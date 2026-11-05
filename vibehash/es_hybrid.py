"""Evo v2's engine for Qwen3.5's hybrid layers (Qwen3.5-0.8B, 2B, 4B, 9B and Qwen3.8-27B share the design): three of
every four layers are Gated DeltaNet linear attention, a fixed-size memory per head (conv state + recurrent state)
instead of a growing KV cache, and every fourth is gated full attention.

The rule is es_fast's (the shared-prompt rule, rank-1 twists, a checker teacher-forcing through the engine that wrote
the answers), on a decoder written for this architecture:

* **The shared reading** of every prompt but its last token is transformers' own forward (untwisted; Gated DeltaNet's
  chunked algorithm), done by worker and checker over the same prompts in one batch, so both get the same states.
* **Everything twisted runs step by step**: each answer token is one decode step, Gated DeltaNet's recurrent rule, for
  the worker answering and for the checker teacher-forcing alike. (A checker reading a whole answer in one forward
  would use the chunked algorithm, which rounds differently from the steps that wrote it.)
* **A pass is always `rows_per_pass(config)` rows**, the same for every worker and checker of a model, padded with idle
  rows, because cuBLAS rounds a row by how many rows share the product. A row's numbers then depend on nothing but the
  row, wherever it sits in the pass. The recurrent state is big (Qwen3.5-4B: 24 layers x 32 heads x 128 x 128 float32,
  48 MB a row), so only one pass of rows is ever resident: rows that finish are replaced by waiting ones at the next
  step (continuous batching), each at its own step and cache slot.
* Each pass is one CUDA graph on CUDA (the buffers are fixed); elsewhere (the CPU tests' tiny model) it runs eagerly.

Twisted linears (TENSORS): full attention's q/k/v/o, linear attention's in_proj_qkv, in_proj_z and out_proj, and the
MLP's gate/up/down; member (j, s) adds s * sigma / sqrt(rank) * (x B_j) A_j^T to each one's output.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import es

# Every projection but linear attention's in_proj_a and in_proj_b (hidden -> one number a value head: each head's
# memory decay and write strength, through softplus/exp and a sigmoid). Those are 0.1% of the projections' weights,
# and a rank-1 twist of their 16-48 outputs moves every head's memory horizon at once rather than what it stores.
TENSORS = ("model.layers.*_proj.weight", "model.layers.*.in_proj_qkv.weight", "model.layers.*.in_proj_z.weight")

# Rows a pass, fixed by the model alone (a worker and its checkers must agree; another value rounds differently). By
# hidden size: 0.8B, 2B, 4B, 9B, 27B. A step's cost is mostly passes over the recurrent state (float32, the model's
# own mamba_ssm_dtype: 18 MB a row in 0.8B, 48 MB in 4B), so rows-steps a second level off early and more rows mostly
# cost memory and make each check (one member's rows in a pass of padding) dearer. Measured on the RTX 4070 SUPER
# (12 GB): 0.8B 2,750 rows-steps/s at 64 (3,200 at 128, twice the memory); 4B 520 at 16 (31 ms a step; 600 at 24,
# 650 at 32), but a round of 64 pairs peaks at 10.9 GiB reserved at 16 and spills into system memory past ~11 (at 32
# it ran 3x slower). 2B, 9B and 27B are provisional, unmeasured.
ROWS_BY_HIDDEN = {1024: 64, 2048: 32, 2560: 16, 4096: 8, 5120: 8}
ROWS_SMALL = 8  # anything else (the tests' tiny models)
_ENGINE: dict = {}


def is_hybrid(model) -> bool:
    return "linear_attention" in (getattr(model.config, "layer_types", None) or ())


def rows_per_pass(cfg) -> int:
    return ROWS_BY_HIDDEN.get(int(cfg.hidden_size), ROWS_SMALL)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    """Qwen3.5's partial rotary embedding: the first cos.shape[-1] numbers of each head rotate, the rest pass."""
    rd = cos.shape[-1]
    xr = x[..., :rd]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, x[..., rd:]], -1)


def _l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _shared(rt, rows):
    """The shared reading (transformers' forward, untwisted) of every prompt but its last token. Returns (ids, mask,
    pos, states): states[l] is (conv inputs [prompts, C, K-1], recurrent state [prompts, Hv, Dk, Dv]) for a linear
    attention layer, (keys, values) [prompts, KVH, W-1, D] for a full attention one."""
    dev = next(rt.model.parameters()).device
    ids, mask = es._left_pad(rows, rt.pad, dev)
    W = ids.shape[1]
    if W < 2:
        raise ValueError("a prompt of one token has nothing to share")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = rt.model(input_ids=ids[:, :W - 1], attention_mask=mask[:, :W - 1], position_ids=pos[:, :W - 1],
                   use_cache=True, logits_to_keep=1)
    K = rt.model.config.linear_conv_kernel_dim
    states = []
    for kind, layer in zip(rt.model.config.layer_types, out.past_key_values.layers):
        if kind == "linear_attention":
            states.append((layer.conv_states[0][..., -(K - 1):], layer.recurrent_states[0]))
        else:
            states.append((layer.keys, layer.values))
    return ids, mask, pos, states


class _Hybrid:
    """Buffers (and CUDA graphs) for decoding R rows through one resident pass of n rows, a KV cache of L slots for
    the full attention layers, answers of up to T tokens. Rows are member-major (member k's answer to prompt q is row
    k * nq + q); row R is a spare that idle slots write to."""

    def __init__(self, rt, linears, n, R, PR, L, T):
        model, cfg = rt.model, rt.model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.rt, self.n, self.R, self.L, self.T, self.dev, self.dt = rt, n, R, L, T, dev, dt
        self.kinds = list(cfg.layer_types)
        self.layers = list(model.model.layers)
        self.embed, self.norm, self.rot = model.model.embed_tokens, model.model.norm, model.model.rotary_emb
        self.lm_w = model.lm_head.weight
        self.H, self.KVH, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.G = self.H // self.KVH
        self.Hk, self.Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.Dk, self.Dv, self.K = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim
        self.C = 2 * self.Hk * self.Dk + self.Hv * self.Dv
        self.names = {id(m): nm for nm, m in linears}
        self.tw = {id(m): (torch.zeros(m.in_features, PR, dtype=dt, device=dev),
                           torch.zeros(PR, m.out_features, dtype=dt, device=dev)) for nm, m in linears}
        z = lambda *s, d=dt: torch.zeros(*s, dtype=d, device=dev)  # noqa: E731
        self.conv, self.rec, self.kc, self.vc = {}, {}, {}, {}
        for l, kind in enumerate(self.kinds):
            if kind == "linear_attention":
                self.conv[l] = z(n, self.C, self.K - 1)
                self.rec[l] = z(n, self.Hv, self.Dk, self.Dv, d=torch.float32)
            else:
                self.kc[l] = z(n, self.KVH, L, self.D)
                self.vc[l] = z(n, self.KVH, L, self.D)
        self.bias = torch.full((n, L), float("-inf"), dtype=dt, device=dev)
        self.zero = z(())
        self.tok = z(n, d=torch.long)
        self.pos = z(n, d=torch.long)
        self.col = z(n, d=torch.long)  # each row's step: the answer token it reads next is number col
        self.slot = z(n, d=torch.long)  # the KV cache slot it writes: W - 1 + col
        self.w1 = z((), d=torch.long)  # W - 1, where every row's answer starts in the cache
        self.rowid = torch.full((n,), R, dtype=torch.long, device=dev)
        self.done = torch.ones(n, dtype=torch.bool, device=dev)
        self.sel = z(n, PR)
        self.ar = torch.arange(n, device=dev)
        self.out = z(R + 1, T, d=torch.long)
        self.ans = z(R + 1, T, d=torch.long)  # teacher forcing: each row's committed answer, its length, each token's
        self.alen = z(R + 1, d=torch.long)  # gap below the step's highest logit and its log-probability
        self.gap = z(R + 1, T, d=torch.float32)
        self.lp = z(R + 1, T, d=torch.float32)
        self.eos = torch.tensor(rt.eos, device=dev)
        self.graphs = {}
        if dev.type == "cuda":
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream()

    def load(self, fac):
        by_name = {nm: i for i, nm in self.names.items()}
        for nm, (A, B) in fac.items():
            Bt, Aa = self.tw[by_name[nm]]
            Bt.copy_(B.permute(1, 0, 2).reshape(Bt.shape))
            Aa.copy_(A.permute(0, 2, 1).reshape(Aa.shape))

    def _lin(self, mod, x):
        y = F.linear(x, mod.weight, mod.bias)
        t = self.tw.get(id(mod))
        if t is None:
            return y
        Bt, Aa = t
        return y.addmm_(torch.mm(x, Bt) * self.sel, Aa)  # the row's twist: one nonzero factor column, times coef

    def _linear_attn(self, l, la, x):
        """Gated DeltaNet's recurrent step for every row (transformers' torch_recurrent_gated_delta_rule's arithmetic,
        the state's products as batched GEMMs)."""
        n, Hk, Hv, Dk, Dv = self.n, self.Hk, self.Hv, self.Dk, self.Dv
        qkv = self._lin(la.in_proj_qkv, x)
        zg = self._lin(la.in_proj_z, x)
        b = F.linear(x, la.in_proj_b.weight)
        a = F.linear(x, la.in_proj_a.weight)
        cs = self.conv[l]
        win = torch.cat([cs, qkv[:, :, None]], -1)  # the causal conv's window: the last K inputs
        cs.copy_(win[:, :, 1:])
        c = F.silu((win.float() * la.conv1d.weight[:, 0, :].float()).sum(-1).to(x.dtype))
        q, k, v = torch.split(c, [Hk * Dk, Hk * Dk, Hv * Dv], -1)
        q, k = q.reshape(n, Hk, Dk).float(), k.reshape(n, Hk, Dk).float()
        if Hv // Hk > 1:
            q, k = q.repeat_interleave(Hv // Hk, dim=1), k.repeat_interleave(Hv // Hk, dim=1)
        B = n * Hv
        q = (_l2norm(q) / (Dk ** 0.5)).reshape(B, 1, Dk)
        k = _l2norm(k).reshape(B, 1, Dk)
        beta = b.sigmoid().float().reshape(B, 1, 1)
        d = (-la.A_log.float().exp() * F.softplus(a.float() + la.dt_bias)).exp().reshape(B, 1, 1)  # the decay
        # S' = d S + k delta^T, delta = (v - d k^T S) beta, out = q^T S' = d q^T S + (q.k) delta: the state is read
        # once for both products, then decayed and updated in place (5 passes over it a step instead of 6).
        Sv = self.rec[l].view(B, Dk, Dv)
        kq = torch.bmm(torch.cat([k, q], 1), Sv) * d  # [B, 2, Dv]: d k^T S, d q^T S
        delta = (v.float().reshape(B, 1, Dv) - kq[:, :1]) * beta
        o = (kq[:, 1:] + torch.bmm(q, k.transpose(1, 2)) * delta).reshape(B, Dv).to(x.dtype)
        Sv.mul_(d)
        Sv.baddbmm_(k.transpose(1, 2), delta)
        o = la.norm(o, zg.reshape(n * Hv, Dv)).reshape(n, Hv * Dv)
        return self._lin(la.out_proj, o)

    def _full_attn(self, l, at, x, cos, sin):
        n, H, KVH, D, G = self.n, self.H, self.KVH, self.D, self.G
        qg = self._lin(at.q_proj, x).view(n, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(n, H * D)
        q = _rope(at.q_norm(q), cos, sin)
        k = _rope(at.k_norm(self._lin(at.k_proj, x).view(n, KVH, D)), cos, sin)
        v = self._lin(at.v_proj, x).view(n, KVH, D)
        kc, vc = self.kc[l], self.vc[l]
        kc[self.ar, :, self.slot] = k
        vc[self.ar, :, self.slot] = v
        o = F.scaled_dot_product_attention(q.reshape(n, KVH, G, D), kc, vc, attn_mask=self.bias[:, None, None, :],
                                           scale=at.scaling)
        return self._lin(at.o_proj, o.reshape(n, H * D) * torch.sigmoid(gate))

    def _pass(self):
        """One decode step's logits for every resident row, their states moved on."""
        self.bias.index_put_((self.ar, self.slot), self.zero)
        h = self.embed(self.tok)
        cos, sin = self.rot(h, self.pos[:, None])
        for l, (kind, layer) in enumerate(zip(self.kinds, self.layers)):
            x = layer.input_layernorm(h)
            if kind == "linear_attention":
                h = h + self._linear_attn(l, layer.linear_attn, x)
            else:
                h = h + self._full_attn(l, layer.self_attn, x, cos, sin)
            x = layer.post_attention_layernorm(h)
            mlp = layer.mlp
            h = h + self._lin(mlp.down_proj, mlp.act_fn(self._lin(mlp.gate_proj, x)) * self._lin(mlp.up_proj, x))
        return F.linear(self.norm(h), self.lm_w)

    def _step(self, forced: bool):
        logits = self._pass()
        rows, col = self.rowid, self.col.clamp(max=self.T - 1)
        if forced:
            nxt = self.ans[rows, col]
            got = logits.gather(1, nxt[:, None])[:, 0].float()
            self.gap.index_put_((rows, col), logits.max(-1).values.float() - got)
            self.lp.index_put_((rows, col), got - torch.logsumexp(logits.float(), -1))
            fin = self.col + 1 >= self.alen[rows]
        else:
            nxt = logits.argmax(-1)
            self.out.index_put_((rows, col), nxt)
            fin = (nxt[:, None] == self.eos).any(-1) | (self.col + 1 >= self.T)
        act = (~self.done).long()
        self.tok.copy_(nxt)
        self.pos.add_(act)
        self.col.add_(act)
        torch.clamp(self.col + self.w1, max=self.L - 1, out=self.slot)
        self.done.logical_or_(fin)
        self.rowid.masked_fill_(self.done, self.R)

    def run(self, forced: bool):
        if self.dev.type != "cuda":
            return self._step(forced)
        g = self.graphs.get(forced)
        if g is not None:
            g.replay()
            return
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            self._step(forced)
            g = torch.cuda.CUDAGraph()
            g.capture_begin(self.pool)
            try:
                self._step(forced)
            finally:
                g.capture_end()
        cur.wait_stream(self.stream)
        self.graphs[forced] = g

    def decode(self, rt, rows, members, fac, sigma, rank, forced: bool, order):
        """Every row in `order` (row ids) through the resident pass, waiting rows taking slots as rows finish."""
        dev, dt, n = self.dev, self.dt, self.n
        nq = len(rows)
        self.load({nm: (A.to(dt), B.to(dt)) for nm, (A, B) in fac.items()})
        r = next(iter(fac.values()))[0].shape[2]
        M = len(members)
        selm = torch.zeros(M, self.sel.shape[1], dtype=dt, device=dev)
        coef = torch.tensor([s * sigma / math.sqrt(rank) for _, s in members], device=dev, dtype=dt)
        for i in range(r):
            selm[torch.arange(M, device=dev), torch.tensor([j * r + i for j, _ in members], device=dev)] = coef
        ids, mask, pos, states = _shared(rt, rows)
        W = ids.shape[1]
        for l, kind in enumerate(self.kinds):  # a row's cache past its prompt starts as zeros, whoever had the slot
            if kind == "full_attention":
                states[l] = tuple(F.pad(s, (0, 0, 0, self.L - (W - 1))) for s in states[l])
        pbias = torch.full((nq, self.L), float("-inf"), dtype=dt, device=dev)
        pbias[:, :W - 1].masked_fill_(mask[:, :W - 1].bool(), 0.0)
        last_tok, last_pos = ids[:, W - 1], pos[:, W - 1]
        self.w1.fill_(W - 1)
        self.done.fill_(True)
        self.rowid.fill_(self.R)
        self.bias.fill_(float("-inf"))
        order = torch.as_tensor(order, dtype=torch.long, device=dev)
        at, total = 0, order.numel()
        while True:
            free = torch.nonzero(self.done).squeeze(1)
            if at < total and free.numel():
                take = min(free.numel(), total - at)
                slots, rid = free[:take], order[at:at + take]
                at += take
                q, k = rid % nq, rid // nq
                for l, kind in enumerate(self.kinds):
                    a, b = states[l]
                    if kind == "linear_attention":
                        self.conv[l][slots] = a[q]
                        self.rec[l][slots] = b[q]
                    else:
                        self.kc[l][slots] = a[q]
                        self.vc[l][slots] = b[q]
                self.bias[slots] = pbias[q]
                self.tok[slots] = last_tok[q]
                self.pos[slots] = last_pos[q]
                self.col[slots] = 0
                self.slot[slots] = W - 1
                self.sel[slots] = selm[k]
                self.rowid[slots] = rid
                self.done[slots] = False
            elif free.numel() == n:
                break
            self.run(forced)


def _engine(rt, linears, fac, R, rows, T):
    """The engine for R rows of these prompts' answers up to T tokens, made once and kept while the shape repeats."""
    W = max(len(r) for r in rows)
    L = -(-(W + T - 1) // 32) * 32
    A0 = next(iter(fac.values()))[0]
    PR = A0.shape[0] * A0.shape[2]
    n = rows_per_pass(rt.model.config)
    key = (id(rt.model), n, R, PR, L, T, tuple(nm for nm, _ in linears))
    eng = _ENGINE.get(key)
    if eng is None:
        _ENGINE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eng = _ENGINE[key] = _Hybrid(rt, linears, n, R, PR, L, T)
    return eng


def _pin_precision():
    # The recurrent state's products are float32 GEMMs: TF32 would round them otherwise, and differently from a
    # checker that left it off.
    torch.backends.cuda.matmul.allow_tf32 = False


def answer(rt, linears, fac, members, rows, T, sigma, rank):
    """Every member's greedy answers to every prompt (es_fast.answer_members on a hybrid model)."""
    _pin_precision()
    nq, M = len(rows), len(members)
    R = M * nq
    eng = _engine(rt, linears, fac, R, rows, T)
    eng.decode(rt, rows, members, fac, sigma, rank, False, torch.arange(R))
    toks = eng.out[:R].tolist()
    answers = []
    for row in toks:
        cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
        answers.append(row[:cut])
    return [answers[k * nq:(k + 1) * nq] for k in range(M)]


def forced(rt, linears, fac, members, rows, answers, T, sigma, rank):
    """Teacher forcing of every member's committed answers (answers[member][prompt]) through the decode steps that
    wrote them: (worst gap below the step's highest logit, mean log-probability) of each, two [members, prompts]
    float tensors (0 for an empty answer)."""
    _pin_precision()
    nq, M = len(rows), len(members)
    R = M * nq
    ans, alen = torch.zeros(R + 1, T, dtype=torch.long), torch.zeros(R + 1, dtype=torch.long)
    for k, member in enumerate(answers):
        for q, a in enumerate(member):
            if len(a) > T:
                raise ValueError(f"an answer of {len(a)} tokens, past the {T} answering stops at")
            ans[k * nq + q, :len(a)] = torch.tensor(a, dtype=torch.long)
            alen[k * nq + q] = len(a)
    eng = _engine(rt, linears, fac, R, rows, T)
    eng.ans.copy_(ans)
    eng.alen.copy_(alen)
    eng.gap.zero_()
    eng.lp.zero_()
    eng.decode(rt, rows, members, fac, sigma, rank, True, torch.nonzero(alen[:R]).squeeze(1).tolist())
    cols = torch.arange(T, device=eng.dev)[None, :] < eng.alen[:R, None]
    gaps = torch.where(cols, eng.gap[:R], 0.0).max(-1).values.view(M, nq)
    lps = torch.where(cols, eng.lp[:R], 0.0).sum(-1) / eng.alen[:R].clamp(min=1)
    return gaps, lps.view(M, nq)
