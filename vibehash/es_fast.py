"""Evo v2's worker plumbing: rank-1 twists drawn from a round's pair seeds, every member's greedy answers, and how
likely each member finds given answers, all under the shared-prompt rule.

The shared-prompt rule: each prompt but its last token is read once with the current untwisted weights; every member
then reads the prompt's last token and writes (or is scored on) its answer with its own twist, so each answer's first
token already comes from the twisted model. Member (j, s) adds s * sigma / sqrt(rank) * (x B_j) A_j^T to every twisted
linear's output.

On CUDA with a bf16 model, answering runs on an engine ported from the speed experiments: one decode for every member's
rows at once from a static KV cache, in passes of ROWS rows, rows dropped as they end (in fixed bucket sizes), and one
CUDA graph per bucket size. A checker teacher-forces committed answers through the same engine (gaps_items), so it
rounds exactly as the worker did. Elsewhere (the CPU tests' tiny model), a plain loop computes the same rule and a
checker reads answers in one forward pass.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import es

# The engine passes through the model ROWS rows at a time, padding the last pass: cuBLAS picks a matrix product's
# kernel, and so its rounding, by the matrix's rows, so a row's numbers then depend on nothing but the row. A checker
# teacher-forcing one member's 16 answers rounds exactly as the worker who wrote them among a thousand (on the same GPU
# and libraries). Measured on the RTX 4070 SUPER (Qwen2.5-0.5B-Instruct): a row's result is the same wherever it sits
# in a pass, and a pass of 256 rows costs about what one of 8 does.
ROWS = 256
BUCKETS = tuple(ROWS * k for k in (16, 12, 8, 6, 4, 3, 2, 1))
_ENGINE: dict = {}


# --- the twists ---------------------------------------------------------------------------------------------------

def stream_layout(linears) -> list[tuple[str, int, int]]:
    """(name, out, in) of every twisted linear, in the order a pair's noise stream covers them: sorted by name, each
    A (out numbers) then B (in numbers)."""
    return sorted((n, m.out_features, m.in_features) for n, m in linears)


def lowrank_factors(linears, pair_seeds: list[int], device="cuda", dtype=torch.bfloat16) -> dict:
    """{name: (A [P, out, 1], B [P, in, 1])} for pairs with these seeds: pair p's numbers are es.noise(seed_p, total),
    the same float32 bits on every machine; the cast to `dtype` is the worker's."""
    layout = stream_layout(linears)
    total = sum(o + i for _, o, i in layout)
    streams = np.stack([es.noise(int(s), total) for s in pair_seeds]) if pair_seeds else np.zeros((0, total), np.float32)
    out, at = {}, 0
    for name, o, i in layout:
        A = torch.from_numpy(np.ascontiguousarray(streams[:, at:at + o]))[:, :, None]
        B = torch.from_numpy(np.ascontiguousarray(streams[:, at + o:at + o + i]))[:, :, None]
        out[name] = (A.to(device, dtype), B.to(device, dtype))
        at += o + i
    return out


class Twist:
    """Some members' twists in every linear's forward while entered. factors: name -> (A [P, m, r], B [P, n, r]);
    coef [P]: each member's sign * sigma / sqrt(r). The batch's rows are member-major: P members x the rows each.
    The correction is added in a batched GEMM's epilogue, in the factors' dtype."""

    def __init__(self, linears, factors, coef):
        self.linears, self.factors, self.coef = linears, factors, coef

    def __enter__(self):
        for n, m in self.linears:
            A, B = self.factors[n]
            m.forward = self._fwd(m, A, B)

    def __exit__(self, *exc):
        for _, m in self.linears:
            del m.forward

    def _fwd(self, m, A, B):
        P, W, bias = A.shape[0], m.weight, m.bias
        At, cf = A.transpose(1, 2), self.coef[:, None, None]

        def f(x):
            y = F.linear(x, W, bias)
            xb = torch.bmm(x.reshape(P, -1, x.shape[-1]), B) * cf
            y.view(P, -1, y.shape[-1]).baddbmm_(xb, At)
            return y
        return f


def _coefs(members, sigma, rank, device, dtype):
    return torch.tensor([s * sigma / math.sqrt(rank) for _, s in members], device=device, dtype=dtype)


def _pick(fac, members, device):
    idx = torch.tensor([j for j, _ in members], device=device)
    return {n: (A[idx], B[idx]) for n, (A, B) in fac.items()}


def _shared_cache(rt, rows):
    """The shared reading: every prompt but its last token, untwisted. Returns (ids, mask, pos, keys/values per layer)."""
    dev = next(rt.model.parameters()).device
    ids, mask = es._left_pad(rows, rt.pad, dev)
    W = ids.shape[1]
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = rt.model(input_ids=ids[:, :W - 1], attention_mask=mask[:, :W - 1], position_ids=pos[:, :W - 1],
                   use_cache=True, logits_to_keep=1)
    return ids, mask, pos, [(layer.keys, layer.values) for layer in out.past_key_values.layers]


# --- answering ------------------------------------------------------------------------------------------------------

def answer_members(rt: es.Runtime, linears, fac: dict, members: list, rows: list, max_tokens: int, sigma: float,
                   rank: int = 1, max_rows: int = 512) -> list:
    """Every member's greedy answers to every prompt under the shared-prompt rule: answers[member][prompt] = token
    ids, ending at an end-of-turn token (kept) or at max_tokens. members: [(pair index j, sign +1/-1)]."""
    p = next(rt.model.parameters())
    with torch.no_grad():
        if _hybrid(rt):
            from . import es_hybrid
            return es_hybrid.answer(rt, linears, fac, members, rows, max_tokens, sigma, rank)
        if p.device.type == "cuda" and p.dtype == torch.bfloat16:
            return _answer_engine(rt, linears, fac, members, rows, max_tokens, sigma, rank)
        return _answer_plain(rt, linears, fac, members, rows, max_tokens, sigma, rank, max_rows)


def _hybrid(rt) -> bool:
    """A Qwen3.5-style model (Gated DeltaNet and full attention layers): es_hybrid's engine, on every device."""
    return "linear_attention" in (getattr(rt.model.config, "layer_types", None) or ())


def _answer_plain(rt, linears, fac, members, rows, T, sigma, rank, max_rows):
    from transformers import DynamicCache
    dev, dt = next(rt.model.parameters()).device, next(rt.model.parameters()).dtype
    ids, mask, pos, kv = _shared_cache(rt, rows)
    nq, W = len(rows), ids.shape[1]
    eos = torch.tensor(rt.eos, device=dev)
    per = max(1, max_rows // nq)
    out = []
    for lo in range(0, len(members), per):
        chunk = members[lo:lo + per]
        P = len(chunk)
        cache = DynamicCache()
        for l, (k, v) in enumerate(kv):
            cache.update(k.repeat(P, 1, 1, 1), v.repeat(P, 1, 1, 1), l)
        am = mask[:, :W - 1].repeat(P, 1)
        tok, at = ids[:, W - 1].repeat(P), pos[:, W - 1].repeat(P)
        done = torch.zeros(P * nq, dtype=torch.bool, device=dev)
        made = []
        sub = {n: (A.to(dt), B.to(dt)) for n, (A, B) in _pick(fac, chunk, dev).items()}
        with Twist(linears, sub, _coefs(chunk, sigma, rank, dev, dt)):
            for _ in range(T):
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                logits = rt.model(input_ids=tok[:, None], attention_mask=am, position_ids=at[:, None],
                                  past_key_values=cache, use_cache=True).logits[:, -1]
                nxt = torch.where(done, torch.full_like(tok, rt.pad), logits.argmax(-1))
                made.append(nxt)
                done = done | torch.isin(nxt, eos)
                tok, at = nxt, at + 1
                if bool(done.all()):
                    break
        toks = torch.stack(made, 1).tolist()
        for k in range(P):
            ans = []
            for row in toks[k * nq:(k + 1) * nq]:
                cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
                ans.append(row[:cut])
            out.append(ans)
    return out


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class _Engine:
    """Buffers and CUDA graphs for decoding up to R rows with a cache of L slots (bf16 on CUDA), in passes of ROWS rows
    (the rows rounded up to a multiple of ROWS: `cap`; the rows past R are padding)."""

    def __init__(self, rt, linears, R, PR, L, T):
        model, cfg = rt.model, rt.model.config
        dev, bf = "cuda", torch.bfloat16
        self.rt, self.R, self.L, self.T = rt, R, L, T
        self.cap = cap = -(-R // ROWS) * ROWS
        self.H, self.KVH = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.G = self.H // self.KVH
        self.layers = list(model.model.layers)
        self.embed, self.norm, self.rot = model.model.embed_tokens, model.model.norm, model.model.rotary_emb
        self.lm_w = model.lm_head.weight
        self.names = {id(m): n for n, m in linears}
        self.tw = {id(m): (torch.zeros(m.in_features, PR, dtype=bf, device=dev),
                           torch.zeros(PR, m.out_features, dtype=bf, device=dev)) for n, m in linears}
        nl = len(self.layers)
        self.kc = [torch.zeros(cap, self.KVH, L, self.D, dtype=bf, device=dev) for _ in range(nl)]
        self.vc = [torch.zeros(cap, self.KVH, L, self.D, dtype=bf, device=dev) for _ in range(nl)]
        self.bias = torch.zeros(cap, L, dtype=bf, device=dev)
        self.tok = torch.zeros(cap, dtype=torch.long, device=dev)
        self.pos = torch.zeros(cap, dtype=torch.long, device=dev)
        self.rowid = torch.zeros(cap, dtype=torch.long, device=dev)
        self.done = torch.ones(cap, dtype=torch.bool, device=dev)
        self.sel = torch.zeros(cap, PR, dtype=bf, device=dev)
        self.out = torch.zeros(R + 1, T, dtype=torch.long, device=dev)
        self.ans = torch.zeros(R + 1, T, dtype=torch.long, device=dev)  # teacher forcing: each row's committed answer,
        self.alen = torch.zeros(R + 1, dtype=torch.long, device=dev)  # its length, and each token's gap below the
        self.gap = torch.zeros(R + 1, T, dtype=torch.float32, device=dev)  # step's highest logit
        self.slot = torch.zeros(1, dtype=torch.long, device=dev)
        self.tcol = torch.zeros(1, dtype=torch.long, device=dev)
        self.eos = torch.tensor(rt.eos, device=dev)
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()

    def load(self, fac):
        by_name = {n: i for i, n in self.names.items()}
        for n, (A, B) in fac.items():
            Bt, Aa = self.tw[by_name[n]]
            Bt.copy_(B.permute(1, 0, 2).reshape(Bt.shape))
            Aa.copy_(A.permute(0, 2, 1).reshape(Aa.shape))

    def _lin(self, mod, x, sel):
        y = F.linear(x, mod.weight, mod.bias)
        t = self.tw.get(id(mod))
        if t is None:
            return y
        Bt, Aa = t
        return y.addmm_(torch.mm(x, Bt) * sel, Aa)  # the row's twist: one nonzero factor column, times coef

    def _pass(self, lo):
        """One decode step's logits (bf16) for rows lo to lo + ROWS, their keys and values written into the cache."""
        H, KVH, D, G, n = self.H, self.KVH, self.D, self.G, ROWS
        hi = lo + n
        mask = self.bias[lo:hi, None, None, :]
        sel = self.sel[lo:hi]
        h = self.embed(self.tok[lo:hi])
        cos, sin = self.rot(h, self.pos[lo:hi, None])
        for l, layer in enumerate(self.layers):
            at, mlp = layer.self_attn, layer.mlp
            res = h
            x = layer.input_layernorm(h)
            q = self._lin(at.q_proj, x, sel).view(n, H, D)
            k = self._lin(at.k_proj, x, sel).view(n, KVH, D)
            v = self._lin(at.v_proj, x, sel).view(n, KVH, D)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            kc, vc = self.kc[l][lo:hi], self.vc[l][lo:hi]
            kc.index_copy_(2, self.slot, k[:, :, None, :])
            vc.index_copy_(2, self.slot, v[:, :, None, :])
            o = F.scaled_dot_product_attention(q.view(n, KVH, G, D), kc, vc, attn_mask=mask, scale=at.scaling)
            h = res + self._lin(at.o_proj, o.reshape(n, H * D), sel)
            res = h
            x = layer.post_attention_layernorm(h)
            x = mlp.act_fn(self._lin(mlp.gate_proj, x, sel)) * self._lin(mlp.up_proj, x, sel)
            h = res + self._lin(mlp.down_proj, x, sel)
        return F.linear(self.norm(h), self.lm_w)

    def _step(self, B):
        self.bias[:B].index_fill_(1, self.slot, 0.0)
        col = self.tcol.expand(ROWS)
        for lo in range(0, B, ROWS):
            nxt = self._pass(lo).argmax(-1)
            self.out.index_put_((self.rowid[lo:lo + ROWS], col), nxt)
            self.done[lo:lo + ROWS] |= (nxt[:, None] == self.eos).any(-1)
            self.tok[lo:lo + ROWS].copy_(nxt)
        self._advance(B)

    def _forced_step(self, B):
        """_step reading each row's committed token instead of choosing one, and recording how far its logit is below
        the pass's highest (0 when it's the greedy choice)."""
        self.bias[:B].index_fill_(1, self.slot, 0.0)
        col = self.tcol.expand(ROWS)
        for lo in range(0, B, ROWS):
            logits = self._pass(lo)
            rows = self.rowid[lo:lo + ROWS]
            want = self.ans[rows, col]
            top = logits.max(-1).values.float()
            self.gap.index_put_((rows, col), top - logits.gather(1, want[:, None])[:, 0].float())
            self.done[lo:lo + ROWS] |= self.alen[rows] <= col + 1
            self.tok[lo:lo + ROWS].copy_(want)
        self._advance(B)

    def _advance(self, B):
        self.pos[:B] += 1
        self.slot += 1
        self.tcol += 1

    def run(self, B, forced=False):
        g = self.graphs.get((forced, B))
        if g is not None:
            g.replay()
            return
        step = self._forced_step if forced else self._step
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            step(B)
            g = torch.cuda.CUDAGraph()
            g.capture_begin(self.pool)
            try:
                step(B)
            finally:
                g.capture_end()
        cur.wait_stream(self.stream)
        self.graphs[(forced, B)] = g

    def compact(self, B, nb):
        keep = torch.nonzero(~self.done[:B]).squeeze(1)
        n = keep.numel()
        for buf in (self.tok, self.pos, self.rowid, self.sel, self.bias):
            buf[:n] = buf[keep]
        for l in range(len(self.layers)):
            self.kc[l][:n] = self.kc[l][keep]
            self.vc[l][:n] = self.vc[l][keep]
        self.done[:n] = False
        self.done[n:nb] = True
        self.rowid[n:nb] = self.R


def _bucket(n, R):
    best = R
    for b in BUCKETS:
        if n <= b < best:
            best = b
    return best


def _answer_engine(rt, linears, fac, members, rows, T, sigma, rank):
    nq, M = len(rows), len(members)
    eng, R = _engine(rt, linears, fac, members, rows, T, sigma, rank)
    _decode(eng, T)
    toks = eng.out[:R].tolist()
    answers = []
    for row in toks:
        cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
        answers.append(row[:cut])
    return [answers[k * nq:(k + 1) * nq] for k in range(M)]


def _gaps_engine(rt, linears, fac, members, rows, answers, T, sigma, rank):
    """Teacher forcing on the answering engine: every member's committed answers (answers[member][prompt]) read
    through the same decode steps that wrote them, so a checker rounds as the worker did. The worst gap of each
    member's answer to each prompt, a [members, prompts] float tensor (0 for an empty answer)."""
    nq, M = len(rows), len(members)
    R = M * nq
    ans, alen = torch.zeros(R + 1, T, dtype=torch.long), torch.zeros(R + 1, dtype=torch.long)
    for k, member in enumerate(answers):
        for q, a in enumerate(member):
            if len(a) > T:
                raise ValueError(f"an answer of {len(a)} tokens, past the {T} answering stops at")
            ans[k * nq + q, :len(a)] = torch.tensor(a, dtype=torch.long)
            alen[k * nq + q] = len(a)
    eng, R = _engine(rt, linears, fac, members, rows, T, sigma, rank)
    eng.ans.copy_(ans)
    eng.alen.copy_(alen)
    eng.gap.zero_()
    eng.done[:R] = eng.alen[:R] == 0
    _decode(eng, T, forced=True)
    cols = torch.arange(T, device="cuda")[None, :] < eng.alen[:R, None]  # (a done row's later steps are padding)
    return torch.where(cols, eng.gap[:R], 0.0).max(-1).values.view(M, nq)


def _engine(rt, linears, fac, members, rows, T, sigma, rank):
    """The engine for len(members) x len(rows) rows, set up as a decode's first step finds it: the round's twists
    loaded, each row's twist selected, the shared reading in its cache, each row about to read its prompt's last token."""
    dev, bf = "cuda", torch.bfloat16
    nq, M = len(rows), len(members)
    R = M * nq
    W = max(len(r) for r in rows)
    L = -(-(W + T - 1) // 32) * 32
    A0 = next(iter(fac.values()))[0]
    r = A0.shape[2]
    PR = A0.shape[0] * r
    key = (id(rt.model), R, PR, L, T, tuple(n for n, _ in linears))
    eng = _ENGINE.get(key)
    if eng is None:
        _ENGINE.clear()
        torch.cuda.empty_cache()
        eng = _ENGINE[key] = _Engine(rt, linears, R, PR, L, T)
    eng.load({n: (A.to(bf), B.to(bf)) for n, (A, B) in fac.items()})
    jr = torch.tensor([j for j, _ in members], device=dev).repeat_interleave(nq)
    cr = _coefs(members, sigma, rank, dev, bf).repeat_interleave(nq)
    eng.sel.zero_()
    for k in range(r):
        eng.sel[torch.arange(R, device=dev), jr * r + k] = cr
    eng.rowid[:R].copy_(torch.arange(R, device=dev))
    eng.rowid[R:] = R  # the padding rows: done from the start, writing to the spare row
    eng.done[R:] = True
    eng.bias[R:].zero_()
    ids, mask, pos, kv = _shared_cache(rt, rows)
    for l, (k, v) in enumerate(kv):
        eng.kc[l][:R, :, :W - 1].copy_(k.repeat(M, 1, 1, 1))
        eng.vc[l][:R, :, :W - 1].copy_(v.repeat(M, 1, 1, 1))
    eng.bias[:R].fill_(float("-inf"))
    eng.bias[:R, :W - 1].masked_fill_(mask[:, :W - 1].bool().repeat(M, 1), 0.0)
    eng.tok[:R] = ids[:, W - 1].repeat(M)
    eng.pos[:R] = pos[:, W - 1].repeat(M)
    eng.done[:R] = False
    eng.slot.fill_(W - 1)
    eng.tcol.fill_(0)
    return eng, R


def _decode(eng, T, forced=False):
    """Up to T decode steps, rows dropped as they end (in fixed bucket sizes)."""
    B = eng.cap
    n = int((~eng.done[:B]).sum())
    for _ in range(T):
        if n == 0:
            break
        nb = _bucket(n, eng.cap)
        if nb < B:
            eng.compact(B, nb)
            B = nb
        eng.run(B, forced)
        n = int((~eng.done[:B]).sum())


# --- scoring ----------------------------------------------------------------------------------------------------------

def _mean_logprob(rt, h, w):
    """Mean per-token log-probability of answer w at hidden states h [P, n, hidden], for P members at once: logits in
    the model's dtype into a fused cross-entropy, so the vocabulary's distribution is never copied to float32."""
    P, n = h.shape[0], h.shape[1]
    logits = rt.model.lm_head(h).reshape(P * n, -1)
    return -F.cross_entropy(logits, w.repeat(P), reduction="none").float().view(P, n).mean(-1)


def gaps_items(rt: es.Runtime, linears, fac: dict, members: list, items: list, sigma: float, rank: int = 1,
               max_rows: int = 512, max_tokens: int | None = None):
    """A checker's teacher forcing under the shared-prompt rule: for each member and each item (prompt ids, answer
    ids), the worst gap over the answer's tokens between the highest logit and the committed token's (0 when every
    token is the greedy choice). A [members, items] float tensor; an empty answer's gap is 0.

    On CUDA with a bf16 model the answers are read through the answering engine's own decode steps (max_tokens: the
    answers' limit, which sizes the engine's cache as answering did), so a checker rounds exactly as the worker did
    however many rows each had; elsewhere in one forward pass."""
    p = next(rt.model.parameters())
    if _hybrid(rt):  # Qwen3.5: through es_hybrid's decode steps on every device
        T = max([max_tokens or 0] + [len(a) for _, a in items])
        if not members or not items or T == 0:
            return torch.zeros(len(members), len(items), device=p.device)
        from . import es_hybrid
        with torch.no_grad():
            return es_hybrid.forced(rt, linears, fac, members, [q for q, _ in items],
                                    [[a for _, a in items]] * len(members), T, sigma, rank)[0]
    if p.device.type == "cuda" and p.dtype == torch.bfloat16:
        T = max([max_tokens or 0] + [len(a) for _, a in items])
        if not members or not items or T == 0:
            return torch.zeros(len(members), len(items), device=p.device)
        with torch.no_grad():
            return _gaps_engine(rt, linears, fac, members, [q for q, _ in items], [[a for _, a in items]] * len(members),
                                T, sigma, rank)
    return _forced(rt, linears, fac, members, items, sigma, rank, max_rows, _worst_gap)


def _worst_gap(rt, h, w):
    logits = rt.model.lm_head(h).float()  # [P, n, vocab]
    return (logits.max(-1).values - logits.gather(-1, w[None, :, None].expand(h.shape[0], -1, 1))[..., 0]).max(-1).values


def score_items(rt: es.Runtime, linears, fac: dict, members: list, items: list, sigma: float, rank: int = 1,
                max_rows: int = 512):
    """How likely each member finds each item's answer, under the shared-prompt rule: a [members, items] float tensor
    of mean per-token log-probabilities. items: [(prompt token ids, answer token ids)], from any round.

    On a Qwen3.5-style model, through es_hybrid's decode steps (so worker and checker get the same numbers)."""
    if _hybrid(rt):
        p = next(rt.model.parameters())
        T = max([0] + [len(a) for _, a in items])
        if not members or not items or T == 0:
            return torch.zeros(len(members), len(items), device=p.device)
        from . import es_hybrid
        with torch.no_grad():
            return es_hybrid.forced(rt, linears, fac, members, [q for q, _ in items],
                                    [[a for _, a in items]] * len(members), T, sigma, rank)[1]
    return _forced(rt, linears, fac, members, items, sigma, rank, max_rows, _mean_logprob)


def _forced(rt, linears, fac, members, items, sigma, rank, max_rows, per_item):
    """Teacher forcing of every item's answer through every member's twisted model (the shared-prompt rule), and
    per_item(rt, hidden [P, n, hidden], answer ids) -> [P] for each item."""
    from transformers import DynamicCache
    dev, dt = next(rt.model.parameters()).device, next(rt.model.parameters()).dtype
    I = len(items)
    out = torch.zeros(len(members), I, device=dev)
    if not I or not members:
        return out
    with torch.no_grad():
        ids, mask, pos, kv = _shared_cache(rt, [p for p, _ in items])
        W = ids.shape[1]
        L = max(len(t) for _, t in items)
        inp = torch.full((I, L), rt.pad, dtype=torch.long, device=dev)
        for j, (_, t) in enumerate(items):
            seq = [int(ids[j, -1])] + list(t[:-1])
            inp[j, :len(seq)] = torch.tensor(seq, device=dev)
        want = [torch.tensor(t, device=dev) for _, t in items]
        am_p, pos0 = mask[:, :W - 1], pos[:, W - 1]
        per = max(1, max_rows // I)
        for lo in range(0, len(members), per):
            chunk = members[lo:lo + per]
            P = len(chunk)
            cache = DynamicCache()
            for l, (k, v) in enumerate(kv):
                cache.update(k.repeat(P, 1, 1, 1), v.repeat(P, 1, 1, 1), l)
            am = torch.cat([am_p.repeat(P, 1), torch.ones(P * I, L, dtype=am_p.dtype, device=dev)], 1)
            pids = (pos0[:, None] + torch.arange(L, device=dev)[None, :]).repeat(P, 1)
            sub = {n: (A.to(dt), B.to(dt)) for n, (A, B) in _pick(fac, chunk, dev).items()}
            with Twist(linears, sub, _coefs(chunk, sigma, rank, dev, dt)):
                h = rt.model.model(input_ids=inp.repeat(P, 1), attention_mask=am, position_ids=pids,
                                   past_key_values=cache, use_cache=True).last_hidden_state
            del cache
            for j, w in enumerate(want):
                if len(w):
                    out[lo:lo + P, j] = per_item(rt, h[torch.arange(P, device=dev) * I + j, :len(w)], w)
            del h
    return out
