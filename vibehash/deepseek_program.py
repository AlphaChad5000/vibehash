"""What one unit of a DeepSeek-V4.1 top-layers job computes, op by op (DEEPSEEK.md §8): program.py's tree for a
deepseek_job.DeepSeekSpec, which a worker trains by (deepseek_tracer.py runs it in PyTorch) and the node referees a
dispute by (referee_deepseek.py's ds_* ops, in NumPy). program.Program.of builds this one for a DeepSeek job.

The tree is program.py's: unit → segment → Adam step → its parts → ops, each op reading named 2-D float32 tensors of
the state and writing one, so any op can be checked on its own given the state before it. A training unit computes in
float32 throughout (deepseek_top.TRAIN), so every op's attrs say num "f32".

A step runs its batch's sequences together, layer by layer, as deepseek_top.train_step_streamed does, so each layer's
routed experts (7.2 GB of Flash's) are read once a pass, not once a sequence:
- the inputs: the records of the step's sequences, their rows concatenated (N rows, sequence after sequence);
- each trained layer forward, on all N rows at once, but for attention, which runs a sequence at a time (rows sliced
  out, then put back together);
- the head and loss, on the learned positions of every sequence (1 / their count is each one's weight);
- each trained layer backward, in reverse, recomputing its forward from its input first (activation checkpointing);
- Adam on every adapter.

The routed experts go by slot: row t·k + j of an expert op is token t's j-th chosen expert's, and the op uses that
expert's weights for it. Which tokens reach which expert is only known once the router has run, but the shapes (N·k
rows) are known before, so the tree follows from the data alone, as a game needs.
"""
from __future__ import annotations

import numpy as np

from .deepseek_config import LORA_MODULES
from .model import batch_indices
from .program import Node, Op, Program, f32

HEAD_BLOCK = 4096  # rows of the head a head op covers (bf16, 5120 wide: 42 MB of Flash's)


class DeepSeekProgram(Program):
    """The tree of one unit of a DeepSeek top-layers fine-tune: program.Program's unit, segments and Adam steps, with
    DeepSeek's step."""

    def __init__(self, spec, data, lr: float, batch_size: int, seed: int, segments: int = 1, per_segment: int = 1):
        from .finetune import OPTIMIZERS

        self.spec, self.data, self.seed, self.cfg, self.cut = spec, data, seed, spec.cfg, spec.top
        self.lr, self.batch_size, self.segments, self.per_segment = lr, batch_size, segments, per_segment
        self.mods = spec.modules()
        self.mod = {name: k for k, (name, _, _) in enumerate(self.mods)}
        b1, b2, eps = (f32(x) for x in OPTIMIZERS[spec.optimizer])
        self.adam = {"b1": b1, "b2": b2, "eps": eps, "c1m": f32(1 - b1), "c2m": f32(1 - b2), "lr": f32(lr)}
        if self.cfg.score_func != "sqrtsoftplus":
            raise NotImplementedError("DeepSeek programs score experts with sqrt(softplus): the router's backward")
        if len({self.cfg.ratio(i) for i in self.cut.trained}) != 1:
            raise NotImplementedError("trained layers that compress by different ratios")
        tokens = data.tokens
        self.starts = np.concatenate([[0], np.cumsum(tokens.lengths)]).astype(np.int64)

    def batch(self, t: int) -> list[int]:
        """Step t's sequences (the job's training sequences' ids), as deepseek_unit.train_segment takes them."""
        train = self.data.train
        return [int(train[i]) for i in batch_indices(self.seed, t, len(train), self.batch_size)]

    def length(self, seq: int) -> int:
        return int(self.data.tokens.lengths[seq])

    def learned(self, seq: int) -> int:
        """Positions of `seq` whose next token is learned: the records' mask (deepseek_worker.sequences)."""
        a = int(self.starts[seq])
        return int(self.data.tokens.learn[a + 1:a + self.length(seq)].sum())

    def step(self, t: int) -> Node:
        def build():
            return _Step(self, t).atoms()
        return Node(f"step {t}", build)


class _Step:
    """Builds one Adam step's ops, keeping track of every tensor's shape as it goes."""

    def __init__(self, p: DeepSeekProgram, t: int):
        self.p, self.cfg, self.spec, self.t = p, p.cfg, p.spec, t
        self.seqs = p.batch(t)
        self.T = [p.length(s) for s in self.seqs]
        self.offs = np.concatenate([[0], np.cumsum(self.T)]).astype(np.int64).tolist()
        self.N = int(self.offs[-1])
        self.P = sum(p.learned(s) for s in self.seqs)
        if self.P == 0:
            raise ValueError(f"step {t}'s sequences have no learned positions")
        self.ic = f32(1.0 / self.P)
        self.shapes = p.theta_shapes(grads=False)
        self.ops: list[Op] = []
        self.s32 = f32(self.spec.scaling)
        c = self.cfg
        self.hc, self.dim, self.k = c.hc_mult, c.dim, c.n_activated_experts
        self.mix = (2 + c.hc_mult) * c.hc_mult
        self.base = {"num": "f32"}
        self.picks_from = None  # where attention's picks are: "q{j}.pk" (an index source's) or "r{j}.pk" (records')

    # --- bookkeeping ---

    def emit(self, kind, out, ins, attrs, shape, drops=()):
        self.ops.append(Op(kind, out, tuple(ins), {**self.base, **attrs}, tuple(drops), tuple(int(s) for s in shape),
                           tuple(self.shapes[i] for i in ins)))
        if out:
            self.shapes[out] = tuple(int(s) for s in shape)
        for d in drops:
            del self.shapes[d]

    def take(self, label: str) -> Node:
        ops, self.ops = self.ops, []
        return Node(label, ops)

    def free(self, label: str, names) -> Node:
        names = tuple(n for n in names if n in self.shapes)
        self.emit("free", "", (), {}, (0, 0), names)
        return self.take(label)

    def live(self, *prefixes) -> tuple[str, ...]:
        return tuple(n for n in self.shapes if n.startswith(prefixes))

    def candidates(self, j: int) -> bool:
        """Whether sequence j's index scores are masked by its candidate blocks (from the records): when it has more
        compressed tokens than the candidate source keeps (Indexer.forward; beyond 16K tokens in Flash)."""
        c = self.cfg
        return 0 <= c.candidate_source_layer < self.p.cut.first and             self.n_comp(j) > c.candidate_topk_blocks * c.candidate_block_size

    def n_comp(self, j: int) -> int:
        """Sequence j's compressed tokens, as the trained layers see them (Attention._side's s // r)."""
        r = self.cfg.ratio(self.p.cut.first)
        return self.T[j] // r if r else 0

    # --- the step ---

    def atoms(self) -> list:
        trained = list(self.p.cut.trained)
        out = [self.inputs()]
        for i in trained:
            out += self.forward_layer(i)
        out.append(self.head(trained[-1] + 1))
        for i in reversed(trained):
            out += self.backward_layer(i, below=i != trained[0])
        out.append(self.p.adam_step(self.t))
        return out

    def record(self, out, seqs, part, shape):
        self.emit("ds_record", out, (), {"seqs": [int(s) for s in seqs], "part": part}, shape)

    def inputs(self) -> Node:
        c, first, N = self.cfg, self.p.cut.first, self.N
        self.record(f"h.{first}", self.seqs, "x", (N, self.hc * self.dim))
        self.record(f"p.{first}", self.seqs, "pre", (N, self.hc))
        crossing = self.p.cut.crossing()
        for j, s in enumerate(self.seqs):
            n = self.n_comp(j)
            if c.ratio(first):
                self.record(f"r{j}.comp", [s], "compress_kv", (n, c.head_dim))
            if "index_k" in crossing and any(i in c.index_source_layers for i in self.p.cut.trained):
                self.record(f"r{j}.ik", [s], "index_k", (n, c.index_head_dim))
                if self.candidates(j):
                    self.record(f"r{j}.cand", [s], "candidates", (self.T[j], n))
            if first not in c.index_source_layers and c.ratio(first) and n:
                self.record(f"r{j}.pk", [s], "picks", (self.T[j], min(c.index_topk, n)))
                self.picks_from = "r"
        return self.take("the step's records")

    # --- projections, with their adapters ---

    def _mod(self, i, t) -> str:
        return f"layers.{i}.{LORA_MODULES[t]}"

    def proj(self, i, t, x, out, shape, drops=()):
        """A projection that LoRA may adapt (attention's four, the shared expert's three): ds_qlinear, plus the
        adapter's share (lora_down, lora_up) when it's trained."""
        name = self._mod(i, t)
        k = self.p.mod.get(name)
        if k is None:
            self.emit("ds_qlinear", out, (x,), {"weight": name}, shape, drops)
            return
        self.emit("ds_qlinear", f"a.{t}0", (x,), {"weight": name}, shape)
        self.emit("lora_down", f"a.{t}a", (x,), {"module": k}, (self.N, self.spec.rank), drops)
        self.emit("lora_up", out, (f"a.{t}0", f"a.{t}a", f"B.{name}"), {"scaling": self.s32}, shape, (f"a.{t}0",))

    def proj_bwd(self, i, t, dy, acc, out, shape, drops=()):
        """dx of projection `t` from dy (plus acc, which is dropped); its adapter's G gets the step's gradient."""
        name = self._mod(i, t)
        ins = (dy,) + ((acc,) if acc else ())
        first = (acc,) if acc else ()
        k = self.p.mod.get(name)
        if k is None:
            self.emit("ds_qlinear_bwd", out, ins, {"weight": name}, shape, first + tuple(drops))
            return
        _, o, _ = self.p.mods[k]
        r, g = self.spec.rank, f"G.{name}"
        self.emit("ds_qlinear_bwd", f"b.{t}x0", ins, {"weight": name}, shape, first)
        self.emit("scale", f"b.{t}s", (dy,), {"c": self.s32}, (self.N, o))
        self.emit("lora_grad", g, (f"b.{t}s", f"a.{t}a") + ((g,) if g in self.shapes else ()), {}, (o, r))
        self.emit("lora_bwd_mid", f"b.{t}m", (f"b.{t}s", f"B.{name}"), {}, (self.N, r), (f"b.{t}s",))
        self.emit("lora_bwd_x", out, (f"b.{t}m", f"b.{t}x0"), {"module": k}, shape,
                  (f"b.{t}m", f"b.{t}x0", f"a.{t}a") + tuple(drops))

    # --- a layer ---

    def _hc(self, i, part):
        c = self.cfg
        p = f"layers.{i}.hc_{part}_"
        return {"scale": p + "scale", "base": p + "base", "hc": c.hc_mult, "iters": c.hc_sinkhorn_iters, "eps": c.hc_eps}

    def _norm(self, i, name, width):
        return {"weight": f"layers.{i}.{name}.weight", "eps": self.cfg.norm_eps, "group": width}

    def _attn(self, i, j):
        c = self.cfg
        a = {"heads": c.n_heads, "window": c.window_size, "scale": f32(c.head_dim ** -0.5),
             "sink": f"layers.{i}.attn.attn_sink", "n_window": min(self.T[j], c.window_size)}
        return a | {"slots": a["n_window"] + self.n_picks(j)}

    def n_picks(self, j) -> int:
        n = self.n_comp(j)
        return min(self.cfg.index_topk, n) if self.cfg.ratio(self.p.cut.first) and n else 0

    def _rope(self, i, hd, inverse=False):
        return {"rope": list(self.cfg.rope_args(i)), "hd": hd, "stride": 1, "inverse": inverse}

    def attn_inputs(self, j):
        """(comp, picks) for sequence j's attention, or () when its layers don't compress."""
        if not self.n_picks(j):
            return ()
        return (f"r{j}.comp", f"{self.picks_from}{j}.pk")

    def forward(self, i, recompute: bool):
        """Layer i's forward from h.{i} and p.{i}. Without recompute: its output h.{i + 1} and p.{i + 1} (the ffn's
        split, whose pre columns the next layer collapses its input by), and nothing else of it kept. With recompute
        (in its backward): what its backward reads, everything but h.{i + 1}."""
        c, N, hc, dim, k = self.cfg, self.N, self.hc, self.dim, self.k
        H, hd = c.n_heads, c.head_dim
        x, pm = f"h.{i}", f"p.{i}"
        self.emit("ds_hc_mixes", "a.am", (x,), {"fn": f"layers.{i}.hc_attn_fn", "eps": c.norm_eps}, (N, self.mix))
        self.emit("ds_hc_split", "a.as", ("a.am",), self._hc(i, "attn"), (N, self.mix))
        self.emit("ds_hc_pre", "a.ap", (x, pm), {"col": 0}, (N, dim))
        self.emit("ds_rmsnorm", "a.x", ("a.ap",), self._norm(i, "attn_norm", dim), (N, dim))
        self.proj(i, "wq_a", "a.x", "a.qa", (N, c.q_lora_rank))
        self.emit("ds_rmsnorm", "a.qr", ("a.qa",), self._norm(i, "attn.q_norm", c.q_lora_rank), (N, c.q_lora_rank))
        self.proj(i, "wq_b", "a.qr", "a.q", (N, H * hd))
        self.proj(i, "wkv", "a.x", "a.k0", (N, hd))
        self.emit("ds_rmsnorm", "a.k1", ("a.k0",), self._norm(i, "attn.kv_norm", hd), (N, hd))
        index = i in c.index_source_layers and not recompute
        if index:
            IH, ID = c.index_n_heads, c.index_head_dim
            self.emit("ds_qlinear", "a.iq", ("a.qr",), {"weight": f"layers.{i}.attn.indexer.wq_b"}, (N, IH * ID))
            self.emit("ds_dense", "a.iw0", ("a.x",), {"weight": f"layers.{i}.attn.indexer.weights_proj.weight"}, (N, IH))
            self.emit("ds_scale", "a.iw", ("a.iw0",), {"c": f32(ID ** -0.5 * IH ** -0.5)}, (N, IH), ("a.iw0",))
        self.ops_attn = self.take(f"layer {i} before attention")
        per_seq = []
        for j, T in enumerate(self.T):
            o0, o1 = self.offs[j], self.offs[j + 1]
            q = f"q{j}."
            self.emit("ds_slice", q + "q", ("a.q",), {"r0": o0, "r1": o1}, (T, H * hd))
            self.emit("ds_rope", q + "qp", (q + "q",), self._rope(i, hd), (T, H * hd), (q + "q",))
            self.emit("ds_slice", q + "k", ("a.k1",), {"r0": o0, "r1": o1}, (T, hd))
            self.emit("ds_rope", q + "kv", (q + "k",), self._rope(i, hd), (T, hd), (q + "k",))
            if index and self.n_picks(j):
                IH, ID, n = c.index_n_heads, c.index_head_dim, self.n_comp(j)
                self.emit("ds_slice", q + "iq", ("a.iq",), {"r0": o0, "r1": o1}, (T, IH * ID))
                self.emit("ds_rope", q + "iqr", (q + "iq",), self._rope(i, ID), (T, IH * ID), (q + "iq",))
                self.emit("ds_slice", q + "iw", ("a.iw",), {"r0": o0, "r1": o1}, (T, IH))
                cand = (f"r{j}.cand",) if self.candidates(j) else ()
                self.emit("ds_index_scores", q + "isc", (q + "iqr", f"r{j}.ik", q + "iw") + cand,
                          {"heads": IH, "ratio": c.ratio(i)}, (T, n), (q + "iqr", q + "iw"))
                self.emit("ds_picks", q + "pk", (q + "isc",), {"ratio": c.ratio(i)}, (T, self.n_picks(j)), (q + "isc",))
                self.picks_from = "q"
            ins = (q + "qp", q + "kv") + self.attn_inputs(j)
            self.emit("ds_attn", q + "o", ins, self._attn(i, j), (T, H * hd), () if recompute else (q + "qp", q + "kv"))
            self.emit("ds_rope", q + "or", (q + "o",), self._rope(i, hd, inverse=True), (T, H * hd), (q + "o",))
            per_seq.append(self.take(f"layer {i} attention, sequence {j}"))
        drop_idx = ("a.iq", "a.iw") if index else ()
        self.emit("ds_cat", "a.o", tuple(f"q{j}.or" for j in range(len(self.T))), {}, (N, H * hd),
                  tuple(f"q{j}.or" for j in range(len(self.T))) + drop_idx + ("a.q", "a.k1"))
        G, R = c.o_groups, c.o_lora_rank
        self.emit("ds_qlinear", "a.oa", ("a.o",), {"weight": f"layers.{i}.attn.wo_a", "groups": G}, (N, G * R),
                  ("a.o",))
        self.proj(i, "wo_b", "a.oa", "a.y", (N, dim))
        self.emit("ds_hc_post", "a.xm", ("a.y", x, "a.as"), {"post": hc, "comb": 2 * hc}, (N, hc * dim))
        self.emit("ds_hc_mixes", "a.fm", ("a.xm",), {"fn": f"layers.{i}.hc_ffn_fn", "eps": c.norm_eps}, (N, self.mix))
        split = f"p.{i + 1}"
        if not recompute:
            self.emit("ds_hc_split", split, ("a.fm",), self._hc(i, "ffn"), (N, self.mix))
        self.emit("ds_hc_pre", "a.fp", ("a.xm", "a.as"), {"col": 0}, (N, dim))
        self.emit("ds_rmsnorm", "a.fx", ("a.fp",), self._norm(i, "ffn_norm", dim), (N, dim))
        # the router
        E, inter = c.n_routed_experts, c.moe_inter_dim
        self.emit("ds_dense", "a.lg", ("a.fx",), {"weight": f"layers.{i}.ffn.gate.weight", "hi": True}, (N, E))
        self.emit("ds_moe_scores", "a.sc", ("a.lg",), {"temp": f32(c.gate_temp), "func": c.score_func}, (N, E))
        self.emit("ds_bias", "a.bs", ("a.sc",), {"bias": f"layers.{i}.ffn.gate.bias"}, (N, E))
        self.emit("ds_experts", "a.ch", ("a.bs",), {}, (N, k), ("a.bs",))
        self.emit("ds_gates", "a.g", ("a.sc", "a.ch"),
                  {"norm": bool(c.norm_topk_prob), "route_scale": f32(c.route_scale)}, (N, k))
        self.emit("ds_reshape", "a.g1", ("a.g",), {}, (N * k, 1), ("a.g",))
        # the routed experts, by slot
        ex = f"layers.{i}.ffn.experts.{{e}}."
        lim = {"limit": f32(c.swiglu_limit)}
        self.emit("ds_moe_linear", "a.eg", ("a.fx", "a.ch"), {"weight": ex + "w1", "per_token": True}, (N * k, inter))
        self.emit("ds_moe_linear", "a.eu", ("a.fx", "a.ch"), {"weight": ex + "w3", "per_token": True}, (N * k, inter))
        self.emit("ds_swiglu", "a.eh", ("a.eg", "a.eu", "a.g1"), lim, (N * k, inter))
        self.emit("ds_moe_linear", "a.ey", ("a.eh", "a.ch"), {"weight": ex + "w2", "per_token": False}, (N * k, dim),
                  ("a.eh",))
        self.emit("ds_reshape", "a.rt", ("a.ey",), {}, (N, k * dim), ("a.ey",))
        # the shared expert
        self.proj(i, "w1", "a.fx", "a.sg", (N, inter))
        self.proj(i, "w3", "a.fx", "a.su", (N, inter))
        self.emit("ds_swiglu", "a.sh", ("a.sg", "a.su"), lim, (N, inter))
        self.proj(i, "w2", "a.sh", "a.sy", (N, dim), ("a.sh",))
        self.emit("ds_moe_sum", "a.mo", ("a.rt", "a.ch", "a.sy"), {}, (N, dim), ("a.rt", "a.sy"))
        if not recompute:
            self.emit("ds_hc_post", f"h.{i + 1}", ("a.mo", "a.xm", split), {"post": hc, "comb": 2 * hc}, (N, hc * dim),
                      self.live("a."))
        return per_seq

    def forward_layer(self, i) -> list:
        per_seq = self.forward(i, recompute=False)
        rest = self.take(f"layer {i} after attention")
        return [Node(f"layer {i} forward", [self.ops_attn, *per_seq, rest])]

    def head(self, L) -> Node:
        c, N, hc, dim = self.cfg, self.N, self.hc, self.dim
        P, V = self.P, c.vocab_size
        self.emit("ds_hc_pre", "f.hp", (f"h.{L}", f"p.{L}"), {"col": 0}, (N, dim))
        self.emit("ds_rmsnorm", "f.x", ("f.hp",), {"weight": "norm.weight", "eps": c.norm_eps, "group": dim}, (N, dim))
        self.record("f.mask", self.seqs, "mask", (N, 1))
        self.record("f.t", self.seqs, "targets", (P, 1))
        self.emit("ds_rows", "f.r", ("f.x", "f.mask"), {}, (P, dim), ("f.x",))
        blocks = [(b, min(V, b + HEAD_BLOCK)) for b in range(0, V, HEAD_BLOCK)]
        w = {"weight": "head.weight"}
        for b0, b1 in blocks:
            prev = ("hd.ms",) if "hd.ms" in self.shapes else ()
            self.emit("ds_head_stats", "hd.ms", ("f.r",) + prev, {**w, "b0": b0, "b1": b1}, (P, 2))
        self.emit("head_lse", "hd.lse", ("hd.ms",), {}, (P, 1), ("hd.ms",))
        for b0, b1 in blocks:
            prev = ("hd.g",) if "hd.g" in self.shapes else ()
            self.emit("ds_head_grad", "hd.g", ("f.r", "hd.lse", "f.t") + prev, {**w, "b0": b0, "b1": b1, "ic": self.ic},
                      (P, dim))
        self.emit("ds_unrows", "f.dy", ("hd.g", "f.mask"), {}, (N, dim), ("hd.g", "hd.lse", "f.r", "f.t", "f.mask"))
        self.emit("rmsnorm_bwd", "f.dhp", ("f.hp", "f.dy"), {"weight": "norm.weight", "group": dim, "eps": c.norm_eps},
                  (N, dim), ("f.dy",))
        self.emit("ds_hc_pre_dx", "g.h", ("f.dhp", f"p.{L}"), {"col": 0}, (N, hc * dim))
        self.emit("ds_hc_pre_dpre", "g.p", ("f.dhp", f"h.{L}"), {}, (N, hc), ("f.dhp", "f.hp", f"h.{L}"))
        return self.take("head and loss")

    def backward_layer(self, i, below: bool) -> list:
        """Layer i's backward: its forward again from its input, then back from g.h and g.p (the gradients of its
        output stream and of its ffn split's pre columns) to the adapters' G, and with `below` (a trained layer under
        it) to the new g.h and g.p of its input."""
        c, N, hc, dim, k = self.cfg, self.N, self.hc, self.dim, self.k
        H, hd = c.n_heads, c.head_dim
        x, pm, split = f"h.{i}", f"p.{i}", f"p.{i + 1}"
        per_seq_fwd = self.forward(i, recompute=True)
        fwd_rest = self.take(f"layer {i} after attention, again")
        recomputed = Node(f"layer {i} forward again", [self.ops_attn, *per_seq_fwd, fwd_rest])
        E, inter = c.n_routed_experts, c.moe_inter_dim
        lim = {"limit": f32(c.swiglu_limit)}
        hcs = {"hc": hc}
        # the ffn's hc_post: h.{i + 1} = hc_post(a.mo, a.xm, split)
        self.emit("ds_hc_post_dy", "b.dmo", ("g.h", split), {"post": hc}, (N, dim))
        self.emit("ds_hc_post_dres", "b.dxm", ("g.h", split), {"comb": 2 * hc, **hcs}, (N, hc * dim))
        self.emit("ds_hc_post_dsplit", "b.dfs", ("g.h", "a.mo", "a.xm", "g.p"), hcs, (N, self.mix),
                  ("g.h", "g.p", "a.mo"))
        # the shared expert
        self.proj_bwd(i, "w2", "b.dmo", None, "b.sdh", (N, inter))
        self.emit("ds_swiglu_bwd", "b.sdg", ("b.sdh", "a.sg", "a.su"), {**lim, "part": "g"}, (N, inter))
        self.emit("ds_swiglu_bwd", "b.sdu", ("b.sdh", "a.sg", "a.su"), {**lim, "part": "u"}, (N, inter), ("b.sdh",))
        self.proj_bwd(i, "w1", "b.sdg", None, "b.sx1", (N, dim), ("b.sdg", "a.sg"))
        self.proj_bwd(i, "w3", "b.sdu", "b.sx1", "b.dfx0", (N, dim), ("b.sdu", "a.su"))
        # the routed experts: each slot's output is added to its token's, so its gradient is the token's
        ex = f"layers.{i}.ffn.experts.{{e}}."
        self.emit("ds_moe_linear_bwd", "b.edh", ("b.dmo", "a.ch"), {"weight": ex + "w2", "per_token": True},
                  (N * k, inter), ("b.dmo",))
        self.emit("ds_swiglu_bwd", "b.edg", ("b.edh", "a.eg", "a.eu", "a.g1"), {**lim, "part": "g"}, (N * k, inter))
        self.emit("ds_swiglu_bwd", "b.edu", ("b.edh", "a.eg", "a.eu", "a.g1"), {**lim, "part": "u"}, (N * k, inter))
        self.emit("ds_swiglu_dgate", "b.edgt", ("b.edh", "a.eg", "a.eu"), lim, (N * k, 1), ("b.edh", "a.eg", "a.eu"))
        self.emit("ds_moe_linear_bwd", "b.ex1", ("b.edg", "a.ch"), {"weight": ex + "w1", "per_token": False},
                  (N * k, dim), ("b.edg",))
        self.emit("ds_moe_linear_bwd", "b.exs", ("b.edu", "a.ch", "b.ex1"), {"weight": ex + "w3", "per_token": False},
                  (N * k, dim), ("b.edu", "b.ex1"))
        self.emit("ds_moe_dx", "b.dfx1", ("b.exs", "a.ch", "b.dfx0"), {}, (N, dim), ("b.exs", "b.dfx0"))
        # the router (the bias steers the choice only: no gradient through it)
        self.emit("ds_reshape", "b.dg", ("b.edgt",), {}, (N, k), ("b.edgt", "a.g1"))
        self.emit("ds_gates_bwd", "b.dsc", ("a.sc", "a.ch", "b.dg"),
                  {"norm": bool(c.norm_topk_prob), "route_scale": f32(c.route_scale)}, (N, E), ("b.dg", "a.sc", "a.ch"))
        self.emit("ds_moe_scores_bwd", "b.dlg", ("a.lg", "b.dsc"), {"temp": f32(c.gate_temp)}, (N, E), ("b.dsc", "a.lg"))
        self.emit("ds_dense_bwd", "b.dfx", ("b.dlg", "b.dfx1"), {"weight": f"layers.{i}.ffn.gate.weight"}, (N, dim),
                  ("b.dlg", "b.dfx1", "a.fx"))
        # the ffn's norm and hc_pre (its input collapsed by the attention's split)
        self.emit("rmsnorm_bwd", "b.dfp", ("a.fp", "b.dfx"), self._norm(i, "ffn_norm", dim), (N, dim),
                  ("b.dfx", "a.fp"))
        self.emit("ds_hc_pre_dx", "b.dxm2", ("b.dfp", "a.as", "b.dxm"), {"col": 0}, (N, hc * dim), ("b.dxm",))
        self.emit("ds_hc_pre_dpre", "b.dap", ("b.dfp", "a.xm"), {}, (N, hc), ("b.dfp",))
        # the ffn's mixes
        self.emit("ds_hc_split_trace", "b.ftr", ("a.fm",), self._hc(i, "ffn"), (N, 16 * (2 * c.hc_sinkhorn_iters + 1)))
        self.emit("ds_hc_split_bwd", "b.dfm", ("b.dfs", "a.fm", "b.ftr"), self._hc(i, "ffn"), (N, self.mix),
                  ("b.dfs", "b.ftr"))
        self.emit("ds_hc_mixes_bwd", "b.dxm3", ("b.dfm", "a.xm", "b.dxm2"), {"fn": f"layers.{i}.hc_ffn_fn",
                                                                          "eps": c.norm_eps}, (N, hc * dim),
                  ("b.dfm", "b.dxm2", "a.fm"))
        # the attention's hc_post: a.xm = hc_post(a.y, h.{i}, a.as)
        self.emit("ds_hc_post_dy", "b.dy", ("b.dxm3", "a.as"), {"post": hc}, (N, dim))
        if below:
            self.emit("ds_hc_post_dres", "b.dres", ("b.dxm3", "a.as"), {"comb": 2 * hc, **hcs}, (N, hc * dim))
            self.emit("ds_hc_post_dsplit", "b.das", ("b.dxm3", "a.y", x, "b.dap"), hcs, (N, self.mix),
                      ("b.dxm3", "b.dap"))
        else:  # the first trained layer's input is the records': nothing to send down
            self.emit("free", "", (), {}, (0, 0), ("b.dxm3", "b.dap"))
        self.emit("free", "", (), {}, (0, 0), ("a.y", "a.xm"))
        after = self.take(f"layer {i} back to the attention")
        # the attention's output projections
        G, R = c.o_groups, c.o_lora_rank
        self.proj_bwd(i, "wo_b", "b.dy", None, "b.doa", (N, G * R), ("b.dy", "a.oa"))
        self.emit("ds_qlinear_bwd", "b.dor", ("b.doa",), {"weight": f"layers.{i}.attn.wo_a", "groups": G}, (N, H * hd),
                  ("b.doa",))
        before_attn = self.take(f"layer {i} back through the output projections")
        per_seq = []
        for j, T in enumerate(self.T):
            o0, o1 = self.offs[j], self.offs[j + 1]
            q = f"q{j}."
            a = self._attn(i, j)
            S = a["slots"]
            keys = (q + "kv",) + self.attn_inputs(j)
            self.emit("ds_slice", q + "dor", ("b.dor",), {"r0": o0, "r1": o1}, (T, H * hd))
            self.emit("ds_rope", q + "do", (q + "dor",), self._rope(i, hd), (T, H * hd), (q + "dor",))
            self.emit("ds_attn_probs", q + "P", (q + "qp",) + keys, a, (T, H * S))
            self.emit("ds_attn_dp", q + "dP", (q + "do",) + keys, a, (T, H * S))
            self.emit("ds_attn_ds", q + "dS", (q + "P", q + "dP"), {"slots": S}, (T, H * S), (q + "dP",))
            self.emit("ds_attn_dq", q + "dq", (q + "dS",) + keys, a, (T, H * hd))
            self.emit("ds_attn_dkv", q + "dkv", (q + "dS", q + "P", q + "qp", q + "do"), a, (T, hd),
                      (q + "dS", q + "P", q + "qp", q + "do", q + "kv"))
            self.emit("ds_rope", q + "dqr", (q + "dq",), self._rope(i, hd, inverse=True), (T, H * hd), (q + "dq",))
            self.emit("ds_rope", q + "dkr", (q + "dkv",), self._rope(i, hd, inverse=True), (T, hd), (q + "dkv",))
            per_seq.append(self.take(f"layer {i} attention back, sequence {j}"))
        n = len(self.T)
        self.emit("ds_cat", "b.dq", tuple(f"q{j}.dqr" for j in range(n)), {}, (N, H * hd),
                  tuple(f"q{j}.dqr" for j in range(n)) + ("b.dor",))
        self.emit("ds_cat", "b.dk", tuple(f"q{j}.dkr" for j in range(n)), {}, (N, hd), tuple(f"q{j}.dkr" for j in range(n)))
        # the queries' and keys' projections and norms
        self.proj_bwd(i, "wq_b", "b.dq", None, "b.dqa", (N, c.q_lora_rank), ("b.dq", "a.qr"))
        self.emit("rmsnorm_bwd", "b.dqa2", ("a.qa", "b.dqa"), self._norm(i, "attn.q_norm", c.q_lora_rank),
                  (N, c.q_lora_rank), ("b.dqa", "a.qa"))
        self.proj_bwd(i, "wq_a", "b.dqa2", None, "b.dxq", (N, dim), ("b.dqa2",))
        self.emit("rmsnorm_bwd", "b.dk2", ("a.k0", "b.dk"), self._norm(i, "attn.kv_norm", hd), (N, hd),
                  ("b.dk", "a.k0"))
        self.proj_bwd(i, "wkv", "b.dk2", "b.dxq", "b.dx", (N, dim), ("b.dk2", "a.x"))
        if below:
            self.emit("rmsnorm_bwd", "b.dap2", ("a.ap", "b.dx"), self._norm(i, "attn_norm", dim), (N, dim),
                      ("b.dx", "a.ap"))
            self.emit("ds_hc_pre_dx", "b.dh", ("b.dap2", pm, "b.dres"), {"col": 0}, (N, hc * dim), ("b.dres",))
            self.emit("ds_hc_pre_dpre", "g.p", ("b.dap2", x), {}, (N, hc), ("b.dap2",))
            self.emit("ds_hc_split_trace", "b.atr", ("a.am",), self._hc(i, "attn"), (N, 16 * (2 * c.hc_sinkhorn_iters + 1)))
            self.emit("ds_hc_split_bwd", "b.dam", ("b.das", "a.am", "b.atr"), self._hc(i, "attn"), (N, self.mix),
                      ("b.das", "b.atr"))
            self.emit("ds_hc_mixes_bwd", "g.h", ("b.dam", x, "b.dh"), {"fn": f"layers.{i}.hc_attn_fn", "eps": c.norm_eps},
                      (N, hc * dim), ("b.dam", "b.dh"))
        # what's left of the layer: its activations, its input (the layer below recomputes from its own), and its
        # split, which only its own backward reads; the picks once their index source is done with; for the first
        # trained layer, the step's records too
        done = tuple(n for n in self.live("a.", "b.", "q") if not n.endswith(".pk")) + (x, split)
        if i in c.index_source_layers or not below:
            done += tuple(n for n in self.live("q") if n.endswith(".pk"))
        if not below:
            done += (pm,) + self.live("r")
        rest = self.take(f"layer {i} back through the queries and keys")
        return [Node(f"layer {i} backward", [recomputed, after, before_attn, *per_seq, rest,
                                             self.free(f"layer {i} done", done)])]


# --- a prep unit ------------------------------------------------------------------------------------------------------
# A layers unit of the split prep (deepseek_graph.plan): frozen layers [a, b) over one chunk's sequences, in the
# reference's arithmetic (num "ref": bf16 activations, FP8 and FP4 rounding as ops of their own, where its kernels
# round). Its segments are its layers (and, for the first piece, the embedding before them), so a dispute bisects a
# unit layer by layer first. Between layers the state is canonical, whatever the pieces: the stream "h" [N, hc·dim],
# the pre_mix "p" [N, hc], and each sequence's side channels that cross the cut above the layer (Cut.crossing): what
# the layers above it still read. So one unit's last state is the next one's first, and a state's root after a layer
# is the same however the stack is cut into pieces.

SIDE_PREFIX = {"compress_kv": "kv", "index_k": "ik", "picks": "pk", "candidates": "cand"}


class PrepProgram:
    """The tree of one prep layers unit (deepseek_graph.Unit, kind "layers") of `cut` over chunk `members` (sequence
    ids) of the token file: its segments, the embedding first if it starts at layer 0, then each of its layers;
    unit() is all of them. `data` has .tokens (the token file), which ds_tokens reads."""

    def __init__(self, cut, data, members, unit):
        self.cut, self.data, self.unit_, self.cfg = cut, data, unit, cut.cfg
        self.seqs = [int(s) for s in members]
        self.T = [int(data.tokens.lengths[s]) for s in self.seqs]
        self.offs = np.concatenate([[0], np.cumsum(self.T)]).astype(np.int64).tolist()
        self.N = int(self.offs[-1])
        self.a, self.b = unit.a, unit.b
        self.layers = list(range(unit.a, unit.b))
        self.first = unit.a == 0
        self._built = None

    @classmethod
    def of(cls, spec, data, unit) -> "PrepProgram":
        """A job's prep unit (deepseek_job.DeepSeekSpec, its data, the graph unit)."""
        return cls(spec.top, data, data.tokens.chunks(spec.chunk_tokens)[unit.chunk], unit)

    @property
    def segments(self) -> int:
        return len(self.layers) + self.first

    def _build(self) -> list:
        if self._built is None:
            self._built = _PrepBuild(self).atoms()
        return self._built

    def segment(self, s: int):
        return self._build()[s]

    def unit(self):
        return Node("unit", lambda: list(self._build()))

    def start_shapes(self) -> dict:
        """The state the unit starts from (the one the unit below hands on), name -> shape: empty at layer 0."""
        return _PrepBuild(self).start_shapes()

    def side_kinds(self, layer: int) -> dict:
        """The side channels in the state entering `layer`: Cut.crossing's {name: source} at it."""
        from .deepseek_config import Cut
        if layer == 0:
            return {}
        return {k: v for k, v in Cut(self.cfg, layer).crossing().items() if k in SIDE_PREFIX}

    def n_comp(self, j: int, src: int) -> int:
        r = self.cfg.ratio(src)
        return self.T[j] // r if r else 0

    def kept(self, layer: int, kind: str, j: int, src: int) -> bool:
        """Whether sequence j's side channel `kind` (from layer src) is in the state entering `layer`. At the cut, the
        state is what the records hold (deepseek_top.crossing_of): candidate blocks only for a sequence with more
        compressed tokens than the candidate source keeps."""
        if not self.n_comp(j, src):
            return False
        c = self.cfg
        return not (layer == self.cut.first and kind == "candidates"
                    and self.n_comp(j, src) <= c.candidate_topk_blocks * c.candidate_block_size)

    def side_shape(self, kind: str, j: int, src: int) -> tuple[int, int]:
        c, n = self.cfg, self.n_comp(j, src)
        return {"compress_kv": (n, c.head_dim), "index_k": (n, c.index_head_dim),
                "picks": (self.T[j], min(c.index_topk, n)), "candidates": (self.T[j], n)}[kind]


class _PrepBuild:
    def __init__(self, p: PrepProgram):
        self.p, self.cfg = p, p.cfg
        self.N, self.T, self.offs = p.N, p.T, p.offs
        c = self.cfg
        self.hc, self.dim, self.k = c.hc_mult, c.dim, c.n_activated_experts
        self.mix = (2 + c.hc_mult) * c.hc_mult
        self.base = {"num": "ref"}
        self.ops: list[Op] = []
        self.shapes = self.start_shapes()

    def n_comp(self, j, src) -> int:
        return self.p.n_comp(j, src)

    def start_shapes(self) -> dict:
        p, c = self.p, self.cfg
        if p.first:
            return {}
        out = {"h": (self.N, c.hc_mult * c.dim), "p": (self.N, c.hc_mult)}
        for kind, src in p.side_kinds(p.a).items():
            for j in range(len(self.T)):
                if p.kept(p.a, kind, j, src):
                    out[f"{SIDE_PREFIX[kind]}{j}"] = p.side_shape(kind, j, src)
        return out

    def emit(self, kind, out, ins, attrs, shape, drops=()):
        self.ops.append(Op(kind, out, tuple(ins), {**self.base, **attrs}, tuple(drops), tuple(int(s) for s in shape),
                           tuple(self.shapes[i] for i in ins)))
        if out:
            self.shapes[out] = tuple(int(s) for s in shape)
        for d in drops:
            del self.shapes[d]

    def take(self, label: str) -> Node:
        ops, self.ops = self.ops, []
        return Node(label, ops)

    def live(self, *prefixes):
        return tuple(n for n in self.shapes if n.startswith(prefixes))

    def round8(self, x, out):
        self.emit("ds_round_fp8", out, (x,), {}, self.shapes[x])

    # --- the unit ---

    def atoms(self) -> list:
        out = []
        if self.p.first:
            out.append(self.embedding())
        for i in self.p.layers:
            out.append(self.layer(i))
        return out

    def embedding(self) -> Node:
        c, N = self.cfg, self.N
        self.emit("ds_tokens", "e.tok", (), {"seqs": self.p.seqs}, (N, 1))
        self.emit("ds_embed", "e.x", ("e.tok",), {"weight": "embed.weight"}, (N, c.dim), ("e.tok",))
        self.emit("ds_tile", "h", ("e.x",), {"n": c.hc_mult}, (N, c.hc_mult * c.dim), ("e.x",))
        self.emit("ds_onehot", "p", (), {"col": 0}, (N, c.hc_mult))
        return self.take("the embedding")

    def _norm(self, i, name, width):
        return {"weight": f"layers.{i}.{name}.weight", "eps": self.cfg.norm_eps, "group": width}

    def _hc(self, i, part):
        c = self.cfg
        p = f"layers.{i}.hc_{part}_"
        return {"scale": p + "scale", "base": p + "base", "hc": c.hc_mult, "iters": c.hc_sinkhorn_iters, "eps": c.hc_eps}

    def _rope(self, i, hd, stride=1, inverse=False):
        return {"rope": list(self.cfg.rope_args(i)), "hd": hd, "stride": stride, "inverse": inverse}

    def _seq(self, src, j, out, width):
        o0, o1 = self.offs[j], self.offs[j + 1]
        self.emit("ds_slice", out, (src,), {"r0": o0, "r1": o1}, (self.T[j], width))

    def layer(self, i) -> Node:
        """Block.forward of layer i over the chunk, from the canonical state, back to it."""
        c, N, hc, dim, k = self.cfg, self.N, self.hc, self.dim, self.k
        H, hd = c.n_heads, c.head_dim
        n = len(self.T)
        parts = []
        if i in c.engram_layer_ids:
            parts.append(self.engram(i))
        self.emit("ds_hc_mixes", "a.am", ("h",), {"fn": f"layers.{i}.hc_attn_fn", "eps": c.norm_eps}, (N, self.mix))
        self.emit("ds_hc_split", "a.as", ("a.am",), self._hc(i, "attn"), (N, self.mix), ("a.am",))
        self.emit("ds_hc_pre", "a.ap", ("h", "p"), {"col": 0}, (N, dim))
        self.emit("ds_rmsnorm", "a.x", ("a.ap",), self._norm(i, "attn_norm", dim), (N, dim), ("a.ap",))
        self.round8("a.x", "a.xq")
        self.emit("ds_qlinear", "a.qa", ("a.xq",), {"weight": f"layers.{i}.attn.wq_a"}, (N, c.q_lora_rank))
        self.emit("ds_rmsnorm", "a.qr", ("a.qa",), self._norm(i, "attn.q_norm", c.q_lora_rank), (N, c.q_lora_rank),
                  ("a.qa",))
        self.round8("a.qr", "a.qrq")
        self.emit("ds_qlinear", "a.q", ("a.qrq",), {"weight": f"layers.{i}.attn.wq_b"}, (N, H * hd))
        self.emit("ds_qlinear", "a.k0", ("a.xq",), {"weight": f"layers.{i}.attn.wkv"}, (N, hd), ("a.xq",))
        self.emit("ds_rmsnorm", "a.k1", ("a.k0",), self._norm(i, "attn.kv_norm", hd), (N, hd), ("a.k0",))
        r = c.ratio(i)
        index = bool(r) and i in c.index_source_layers
        if index:
            IH, ID = c.index_n_heads, c.index_head_dim
            self.emit("ds_qlinear", "a.iq", ("a.qrq",), {"weight": f"layers.{i}.attn.indexer.wq_b"}, (N, IH * ID))
            self.emit("ds_dense", "a.iw0", ("a.x",), {"weight": f"layers.{i}.attn.indexer.weights_proj.weight"}, (N, IH))
            self.emit("ds_scale", "a.iw", ("a.iw0",), {"c": f32(ID ** -0.5 * IH ** -0.5)}, (N, IH), ("a.iw0",))
        self.emit("free", "", (), {}, (0, 0), ("a.qrq",))
        if r and i in c.kv_source_layers and r == 1:
            self.emit("ds_dense", "a.l0", ("a.x",), {"weight": f"layers.{i}.attn.compressor.wkv.weight"}, (N, hd))
        parts.append(self.take(f"layer {i} before attention"))
        for j, T in enumerate(self.T):
            q = f"q{j}."
            nj = self.n_comp(j, i) if r else 0
            if r and i in c.kv_source_layers and nj:  # its compressed keys (and index keys): Compressor, Indexer.keys
                if r == 1:
                    self._seq("a.l0", j, q + "l0", hd)
                else:
                    self._seq("a.x", j, q + "xs", dim)
                    self.emit("ds_compress", q + "l0", (q + "xs",),
                              {"wkv": f"layers.{i}.attn.compressor.wkv.weight",
                               "wgate": f"layers.{i}.attn.compressor.wgate.weight", "ratio": r}, (nj, hd), (q + "xs",))
                self.emit("ds_rmsnorm", q + "lat", (q + "l0",), self._norm(i, "attn.compressor.norm", hd), (nj, hd),
                          (q + "l0",))
                if index:
                    ID = c.index_head_dim
                    self.emit("ds_dense", q + "ik0", (q + "lat",), {"weight": f"layers.{i}.attn.indexer.wk.weight"},
                              (nj, ID))
                    self.emit("ds_rmsnorm", q + "ik1", (q + "ik0",), self._norm(i, "attn.indexer.k_norm", ID), (nj, ID),
                              (q + "ik0",))
                    self.emit("ds_rope", q + "ik2", (q + "ik1",), self._rope(i, ID, stride=r), (nj, ID), (q + "ik1",))
                    self.emit("ds_round_fp4", f"ik{j}", (q + "ik2",), {"block": 32, "e4m3": False}, (nj, ID),
                              (q + "ik2",))
                self.emit("ds_rope", q + "kr", (q + "lat",), self._rope(i, hd, stride=r), (nj, hd), (q + "lat",))
                self.emit("ds_round_fp4", f"kv{j}", (q + "kr",), {"block": 16, "e4m3": True}, (nj, hd), (q + "kr",))
            self._seq("a.q", j, q + "q", H * hd)
            self.emit("ds_rope", q + "qp", (q + "q",), self._rope(i, hd), (T, H * hd), (q + "q",))
            self._seq("a.k1", j, q + "k", hd)
            self.emit("ds_rope", q + "kr2", (q + "k",), self._rope(i, hd), (T, hd), (q + "k",))
            self.round8(q + "kr2", q + "kv")
            self.emit("free", "", (), {}, (0, 0), (q + "kr2",))
            if index and nj:
                IH, ID = c.index_n_heads, c.index_head_dim
                self._seq("a.iq", j, q + "iq", IH * ID)
                self.emit("ds_rope", q + "iq1", (q + "iq",), self._rope(i, ID), (T, IH * ID), (q + "iq",))
                self.emit("ds_round_fp4", q + "iq2", (q + "iq1",), {"block": 32, "e4m3": False}, (T, IH * ID),
                          (q + "iq1",))
                self._seq("a.iw", j, q + "iw", IH)
                cand = (f"cand{j}",) if 0 <= c.candidate_source_layer < i and f"cand{j}" in self.shapes else ()
                self.emit("ds_index_scores", q + "isc", (q + "iq2", f"ik{j}", q + "iw") + cand,
                          {"heads": IH, "ratio": r}, (T, nj), (q + "iq2", q + "iw"))
                if i == c.candidate_source_layer:
                    self.emit("ds_candidates", f"cand{j}", (q + "isc",),
                              {"block": c.candidate_block_size, "topk_blocks": c.candidate_topk_blocks, "ratio": r},
                              (T, nj))
                self.emit("ds_picks", f"pk{j}", (q + "isc",), {"ratio": r}, (T, min(c.index_topk, nj)), (q + "isc",))
            kv_n = self.shapes.get(f"kv{j}", (0,))[0] if r else 0
            extra = (f"kv{j}", f"pk{j}") if r and kv_n and f"pk{j}" in self.shapes else ()
            att = {"heads": H, "window": c.window_size, "scale": f32(hd ** -0.5), "sink": f"layers.{i}.attn.attn_sink",
                   "n_window": min(T, c.window_size)}
            self.emit("ds_attn", q + "o", (q + "qp", q + "kv") + extra, att, (T, H * hd), (q + "qp", q + "kv"))
            self.emit("ds_rope", q + "or", (q + "o",), self._rope(i, hd, inverse=True), (T, H * hd), (q + "o",))
            parts.append(self.take(f"layer {i} attention, sequence {j}"))
        drops = ("a.q", "a.k1", "a.x") + (("a.iq", "a.iw") if index else ()) + self.live("a.l0")
        self.emit("ds_cat", "a.o", tuple(f"q{j}.or" for j in range(n)), {}, (N, H * hd),
                  tuple(f"q{j}.or" for j in range(n)) + drops)
        G, R = c.o_groups, c.o_lora_rank
        self.round8("a.o", "a.oq")
        self.emit("ds_qlinear", "a.oa", ("a.oq",), {"weight": f"layers.{i}.attn.wo_a", "groups": G}, (N, G * R),
                  ("a.o", "a.oq"))
        self.round8("a.oa", "a.oaq")
        self.emit("ds_qlinear", "a.y", ("a.oaq",), {"weight": f"layers.{i}.attn.wo_b"}, (N, dim), ("a.oa", "a.oaq"))
        self.emit("ds_hc_post", "a.xm", ("a.y", "h", "a.as"), {"post": hc, "comb": 2 * hc}, (N, hc * dim), ("a.y",))
        self.emit("ds_hc_mixes", "a.fm", ("a.xm",), {"fn": f"layers.{i}.hc_ffn_fn", "eps": c.norm_eps}, (N, self.mix))
        self.emit("ds_hc_split", "a.fs", ("a.fm",), self._hc(i, "ffn"), (N, self.mix), ("a.fm",))
        self.emit("ds_hc_pre", "a.fp", ("a.xm", "a.as"), {"col": 0}, (N, dim), ("a.as",))
        self.emit("ds_rmsnorm", "a.fx", ("a.fp",), self._norm(i, "ffn_norm", dim), (N, dim), ("a.fp",))
        E, inter = c.n_routed_experts, c.moe_inter_dim
        self.emit("ds_dense", "a.lg", ("a.fx",), {"weight": f"layers.{i}.ffn.gate.weight", "hi": True}, (N, E))
        self.emit("ds_moe_scores", "a.sc", ("a.lg",), {"temp": f32(c.gate_temp), "func": c.score_func}, (N, E),
                  ("a.lg",))
        self.emit("ds_bias", "a.bs", ("a.sc",), {"bias": f"layers.{i}.ffn.gate.bias"}, (N, E))
        self.emit("ds_experts", "a.ch", ("a.bs",), {}, (N, k), ("a.bs",))
        self.emit("ds_gates", "a.g", ("a.sc", "a.ch"),
                  {"norm": bool(c.norm_topk_prob), "route_scale": f32(c.route_scale)}, (N, k), ("a.sc",))
        self.emit("ds_reshape", "a.g1", ("a.g",), {}, (N * k, 1), ("a.g",))
        self.round8("a.fx", "a.fxq")
        ex = f"layers.{i}.ffn.experts.{{e}}."
        lim = {"limit": f32(c.swiglu_limit)}
        self.emit("ds_moe_linear", "a.eg", ("a.fxq", "a.ch"), {"weight": ex + "w1", "per_token": True}, (N * k, inter))
        self.emit("ds_moe_linear", "a.eu", ("a.fxq", "a.ch"), {"weight": ex + "w3", "per_token": True}, (N * k, inter))
        self.emit("ds_swiglu", "a.eh", ("a.eg", "a.eu", "a.g1"), lim, (N * k, inter), ("a.eg", "a.eu", "a.g1"))
        self.round8("a.eh", "a.ehq")
        self.emit("ds_moe_linear", "a.ey", ("a.ehq", "a.ch"), {"weight": ex + "w2", "per_token": False}, (N * k, dim),
                  ("a.eh", "a.ehq"))
        self.emit("ds_reshape", "a.rt", ("a.ey",), {}, (N, k * dim), ("a.ey",))
        sh = f"layers.{i}.ffn.shared_experts."
        self.emit("ds_qlinear", "a.sg", ("a.fxq",), {"weight": sh + "w1"}, (N, inter))
        self.emit("ds_qlinear", "a.su", ("a.fxq",), {"weight": sh + "w3"}, (N, inter), ("a.fxq", "a.fx"))
        self.emit("ds_swiglu", "a.sh", ("a.sg", "a.su"), lim, (N, inter), ("a.sg", "a.su"))
        self.round8("a.sh", "a.shq")
        self.emit("ds_qlinear", "a.sy", ("a.shq",), {"weight": sh + "w2"}, (N, dim), ("a.sh", "a.shq"))
        self.emit("ds_moe_sum", "a.mo", ("a.rt", "a.ch", "a.sy"), {}, (N, dim), ("a.rt", "a.ch", "a.sy"))
        self.emit("ds_hc_post", "h", ("a.mo", "a.xm", "a.fs"), {"post": hc, "comb": 2 * hc}, (N, hc * dim),
                  ("a.mo", "a.xm"))
        self.emit("ds_cols", "p", ("a.fs",), {"c0": 0, "c1": hc}, (N, hc), ("a.fs",))
        # the canonical state: what crosses the cut above this layer, nothing else
        keep = {f"{SIDE_PREFIX[kd]}{j}" for kd, src in self.p.side_kinds(i + 1).items() for j in range(n)
                if self.p.kept(i + 1, kd, j, src)}
        gone = tuple(nm for nm in self.shapes if nm not in ("h", "p") and nm not in keep)
        if gone:
            self.emit("free", "", (), {}, (0, 0), gone)
        parts.append(self.take(f"layer {i} after attention"))
        return Node(f"layer {i}", parts)

    def engram(self, i) -> Node:
        """Engram.forward before layer i's block: each token's n-gram rows, its wkv, and the gate into the stream."""
        from .deepseek_config import EngramLayout

        c, N, dim = self.cfg, self.N, self.dim
        idx = c.engram_layer_ids.index(i)
        layout = EngramLayout.of(c)
        flat = [p for size in layout.primes[idx] for p in size]
        cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads
        attrs = {"n": c.engram_max_ngram_size, "multipliers": list(layout.multipliers[idx]),
                 "primes": [list(p) for p in layout.primes[idx]], "offsets": np.cumsum([0, *flat[:-1]]).tolist(),
                 "pad_id": int(c.engram_pad_id), "token_map": "engram.token_map", "table": f"layers.{i}.engram.embed"}
        self.emit("ds_tokens", "g.tok", (), {"seqs": self.p.seqs}, (N, 1))
        for j, T in enumerate(self.T):
            self._seq("g.tok", j, f"g{j}.tok", 1)
            self.emit("ds_engram_rows", f"g{j}.rows", (f"g{j}.tok",), {**attrs, "layer": i}, (T, cols * c.engram_head_dim),
                      (f"g{j}.tok",))
        self.emit("ds_cat", "g.rows", tuple(f"g{j}.rows" for j in range(len(self.T))), {},
                  (N, cols * c.engram_head_dim), tuple(f"g{j}.rows" for j in range(len(self.T))) + ("g.tok",))
        self.round8("g.rows", "g.rq")
        self.emit("ds_qlinear", "g.kv", ("g.rq",), {"weight": f"layers.{i}.engram.wkv"}, (N, (c.hc_mult + 1) * dim),
                  ("g.rows", "g.rq"))
        self.emit("ds_engram_gate", "h", ("h", "g.kv"), {"q_weight": f"layers.{i}.engram.q_weight",
                                                       "k_weight": f"layers.{i}.engram.k_weight", "eps": c.norm_eps},
                  (N, c.hc_mult * dim), ("g.kv",))
        return self.take(f"layer {i} engram")
