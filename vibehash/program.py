"""What one unit of a fine-tune computes, op by op: the program a worker trains by (tracer.py runs it in PyTorch) and a
node referees a dispute by (dispute.py, referee.py), without PyTorch.

A unit is a tree. Its children are its segments; a segment's, its Adam steps; a step's, one sequence of its batch at a
time (a micro-batch, adding that sequence's gradients into G) and then the Adam update; a micro-batch's, the embedding,
each layer forward, the head and loss, and each layer backward; and those are lists of ops. Every op reads named 2-D
tensors of the state and writes one (merkle.py hashes a state), so the state between any two ops is well defined, and
any op can be checked on its own given the state before it.

How the arithmetic goes is the same as llm.py's model (float32, frozen weights cast exactly, attention written out),
with the backward written out too, in place of autograd's: the referee has to know each op's arithmetic. A layer's
activations are dropped after its forward and recomputed in its backward, as llm.py's activation checkpointing does.
The head works through the vocabulary VOCAB_BLOCK words at a time, keeping a running max and sum for the log-softmax,
so no op needs more than a block of the head's weights.
"""
from __future__ import annotations

import functools
import math
import types
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .arch import EMBED, HEAD, MODULES, NORM, Arch
from .finetune import OPTIMIZERS, ipow
from .model import batch_indices

K = 64  # most children one split has
VOCAB_BLOCK = 4096  # words of the vocabulary a head op covers: a block of the head is 8 MB (bf16, 1024 wide) to 20 MB


def f32(x: float) -> float:
    return float(np.float32(x))


@dataclass
class Op:
    kind: str
    out: str  # "" for "free", which only drops tensors
    ins: tuple[str, ...] = ()
    attrs: dict = field(default_factory=dict)
    drops: tuple[str, ...] = ()
    shape: tuple[int, int] = (0, 0)  # the output's [rows, cols]
    ishapes: tuple[tuple[int, int], ...] = ()  # each input's

    def describe(self) -> str:
        return f"{self.kind} {', '.join(self.ins)} → {self.out or '(frees ' + ', '.join(self.drops) + ')'}"


class Node:
    """A part of the tree: its atoms (children) are Ops and Nodes, built when first asked for."""

    def __init__(self, label: str, build: Callable[[], list] | list):
        self.label = label
        self._build, self._atoms = (build, None) if callable(build) else (None, list(build))

    @property
    def atoms(self) -> list:
        if self._atoms is None:
            self._atoms = self._build()
        return self._atoms

    def ops(self):
        """Every op under this node, in order."""
        for a in self.atoms:
            if isinstance(a, Op):
                yield a
            else:
                yield from a.ops()


def groups(lo: int, hi: int, k: int = K) -> list[tuple[int, int]]:
    """[lo, hi) cut into min(k, hi − lo) runs as even as can be."""
    n = hi - lo
    m = min(k, n)
    return [(lo + n * i // m, lo + n * (i + 1) // m) for i in range(m)]


def module_names(spec) -> list[tuple[str, int, int]]:
    return spec.modules()


def split_theta(spec, theta: np.ndarray) -> dict[str, np.ndarray]:
    """A checkpoint [B | m | v] as the state's tensors: B, m and v of each adapted module, [out, rank] float32."""
    theta = np.asarray(theta, dtype=np.float32)
    pb, pm, pv = spec.parts
    out, at = {}, 0
    for name, o, _ in spec.modules():
        n = o * spec.rank
        for prefix, part in (("B", pb), ("m", pm), ("v", pv)):
            out[f"{prefix}.{name}"] = theta[part][at:at + n].reshape(o, spec.rank)
        at += n
    return out


def join_theta(spec, state: dict) -> np.ndarray:
    """The checkpoint vector a state's B, m and v make."""
    parts = []
    for prefix in ("B", "m", "v"):
        parts += [np.asarray(state[f"{prefix}.{name}"], dtype=np.float32).reshape(-1) for name, _, _ in spec.modules()]
    return np.concatenate(parts).astype(np.float64)


class Program:
    """The tree of one unit of a fine-tune whose seed is `seed`: `segments` segments of `per_segment` Adam steps at
    learning rate `lr`, each step on `batch_size` sequences of `data`."""

    def __init__(self, spec, data, lr: float, batch_size: int, seed: int, segments: int = 1, per_segment: int = 1):
        self.spec, self.data, self.seed = spec, data, seed
        self.lr, self.batch_size, self.segments, self.per_segment = lr, batch_size, segments, per_segment
        self.arch = Arch.from_hf_config(self.data.parsed_config())
        self.mods = self.spec.modules()
        self.mod = {}
        for k, (name, _, _) in enumerate(self.mods):
            layer, rest = name.split(".", 3)[2], name.split(".", 3)[3]
            target = next(t for t, m in MODULES.items() if m == rest)
            self.mod[(int(layer), target)] = k
        b1, b2, eps = (f32(x) for x in OPTIMIZERS[self.spec.optimizer])
        self.adam = {"b1": b1, "b2": b2, "eps": eps, "c1m": f32(1 - b1), "c2m": f32(1 - b2), "lr": f32(lr)}
        self.head_weight = EMBED if self.arch.tied else HEAD

    @classmethod
    def of(cls, job, seed: int) -> "Program":
        if getattr(job, "deepseek", False):  # a DeepSeek top-layers job's units (deepseek_program.py)
            from .deepseek_program import DeepSeekProgram
            cls = DeepSeekProgram
        return cls(job.spec, job.data, job.lr, job.batch_size, seed, job.n_segments, job.checkpoint_every)

    # --- the levels ---

    def unit(self) -> Node:
        return Node("unit", lambda: [self.segment(s) for s in range(self.segments)])

    def segment(self, s: int) -> Node:
        c = self.per_segment
        return Node(f"segment {s}", lambda: [self.step(t) for t in range(s * c, (s + 1) * c)])

    def batch(self, t: int) -> np.ndarray:
        return batch_indices(self.seed, t, self.data.n_train, self.batch_size)

    def step(self, t: int) -> Node:
        def build():
            idx = self.batch(t)
            count = int(sum(int(self.data.train_mask[i, 1:int(self.data.train_len[i])].sum()) for i in idx))
            return [self.micro(t, j, int(i), count) for j, i in enumerate(idx)] + [self.adam_step(t)]
        return Node(f"step {t}", build)

    def theta_shapes(self, grads: bool) -> dict[str, tuple[int, int]]:
        r = self.spec.rank
        shapes = {f"{p}.{n}": (o, r) for n, o, _ in self.mods for p in ("B", "m", "v")}
        if grads:
            shapes |= {f"G.{n}": (o, r) for n, o, _ in self.mods}
        return shapes

    def adam_step(self, t: int) -> Node:
        def build():
            a = self.adam
            c = {"c1": f32(1 - ipow(a["b1"], t + 1)), "c2": f32(1 - ipow(a["b2"], t + 1)), "eps": a["eps"],
                 "lr": a["lr"]}
            ops, r = [], self.spec.rank
            for n, o, _ in self.mods:
                s = (o, r)
                ops += [Op("adam_m", f"m.{n}", (f"m.{n}", f"G.{n}"), {"b1": a["b1"], "c1m": a["c1m"]}, (), s, (s, s)),
                        Op("adam_v", f"v.{n}", (f"v.{n}", f"G.{n}"), {"b2": a["b2"], "c2m": a["c2m"]}, (), s, (s, s)),
                        Op("adam_b", f"B.{n}", (f"B.{n}", f"m.{n}", f"v.{n}"), c, (f"G.{n}",), s, (s, s, s))]
            return ops
        return Node(f"step {t} Adam", build)

    def micro(self, t: int, j: int, seq: int, count: int) -> Node:
        def build():
            return _Micro(self, seq, count, first=j == 0).atoms()
        return Node(f"step {t} sequence {j} (#{seq})", build)


class _Micro:
    """Builds one sequence's forward and backward, keeping track of every tensor's shape as it goes."""

    def __init__(self, p: Program, seq: int, count: int, first: bool):
        self.p, self.a, self.spec, self.seq = p, p.arch, p.spec, seq
        self.T = int(p.data.train_len[seq])
        mask = p.data.train_mask[seq, :self.T]
        self.P = int(mask[1:].sum())
        self.ic = f32(1.0 / count)
        self.shapes = p.theta_shapes(grads=not first)
        self.ops: list[Op] = []
        self.s32 = f32(self.spec.scaling)

    def emit(self, kind, out, ins, attrs, shape, drops=()):
        self.ops.append(Op(kind, out, tuple(ins), attrs, tuple(drops), tuple(shape), tuple(self.shapes[i] for i in ins)))
        if out:
            self.shapes[out] = tuple(shape)
        for d in drops:
            del self.shapes[d]

    def take(self, label: str) -> Node:
        ops, self.ops = self.ops, []
        return Node(label, ops)

    def atoms(self) -> list:
        L = self.a.layers
        self.emit("embed", "h.0", (), {"seq": self.seq, "weight": EMBED}, (self.T, self.a.hidden))
        out = [self.ops.pop()]
        for i in range(L):
            self.forward(i, recompute=False)
            out.append(self.take(f"layer {i} forward"))
        self.head()
        out.append(self.take("head and loss"))
        for i in reversed(range(L)):
            self.backward(i)
            out.append(self.take(f"layer {i} backward"))
        return out

    # --- pieces ---

    def _w(self, i, t, part="weight"):
        return f"model.layers.{i}.{MODULES[t]}.{part}"

    def proj(self, i, t, x, out):
        o, _ = self.spec.shape(t)
        bias = self._w(i, t, "bias") if self.a.qkv_bias and t in ("q", "k", "v") else None
        k = self.p.mod.get((i, t))
        if k is None:
            self.emit("linear", out, (x,), {"weight": self._w(i, t), "bias": bias}, (self.T, o))
            return
        name = self.p.mods[k][0]
        self.emit("linear", f"a.{t}0", (x,), {"weight": self._w(i, t), "bias": bias}, (self.T, o))
        self.emit("lora_down", f"a.{t}a", (x,), {"module": k}, (self.T, self.spec.rank))
        self.emit("lora_up", out, (f"a.{t}0", f"a.{t}a", f"B.{name}"), {"scaling": self.s32}, (self.T, o), (f"a.{t}0",))

    def proj_bwd(self, i, t, dy, acc, out, drops=()):
        """dx of projection `t` from dy, plus `acc` if given; G of its adapter gets this sequence's gradient. `drops`
        go with the last op (`acc` is always dropped)."""
        _, inn = self.spec.shape(t)
        ins = (dy,) + ((acc,) if acc else ())
        first = ((acc,) if acc else ())
        k = self.p.mod.get((i, t))
        if k is None:
            self.emit("linear_bwd", out, ins, {"weight": self._w(i, t)}, (self.T, inn), first + tuple(drops))
            return
        name, o, _ = self.p.mods[k]
        r, g = self.spec.rank, f"G.{name}"
        self.emit("linear_bwd", f"b.{t}x0", ins, {"weight": self._w(i, t)}, (self.T, inn), first)
        self.emit("scale", f"b.{t}s", (dy,), {"c": self.s32}, (self.T, o))
        self.emit("lora_grad", g, (f"b.{t}s", f"a.{t}a") + ((g,) if g in self.shapes else ()), {}, (o, r))
        self.emit("lora_bwd_mid", f"b.{t}m", (f"b.{t}s", f"B.{name}"), {}, (self.T, r), (f"b.{t}s",))
        self.emit("lora_bwd_x", out, (f"b.{t}m", f"b.{t}x0"), {"module": k}, (self.T, inn),
                  (f"b.{t}m", f"b.{t}x0") + tuple(drops))

    def rope(self):
        a = self.a
        return {"hd": a.head_dim, "theta": a.rope_theta, "scaling": list(a.rope_scaling)}

    def heads(self):
        a = self.a
        return {"H": a.heads, "KV": a.kv_heads, "hd": a.head_dim}

    def forward(self, i, recompute):
        a, T, d = self.a, self.T, self.a.hidden
        p = f"model.layers.{i}."
        norm = {"group": d, "eps": a.norm_eps}
        self.emit("rmsnorm", "a.x", (f"h.{i}",), {"weight": p + "input_layernorm.weight", **norm}, (T, d))
        for t in ("q", "k", "v"):
            self.proj(i, t, "a.x", f"a.{t}")
        q, k = "a.q", "a.k"
        if a.qk_norm:
            hn = {"group": a.head_dim, "eps": a.norm_eps}
            self.emit("rmsnorm", "a.qn", (q,), {"weight": p + "self_attn.q_norm.weight", **hn}, self.shapes[q])
            self.emit("rmsnorm", "a.kn", (k,), {"weight": p + "self_attn.k_norm.weight", **hn}, self.shapes[k])
            q, k = "a.qn", "a.kn"
        if i not in a.nope:
            self.emit("rope", "a.qr", (q,), self.rope(), self.shapes[q])
            self.emit("rope", "a.kr", (k,), self.rope(), self.shapes[k])
            q, k = "a.qr", "a.kr"
        self.qk = (q, k)
        att = {**self.heads(), "c": f32(1.0 / math.sqrt(a.head_dim))}
        self.emit("attn_scores", "a.s", (q, k), att, (a.heads * T, T))
        self.emit("softmax", "a.sm", ("a.s",), {}, (a.heads * T, T), ("a.s",))
        self.emit("attn_mix", "a.o", ("a.sm", "a.v"), self.heads(), (T, a.heads * a.head_dim))
        self.proj(i, "o", "a.o", "a.y")
        self.emit("add", "a.r", (f"h.{i}", "a.y"), {}, (T, d), ("a.y",))
        self.emit("rmsnorm", "a.x2", ("a.r",), {"weight": p + "post_attention_layernorm.weight", **norm}, (T, d))
        self.proj(i, "gate", "a.x2", "a.g")
        self.proj(i, "up", "a.x2", "a.u")
        self.emit("silu_mul", "a.m", ("a.g", "a.u"), {}, (T, a.ffn))
        self.proj(i, "down", "a.m", "a.d")
        if not recompute:
            live = tuple(n for n in self.shapes if n.startswith("a."))
            self.emit("add", f"h.{i + 1}", ("a.r", "a.d"), {}, (T, d), live)

    def head(self):
        a, T, d = self.a, self.T, self.a.hidden
        L, V = a.layers, a.vocab
        self.emit("rmsnorm", "f.x", (f"h.{L}",), {"weight": NORM, "group": d, "eps": a.norm_eps}, (T, d))
        blocks = [(b, min(V, b + VOCAB_BLOCK)) for b in range(0, V, VOCAB_BLOCK)]
        if self.P:
            w = {"seq": self.seq, "weight": self.p.head_weight}
            for b0, b1 in blocks:
                prev = ("hd.ms",) if "hd.ms" in self.shapes else ()
                self.emit("head_stats", "hd.ms", ("f.x",) + prev, {**w, "b0": b0, "b1": b1}, (self.P, 2))
            self.emit("head_lse", "hd.lse", ("hd.ms",), {}, (self.P, 1), ("hd.ms",))
            for b0, b1 in blocks:
                prev = ("hd.g",) if "hd.g" in self.shapes else ()
                self.emit("head_grad", "hd.g", ("f.x", "hd.lse") + prev, {**w, "b0": b0, "b1": b1, "ic": self.ic},
                          (self.P, d))
            self.emit("scatter", "f.dy", ("hd.g",), {"seq": self.seq}, (T, d), ("hd.g", "hd.lse", "f.x"))
        else:
            self.emit("scatter", "f.dy", (), {"seq": self.seq}, (T, d), ("f.x",))
        self.emit("rmsnorm_bwd", "g.h", (f"h.{L}", "f.dy"), {"weight": NORM, "group": d, "eps": a.norm_eps}, (T, d),
                  ("f.dy", f"h.{L}"))

    def backward(self, i):
        a, T, d = self.a, self.T, self.a.hidden
        p = f"model.layers.{i}."
        self.forward(i, recompute=True)
        q, k = self.qk
        self.proj_bwd(i, "down", "g.h", None, "b.dm")
        self.emit("silu_bwd_g", "b.dg", ("a.g", "a.u", "b.dm"), {}, (T, a.ffn))
        self.emit("silu_bwd_u", "b.du", ("a.g", "b.dm"), {}, (T, a.ffn), ("b.dm",))
        self.proj_bwd(i, "gate", "b.dg", None, "b.x2g", ("b.dg",))
        self.proj_bwd(i, "up", "b.du", "b.x2g", "b.dx2", ("b.du",))
        norm = {"group": d, "eps": a.norm_eps}
        self.emit("rmsnorm_bwd", "b.dr", ("a.r", "b.dx2", "g.h"), {"weight": p + "post_attention_layernorm.weight", **norm},
                  (T, d), ("b.dx2",))
        self.proj_bwd(i, "o", "b.dr", None, "b.do")
        att = {**self.heads(), "c": f32(1.0 / math.sqrt(a.head_dim))}
        self.emit("attn_dp", "b.dp", ("b.do", "a.v"), self.heads(), (a.heads * T, T))
        self.emit("attn_dv", "b.dv", ("a.sm", "b.do"), self.heads(), self.shapes["a.v"], ("b.do",))
        self.emit("softmax_bwd", "b.ds", ("a.sm", "b.dp"), {}, (a.heads * T, T), ("b.dp",))
        self.emit("attn_dq", "b.dq", ("b.ds", k), att, self.shapes[q])
        self.emit("attn_dk", "b.dk", ("b.ds", q), att, self.shapes[k], ("b.ds",))
        dq, dk = "b.dq", "b.dk"
        if i not in a.nope:
            self.emit("rope_bwd", "b.dq1", (dq,), self.rope(), self.shapes[dq], (dq,))
            self.emit("rope_bwd", "b.dk1", (dk,), self.rope(), self.shapes[dk], (dk,))
            dq, dk = "b.dq1", "b.dk1"
        if a.qk_norm:
            hn = {"group": a.head_dim, "eps": a.norm_eps}
            self.emit("rmsnorm_bwd", "b.dq2", ("a.q", dq), {"weight": p + "self_attn.q_norm.weight", **hn},
                      self.shapes[dq], (dq,))
            self.emit("rmsnorm_bwd", "b.dk2", ("a.k", dk), {"weight": p + "self_attn.k_norm.weight", **hn},
                      self.shapes[dk], (dk,))
            dq, dk = "b.dq2", "b.dk2"
        self.proj_bwd(i, "q", dq, None, "b.xq", (dq,))
        self.proj_bwd(i, "k", dk, "b.xq", "b.xk", (dk,))
        self.proj_bwd(i, "v", "b.dv", "b.xk", "b.dx", ("b.dv",))
        self.emit("rmsnorm_bwd", "g.h", (f"h.{i}", "b.dx", "b.dr"), {"weight": p + "input_layernorm.weight", **norm},
                  (T, d), ("b.dx", "b.dr"))
        live = tuple(n for n in self.shapes if n[:2] in ("a.", "b.")) + (f"h.{i}",) + (("g.h",) if i == 0 else ())
        self.emit("free", "", (), {}, (0, 0), live)


@functools.lru_cache(maxsize=64)
def dispute_size(spec, batch_size: int, segments: int, per_segment: int) -> tuple[int, int]:
    """(the most moves a dispute over one unit takes, the most bytes of new tensors one split names), from the spec
    alone: a node sizes a round's dispute phase before it has the job's data. A sequence of seq_len tokens stands in
    for the data, since how many ops a layer, the head and Adam have, and their shapes, depend on lengths alone.

    Moves: for each level from the unit down to an op, a split and a pick for every 64-fold narrowing of its atoms (one
    at least, to descend into it), then the evidence. Bytes: what the states at a split's boundaries hold that the
    state before it didn't (the tensors written in a child that are still there at its end): above a sequence, the
    adapters' B, m and v (between steps and segments) or G (between a step's sequences); below, activations.

    A DeepSeek spec's program (deepseek_program.py) is built from the job's records, so until it counts its own, its
    step is taken as five rounds deep (inputs, a layer, its attention by sequence, ops) and a split as its adapters'."""
    if hasattr(spec, "cut"):  # deepseek_job.DeepSeekSpec
        def levels(n):
            k = 1
            while n > K:
                n, k = -(-n // K), k + 1
            return k
        state = 4 * spec.n_params  # B, m and v
        return 2 * (levels(segments) + levels(per_segment) + 5) + 1, (min(K, max(segments, per_segment)) - 1) * state
    d, T = dict(spec.dims), spec.seq_len
    config = {"model_type": d["family"], "vocab_size": d["vocab"], "hidden_size": d["hidden"],
              "num_hidden_layers": d["layers"], "num_attention_heads": d["heads"], "num_key_value_heads": d["kv_heads"],
              "head_dim": d["head_dim"], "intermediate_size": d["ffn"], "tie_word_embeddings": d["tied"]}
    data = types.SimpleNamespace(parsed_config=lambda: config, n_train=1, train_len=np.array([T]),
                                 train_mask=np.ones((1, T), dtype=np.uint8))
    p = Program(spec, data, 1e-4, 1, 0)

    def rounds(n):
        k = 1
        while n > K:
            n, k = -(-n // K), k + 1
        return k

    def below(atoms):
        return rounds(len(atoms)) + max((below(a.atoms) for a in atoms if isinstance(a, Node)), default=0)

    def split(atoms):
        ends, alive, total = {g1 for _, g1 in groups(0, len(atoms))}, {}, 0
        for i, a in enumerate(atoms):
            for op in [a] if isinstance(a, Op) else a.ops():
                if op.out:
                    alive[op.out] = op.shape
                for n in op.drops:
                    alive.pop(n, None)
            if i + 1 in ends and i + 1 < len(atoms):
                total += sum(4 * r * c for r, c in alive.values())
                alive = {}
        return total

    micros = [_Micro(p, 0, T, first).atoms() for first in (True, False)]
    adam = p.adam_step(0).atoms
    moves = 2 * (rounds(segments) + rounds(per_segment) + rounds(batch_size + 1)
                 + max(*(below(m) for m in micros), below(adam))) + 1
    g = 4 * spec.n_adapter
    nodes = [m for m in micros] + [a.atoms for m in micros for a in m if isinstance(a, Node)] + [adam]
    most = max([(min(K, segments) - 1) * 3 * g, (min(K, per_segment) - 1) * 3 * g, (min(K, batch_size + 1) - 1) * g]
               + [split(n) for n in nodes])
    return moves, most
