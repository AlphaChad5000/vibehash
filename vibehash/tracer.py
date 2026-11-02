"""A fine-tune's training in PyTorch, op by op as program.py lays it out: what finetune.train_segment runs, and what a
worker re-runs in a dispute to show (or to check) any state between two ops.

Training and a dispute's re-run are one computation: the same ops, kernels and inputs on the same device give the same
bits (TF32 off, deterministic kernels: compute.TorchBackend), so the states a worker posts in a dispute are the ones its
training went through, and they chain to the checkpoints it committed to. Every op makes a new tensor (nothing is
changed in place), so a state is just a dict of tensors and a snapshot of it is a shallow copy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .program import Node, Op

_F32 = torch.float32


def _rot(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Executor:
    """Runs a Program's ops on states of `rt` (finetune._Runtime: the base model on one device). `tamper`, for tests,
    is a worker that cheats in one op: tamper(where, op) gives a factor its output is multiplied by, or None."""

    def __init__(self, rt, spec, program, tamper=None):
        self.model, self.device, self.spec, self.prog, self.tamper = rt.model, rt.device, spec, program, tamper
        self.a = {k: a for k, (_, a, _, _) in enumerate(rt.lora[spec])}
        self.where: tuple = ()
        self._seq: dict[int, tuple] = {}
        self._head: tuple = (None, None)  # the head as float32 while a sequence's head and loss run

    def w(self, name: str) -> torch.Tensor:
        return self.model.w[name]

    def seq(self, i: int):
        """One training sequence: (ids, its target positions, their targets) on the device."""
        if i not in self._seq:
            d = self.prog.data
            n = int(d.train_len[i])
            ids = torch.from_numpy(d.train_ids[i, :n].astype("int64")).to(self.device)
            pos = torch.from_numpy((d.train_mask[i, 1:n] != 0).nonzero()[0].astype("int64")).to(self.device)
            self._seq = {i: (ids, pos, ids[pos + 1])}
        return self._seq[i]

    def head32(self, name: str) -> torch.Tensor:
        """The head's weights cast to float32 once for all of a sequence's head ops (each slices its block): the cast
        is exact, so it's the same numbers as casting each block, and it's what llm.py's _HeadLoss holds too."""
        if self._head[0] != name:
            self._head = (name, self.w(name).to(_F32))
        return self._head[1]

    def tables(self, t: int):
        return self.model._tables(t)

    # --- running ---

    def run(self, op: Op, st: dict) -> None:
        out = RUN[op.kind](self, op, st)
        if out is not None:
            if self.tamper is not None:
                f = self.tamper(self.where, op)
                if f is not None:
                    out = out * f
            st[op.out] = out.contiguous()
        for d in op.drops:
            del st[d]

    def run_atoms(self, node: Node, lo: int, hi: int, st: dict, where: tuple = ()) -> None:
        for i in range(lo, hi):
            a = node.atoms[i]
            if isinstance(a, Op):
                self.where = where
                self.run(a, st)
            else:
                self.run_atoms(a, 0, len(a.atoms), st, where + (a.label,))
                if a.label == "head and loss":
                    self._head = (None, None)


# --- the ops, as referee.py checks them -------------------------------------------------------------------------------

def _x(st, op, i=0):
    return st[op.ins[i]]


def _acc(st, op, i, y):
    return st[op.ins[i]] + y if len(op.ins) > i else y


def r_embed(ex, op, st):
    ids, _, _ = ex.seq(op.attrs["seq"])
    return ex.w(op.attrs["weight"])[ids].to(_F32)


def _norm_rsqrt(xs, eps):
    return torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)


def r_rmsnorm(ex, op, st):
    x = _x(st, op)
    xs = x.view(x.shape[0], -1, op.attrs["group"])
    return (ex.w(op.attrs["weight"]) * (xs * _norm_rsqrt(xs, op.attrs["eps"]))).view(x.shape)


def r_linear(ex, op, st):
    y = F.linear(_x(st, op), ex.w(op.attrs["weight"]).to(_F32))
    bias = op.attrs.get("bias")
    return y + ex.w(bias) if bias else y


def r_lora_down(ex, op, st):
    return _x(st, op) @ ex.a[op.attrs["module"]].T


def r_lora_up(ex, op, st):
    base, xa, b = (st[n] for n in op.ins)
    return base + (xa @ b.T) * op.attrs["scaling"]


def r_rope(ex, op, st):
    x = _x(st, op)
    cos, sin = ex.tables(x.shape[0])
    xs = x.view(x.shape[0], -1, op.attrs["hd"])
    return (xs * cos[:, None] + _rot(xs) * sin[:, None]).view(x.shape)


def _dims(op):
    a = op.attrs
    return a["H"], a["KV"], a["hd"], a["H"] // a["KV"]


def _q4(x, KV, G, hd):  # [T, H·hd] -> [KV, G, T, hd]
    return x.view(x.shape[0], KV, G, hd).permute(1, 2, 0, 3)


def _kv4(x, KV, hd):  # [T, KV·hd] -> [KV, 1, T, hd]
    return x.view(x.shape[0], KV, hd).permute(1, 0, 2).unsqueeze(1)


def r_attn_scores(ex, op, st):
    q, k = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = q.shape[0]
    mask = torch.full((T, T), float("-inf"), device=q.device, dtype=_F32).triu(1)
    return ((_q4(q, KV, G, hd) @ _kv4(k, KV, hd).transpose(-1, -2)) * op.attrs["c"] + mask).reshape(H * T, T)


def r_softmax(ex, op, st):
    return torch.softmax(_x(st, op), dim=-1)


def r_attn_mix(ex, op, st):
    p, v = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = v.shape[0]
    return (p.view(KV, G, T, T) @ _kv4(v, KV, hd)).permute(2, 0, 1, 3).reshape(T, H * hd)


def r_add(ex, op, st):
    return st[op.ins[0]] + st[op.ins[1]]


def r_silu_mul(ex, op, st):
    return F.silu(st[op.ins[0]]) * st[op.ins[1]]


def _head(ex, op, st):
    _, pos, tgt = ex.seq(op.attrs["seq"])
    w = ex.head32(op.attrs["weight"])[op.attrs["b0"]:op.attrs["b1"]]
    return F.linear(st[op.ins[0]][pos], w), w, tgt


def r_head_stats(ex, op, st):
    lo, _, _ = _head(ex, op, st)
    mb = lo.max(-1).values
    if len(op.ins) > 1:
        prev = st[op.ins[1]]
        m = torch.maximum(prev[:, 0], mb)
        s = prev[:, 1] * torch.exp(prev[:, 0] - m) + torch.exp(lo - m[:, None]).sum(-1)
    else:
        m, s = mb, torch.exp(lo - mb[:, None]).sum(-1)
    return torch.stack((m, s), dim=-1)


def r_head_lse(ex, op, st):
    ms = _x(st, op)
    return ms[:, :1] + torch.log(ms[:, 1:2])


def r_head_grad(ex, op, st):
    lo, w, tgt = _head(ex, op, st)
    q = torch.exp(lo - st[op.ins[1]])
    b0, b1 = op.attrs["b0"], op.attrs["b1"]
    rows = ((tgt >= b0) & (tgt < b1)).nonzero()[:, 0]
    if len(rows):
        q[rows, tgt[rows] - b0] -= 1
    return _acc(st, op, 2, (q * op.attrs["ic"]) @ w)


def r_scatter(ex, op, st):
    ids, pos, _ = ex.seq(op.attrs["seq"])
    out = torch.zeros(len(ids), ex.prog.arch.hidden, device=ids.device, dtype=_F32)
    if op.ins:
        out[pos] = st[op.ins[0]]
    return out


def r_rmsnorm_bwd(ex, op, st):
    x, dy = st[op.ins[0]], st[op.ins[1]]
    g = op.attrs["group"]
    xs, dys = x.view(x.shape[0], -1, g), dy.view(x.shape[0], -1, g)
    r = _norm_rsqrt(xs, op.attrs["eps"])
    dyw = dys * ex.w(op.attrs["weight"])
    dm = (dyw * xs).mean(-1, keepdim=True)
    return _acc(st, op, 2, (r * (dyw - xs * ((r * r) * dm))).view(x.shape))


def r_linear_bwd(ex, op, st):
    return _acc(st, op, 1, _x(st, op) @ ex.w(op.attrs["weight"]).to(_F32))


def r_scale(ex, op, st):
    return _x(st, op) * op.attrs["c"]


def r_lora_grad(ex, op, st):
    return _acc(st, op, 2, st[op.ins[0]].T @ st[op.ins[1]])


def r_lora_bwd_mid(ex, op, st):
    return st[op.ins[0]] @ st[op.ins[1]]


def r_lora_bwd_x(ex, op, st):
    return st[op.ins[1]] + st[op.ins[0]] @ ex.a[op.attrs["module"]]


def r_silu_bwd_g(ex, op, st):
    g, u, dm = (st[n] for n in op.ins)
    sig = torch.sigmoid(g)
    return (dm * u) * (sig * (1 + g * (1 - sig)))


def r_silu_bwd_u(ex, op, st):
    return st[op.ins[1]] * F.silu(st[op.ins[0]])


def r_attn_dp(ex, op, st):
    do, v = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = v.shape[0]
    return (_q4(do, KV, G, hd) @ _kv4(v, KV, hd).transpose(-1, -2)).reshape(H * T, T)


def r_attn_dv(ex, op, st):
    p, do = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = do.shape[0]
    return (p.view(KV, G, T, T).transpose(-1, -2) @ _q4(do, KV, G, hd)).sum(1).permute(1, 0, 2).reshape(T, KV * hd)


def r_softmax_bwd(ex, op, st):
    p, dp = (st[n] for n in op.ins)
    return p * (dp - (dp * p).sum(-1, keepdim=True))


def r_attn_dq(ex, op, st):
    ds, k = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = k.shape[0]
    return ((ds * op.attrs["c"]).view(KV, G, T, T) @ _kv4(k, KV, hd)).permute(2, 0, 1, 3).reshape(T, H * hd)


def r_attn_dk(ex, op, st):
    ds, q = (st[n] for n in op.ins)
    H, KV, hd, G = _dims(op)
    T = q.shape[0]
    dss = (ds * op.attrs["c"]).view(KV, G, T, T)
    return (dss.transpose(-1, -2) @ _q4(q, KV, G, hd)).sum(1).permute(1, 0, 2).reshape(T, KV * hd)


def r_rope_bwd(ex, op, st):
    dy = _x(st, op)
    cos, sin = ex.tables(dy.shape[0])
    hd = op.attrs["hd"]
    ys = dy.view(dy.shape[0], -1, hd)
    a, b = ys * cos[:, None], ys * sin[:, None]
    return (a + torch.cat((b[..., hd // 2:], -b[..., :hd // 2]), dim=-1)).view(dy.shape)


def r_adam_m(ex, op, st):
    m, g = (st[n] for n in op.ins)
    return m * op.attrs["b1"] + g * op.attrs["c1m"]


def r_adam_v(ex, op, st):
    v, g = (st[n] for n in op.ins)
    return v * op.attrs["b2"] + (g * g) * op.attrs["c2m"]


def r_adam_b(ex, op, st):
    b, m, v = (st[n] for n in op.ins)
    a = op.attrs
    return b - ((m / a["c1"]) / ((v / a["c2"]).sqrt() + a["eps"])) * a["lr"]


def r_free(ex, op, st):
    return None


RUN = {name[2:]: fn for name, fn in globals().items() if name.startswith("r_")}


# --- disputes: a worker's side (dispute.py has the rules) -------------------------------------------------------------

HARD = 1e-2  # a child whose result is this far (relative) from a recompute is taken at once; else the furthest
CANDIDATES = 32  # tiles of a picked op the challenger checks with the referee before naming one (or withdrawing)


def node_at(prog, root, path) -> tuple[Node, tuple]:
    """The node a game is at, and the labels training runs its ops under (tracer.Executor's `where`)."""
    n = prog.segment(int(root[1])) if root[0] == "segment" else prog.unit()
    labels = []
    for i in path:
        n = n.atoms[i]
        labels.append(n.label)
    skip = 0 if root[0] == "segment" else 1  # a unit's first level is segments, which training doesn't label
    return n, tuple(labels[skip:])


class TorchWeights:
    """The base model's weights as a worker has them loaded, for running the referee itself (dispute.Outside)."""

    _NAMES = {torch.bfloat16: ("BF16", torch.int16), torch.float16: ("F16", torch.int16), torch.float32: ("F32", torch.int32)}

    def __init__(self, model):
        self.model = model

    def shape(self, name):
        if name not in self.model.w:
            from .dispute import WeightsUnavailable
            raise WeightsUnavailable(f"the base model has no {name}")
        t = self.model.w[name]
        return self._NAMES[t.dtype][0], tuple(t.shape)

    def read(self, name, r0, r1) -> bytes:
        t = self.model.w[name]
        t = t[r0:r1] if t.ndim > 1 else t
        return t.contiguous().view(self._NAMES[t.dtype][1]).cpu().numpy().tobytes()


class _AsNumpy(dict):
    """A state's tensors as float32 NumPy arrays, converted when first read."""

    def __init__(self, st: dict):
        super().__init__()
        self.st = st

    def __missing__(self, name):
        a = self.st[name].detach().cpu().numpy()
        self[name] = a
        return a

    def shapes(self) -> dict:
        return {n: tuple(t.shape) for n, t in self.st.items()}


class Party:
    """One worker's side of its disputes: states it has made or fetched (by root), and how to run them.
    `remote` has put(bytes) -> blob id and get(blob id) -> bytes (the node's blob store)."""

    def __init__(self, rt, spec, data, remote, tamper=None):
        from . import merkle

        self.rt, self.spec, self.data, self.remote, self.tamper, self.merkle = rt, spec, data, remote, tamper, merkle
        self.states: dict[str, tuple[dict, bytes]] = {}
        self.tensors: dict[str, torch.Tensor] = {}  # by blob id
        self.hashes: dict[int, tuple] = {}  # id(tensor) -> (tensor, root, blob): every op makes a new tensor

    def executor(self, prog, honest: bool) -> Executor:
        return Executor(self.rt, self.spec, prog, None if honest else self.tamper)

    def source(self):
        """The base model's weights as this worker holds them, for running the referee itself (dispute.Outside)."""
        return TorchWeights(self.rt.model)

    def load(self, root: str) -> tuple[dict, bytes]:
        if root not in self.states:
            m = self.remote.get(root)
            if self.merkle.state_root(m) != root:
                raise ValueError("the node sent another state than the one asked for")
            st = {}
            for name, (troot, blob, rows, cols) in self.merkle.parse_manifest(m).items():
                if blob not in self.tensors:
                    raw = self.remote.get(blob)
                    a = torch.frombuffer(bytearray(raw), dtype=_F32).reshape(rows, cols)
                    self.tensors[blob] = a.to(self.rt.device)
                    self.hashes[id(self.tensors[blob])] = (self.tensors[blob], troot, blob)
                st[name] = self.tensors[blob]
            self.states[root] = (st, m)
        st, m = self.states[root]
        return dict(st), m

    def publish(self, st: dict) -> str:
        entries, new = {}, {}
        for name, t in st.items():
            hit = self.hashes.get(id(t))
            if hit is None or hit[0] is not t:
                a = t.detach().cpu().numpy()
                hit = (t, self.merkle.root_of(a), self.merkle.blob_of(a))
                self.hashes[id(t)] = hit
                new[hit[2]] = a
            entries[name] = (hit[1], hit[2], t.shape[0], t.shape[1])
            self.tensors.setdefault(hit[2], t)
        m = self.merkle.manifest_bytes(entries)
        for blob, a in new.items():
            self.remote.put(a.astype("<f4").tobytes())
        root = self.remote.put(m)
        self.states[root] = (dict(st), m)
        return root

    def forget(self) -> None:
        self.states.clear()
        self.tensors.clear()
        self.hashes.clear()

    # --- the defender ---

    def split(self, prog, view: dict) -> list[str]:
        """The state roots at the boundaries of the current range's children, as its training went."""
        from .program import groups

        node, where = node_at(prog, view["root"], view["path"])
        st, _ = self.load(view["a"])
        ex, roots = self.executor(prog, honest=False), [view["a"]]
        for g0, g1 in groups(view["lo"], view["hi"])[:-1]:
            ex.run_atoms(node, g0, g1, st, where)
            roots.append(self.publish(st))
        return roots + [view["b"]]

    def evidence(self, prog, view: dict) -> bytes:
        from .dispute import Outside, make_evidence

        node, _ = node_at(prog, view["root"], view["path"])
        op = node.atoms[view["lo"]]
        (before, mb), (after, ma) = self.load(view["a"]), self.load(view["b"])
        out = Outside(self.spec, self.data, self.source(), keep=False)
        b, a = _AsNumpy(before), _AsNumpy(after)
        evidence, _, _ = make_evidence(op, view["tile"], b, a, (mb, ma), out, shapes=b.shapes())
        return evidence

    # --- the challenger ---

    def pick(self, prog, view: dict) -> dict:
        """The first child that doesn't come out of a recompute from the defender's state before it (else the one
        furthest off), and for an op, its worst tile. A replica that finds nothing wrong with the other's unit
        concedes, which costs it nothing before its first pick."""
        from .program import groups

        node, where = node_at(prog, view["root"], view["path"])
        ex, roots = self.executor(prog, honest=True), view["roots"]
        best = (-1.0, 0)
        for c, (g0, g1) in enumerate(groups(view["lo"], view["hi"])):
            mine, _ = self.load(roots[c])
            theirs, _ = self.load(roots[c + 1])
            ex.run_atoms(node, g0, g1, mine, where)
            dev = deviation(mine, theirs)
            best = max(best, (dev, -c))
            if dev >= HARD:
                break
        if view.get("why") == "replicas" and not view.get("played") and best[0] < HARD:
            return {"concede": True}
        child = -best[1]
        g0, g1 = groups(view["lo"], view["hi"])[child]
        atom = node.atoms[g0] if g1 - g0 == 1 else None
        move = {"pick": child}
        if isinstance(atom, Op) and atom.out:
            tile = self.worst_tile(prog, atom, roots[child], roots[child + 1])
            if tile is None:  # nothing the referee would refuse: play on and lose, or withdraw
                return {"concede": True}
            move["tile"] = tile
        return move

    def worst_tile(self, prog, op: Op, a: str, b: str) -> int | None:
        """The tile of the defender's output of `op` the referee would refuse, among the CANDIDATES furthest from a
        recompute, or None: then the challenger can only withdraw (dispute.play's concede), not win."""
        from . import merkle, referee
        from .dispute import Inputs, Outside

        before, _ = self.load(a)
        after, _ = self.load(b)
        mine = dict(before)
        self.executor(prog, honest=True).run(op, mine)
        if op.out not in after:
            return 0
        n = merkle.n_leaves(*op.shape)
        x, y = mine[op.out].reshape(-1), after[op.out].reshape(-1)
        pad = n * merkle.LEAF - x.numel()
        diff = torch.nn.functional.pad(_diff(x, y), (0, pad)).view(n, -1).amax(1)
        size = torch.nn.functional.pad(_size(y), (0, pad)).view(n, -1).amax(1)
        off = torch.nan_to_num(diff / (size + 1e-30), nan=float("inf"))
        order = torch.argsort(off, descending=True)[:CANDIDATES].tolist()
        out = Outside(self.spec, self.data, self.source(), keep=False)
        full = _AsNumpy(before)
        theirs = after[op.out].detach().cpu().numpy().reshape(-1)
        for tile in order:
            said = theirs[tile * merkle.LEAF:(tile + 1) * merkle.LEAF]
            try:
                ok, _ = referee.judge(op, Inputs(full.shapes(), out, full=full), tile, said)
            except referee.Unresolved:  # the referee can't say: not a tile to win on
                continue
            if not ok:
                return tile
        return None


def _diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a − b|, 0 where they're equal (the attention mask's −∞ included), ∞ where either is NaN."""
    d = torch.where(a == b, torch.zeros_like(a), (a - b).abs())
    return torch.nan_to_num(d, nan=float("inf"))


def _size(b: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(b), b.abs(), torch.zeros_like(b))


def deviation(mine: dict, theirs: dict) -> float:
    """How far a state is from a recompute of it: the largest relative difference of any tensor, inf if they don't
    hold the same tensors."""
    if set(mine) != set(theirs):
        return float("inf")
    worst = 0.0
    for name, a in mine.items():
        b = theirs[name]
        if a is b:
            continue
        if a.shape != b.shape:
            return float("inf")
        d = float((_diff(a, b).max() / (_size(b).max() + 1e-30)).item()) if a.numel() else 0.0
        if d != d:
            return float("inf")
        worst = max(worst, d)
    return worst
