"""A DeepSeek-V4.1 top-layers unit in PyTorch, op by op as deepseek_program.py lays it out: what deepseek_unit trains
by, and what a worker re-runs in a dispute to show (or to check) any state between two ops (tracer.py's Executor and
Party, for DeepSeek's ops).

Each op computes exactly the expression its referee op (referee_deepseek.py) names and bounds, in that order, mostly
by calling deepseek.py's own functions on the op's inputs: so a unit's states are what the model computes, and the
node's referee checks each one. Every op makes a new tensor (nothing is changed in place), so a state is a dict of
tensors and a snapshot of it a shallow copy. States hold float32; under "ref" (prep units' bf16) an op computes in the
activation dtype and its result is widened back, exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import deepseek as ds
from . import tracer as tr
from .program import Op

_F32 = torch.float32
NUM = {"f32": ds.Numerics(torch.float32, quantize=False), "ref": ds.Numerics()}


class Executor(tr.Executor):
    """Runs a DeepSeekProgram's ops on states of `rt` (deepseek_unit's runtime: the trained layers and the head on
    one device, the routed experts wherever their tensors are, brought over an expert at a time). `tamper`, for
    tests: tamper(where, op) gives a factor an op's output is multiplied by, or None."""

    def __init__(self, rt, spec, program, tamper=None):
        self.model, self.device, self.spec, self.prog, self.tamper = rt.model, rt.device, spec, program, tamper
        self.state = rt.weights
        self.a = {k: torch.from_numpy(a).to(self.device) for k, a in enumerate(spec.lora_a())}
        self.where: tuple = ()
        self._head: tuple = (None, None)
        self._records: dict[int, dict] = {}
        # a prep unit's engram rows: deepseek_prep.Slice by layer (the rows units' hand-offs) and the token map;
        # without them, the tables and "engram.token_map" in the weights (tests' random weights)
        self.slices: dict = {}
        self.token_map: torch.Tensor | None = None

    def w(self, name: str) -> torch.Tensor:
        return self.state[name]

    def record(self, seq: int) -> dict:
        if seq not in self._records:
            if len(self._records) > 64:
                self._records.clear()
            self._records[seq] = self.prog.data.records.record(seq)
        return self._records[seq]

    def run(self, op: Op, st: dict) -> None:
        out = (RUN.get(op.kind) or tr.RUN[op.kind])(self, op, st)
        if out is not None:
            if self.tamper is not None:
                f = self.tamper(self.where, op)
                if f is not None:
                    out = out * f
            st[op.out] = out.contiguous()
        for d in op.drops:
            del st[d]


def _num(op) -> ds.Numerics:
    return NUM[op.attrs.get("num", "ref")]


def _in(st, op, i=0):
    """Input i in the op's activation dtype (float32 states hold bf16 numbers exactly)."""
    return st[op.ins[i]].to(_num(op).dtype)


def _out(t):
    return t.to(_F32)


def _acc(st, op, i, y):
    return st[op.ins[i]] + y if len(op.ins) > i else y


def _vec(ex, name, hi):
    return ex.w(name).to(ex.device, hi)


# --- inputs and moving rows ---------------------------------------------------------------------------------------------

def r_ds_record(ex, op, st):
    part, out = op.attrs["part"], []
    for s in op.attrs["seqs"]:
        r = ex.record(int(s))
        if part == "x":
            t = r["x"].flatten(1)
        elif part == "mask":
            t = r["mask"][:, None]
        elif part == "targets":
            tok = r["tokens"].long()
            t = torch.cat((tok[1:], tok[:1]))[r["mask"].bool()][:, None]
        else:
            t = r[part]
        out.append(t.to(ex.device, _F32))
    return torch.cat(out, 0)


def r_ds_slice(ex, op, st):
    return st[op.ins[0]][op.attrs["r0"]:op.attrs["r1"]]


def r_ds_cat(ex, op, st):
    return torch.cat([st[n] for n in op.ins], 0)


def r_ds_reshape(ex, op, st):
    return st[op.ins[0]].reshape(op.shape)


def r_ds_rows(ex, op, st):
    return st[op.ins[0]][st[op.ins[1]][:, 0] == 1]


def r_ds_unrows(ex, op, st):
    src, mask = st[op.ins[0]], st[op.ins[1]][:, 0] == 1
    out = torch.zeros(op.shape, device=src.device, dtype=_F32)
    out[mask] = src
    return out


# --- projections ----------------------------------------------------------------------------------------------------------

def _dequant(ex, base: str, dtype) -> torch.Tensor:
    w, s = ex.w(base + ".weight"), ex.w(base + ".scale")
    return ds.dequant(w.to(ex.device, non_blocking=True), s.to(ex.device, non_blocking=True), dtype)


def r_ds_round_fp8(ex, op, st):
    return _out(ds.round_fp8(_in(st, op), op.attrs.get("block", ds.BLOCK)))


def r_ds_round_fp4(ex, op, st):
    a = op.attrs
    return _out(ds.round_fp4(_in(st, op), a.get("block", ds.BLOCK), a.get("e4m3", False)))


def r_ds_qlinear(ex, op, st):
    x = _in(st, op)
    g = op.attrs.get("groups", 1)
    w = _dequant(ex, op.attrs["weight"], x.dtype)
    if g == 1:
        return _out(ds._matmul(x, w, 1))
    return _out(ds._matmul(x.view(x.shape[0], g, -1), w, g).flatten(1))


def r_ds_qlinear_bwd(ex, op, st):
    dy = st[op.ins[0]]
    g = op.attrs.get("groups", 1)
    w = _dequant(ex, op.attrs["weight"], _F32)
    if g == 1:  # as _FrozenMatmul's backward: dy · W, a linear with Wᵀ laid out row by row
        dx = F.linear(dy, w.t().contiguous())
    else:
        dx = torch.einsum("...gr,grd->...gd", dy.view(dy.shape[0], g, -1), w.view(g, -1, w.size(1))).flatten(1)
    return _acc(st, op, 1, dx)


def r_ds_dense(ex, op, st):
    num = _num(op)
    dt = num.hi if op.attrs.get("hi") else num.dtype
    return _out(F.linear(st[op.ins[0]].to(dt), ex.w(op.attrs["weight"]).to(ex.device, dt)))


def r_ds_dense_bwd(ex, op, st):
    return _acc(st, op, 1, st[op.ins[0]] @ ex.w(op.attrs["weight"]).to(ex.device, _F32))


def r_ds_scale(ex, op, st):
    return _out(_in(st, op) * op.attrs["c"])


def r_ds_rmsnorm(ex, op, st):
    x, num = _in(st, op), _num(op)
    g = op.attrs.get("group", op.shape[1])
    h = x.view(x.shape[0], -1, g).to(num.hi)
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + op.attrs["eps"])
    return _out((_vec(ex, op.attrs["weight"], num.hi) * h).to(x.dtype).view(x.shape))


def r_ds_rope(ex, op, st):
    a, x, num = op.attrs, _in(st, op), _num(op)
    T, stride = x.shape[0], a.get("stride", 1)
    cos, sin = ds._rope_table(T * stride, *a["rope"])
    cos, sin = cos[::stride][:T], sin[::stride][:T]
    y = ds._rope(x.view(1, T, -1, a["hd"]), cos, sin, num, inverse=bool(a.get("inverse")))
    return _out(y.reshape(x.shape))


# --- hyper-connections ------------------------------------------------------------------------------------------------------

def r_ds_hc_mixes(ex, op, st):
    num = _num(op)
    h = st[op.ins[0]].to(num.hi)
    return _out(F.linear(h, _vec(ex, op.attrs["fn"], num.hi))
                * torch.rsqrt(h.square().mean(-1, keepdim=True) + op.attrs["eps"]))


def _split_args(ex, op):
    a, hi = op.attrs, _num(op).hi
    return _vec(ex, a["scale"], hi), _vec(ex, a["base"], hi), a["hc"], a["iters"], a["eps"]


def r_ds_hc_split(ex, op, st):
    m = st[op.ins[0]].to(_num(op).hi)
    pre, post, comb = ds._sinkhorn_split(m, *_split_args(ex, op))
    return _out(torch.cat((pre, post, comb.flatten(-2)), -1))


def r_ds_hc_split_trace(ex, op, st):
    """_sinkhorn_split's comb, every iterate: the same expressions, kept."""
    m = st[op.ins[0]].to(_num(op).hi)
    scale, base, hc, iters, eps = _split_args(ex, op)
    soft = (m[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc)).softmax(-1)
    comb = soft + eps
    out = [soft, comb]
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    out.append(comb)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        out.append(comb)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        out.append(comb)
    return _out(torch.cat([t.flatten(-2) for t in out], -1))


def r_ds_hc_split_bwd(ex, op, st):
    d, m, trace = (st[n] for n in op.ins)
    scale, base, hc, iters, eps = _split_args(ex, op)

    def through_sigmoid(lo, hi, k, two):
        s = torch.sigmoid(m[:, lo:hi] * scale[k] + base[lo:hi])
        g = d[:, lo:hi] * (s * (1 - s))
        return (g * 2.0 if two else g) * scale[k]
    its = trace.view(trace.shape[0], -1, hc, hc)  # [softmax, c0, c1, ...]
    soft, cs = its[:, 0], its[:, 1:]
    kinds = ["col"] + ["row", "col"] * (iters - 1)
    dy = d[:, 2 * hc:].view(-1, hc, hc)
    for t in reversed(range(len(kinds))):  # step t takes cs[t] to cs[t + 1]
        line = -1 if kinds[t] == "row" else -2
        a, y = cs[:, t], cs[:, t + 1]
        dy = (dy - (y * dy).sum(line, keepdim=True)) / (a.sum(line, keepdim=True) + eps)
    dz = soft * (dy - (soft * dy).sum(-1, keepdim=True))
    return torch.cat((through_sigmoid(0, hc, 0, False), through_sigmoid(hc, 2 * hc, 1, True),
                      (dz * scale[2]).flatten(1)), -1)


def _stream(t, hc):
    return t.view(t.shape[0], hc, -1)


def r_ds_hc_pre(ex, op, st):
    num = _num(op)
    dim, col = op.shape[1], op.attrs.get("col", 0)
    x = _in(st, op)
    hc = x.shape[1] // dim
    pre = st[op.ins[1]][:, col:col + hc].to(num.hi)
    return _out(ds.hc_pre(_stream(x, hc)[:, None], pre[:, None], num.hi)[:, 0])


def r_ds_hc_post(ex, op, st):
    a, num = op.attrs, _num(op)
    y, res = _in(st, op, 0), _in(st, op, 1)
    hc = op.shape[1] // y.shape[1]
    sp = st[op.ins[2]].to(num.hi)
    post, comb = sp[:, a["post"]:a["post"] + hc], sp[:, a["comb"]:a["comb"] + hc * hc].view(-1, hc, hc)
    return _out(ds.hc_post(y[None], _stream(res, hc)[None], post[None], comb[None], num.hi)[0].flatten(1))


def r_ds_hc_pre_dx(ex, op, st):
    dout = st[op.ins[0]]
    hc = op.shape[1] // dout.shape[1]
    col = op.attrs.get("col", 0)
    pre = st[op.ins[1]][:, col:col + hc]
    return _acc(st, op, 2, (pre[:, :, None] * dout[:, None, :]).flatten(1))


def r_ds_hc_pre_dpre(ex, op, st):
    dout, x = st[op.ins[0]], st[op.ins[1]]
    return torch.einsum("nd,nkd->nk", dout, _stream(x, x.shape[1] // dout.shape[1]))


def r_ds_hc_post_dy(ex, op, st):
    dout, sp = st[op.ins[0]], st[op.ins[1]]
    hc = dout.shape[1] // op.shape[1]
    post = sp[:, op.attrs["post"]:op.attrs["post"] + hc]
    return (post[:, :, None] * _stream(dout, hc)).sum(1)


def r_ds_hc_post_dres(ex, op, st):
    dout, sp = st[op.ins[0]], st[op.ins[1]]
    hc = op.attrs["hc"]
    comb = sp[:, op.attrs["comb"]:op.attrs["comb"] + hc * hc].view(-1, hc, hc)
    return _acc(st, op, 2, torch.einsum("njk,nkd->njd", comb, _stream(dout, hc)).flatten(1))


def r_ds_hc_post_dsplit(ex, op, st):
    dout, y, res = st[op.ins[0]], st[op.ins[1]], st[op.ins[2]]
    hc = op.attrs["hc"]
    d, r = _stream(dout, hc), _stream(res, hc)
    dpre = st[op.ins[3]] if len(op.ins) > 3 else torch.zeros(dout.shape[0], hc, device=dout.device, dtype=_F32)
    dpost = torch.einsum("nkd,nd->nk", d, y)
    dcomb = torch.einsum("njd,nkd->njk", r, d).flatten(1)
    return torch.cat((dpre, dpost, dcomb), -1)


def r_ds_hc_mixes_bwd(ex, op, st):
    dmix, h = st[op.ins[0]], st[op.ins[1]]
    fn = _vec(ex, op.attrs["fn"], _F32)
    n = h.shape[1]
    lin = F.linear(h, fn)
    rs = torch.rsqrt(h.square().mean(-1, keepdim=True) + op.attrs["eps"])
    d1 = (dmix * rs) @ fn
    q = (dmix * lin).sum(-1, keepdim=True)
    coef = (q * ((rs * rs) * rs)) * float(torch.tensor(-1.0 / n, dtype=_F32))
    return _acc(st, op, 2, d1 + h * coef)


# --- the indexer ------------------------------------------------------------------------------------------------------------

def r_ds_index_scores(ex, op, st):
    a = op.attrs
    q, k, w = (_in(st, op, i) for i in range(3))
    T, H = q.shape[0], a["heads"]
    sc = torch.einsum("bshd,btd->bsht", q.view(1, T, H, -1), k[None])
    out = (sc.relu() * w[None].unsqueeze(-1)).sum(dim=2)[0]
    visible = (torch.arange(1, T + 1, device=q.device) // a["ratio"]).unsqueeze(-1)
    out = out.masked_fill(torch.arange(k.shape[0], device=q.device) >= visible, float("-inf"))
    if len(op.ins) > 3:  # the candidate source's blocks (Indexer.forward, for a layer above it)
        out = out.masked_fill(st[op.ins[3]] == 0, float("-inf"))
    return _out(out)


def r_ds_picks(ex, op, st):
    score = st[op.ins[0]]
    T, n = score.shape
    idx = score.topk(min(op.shape[1], n), dim=-1, sorted=False).indices.sort(dim=-1).values
    visible = (torch.arange(1, T + 1, device=score.device) // op.attrs["ratio"]).unsqueeze(-1)
    return torch.where(idx < visible, idx, -1).to(_F32)


# --- attention ----------------------------------------------------------------------------------------------------------------

def r_ds_attn(ex, op, st):
    a, num = op.attrs, _num(op)
    q, kv = _in(st, op, 0), _in(st, op, 1)
    T, H = q.shape[0], a["heads"]
    comp = picks = None
    if len(op.ins) > 2:
        comp = _in(st, op, 2)[None]
        picks = st[op.ins[3]].long()[None]
    o = ds.sparse_attention(q.view(1, T, H, -1), kv[None], comp, picks, _vec(ex, a["sink"], _F32), a["scale"],
                            a["window"], num)
    return _out(o.reshape(T, -1))


def _keys(op, st, kv_at=1):
    """Each query's key slots (its window, oldest first, then its picks): keys [T, S, hd] and which are real."""
    a = op.attrs
    kv = st[op.ins[kv_at]]
    T, dev = kv.shape[0], kv.device
    t = torch.arange(T, device=dev)[:, None]
    widx = (t - a["window"] + 1).clamp_min(0) + torch.arange(a["n_window"], device=dev)[None]
    valid = widx <= t
    keys = kv[widx.clamp(max=T - 1)]
    if len(op.ins) > kv_at + 1:
        comp, picks = st[op.ins[kv_at + 1]], st[op.ins[kv_at + 2]].long()
        keys = torch.cat((keys, comp[picks.clamp_min(0)]), 1)
        valid = torch.cat((valid, picks >= 0), 1)
    return keys, valid, widx


def r_ds_attn_probs(ex, op, st):
    a = op.attrs
    q = st[op.ins[0]]
    keys, valid, _ = _keys(op, st)
    T, H = q.shape[0], a["heads"]
    s = torch.einsum("thd,tsd->ths", q.view(T, H, -1), keys) * a["scale"]
    s = s.masked_fill(~valid[:, None], float("-inf"))
    m = s.amax(-1, keepdim=True)
    e = torch.exp(s - m)
    den = e.sum(-1, keepdim=True) + torch.exp(_vec(ex, a["sink"], _F32)[None, :, None] - m)
    return (e / den).masked_fill(~valid[:, None], 0.0).flatten(1)


def r_ds_attn_dp(ex, op, st):
    a = op.attrs
    do = st[op.ins[0]]
    keys, valid, _ = _keys(op, st)
    T, H = do.shape[0], a["heads"]
    return torch.einsum("thd,tsd->ths", do.view(T, H, -1), keys).masked_fill(~valid[:, None], 0.0).flatten(1)


def r_ds_attn_ds(ex, op, st):
    p, dp = st[op.ins[0]], st[op.ins[1]]
    S = op.attrs["slots"]
    p, dp = p.view(p.shape[0], -1, S), dp.view(dp.shape[0], -1, S)
    return (p * (dp - (p * dp).sum(-1, keepdim=True))).flatten(1)


def r_ds_attn_dq(ex, op, st):
    a = op.attrs
    ds_ = st[op.ins[0]]
    keys, _, _ = _keys(op, st)
    T = ds_.shape[0]
    return (torch.einsum("ths,tsd->thd", ds_.view(T, -1, a["slots"]), keys) * a["scale"]).flatten(1)


def r_ds_attn_dkv(ex, op, st):
    a = op.attrs
    ds_, p, q, do = (st[n] for n in op.ins[:4])
    T, H, S, nw = q.shape[0], a["heads"], a["slots"], a["n_window"]
    hd = op.shape[1]
    t = torch.arange(T, device=q.device)[:, None]
    widx = (t - a["window"] + 1).clamp_min(0) + torch.arange(nw, device=q.device)[None]
    valid = widx <= t
    coef = ds_.view(T, H, S)[:, :, :nw] * a["scale"]
    each = torch.einsum("thj,thd->tjd", coef, q.view(T, H, hd)) + torch.einsum("thj,thd->tjd", p.view(T, H, S)[:, :, :nw],
                                                                             do.view(T, H, hd))
    each = each.masked_fill(~valid[..., None], 0.0)
    out = torch.zeros(T, hd, device=q.device, dtype=_F32).index_add_(0, widx.clamp(max=T - 1).flatten(), each.flatten(0, 1))
    return _acc(st, op, 4, out)


# --- the router and the experts ----------------------------------------------------------------------------------------

def r_ds_moe_scores(ex, op, st):
    num = _num(op)
    s = st[op.ins[0]].to(num.hi) / op.attrs.get("temp", 1.0)
    return _out(s.sigmoid() if op.attrs.get("func") == "sigmoid" else F.softplus(s).sqrt())


def r_ds_bias(ex, op, st):
    return st[op.ins[0]] + _vec(ex, op.attrs["bias"], _F32)


def r_ds_experts(ex, op, st):
    return st[op.ins[0]].topk(op.shape[1], dim=-1).indices.to(_F32)


def r_ds_gates(ex, op, st):
    a = op.attrs
    scores, chosen = st[op.ins[0]], st[op.ins[1]].long()
    g = scores.gather(1, chosen)
    if a.get("norm", True) and chosen.shape[1] > 1:
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-20)
    return g * a.get("route_scale", 1.0)


def _by_expert(chosen):
    """[(expert, its slots' rows)], experts ascending: row t·k + j is token t's slot j."""
    flat = chosen.long().flatten()
    return [(int(e), (flat == e).nonzero()[:, 0]) for e in torch.unique(flat).tolist()]


def r_ds_moe_linear(ex, op, st):
    a = op.attrs
    src, chosen = _in(st, op, 0), st[op.ins[1]]
    k = chosen.shape[1]
    out = torch.zeros(op.shape, device=src.device, dtype=src.dtype)
    for e, rows in _by_expert(chosen):
        x = src[rows // k] if a["per_token"] else src[rows]
        out[rows] = F.linear(x, _dequant(ex, a["weight"].format(e=e), src.dtype))
    return _out(out)


def r_ds_moe_linear_bwd(ex, op, st):
    a = op.attrs
    dy, chosen = st[op.ins[0]], st[op.ins[1]]
    k = chosen.shape[1]
    out = torch.zeros(op.shape, device=dy.device, dtype=_F32)
    for e, rows in _by_expert(chosen):
        d = dy[rows // k] if a.get("per_token") else dy[rows]
        out[rows] = F.linear(d, _dequant(ex, a["weight"].format(e=e), _F32).t().contiguous())
    return _acc(st, op, 2, out)


def _ascending(slots, chosen):
    """A token's k slots (slots [N, k, d]) in ascending expert order, summed in it."""
    order = chosen.long().argsort(dim=-1, stable=True)
    s = slots.gather(1, order[..., None].expand(-1, -1, slots.shape[-1]))
    y = s[:, 0]
    for j in range(1, s.shape[1]):
        y = y + s[:, j]
    return y


def r_ds_moe_sum(ex, op, st):
    num = _num(op)
    routed, chosen, shared = st[op.ins[0]], st[op.ins[1]], st[op.ins[2]]
    y = _ascending(routed.view(routed.shape[0], chosen.shape[1], -1).to(num.hi), chosen)
    return _out((y + shared.to(num.hi)).to(num.dtype))


def r_ds_moe_dx(ex, op, st):
    dxs, chosen = st[op.ins[0]], st[op.ins[1]]
    return _acc(st, op, 2, _ascending(dxs.view(chosen.shape[0], chosen.shape[1], -1), chosen))


def _swiglu_in(st, op, lim, hi):
    g, u = st[op.ins[0]].to(hi), st[op.ins[1]].to(hi)
    if lim > 0:
        u, g = u.clamp(-lim, lim), g.clamp(max=lim)
    return g, u


def r_ds_swiglu(ex, op, st):
    num = _num(op)
    g, u = _swiglu_in(st, op, op.attrs.get("limit", 0), num.hi)
    h = F.silu(g) * u
    if len(op.ins) > 2:
        h = st[op.ins[2]].to(num.hi) * h
    return _out(h.to(num.dtype))


def r_ds_swiglu_bwd(ex, op, st):
    a = op.attrs
    lim = a.get("limit", 0)
    dh, g, u = (st[n] for n in op.ins[:3])
    gc, uc = (g.clamp(max=lim), u.clamp(-lim, lim)) if lim > 0 else (g, u)
    dg_ = dh * st[op.ins[3]] if len(op.ins) > 3 else dh
    if a["part"] == "u":
        y = dg_ * F.silu(gc)
        keep = u.abs() <= lim if lim > 0 else None
    else:
        sig = torch.sigmoid(gc)
        y = (dg_ * uc) * (sig * (1 + gc * (1 - sig)))
        keep = g <= lim if lim > 0 else None
    return y if keep is None else torch.where(keep, y, torch.zeros_like(y))


def r_ds_swiglu_dgate(ex, op, st):
    lim = op.attrs.get("limit", 0)
    dh, g, u = (st[n] for n in op.ins[:3])
    gc, uc = (g.clamp(max=lim), u.clamp(-lim, lim)) if lim > 0 else (g, u)
    return (dh * (F.silu(gc) * uc)).sum(-1, keepdim=True)


def r_ds_gates_bwd(ex, op, st):
    a = op.attrs
    scores, chosen, dg = st[op.ins[0]], st[op.ins[1]].long(), st[op.ins[2]]
    s = scores.gather(1, chosen)
    if a.get("norm", True) and chosen.shape[1] > 1:
        S = s.sum(-1, keepdim=True) + 1e-20
        G = (dg * s).sum(-1, keepdim=True)
        d = dg / S - G / (S * S)
    else:
        d = dg
    d = d * a.get("route_scale", 1.0)
    return torch.zeros_like(scores).scatter(1, chosen, d)


def r_ds_moe_scores_bwd(ex, op, st):
    t = op.attrs.get("temp", 1.0)
    z = st[op.ins[0]] / t
    score = F.softplus(z).sqrt()
    dsp = torch.where(z > 20, torch.ones_like(z), torch.sigmoid(z))
    return ((st[op.ins[1]] / (score * 2.0)) * dsp) / t


# --- the head and the loss ------------------------------------------------------------------------------------------

def _logits(ex, op, st):
    w = ex.w(op.attrs["weight"])[op.attrs["b0"]:op.attrs["b1"]].to(ex.device, _F32)  # a block cast: exact
    return F.linear(st[op.ins[0]], w), w


def r_ds_head_stats(ex, op, st):
    lo, _ = _logits(ex, op, st)
    mb = lo.max(-1).values
    if len(op.ins) > 1:
        prev = st[op.ins[1]]
        m = torch.maximum(prev[:, 0], mb)
        s = prev[:, 1] * torch.exp(prev[:, 0] - m) + torch.exp(lo - m[:, None]).sum(-1)
    else:
        m, s = mb, torch.exp(lo - mb[:, None]).sum(-1)
    return torch.stack((m, s), dim=-1)


def r_ds_head_grad(ex, op, st):
    lo, w = _logits(ex, op, st)
    q = torch.exp(lo - st[op.ins[1]])
    tgt = st[op.ins[2]][:, 0].long()
    b0, b1 = op.attrs["b0"], op.attrs["b1"]
    rows = ((tgt >= b0) & (tgt < b1)).nonzero()[:, 0]
    if len(rows):
        q[rows, tgt[rows] - b0] -= 1
    return _acc(st, op, 3, (q * op.attrs["ic"]) @ w)


# --- a prep unit's: the embedding, engram, the compressor, candidate blocks -------------------------------------------

def r_ds_tokens(ex, op, st):
    t = ex.prog.data.tokens
    starts = np.concatenate([[0], np.cumsum(t.lengths)]).astype(np.int64)
    ids = np.concatenate([t.tokens[starts[s]:starts[s] + int(t.lengths[s])] for s in op.attrs["seqs"]])
    return torch.from_numpy(ids.astype(np.float32))[:, None].to(ex.device)


def r_ds_embed(ex, op, st):
    tok = st[op.ins[0]][:, 0].long()
    return _out(F.embedding(tok, ex.w(op.attrs.get("weight", "embed.weight")).to(ex.device)).to(_num(op).dtype))


def r_ds_tile(ex, op, st):
    x = st[op.ins[0]]
    return x.unsqueeze(1).repeat(1, op.attrs["n"], 1).flatten(1)


def r_ds_onehot(ex, op, st):
    out = torch.zeros(op.shape, device=ex.device, dtype=_F32)
    out[:, op.attrs["col"]] = 1
    return out


def r_ds_cols(ex, op, st):
    return st[op.ins[0]][:, op.attrs["c0"]:op.attrs["c1"]]


def r_ds_engram_rows(ex, op, st):
    """Engram.rows of one sequence's n-grams (engram_hashes from its tokens), from the rows units' slices."""
    a, num, cfg = op.attrs, _num(op), ex.spec.cfg
    layer = a["layer"]
    tok = st[op.ins[0]][:, 0].long()[None].cpu()
    tmap = ex.token_map if ex.token_map is not None else ex.w("engram.token_map")
    hashes = ds.engram_hashes(tok, tmap.cpu(), cfg)[:, :, cfg.engram_layer_ids.index(layer)]
    if layer in ex.slices:
        rows = ex.slices[layer].rows(hashes, num.dtype)
    else:
        base = f"layers.{layer}.engram.embed"
        v = F.embedding(hashes, ex.w(base + ".weight")).float().unflatten(-1, (-1, ds.BLOCK))
        rows = (v * F.embedding(hashes, ex.w(base + ".scale")).float().unsqueeze(-1)).flatten(-2).to(num.dtype)
    return _out(rows[0].flatten(1).to(ex.device))


def r_ds_engram_gate(ex, op, st):
    """Engram.forward after its wkv, as written there."""
    a, num = op.attrs, _num(op)
    hi = num.hi
    x, kv = _in(st, op, 0), _in(st, op, 1)
    dim = kv.shape[1] - x.shape[1]
    hc = x.shape[1] // dim
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.to(hi).unflatten(-1, (hc, dim))
    h = x.view(-1, hc, dim).to(hi)
    rstd = torch.rsqrt(h.square().mean(-1) + a["eps"]) * torch.rsqrt(key.square().mean(-1) + a["eps"])
    dot = (h * (_vec(ex, a["q_weight"], hi) * _vec(ex, a["k_weight"], hi)) * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return _out((h + gate.unsqueeze(-1) * value.to(hi).unsqueeze(-2)).to(x.dtype).flatten(1))


def r_ds_compress(ex, op, st):
    """Compressor.forward's pooling (ratio > 1) of one sequence, before its norm."""
    a, num = op.attrs, _num(op)
    x = _in(st, op)
    r, hi = a["ratio"], num.hi
    n = x.shape[0] // r
    h = x.to(hi)
    kv = F.linear(h, ex.w(a["wkv"]).to(ex.device, hi))[:n * r].unflatten(0, (n, r))
    score = F.linear(h, ex.w(a["wgate"]).to(ex.device, hi))[:n * r].unflatten(0, (n, r))
    return _out((kv * score.softmax(dim=1)).sum(dim=1).to(x.dtype))


def r_ds_candidates(ex, op, st):
    a = op.attrs
    score = st[op.ins[0]]
    visible = (torch.arange(1, score.shape[0] + 1, device=score.device) // a["ratio"]).unsqueeze(-1)
    return ds.candidate_blocks(score[None], visible, a["topk_blocks"], a["block"])[0].to(_F32)


RUN = {name[2:]: fn for name, fn in globals().items() if name.startswith("r_ds_")}


# --- disputes: a worker's side ------------------------------------------------------------------------------------------

class Party(tr.Party):
    """tracer.Party for a DeepSeek unit: its executor, and the weights the referee reads, as the worker holds them."""

    def executor(self, prog, honest: bool) -> Executor:
        return Executor(self.rt, self.spec, prog, None if honest else self.tamper)

    def source(self):
        from .referee_deepseek import RecordSource, Sources, StateWeights
        return Sources(StateWeights(self.rt.weights), RecordSource(self.data))
