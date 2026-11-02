"""The node's referee for a fine-tune's dispute (dispute.py): one op of one training step, a few hundred of its output
numbers, recomputed with NumPy on the CPU and judged against what the worker said they were.

A worker computes in float32, on whatever GPU or CPU it has, so its numbers aren't the exact ones; nor would an
honest CPU's be. The referee works out, in float64, both the exact value of each output number given the worker's own
committed inputs, and a bound on how far any honest float32 computation of it can land (`E`: running error analysis).
Every op is the sequence of float32 operations tracer.py runs, and each step of it adds what one rounding can:
    - a correctly rounded operation (+ − × ÷ √): u·|result|, with u = 2⁻²⁴;
    - exp, log, rsqrt: 4 ulps (CUDA's are 1 to 2, the CPU's at most 1);
    - a dot product or sum of K terms: C · u · (√K · M + ‖terms‖₂), where M is the largest partial sum any honest order
      makes (spread()). Not the worst case γ_K · Σ|terms| (Higham §3.1), which allows K roundings all one way: that's
      about K times more than any real order makes (1.5% of an output at K = 2560 against 3e-7 of honest error), room
      for a worker to compute in TF32 or steer every op and still pass.
Errors in a step's inputs are carried through by the operation's derivative. A worker's number passes when it's within
MARGIN times the bound (for the order a kernel happens to use inside a step, like multiplying by 1/x for ÷x), plus
2⁻³⁰ of the value, plus 2⁻¹²⁶ (for GPUs that flush subnormals). −∞ where the causal mask puts it must be exact.

The sum bound is statistical, not a worst case. A float sum of K terms, in any order, rounds K times, each time by at
most u of a partial sum. The orders real kernels use (sequential, blocked, split-K, pairwise trees and strided
accumulators of up to 32 lanes, as GPU reductions are) only ever form partial sums of runs of terms that are
contiguous at some stride up to 32, whose sizes spread() takes the largest of, M. Rounding errors behave as
independent zero-mean draws of size at most u·M, so their total lies within C·u·√K·M except with a probability
Hoeffding bounds by 2·exp(-C²·3/2); and each product's own rounding adds u·‖terms‖₂ the same way. C is calibrated
(SPREAD_C): honest CPU and CUDA traces of real models stay within a small part of it (tests/test_dispute.py,
CalibrationTests).

Deterministic on every machine. Products of two float32s are exact in float64, and math.fsum rounds a sum exactly, so a
dot product's value is its exact value correctly rounded. Everything else uses only + − × ÷ √, which IEEE 754 rounds
correctly on every platform, applied element by element by NumPy: no pairwise sums (whose order NumPy may change), and
no C library: exp, log, sin and cos are written below from those operations. So two validators, whatever their CPU, OS
or NumPy, compute bit-identical values and bounds, and so identical verdicts. (The workers' kernels are where libm and
CUDA differ by an ulp or two, which is what the 4-ulp allowance is for.)

The table below is keyed by op kind; program.py says which ops a step runs. An op family with other arithmetic, like
DeepSeek's FP8 and FP4 dequantization, its mixture-of-experts router (where a near-tie between experts would need a
margin rule) or hyper-connections, is added as more entries here and in tracer.py.
"""
from __future__ import annotations

import math

import numpy as np

from .merkle import LEAF

U = 2.0 ** -24  # float32's unit roundoff
TRANS = 8.0  # exp, log and rsqrt: up to 4 ulps, which is 8u
MARGIN, REL, FLOOR = 1.25, 2.0 ** -30, 2.0 ** -126
SPREAD_C = 6.0  # the sum bound's multiple of its statistical scale: about 7 standard deviations (see the docstring)
STRIDES = (1, 2, 4, 8, 16, 32)  # the strides at which honest orders accumulate runs of terms


def f32(x: float) -> float:
    return float(np.float32(x))


def gamma(k: int) -> float:
    return k * U / (1 - k * U)


# --- exp, log, sin and cos from + − × ÷ alone -------------------------------------------------------------------------

_LN2_HI, _LN2_LO = 6.93147180369123816490e-01, 1.90821492927058770002e-10  # fdlibm's split: k·HI is exact
_INV_LN2 = 1.44269504088896338700e+00
_EXP_C = [1.0 / math.factorial(n) for n in range(14)]  # e^r to r¹³/13!, for |r| ≤ ln2/2: below 2⁻⁵³ after that
_LOG_C = [2.0 / (2 * n + 1) for n in range(12)]  # log((1+s)/(1−s)) = 2(s + s³/3 + …), |s| ≤ 0.172
_PIO2 = (1.57079632673412561417e+00, 6.07710050630396597660e-11, 2.02226624871116645580e-21)  # π/2 in 33-bit parts
_INV_PIO2 = 6.36619772367581382433e-01


def dexp(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x > 0, np.inf, 0.0)
    ok = (x > -746.0) & (x < 710.0)
    xs = x[ok]
    k = np.rint(xs * _INV_LN2)
    r = (xs - k * _LN2_HI) - k * _LN2_LO
    p = np.full_like(r, _EXP_C[13])
    for c in reversed(_EXP_C[:13]):
        p = p * r + c
    out[ok] = np.ldexp(p, k.astype(np.int64))
    out[np.isnan(x)] = np.nan
    return out


def dlog(x) -> np.ndarray:
    """Natural log of positive numbers (−∞ at 0, NaN below)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    out[x == 0] = -np.inf
    out[x == np.inf] = np.inf
    ok = (x > 0) & (x < np.inf)
    m, e = np.frexp(x[ok])
    low = m < 0.70710678118654752440
    m = np.where(low, m * 2, m)
    e = np.where(low, e - 1, e).astype(np.float64)
    f = m - 1.0
    s = f / (2.0 + f)
    s2 = s * s
    t = np.full_like(s, _LOG_C[11])
    for c in reversed(_LOG_C[:11]):
        t = t * s2 + c
    out[ok] = e * _LN2_HI + (s * t + e * _LN2_LO)
    return out


def dsincos(x) -> tuple[np.ndarray, np.ndarray]:
    """sin and cos of |x| < 2²⁰ (rotary angles: positions times frequencies up to 1)."""
    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x * _INV_PIO2)
    r = ((x - k * _PIO2[0]) - k * _PIO2[1]) - k * _PIO2[2]
    r2 = r * r
    s = np.full_like(r, (-1) ** 8 / math.factorial(17))
    for n in range(7, -1, -1):
        s = s * r2 + (-1) ** n / math.factorial(2 * n + 1)
    s = s * r
    c = np.full_like(r, (-1) ** 9 / math.factorial(18))
    for n in range(8, -1, -1):
        c = c * r2 + (-1) ** n / math.factorial(2 * n)
    q = k.astype(np.int64) % 4
    sin = np.select([q == 0, q == 1, q == 2], [s, c, -s], -c)
    cos = np.select([q == 0, q == 1, q == 2], [c, -s, -c], s)
    return sin, cos


def rope_tables(arch, length: int) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin, [length, head_dim], as llm.rope_table makes them (rounded to float32), from + − × ÷ alone."""
    d = arch.head_dim
    inv = dexp(-(np.arange(0, d, 2, dtype=np.float64) / d) * dlog(np.float64(arch.rope_theta)))
    if arch.rope_scaling:
        factor, low, high, old = arch.rope_scaling
        wavelen = (2 * math.pi) / inv
        scaled = np.where(wavelen > old / low, inv / factor, inv)
        smooth = (old / wavelen - low) / (high - low)
        medium = (wavelen >= old / high) & (wavelen <= old / low)
        inv = np.where(medium, (1 - smooth) * scaled / factor + smooth * scaled, scaled)
    angles = np.arange(length, dtype=np.float64)[:, None] * inv[None, :]
    angles = np.concatenate([angles, angles], axis=1)
    sin, cos = dsincos(angles)
    return cos.astype(np.float32).astype(np.float64), sin.astype(np.float32).astype(np.float64)


# --- numbers with a bound on how far float32 can take them -----------------------------------------------------------

class E:
    """`v`: a value worked out in float64; `e`: how far from it a float32 computation of it can be."""
    __slots__ = ("v", "e")

    def __init__(self, v, e=None):
        self.v = np.asarray(v, dtype=np.float64)
        e = np.zeros(self.v.shape) if e is None else np.asarray(e, dtype=np.float64)
        self.e = e if e.shape == self.v.shape else np.broadcast_to(e, self.v.shape).copy()

    def __getitem__(self, i):
        return E(self.v[i], self.e[i])

    def __neg__(self):
        return E(-self.v, self.e)


def _r(v, e, k=1.0) -> E:
    """The float32 rounding of a result: k·u of it, at most."""
    with np.errstate(invalid="ignore", over="ignore"):
        return E(v, e + k * U * (np.abs(v) + e))


def add(a: E, b: E) -> E:
    return _r(a.v + b.v, a.e + b.e)


def sub(a: E, b: E) -> E:
    return _r(a.v - b.v, a.e + b.e)


def mul(a: E, b: E) -> E:
    return _r(a.v * b.v, np.abs(a.v) * b.e + np.abs(b.v) * a.e + a.e * b.e)


def scale(a: E, c: float) -> E:
    """× a float32 constant."""
    return _r(a.v * c, abs(c) * a.e)


def div(a: E, b: E) -> E:
    with np.errstate(divide="ignore", invalid="ignore"):
        v = a.v / b.v
        lo = np.abs(b.v) - b.e
        e = np.where(lo > 0, (a.e + np.abs(v) * b.e) / np.where(lo > 0, lo, 1.0), np.inf)
    return _r(v, e)


def sqrt(a: E) -> E:
    v = np.sqrt(np.maximum(a.v, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(a.e > 0, a.e / (np.sqrt(np.maximum(a.v - a.e, 0.0)) + v), 0.0)
    return _r(v, e)


def rsqrt(a: E) -> E:
    lo = a.v - a.e
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1.0 / np.sqrt(a.v)
        e = np.where(lo > 0, a.e / (2 * lo * np.sqrt(np.where(lo > 0, lo, 1.0))), np.inf)  # |d(x^-½)/dx| at its worst
    return _r(v, e, TRANS)


def exp(a: E) -> E:
    v = dexp(a.v)
    with np.errstate(invalid="ignore", over="ignore"):
        e = np.where(a.e > 0, v * a.e * dexp(a.e), 0.0)  # e^x·(e^δ − 1) ≤ e^x·δ·e^δ
    return _r(v, e, TRANS)


def log(a: E) -> E:
    lo = a.v - a.e
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(lo > 0, a.e / np.where(lo > 0, lo, 1.0), np.inf)
    return _r(dlog(a.v), e, TRANS)


def fsum_rows(x: np.ndarray) -> np.ndarray:
    """Each row's exact sum, correctly rounded: math.fsum, never NumPy's pairwise sums."""
    x = np.asarray(x, dtype=np.float64)
    return np.array([math.fsum(row) for row in x.reshape(-1, x.shape[-1]).tolist()]).reshape(x.shape[:-1])


def spread(t: np.ndarray) -> np.ndarray:
    """√K · M + ‖t‖₂ along the last axis: the scale of the rounding error any honest order makes summing K terms (see
    the module's docstring). M is the largest |sum| of a run of terms contiguous at a stride in STRIDES, from running
    sums (NumPy's cumsum is sequential, so every machine gets the same M)."""
    t = np.asarray(t, dtype=np.float64)
    k = t.shape[-1]
    m = np.zeros(t.shape[:-1])
    for s in STRIDES:
        if s > k:
            break
        pad = (-k) % s
        tt = np.concatenate([t, np.zeros(t.shape[:-1] + (pad,))], -1) if pad else t
        tt = tt.reshape(t.shape[:-1] + (-1, s))
        c = np.cumsum(tt, axis=-2)
        hi = np.maximum(c.max(-2), 0.0)
        lo = np.minimum(c.min(-2), 0.0)
        m = np.maximum(m, (hi - lo).max(-1))
    return math.sqrt(k) * m + np.sqrt(np.einsum("...i,...i->...", t, t))


def esum(a: E) -> E:
    """The sum along the last axis of K terms, in any honest order."""
    ae = fsum_rows(a.e) if np.any(a.e) else np.zeros(a.v.shape[:-1])
    return E(fsum_rows(a.v), ae * (1 + 2 * U) + SPREAD_C * U * spread(a.v))


def contract(a: E, b: np.ndarray) -> E:
    """a [n, K] · bᵀ, b [m, K] exact: [n, m], a dot product of K terms each."""
    a_v, a_e, b = np.atleast_2d(a.v), np.atleast_2d(a.e), np.atleast_2d(np.asarray(b, dtype=np.float64))
    n, m = a_v.shape[0], b.shape[0]
    p = a_v[:, None, :] * b[None, :, :]
    ae = fsum_rows(a_e[:, None, :] * np.abs(b)[None, :, :]) if np.any(a_e) else np.zeros((n, m))
    return E(fsum_rows(p), ae * (1 + 2 * U) + SPREAD_C * U * spread(p))


def bcast(a: E) -> E:
    """A per-row value [n] as [n, 1], to go with a row of a group."""
    return E(a.v[:, None], a.e[:, None])


def _cat(*parts: E) -> E:
    return E(np.concatenate([p.v for p in parts]), np.concatenate([p.e for p in parts]))


# --- the ops ---------------------------------------------------------------------------------------------------------
# Each returns E for output row `r`, columns [c0, c1) of the op's 2-D output, reading through `x` (an Inputs): its
# input tensors' numbers (proven against the state), the base model's weights and the job's data.

def _norm(x, xs: np.ndarray, g: int, eps: float) -> E:
    """rsqrt(mean(x²) + eps) for each group of a row, as rmsnorm and its backward compute it."""
    sq = _r(xs * xs, 0.0)
    s = esum(sq)
    ms = _r(s.v / g, s.e / g, 2.0)  # ÷g, or ×(1/g)
    return rsqrt(_r(ms.v + f32(eps), ms.e))


def op_embed(op, x, r, c0, c1):
    ids, _ = x.tokens(op.attrs["seq"])
    return E(x.weight(op.attrs["weight"], int(ids[r]), int(ids[r]) + 1)[0, c0:c1])


def op_rmsnorm(op, x, r, c0, c1):
    g = op.attrs["group"]
    xs = x.row(op.ins[0], r).reshape(-1, g)
    w = x.weight(op.attrs["weight"], 0, 1)[0]
    y = mul(E(w[None, :]), mul(E(xs), bcast(_norm(x, xs, g, op.attrs["eps"]))))
    return E(y.v.reshape(-1)[c0:c1], y.e.reshape(-1)[c0:c1])


def op_linear(op, x, r, c0, c1):
    y = contract(E(x.row(op.ins[0], r)[None, :]), x.weight(op.attrs["weight"], c0, c1))[0]
    bias = op.attrs.get("bias")
    return add(y, E(x.weight(bias, 0, 1)[0, c0:c1])) if bias else y


def op_lora_down(op, x, r, c0, c1):
    return contract(E(x.row(op.ins[0], r)[None, :]), x.lora_a(op.attrs["module"])[c0:c1])[0]


def op_lora_up(op, x, r, c0, c1):
    base, xa, b = op.ins
    t = contract(E(x.row(xa, r)[None, :]), x.get(b, c0, c1))[0]
    return add(E(x.get(base, r, r + 1, c0, c1)[0]), scale(t, f32(op.attrs["scaling"])))


def _tables(op, x, r):
    cos, sin = x.rope(op, r)
    return E(cos, 2 * U * np.abs(cos)), E(sin, 2 * U * np.abs(sin))  # a worker's table may differ by an ulp


def op_rope(op, x, r, c0, c1):
    hd = op.attrs["hd"]
    xs = x.row(op.ins[0], r).reshape(-1, hd)
    half = hd // 2
    rot = np.concatenate([-xs[:, half:], xs[:, :half]], axis=1)
    cos, sin = _tables(op, x, r)
    y = add(mul(E(xs), E(cos.v[None], cos.e[None])), mul(E(rot), E(sin.v[None], sin.e[None])))
    return E(y.v.reshape(-1)[c0:c1], y.e.reshape(-1)[c0:c1])


def _heads(op):
    a = op.attrs
    return a["H"], a["KV"], a["hd"], a["H"] // a["KV"]


def op_attn_scores(op, x, r, c0, c1):
    q, k = op.ins
    _, _, hd, G = _heads(op)
    T = op.shape[1]
    h, tq = divmod(r, T)
    g = h // G
    s = contract(E(x.get(q, tq, tq + 1, h * hd, (h + 1) * hd)), x.get(k, c0, c1, g * hd, (g + 1) * hd))[0]
    s = scale(s, f32(op.attrs["c"]))
    masked = np.arange(c0, c1) > tq
    return E(np.where(masked, -np.inf, s.v), np.where(masked, 0.0, s.e))


def op_softmax(op, x, r, c0, c1):
    s = x.row(op.ins[0], r)
    fin = np.isfinite(s)
    z = _r(s[fin] - s[fin].max(), 0.0)
    ex = exp(z)
    p = div(ex, esum(E(ex.v[None], ex.e[None]))[0:1])
    v, e = np.zeros(len(s)), np.zeros(len(s))
    v[fin], e[fin] = p.v, p.e
    return E(v[c0:c1], e[c0:c1])


def _by_head(c0, c1, hd):
    """The columns [c0, c1) of a [T, heads·hd] tensor, split at head boundaries: (head, first, end) within it."""
    out, c = [], c0
    while c < c1:
        h, j = divmod(c, hd)
        end = min(c1, (h + 1) * hd)
        out.append((h, j, j + end - c))
        c = end
    return out


def op_attn_mix(op, x, r, c0, c1):
    p, v = op.ins
    _, _, hd, G = _heads(op)
    T = x.shape(p)[1]
    parts = []
    for h, j0, j1 in _by_head(c0, c1, hd):
        g = h // G
        parts.append(contract(E(x.row(p, h * T + r)[None]), x.get(v, 0, T, g * hd + j0, g * hd + j1).T)[0])
    return _cat(*parts)


def op_add(op, x, r, c0, c1):
    a, b = op.ins
    return add(E(x.get(a, r, r + 1, c0, c1)[0]), E(x.get(b, r, r + 1, c0, c1)[0]))


OVERFLOW = 88.0  # float32's e^x overflows to ∞ just above 88.72: kernels' e^−g may, for g below −88


def _sigmoid(g: np.ndarray) -> E:
    """1/(1 + e^−g). Below −OVERFLOW, float32's e^−g may be ∞, so an honest kernel gets 0 where the value is under
    2⁻¹²⁶ (or a subnormal, by e^g/(1 + e^g)): any number from the value to 0 passes, worked out as e^g/(1 + e^g),
    which doesn't overflow (and is exactly 0 below −746, where the bound used to be ∞ and anything passed)."""
    g = np.asarray(g, dtype=np.float64)
    far = g < -OVERFLOW
    s = div(E(1.0), add(E(1.0), exp(E(np.where(far, 0.0, -g)))))
    t = dexp(np.where(far, g, 0.0))
    t = t / (1.0 + t)
    return E(np.where(far, t, s.v), np.where(far, t, s.e))


def _silu(g: np.ndarray) -> E:
    """g/(1 + e^−g), and below −OVERFLOW g·e^g/(1 + e^g), anything from it to 0 (see _sigmoid)."""
    g = np.asarray(g, dtype=np.float64)
    far = g < -OVERFLOW
    s = div(E(g), add(E(1.0), exp(E(np.where(far, 0.0, -g)))))
    t = dexp(np.where(far, g, 0.0))
    t = g * (t / (1.0 + t))
    return E(np.where(far, t, s.v), np.where(far, np.abs(t), s.e))


def op_silu_mul(op, x, r, c0, c1):
    g, u = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    return mul(_silu(g), E(u))


def _positions(x, seq):
    ids, mask = x.tokens(seq)
    pos = np.flatnonzero(mask[1:]).astype(np.int64)
    return pos, ids[pos + 1]


def _logits(op, x, r) -> tuple[E, int, int]:
    """One target position's logits over the op's block of the vocabulary: (logits, its position, its target)."""
    pos, tgt = _positions(x, op.attrs["seq"])
    f = x.row(op.ins[0], int(pos[r]))
    w = x.weight(op.attrs["weight"], op.attrs["b0"], op.attrs["b1"])
    return contract(E(f[None]), w)[0], w, int(tgt[r])


def op_head_stats(op, x, r, c0, c1):
    lo, _, _ = _logits(op, x, r)
    mb = E(lo.v.max(), lo.e.max())
    if len(op.ins) > 1:
        pm, ps = x.row(op.ins[1], r)
        m = E(max(pm, float(mb.v)), mb.e)
        s = add(mul(E(ps), exp(sub(E(pm), m))), esum(exp(sub(lo, m))))
    else:
        m, s = mb, esum(exp(sub(lo, mb)))
    out = _cat(E(np.atleast_1d(m.v), np.atleast_1d(m.e)), E(np.atleast_1d(s.v), np.atleast_1d(s.e)))
    return out[c0:c1]


def op_head_lse(op, x, r, c0, c1):
    m, s = x.row(op.ins[0], r)
    return _cat(add(E([m]), log(E([s]))))[c0:c1]


def op_head_grad(op, x, r, c0, c1):
    lo, w, tgt = _logits(op, x, r)
    lse = x.row(op.ins[1], r)[0]
    ex = exp(sub(lo, E(lse)))
    b0 = op.attrs["b0"]
    if b0 <= tgt < op.attrs["b1"]:
        one = sub(ex[tgt - b0:tgt - b0 + 1], E([1.0]))
        ex = E(ex.v.copy(), ex.e.copy())
        ex.v[tgt - b0], ex.e[tgt - b0] = one.v[0], one.e[0]
    q = scale(ex, f32(op.attrs["ic"]))
    g = contract(E(q.v[None], q.e[None]), w[:, c0:c1].T)[0]
    return add(E(x.get(op.ins[2], r, r + 1, c0, c1)[0]), g) if len(op.ins) > 2 else g


def op_scatter(op, x, r, c0, c1):
    pos, _ = _positions(x, op.attrs["seq"])
    hit = np.flatnonzero(pos == r)
    if not len(hit):
        return E(np.zeros(c1 - c0))
    return E(x.get(op.ins[0], int(hit[0]), int(hit[0]) + 1, c0, c1)[0])


def op_rmsnorm_bwd(op, x, r, c0, c1):
    g = op.attrs["group"]
    xs, dys = (x.row(n, r).reshape(-1, g) for n in op.ins[:2])
    w = x.weight(op.attrs["weight"], 0, 1)[0]
    rr = bcast(_norm(x, xs, g, op.attrs["eps"]))
    dyw = mul(E(dys), E(w[None, :]))
    ds = esum(mul(dyw, E(xs)))
    dm = bcast(_r(ds.v / g, ds.e / g, 2.0))
    dx = mul(rr, sub(dyw, mul(E(xs), mul(mul(rr, rr), dm))))
    dx = E(dx.v.reshape(-1)[c0:c1], dx.e.reshape(-1)[c0:c1])
    return add(E(x.get(op.ins[2], r, r + 1, c0, c1)[0]), dx) if len(op.ins) > 2 else dx


def op_linear_bwd(op, x, r, c0, c1):
    dy = x.row(op.ins[0], r)
    w = x.weight(op.attrs["weight"], 0, x.weight_rows(op.attrs["weight"]))[:, c0:c1]
    y = contract(E(dy[None]), w.T)[0]
    return add(E(x.get(op.ins[1], r, r + 1, c0, c1)[0]), y) if len(op.ins) > 1 else y


def op_scale(op, x, r, c0, c1):
    return scale(E(x.get(op.ins[0], r, r + 1, c0, c1)[0]), f32(op.attrs["c"]))


def op_lora_grad(op, x, r, c0, c1):
    dys, xa = op.ins[:2]
    T = x.shape(dys)[0]
    y = contract(E(x.get(dys, 0, T, r, r + 1)[:, 0][None]), x.get(xa, 0, T, c0, c1).T)[0]
    return add(E(x.get(op.ins[2], r, r + 1, c0, c1)[0]), y) if len(op.ins) > 2 else y


def op_lora_bwd_mid(op, x, r, c0, c1):
    dys, b = op.ins
    return contract(E(x.row(dys, r)[None]), x.get(b, 0, x.shape(b)[0], c0, c1).T)[0]


def op_lora_bwd_x(op, x, r, c0, c1):
    mid, acc = op.ins
    y = contract(E(x.row(mid, r)[None]), x.lora_a(op.attrs["module"])[:, c0:c1].T)[0]
    return add(E(x.get(acc, r, r + 1, c0, c1)[0]), y)


def op_silu_bwd_g(op, x, r, c0, c1):
    g, u, dm = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    sig = _sigmoid(g)
    dsig = mul(sig, add(E(1.0), mul(E(g), sub(E(1.0), sig))))
    return mul(mul(E(dm), E(u)), dsig)


def op_silu_bwd_u(op, x, r, c0, c1):
    g, dm = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    return mul(E(dm), _silu(g))


def op_attn_dp(op, x, r, c0, c1):
    do, v = op.ins
    _, _, hd, G = _heads(op)
    T = op.shape[1]
    h, tq = divmod(r, T)
    g = h // G
    return contract(E(x.get(do, tq, tq + 1, h * hd, (h + 1) * hd)), x.get(v, c0, c1, g * hd, (g + 1) * hd))[0]


def _grouped(op, x, r, c0, c1, rows_of, cols_of, c=None):
    """Σ over a key/value head's query heads and every query position: attn_dv (rows_of p, cols_of do) and attn_dk
    (rows_of ds·c, cols_of q). Column `r` of each query head's [T, T] block, against that head's columns."""
    a_name, b_name = rows_of, cols_of
    _, _, hd, G = _heads(op)
    T = x.shape(a_name)[1]
    parts = []
    for g, j0, j1 in _by_head(c0, c1, hd):
        heads = range(g * G, (g + 1) * G)
        col = np.concatenate([x.get(a_name, h * T, (h + 1) * T, r, r + 1)[:, 0] for h in heads])
        a = E(col) if c is None else scale(E(col), f32(c))
        b = np.concatenate([x.get(b_name, 0, T, h * hd + j0, h * hd + j1) for h in heads])
        parts.append(contract(E(a.v[None], a.e[None]), b.T)[0])
    return _cat(*parts)


def op_attn_dv(op, x, r, c0, c1):
    return _grouped(op, x, r, c0, c1, op.ins[0], op.ins[1])


def op_attn_dk(op, x, r, c0, c1):
    return _grouped(op, x, r, c0, c1, op.ins[0], op.ins[1], op.attrs["c"])


def op_softmax_bwd(op, x, r, c0, c1):
    p, dp = (x.row(n, r) for n in op.ins)
    s = esum(mul(E(dp[None]), E(p[None])))
    return mul(E(p[c0:c1]), sub(E(dp[c0:c1]), s))


def op_attn_dq(op, x, r, c0, c1):
    ds, k = op.ins
    _, _, hd, G = _heads(op)
    T = x.shape(ds)[1]
    parts = []
    for h, j0, j1 in _by_head(c0, c1, hd):
        g = h // G
        a = scale(E(x.row(ds, h * T + r)), f32(op.attrs["c"]))
        parts.append(contract(E(a.v[None], a.e[None]), x.get(k, 0, T, g * hd + j0, g * hd + j1).T)[0])
    return _cat(*parts)


def op_rope_bwd(op, x, r, c0, c1):
    hd = op.attrs["hd"]
    dy = x.row(op.ins[0], r).reshape(-1, hd)
    half = hd // 2
    cos, sin = _tables(op, x, r)
    a = mul(E(dy), E(cos.v[None], cos.e[None]))
    b = mul(E(dy), E(sin.v[None], sin.e[None]))
    rb = E(np.concatenate([b.v[:, half:], -b.v[:, :half]], axis=1), np.concatenate([b.e[:, half:], b.e[:, :half]], axis=1))
    y = add(a, rb)
    return E(y.v.reshape(-1)[c0:c1], y.e.reshape(-1)[c0:c1])


def op_adam_m(op, x, r, c0, c1):
    m, g = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    return add(scale(E(m), op.attrs["b1"]), scale(E(g), op.attrs["c1m"]))


def op_adam_v(op, x, r, c0, c1):
    v, g = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    return add(scale(E(v), op.attrs["b2"]), scale(mul(E(g), E(g)), op.attrs["c2m"]))


def op_adam_b(op, x, r, c0, c1):
    b, m, v = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins)
    a = op.attrs
    den = add(sqrt(div(E(v), E(a["c2"]))), E(a["eps"]))
    return sub(E(b), scale(div(div(E(m), E(a["c1"])), den), a["lr"]))


OPS = {name[3:]: fn for name, fn in globals().items() if name.startswith("op_")}
EXACT = {"embed", "scatter"}  # copies: no rounding at all


class Missing(LookupError):
    """The referee asked for a number the evidence doesn't prove."""


class Unresolved(ArithmeticError):
    """A number of the tile has no finite bound (a division by an interval around 0, say): the referee can't say
    either way, so the game ends unchecked. An infinite bound used to let any number pass."""


def tile_rows(shape: tuple[int, int], leaf: int) -> list[tuple[int, int, int]]:
    """The output numbers a tile (one leaf of the output's Merkle tree) covers, as (row, first column, end)."""
    rows, cols = shape
    start, end = leaf * LEAF, min(rows * cols, (leaf + 1) * LEAF)
    out, at = [], start
    while at < end:
        r, c = divmod(at, cols)
        stop = min(end, (r + 1) * cols)
        out.append((r, c, c + stop - at))
        at = stop
    return out


def judge(op, x, leaf: int, said: np.ndarray) -> tuple[bool, str]:
    """Whether the worker's numbers `said` for one tile of `op`'s output are what float32 can make of its inputs.
    (passed, why) — `why` names the worst number either way, for the record."""
    want = [OPS[op.kind](op, x, r, c0, c1) for r, c0, c1 in tile_rows(op.shape, leaf)]
    v = np.concatenate([w.v for w in want])
    e = np.concatenate([w.e for w in want])
    said = np.asarray(said, dtype=np.float64)
    if said.shape != v.shape:
        return False, "the tile has the wrong size"
    masked = np.isinf(v)
    loose = ~masked & ~(np.isfinite(e) & ~np.isnan(v))
    if loose.any():
        i = int(np.argmax(loose))
        raise Unresolved(f"{op.kind} → {op.out}, number {leaf * LEAF + i}: the referee can't bound it "
                         f"(exact {float(v[i])!r}, bound {float(e[i])!r})")
    allowed = MARGIN * e + REL * np.abs(v) + FLOOR
    with np.errstate(invalid="ignore"):
        off = np.where(masked, np.where(said == v, 0.0, np.inf), np.abs(said - v) / allowed)
    off = np.where(np.isnan(off), np.inf, off)
    i = int(np.argmax(off))
    worst = float(off[i])
    # Python's own float formatting, not NumPy's (which changed between versions): the record keeps this text, and
    # every validator has to write it byte for byte the same.
    detail = (f"{op.kind} → {op.out}, number {leaf * LEAF + i}: said {float(said[i])!r}, exact {float(v[i])!r}, "
              f"allowed ±{float(allowed[i]):.3g} ({worst:.3g} of that)")
    return bool(worst <= 1.0), detail
