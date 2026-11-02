"""The referee's DeepSeek-V4.1 ops (DEEPSEEK.md §8): one op of a prep unit or of a top-layer training unit, a tile of
its output recomputed with NumPy, as referee.py does for Qwen-style models. Registered into referee.OPS under
"ds_" names.

Two arithmetics, as attrs["num"]:
- "ref", deepseek.Numerics(): the reference's, which prep units run. Activations are bf16; the code upcasts to
  float32 where the reference does; FP8 and FP4 activation rounding are ops of their own.
- "f32", deepseek_top.TRAIN: float32 throughout, no activation rounding, which training units run.
Each op is the torch expression in deepseek.py its docstring names, and its bound follows that expression step by step
(referee.py's E). A result cast to bf16 adds bf16's unit roundoff, 2⁻⁸ of it. Matmuls and sums are assumed to
accumulate in at least float32: cuBLAS with reduced-precision reductions off (deepseek.configure_torch), and oneDNN
on CPUs, do.

Exact ops (decoding weights, the FP8/FP4 rounding, engram hashes and rows) are judged by equality: they're
recomputed exactly here, in float32 where the reference computes in float32 (NumPy's float32 operations are IEEE
correctly rounded, as torch's are). The one float op whose rounding IEEE doesn't pin, powf in the rotary table, is
bounded, not trusted.

Weights are read by checkpoint name: a quantised matrix as NAME.weight plus NAME.scale (FP8 E4M3 with a ue8m0 scale
per 32x32 block, or FP4 E2M1 packed two to a byte with one per 32 inputs), the rest as bf16 or float32, all decoded
exactly in float64 (dispute.decode).
"""
from __future__ import annotations

import numpy as np

from . import referee as rf
from .referee import E, add, div, exp

U_BF16 = 2.0 ** -8  # bf16's unit roundoff (8 significand bits)
FP8_MAX, FP4_MAX, BLOCK = 448.0, 6.0, 32
F32 = np.float32


# --- exact formats ----------------------------------------------------------------------------------------------------

def _e4m3_table() -> np.ndarray:
    """Every E4M3 (fn) byte's value: 4 exponent bits (bias 7), 3 mantissa bits, no infinities, 0x7f and 0xff NaN."""
    out = np.empty(256)
    for b in range(256):
        sign, e, m = -1.0 if b & 0x80 else 1.0, (b >> 3) & 0xF, b & 7
        if e == 0xF and m == 7:
            out[b] = np.nan
        elif e == 0:
            out[b] = sign * m * 2.0 ** -9
        else:
            out[b] = sign * (1 + m / 8) * 2.0 ** (e - 7)
    return out


E4M3 = _e4m3_table()
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def decode_e8m0(raw: np.ndarray) -> np.ndarray:
    """ue8m0 scales: 2^(byte - 127); 255 is NaN."""
    b = np.asarray(raw, dtype=np.uint8).astype(np.int64)
    return np.where(b == 255, np.nan, np.ldexp(1.0, b - 127))


def _exponent(a: np.ndarray) -> np.ndarray:
    """floor(log2(a)) of positive float64s, exactly (frexp, not log2's rounding)."""
    _, e = np.frexp(a)
    return e.astype(np.int64) - 1


def round_e4m3(v: np.ndarray) -> np.ndarray:
    """Values in [-448, 448] rounded to the nearest E4M3 number, ties to even, as torch's cast to float8_e4m3fn does:
    steps of 2^(e-3) at exponent e, and 2^-9 below the smallest normal, 2^-6."""
    v = np.asarray(v, dtype=np.float64)
    a = np.abs(v)
    e = np.maximum(_exponent(np.where(a > 0, a, 1.0)), -6)
    step = np.ldexp(1.0, e - 3)
    return np.copysign(np.rint(a / step) * step, v)


def fp4_grid(v: np.ndarray) -> np.ndarray:
    """deepseek._fp4_grid: the nearest E2M1 value to each of v in [-6, 6], ties to even."""
    v = np.asarray(v, dtype=np.float64)
    a = np.abs(v)
    r = np.where(a < 2, np.rint(a * 2) / 2, np.where(a < 4, np.rint(a), np.rint(a / 2) * 2))
    return np.copysign(r, v)


def pow2_ceil(x) -> np.ndarray:
    """deepseek._pow2_ceil: 2^ceil(log2(x)) of positive normal float32s, from their bits."""
    bits = np.asarray(x, dtype=F32).view(np.int32).astype(np.int64)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0)
    return np.ldexp(1.0, e)


def round_bf16(v) -> np.ndarray:
    """Float32 values rounded to bf16, ties to even: what `.to(torch.bfloat16)` does to them."""
    bits = np.asarray(v, dtype=F32).view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return bits.astype(np.uint32).view(F32).astype(np.float64)


def round_fp8(row, block: int = BLOCK) -> np.ndarray:
    """deepseek.round_fp8 of one row: per `block` values, the scale 2^ceil(log2(max(amax, 1e-4) · f32(1/448))), that
    product rounded to float32, then each value the nearest E4M3 number times it."""
    b = np.asarray(row, dtype=np.float64).reshape(-1, block)
    amax = np.maximum(np.abs(b).max(-1, keepdims=True), float(F32(1e-4)))
    s = pow2_ceil(F32(amax) * F32(1 / FP8_MAX))
    return (round_e4m3(np.clip(b / s, -FP8_MAX, FP8_MAX)) * s).reshape(-1)


def round_fp4(row, block: int = BLOCK, e4m3_scale: bool = False) -> np.ndarray:
    """deepseek.round_fp4 of one row: E2M1 values times one scale per `block`, either 2^ceil(log2(amax · f32(1/6)))
    or (e4m3_scale) the E4M3 number nearest f32(max(amax, 6·2^-9) / 6), capped at 448."""
    b = np.asarray(row, dtype=np.float64).reshape(-1, block)
    amax = np.abs(b).max(-1, keepdims=True)
    if e4m3_scale:
        s = round_e4m3(np.minimum((F32(np.maximum(amax, FP4_MAX * 2.0 ** -9)) / F32(FP4_MAX)).astype(np.float64),
                                  FP8_MAX))
    else:
        s = pow2_ceil(F32(np.maximum(amax, FP4_MAX * 2.0 ** -126)) * F32(1 / FP4_MAX))
    q = np.clip((F32(b) / F32(s)).astype(np.float64), -FP4_MAX, FP4_MAX)
    return (fp4_grid(q) * s).reshape(-1)


# --- reading DeepSeek's weights ----------------------------------------------------------------------------------------

def qrows(x, base: str, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of a projection, dequantised exactly: base.weight times base.scale (FP8 32x32 blocks or FP4 32
    inputs); or base.weight itself, when it's bf16 or float32 and has no scale."""
    w = x.weight(base + ".weight", r0, r1)
    try:
        sdtype, _ = x.outside.source.shape(base + ".scale")
    except Exception:  # noqa: BLE001 - no scale: a plain matrix
        return w
    if sdtype != "F8_E8M0":
        raise rf.Missing(f"{base}.scale is {sdtype}, not ue8m0")
    wdtype, wshape = x.outside.source.shape(base + ".weight")
    _, sshape = x.outside.source.shape(base + ".scale")
    if wdtype == "I8" or sshape[0] == wshape[0]:  # FP4, or an FP8 table (engram's): a scale per 32 of each row
        s = x.weight(base + ".scale", r0, r1)
        return (w.reshape(r1 - r0, -1, BLOCK) * s[:, :, None]).reshape(r1 - r0, -1)
    b0 = r0 // BLOCK
    s = x.weight(base + ".scale", b0, (r1 - 1) // BLOCK + 1)  # FP8: one per 32x32 block
    rows = np.repeat(s, BLOCK, axis=0)[r0 - b0 * BLOCK:r1 - b0 * BLOCK]
    return w * np.repeat(rows, BLOCK, axis=1)[:, :w.shape[1]]


def vec(x, name: str) -> np.ndarray:
    """A 1-D parameter (a norm's weight, a bias, the sinks), exactly."""
    return x.weight(name, 0, 1)[0]


# --- bounds -----------------------------------------------------------------------------------------------------------

def cast(a: E, num: str) -> E:
    """A result rounded to the activation dtype: bf16 under "ref"; nothing more under "f32"."""
    if num == "f32":
        return a
    with np.errstate(invalid="ignore", over="ignore"):
        return E(a.v, a.e + U_BF16 * (np.abs(a.v) + a.e))


def sigmoid(a: E) -> E:
    """torch.sigmoid: 1 / (1 + exp(-a)), each step rounded."""
    return div(E(1.0), add(E(1.0), exp(-a)))


class StateWeights:
    """A weight source (dispute.Outside's) over a dict of torch tensors as DeepSeek stores them: what a worker reads
    the referee's inputs from to build its evidence, and what tests use. The node's is dispute.HubWeights."""

    NAMES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32", "float8_e4m3fn": "F8_E4M3",
             "float8_e8m0fnu": "F8_E8M0", "int8": "I8", "int32": "I32", "int64": "I64"}

    def __init__(self, state: dict):
        self.state = state

    def shape(self, name):
        t = self.state.get(name)
        if t is None:
            from .dispute import WeightsUnavailable
            raise WeightsUnavailable(f"no {name}")
        return self.NAMES[str(t.dtype).replace("torch.", "")], tuple(t.shape)

    def read(self, name, r0, r1) -> bytes:
        import torch

        t = self.state[name]
        t = t[r0:r1] if t.dim() > 1 else t
        width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        return t.detach().contiguous().cpu().view(width).numpy().tobytes()


# --- the ops ----------------------------------------------------------------------------------------------------------
# Each takes (op, x, r, c0, c1) and gives E for row r, columns [c0, c1) of op.shape, as referee.OPS do. `x` reads the
# state before the op (x.row, x.get) and the base model's weights (x.weight).

def _num(op) -> str:
    return op.attrs.get("num", "ref")


def op_ds_round_fp8(op, x, r, c0, c1):
    """num.fp8(x): deepseek.round_fp8, exact."""
    return E(round_fp8(x.row(op.ins[0], r), op.attrs.get("block", BLOCK))[c0:c1])


def op_ds_round_fp4(op, x, r, c0, c1):
    """num.fp4(x, block, e4m3_scale): deepseek.round_fp4, exact."""
    a = op.attrs
    return E(round_fp4(x.row(op.ins[0], r), a.get("block", BLOCK), a.get("e4m3", False))[c0:c1])


def _segments(c0, c1, width):
    """Columns [c0, c1) split where a group of `width` ends: (group, first, end) within it."""
    out, c = [], c0
    while c < c1:
        g, j = divmod(c, width)
        end = min(c1, (g + 1) * width)
        out.append((g, j, j + end - c))
        c = end
    return out


def op_ds_qlinear(op, x, r, c0, c1):
    """QLinear.matmul (deepseek._matmul): x · Wᵀ with W dequantised, x already rounded to FP8 under "ref". With
    groups > 1 (wo_a), group g of x's row meets only W's g-th block of rows (deepseek._matmul's einsum)."""
    a, xs = op.attrs, x.row(op.ins[0], r)
    groups = a.get("groups", 1)
    out_g, in_g = op.shape[1] // groups, op.ishapes[0][1] // groups
    parts = []
    for g, j0, j1 in _segments(c0, c1, out_g):
        w = qrows(x, a["weight"], g * out_g + j0, g * out_g + j1)
        parts.append(rf.contract(E(xs[g * in_g:(g + 1) * in_g][None]), w)[0])
    return cast(rf._cat(*parts), _num(op))


def op_ds_dense(op, x, r, c0, c1):
    """F.linear(x, W) for a matrix stored as is (bf16 or float32): the router's, the compressor's r = 1 projection,
    the indexer's wk and weights_proj, the head. attrs hi: computed in num.hi and left there (MoE.scores, the
    head); else in the activation dtype."""
    y = rf.contract(E(x.row(op.ins[0], r)[None]), x.weight(op.attrs["weight"], c0, c1))[0]
    return y if op.attrs.get("hi") else cast(y, _num(op))


def op_ds_scale(op, x, r, c0, c1):
    """x · c for a float32 constant c (the indexer's head weights times D^-1/2 H^-1/2), in the activation dtype."""
    return cast(rf.scale(E(x.get(op.ins[0], r, r + 1, c0, c1)[0]), rf.f32(op.attrs["c"])), _num(op))


def op_ds_rmsnorm(op, x, r, c0, c1):
    """RMSNorm.forward: weight · (h · rsqrt(mean(h²) + eps)) in num.hi, cast back. attrs group: normalised per that
    many columns (default: the whole row)."""
    g = op.attrs.get("group", op.shape[1])
    xs = x.row(op.ins[0], r).reshape(-1, g)
    w = x.weight(op.attrs["weight"], 0, 1)[0]
    y = rf.mul(E(w[None, :]), rf.mul(E(xs), rf.bcast(rf._norm(x, xs, g, op.attrs["eps"]))))
    return cast(E(y.v.reshape(-1)[c0:c1], y.e.reshape(-1)[c0:c1]), _num(op))


# --- rotary positions -------------------------------------------------------------------------------------------------

POWF_ULPS = 1.5  # powf's error in the rotary table's frequencies: faithful (under 1 ulp) in glibc, Sleef and CUDA


def rope_table(args, positions) -> tuple[E, E]:
    """cos and sin [len(positions), dim/2] of deepseek._rope_table(seqlen, *args) at those positions, with bounds
    covering how float32 computes them there: base ** t (powf, POWF_ULPS of error), then 1 / it, YaRN's blend and
    positions times frequencies each correctly rounded, then cos and sin (an ulp of 1)."""
    import math

    dim, original_seq_len, base, factor, beta_fast, beta_slow = args
    t = (np.arange(0, dim, 2, dtype=F32) / F32(dim)).astype(np.float64)
    p = rf.dexp(t * rf.dlog(np.float64(base)))  # base ** t, near enough: the bound covers powf's own error
    p32 = F32(p).astype(np.float64)
    freqs = rf.div(E(1.0), E(p32, POWF_ULPS * np.spacing(F32(p)).astype(np.float64) + np.abs(p - p32)))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = rf._r((np.arange(dim // 2, dtype=np.float64) - low) / float(F32(max(high - low, 1e-3))), 0.0)
        smooth = rf.sub(E(1.0), E(np.clip(ramp.v, 0, 1), ramp.e))
        fq = float(F32(factor))
        slow = rf._r(freqs.v / fq, freqs.e / fq)
        freqs = rf.add(rf.mul(slow, rf.sub(E(1.0), smooth)), rf.mul(freqs, smooth))
    pos = np.asarray(positions, dtype=np.float64)[:, None]
    ang = rf._r(pos * freqs.v[None, :], pos * freqs.e[None, :])
    s, c = rf.dsincos(ang.v)
    err = ang.e + 2.0 ** -23  # |d cos| ≤ |d angle|, plus the sincos implementation's ulp
    return E(c, err), E(s, err)


def op_ds_rope(op, x, r, c0, c1):
    """deepseek._rope: the last rd dims of each head of `hd` taken in adjacent pairs (x0, x1) and rotated by the
    position's angle, (x0 cos - x1 sin, x0 sin + x1 cos) in num.hi (sin negated with inverse), cast back; the other
    dims copied. attrs: rope (Config.rope_args' tuple), hd, stride (positions per row: a compressed token sits at its
    group's first), inverse."""
    a = op.attrs
    hd, rope = a["hd"], a["rope"]
    rd = rope[0]
    row = x.row(op.ins[0], r).reshape(-1, hd)
    cos, sin = rope_table(rope, [r * a.get("stride", 1)])
    if a.get("inverse"):
        sin = -sin
    x0, x1 = E(row[:, hd - rd::2]), E(row[:, hd - rd + 1::2])
    t0 = rf.sub(rf.mul(x0, cos), rf.mul(x1, sin))
    t1 = rf.add(rf.mul(x0, sin), rf.mul(x1, cos))
    n = row.shape[0]
    tail = cast(E(np.stack([t0.v, t1.v], -1).reshape(n, rd), np.stack([t0.e, t1.e], -1).reshape(n, rd)), _num(op))
    v = np.concatenate([row[:, :hd - rd], tail.v], 1).reshape(-1)
    e = np.concatenate([np.zeros((n, hd - rd)), tail.e], 1).reshape(-1)
    return E(v[c0:c1], e[c0:c1])


# --- hyper-connections ------------------------------------------------------------------------------------------------

def _sum(a: E, axis: int) -> E:
    """A sum along one axis of a small array, in any order, kept broadcastable."""
    k = a.v.shape[axis]
    v = np.apply_along_axis(math_fsum, axis, a.v)
    s_abs = np.apply_along_axis(math_fsum, axis, np.abs(a.v))
    ae = np.apply_along_axis(math_fsum, axis, a.e)
    v, ae, s_abs = (np.expand_dims(t, axis) for t in (v, ae, s_abs))
    return E(v, ae + rf.gamma(k) * (s_abs + ae))


def math_fsum(t) -> float:
    import math
    return math.fsum(t.tolist())


def op_ds_hc_mixes(op, x, r, c0, c1):
    """Block.hc_mixes' first half: F.linear(h, fn) · rsqrt(mean(h²) + eps), h the flattened stream in num.hi. The
    result stays in num.hi."""
    h = x.row(op.ins[0], r)
    y = rf.contract(E(h[None]), x.weight(op.attrs["fn"], c0, c1))[0]
    return rf.mul(y, rf.bcast(rf._norm(x, h[None], h.size, op.attrs["eps"]))[0])


def hc_split(m: np.ndarray, sc: np.ndarray, base: np.ndarray, hc: int, iters: int, eps: float) -> E:
    """deepseek._sinkhorn_split of one token's mixes, as [pre (hc) | post (hc) | comb (hc·hc, row-major)]."""
    m, e32 = E(m), rf.f32(eps)

    def lin(lo, hi, k):
        return rf.add(rf.scale(m[lo:hi], rf.f32(sc[k])), E(base[lo:hi]))
    pre = rf.add(sigmoid(lin(0, hc, 0)), E(e32))
    post = rf.scale(sigmoid(lin(hc, 2 * hc, 1)), 2.0)
    z = lin(2 * hc, (2 + hc) * hc, 2)
    z = E(z.v.reshape(hc, hc), z.e.reshape(hc, hc))
    top = E(z.v.max(-1, keepdims=True), z.e.max(-1, keepdims=True))
    ex = rf.exp(rf.sub(z, top))
    comb = sinkhorn(rf.add(rf.div(ex, _sum(ex, 1)), E(e32)), iters, e32)
    return rf._cat(pre, post, E(comb.v.reshape(-1), comb.e.reshape(-1)))


def sinkhorn(k: E, iters: int, eps: float) -> E:
    """The kernel's normalisations of a positive matrix k, columns first, then rows and columns iters - 1 times, each
    a / (sum + eps), with a bound that doesn't double at every step as elementwise bounds would.

    Every step only rescales rows or columns, so the float iterate is the exact one with each element times
    exp(ρ_i + κ_j + π_ij): a row factor ρ, a column factor κ and an elementwise part π, which only the divisions'
    own rounding adds to (an ulp a step). A row step divides by the row's float sum: that is, by e^ρ_i times a
    weighted mean of e^(κ_k + π_ik) and a rounding of the sum, so ρ_i is replaced by (nearly) minus that mean: ρ's
    old value cancels, and the bound grows by a few roundings a step rather than doubling. (With eps the
    cancellation is (1 - λ) short of whole, λ = s/(s + eps) near 1: carried exactly as (1 - λ)ρ - λ(σ + τ).)
    The intervals of ρ, κ and the bound on |π| are propagated step by step against the exact iterate, computed in
    float64 alongside."""
    a = k.v.astype(np.float64).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(k.e < a, np.log1p(-k.e / a), -np.inf)
        p = float(np.max(np.maximum(np.log1p(k.e / a), -lo)))
    n = a.shape[0]
    u_div = -np.log1p(-rf.U)  # a correctly rounded division or addition, in log terms
    tau = -np.log1p(-rf.gamma(n))  # a float sum of n positive terms
    rho = np.zeros((n, 2))  # [lo, hi] per row
    kap = np.zeros((n, 2))  # per column
    for step in ["col"] + ["row", "col"] * (iters - 1):
        mat, own, other = (a, rho, kap) if step == "row" else (a.T, kap, rho)
        s = np.array([math_fsum(r) for r in mat])
        w = mat / s[:, None]
        sig_lo = np.log(w @ np.exp(other[:, 0] - p))
        sig_hi = np.log(w @ np.exp(other[:, 1] + p))
        d_lo = np.minimum(0.0, own[:, 0] + sig_lo - tau)
        d_hi = np.maximum(0.0, own[:, 1] + sig_hi + tau)
        lam = np.stack([s * np.exp(d_lo) / (s * np.exp(d_lo) + eps), s * np.exp(d_hi) / (s * np.exp(d_hi) + eps)], 1)
        q = np.stack([sig_lo - tau, sig_hi + tau], 1)
        corners = np.stack([(1 - lam[:, i]) * own[:, j] - lam[:, i] * q[:, l]
                            for i in (0, 1) for j in (0, 1) for l in (0, 1)], 1)
        new = np.stack([corners.min(1) - u_div, corners.max(1) + u_div], 1)  # and the rounding of sum + eps
        if step == "row":
            rho, a = new, a / (s + eps)[:, None]
        else:
            kap, a = new, a / (s + eps)[None, :]
        p += u_div  # the division's own rounding, elementwise
    hi = rho[:, 1][:, None] + kap[:, 1][None, :] + p
    lo = rho[:, 0][:, None] + kap[:, 0][None, :] - p
    e = a * np.maximum(np.expm1(hi), -np.expm1(lo)) + 1e-12 * a  # (and float64's own rounding of the exact iterate)
    return E(a, e)


def op_ds_hc_split(op, x, r, c0, c1):
    """deepseek._sinkhorn_split: pre, post and comb of one token, from its mixes, in num.hi."""
    a = op.attrs
    return hc_split(x.row(op.ins[0], r), vec(x, a["scale"]), vec(x, a["base"]), a["hc"], a["iters"], a["eps"])[c0:c1]


def op_ds_hc_pre(op, x, r, c0, c1):
    """deepseek.hc_pre: Σ_k pre_k · x_k over the stream's hc copies, in num.hi, cast back. ins (stream, mixes);
    attrs col: where pre starts in the mixes' row."""
    dim = op.shape[1]
    hc, col = op.ishapes[0][1] // dim, op.attrs.get("col", 0)
    pre = x.get(op.ins[1], r, r + 1, col, col + hc)[0]
    xs = np.stack([x.get(op.ins[0], r, r + 1, k * dim + c0, k * dim + c1)[0] for k in range(hc)], -1)  # [cols, hc]
    return cast(rf.esum(rf.mul(E(pre[None, :]), E(xs))), _num(op))


def op_ds_hc_post(op, x, r, c0, c1):
    """deepseek.hc_post: out_k = post_k · y + Σ_j comb_jk · residual_j, the sum over j from 0 in order, in num.hi,
    cast back. ins (y, residual stream, split row); attrs post, comb: where they start in the split row."""
    a = op.attrs
    dim = op.ishapes[0][1]
    hc = op.shape[1] // dim
    post = x.get(op.ins[2], r, r + 1, a["post"], a["post"] + hc)[0]
    comb = x.get(op.ins[2], r, r + 1, a["comb"], a["comb"] + hc * hc)[0].reshape(hc, hc)
    parts = []
    for k, j0, j1 in _segments(c0, c1, dim):
        y = E(x.get(op.ins[0], r, r + 1, j0, j1)[0])
        res = [E(x.get(op.ins[1], r, r + 1, j * dim + j0, j * dim + j1)[0]) for j in range(hc)]
        mixed = rf.mul(E(comb[0, k]), res[0])
        for j in range(1, hc):
            mixed = rf.add(mixed, rf.mul(E(comb[j, k]), res[j]))
        parts.append(cast(rf.add(rf.mul(E(post[k]), y), mixed), _num(op)))
    return rf._cat(*parts)


# --- the compressor and the indexer -----------------------------------------------------------------------------------

def op_ds_compress(op, x, r, c0, c1):
    """Compressor.forward's pooling (ratio > 1), before its norm: for compressed token r, the softmax over its ratio
    tokens of h·wgate weighting h·wkv, per column, in num.hi, the sum cast back. attrs wkv, wgate, ratio."""
    a = op.attrs
    k = a["ratio"]
    h = x.get(op.ins[0], r * k, (r + 1) * k)  # [ratio, dim]
    kv = rf.contract(E(h), x.weight(a["wkv"], c0, c1))  # [ratio, cols]
    sc = rf.contract(E(h), x.weight(a["wgate"], c0, c1))
    top = E(sc.v.max(0, keepdims=True), sc.e.max(0, keepdims=True))
    ex = rf.exp(rf.sub(sc, top))
    p = rf.div(ex, _sum(ex, 0))
    tot = _sum(rf.mul(kv, p), 0)
    return cast(E(tot.v[0], tot.e[0]), _num(op))


def op_ds_index_scores(op, x, r, c0, c1):
    """Indexer.scores for query r: Σ_h relu(q_h · k_t) · w_h for each compressed token t, -inf for those it can't
    see yet ((r + 1) // ratio of them). The einsum, the product and the sum each round to the activation dtype.
    ins (q [T, H·D], index keys [n, D], w [T, H]); attrs heads, ratio."""
    a, num = op.attrs, _num(op)
    H = a["heads"]
    D = op.ishapes[0][1] // H
    visible = (r + 1) // a["ratio"]
    q = x.row(op.ins[0], r).reshape(H, D)
    w = x.row(op.ins[2], r)
    v, e = np.full(c1 - c0, -np.inf), np.zeros(c1 - c0)
    hi = min(c1, visible)
    if hi > c0:
        k = x.get(op.ins[1], c0, hi)  # [cols, D]
        sc = cast(rf.contract(E(q), k), num)  # [H, cols]
        sc = E(np.maximum(sc.v, 0), np.where(sc.v + sc.e > 0, sc.e, 0.0))  # relu: within e of 0 stays within e
        prod = cast(rf.mul(sc, E(w[:, None])), num)
        tot = cast(_sum(prod, 0), num)
        v[:hi - c0], e[:hi - c0] = tot.v[0], tot.e[0]
    return E(v, e)


# --- the MoE's experts ------------------------------------------------------------------------------------------------

def op_ds_swiglu(op, x, r, c0, c1):
    """Expert.forward's middle: g and u in num.hi, u clamped to ±limit and g below +limit (limit > 0), silu(g) · u,
    times the token's gate if given, cast back. ins (g, u[, gate [n, 1]])."""
    lim = op.attrs.get("limit", 0)
    g, u = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins[:2])
    if lim > 0:
        g, u = np.minimum(g, lim), np.clip(u, -lim, lim)
    h = rf.mul(rf._silu(g), E(u))
    if len(op.ins) > 2:
        h = rf.mul(E(x.get(op.ins[2], r, r + 1, 0, 1)[0]), h)
    return cast(h, _num(op))


# --- sparse attention -------------------------------------------------------------------------------------------------

KERNEL_BLOCK = 64  # deepseek.KERNEL_BLOCK: keys the reference's kernel takes at a time


def _emax(a: E) -> E:
    """The max of approximate values: within the largest of their bounds of the max of the exact ones."""
    if a.v.size == 0:
        return E(np.array(-np.inf), np.array(0.0))
    return E(a.v.max(), a.e.max())


def attend(qh: np.ndarray, win: np.ndarray, sel: np.ndarray, n_window: int, sink: float, scale_: float, num: str) -> E:
    """deepseek._attend for one query and one head: its window keys `win` [w, d] (oldest first) and picked keys `sel`
    [k, d] (the valid ones, in pick order), with the kernel's arithmetic. Scores q·k·scale in float32; each key's
    probability exp(s - m_key), m_key the running max over the kernel's 64-key blocks up to the key's own (its list
    position: window keys from 0, picks from n_window); the denominator from those times exp(m_key - m) plus
    exp(sink - m); the probabilities rounded to the activation dtype before they weight the values; o / denom cast."""
    c = rf.f32(scale_)
    keys = np.concatenate([win, sel]) if len(sel) else win
    slots = np.concatenate([np.arange(len(win)), n_window + np.arange(len(sel))]).astype(np.int64)
    if len(keys) == 0:  # a query that sees nothing: m = -1e30, o = 0
        return E(np.zeros(qh.shape[0]))
    s = rf.scale(rf.contract(E(qh[None]), keys)[0], c)
    blocks = slots // KERNEL_BLOCK
    best = [_emax(s[blocks == b]) for b in range(int(blocks.max()) + 1)]
    run, m_blk = E(np.array(-np.inf), np.array(0.0)), []
    for b in best:  # cummax, floored at -1e30 as the kernel's running max
        run = E(np.maximum(run.v, b.v), np.maximum(run.e, b.e)) if np.isfinite(b.v) else run
        m_blk.append(E(np.maximum(run.v, -1e30), run.e))
    m = m_blk[-1]
    m_key = E(np.array([m_blk[b].v for b in blocks]), np.array([m_blk[b].e for b in blocks]))
    p = rf.exp(rf.sub(s, m_key))
    rescale = rf.exp(rf.sub(m_key, E(np.full(len(keys), m.v), np.full(len(keys), m.e))))
    denom = rf.add(rf.esum(rf.mul(p, rescale)[None])[0:1], rf.exp(rf.sub(E(np.array([sink])), E(np.array([m.v]), np.array([m.e])))))
    pr = rf.mul(cast(p, num), rescale)
    o = rf.contract(E(pr.v[:len(win)][None], pr.e[:len(win)][None]), win.T)[0]
    if len(sel):
        o = rf.add(o, rf.contract(E(pr.v[len(win):][None], pr.e[len(win):][None]), sel.T)[0])
    return cast(rf.div(o, E(np.full(o.v.shape, denom.v[0]), np.full(o.v.shape, denom.e[0]))), num)


def op_ds_attn(op, x, r, c0, c1):
    """deepseek.sparse_attention for query r, the heads its columns cover: its window (the last `window` tokens up
    to it) and its picks of the compressed keys (-1: none), with one shared kv head and a sink per head. ins (q
    [T, heads·hd], kv [T, hd], and with compression comp [n, hd] and picks [T, k]); attrs heads, window, scale, sink,
    n_window (min(T, window): where the picks' places in the kernel's list start)."""
    a, num = op.attrs, _num(op)
    hd = op.ishapes[1][1]
    win = x.get(op.ins[1], max(0, r - a["window"] + 1), r + 1)
    sel = np.zeros((0, hd))
    if len(op.ins) > 2:
        picks = x.row(op.ins[3], r)
        valid = [int(p) for p in picks if p >= 0]
        if len(valid):
            sel = np.concatenate([x.get(op.ins[2], i, i + 1) for i in valid])
    sinks = vec(x, a["sink"])
    parts = []
    for h, j0, j1 in _segments(c0, c1, hd):
        qh = x.get(op.ins[0], r, r + 1, h * hd, (h + 1) * hd)[0]
        parts.append(attend(qh, win, sel, a["n_window"], float(sinks[h]), a["scale"], num)[j0:j1])
    return rf._cat(*parts)


# --- the router ---------------------------------------------------------------------------------------------------------

def op_ds_moe_scores(op, x, r, c0, c1):
    """MoE.scores after its F.linear: s / gate_temp, then sqrt(softplus(s)) (softplus is s itself beyond 20), or
    sigmoid; in num.hi. attrs temp, func ("sqrtsoftplus" or "sigmoid")."""
    a = op.attrs
    s = E(x.get(op.ins[0], r, r + 1, c0, c1)[0])
    t = rf.f32(a.get("temp", 1.0))
    s = rf._r(s.v / t, s.e / t) if t != 1.0 else s
    if a.get("func", "sqrtsoftplus") == "sigmoid":
        return sigmoid(s)
    sp = rf.log(rf.add(E(1.0), rf.exp(s)))
    big = s.v > 20
    return rf.sqrt(E(np.where(big, s.v, sp.v), np.where(big, s.e, sp.e)))


def op_ds_bias(op, x, r, c0, c1):
    """scores + bias (the gate's, which steers which experts are chosen): a float32 add."""
    return rf.add(E(x.get(op.ins[0], r, r + 1, c0, c1)[0]), E(vec(x, op.attrs["bias"])[c0:c1]))


def op_ds_gates(op, x, r, c0, c1):
    """MoE.route's gates: the unbiased scores of the chosen experts, normalised by their sum + 1e-20 when attrs
    norm, times route_scale; in num.hi. ins (scores [T, E], chosen [T, k])."""
    a = op.attrs
    chosen = x.row(op.ins[1], r).astype(np.int64)
    g = E(np.array([x.get(op.ins[0], r, r + 1, int(c), int(c) + 1)[0, 0] for c in chosen]))
    if a.get("norm", True) and len(chosen) > 1:
        g = rf.div(g, rf.add(rf.esum(g[None])[0:1], E(np.array([rf.f32(1e-20)]))))
    return rf.scale(g, rf.f32(a.get("route_scale", 1.0)))[c0:c1]

# --- the embedding and engram -----------------------------------------------------------------------------------------

def _token(x, name, t) -> int:
    v = x.get(name, t, t + 1, 0, 1)[0, 0]
    if not (np.isfinite(v) and v == int(v) and v >= 0):
        raise rf.Missing(f"token {t} isn't a token id")
    return int(v)


def op_ds_embed(op, x, r, c0, c1):
    """Model.start: the embedding's row for token r (stored bf16, exact in the activations). ins (tokens [T, 1])."""
    tok = _token(x, op.ins[0], r)
    return E(x.weight(op.attrs.get("weight", "embed.weight"), tok, tok + 1)[0, c0:c1])


def engram_rows(ids: list[int], layer: int, a: dict) -> list[int]:
    """deepseek.engram_hashes for the last position of `ids` (the normalised tokens up to it, oldest first): its
    n-grams' table rows, n = 2..N, each head's, as Python integers (exact: no int64 to overflow). attrs: primes
    [size - 2][head] and multipliers [look-back] of this engram layer (EngramLayout), offsets (each (size, head)'s
    first row), pad (the normalised pad id), n (N)."""
    n, mult = a["n"], a["multipliers"]
    back = [ids[-1 - k] if k < len(ids) else a["pad"] for k in range(n)]
    rolling, out, at = back[0] * mult[0], [], 0
    for i in range(1, n):
        rolling ^= back[i] * mult[i]
        for p in a["primes"][i - 1]:
            out.append(rolling % p + a["offsets"][at])
            at += 1
    return out


def op_ds_engram_rows(op, x, r, c0, c1):
    """Engram.rows for token r: its n-grams' rows of the layer's FP8 table (ue8m0 scale per 32), dequantised exactly,
    concatenated [cols · head_dim]. The rows follow from the text alone: ins (tokens [T, 1]); the token map (token id
    to normalised id) is read as a table (attrs token_map), the rows as attrs table's .weight and .scale."""
    a = op.attrs
    ids = [int(_lookup(x, a["token_map"], _token(x, op.ins[0], t))) for t in range(max(0, r - a["n"] + 1), r + 1)]
    rows = engram_rows(ids, 0, a)
    hd = op.shape[1] // len(rows)
    vals = np.concatenate([qrows(x, a["table"], row, row + 1)[0] for row in rows[c0 // hd:(c1 - 1) // hd + 1]])
    lo = (c0 // hd) * hd
    return E(vals[c0 - lo:c1 - lo])


def _lookup(x, table: str, i: int) -> float:
    """Entry i of a 1-D integer table (the token map), exactly."""
    dtype, shape = x.outside.source.shape(table)
    if len(shape) != 1 or not 0 <= i < shape[0]:
        raise rf.Missing(f"entry {i} of {table}")
    return float(x.weight(table, 0, 1)[0, i])


def op_ds_engram_gate(op, x, r, c0, c1):
    """Engram.forward after its wkv: per stream copy k, the gate sigmoid(signed sqrt(max(|dot|, 1e-6))) of
    dot = Σ_c h·(q_w·k_w)·key · rsqrt(mean(h²) + eps) · rsqrt(mean(key²) + eps) · dim^-1/2, then h + gate · value,
    in num.hi, cast back. ins (stream [T, hc·dim], wkv's output [T, (hc + 1)·dim]: hc keys, then the value);
    attrs q_weight, k_weight ([hc, dim]), eps."""
    a, num = op.attrs, _num(op)
    dim = op.ishapes[1][1] - op.ishapes[0][1]
    hc = op.ishapes[0][1] // dim
    kv = x.row(op.ins[1], r)
    parts = []
    for k, j0, j1 in _segments(c0, c1, dim):
        h = x.get(op.ins[0], r, r + 1, k * dim, (k + 1) * dim)[0]
        key = kv[k * dim:(k + 1) * dim]
        qw, kw = x.weight(a["q_weight"], k, k + 1)[0], x.weight(a["k_weight"], k, k + 1)[0]
        qk = rf.mul(E(qw), E(kw))
        dot = rf.esum(rf.mul(rf.mul(E(h), qk), E(key))[None])[0:1]
        rstd = rf.mul(rf._norm(x, h[None], dim, a["eps"]), rf._norm(x, key[None], dim, a["eps"]))
        dot = rf.scale(rf.mul(dot, rstd), rf.f32(dim ** -0.5))
        mag = rf.sqrt(E(np.maximum(np.abs(dot.v), 1e-6), dot.e))
        gate = sigmoid(E(np.copysign(mag.v, dot.v), np.where(np.abs(dot.v) <= dot.e, mag.v + mag.e, mag.e)))
        val = E(kv[hc * dim + j0:hc * dim + j1])
        g = E(np.full(j1 - j0, gate.v[0]), np.full(j1 - j0, gate.e[0]))
        parts.append(cast(rf.add(E(h[j0:j1]), rf.mul(g, val)), num))
    return rf._cat(*parts)


def op_ds_moe_sum(op, x, r, c0, c1):
    """MoE.forward's sum and combine: token r's routed outputs added in float32 in ascending expert order (the order
    its experts run), then the shared expert's, cast back. ins (routed [T, k·dim]: slot j the output of the expert in
    chosen[r, j], chosen [T, k], shared [T, dim])."""
    dim = op.shape[1]
    chosen = x.row(op.ins[1], r).astype(np.int64)
    order = np.argsort(chosen, kind="stable")
    y = None
    for j in order:
        o = E(x.get(op.ins[0], r, r + 1, int(j) * dim + c0, int(j) * dim + c1)[0])
        y = o if y is None else rf.add(y, o)
    return cast(rf.add(y, E(x.get(op.ins[2], r, r + 1, c0, c1)[0])), _num(op))

# --- the backward, for training units (float32: "f32") -----------------------------------------------------------------
# Training units compute float32 throughout (deepseek_top.TRAIN), and their backward is written out op by op, each the
# formula its docstring gives in that order: the tracer computes exactly these, and the referee bounds them. Ops that
# add into an existing gradient take it as an optional last input (the accumulator), added last.

def _acc(op, x, r, c0, c1, y: E, at: int) -> E:
    return rf.add(E(x.get(op.ins[at], r, r + 1, c0, c1)[0]), y) if len(op.ins) > at else y


def _contract_rows(dy: np.ndarray, rows_of, n_rows: int, chunk: int = 4096) -> E:
    """Σ_o dy[o] · W[o, cols] over every row o of W, read `chunk` rows at a time (rows_of(o0, o1) -> [o1-o0, cols]):
    exact partial sums (fsum), and the bound of one K-term dot product of float32s in any order."""
    parts, absparts = [], []
    for o0 in range(0, n_rows, chunk):
        o1 = min(n_rows, o0 + chunk)
        p = dy[o0:o1, None] * rows_of(o0, o1)
        parts.append(rf.fsum_rows(p.T))
        absparts.append(rf.fsum_rows(np.abs(p).T))
    v = rf.fsum_rows(np.stack(parts, -1))
    s_abs = rf.fsum_rows(np.stack(absparts, -1))
    return E(v, (rf.gamma(n_rows) + 2.0 ** -50) * s_abs)  # (and the float64 partial sums' own rounding)


def op_ds_qlinear_bwd(op, x, r, c0, c1):
    """QLinear's backward to its input: dx = dy · W (W dequantised), group by group for wo_a, then + acc. ins (dy[,
    acc]); attrs weight, groups."""
    a = op.attrs
    groups = a.get("groups", 1)
    out_g, in_g = op.ishapes[0][1] // groups, op.shape[1] // groups
    dy = x.row(op.ins[0], r)
    parts = []
    for g, j0, j1 in _segments(c0, c1, in_g):
        def rows(o0, o1, g=g, j0=j0, j1=j1):
            return qrows(x, a["weight"], g * out_g + o0, g * out_g + o1)[:, j0:j1]
        parts.append(_contract_rows(dy[g * out_g:(g + 1) * out_g], rows, out_g))
    return _acc(op, x, r, c0, c1, rf._cat(*parts), 1)


def op_ds_dense_bwd(op, x, r, c0, c1):
    """A plain matrix's backward to its input: dx = dy · W, then + acc. ins (dy[, acc]); attrs weight."""
    dy = x.row(op.ins[0], r)
    y = _contract_rows(dy, lambda o0, o1: x.weight(op.attrs["weight"], o0, o1)[:, c0:c1], op.ishapes[0][1])
    return _acc(op, x, r, c0, c1, y, 1)


def op_ds_hc_pre_dx(op, x, r, c0, c1):
    """hc_pre's backward to the stream: dx_k = pre_k · dout, then + acc. ins (dout [T, dim], mix[, acc]); attrs col."""
    dim, col = op.ishapes[0][1], op.attrs.get("col", 0)
    parts = []
    for k, j0, j1 in _segments(c0, c1, dim):
        pre = x.get(op.ins[1], r, r + 1, col + k, col + k + 1)[0, 0]
        parts.append(rf.mul(E(pre), E(x.get(op.ins[0], r, r + 1, j0, j1)[0])))
    return _acc(op, x, r, c0, c1, rf._cat(*parts), 2)


def op_ds_hc_pre_dpre(op, x, r, c0, c1):
    """hc_pre's backward to pre: dpre_k = dout · x_k (over dim). ins (dout [T, dim], stream [T, hc·dim])."""
    dim = op.ishapes[0][1]
    dout = x.row(op.ins[0], r)
    xs = np.stack([x.get(op.ins[1], r, r + 1, k * dim, (k + 1) * dim)[0] for k in range(c0, c1)])
    return rf.contract(E(dout[None]), xs)[0]


def op_ds_hc_post_dy(op, x, r, c0, c1):
    """hc_post's backward to y: dy = Σ_k post_k · dout_k. ins (dout [T, hc·dim], split); attrs post."""
    dim = op.shape[1]
    hc = op.ishapes[0][1] // dim
    post = x.get(op.ins[1], r, r + 1, op.attrs["post"], op.attrs["post"] + hc)[0]
    d = np.stack([x.get(op.ins[0], r, r + 1, k * dim + c0, k * dim + c1)[0] for k in range(hc)], -1)
    return rf.esum(rf.mul(E(post[None, :]), E(d)))


def op_ds_hc_post_dres(op, x, r, c0, c1):
    """hc_post's backward to the residual stream: dres_j = Σ_k comb_jk · dout_k, then + acc. ins (dout, split[,
    acc]); attrs comb."""
    hc = op.attrs["hc"]
    dim = op.shape[1] // hc
    comb = x.get(op.ins[1], r, r + 1, op.attrs["comb"], op.attrs["comb"] + hc * hc)[0].reshape(hc, hc)
    parts = []
    for j, j0, j1 in _segments(c0, c1, dim):
        d = np.stack([x.get(op.ins[0], r, r + 1, k * dim + j0, k * dim + j1)[0] for k in range(hc)], -1)
        parts.append(rf.esum(rf.mul(E(comb[j][None, :]), E(d))))
    return _acc(op, x, r, c0, c1, rf._cat(*parts), 2)


def op_ds_hc_post_dsplit(op, x, r, c0, c1):
    """hc_post's backward to its split row: zero for pre, dpost_k = dout_k · y, dcomb_jk = dout_k · res_j (dots over
    dim). ins (dout [T, hc·dim], y [T, dim], residual [T, hc·dim]); attrs hc."""
    hc = op.attrs["hc"]
    dim = op.ishapes[1][1]
    dout = x.row(op.ins[0], r).reshape(hc, dim)
    y, res = x.row(op.ins[1], r), x.row(op.ins[2], r).reshape(hc, dim)
    dpost = rf.contract(E(dout), y[None])[:, 0]
    dcomb = rf.contract(E(res), dout)  # [j, k]
    out = rf._cat(E(np.zeros(hc)), E(dpost.v, dpost.e), E(dcomb.v.reshape(-1), dcomb.e.reshape(-1)))
    return out[c0:c1]


def op_ds_hc_mixes_bwd(op, x, r, c0, c1):
    """hc_mixes' backward to the stream h: with lin = h·fnᵀ and rs = rsqrt(mean(h²) + eps) recomputed,
    dh = (dmix · rs)·fn + h · ((Σ_m dmix_m · lin_m) · rs³ · (-1/n)), then + acc. ins (dmix [T, 24], stream[, acc]);
    attrs fn, eps."""
    a = op.attrs
    h = x.row(op.ins[1], r)
    n = h.size
    dmix = x.row(op.ins[0], r)
    fn = x.weight(a["fn"], 0, x.weight_rows(a["fn"]))
    lin = rf.contract(E(h[None]), fn)[0]
    rs = rf._norm(x, h[None], n, a["eps"])
    g = rf.mul(E(dmix), E(np.full(dmix.shape, rs.v[0]), np.full(dmix.shape, rs.e[0])))
    d1 = rf.contract(E(g.v[None], g.e[None]), fn[:, c0:c1].T)[0]
    q = rf.esum(rf.mul(E(dmix), lin)[None])
    coef = rf.scale(rf.mul(q, rf.mul(rf.mul(rs, rs), rs)), rf.f32(-1.0 / n))
    d2 = rf.mul(E(h[c0:c1]), E(np.full(c1 - c0, coef.v[0]), np.full(c1 - c0, coef.e[0])))
    return _acc(op, x, r, c0, c1, rf.add(d1, d2), 2)


def op_ds_swiglu_bwd(op, x, r, c0, c1):
    """ds_swiglu's backward (h = gate · silu(g') · u', g' = min(g, L), u' = clamp(u, ±L)), to g or u by attrs part:
    dg = ((dh · gate) · u') · silu'(g') where g ≤ L, silu'(z) = σ(z) · (1 + z · (1 - σ(z))); du = ((dh · gate) ·
    silu(g')) where |u| ≤ L. ins (dh, g, u, gate [n, 1]); attrs limit, part ("g" or "u")."""
    a = op.attrs
    lim = a.get("limit", 0)
    dh, g, u = (x.get(n, r, r + 1, c0, c1)[0] for n in op.ins[:3])
    gate = x.get(op.ins[3], r, r + 1, 0, 1)[0, 0]
    gc, uc = (np.minimum(g, lim), np.clip(u, -lim, lim)) if lim > 0 else (g, u)
    dg_ = rf.mul(E(dh), E(gate))
    if a["part"] == "u":
        y = rf.mul(dg_, rf._silu(gc))
        keep = (np.abs(u) <= lim) if lim > 0 else np.ones_like(u, dtype=bool)
    else:
        sig = sigmoid(E(gc))
        dsil = rf.mul(sig, rf.add(E(1.0), rf.mul(E(gc), rf.sub(E(1.0), sig))))
        y = rf.mul(rf.mul(dg_, E(uc)), dsil)
        keep = (g <= lim) if lim > 0 else np.ones_like(g, dtype=bool)
    return E(np.where(keep, y.v, 0.0), np.where(keep, y.e, 0.0))


def op_ds_swiglu_dgate(op, x, r, c0, c1):
    """ds_swiglu's backward to the token's gate: Σ_c dh_c · (silu(g'_c) · u'_c). ins (dh, g, u); attrs limit."""
    lim = op.attrs.get("limit", 0)
    dh, g, u = (x.row(n, r) for n in op.ins[:3])
    gc, uc = (np.minimum(g, lim), np.clip(u, -lim, lim)) if lim > 0 else (g, u)
    return rf.esum(rf.mul(E(dh), rf.mul(rf._silu(gc), E(uc)))[None])[0:1]


def op_ds_gates_bwd(op, x, r, c0, c1):
    """ds_gates' backward to the scores [T, E]: zero but for the chosen; with S = Σ_chosen s + 1e-20 (norm) and
    G = Σ_i dg_i · s_i, ds_j = route_scale · (dg_j / S - G / (S · S)); without norm, route_scale · dg_j.
    ins (scores, chosen, dgates [T, k]); attrs norm, route_scale."""
    a = op.attrs
    chosen = x.row(op.ins[1], r).astype(np.int64)
    dg = E(x.row(op.ins[2], r))
    s = E(np.array([x.get(op.ins[0], r, r + 1, int(c), int(c) + 1)[0, 0] for c in chosen]))
    rs = rf.f32(a.get("route_scale", 1.0))
    if a.get("norm", True) and len(chosen) > 1:
        S = rf.add(rf.esum(s[None])[0:1], E(np.array([rf.f32(1e-20)])))
        G = rf.esum(rf.mul(dg, s)[None])[0:1]
        full = lambda t: E(np.full(len(chosen), t.v[0]), np.full(len(chosen), t.e[0]))  # noqa: E731
        d = rf.sub(rf.div(dg, full(S)), full(rf.div(G, rf.mul(S, S))))
    else:
        d = dg
    d = rf.scale(d, rs)
    v, e = np.zeros(c1 - c0), np.zeros(c1 - c0)
    for j, c in enumerate(chosen):
        if c0 <= c < c1:
            v[c - c0], e[c - c0] = d.v[j], d.e[j]
    return E(v, e)


def op_ds_moe_scores_bwd(op, x, r, c0, c1):
    """MoE.scores' backward to its logits (sqrt(softplus(l / temp))): ds · (1 / (2 · score)) · σ(l / temp) / temp
    (σ: softplus' derivative; 1 beyond 20). ins (logits, dscores); attrs temp."""
    t = rf.f32(op.attrs.get("temp", 1.0))
    lg = x.get(op.ins[0], r, r + 1, c0, c1)[0]
    ds_ = E(x.get(op.ins[1], r, r + 1, c0, c1)[0])
    z = rf._r(lg / t, np.zeros_like(lg)) if t != 1.0 else E(lg)
    sp = rf.log(rf.add(E(1.0), rf.exp(z)))
    big = z.v > 20
    sp = E(np.where(big, z.v, sp.v), np.where(big, z.e, sp.e))
    score = rf.sqrt(sp)
    dsp = sigmoid(z)
    dsp = E(np.where(big, 1.0, dsp.v), np.where(big, 0.0, dsp.e))
    y = rf.mul(rf.div(ds_, rf.scale(score, 2.0)), dsp)
    return rf._r(y.v / t, y.e / t) if t != 1.0 else y


# --- attention's backward ---------------------------------------------------------------------------------------------
# In float32 the forward's rounding of the probabilities is the identity and its running maxima cancel, so the backward
# is a plain softmax's (with the sink in the denominator), over each query's key slots: its window, oldest first, then
# its picks (slots n_window ...). Probabilities are recomputed (ds_attn_probs), then dP = do · v, dS = P · (dP - δ),
# dq = scale · Σ dS · k, and each window key's gradient sums every query and head that saw it.

def _keys(op, x, r):
    """Query r's keys: (window rows [w, hd], picked rows [k', hd] (valid ones), their slots)."""
    a = op.attrs
    hd = op.ishapes[1][1]
    win = x.get(op.ins[1], max(0, r - a["window"] + 1), r + 1)
    sel, slots = np.zeros((0, hd)), list(range(len(win)))
    if len(op.ins) > 2 and op.ins[2]:
        picks = x.row(op.ins[3], r)
        valid = [(j, int(p)) for j, p in enumerate(picks) if p >= 0]
        if valid:
            sel = np.concatenate([x.get(op.ins[2], i, i + 1) for _, i in valid])
            slots += [a["n_window"] + j for j, _ in valid]
    return win, sel, slots


def probs(qh, keys, sink, scale_) -> E:
    """P_j = exp(s_j - m) / (Σ exp(s - m) + exp(sink - m)), s = (q · k) · scale, m = max s (one query, one head)."""
    if len(keys) == 0:
        return E(np.zeros(0))
    s = rf.scale(rf.contract(E(qh[None]), keys)[0], rf.f32(scale_))
    m = _emax(s)
    ex = rf.exp(rf.sub(s, E(np.full(len(keys), m.v), np.full(len(keys), m.e))))
    D = rf.add(rf.esum(ex[None])[0:1], rf.exp(rf.sub(E(np.array([sink])), E(np.array([m.v]), np.array([m.e])))))
    return rf.div(ex, E(np.full(len(keys), D.v[0]), np.full(len(keys), D.e[0])))


def _slots_row(op, values: dict, width: int) -> tuple[np.ndarray, np.ndarray]:
    v, e = np.zeros(width), np.zeros(width)
    for s, (vv, ee) in values.items():
        v[s], e[s] = vv, ee
    return v, e


def op_ds_attn_probs(op, x, r, c0, c1):
    """Query r's probabilities, [heads · slots]: slot j < n_window its j-th window key (oldest first), slot
    n_window + i its i-th pick; zero where there's no key. ins and attrs as ds_attn's, plus slots (n_window + k)."""
    a = op.attrs
    hd, S = op.ishapes[1][1], a["slots"]
    win, sel, slots = _keys(op, x, r)
    keys = np.concatenate([win, sel]) if len(sel) else win
    sinks = vec(x, a["sink"])
    v, e = np.zeros(op.shape[1]), np.zeros(op.shape[1])
    for h in range(c0 // S, (c1 - 1) // S + 1):
        p = probs(x.get(op.ins[0], r, r + 1, h * hd, (h + 1) * hd)[0], keys, float(sinks[h]), a["scale"])
        for j, s in enumerate(slots):
            v[h * S + s], e[h * S + s] = p.v[j], p.e[j]
    return E(v[c0:c1], e[c0:c1])


def op_ds_attn_dp(op, x, r, c0, c1):
    """dP_j = do · v_j for query r's key slots (the keys are the values), [heads · slots]. ins (do [T, heads·hd],
    kv, comp, picks); attrs window, n_window, slots."""
    a = op.attrs
    hd, S = op.ishapes[1][1], a["slots"]
    win, sel, slots = _keys(op, x, r)
    keys = np.concatenate([win, sel]) if len(sel) else win
    v, e = np.zeros(op.shape[1]), np.zeros(op.shape[1])
    for h in range(c0 // S, (c1 - 1) // S + 1):
        if len(keys):
            d = rf.contract(E(x.get(op.ins[0], r, r + 1, h * hd, (h + 1) * hd)), keys)[0]
            for j, s in enumerate(slots):
                v[h * S + s], e[h * S + s] = d.v[j], d.e[j]
    return E(v[c0:c1], e[c0:c1])


def op_ds_attn_ds(op, x, r, c0, c1):
    """dS = P · (dP - δ), δ = Σ_j P_j · dP_j per head. ins (P, dP) [T, heads · slots]; attrs slots."""
    S = op.attrs["slots"]
    out = []
    for h, j0, j1 in _segments(c0, c1, S):
        p, dp = (x.get(n, r, r + 1, h * S, (h + 1) * S)[0] for n in op.ins)
        delta = rf.esum(rf.mul(E(p), E(dp))[None])
        out.append(rf.mul(E(p[j0:j1]), rf.sub(E(dp[j0:j1]), E(np.full(j1 - j0, delta.v[0]), np.full(j1 - j0, delta.e[0])))))
    return rf._cat(*out)


def op_ds_attn_dq(op, x, r, c0, c1):
    """dq = scale · Σ_j dS_j · k_j, [heads · hd]. ins (dS, kv, comp, picks); attrs window, n_window, slots, scale."""
    a = op.attrs
    hd, S = op.ishapes[1][1], a["slots"]
    win, sel, slots = _keys(op, x, r)
    keys = np.concatenate([win, sel]) if len(sel) else win
    parts = []
    for h, j0, j1 in _segments(c0, c1, hd):
        ds_ = x.get(op.ins[0], r, r + 1, h * S, (h + 1) * S)[0][slots] if len(slots) else np.zeros(0)
        d = rf.contract(E(ds_[None]), keys[:, j0:j1].T)[0] if len(slots) else E(np.zeros(j1 - j0))
        parts.append(rf.scale(d, rf.f32(a["scale"])))
    return rf._cat(*parts)


def op_ds_attn_dkv(op, x, r, c0, c1):
    """Window key r's gradient (as key and as value), [hd]: Σ over the queries t in [r, r + window) and every head h
    of scale · dS[t, h, slot] · q[t, h] + P[t, h, slot] · do[t, h], slot = r - max(0, t - window + 1); then + acc.
    ins (dS, P, q, do[, acc]); attrs heads, window, scale, slots."""
    a = op.attrs
    H, W, S = a["heads"], a["window"], a["slots"]
    T = op.ishapes[0][0]
    hd = op.shape[1]
    c = rf.f32(a["scale"])
    coef, rows = [], []
    for t in range(r, min(T, r + W)):
        slot = r - max(0, t - W + 1)
        dsr, pr = x.row(op.ins[0], t), x.row(op.ins[1], t)
        for h in range(H):
            coef += [dsr[h * S + slot], pr[h * S + slot]]
            rows += [x.get(op.ins[2], t, t + 1, h * hd + c0, h * hd + c1)[0], x.get(op.ins[3], t, t + 1, h * hd + c0, h * hd + c1)[0]]
    if not rows:
        return _acc(op, x, r, c0, c1, E(np.zeros(c1 - c0)), 4)
    coef = np.array(coef)
    ce = E(np.where(np.arange(len(coef)) % 2 == 0, coef * c, coef),
           np.where(np.arange(len(coef)) % 2 == 0, rf.U * np.abs(coef * c), 0.0))  # scale · dS, rounded
    y = rf.contract(E(ce.v[None], ce.e[None]), np.array(rows).T)[0]
    return _acc(op, x, r, c0, c1, y, 4)

# --- choices: judged by whole rows (referee.CHOICES), exactly, against the committed scores ----------------------------
# A choice is a function of scores the state already holds, and those were judged in their own op (within their
# bounds). So a worker may break near-ties either way, as its own scores fall, but may choose nothing its scores don't
# allow: the rules are deepseek.choice_gap's and candidate_gap's, for one row.

def _ints(row) -> np.ndarray | None:
    row = np.asarray(row, dtype=np.float64)
    if not np.all(np.isfinite(row)) or not np.array_equal(row, np.round(row)):
        return None
    return row.astype(np.int64)


def gap_ok(scores: np.ndarray, chosen: np.ndarray, k: int) -> tuple[bool, str]:
    """deepseek.choice_gap ≤ 0 for one row: distinct options in range (-1 only as padding), as many as there are
    finite scores up to k, and no option left out scoring above one chosen."""
    n = len(scores)
    if np.any(chosen < -1) or np.any(chosen >= n):
        return False, "it names an option out of range"
    real = chosen[chosen >= 0]
    if len(set(real.tolist())) != len(real):
        return False, "it names an option twice"
    need = min(int(np.isfinite(scores).sum()), k)
    if len(real) < need:
        return False, f"it leaves a slot empty while {need} options have a finite score"
    mask = np.zeros(n, dtype=bool)
    mask[real] = True
    worst = scores[mask].min() if mask.any() else np.inf
    best = scores[~mask].max() if (~mask).any() else -np.inf
    if best > worst and best != -np.inf:
        return False, f"it leaves out an option scoring {float(best)!r}, above a chosen one's {float(worst)!r}"
    return True, ""


def choice_ds_experts(op, x, r, row):
    """MoE.route's choice: k distinct experts, a top-k of the biased scores (deepseek.check_experts, choice_gap)."""
    chosen = _ints(row)
    if chosen is None:
        return False, "the experts aren't whole numbers"
    if np.any(chosen < 0):
        return False, "every token goes to k experts: no -1"
    return gap_ok(x.row(op.ins[0], r), chosen, len(chosen))


def choice_ds_picks(op, x, r, row):
    """Indexer.forward's picks for query r: ascending positions it can see, -1 (none) only after them and only where
    fewer than k positions are open (deepseek.check_picks), and a top-k of its index scores, -inf where it can't see
    or outside its candidate blocks (choice_gap)."""
    picks = _ints(row)
    if picks is None:
        return False, "the picks aren't whole numbers"
    visible = (r + 1) // op.attrs["ratio"]
    valid = picks >= 0
    if np.any(picks[valid] >= visible):
        return False, f"it picks a position query {r} can't see yet"
    if np.any(valid[1:] & ~valid[:-1]) or np.any(np.diff(picks[valid]) <= 0):
        return False, "picks are ascending, with any -1 after them"
    return gap_ok(x.row(op.ins[0], r), picks, len(picks))


def choice_ds_candidates(op, x, r, row):
    """deepseek.candidate_blocks for query r, as a 0/1 mask over the compressed tokens: whole blocks, at most
    topk_blocks of them, none unreachable, the block of its newest visible position always, and the rest the best
    blocks by their best score (candidate_gap)."""
    a = op.attrs
    blk, top = a["block"], a["topk_blocks"]
    mask = np.asarray(row, dtype=np.float64)
    if not np.all((mask == 0) | (mask == 1)):
        return False, "a candidate mask is 0 or 1"
    score = x.row(op.ins[0], r)
    width = len(score)
    nb = -(-width // blk)
    pad = np.full(nb * blk, -np.inf)
    pad[:width] = score
    blocks = pad.reshape(nb, blk).max(1)
    pinned = ((r + 1) // a["ratio"] - 1) // blk
    if 0 <= pinned < nb:
        blocks[pinned] = np.inf
    m = np.zeros(nb * blk, dtype=bool)
    m[:width] = mask == 1
    kept = m.reshape(nb, blk).any(1)
    if not np.array_equal(np.repeat(kept, blk)[:width], mask == 1):
        return False, "the mask isn't whole blocks"
    if kept.sum() > top or np.any(kept & (blocks == -np.inf)):
        return False, f"it keeps more than {top} blocks, or one this query can't reach"
    need = min(int((blocks > -np.inf).sum()), top)
    if kept.sum() < need:
        return False, f"it keeps {int(kept.sum())} blocks where {need} are open"
    worst = blocks[kept].min() if kept.any() else np.inf
    best = blocks[~kept].max() if (~kept).any() else -np.inf
    if best > worst and best != -np.inf:
        return False, "it leaves out a block scoring above one it keeps (or the pinned one)"
    return True, ""

OPS = {name[3:]: fn for name, fn in globals().items() if name.startswith("op_ds_")}
EXACT = {"ds_round_fp8", "ds_round_fp4", "ds_embed", "ds_engram_rows"}
# Ops whose result is cast to the activation dtype: under "ref", every number of an honest tile is a bf16 number
# (or -inf, the indexer's mask). Rounding error of a bf16 step can't be caught by a bound, but a number between
# two bf16 numbers can be refused exactly.
CAST = {"ds_qlinear", "ds_scale", "ds_rmsnorm", "ds_rope", "ds_hc_pre", "ds_hc_post", "ds_compress", "ds_index_scores",
        "ds_swiglu", "ds_dense", "ds_attn", "ds_engram_gate", "ds_moe_sum"}


def bf16_only(op, said: np.ndarray):
    if _num(op) != "ref" or (op.kind == "ds_dense" and op.attrs.get("hi")):
        return None
    fin = np.isfinite(said)
    if not np.array_equal(round_bf16(said[fin]), said[fin]) or np.isnan(said).any() or (np.isinf(said) & (said > 0)).any():
        return "a number that isn't a bf16 number, as every output of this op is"
    return None


CHOICES = {name[7:]: fn for name, fn in globals().items() if name.startswith("choice_ds_")}
rf.OPS.update(OPS)
rf.EXACT.update(EXACT)
rf.CHOICES.update(CHOICES)
rf.FORMATS.update({k: bf16_only for k in CAST})
