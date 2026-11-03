"""Evolution strategies on the chain, a prototype: the blockchain becomes the AI.

Each round of an `es` job is one step of evolution strategies (Salimans et al., 2017; at the scale of language models,
Qiu et al., "Evolution Strategies at Scale", arXiv 2509.24372). Workers run the model with a small random nudge to some
of its weights, each nudge named by a seed and tried both ways (+σε and −σε, an antithetic pair), answer the round's
problems greedily, and the answers are scored; checkers re-check a sample. The block then holds only seeds and scores,
and every node computes the model's update from them with the arithmetic below. So the model is a pure function of the
chain: a fresh node rebuilds it from genesis and the blocks' records alone (`rebuild`, `rebuild_chain`), with no weight
files to trust and no GPU. What the node keeps is not the weights but how far training has moved them from the pinned
base model (`delta`), which is zero at the start and needs nothing but seeds and scores to update.

Everything below is pinned so that every machine gets the same bits, and is versioned (`RULE`) so a change is a rule
upgrade, never a silent fork:

* **Noise** (`noise`). ε for a seed is a stream of standard normals, one per perturbed number, in the perturbed
  tensors' order (sorted by name, each flattened row-major). Its bits come from Philox4x64-10 (numpy's
  `Philox(key=seed).random_raw()`: a counter-based generator whose raw stream numpy keeps stable across versions, and
  which `test_es` pins by a known answer), so any stretch of it can be made without the rest. Each two raw words make
  two normals by Box-Muller in float64 (u1 = ((w0 >> 11) + 1)·2⁻⁵³ in (0, 1], u2 = (w1 >> 11)·2⁻⁵³ in [0, 1);
  √(−2 ln u1)·cos 2πu2, then ·sin 2πu2), rounded to float32. ln, sin and cos are this module's own (`_ln`, `_sincos`):
  series in +, −, ×, ÷ and √, which IEEE 754 rounds exactly, after exact range reductions. A library's ln or cos can
  differ by an ulp between machines (numpy's SIMD paths, libm), and one such ulp in twenty million numbers would fork
  the chain.
* **The update** (`step`): see its docstring for the exact order of every operation.
* **The problems** (`problem`). Countdown: some numbers and a target, built by combining the numbers with random
  operations first and then hiding the expression, so every problem is solvable. Drawn from the round's seed with
  `crypto.Draw` (hash-based, version-stable), so they're fresh every round, known to nobody before registration closes,
  and need no dataset. `reward` scores an answer exactly (fractions, no eval), so the node itself checks every score.
* **Difficulty** (`LEVELS`, `step`'s retarget): the state carries a level, which a deterministic rule moves every K
  rounds to keep the mean reward near a target. ES learns from differences between rewards: at 4% solved almost every
  nudge scores 0 both ways and there's no signal; near 50% there's the most.

The perturbed tensors are a small named set (`EsSpec.tensors`: glob patterns over the Hugging Face parameter names,
`PRESETS`), so the state fits a 2 GB node. `Group` puts several members' weights into one batch on the GPU, `generate`
answers greedily, and `check` teacher-forces a committed answer through the perturbed model: each token must be the
greedy choice within a small logit margin, which absorbs the rounding differences between a worker's GPU and a
checker's. The worker can't use vLLM (Windows), so generation is plain PyTorch through transformers.

Measured on the RTX 4070 SUPER with Qwen2.5-0.5B-Instruct (float32, TF32 off): CUDA's and the CPU's logits at answer
tokens differ by at most 1.3e-4 (logits up to 31), and in 1,728 honest answer tokens none was off the greedy choice on
either device; answers regenerated on the CPU were identical, rewards too. Forged answers (the right expression, which
the model didn't write) sat 1.15 to 3.6 logits below the greedy choice at their worst token. So `margin` is 0.05: 400×
the devices' difference, 23× under the smallest forgery's.

Not built yet: refereeing a checker's rejection (a dispute over one token's logits, the node recomputing one op as for
fine-tunes; until then a rejected claim is unchecked, unpaid, nobody slashed); canaries that catch a checker approving
without checking; fetching the base from the Hub by its manifest (`register` points at a local folder); racing (each
member answers a few problems first and only the promising finish, in a pinned order); a block version for a public
rollout (a genesis rule gates it today: ledger.ES_GATE).

Not copied from the paper's code (its licence is academic only): this is written from the method's description.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .crypto import Draw

RULE = 1  # the update rule, noise and problems; a record names it, and a change is a rule upgrade
CTRL = 8  # control words at the end of the state: RULE, level, steps, rounds since retarget, reward sum, last mean, 0, 0
MAX_N = 50_000_000  # perturbed numbers at most: the state (×3 with Adam) has to fit a small node
_CHUNK = 1 << 20  # normals made at once (a multiple of 4: Philox makes 4 words a counter)
_BASE = re.compile(r"^hf:([A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?)@([0-9a-f]{40})$", re.ASCII)
_SHA = re.compile(r"^[0-9a-f]{64}$")
OPTIMIZERS = ("sgd", "momentum", "adam")
REWARDS = ("binary", "partial")
# Named sets of tensors to perturb (Hugging Face names, Qwen2/Llama layout). A preset names its patterns in the spec,
# never the preset itself, so what a job perturbs can't change under it.
PRESETS = {
    # Every RMSNorm's scale and every bias: 71,552 numbers in Qwen2.5-0.5B, tiny and with a lot of leverage.
    "norms+biases": ("model.norm.weight", "model.layers.*.input_layernorm.weight",
                     "model.layers.*.post_attention_layernorm.weight", "model.layers.*.self_attn.*_proj.bias"),
    # One layer's feed-forward block (gate, up, down): 13,074,432 numbers in Qwen2.5-0.5B. "{layer}" is filled in.
    "mlp": ("model.layers.{layer}.mlp.gate_proj.weight", "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.down_proj.weight"),
    # Every attention output projection: 19,267,584 numbers in Qwen2.5-0.5B.
    "attn-out": ("model.layers.*.self_attn.o_proj.weight",),
}


# --- deterministic arithmetic ---------------------------------------------------------------------------------------

_LN2 = 0.6931471805599453  # ln 2, the nearest double
_SQRT_HALF = 0.7071067811865476
_QUARTER_PI = 0.7853981633974483  # π/4, the nearest double
# Series coefficients, each the nearest double to a rational (CPython rounds int/int division exactly).
_ATANH = tuple(1 / (2 * k + 1) for k in range(12))  # ln m = 2f·Σ f^2k/(2k+1), f = (m−1)/(m+1), |f| ≤ 0.172
_SIN = tuple((-1) ** k / math.factorial(2 * k + 1) for k in range(9))  # x·Σ (−x²)^k/(2k+1)!, |x| ≤ π/4
_COS = tuple((-1) ** k / math.factorial(2 * k) for k in range(10))  # Σ (−x²)^k/(2k)!


def _horner(coefs, x2: np.ndarray) -> np.ndarray:
    """Σ coefs[k]·x2^k, from the top: a fixed order of exactly rounded multiplies and adds."""
    acc = np.full_like(x2, coefs[-1])
    for c in reversed(coefs[:-1]):
        acc = acc * x2 + c
    return acc


def _ln(x: np.ndarray) -> np.ndarray:
    """ln x for float64 x in (0, 1], to within a few ulps, the same bits on every IEEE machine. x = m·2^e exactly
    (frexp), m moved into [√½, √2) by an exact doubling, then e·ln 2 + 2f·Σ f^2k/(2k+1) with f = (m−1)/(m+1)."""
    m, e = np.frexp(x)
    low = m < _SQRT_HALF
    m = np.where(low, m * 2.0, m)
    e = np.where(low, e - 1, e).astype(np.float64)
    f = (m - 1.0) / (m + 1.0)
    return e * _LN2 + 2.0 * f * _horner(_ATANH, f * f)


def _sincos(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sin 2πu, cos 2πu) for float64 u in [0, 1). The angle's octant comes from u exactly (8u is exact, its floor
    and fraction too), the angle within it is t·π/4 with t in [0, 1] (mirrored, 1 − t, in odd octants, also exact),
    and the rest is two series and a quarter-turn rotation made of swaps and sign flips."""
    v = u * 8.0
    o = np.floor(v)
    t = v - o
    oi = o.astype(np.int64)
    odd = (oi & 1).astype(bool)
    x = np.where(odd, 1.0 - t, t) * _QUARTER_PI
    x2 = x * x
    s = x * _horner(_SIN, x2)
    c = _horner(_COS, x2)
    s = np.where(odd, -s, s)  # an odd octant's angle is the next quarter turn less x
    q = ((oi + odd) >> 1) & 3
    sin = np.select([q == 0, q == 1, q == 2], [s, c, -s], -c)
    cos = np.select([q == 0, q == 1, q == 2], [c, -s, -c], s)
    return sin, cos


def _box_muller(raw: np.ndarray) -> np.ndarray:
    """Normals from raw 64-bit words, two from each two, rounded to float32 (see the module's docstring)."""
    a = (raw[0::2] >> np.uint64(11)).astype(np.float64)
    b = (raw[1::2] >> np.uint64(11)).astype(np.float64)
    u1 = (a + 1.0) * 2.0 ** -53
    u2 = b * 2.0 ** -53
    r = np.sqrt(-2.0 * _ln(u1))
    sin, cos = _sincos(u2)
    z = np.empty(raw.size, np.float64)
    z[0::2] = r * cos
    z[1::2] = r * sin
    return z.astype(np.float32)


_cache: dict = {}  # (seed, n) -> a whole stream, read-only, newest last; off unless cache_noise sets a size
_cache_room = [0]
_cache_lock = threading.Lock()


def cache_noise(max_mb: int) -> None:
    """Keep up to max_mb of whole noise streams in memory (0: none, the default). In one process that plays several
    parts (a testnet: the node, workers answering and checking, each worker following the chain), a pair's ε is then
    made once a round instead of six times. It changes no bits, only how often they're worked out."""
    with _cache_lock:
        _cache.clear()
        _cache_room[0] = max_mb * 2**20


def noise(seed: int, n: int, start: int = 0) -> np.ndarray:
    """Elements [start, start + n) of seed's stream of standard normals, float32: the same bits on every machine, and
    any stretch of it the same as that stretch of the whole (Philox is counter-based: word i is counter i // 4's)."""
    if not 0 <= seed < 2**128 or n < 0 or start < 0:
        raise ValueError("a seed is 0..2^128, and a stretch of the stream starts at 0 or later")
    if start == 0 and 0 < 4 * n <= _cache_room[0]:
        with _cache_lock:
            hit = _cache.pop((seed, n), None)
            if hit is not None:
                _cache[(seed, n)] = hit
                return hit
        out = _noise(seed, n, 0)
        out.flags.writeable = False
        with _cache_lock:
            _cache[(seed, n)] = out
            while sum(a.nbytes for a in _cache.values()) > _cache_room[0]:
                _cache.pop(next(iter(_cache)))
        return out
    return _noise(seed, n, start)


def _noise(seed: int, n: int, start: int) -> np.ndarray:
    out = np.empty(n, np.float32)
    pos = 0
    while pos < n:
        a = start + pos
        lo = a - a % 4
        hi = min(start + n, lo + _CHUNK)
        words = (hi - lo + 3) // 4 * 4
        raw = np.random.Philox(key=seed, counter=lo // 4).random_raw(words)
        take = hi - a
        out[pos:pos + take] = _box_muller(raw)[a - lo:a - lo + take]
        pos += take
    return out


def derive(*parts) -> int:
    """A 64-bit seed from its parts (bytes, ints, strings), each length-prefixed into one SHA-256."""
    h = hashlib.sha256(b"vibehash-es")
    for p in parts:
        b = p if isinstance(p, bytes) else p.to_bytes(16, "big") if isinstance(p, int) else str(p).encode()
        h.update(len(b).to_bytes(4, "big") + b)
    return int.from_bytes(h.digest()[:8], "big")


def pair_seed(round_seed: int, j: int) -> int:
    """The seed of a round's j-th antithetic pair."""
    return derive(b"pair", round_seed, j)


def problems_seed(round_seed: int) -> int:
    return derive(b"problems", round_seed)


EVAL_SEED = derive(b"held-out eval", 0)  # the fixed held-out set's (never a round's: those come from block hashes)


def _ipow(x: float, n: int) -> float:
    """x**n by repeated squaring, exactly rounded steps in a fixed order (C's pow may differ by an ulp by machine)."""
    out, base = 1.0, x
    while n:
        if n & 1:
            out *= base
        base *= base
        n >>= 1
    return out


# --- Countdown -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    """A Countdown difficulty: how many numbers, from 1..top, which operations build the target, and its range."""
    count: int
    top: int
    ops: str
    lo: int
    hi: int


# Easiest first; a job's retarget moves along this table. 5 and 6 are the two halves of the Countdown-Tasks-3to4 set ES
# papers use (3 or 4 numbers below 100, targets up to 100); 6, four numbers, is the default start. Qwen2.5-0.5B-Instruct
# solves, greedy, with messages' wording, on 200 held-out problems a level: 41, 28, 18.5, 13, 11.5, 15.5, 3 and 1%.
# A model that small reaches for three numbers whatever it's given (two numbers: 4.5%), so the easy end is sums of three.
LEVELS = (
    Level(3, 10, "+", 1, 30),
    Level(3, 20, "+", 1, 60),
    Level(3, 10, "+-", 1, 30),
    Level(3, 20, "+-*", 1, 100),
    Level(3, 50, "+-*/", 1, 100),
    Level(3, 99, "+-*/", 1, 100),
    Level(4, 99, "+-*/", 1, 100),
    Level(5, 99, "+-*/", 1, 200),
)


@dataclass(frozen=True)
class Problem:
    numbers: tuple[int, ...]
    target: int
    solution: str  # how it was built (never shown to the model)


def problem(seed: int, i: int, level: int) -> Problem:
    """The i-th problem of a seed's set at a level: numbers drawn, then combined two at a time with random allowed
    operations (a − b only when a > b, a ÷ b only when b > 1 divides a) until one value is left; kept if that's in the
    level's range, else drawn again. Every number is used once, so it's solvable by construction."""
    lv = LEVELS[level]
    d = Draw(b"es-countdown", seed.to_bytes(16, "big"), i.to_bytes(8, "big"), level.to_bytes(2, "big"))
    for _ in range(100_000):
        nums = [1 + d.below(lv.top) for _ in range(lv.count)]
        pool, ok = [(n, str(n)) for n in nums], True
        while ok and len(pool) > 1:
            (a, ea), (b, eb) = pool.pop(d.below(len(pool))), pool.pop(d.below(len(pool)))
            op = lv.ops[d.below(len(lv.ops))]
            if op in "-/" and a < b:
                (a, ea), (b, eb) = (b, eb), (a, ea)
            if op == "-":
                ok, v = a > b, a - b
            elif op == "/":
                ok, v = b > 1 and a % b == 0, a // max(b, 1)
            else:
                v = a + b if op == "+" else a * b
            pool.append((v, f"({ea} {op} {eb})"))
        if ok and lv.lo <= pool[0][0] <= lv.hi:
            return Problem(tuple(nums), pool[0][0], pool[0][1][1:-1])
    raise RuntimeError("no problem found")  # pragma: no cover - each try succeeds about a third of the time


def problems(seed: int, n: int, level: int) -> list[Problem]:
    return [problem(seed, i, level) for i in range(n)]


def heldout(n: int = 200, level: int = 6) -> list[Problem]:
    """The fixed held-out evaluation set: from EVAL_SEED, which no round's problems come from."""
    return problems(EVAL_SEED, n, level)


def messages(p: Problem) -> list[dict]:
    """The chat a problem is asked as. It asks for the expression alone: measured on Qwen2.5-0.5B-Instruct (200
    held-out problems, levels 2-4), asking it to reason first left 75-88% of answers unfinished at 256 tokens and
    1.5-3.5% right, while this wording gets 11-16% right in 16 tokens on average. Its example uses two of three
    numbers: one shaped like "(2 + 3) * 4" was copied, shape and all, into most answers (1-3% right)."""
    nums = ", ".join(str(n) for n in p.numbers)
    return [{"role": "user", "content":
             f"Using the numbers [{nums}], write an arithmetic expression that equals {p.target}. You may use + - * / "
             "and parentheses, and each number at most once. Reply with only the expression between <answer> and "
             "</answer>. For example, for the numbers [4, 6, 9] and the target 15, reply <answer>6 + 9</answer>."}]


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_TOKEN = re.compile(r"\s*(\d+|[-+*/()])")
MAX_EXPR = 200


def parse(expr: str) -> tuple[Fraction, list[int]] | None:
    """An answer's expression: its exact value and the numbers it uses, or None if it isn't one. Integers, + − × ÷
    and parentheses (no unary minus); an expression may end in "= <number>", which is ignored. ÷ by zero is None."""
    expr = expr.strip()
    if "=" in expr:
        left, _, right = expr.partition("=")
        if not re.fullmatch(r"\s*\d+(\.0+)?\s*", right):
            return None
        expr = left
    if not expr or len(expr) > MAX_EXPR:
        return None
    toks, pos = [], 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m:
            if expr[pos:].strip():
                return None
            break
        toks.append(m.group(1))
        pos = m.end()
    used: list[int] = []
    at = [0]

    def peek():
        return toks[at[0]] if at[0] < len(toks) else None

    def atom(depth):
        t = peek()
        if t is None or depth > 32:
            raise ValueError
        at[0] += 1
        if t == "(":
            v = sum_(depth + 1)
            if peek() != ")":
                raise ValueError
            at[0] += 1
            return v
        if not t.isdigit():
            raise ValueError
        used.append(int(t))
        return Fraction(int(t))

    def product(depth):
        v = atom(depth)
        while peek() in ("*", "/"):
            op = toks[at[0]]
            at[0] += 1
            w = atom(depth)
            if op == "/" and w == 0:
                raise ValueError
            v = v * w if op == "*" else v / w
        return v

    def sum_(depth):
        v = product(depth)
        while peek() in ("+", "-"):
            op = toks[at[0]]
            at[0] += 1
            w = product(depth)
            v = v + w if op == "+" else v - w
        return v
    try:
        v = sum_(0)
    except ValueError:
        return None
    return (v, used) if at[0] == len(toks) else None


def reward(p: Problem, text: str, kind: str = "binary") -> float:
    """An answer's score, exactly. The answer is the last <answer>…</answer> in the text. "binary": 1 if its
    expression uses only the given numbers (each at most as often as given) and equals the target, else 0. "partial",
    a denser signal: 0 with no expression; 0.1 if it uses other numbers; 0.2 + 0.6·max(0, 1 − |value − target|/target)
    if it uses only the given ones; 1 if it's right. Worked in fractions and rounded once to a float."""
    found = _ANSWER.findall(text)
    parsed = parse(found[-1]) if found else None
    if parsed is None:
        return 0.0
    value, used = parsed
    left = list(p.numbers)
    fair = True
    for n in used:
        if n in left:
            left.remove(n)
        else:
            fair = False
    if fair and value == p.target:
        return 1.0
    if kind == "binary":
        return 0.0
    if not fair:
        return 0.1
    close = max(Fraction(0), 1 - abs(value - p.target) / p.target)
    return float(Fraction(1, 5) + Fraction(3, 5) * close)


def mean_reward(scores: list[float]) -> float:
    """A member's reward: its problems' scores summed in order in float64, divided by their count."""
    total = 0.0
    for s in scores:
        total += s
    return total / len(scores)


# --- the job's spec --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EsSpec:
    """An `es` job's model and method: the pinned base, which tensors are perturbed, and the ES settings. ModelSpec's
    interface (precision, dtype, n_params, round, init_params, fresh, flops_per_step), so job code can read it."""
    base: str  # hf:org/name@<40-hex commit>
    manifest: str  # sha256 of the base's hub.Manifest
    dims: tuple[tuple[str, object], ...]  # the architecture's sizes (finetune.DIMS), for pricing the work
    tensors: tuple[str, ...]  # glob patterns over parameter names; what they select, sorted by name, is perturbed
    n: int  # numbers they select (checked against the model, as is tensors_sha)
    tensors_sha: str  # sha256 over "name shape\n" of each selected tensor, in order
    sigma: float = 1e-3  # the nudge's scale
    alpha: float = 5e-7  # the step: θ += alpha·g (sgd); see step
    pairs: int = 8  # antithetic pairs a round: a population of 2·pairs
    pairs_per_unit: int = 2
    problems: int = 32  # problems a round, every member answering all of them
    max_tokens: int = 256
    task: str = "countdown"
    reward: str = "binary"
    level: int = 6  # the starting difficulty (LEVELS): four numbers below 100
    retarget: tuple[int, float, float] = (0, 0.5, 0.1)  # every K rounds (0: never), toward a mean reward, ± a band
    optimizer: str = "sgd"
    betas: tuple[float, float] = (0.9, 0.999)  # momentum's first; Adam's both
    eps: float = 1e-8  # Adam's
    check: int = 2  # members of a claim a checker re-checks (beacon-chosen)
    margin: float = 0.05  # logits a committed token may be below the greedy choice's (rounding across devices)
    rule: int = RULE
    precision: str = "float32"  # the state's

    kind = "es"

    def __post_init__(self):
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", tuple(sorted(self.dims.items())))
        for name in ("tensors", "retarget", "betas"):
            v = getattr(self, name)
            if isinstance(v, list):
                object.__setattr__(self, name, tuple(v))

    @property
    def model(self) -> dict:
        return dict(self.dims)

    @property
    def repo(self) -> str:
        return _BASE.match(self.base).group(1)

    @property
    def commit(self) -> str:
        return _BASE.match(self.base).group(2)

    @property
    def dtype(self):
        return np.float32

    @property
    def units(self) -> int:
        return self.pairs // self.pairs_per_unit

    @property
    def members(self) -> int:
        """Members of a unit: each of its pairs both ways."""
        return 2 * self.pairs_per_unit

    @property
    def moments(self) -> int:
        return {"sgd": 0, "momentum": 1, "adam": 2}[self.optimizer]

    @property
    def n_params(self) -> int:
        """The state's length: delta (how far training has moved the perturbed numbers from the base), the
        optimiser's moments, and the control words."""
        return (1 + self.moments) * self.n + CTRL

    def problems_list(self) -> list[str]:
        """Why the spec can't run, or []."""
        out = []
        if not isinstance(self.base, str) or not _BASE.match(self.base):
            out.append("base is hf:org/name@<40-hex commit>")
        if not isinstance(self.manifest, str) or not _SHA.match(self.manifest):
            out.append("manifest is a sha256")
        if not isinstance(self.tensors_sha, str) or not _SHA.match(self.tensors_sha):
            out.append("tensors_sha is a sha256")
        if not (self.tensors and len(self.tensors) <= 16 and all(isinstance(t, str) and 0 < len(t) <= 96
                                                                 and re.fullmatch(r"[\w.*?\[\]-]+", t) for t in self.tensors)):
            out.append("tensors is 1-16 glob patterns over parameter names")
        if not (isinstance(self.n, int) and 0 < self.n <= MAX_N):
            out.append(f"n is 1-{MAX_N:,} perturbed numbers")
        if not (0 < self.sigma <= 1 and 0 < self.alpha <= 10):
            out.append("sigma is in (0, 1] and alpha in (0, 10]")
        if not (isinstance(self.pairs, int) and isinstance(self.pairs_per_unit, int) and 1 <= self.pairs_per_unit
                <= self.pairs <= 4096 and self.pairs % self.pairs_per_unit == 0):
            out.append("pairs is 1-4096, a multiple of pairs_per_unit")
        if not (isinstance(self.problems, int) and 1 <= self.problems <= 512 and isinstance(self.max_tokens, int)
                and 1 <= self.max_tokens <= 2048):
            out.append("problems is 1-512 and max_tokens 1-2048")
        if self.task != "countdown" or self.reward not in REWARDS:
            out.append(f"task is countdown, and reward one of {', '.join(REWARDS)}")
        if not (isinstance(self.level, int) and 0 <= self.level < len(LEVELS)):
            out.append(f"level is 0-{len(LEVELS) - 1}")
        k, target, band = self.retarget
        if not (isinstance(k, int) and 0 <= k <= 1000 and 0 <= target <= 1 and 0 <= band <= 1):
            out.append("retarget is [every K rounds (0: never), target mean reward, band]")
        if self.optimizer not in OPTIMIZERS or not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1 and self.eps > 0):
            out.append(f"optimizer is one of {', '.join(OPTIMIZERS)}, betas in [0, 1), eps positive")
        if not (isinstance(self.check, int) and 1 <= self.check <= self.members and 0 <= self.margin <= 10):
            out.append("check is 1 to a unit's members, margin 0-10")
        if self.rule != RULE:
            out.append(f"this code knows ES rule {RULE}")
        if self.precision != "float32":
            out.append("an ES state is float32")
        return out

    def round(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=np.float32).astype(np.float64)

    def init_params(self, seed: int = 0) -> np.ndarray:
        """The state before any round: delta and moments zero (the base model itself), the control words at the
        rule, the starting level, and nothing counted yet."""
        out = np.zeros(self.n_params, np.float32)
        out[-CTRL] = RULE
        out[-CTRL + 1] = self.level
        return out

    def fresh(self, theta: np.ndarray) -> np.ndarray:
        return theta

    @property
    def base_params(self) -> int:
        """The base model's weights in all (roughly, from its dims), to price the work."""
        m = self.model
        attn = m["hidden"] * m["head_dim"] * (2 * m["heads"] + 2 * m["kv_heads"])
        layer = attn + 3 * m["hidden"] * m["ffn"] + 2 * m["hidden"]
        return m["layers"] * layer + m["vocab"] * m["hidden"] * (1 if m["tied"] else 2)

    def flops_per_step(self, batch_size: int = 1) -> int:
        """A unit's nominal FLOPs (a unit is one step): every member answering every problem, at 2 a weight a token
        for a prompt of PROMPT_TOKENS and max_tokens generated."""
        return 2 * self.base_params * (PROMPT_TOKENS + self.max_tokens) * self.problems * self.members * batch_size

    def check_flops(self) -> int:
        """A check's: one forward pass over its sampled members' prompts and answers."""
        return 2 * self.base_params * (PROMPT_TOKENS + self.max_tokens) * self.problems * self.check

    def to_json(self) -> dict:
        return {"base": self.base, "manifest": self.manifest, "model": self.model,
                "es": {"tensors": list(self.tensors), "n": self.n, "tensors_sha": self.tensors_sha, "sigma": self.sigma,
                       "alpha": self.alpha, "pairs": self.pairs, "pairs_per_unit": self.pairs_per_unit,
                       "problems": self.problems, "max_tokens": self.max_tokens, "task": self.task,
                       "reward": self.reward, "level": self.level, "retarget": list(self.retarget),
                       "optimizer": self.optimizer, "betas": list(self.betas), "eps": self.eps, "check": self.check,
                       "margin": self.margin, "rule": self.rule}}

    @classmethod
    def from_json(cls, d: dict) -> "EsSpec":
        e, model = d["es"], d["model"]
        if not isinstance(e, dict) or not isinstance(model, dict) or not isinstance(e["tensors"], list):
            raise TypeError("es and model are objects, and es's tensors a list")
        k, target, band = e["retarget"]
        b1, b2 = e["betas"]
        known = {"tensors", "n", "tensors_sha", "sigma", "alpha", "pairs", "pairs_per_unit", "problems", "max_tokens",
                 "task", "reward", "level", "retarget", "optimizer", "betas", "eps", "check", "margin", "rule"}
        if set(e) - known:
            raise ValueError(f"an ES spec has no field {sorted(set(e) - known)[0]!r}")
        return cls(base=str(d["base"]), manifest=str(d["manifest"]), dims=dict(model),
                   tensors=tuple(str(t) for t in e["tensors"]), n=int(e["n"]), tensors_sha=str(e["tensors_sha"]),
                   sigma=float(e["sigma"]), alpha=float(e["alpha"]), pairs=int(e["pairs"]),
                   pairs_per_unit=int(e["pairs_per_unit"]), problems=int(e["problems"]),
                   max_tokens=int(e["max_tokens"]), task=str(e["task"]), reward=str(e["reward"]),
                   level=int(e["level"]), retarget=(int(k), float(target), float(band)),
                   optimizer=str(e["optimizer"]), betas=(float(b1), float(b2)), eps=float(e["eps"]),
                   check=int(e["check"]), margin=float(e["margin"]), rule=int(e["rule"]))


PROMPT_TOKENS = 96  # about what a Countdown prompt takes in a chat template, for pricing


def is_es_spec(d) -> bool:
    """Whether a parsed job spec is an ES job's: an "es" object and a model without n_in or lora. Nodes before ES
    jobs read it as an MLP's spec, whose model must have n_in, and refuse it."""
    return isinstance(d, dict) and isinstance(d.get("es"), dict) and isinstance(d.get("model"), dict) \
        and "n_in" not in d["model"] and "lora" not in d


ES_KEYS = {"id", "fee_per_round", "base", "manifest", "model", "es", "init_seed", "units_per_round", "memory_mb",
           "compute_s"}


# --- the state and the update rule ----------------------------------------------------------------------------------

def level_of(state: np.ndarray) -> int:
    return int(state[-CTRL + 1])


def delta_of(spec: EsSpec, state: np.ndarray) -> np.ndarray:
    return np.asarray(state[:spec.n], np.float32)


def shaped(rewards: list[float]) -> list[float]:
    """Z-scores, the reward shaping: (r − mean)/(sd + 1e-8), mean and sd over the population (sd's divisor N), each
    sum a Python float added left to right, √ exactly rounded."""
    n = len(rewards)
    mu = 0.0
    for r in rewards:
        mu += r
    mu /= n
    var = 0.0
    for r in rewards:
        var += (r - mu) * (r - mu)
    sd = math.sqrt(var / n)
    return [(r - mu) / (sd + 1e-8) for r in rewards]


def step(spec: EsSpec, state: np.ndarray, seed: int, update: list) -> np.ndarray:
    """The state after one round, from the state before, the round's seed and its accepted pairs' rewards
    `update` = [[j, r_plus, r_minus], ...] sorted by pair index j (each j once). Rule 1, exactly:

    1. Rewards in member order r = [r⁺_j0, r⁻_j0, r⁺_j1, ...] (N = 2·len(update)); r̂ = shaped(r).
    2. Each pair's coefficient c_j = r̂⁺_j − r̂⁻_j (a Python float), since −ε is the minus member's nudge.
    3. g = Σ_j c_j·ε_j, in float64, adding the pairs in order of j (a pair with c_j = 0 adds nothing and is skipped),
       ε_j = noise(pair_seed(seed, j), n) as float64; then g = g·(1/(N·σ)), the factor a Python float. So
       g = 1/(N·σ)·Σ_i r̂_i·ε_i, the ES estimate of the reward's gradient.
    4. The optimiser, elementwise in float64 from the stored float32 values, each stored value rounded to float32
       once at the end (D: delta; m, v: moments; t: steps so far, this one included):
         sgd:       D' = D + alpha·g                  (so θ += alpha/(N·σ)·Σ r̂_i·ε_i)
         momentum:  m' = β1·m + g;  D' = D + alpha·m'
         adam:      m' = β1·m + (1−β1)·g;  v' = β2·v + (1−β2)·(g·g);  m̂ = m'/(1−β1^t);  v̂ = v'/(1−β2^t);
                    D' = D + (alpha·m̂)/(√v̂ + eps)       (β^t by repeated squaring, _ipow)
    5. Control words: steps t; last mean reward; and with retarget (K, target, band), the round's mean reward is added
       to a sum, and every K steps the level moves up one if sum/K > target + band, down one if below target − band
       (within LEVELS), and the sum and count restart. Each stored float32.

    Done in stretches of _CHUNK numbers, which changes nothing: every operation is elementwise. A round with nothing
    accepted leaves the state exactly as it was. Paper's form: the reference code's θ += α_p/N·Σ r̂_i·ε_i (its
    α_p = σ/2) is alpha = α_p·σ here."""
    state = np.asarray(state, np.float32)
    if not update:
        return state.copy()
    js = [int(u[0]) for u in update]
    if js != sorted(set(js)):
        raise ValueError("an update's pairs are sorted by index, each once")
    rewards = []
    for _, rp, rm in update:
        rewards += [float(rp), float(rm)]
    z = shaped(rewards)
    coef = [z[2 * k] - z[2 * k + 1] for k in range(len(update))]
    n_members, n = len(rewards), spec.n
    scale = 1.0 / (n_members * spec.sigma)
    out = state.copy()
    ctrl = out[-CTRL:]
    t = int(ctrl[2]) + 1
    b1, b2 = spec.betas
    c1, c2 = 1.0 - _ipow(b1, t), 1.0 - _ipow(b2, t)
    seeds = [(pair_seed(seed, j), c) for j, c in zip(js, coef) if c != 0.0]
    whole = 4 * n <= _cache_room[0]  # each ε whole, kept for the round's other users (cache_noise), else in stretches
    total = np.zeros(n, np.float64) if whole else None
    for s, c in seeds if whole else ():
        e = noise(s, n)
        for lo in range(0, n, _CHUNK):
            total[lo:lo + _CHUNK] += e[lo:lo + _CHUNK].astype(np.float64) * c
    for lo in range(0, n, _CHUNK):
        hi = min(n, lo + _CHUNK)
        if whole:
            g = total[lo:hi]
        else:
            g = np.zeros(hi - lo, np.float64)
            for s, c in seeds:
                g += noise(s, hi - lo, lo).astype(np.float64) * c
        g = g * scale
        d = out[lo:hi].astype(np.float64)
        if spec.optimizer == "sgd":
            d = d + g * spec.alpha
        elif spec.optimizer == "momentum":
            m = out[n + lo:n + hi].astype(np.float64) * b1 + g
            out[n + lo:n + hi] = m.astype(np.float32)
            d = d + m * spec.alpha
        else:
            m = out[n + lo:n + hi].astype(np.float64) * b1 + g * (1.0 - b1)
            v = out[2 * n + lo:2 * n + hi].astype(np.float64) * b2 + (g * g) * (1.0 - b2)
            out[n + lo:n + hi] = m.astype(np.float32)
            out[2 * n + lo:2 * n + hi] = v.astype(np.float32)
            d = d + ((m / c1) * spec.alpha) / (np.sqrt(v / c2) + spec.eps)
        out[lo:hi] = d.astype(np.float32)
    mu = 0.0
    for r in rewards:
        mu += r
    mu /= n_members
    ctrl[2] = t
    ctrl[5] = mu
    k, target, band = spec.retarget
    if k:
        since, total = int(ctrl[3]) + 1, float(ctrl[4]) + mu
        if since >= k:
            avg, level = total / k, int(ctrl[1])
            if avg > target + band and level < len(LEVELS) - 1:
                level += 1
            elif avg < target - band and level > 0:
                level -= 1
            ctrl[1], since, total = level, 0, 0.0
        ctrl[3], ctrl[4] = since, total
    return out


def state_hash(state: np.ndarray) -> str:
    """The state's params_hash (model.params_hash), hex."""
    from .model import params_hash
    return params_hash(state).hex()


def rebuild(spec: EsSpec, history: list[dict], state: np.ndarray | None = None) -> np.ndarray:
    """The state after `history`, each round's record entry {"seed": hex, "update": [[j, r+, r−], ...]} in chain
    order, from the start (or `state`). Pure arithmetic: no GPU, no model, no weight files."""
    state = spec.init_params() if state is None else np.asarray(state, np.float32)
    for h in history:
        state = step(spec, state, int(h["seed"], 16), h["update"])
    return state


def rebuild_chain(blocks, record_of, job_id: str, spec: EsSpec | None = None) -> tuple[np.ndarray, int]:
    """A job's state rebuilt from a chain alone: its blocks (ledger.Block, in order) and `record_of(height)`, each
    round's record. Every record is checked against the record_hash its block commits to, and every round's state
    against the params_hash its block's summary holds; ValueError at the first that differs. (The blocks themselves
    are the caller's to validate, as Chain.append does.) Returns (state, rounds rebuilt)."""
    from .crypto import h256
    from .ledger import canonical
    from .tasks import TrainingJob

    state, rounds = None, 0
    for b in blocks:
        s = b.summary
        if s is None or s.job_id != job_id:
            if spec is None and b.height:  # the job's post_job, whose spec this needs
                for tx in b.txs:
                    if tx.kind == "post_job":
                        try:
                            job = TrainingJob.from_spec_json(tx.memo, tx.sender, tx.amount)
                        except ValueError:
                            continue
                        if job.job_id == job_id:
                            spec = job.spec
            continue
        if spec is None:
            raise ValueError(f"no post_job for {job_id} before block {b.height}")
        if state is None:
            state = spec.init_params()
        record = record_of(b.height)
        if h256(canonical(record)).hex() != s.record_hash:
            raise ValueError(f"block {b.height}'s record isn't the one it commits to")
        h = record["es"]
        if h.get("rule") != spec.rule:
            raise ValueError(f"block {b.height}'s round follows ES rule {h.get('rule')}, this code rule {spec.rule}")
        state = step(spec, state, int(h["seed"], 16), h["update"])
        if state_hash(state) != s.params_hash:
            raise ValueError(f"block {b.height}: rebuilt state {state_hash(state)[:16]}… but the block says "
                             f"{s.params_hash[:16]}…")
        rounds += 1
    if state is None:
        state = spec.init_params() if spec is not None else None
    return state, rounds


# --- the model (PyTorch) ---------------------------------------------------------------------------------------------

LOCAL: dict[str, str] = {}  # manifest sha256 -> a local folder holding the base (Hugging Face layout): tests, testnets
_runtimes: dict = {}
_lock = threading.Lock()


def register(manifest: str, folder) -> None:
    """Say where a base model is on this computer (instead of fetching it from the Hub)."""
    LOCAL[manifest] = str(folder)


def select(named, patterns) -> list[tuple[str, tuple]]:
    """The tensors `patterns` pick from `named` ((name, shape) pairs), sorted by name. Only RMSNorm scales and
    linear layers' weights and biases can be perturbed (never the embedding, which the head shares)."""
    out = []
    for name, shape in named:
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            if "embed" in name or "lm_head" in name:
                raise ValueError(f"{name}: the embedding and head can't be perturbed")
            out.append((name, tuple(int(s) for s in shape)))
    return sorted(out)


def tensors_digest(chosen) -> tuple[int, str]:
    """(numbers, sha256 over "name shape" lines) of a selection: what the spec pins."""
    h = hashlib.sha256()
    total = 0
    for name, shape in chosen:
        h.update(f"{name} {'x'.join(map(str, shape))}\n".encode())
        total += math.prod(shape)
    return total, h.hexdigest()


def preset(name: str, layer: int | None = None) -> tuple[str, ...]:
    pats = PRESETS[name]
    return tuple(p.format(layer=layer) for p in pats) if "{layer}" in "".join(pats) else pats


@dataclass
class Runtime:
    """A base model on one device for ES work: transformers' model in float32 with TF32 off, its tokenizer, and the
    perturbed tensors (name, shape, module, attribute, base values) in the spec's order. Shared by the threads of a
    process (a testnet's workers), which take turns on it (`busy`)."""
    model: object
    tokenizer: object
    device: str
    eos: tuple[int, ...]
    pad: int
    targets: list = field(default_factory=list)
    busy: object = field(default_factory=threading.Lock)


def _load(folder: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, attn_implementation="sdpa")
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(folder)
    gen = json.loads((Path(folder) / "generation_config.json").read_text()) if (Path(folder) / "generation_config.json").exists() else {}
    eos = gen.get("eos_token_id", tok.eos_token_id)
    eos = tuple(eos) if isinstance(eos, list) else (eos,)
    pad = gen.get("pad_token_id", tok.pad_token_id if tok.pad_token_id is not None else eos[0])
    return model, tok, eos, pad


def runtime(spec: EsSpec, device: str) -> Runtime:
    """The spec's base on `device`, loaded once per process; its perturbed tensors checked against the spec."""
    key = (spec.manifest, device, spec.tensors, spec.n, spec.tensors_sha)
    with _lock:
        rt = _runtimes.get(key)
        if rt is None:
            folder = LOCAL.get(spec.manifest)
            if folder is None:
                raise RuntimeError(f"{spec.repo}'s weights aren't on this computer: es.register(manifest, folder) "
                                   "(fetching an ES job's base from the Hub isn't built yet)")
            model, tok, eos, pad = _load(folder, device)
            rt = Runtime(model, tok, device, eos, pad)
            params = dict(model.named_parameters())
            chosen = select(((n, p.shape) for n, p in params.items()), spec.tensors)
            if tensors_digest(chosen) != (spec.n, spec.tensors_sha):
                raise ValueError("the spec's tensors aren't what its patterns pick from this model")
            for name, shape in chosen:
                mod_name, attr = name.rsplit(".", 1)
                mod = model.get_submodule(mod_name)
                rt.targets.append((name, shape, mod, attr, params[name].detach().clone()))
            _runtimes[key] = rt
        return rt


def unload() -> None:
    with _lock:
        _runtimes.clear()


def _is_norm(mod) -> bool:
    return hasattr(mod, "variance_epsilon") or type(mod).__name__.endswith("RMSNorm")


class Group:
    """P members' weights in the model at once: in a batch of P·B rows, rows [p·B, (p+1)·B) run member p, whose
    perturbed tensors are float32(float64(base) + float64(delta) + s·σ·float64(ε)) with s its sign (+1, −1, or 0 for
    the unperturbed model). Each targeted module's forward is swapped for a grouped one (a batched matmul, or a scale
    by group) while the group is entered. A member is (pair seed, sign)."""

    def __init__(self, rt: Runtime, spec: EsSpec, delta: np.ndarray, members: list[tuple[int, int]]):
        self.rt, self.spec, self.members, self.P = rt, spec, members, len(members)
        self.delta = np.asarray(delta, np.float32)
        self._saved = []

    def _stacks(self):
        import torch

        rt, spec = self.rt, self.spec
        eps = {}
        for s, sign in self.members:
            if sign and s not in eps:
                eps[s] = noise(s, spec.n)
        out, at = {}, 0
        for name, shape, mod, attr, base in rt.targets:
            size = math.prod(shape)
            b = base.to(torch.float64) + torch.from_numpy(self.delta[at:at + size]).to(rt.device, torch.float64).view(shape)
            rows = []
            for s, sign in self.members:
                if sign:
                    e = torch.tensor(eps[s][at:at + size], device=rt.device, dtype=torch.float64).view(shape)
                    rows.append((b + e * (sign * spec.sigma)).to(torch.float32))
                else:
                    rows.append(b.to(torch.float32))
            out[name] = torch.stack(rows)
            at += size
        return out

    def __enter__(self):
        stacks = self._stacks()
        by_module: dict = {}
        for name, shape, mod, attr, base in self.rt.targets:
            by_module.setdefault(id(mod), (mod, {}))[1][attr] = stacks[name]
        for mod, w in by_module.values():
            self._saved.append((mod, mod.forward))
            mod.forward = self._grouped(mod, w)
        return self

    def __exit__(self, *exc):
        for mod, fwd in reversed(self._saved):
            mod.forward = fwd
        self._saved = []

    def _grouped(self, mod, w: dict):
        import torch

        P = self.P
        if _is_norm(mod):
            scale, eps = w["weight"], mod.variance_epsilon

            def norm(h):
                dt = h.dtype
                x = h.to(torch.float32)
                x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                x = x.to(dt).reshape(P, -1, x.shape[-1])
                return (scale[:, None, :] * x).reshape(h.shape)
            return norm
        weight, bias = w.get("weight"), w.get("bias")

        def linear(x):
            xs = x.reshape(P, -1, x.shape[-1])
            y = torch.bmm(xs, weight.transpose(1, 2)) if weight is not None else torch.matmul(xs, mod.weight.t())
            if bias is not None:
                y = y + bias[:, None, :]
            elif mod.bias is not None:
                y = y + mod.bias
            return y.reshape(*x.shape[:-1], y.shape[-1])
        return linear


def prompt_ids(rt: Runtime, ps: list[Problem]) -> list[list[int]]:
    tok = rt.tokenizer
    out = []
    for p in ps:
        ids = tok.apply_chat_template(messages(p), add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        out.append(list(ids))
    return out


def _left_pad(rows: list[list[int]], pad: int, device: str):
    import torch

    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = torch.tensor(r, dtype=torch.long)
        mask[i, width - len(r):] = 1
    return ids.to(device), mask.to(device)


def generate(rt: Runtime, rows: list[list[int]], max_tokens: int) -> list[list[int]]:
    """Greedy answers to prompts `rows` (inside a Group, P·B of them, group-major), up to max_tokens each: the token
    with the highest logit every step (the first on a tie), stopping at an end-of-turn token, which the answer keeps."""
    import torch

    ids, mask = _left_pad(rows, rt.pad, rt.device)
    eos = torch.tensor(rt.eos, device=rt.device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = rt.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        done = torch.isin(nxt, eos)
        made, last = [nxt], pos[:, -1:]
        for _ in range(max_tokens - 1):
            if bool(done.all()):
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            last = last + 1
            out = rt.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last, past_key_values=cache,
                           use_cache=True)
            cache = out.past_key_values
            nxt = torch.where(done, torch.full_like(nxt, rt.pad), out.logits[:, -1].argmax(-1))
            made.append(nxt)
            done = done | torch.isin(nxt, eos)
    toks = torch.stack(made, 1).cpu().tolist()
    answers = []
    for row in toks:
        cut = next((i + 1 for i, t in enumerate(row) if t in rt.eos), len(row))
        answers.append(row[:cut])
    return answers


def text_of(rt: Runtime, answer: list[int]) -> str:
    return rt.tokenizer.decode([t for t in answer if t not in rt.eos], skip_special_tokens=True)


def gaps(rt: Runtime, rows: list[list[int]], answers: list[list[int]]) -> list[list[float]]:
    """Teacher forcing (inside a Group): for each row's answer token, how far its logit is below the highest at that
    position (0 when it's the greedy choice), from one forward pass over prompt + answer."""
    import torch

    seqs = [r + a for r, a in zip(rows, answers)]
    ids, mask = _left_pad(seqs, rt.pad, rt.device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    width = ids.shape[1]
    out = []
    with torch.no_grad():
        hidden = rt.model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
        for i, a in enumerate(answers):
            if not a:
                out.append([])
                continue
            at = torch.arange(width - len(a) - 1, width - 1, device=rt.device)
            want = torch.tensor(a, device=rt.device)
            row = []
            for lo in range(0, len(a), 512):
                logits = rt.model.lm_head(hidden[i, at[lo:lo + 512]])
                top = logits.max(-1).values
                row += (top - logits.gather(1, want[lo:lo + 512, None])[:, 0]).tolist()
            out.append(row)
    return out


def unit_members(spec: EsSpec, round_seed: int, index: int) -> list[tuple[int, int, int]]:
    """A unit's members, in order: (pair index j, pair seed, sign), each pair + then −."""
    out = []
    for k in range(spec.pairs_per_unit):
        j = index * spec.pairs_per_unit + k
        s = pair_seed(round_seed, j)
        out += [(j, s, 1), (j, s, -1)]
    return out


def answer(rt: Runtime, spec: EsSpec, delta: np.ndarray, members: list[tuple[int, int]], ps: list[Problem],
           max_rows: int = 256) -> list[list[list[int]]]:
    """Every member's greedy answers (token ids) to every problem, as many members at once as max_rows allows."""
    rows = prompt_ids(rt, ps)
    per = max(1, max_rows // max(1, len(ps)))
    out = []
    with rt.busy:
        for lo in range(0, len(members), per):
            group = members[lo:lo + per]
            with Group(rt, spec, delta, group):
                got = generate(rt, rows * len(group), spec.max_tokens)
            out += [got[k * len(ps):(k + 1) * len(ps)] for k in range(len(group))]
    return out


def claim(rt: Runtime, spec: EsSpec, delta: np.ndarray, round_seed: int, level: int, index: int,
          max_rows: int = 256) -> dict:
    """A unit's work: each member's answers (ids and text) to the round's problems and its reward, as a worker commits
    and reveals it. {"answers": [[{"ids", "text"} per problem] per member], "rewards": [[r+, r−] per pair]}."""
    ps = problems(problems_seed(round_seed), spec.problems, level)
    members = unit_members(spec, round_seed, index)
    got = answer(rt, spec, delta, [(s, sign) for _, s, sign in members], ps, max_rows)
    answers = [[{"ids": a, "text": text_of(rt, a)} for a in member] for member in got]
    scores = [mean_reward([reward(p, a["text"], spec.reward) for p, a in zip(ps, member)]) for member in answers]
    return {"answers": answers, "rewards": [[scores[2 * k], scores[2 * k + 1]] for k in range(spec.pairs_per_unit)]}


def claim_bytes(unit_id: str, answers: list) -> bytes:
    """A unit's answers as the blob a worker uploads and commits to: canonical JSON."""
    return json.dumps({"unit": unit_id, "answers": answers}, sort_keys=True, separators=(",", ":")).encode()


def claim_root(answers_sha: str, rewards: list) -> bytes:
    """What a commitment binds: the answers blob's sha256 and the rewards claimed for it."""
    from .crypto import h256
    return h256(b"es-claim", bytes.fromhex(answers_sha),
                json.dumps(rewards, separators=(",", ":")).encode())


def rescore(spec: EsSpec, round_seed: int, level: int, raw: bytes, unit_id: str) -> list | None:
    """The rewards a claim's answers earn, from its blob, or None if the blob isn't a unit's answers in shape: what
    the node checks every claim's rewards against (no model needed)."""
    try:
        d = json.loads(raw)
        if not isinstance(d, dict) or d.get("unit") != unit_id:
            return None
        answers = d["answers"]
        if not (isinstance(answers, list) and len(answers) == spec.members
                and all(isinstance(m, list) and len(m) == spec.problems for m in answers)):
            return None
        ps = problems(problems_seed(round_seed), spec.problems, level)
        scores = []
        for member in answers:
            for a in member:
                if not (isinstance(a, dict) and isinstance(a.get("text"), str) and isinstance(a.get("ids"), list)
                        and len(a["ids"]) <= spec.max_tokens and all(type(t) is int for t in a["ids"])):
                    return None
            scores.append(mean_reward([reward(p, a["text"], spec.reward) for p, a in zip(ps, member)]))
    except (ValueError, KeyError, TypeError):
        return None
    return [[scores[2 * k], scores[2 * k + 1]] for k in range(spec.pairs_per_unit)]


def sample(spec: EsSpec, beacon: bytes, secret: bytes, unit_id: str, worker: str) -> list[int]:
    """Which of a claim's members (indices into its unit's) its checkers re-check: `check` of them, from the beacon
    and the round secret, so nobody knows which before the claim is committed."""
    d = Draw(b"es-sample", beacon, secret, unit_id.encode(), worker.encode())
    left, out = list(range(spec.members)), []
    while len(out) < spec.check:
        out.append(left.pop(d.below(len(left))))
    return sorted(out)


def check(rt: Runtime, spec: EsSpec, delta: np.ndarray, round_seed: int, level: int, index: int, raw: bytes,
          chosen: list[int], unit_id: str) -> tuple[bool, int | None, str]:
    """A checker's verdict on a claim's sampled members: (ok, the first bad member or None, why). Each sampled member's
    answers must be token ids whose text is their decoding, end at an end-of-turn token or at max_tokens, and be the
    greedy choice of the member's perturbed model at every token to within `margin` (teacher-forced: one forward pass).
    The rewards are the node's to check (rescore)."""
    try:
        answers = json.loads(raw)["answers"]
    except (ValueError, KeyError, TypeError):
        return False, chosen[0], "the answers aren't readable"
    ps = problems(problems_seed(round_seed), spec.problems, level)
    rows = prompt_ids(rt, ps)
    members = unit_members(spec, round_seed, index)
    for k in chosen:
        try:
            got = [list(map(int, a["ids"])) for a in answers[k]]
            texts = [a["text"] for a in answers[k]]
        except (KeyError, TypeError, ValueError, IndexError):
            return False, k, "a member's answers aren't readable"
        for a, t in zip(got, texts):
            if t != text_of(rt, a):
                return False, k, "an answer's text isn't its tokens' decoding"
            if len(a) > spec.max_tokens or (len(a) < spec.max_tokens and (not a or a[-1] not in rt.eos)):
                return False, k, "an answer stops before an end-of-turn token or max_tokens"
        _, s, sign = members[k]
        with rt.busy, Group(rt, spec, delta, [(s, sign)]):
            g = gaps(rt, rows, got)
        worst = max((x for row in g for x in row), default=0.0)
        if worst > spec.margin:
            return False, k, f"a token {worst:.4f} below the greedy choice's logit"
    return True, None, ""


def evaluate(rt: Runtime, spec: EsSpec, delta: np.ndarray, ps: list[Problem], max_rows: int = 256,
             max_tokens: int | None = None) -> float:
    """pass@1 of the unperturbed model θ = base + delta on `ps`, greedy."""
    rows = prompt_ids(rt, ps)
    right = 0
    with rt.busy, Group(rt, spec, delta, [(0, 0)]):
        for lo in range(0, len(ps), max_rows):
            got = generate(rt, rows[lo:lo + max_rows], max_tokens or spec.max_tokens)
            right += sum(reward(p, text_of(rt, a), "binary") == 1.0 for p, a in zip(ps[lo:lo + max_rows], got))
    return right / len(ps)
