"""Mixture-of-experts: a feed-forward block that sends each token to a few of many experts, and how a checker checks it.

A design prototype (SHARDING.md §7), not yet part of the protocol: the drop-in for a transformer's MLP that the biggest
open models use. A linear router scores every expert for each token, the token goes to its `top_k` best, and the
block's output is their SwiGLU outputs weighted by gates (the chosen experts' scores, normalised over the k), plus the
outputs of any shared experts, which every token goes through (DeepSeek's design). It works on [tokens, d_model]
activations; a transformer's [batch, seq, d_model] reshapes to that and back.

The top-k choice is where replay breaks. It's a jump: when two experts' scores agree to the seventh digit, a checker
whose hardware rounds differently from the worker's can pick the other one, and then that token's output is a different
expert's, nowhere near any tolerance. So a worker commits to its routing (which experts each token went to, and their
gates) along with the tensors at its edges. A checker replays with the committed routing (check_block), and separately
checks that each committed choice was the router's own top-k pick up to a tie margin (check_routing). With the routing
fixed, the rest of the block is continuous, and the job's tolerance absorbs rounding as it always has (DESIGN §8).

Fixing the routing also makes every expert a cell of its own, as shards.py makes every shard one: an expert's outputs
are a pure function of its weights and the tokens routed to it. A checker holding that one expert checks it
(check_expert), and one holding the router and the shared experts checks the routing and how the experts' outputs were
combined (check_router). MoETrace lays out the commitments, extending shards.Trace's.

There's no auxiliary load-balancing loss: it would tie each token's gradient to every other token's routing, for
nothing a replay needs. The natural later option is DeepSeek-V3's bias-based balancing, a per-expert bias added to the
scores when choosing (never to the gates) and nudged between steps by each expert's load. That keeps a step a pure
function, with the bias committed like the weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .crypto import h256, merkle_proof, merkle_root
from .model import PRECISIONS
from .shards import KINDS, _close, _proves, tensor_hash

# How far, in the router's logits, a committed choice may lose to an expert it left out: no left-out expert may beat a
# chosen one by more than this. Logits are log-scores up to a per-token constant, so this is a relative margin on the
# scores (within a factor e^-1e-4, about 0.01%). Relative, because honest rounding is: an absolute margin on
# probabilities would let a confident router's lower slots go to any expert at all, every runner-up's score being near
# zero. Honest float32 and float64 routers measured at most 4e-6 apart (widths 16 to 7168, 8 to 256 experts, logits
# ~N(0, 1) as an RMS-normalised input gives), so this leaves 25 times headroom, and it's the size of a float32 job's
# tolerance. What it lets a cheat do: on the tokens with such a near-tie (under 0.2% of them for 8 to 256 experts),
# swap an expert for one the router rates as good to within 0.01%. That saves no work, since every token still goes to
# k experts. A job could set its own, measured across GPU vendors as tolerances are; either way the router has to run in
# float32 or better even where the experts don't (bfloat16 logits are only good to about 1e-2).
ROUTING_MARGIN = 1e-4

MATRICES = ("gate", "up", "down")  # each expert's SwiGLU: down(silu(x @ gate) * (x @ up)), Llama's MLP
CHEATS = ("skip", "noisy")  # one expert's work faked, for tests, like shards.CHEATS

# A block's leaves (MoETrace): the dense part's (the router and shared experts, which every token goes through), then
# the routing, then each expert's. The dense part's include the block's four edge tensors in shards.KINDS order.
DENSE_KINDS = ("weights", *KINDS, "grad")
EXPERT_KINDS = ("weights", "x", "dy", "y", "dx", "grad")  # in: its tokens, their gradient; out: its outputs, theirs
SENT = ("x", "dy", "y", "dx")  # what crosses between the router's side and an expert: the all-to-all's traffic


@dataclass(frozen=True)
class MoESpec:
    d_model: int = 16
    d_ff: int = 32  # each expert's hidden width; MoE experts are narrow (DeepSeek-V3: 2048, in a 7168-wide model)
    n_experts: int = 8
    top_k: int = 2  # at least 2: with one, normalising over the chosen makes every gate 1 and the router untrainable
    n_shared: int = 0  # always-on experts every token goes through, at full weight (DeepSeek-V3 has 1)
    precision: str = "float64"  # as model.ModelSpec
    lora_rank: int = 0  # 0 trains every weight; otherwise each expert matrix gets an adapter and its base is frozen
    lora_alpha: float = 16.0

    @property
    def dtype(self):
        return PRECISIONS[self.precision]

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_rank

    def round(self, a) -> np.ndarray:
        """Snap to the job's precision, carried as float64, as ModelSpec.round."""
        return np.asarray(a, dtype=self.dtype).astype(np.float64)

    def _mlp(self, prefix: str) -> list[tuple[str, tuple[int, int]]]:
        d, f, r = self.d_model, self.d_ff, self.lora_rank
        out = []
        for m, (a, b) in zip(MATRICES, ((d, f), (d, f), (f, d))):
            out.append((f"{prefix}.{m}", (a, b)))
            if r:
                out += [(f"{prefix}.{m}.lora_a", (a, r)), (f"{prefix}.{m}.lora_b", (r, b))]
        return out

    def layout(self, expert: int | None = None) -> list[tuple[str, tuple[int, ...]]]:
        """(name, shape) of every array in one part of the block, in order: expert `expert`'s, or with None the dense
        part every token goes through (the router and the shared experts). Each part's arrays are its own, never a
        slice of one [n_experts, ...] stack, so a worker or a checker can hold any subset of experts, and each part
        packs into one flat vector (pack) as a shard's weights are one slice of model.py's."""
        if expert is not None:
            return self._mlp(f"expert.{expert}")
        shared = [a for s in range(self.n_shared) for a in self._mlp(f"shared.{s}")]
        return [("router", (self.d_model, self.n_experts)), *shared]

    def trains(self, name: str) -> bool:
        """With LoRA, experts' base weights are frozen and only their adapters train. The router trains in full either
        way: it's d_model x n_experts (1.8M weights for DeepSeek-V3's 7168 x 256)."""
        return not self.lora_rank or name == "router" or name.endswith((".lora_a", ".lora_b"))

    def names(self, expert: int | None = None, trainable: bool = False) -> list[str]:
        return [n for n, _ in self.layout(expert) if not trainable or self.trains(n)]

    def part_size(self, expert: int | None = None, trainable: bool = False) -> int:
        return sum(math.prod(s) for n, s in self.layout(expert) if not trainable or self.trains(n))

    @property
    def n_params(self) -> int:
        return self.part_size() + self.n_experts * self.part_size(0)

    @property
    def active_params(self) -> int:
        """Weights one token goes through: the router, the shared experts, and k routed ones."""
        return self.part_size() + self.top_k * self.part_size(0)

    def init_params(self, seed: int) -> dict[str, np.ndarray]:
        """Named arrays, float64 holding values of the job's precision. LoRA's B starts at zero, so the adapters start
        as no change at all."""
        rng = np.random.default_rng(seed)
        params = {}
        for part in (None, *range(self.n_experts)):
            for name, shape in self.layout(part):
                fresh = np.zeros(shape) if name.endswith(".lora_b") else rng.normal(0, 1 / np.sqrt(shape[0]), shape)
                params[name] = self.round(fresh)
        return params


def pack(spec: MoESpec, arrays: dict, expert: int | None = None, trainable: bool = False) -> np.ndarray:
    """One part's arrays (expert `expert`'s, or the dense part's) as one flat float64 vector in layout order: what a
    worker holding that part stores, hashes and sends. With `trainable`, only the arrays that train, for gradients."""
    return np.concatenate([np.asarray(arrays[n], dtype=np.float64).ravel() for n in spec.names(expert, trainable)])


def unpack(spec: MoESpec, vector: np.ndarray, expert: int | None = None, trainable: bool = False) -> dict[str, np.ndarray]:
    """pack's inverse: views into `vector`, by name."""
    out, i = {}, 0
    for name, shape in spec.layout(expert):
        if not trainable or spec.trains(name):
            out[name] = vector[i:i + math.prod(shape)].reshape(shape)
            i += math.prod(shape)
    if i != len(vector):
        raise ValueError(f"{len(vector)} values for a part of {i}")
    return out


@dataclass(frozen=True, eq=False)
class Routing:
    """Which experts each token went to, and their gates: what a worker commits to, so that a checker replays its
    choices rather than remaking them (the module's docstring says why)."""
    experts: np.ndarray  # [tokens, k] expert indices, best first
    gates: np.ndarray  # [tokens, k] each one's weight in the token's output; a token's gates sum to 1

    def groups(self, n_experts: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per expert, the tokens routed to it (in token order) and which of their k slots chose it: the batch that
        expert runs on. Only for a well-formed routing (_malformed says it is)."""
        k = self.experts.shape[1]
        flat = self.experts.ravel()
        order = np.argsort(flat, kind="stable")  # by expert, then by token: a function of the routing alone
        return [(o // k, o % k) for o in np.split(order, np.cumsum(np.bincount(flat, minlength=n_experts))[:-1])]

    def hash(self) -> bytes:
        return h256(repr(self.experts.shape).encode(), np.ascontiguousarray(self.experts, dtype=np.int64).tobytes(),
                    np.ascontiguousarray(self.gates, dtype=np.float64).tobytes())


def _malformed(spec: MoESpec, routing, n_tokens: int) -> str | None:
    e, g = getattr(routing, "experts", None), getattr(routing, "gates", None)
    if not isinstance(e, np.ndarray) or not isinstance(g, np.ndarray) or e.shape != (n_tokens, spec.top_k) \
            or g.shape != e.shape:
        return f"it doesn't give each of the {n_tokens} tokens {spec.top_k} experts and gates"
    if e.dtype.kind not in "iu" or (e.size and (e.min() < 0 or e.max() >= spec.n_experts)):
        return "it names experts that don't exist"
    s = np.sort(e, axis=1)
    if np.any(s[:, 1:] == s[:, :-1]):
        return "it sends a token to the same expert twice"
    return None


# --- the block's maths, in the job's precision ---------------------------------------------------------------------

def _w(spec: MoESpec, params: dict, name: str) -> np.ndarray:
    return np.asarray(params[name], dtype=spec.dtype)


def _logits(spec: MoESpec, params: dict, x: np.ndarray) -> np.ndarray:
    return x @ _w(spec, params, "router")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _top_k(logits: np.ndarray, k: int) -> np.ndarray:
    """Each token's k best experts, best first. Chosen on the logits, which order the experts exactly as the scores do
    but without the ties rounding can make among small probabilities; exact ties go to the lower index, so the choice
    is a function of the bits."""
    return np.argsort(-logits, axis=1, kind="stable")[:, :k]


def _gates(scores: np.ndarray, experts: np.ndarray) -> np.ndarray:
    chosen = np.take_along_axis(scores, experts, axis=1)
    return chosen / chosen.sum(axis=1, keepdims=True)


def _linear(spec: MoESpec, params: dict, name: str, x: np.ndarray):
    """x @ W, plus LoRA's scale * (x @ A) @ B when the job has adapters. Returns (that, x @ A for the backward)."""
    z = x @ _w(spec, params, name)
    if not spec.lora_rank:
        return z, None
    u = x @ _w(spec, params, f"{name}.lora_a")
    return z + spec.dtype(spec.lora_scale) * (u @ _w(spec, params, f"{name}.lora_b")), u


def _linear_back(spec: MoESpec, params: dict, name: str, x: np.ndarray, u, dz: np.ndarray, grads: dict) -> np.ndarray:
    """Adds the weights' gradients to `grads` (the adapters' only, with LoRA) and returns the input's."""
    dx = dz @ _w(spec, params, name).T
    if not spec.lora_rank:
        grads[name] = x.T @ dz
        return dx
    s = spec.dtype(spec.lora_scale)
    du = s * (dz @ _w(spec, params, f"{name}.lora_b").T)
    grads[f"{name}.lora_a"] = x.T @ du
    grads[f"{name}.lora_b"] = s * (u.T @ dz)
    return dx + du @ _w(spec, params, f"{name}.lora_a").T


def _swiglu(spec: MoESpec, params: dict, prefix: str, x: np.ndarray):
    """One expert on its tokens. Returns (its output, what its backward needs)."""
    a, ua = _linear(spec, params, f"{prefix}.gate", x)
    b, ub = _linear(spec, params, f"{prefix}.up", x)
    sig = 0.5 + 0.5 * np.tanh(0.5 * a)  # the logistic function, without exp overflowing for very negative a
    h = a * sig * b
    y, uh = _linear(spec, params, f"{prefix}.down", h)
    return y, (x, a, b, sig, h, ua, ub, uh)


def _swiglu_back(spec: MoESpec, params: dict, prefix: str, saved, dy: np.ndarray):
    """Returns (the gradient for the expert's input, its weights' gradients by name)."""
    x, a, b, sig, h, ua, ub, uh = saved
    grads: dict = {}
    dh = _linear_back(spec, params, f"{prefix}.down", h, uh, dy, grads)
    dx = _linear_back(spec, params, f"{prefix}.gate", x, ua, dh * b * sig * (1 + a * (1 - sig)), grads)  # silu'(a)
    return dx + _linear_back(spec, params, f"{prefix}.up", x, ub, dh * a * sig, grads), grads


class LocalExperts:
    """Runs routed experts on weights this process holds: forward's default. The block only ever hands an expert its
    tokens and takes back its outputs (and the same for gradients), so other runners fit the same two calls: wave 2's
    expert parallelism would send the tokens to the worker holding the expert (the all-to-all), and check_router replays
    with the tensors a worker committed to."""

    def __init__(self, spec: MoESpec, params: dict):
        self.spec, self.params = spec, params

    def forward(self, e: int, x: np.ndarray):
        return _swiglu(self.spec, self.params, f"expert.{e}", x)

    def backward(self, e: int, saved, dy: np.ndarray):
        return _swiglu_back(self.spec, self.params, f"expert.{e}", saved, dy)


def forward(spec: MoESpec, params: dict, x: np.ndarray, routing: Routing | None = None, experts=None):
    """The block on [tokens, d_model] activations. Returns (output, what backward needs). Without a routing it routes
    the tokens itself; with a committed one it sends each token where that says. Either way it computes the gates from
    its own scores, because they're continuous in the weights and the router's gradient flows through them (check_routing
    holds the committed gates to these). saved["routing"] is the routing it used. `experts` runs the routed experts
    (default: LocalExperts on `params`); with another runner, `params` needs only the dense part's arrays."""
    dt = spec.dtype
    x = np.asarray(x, dtype=dt)
    logits = _logits(spec, params, x)
    if routing is None:
        chosen = _top_k(logits, spec.top_k)
    else:
        reason = _malformed(spec, routing, len(x))
        if reason:
            raise ValueError(f"routing: {reason}")
        chosen = np.asarray(routing.experts, dtype=np.int64)
    routing = Routing(chosen, _gates(_softmax(logits), chosen))
    groups = routing.groups(spec.n_experts)
    experts = experts or LocalExperts(spec, params)
    shared = [_swiglu(spec, params, f"shared.{s}", x) for s in range(spec.n_shared)]
    routed = [experts.forward(e, x[ids]) for e, (ids, _) in enumerate(groups)]
    # Summed in a fixed order, shared experts and then routed ones by index, so the same inputs give the same bits.
    out = np.zeros_like(x)
    for y, _ in shared:
        out += y
    for (ids, slots), (y, _) in zip(groups, routed):
        out[ids] += routing.gates[ids, slots][:, None] * np.asarray(y, dtype=dt)  # a token visits an expert at most once
    return out, {"x": x, "routing": routing, "groups": groups, "shared": shared, "routed": routed, "experts": experts}


def backward(spec: MoESpec, params: dict, saved: dict, g_out: np.ndarray):
    """Returns (the gradient for the block's input, the gradients of its trainable arrays by name). The router's comes
    through the gates: they're the chosen experts' scores over their sum, which is a softmax over just the chosen
    logits (the full softmax's denominator cancels), so only those k logits get a gradient. The choice itself is a
    jump and has none."""
    dt = spec.dtype
    g_out = np.asarray(g_out, dtype=dt)
    x, routing, experts = saved["x"], saved["routing"], saved["experts"]
    g_in, grads = np.zeros_like(x), {}
    for s, (_, cache) in enumerate(saved["shared"]):
        dx, g = _swiglu_back(spec, params, f"shared.{s}", cache, g_out)
        g_in += dx
        grads |= g
    dgates = np.zeros_like(routing.gates)
    for e, ((ids, slots), (y, cache)) in enumerate(zip(saved["groups"], saved["routed"])):
        g = g_out[ids]
        dgates[ids, slots] = (g * np.asarray(y, dtype=dt)).sum(axis=1)
        dx, eg = experts.backward(e, cache, routing.gates[ids, slots][:, None] * g)
        g_in[ids] += np.asarray(dx, dtype=dt)
        grads |= eg
    d_chosen = routing.gates * (dgates - (dgates * routing.gates).sum(axis=1, keepdims=True))
    dlogits = np.zeros((len(x), spec.n_experts), dtype=dt)
    np.put_along_axis(dlogits, routing.experts, d_chosen, axis=1)
    grads["router"] = x.T @ dlogits
    g_in += dlogits @ _w(spec, params, "router").T
    return g_in, grads


# --- checking the routing, and replaying with it -------------------------------------------------------------------

def route(spec: MoESpec, params: dict, x: np.ndarray) -> tuple[Routing, np.ndarray]:
    """The router's own choice for each token, and every expert's score (softmax probabilities, [tokens, n_experts])."""
    x = np.asarray(x, dtype=spec.dtype)
    logits = _logits(spec, params, x)
    scores, chosen = _softmax(logits), _top_k(logits, spec.top_k)
    return Routing(chosen, _gates(scores, chosen)), scores


def check_routing(spec: MoESpec, params: dict, x: np.ndarray, committed: Routing,
                  margin: float = ROUTING_MARGIN) -> str | None:
    """None if `committed` is a routing the router could have made here, else what's wrong. For every token: no expert
    it left out may beat one it chose by more than `margin` in the logits (ROUTING_MARGIN says why logits and why that
    size). That makes each chosen expert's score within the margin of the k-th best, and more: the weaker rule would let
    a worker drop the top expert for one tied with the k-th. And the gates must be the chosen experts' scores normalised
    over the k, within the same margin relatively (a margin's error in the logits moves a gate by that fraction)."""
    x = np.asarray(x, dtype=spec.dtype)
    reason = _malformed(spec, committed, len(x))
    if reason:
        return reason
    logits = _logits(spec, params, x)
    chosen = committed.experts.astype(np.int64)
    worst_in = np.take_along_axis(logits, chosen, axis=1).min(axis=1)
    left_out = logits.copy()
    np.put_along_axis(left_out, chosen, -np.inf, axis=1)
    best_out = left_out.max(axis=1)  # -inf when every expert was chosen
    beaten = np.flatnonzero(best_out - worst_in > margin)
    if beaten.size:
        t = beaten[0]
        loser = chosen[t, np.take_along_axis(logits, chosen, axis=1)[t].argmin()]
        return (f"token {t}: it left out expert {left_out[t].argmax()}, which beats chosen expert {loser} by "
                f"{best_out[t] - worst_in[t]:.3g}, over the margin of {margin:g}")
    expected = _gates(_softmax(logits), chosen)
    gates = np.asarray(committed.gates, dtype=np.float64)
    off = ~np.isfinite(gates) | (np.abs(gates - expected) > margin * expected + np.finfo(spec.dtype).eps)
    if off.any():
        t = np.flatnonzero(off.any(axis=1))[0]
        return f"token {t}: its gates aren't its experts' scores normalised over the {spec.top_k}"
    return None


def check_block(spec: MoESpec, params: dict, x: np.ndarray, committed: Routing, claimed_out: np.ndarray,
                claimed_grads: dict, tolerance: float, g_out: np.ndarray, margin: float = ROUTING_MARGIN) -> str | None:
    """A whole block's step checked by replay: None if it holds up, else what's wrong. The routing must pass
    check_routing; then the block is replayed with it, from the input `x` and the gradient `g_out` the layer above sent
    back. Its output and gradients must match within the job's tolerance. `claimed_grads` holds the gradient of every
    trainable array by name, and the input's as "x"."""
    reason = check_routing(spec, params, x, committed, margin)
    if reason:
        return reason
    out, saved = forward(spec, params, x, committed)
    if not _close(out, claimed_out, tolerance):
        return "its output isn't what the experts its routing chose make"
    g_in, grads = backward(spec, params, saved, g_out)
    grads = {"x": g_in, **grads}
    if not isinstance(claimed_grads, dict) or set(claimed_grads) != set(grads):
        return "it claims gradients for the wrong arrays"
    for name, g in grads.items():
        if not _close(g, claimed_grads[name], tolerance):
            return "the gradient it sent back isn't what backpropagation gives" if name == "x" \
                else f"the gradient of {name} isn't what backpropagation gives"
    return None


# --- expert-level cells ---------------------------------------------------------------------------------------------

@dataclass
class MoETrace:
    """What a worker keeps from one step of an MoE block, laid out so each expert is a cell of its own. It extends
    shards.Trace's commitment. The block's four edge tensors are the same KINDS, hashed the same way, so a neighbouring
    shard's commitment must name the same ones (shards.edges_agree). After them come the routing and, per expert, its
    weights, the tokens it was sent, the gradient that came back for them, what it returned each way, and its weights'
    gradient (EXPERT_KINDS). In a unit, a shard holding MoE layers would add these leaves for every step, and give each
    expert's weights a leaf of its own at every checkpoint, so that a challenged expert opens alone."""
    dense: dict  # DENSE_KINDS: the router's and shared experts' weights and gradient (flat), and the block's edges
    routing: Routing
    experts: list[dict]  # per expert, EXPERT_KINDS

    def leaves(self) -> list[bytes]:
        return [tensor_hash(self.dense[k]) for k in DENSE_KINDS] + [self.routing.hash()] \
            + [tensor_hash(ex[k]) for ex in self.experts for k in EXPERT_KINDS]

    def root(self) -> bytes:
        return merkle_root(self.leaves())


def leaf_index(kind: str, expert: int | None = None) -> int:
    """Where a leaf sits in MoETrace.leaves: the dense part's kind, "routing", or expert `expert`'s kind."""
    if expert is None:
        return len(DENSE_KINDS) if kind == "routing" else DENSE_KINDS.index(kind)
    return len(DENSE_KINDS) + 1 + len(EXPERT_KINDS) * expert + EXPERT_KINDS.index(kind)


class _Worker(LocalExperts):
    """A worker's own experts, keeping every tensor that crossed between them and the router's side: what it commits
    to. `cheats` maps an expert to one of CHEATS, for tests: its outputs zeroed (it never ran, and returns nothing
    back either) or noisy. Either way the worker combines the fake as if it were real, so its block hangs together."""

    def __init__(self, spec: MoESpec, params: dict, cheats: dict[int, str]):
        super().__init__(spec, params)
        self.cheats, self.sent = cheats, [{} for _ in range(spec.n_experts)]

    def forward(self, e, x):
        y, saved = super().forward(e, x)
        if self.cheats.get(e) == "skip":
            y = np.zeros_like(y)
        elif self.cheats.get(e) == "noisy":
            y = (y + np.random.default_rng(e).normal(0, 0.01, y.shape)).astype(y.dtype)
        self.sent[e].update(x=x, y=y)
        return y, saved

    def backward(self, e, saved, dy):
        dx, grads = super().backward(e, saved, dy)
        if self.cheats.get(e) == "skip":
            dx, grads = np.zeros_like(dx), {n: np.zeros_like(g) for n, g in grads.items()}
        self.sent[e].update(dy=dy, dx=dx, grad=pack(self.spec, grads, e, trainable=True))
        return dx, grads


def trace_block(spec: MoESpec, params: dict, x: np.ndarray, g_out: np.ndarray, routing: Routing | None = None,
                cheats: dict[int, str] | None = None) -> MoETrace:
    """One step of the block as a worker runs it: forward (routing the tokens itself, unless told how), backward from
    the gradient the layer above sent back, and everything it commits to."""
    worker = _Worker(spec, params, cheats or {})
    out, saved = forward(spec, params, x, routing, worker)
    g_in, grads = backward(spec, params, saved, g_out)
    dense = {"weights": pack(spec, params), "h_in": saved["x"], "g_out": np.asarray(g_out, dtype=spec.dtype),
             "h_out": out, "g_in": g_in, "grad": pack(spec, grads, trainable=True)}
    experts = [{"weights": pack(spec, params, e), **sent} for e, sent in enumerate(worker.sent)]
    return MoETrace(dense, saved["routing"], experts)


def _proofs(trace: MoETrace, indices) -> dict:
    leaves = trace.leaves()
    return {i: merkle_proof(leaves, i) for i in indices}


def open_expert(trace: MoETrace, e: int) -> dict:
    """What a worker reveals for a challenged expert: its weights, its tokens in and outputs out, both gradients and its
    weights' gradient, with Merkle proofs that they're what it committed to. Nothing of any other expert's."""
    return {**trace.experts[e], "proofs": _proofs(trace, [leaf_index(k, e) for k in EXPERT_KINDS])}


def check_expert(spec: MoESpec, root: bytes, e: int, cell: dict, tolerance: float) -> str | None:
    """A checker's job for one expert: None if it holds up, else what's wrong. It takes no model, only the opened cell:
    that expert's weights and the tokens routed to it, never the router, another expert or another token, so a checker
    needs one expert's memory, 1/n_experts of the routed weights. It replays the expert on the tokens it was sent and
    the gradient that came back (check_router ties those to the block), and its outputs, the gradient it returned and
    its weights' gradient must match within the job's tolerance."""
    proofs = cell.get("proofs", {})
    for k in EXPERT_KINDS:
        i, value = leaf_index(k, e), cell.get(k)
        if not isinstance(value, np.ndarray) or i not in proofs or not _proves(tensor_hash(value), proofs[i], root, i):
            return f"its {k} isn't what the worker committed to"
    if cell["weights"].shape != (spec.part_size(e),) or cell["x"].ndim != 2 or cell["x"].shape[1] != spec.d_model \
            or cell["dy"].shape != cell["x"].shape:
        return "its tensors aren't the shapes of an expert's"
    weights, dt = unpack(spec, cell["weights"], e), spec.dtype
    y, saved = _swiglu(spec, weights, f"expert.{e}", np.asarray(cell["x"], dtype=dt))
    if not _close(y, cell["y"], tolerance):
        return "its outputs aren't what its weights make of the tokens it was sent"
    dx, grads = _swiglu_back(spec, weights, f"expert.{e}", saved, np.asarray(cell["dy"], dtype=dt))
    if not _close(dx, cell["dx"], tolerance):
        return "the gradient it sent back isn't what its weights make"
    if not _close(pack(spec, grads, e, trainable=True), cell["grad"], tolerance):
        return "its weights' gradient isn't what backpropagation gives"
    return None


def open_router(trace: MoETrace) -> dict:
    """What a worker reveals for the router's cell: the dense part's weights and gradient, the block's edge tensors,
    the routing, and what went to and from each expert (SENT), with proofs. No routed expert's weights."""
    idx = [leaf_index(k) for k in DENSE_KINDS] + [leaf_index("routing")] \
        + [leaf_index(k, e) for e in range(len(trace.experts)) for k in SENT]
    return {**trace.dense, "routing": trace.routing, "experts": [{k: ex[k] for k in SENT} for ex in trace.experts],
            "proofs": _proofs(trace, idx)}


class _Committed:
    """Experts that return what the worker committed to, and remember the gradient they were handed."""

    def __init__(self, cells: list[dict], dtype):
        self.cells, self.dtype, self.handed = cells, dtype, {}

    def forward(self, e, x):
        return np.asarray(self.cells[e]["y"], dtype=self.dtype), None

    def backward(self, e, saved, dy):
        self.handed[e] = dy
        return np.asarray(self.cells[e]["dx"], dtype=self.dtype), {}


def check_router(spec: MoESpec, root: bytes, cell: dict, tolerance: float, margin: float = ROUTING_MARGIN) -> str | None:
    """The other kind of cell: the router, the shared experts, and how the experts' outputs were combined. None if it
    holds up, else what's wrong. It needs the dense part's weights and the tensors that went to and from the experts,
    never a routed expert's weights. The routing must pass check_routing, and each expert must have been sent exactly
    the tokens the routing gave it: a gather, so bit for bit. Then the block is replayed with the experts' committed
    outputs in place of the experts: its output, the gradient it handed each expert, the one it sent back and its
    weights' gradient must match within the job's tolerance. Every leaf of a trace is checked by this cell or an
    expert's, so, as with shards, checking cells one at a time checks the block, and a lie fails only the cell it's in."""
    proofs, opened = cell.get("proofs", {}), cell.get("experts")
    if not isinstance(opened, list) or len(opened) != spec.n_experts or not all(isinstance(ex, dict) for ex in opened):
        return f"it doesn't open {spec.n_experts} experts' tensors"
    leaves = [(leaf_index(k), cell.get(k)) for k in DENSE_KINDS]
    leaves += [(leaf_index(k, e), ex.get(k)) for e, ex in enumerate(opened) for k in SENT]
    for i, value in leaves:
        if not isinstance(value, np.ndarray) or i not in proofs or not _proves(tensor_hash(value), proofs[i], root, i):
            return f"leaf {i} isn't what the worker committed to"
    routing, i = cell.get("routing"), leaf_index("routing")
    if not isinstance(routing, Routing) or i not in proofs or not _proves(routing.hash(), proofs[i], root, i):
        return "its routing isn't what the worker committed to"
    if cell["weights"].shape != (spec.part_size(),):
        return "its weights aren't the router's and shared experts'"
    if cell["h_in"].ndim != 2 or cell["h_in"].shape[1] != spec.d_model \
            or any(cell[k].shape != cell["h_in"].shape for k in ("g_out", "h_out", "g_in")):
        return "its edge tensors aren't the block's shape"
    params, dt = unpack(spec, cell["weights"]), spec.dtype
    x = np.asarray(cell["h_in"], dtype=dt)
    reason = check_routing(spec, params, x, routing, margin)
    if reason:
        return reason
    for e, ((ids, _), ex) in enumerate(zip(routing.groups(spec.n_experts), opened)):
        if ex["x"].shape != (len(ids), spec.d_model) or not np.array_equal(ex["x"], x[ids]):
            return f"expert {e} wasn't sent the tokens the routing gave it"
        if any(ex[k].shape != ex["x"].shape for k in ("dy", "y", "dx")):
            return f"expert {e}'s tensors aren't the shape of its tokens"
    committed = _Committed(opened, dt)
    out, saved = forward(spec, params, x, routing, committed)
    if not _close(out, cell["h_out"], tolerance):
        return "its output isn't its experts' outputs combined by their gates"
    g_in, grads = backward(spec, params, saved, cell["g_out"])
    for e, ex in enumerate(opened):
        if not _close(committed.handed[e], ex["dy"], tolerance):
            return f"the gradient it handed expert {e} isn't the incoming one times the gates"
    if not _close(g_in, cell["g_in"], tolerance):
        return "the gradient it sent back isn't what its experts and router give"
    if not _close(pack(spec, grads, trainable=True), cell["grad"], tolerance):
        return "its router's or shared experts' gradient isn't what backpropagation gives"
    return None
