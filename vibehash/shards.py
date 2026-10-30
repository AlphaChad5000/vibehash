"""Sharded units: training a model too big for one computer, checked without anyone running all of it.

A design prototype (SHARDING.md), not yet part of the protocol. The model is split into shards, contiguous blocks of
layers, each held by a different worker of a team. A training step runs as a pipeline: activations go forward from
shard to shard, gradients come back, and each shard updates its own weights. Every worker commits to its shard's
checkpoints and to every tensor that crossed its edges, so a unit's work falls into cells, one shard by one segment,
and a checker can replay any one cell holding only that shard's weights and the tensors at its two edges. Neighbouring
shards' commitments must name identical edge tensors, so each cell's inputs are what its neighbours really sent, and
checking cells one at a time checks the whole unit.

The maths is model.py's, cut at layer boundaries with the same operations in the same order, so a sharded unit trains
to exactly the weights the whole-model reference does (tests/test_shards.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crypto import h256, merkle_proof, merkle_root, merkle_verify
from .model import Dataset, ModelSpec, batch_indices, relative_distance
from .verification import reproduces

# Per step, what crossed a shard's edges: activations in from the shard before it, gradients in from the one after,
# and what it sent out each way. The first shard reads its input from the dataset and the last its gradient from the
# labels, so those are None.
KINDS = ("h_in", "g_out", "h_out", "g_in")
CHEATS = ("noisy_output", "zero_gradient", "frozen")  # for tests, like worker.py's behaviours


@dataclass(frozen=True)
class Shard:
    index: int
    lo: int  # its first layer
    hi: int  # one past its last
    n_layers: int  # in the whole model

    @property
    def first(self) -> bool:
        return self.lo == 0

    @property
    def last(self) -> bool:
        return self.hi == self.n_layers

    def span(self, spec: ModelSpec) -> slice:
        """Where its weights sit in the model's flat vector. Layers are stored in order (W1, b1, W2, b2, ...), so a
        block of layers is one contiguous slice, and a shard's weights are a plain cut of the whole model's."""
        w = spec.widths
        start = sum(a * b + b for a, b in zip(w[:self.lo], w[1:self.lo + 1]))
        return slice(start, start + sum(a * b + b for a, b in zip(w[self.lo:self.hi], w[self.lo + 1:self.hi + 1])))

    def layers(self, spec: ModelSpec, params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(W, b), ...] views into the shard's own weights."""
        out, i, w = [], 0, spec.widths
        for a, b in zip(w[self.lo:self.hi], w[self.lo + 1:self.hi + 1]):
            out.append((params[i:i + a * b].reshape(a, b), params[i + a * b:i + a * b + b]))
            i += a * b + b
        return out


def split(spec: ModelSpec, n: int) -> list[Shard]:
    """`n` shards of whole layers with about the same number of weights each, so each fits about the same memory."""
    sizes = [a * b + b for a, b in zip(spec.widths, spec.widths[1:])]
    if not 1 <= n <= len(sizes):
        raise ValueError(f"a {len(sizes)}-layer model splits into 1 to {len(sizes)} shards")
    cuts, run, total = [0], 0, sum(sizes)
    for i, size in enumerate(sizes[:-1]):
        run += size
        left = n - len(cuts)  # shards still to start
        if left and (run >= total * len(cuts) / n or len(sizes) - (i + 1) == left):
            cuts.append(i + 1)
    cuts.append(len(sizes))
    return [Shard(k, lo, hi, len(sizes)) for k, (lo, hi) in enumerate(zip(cuts, cuts[1:]))]


def tensor_hash(a: np.ndarray | None) -> bytes:
    if a is None:
        return h256(b"none")
    return h256(repr(a.shape).encode(), np.ascontiguousarray(a, dtype=np.float64).tobytes())


# --- one shard's share of a training step (model._forward and model._gradient, cut at layer boundaries) ------------

def _forward(shard: Shard, layers: list, h_in: np.ndarray):
    """Returns (each layer's input, the shard's output): activations for the next shard, or the softmax if it's last."""
    acts = [h_in]
    for w, b in layers if not shard.last else layers[:-1]:
        acts.append(np.tanh(acts[-1] @ w + b))
    if not shard.last:
        return acts, acts[-1]
    w, b = layers[-1]
    logits = acts[-1] @ w + b
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return acts, probs


def _backward(shard: Shard, layers: list, acts: list, out: np.ndarray, g_out: np.ndarray | None, y: np.ndarray):
    """Returns (the gradient to send back, or None from the first shard; the gradient of the shard's own weights).
    Edges carry the gradient with respect to the activations, before tanh's derivative: the shard that applied the
    tanh applies its derivative too, which is the whole model's `(d @ w.T) * (1 - a ** 2)` cut in two, unchanged."""
    if shard.last:
        d = out.copy()
        d[np.arange(len(y)), y] -= 1
        d /= len(y)
    else:
        d = g_out * (1 - acts[-1] ** 2)
    grads = []
    for j in range(len(layers) - 1, -1, -1):
        w, _ = layers[j]
        grads.append((acts[j].T @ d, d.sum(axis=0)))
        if j:
            d = (d @ w.T) * (1 - acts[j] ** 2)
    g_in = None if shard.first else d @ layers[0][0].T
    return g_in, np.concatenate([g.ravel() for gw, gb in reversed(grads) for g in (gw, gb)])


# --- a team training a unit ----------------------------------------------------------------------------------------

@dataclass
class Trace:
    """What one shard's worker keeps from a unit: its weights every `checkpoint_every` steps, and every tensor that
    crossed its edges, step by step. It commits to all of it (`root`) before the beacon picks which cells to check."""
    shard: Shard
    checkpoints: list[np.ndarray]
    tensors: list[dict]

    def leaves(self) -> list[bytes]:
        """Checkpoints first, then each step's four edge tensors in KINDS order (`leaf_index`)."""
        return [tensor_hash(c) for c in self.checkpoints] + [tensor_hash(t[k]) for t in self.tensors for k in KINDS]

    def root(self) -> bytes:
        return merkle_root(self.leaves())


def leaf_index(n_checkpoints: int, step: int, kind: str) -> int:
    return n_checkpoints + 4 * step + KINDS.index(kind)


def train_sharded(spec: ModelSpec, shards: list[Shard], data: Dataset, theta: np.ndarray, unit_seed: int, n_steps: int,
                  checkpoint_every: int, lr: float, batch_size: int, cheats: dict[int, str] | None = None) -> list[Trace]:
    """A unit trained as a pipeline, one worker per shard, in one process: what each worker would end up holding.
    `cheats` maps a shard to one of CHEATS. Each step runs every shard's forward in order, then every backward in
    reverse, each shard updating its own weights once its backward is done; nothing else uses them afterwards."""
    cheats, dt = cheats or {}, spec.dtype
    lr = dt(lr)
    params = [np.asarray(theta[s.span(spec)], dtype=dt).copy() for s in shards]
    traces = [Trace(s, [p.astype(np.float64)], []) for s, p in zip(shards, params)]
    noise = np.random.default_rng(unit_seed)  # only for the noisy_output cheat
    n = len(data.y_train)
    for step in range(n_steps):
        idx = batch_indices(unit_seed, step, n, batch_size)
        h, y = data.x_train[idx].astype(dt), data.y_train[idx]
        rec, saved = [dict.fromkeys(KINDS) for _ in shards], []
        for k, s in enumerate(shards):
            layers = s.layers(spec, params[k])
            acts, out = _forward(s, layers, h)
            if not s.first:
                rec[k]["h_in"] = h
            if not s.last:
                if cheats.get(k) == "noisy_output":
                    out = (out + noise.normal(0, 0.01, out.shape)).astype(dt)
                rec[k]["h_out"] = out
            saved.append((layers, acts, out))
            h = out
        g = None
        for k in range(len(shards) - 1, -1, -1):
            s, (layers, acts, out) = shards[k], saved[k]
            if not s.last:
                rec[k]["g_out"] = g
            g, grad = _backward(s, layers, acts, out, g, y)
            if not s.first:
                if cheats.get(k) == "zero_gradient":
                    g = np.zeros_like(g)
                rec[k]["g_in"] = g
            if cheats.get(k) != "frozen":
                params[k] -= lr * grad
        for k, t in enumerate(traces):
            t.tensors.append(rec[k])
            if (step + 1) % checkpoint_every == 0:
                t.checkpoints.append(params[k].astype(np.float64))
    return traces


def assemble(spec: ModelSpec, shards: list[Shard], parts: list[np.ndarray]) -> np.ndarray:
    """The whole model's flat vector from its shards' weights."""
    theta = np.empty(spec.n_params)
    for s, p in zip(shards, parts):
        theta[s.span(spec)] = p
    return theta


# --- checking one cell ---------------------------------------------------------------------------------------------

def open_cell(trace: Trace, seg: int, checkpoint_every: int) -> dict:
    """What a worker reveals for a challenged cell: its weights at both ends of the segment, the edge tensors of its
    steps, and Merkle proofs that all of them are what it committed to."""
    c, leaves, n_ckpt = checkpoint_every, trace.leaves(), len(trace.checkpoints)
    steps = range(seg * c, (seg + 1) * c)
    idx = [seg, seg + 1] + [leaf_index(n_ckpt, t, k) for t in steps for k in KINDS]
    return {"before": trace.checkpoints[seg], "after": trace.checkpoints[seg + 1],
            "tensors": trace.tensors[seg * c:(seg + 1) * c], "proofs": {i: merkle_proof(leaves, i) for i in idx}}


def _proves(leaf: bytes, proof: list, root: bytes, index: int) -> bool:
    """A Merkle proof of `leaf` at position `index`. The proof's left/right steps spell out where the leaf sits, so
    they must match `index`, or a proof of one leaf would pass for another position (say, a checkpoint's hash offered
    as an edge tensor's)."""
    return all(is_left == bool((index >> level) & 1) for level, (_, is_left) in enumerate(proof)) \
        and merkle_verify(leaf, proof, root)


def _close(a: np.ndarray, claim, tolerance: float) -> bool:
    return isinstance(claim, np.ndarray) and claim.shape == a.shape and relative_distance(claim, a) <= tolerance


def check_cell(spec: ModelSpec, shard: Shard, data: Dataset, root: bytes, seg: int, cell: dict, unit_seed: int,
               n_steps: int, checkpoint_every: int, lr: float, batch_size: int, tolerance: float) -> str | None:
    """A checker's job: None if the cell holds up, else what's wrong. It needs this one shard's weights (the cell's
    two checkpoints) and the tensors at its edges, never the rest of the model: it replays the segment from the inputs
    the worker committed to (what its neighbours sent, which edges_agree ties to their commitments), so honest
    floating-point differences in other shards never add up here. Each step's outputs must match the replay within
    the job's tolerance, and the weights at the segment's end must pass the same test a whole-model replay uses."""
    c, n_ckpt = checkpoint_every, n_steps // checkpoint_every + 1
    steps = range(seg * c, (seg + 1) * c)
    proofs = cell["proofs"]
    opened = {seg: cell["before"], seg + 1: cell["after"]}
    opened |= {leaf_index(n_ckpt, t, k): rec[k] for t, rec in zip(steps, cell["tensors"]) for k in KINDS}
    for i, value in opened.items():
        if i not in proofs or not _proves(tensor_hash(value), proofs[i], root, i):
            return f"leaf {i} isn't what the worker committed to"
    dt = spec.dtype
    lr, params, n = dt(lr), np.asarray(cell["before"], dtype=dt).copy(), len(data.y_train)
    for t, rec in zip(steps, cell["tensors"]):
        idx = batch_indices(unit_seed, t, n, batch_size)
        x, y = data.x_train[idx].astype(dt), data.y_train[idx]
        layers = shard.layers(spec, params)
        if not shard.first and not isinstance(rec["h_in"], np.ndarray):
            return f"step {t}: no input activations"
        acts, out = _forward(shard, layers, x if shard.first else rec["h_in"].astype(dt))
        if not shard.last and not _close(out, rec["h_out"], tolerance):
            return f"step {t}: the activations it sent on aren't what its layers make"
        if not shard.last and not isinstance(rec["g_out"], np.ndarray):
            return f"step {t}: no incoming gradient"
        g_in, grad = _backward(shard, layers, acts, out, None if shard.last else rec["g_out"].astype(dt), y)
        if not shard.first and not _close(g_in, rec["g_in"], tolerance):
            return f"step {t}: the gradient it sent back isn't what its layers make"
        params -= lr * grad
    if not reproduces(np.asarray(cell["after"]), params.astype(np.float64), np.asarray(cell["before"]), tolerance, spec):
        return "its weights at the segment's end aren't what training gives"
    return None


def open_edges(trace: Trace, steps) -> dict:
    """The hashes (with proofs) of a worker's edge tensors for `steps`, for comparing with its neighbours'."""
    leaves, n_ckpt = trace.leaves(), len(trace.checkpoints)
    return {(t, k): (leaves[i], merkle_proof(leaves, i)) for t in steps for k in KINDS
            for i in [leaf_index(n_ckpt, t, k)]}


def edges_agree(lower_root: bytes, lower: dict, upper_root: bytes, upper: dict, n_checkpoints: int, steps) -> str | None:
    """Two neighbouring shards' commitments must name the same tensors where they meet: what the lower one sent on is
    what the upper one took in, and the gradient the upper one sent back is what the lower one took in. A hash
    comparison, no compute. (Signed edge messages say which side lied when they don't: SHARDING.md.)"""
    for root, opened in ((lower_root, lower), (upper_root, upper)):
        for t in steps:
            for k in KINDS:
                if (t, k) not in opened or not _proves(*opened[(t, k)], root, leaf_index(n_checkpoints, t, k)):
                    return f"step {t}: {k} isn't what was committed to"
    for t in steps:
        if lower[(t, "h_out")][0] != upper[(t, "h_in")][0]:
            return f"step {t}: the activations sent on aren't the ones the next shard took in"
        if lower[(t, "g_out")][0] != upper[(t, "g_in")][0]:
            return f"step {t}: the gradient taken in isn't the one the next shard sent back"
    return None
