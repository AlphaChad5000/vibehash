"""Sharded units: training a model too big for one computer, checked without anyone running all of it.

The core of SHARDING.md, and the protocol's side of it: coordinator.py runs sharded jobs' rounds with these pieces,
and protocol.py's in-process teams train with train_sharded. The model is split into shards, contiguous blocks of
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

import math
from dataclasses import dataclass, field

import numpy as np

from .blobs import check_id
from .crypto import h256, merkle_root, merkle_verify
from .model import Dataset, ModelSpec, batch_indices, relative_distance
from .verification import _swap_leaf, challenge_segments, reproduces, rescale_update
from .wire import WireError, proof_in, proof_out

# Per step, what crossed a shard's edges: activations in from the shard before it, gradients in from the one after,
# and what it sent out each way. The first shard reads its input from the dataset and the last its gradient from the
# labels, so those are None.
KINDS = ("h_in", "g_out", "h_out", "g_in")
# For tests, like worker.py's behaviours. CHEATS lie in what they compute, so their own cells fail: noisy activations
# sent on (the last shard, which sends none, adds the noise to the gradient it sends back), zeroed gradients sent back
# (the first shard, which sends none, skips its backward pass instead), or weights left untrained. EDGE_CHEATS compute
# honestly but lie about what crossed an edge, so their cells pass and the edge doesn't: one commits to the tensor it
# should have sent while sending another, the other takes in something other than it was sent.
CHEATS = ("noisy_output", "zero_gradient", "frozen")
EDGE_CHEATS = ("lies_sent", "lies_received")
RECEIVED = {"h_out": "h_in", "g_in": "g_out"}  # a tensor a shard sends, as the shard it goes to records it
SENT = {v: k for k, v in RECEIVED.items()}


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

    def size(self, spec: ModelSpec) -> int:
        span = self.span(spec)
        return span.stop - span.start

    def layers(self, spec: ModelSpec, params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(W, b), ...] views into the shard's own weights."""
        out, i, w = [], 0, spec.widths
        for a, b in zip(w[self.lo:self.hi], w[self.lo + 1:self.hi + 1]):
            out.append((params[i:i + a * b].reshape(a, b), params[i + a * b:i + a * b + b]))
            i += a * b + b
        return out

    def shapes(self, spec: ModelSpec, batch_size: int) -> dict[str, tuple | None]:
        """Each edge tensor's shape at every step: one row per example, as wide as the layer at that edge; None where
        the shard has no neighbour."""
        into, out = (batch_size, spec.widths[self.lo]), (batch_size, spec.widths[self.hi])
        return {"h_in": None if self.first else into, "g_in": None if self.first else into,
                "h_out": None if self.last else out, "g_out": None if self.last else out}


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


# --- what a sharded job asks of each shard's worker -----------------------------------------------------------------

def cell_flops(spec: ModelSpec, shard: Shard, batch_size: int, steps: int) -> int:
    """A shard's training FLOPs over `steps` steps, ModelSpec.flops_per_step's rule for its weights alone: so a unit's
    shards add up to exactly the whole unit's FLOPs, and pay per verified cell is pay per FLOP, as for whole units."""
    return 6 * shard.size(spec) * batch_size * steps


def memory_mb(job_memory_mb: int, spec: ModelSpec, shard: Shard) -> int:
    """The memory a worker needs to hold a shard, to train or check its cells: the job's `memory_mb` is what the whole
    model needs, and a shard needs its share of it, by weights."""
    return max(1, math.ceil(job_memory_mb * shard.size(spec) / spec.n_params))


def segment_steps(segments, checkpoint_every: int) -> set[int]:
    return {t for s in segments for t in range(s * checkpoint_every, (s + 1) * checkpoint_every)}


def challenged(beacon: bytes, unit_id: str, team: list[str], worker: str, n_segments: int, checkpoint_every: int,
               q: int) -> tuple[list[int], list[int]]:
    """What the beacon asks of one team member for a unit: the segments of its shard whose cells it opens (picked as a
    whole unit's are, by verification.challenge_segments), and the steps at which it opens its edge hashes: those of
    its own challenged cells and of its neighbours', so every challenged cell's inputs and outputs can be compared
    with what the shard on the other side of each edge committed to."""
    k = team.index(worker)
    near = [challenge_segments(beacon, unit_id, a, n_segments, q) for a in team[max(0, k - 1):k + 2]]
    return (challenge_segments(beacon, unit_id, worker, n_segments, q),
            sorted(set().union(*(segment_steps(s, checkpoint_every) for s in near))))


def edge_message(chain_id: str, round_no: int, unit_id: str, step: int, kind: str, digest: bytes) -> bytes:
    """What a shard's worker signs for each tensor it sends a neighbour (`kind` as it records it: "h_out" forward,
    "g_in" back), and the receiver keeps: the edge's receipt. If the two commitments disagree about that tensor, it
    shows which side committed to something other than what crossed (SHARDING.md §3)."""
    return b"edge|" + h256(chain_id.encode(), round_no.to_bytes(8, "big"), unit_id.encode(), step.to_bytes(8, "big"),
                           kind.encode(), digest)


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
    # (step, kind as this shard recorded it) -> its neighbour's signed message naming what it sent (edge_message):
    # the proof, if their commitments disagree, of which side committed to something other than what crossed.
    received: dict = field(default_factory=dict)

    def leaves(self) -> list[bytes]:
        """Checkpoints first, then each step's four edge tensors in KINDS order (`leaf_index`)."""
        return [tensor_hash(c) for c in self.checkpoints] + [tensor_hash(t[k]) for t in self.tensors for k in KINDS]

    def root(self) -> bytes:
        return merkle_root(self.leaves())


def leaf_index(n_checkpoints: int, step: int, kind: str) -> int:
    return n_checkpoints + 4 * step + KINDS.index(kind)


def train_sharded(spec: ModelSpec, shards: list[Shard], data: Dataset, theta: np.ndarray, unit_seed: int, n_steps: int,
                  checkpoint_every: int, lr: float, batch_size: int, cheats: dict[int, str] | None = None,
                  sign: list | None = None) -> list[Trace]:
    """A unit trained as a pipeline, one worker per shard, in one process: what each worker would end up holding.
    `cheats` maps a shard to one of CHEATS or EDGE_CHEATS. Each step runs every shard's forward in order, then every
    backward in reverse, each shard updating its own weights once its backward is done; nothing else uses them
    afterwards. `sign`, one callable per shard, is what that shard's worker signs for each tensor it sends a
    neighbour: sign[k](step, kind as sent, tensor_hash) gives the message the receiving shard's trace keeps."""
    cheats, dt = cheats or {}, spec.dtype
    lr = dt(lr)
    params = [np.asarray(theta[s.span(spec)], dtype=dt).copy() for s in shards]
    traces = [Trace(s, [p.astype(np.float64)], []) for s, p in zip(shards, params)]
    noise = np.random.default_rng(unit_seed)  # only for the cheats

    def perturbed(a: np.ndarray) -> np.ndarray:
        return (a + noise.normal(0, 0.01, a.shape)).astype(dt)

    def send(k: int, to: int, step: int, kind: str, tensor: np.ndarray) -> np.ndarray:
        """Shard k hands `tensor` (as it recorded it) to shard `to`, signed; returns what `to` goes on to use."""
        if cheats.get(k) == "lies_sent":
            tensor = perturbed(tensor)
        if sign is not None:
            traces[to].received[(step, RECEIVED[kind])] = sign[k](step, kind, tensor_hash(tensor))
        if cheats.get(to) == "lies_received":
            tensor = perturbed(tensor)
        return tensor

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
                    out = perturbed(out)
                rec[k]["h_out"] = out
                h = send(k, k + 1, step, "h_out", out)
            saved.append((layers, acts, out))
        g = None
        for k in range(len(shards) - 1, -1, -1):
            s, (layers, acts, out) = shards[k], saved[k]
            if not s.last:
                rec[k]["g_out"] = g
            g, grad = _backward(s, layers, acts, out, g, y)
            if not s.first:
                if cheats.get(k) == "zero_gradient":
                    g = np.zeros_like(g)
                elif cheats.get(k) == "noisy_output" and s.last:
                    g = perturbed(g)
                rec[k]["g_in"] = g
                g = send(k, k - 1, step, "g_in", g)
            elif cheats.get(k) == "zero_gradient":
                grad = np.zeros_like(grad)
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

def proofs(leaves: list[bytes], indices) -> dict[int, list]:
    """crypto.merkle_proof for many leaves of one tree, building the tree once: a cell opens dozens of leaves."""
    levels = [[h256(b"leaf", leaf) for leaf in leaves]]
    while len(levels[-1]) > 1:
        level = levels[-1]
        if len(level) % 2:
            level.append(level[-1])
        levels.append([h256(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)])
    out = {}
    for index in indices:
        proof, i = [], index
        for level in levels[:-1]:
            proof.append((level[i ^ 1], (i ^ 1) < i))
            i //= 2
        out[index] = proof
    return out


def root_from(leaf: bytes, proof: list) -> bytes:
    """The root a Merkle proof of `leaf` leads to."""
    node = h256(b"leaf", leaf)
    for sibling, is_left in proof:
        node = h256(b"node", sibling, node) if is_left else h256(b"node", node, sibling)
    return node


def open_cell(trace: Trace, seg: int, checkpoint_every: int, leaves: list[bytes] | None = None) -> dict:
    """What a worker reveals for a challenged cell: its weights at both ends of the segment, the edge tensors of its
    steps, and Merkle proofs that all of them are what it committed to. `leaves`: the trace's, if already hashed."""
    c, leaves, n_ckpt = checkpoint_every, leaves or trace.leaves(), len(trace.checkpoints)
    steps = range(seg * c, (seg + 1) * c)
    idx = [seg, seg + 1] + [leaf_index(n_ckpt, t, k) for t in steps for k in KINDS]
    return {"before": trace.checkpoints[seg], "after": trace.checkpoints[seg + 1],
            "tensors": trace.tensors[seg * c:(seg + 1) * c], "proofs": proofs(leaves, idx)}


def proves(leaf: bytes, proof: list, root: bytes, index: int) -> bool:
    """A Merkle proof of `leaf` at position `index`. The proof's left/right steps spell out where the leaf sits, so
    they must match `index`, or a proof of one leaf would pass for another position (say, a checkpoint's hash offered
    as an edge tensor's)."""
    return all(is_left == bool((index >> level) & 1) for level, (_, is_left) in enumerate(proof)) \
        and merkle_verify(leaf, proof, root)


def _close(a: np.ndarray, claim, tolerance: float) -> bool:
    return isinstance(claim, np.ndarray) and claim.shape == a.shape and relative_distance(claim, a) <= tolerance


def cell_unproven(root: bytes, seg: int, cell: dict, n_steps: int, checkpoint_every: int) -> str | None:
    """None if every checkpoint and tensor of an opened cell is what the worker committed to, in its own slot."""
    c, n_ckpt = checkpoint_every, n_steps // checkpoint_every + 1
    steps, cell_proofs = range(seg * c, (seg + 1) * c), cell["proofs"]
    if len(cell["tensors"]) != c:
        return f"the cell has {len(cell['tensors'])} steps' tensors, not {c}"
    opened = {seg: cell["before"], seg + 1: cell["after"]}
    opened |= {leaf_index(n_ckpt, t, k): rec[k] for t, rec in zip(steps, cell["tensors"]) for k in KINDS}
    for i, value in opened.items():
        if i not in cell_proofs or not proves(tensor_hash(value), cell_proofs[i], root, i):
            return f"leaf {i} isn't what the worker committed to"
    return None


def misshapen(spec: ModelSpec, shard: Shard, cell: dict, batch_size: int) -> str | None:
    """None if an opened cell's arrays have the shapes its shard's do, so replaying it can't fail on bad input."""
    n, shapes = shard.size(spec), shard.shapes(spec, batch_size)
    for name in ("before", "after"):
        if not isinstance(cell[name], np.ndarray) or cell[name].shape != (n,) or not np.all(np.isfinite(cell[name])):
            return f"its {name} weights aren't {n} finite numbers"
    for rec in cell["tensors"]:
        for k in KINDS:
            want, got = shapes[k], rec[k]
            if (want is None) != (got is None) or (got is not None and (not isinstance(got, np.ndarray) or got.shape != want)):
                return f"its {k} isn't shaped like the tensors at its edge"
    return None


def check_cell(spec: ModelSpec, shard: Shard, data: Dataset, root: bytes, seg: int, cell: dict, unit_seed: int,
               n_steps: int, checkpoint_every: int, lr: float, batch_size: int, tolerance: float) -> str | None:
    """A checker's job: None if the cell holds up, else what's wrong. It needs this one shard's weights (the cell's
    two checkpoints) and the tensors at its edges, never the rest of the model: it replays the segment from the inputs
    the worker committed to (what its neighbours sent, which edges_agree ties to their commitments), so honest
    floating-point differences in other shards never add up here. Each step's outputs must match the replay within
    the job's tolerance, and the weights at the segment's end must pass the same test a whole-model replay uses."""
    steps = range(seg * checkpoint_every, (seg + 1) * checkpoint_every)
    unproven = cell_unproven(root, seg, cell, n_steps, checkpoint_every)
    if unproven:
        return unproven
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


def open_edges(trace: Trace, steps, leaves: list[bytes] | None = None) -> dict:
    """The hashes (with proofs) of a worker's edge tensors for `steps`, for comparing with its neighbours'."""
    leaves, n_ckpt = leaves or trace.leaves(), len(trace.checkpoints)
    idx = {(t, k): leaf_index(n_ckpt, t, k) for t in steps for k in KINDS}
    opened = proofs(leaves, idx.values())
    return {key: (leaves[i], opened[i]) for key, i in idx.items()}


def edges_agree(lower_root: bytes, lower: dict, upper_root: bytes, upper: dict, n_checkpoints: int, steps) -> str | None:
    """Two neighbouring shards' commitments must name the same tensors where they meet: what the lower one sent on is
    what the upper one took in, and the gradient the upper one sent back is what the lower one took in. A hash
    comparison, no compute. (Signed edge messages say which side lied when they don't: SHARDING.md.)"""
    for root, opened in ((lower_root, lower), (upper_root, upper)):
        for t in steps:
            for k in KINDS:
                if (t, k) not in opened or not proves(*opened[(t, k)], root, leaf_index(n_checkpoints, t, k)):
                    return f"step {t}: {k} isn't what was committed to"
    for t in steps:
        if lower[(t, "h_out")][0] != upper[(t, "h_in")][0]:
            return f"step {t}: the activations sent on aren't the ones the next shard took in"
        if lower[(t, "g_out")][0] != upper[(t, "g_in")][0]:
            return f"step {t}: the gradient taken in isn't the one the next shard sent back"
    return None


def disagreements(lower: dict, upper: dict, steps) -> list[tuple[int, str]]:
    """Where two neighbours' opened (and proven) edge hashes name different tensors: the first such step each way, as
    (step, kind its receiver recorded it as): "h_in" for activations the upper shard took in, "g_out" for a gradient
    the lower one took in. Each is settled by the receiver's signed edge message (coordinator.py)."""
    out = []
    for low, up, received in (("h_out", "h_in", "h_in"), ("g_out", "g_in", "g_out")):
        t = next((t for t in steps if lower[(t, low)][0] != upper[(t, up)][0]), None)
        if t is not None:
            out.append((t, received))
    return out


# --- a team member's reveal, and canaries made of cells ------------------------------------------------------------

@dataclass
class ShardReveal:
    """What one team member publishes for a unit after the beacon: its trace's root and the salt its commitment hid it
    with; its shard's weights where the unit started and ended (`ends`: checkpoint index -> (weights, proof)); the
    cells the beacon challenged (segment -> open_cell); and its edge hashes at the steps it and its neighbours were
    challenged on (open_edges). Arrays are None when a replay reads it from a record without them."""
    unit_id: str
    worker: str
    root: bytes
    salt: bytes
    ends: dict[int, tuple]
    cells: dict[int, dict]
    edges: dict[tuple[int, str], tuple[bytes, list]]

    @property
    def final(self) -> np.ndarray | None:
        return self.ends[max(self.ends)][0]


def open_shard(trace: Trace, unit_id: str, worker: str, root: bytes, salt: bytes, segments: list[int], steps,
               checkpoint_every: int, leaves: list[bytes] | None = None) -> ShardReveal:
    leaves = leaves or trace.leaves()
    last = len(trace.checkpoints) - 1
    ends = proofs(leaves, [0, last])
    return ShardReveal(unit_id, worker, root, salt, {i: (trace.checkpoints[i], ends[i]) for i in (0, last)},
                       {s: open_cell(trace, s, checkpoint_every, leaves) for s in segments}, open_edges(trace, steps, leaves))


def reveal_out(rv: ShardReveal, put) -> dict:
    """A reveal as the record holds it: arrays by blob id (`put(array) -> id`), hashes and proofs inline, so a replay
    can compare every edge without any array."""
    def blob(a):
        return None if a is None else put(a)
    return {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.root.hex(), "salt": rv.salt.hex(),
            "ends": {str(i): [blob(a), proof_out(p)] for i, (a, p) in sorted(rv.ends.items())},
            "cells": {str(s): {"before": blob(c["before"]), "after": blob(c["after"]),
                               "tensors": [{k: blob(t[k]) for k in KINDS} for t in c["tensors"]],
                               "proofs": {str(i): proof_out(p) for i, p in sorted(c["proofs"].items())}}
                      for s, c in sorted(rv.cells.items())},
            "edges": [[t, k, leaf.hex(), proof_out(p)]
                      for (t, k), (leaf, p) in sorted(rv.edges.items(), key=lambda e: (e[0][0], KINDS.index(e[0][1])))]}


def reveal_in(d: dict, load=None, shapes: dict | None = None) -> ShardReveal:
    """A recorded reveal. With `load` (blob id -> array), its arrays too, edge tensors reshaped to `shapes`
    (Shard.shapes); without, they're None, which is all a replay needs: the hashes it compares are in the record."""
    def blob(bid, kind=None):
        if bid is None or load is None:
            return None
        a = load(check_id(bid))
        want = (shapes or {}).get(kind)
        if a is not None and want is not None and a.size == math.prod(want):
            a = a.reshape(want)
        return a
    try:
        ends = {int(i): (blob(bid), proof_in(p)) for i, (bid, p) in d["ends"].items()}
        cells = {int(s): {"before": blob(c["before"]), "after": blob(c["after"]),
                          "tensors": [{k: blob(t[k], k) for k in KINDS} for t in c["tensors"]],
                          "proofs": {int(i): proof_in(p) for i, p in c["proofs"].items()}} for s, c in d["cells"].items()}
        edges = {(int(t), str(k)): (bytes.fromhex(leaf), proof_in(p)) for t, k, leaf, p in d["edges"]}
        return ShardReveal(str(d["unit_id"]), str(d["worker"]), bytes.fromhex(d["merkle_root"]), bytes.fromhex(d["salt"]),
                           ends, cells, edges)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise WireError(f"bad sharded reveal: {e}") from e


def plant_cell(spec: ModelSpec, cell: dict, seg: int, n_leaves: int, draw) -> tuple[dict, bytes]:
    """A canary cell: a real cell's openings with its update rescaled (verification.rescale_update, record version
    2's construction), and Merkle proofs of the tree that would hold it. Returns (cell, root). Its steps' tensors are
    the real ones and its weights start where the real cell's do, so every step replays as honest work and only the
    weights it ends at fail, as a cell whose worker faked its update would. The root is a fresh tree's, which nobody
    outside the team sees until the round is sealed."""
    forged = rescale_update(spec, cell["before"], cell["after"], draw)
    leaf = tensor_hash(forged)
    swapped = _swap_leaf(cell["proofs"], seg + 1, leaf, n_leaves)
    return dict(cell, after=forged, proofs=swapped), root_from(leaf, swapped[seg + 1])
