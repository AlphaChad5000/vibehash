"""Checks that a submitted work unit was really computed.

Layers, cheapest first:
  1. commit–reveal  — the reveal must match the signed commitment
  2. openings       — every checkpoint the beacon demands is present and in the Merkle tree
  3. sanity         — finite, starts at the round's model, final == last checkpoint, bounded update
  4. replay         — a verifier re-runs the challenged segments and compares
  5. redundancy     — replicated units must agree (handled in coordinator.py)
"""
from __future__ import annotations

import numpy as np

from .crypto import h256, merkle_verify, verify
from .model import params_hash, relative_distance, train_segment
from .tasks import Commitment, Reveal, TrainingJob, WorkUnit, commit_digest


def check_commitment_signature(c: Commitment) -> bool:
    return verify(c.pubkey, c.message(), c.signature)


def check_reveal(c: Commitment, r: Reveal) -> bool:
    return c.unit_id == r.unit_id and c.worker == r.worker and c.digest == commit_digest(r.unit_id, c.pubkey, r.merkle_root, r.salt)


def challenge_segments(beacon: bytes, unit_id: str, worker: str, n_segments: int, q: int) -> list[int]:
    seed = int.from_bytes(h256(beacon, unit_id.encode(), worker.encode())[:8], "big")
    # numpy doesn't promise this stream stays the same across versions, and workers, checkers and every replay must
    # agree on it; tests/test_record_v2.py pins its output, so a numpy that changed it fails the tests.
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_segments, size=min(q, n_segments), replace=False).tolist())


def required_openings(job: TrainingJob, beacon: bytes, unit_id: str, worker: str, q: int) -> list[int]:
    """Checkpoints a worker must publish: first, last, and both ends of each challenged segment."""
    need = {0, job.n_segments}
    for s in challenge_segments(beacon, unit_id, worker, job.n_segments, q):
        need.update((s, s + 1))
    return sorted(need)


def check_openings(job: TrainingJob, r: Reveal, required: list[int]) -> bool:
    n = job.spec.n_params
    for i in required:
        if i not in r.openings:
            return False
        params, proof = r.openings[i]
        if params.shape != (n,) or not merkle_verify(params_hash(params), proof, r.merkle_root):
            return False
    return True


def check_sanity(job: TrainingJob, unit: WorkUnit, start: np.ndarray, r: Reveal) -> bool:
    final = r.final_params
    if final.shape != start.shape or not np.all(np.isfinite(final)):
        return False
    if relative_distance(final, start) > job.max_update_ratio:
        return False
    # Training must start from the round's model and end where the worker says it did.
    first, last = r.openings[0][0], r.openings[job.n_segments][0]
    return params_hash(first) == unit.start_hash and np.array_equal(last, final)


# How far a claim may be from the replay, as a fraction of what the training itself changed. A job's
# `tolerance` alone isn't enough: for big or nearly trained models one segment's whole update can be
# smaller than it, and then resubmitting the start weights untouched would pass. Honest CPU <-> GPU
# differences measured under 0.1% of the update (DESIGN §8).
UPDATE_TOLERANCE = 0.01


def reproduces(claim: np.ndarray, expected: np.ndarray, start: np.ndarray, tolerance: float, spec) -> bool:
    """Whether `claim` is an honest result of training from `start`, which a replay says ends at `expected`:
    within `tolerance` of the weights, and within UPDATE_TOLERANCE of the update (no finer than one rounding
    step of the job's precision, so a segment that barely changes anything can't fail on rounding alone)."""
    if relative_distance(claim, expected) > tolerance:
        return False
    err = np.linalg.norm(claim - expected)
    return err <= max(UPDATE_TOLERANCE * np.linalg.norm(expected - start), np.finfo(spec.dtype).eps * np.linalg.norm(expected))


def replay_segment(job: TrainingJob, unit: WorkUnit, seg: int, before: np.ndarray, after: np.ndarray, backend=None) -> bool:
    c = job.checkpoint_every
    expected = train_segment(job.spec, job.data, before, unit.seed, seg * c, c, job.lr, job.batch_size, backend)
    return reproduces(after, expected, before, job.tolerance, job.spec)


# How far a canary (coordinator.py) moves one segment's update away from what the training did: 15-25%, so 15 to 25
# times UPDATE_TOLERANCE and every honest replay rejects it, yet no further than honest segments' updates vary in size
# from one segment to the next (about ±20% at batch size 32), so its size doesn't give it away. Near full-batch jobs,
# whose neighbouring segments differ by ~2%, it can still show; but so would any fake far enough off to fail safely.
CANARY_SCALE = (0.15, 0.25)


def plant_canary(job: TrainingJob, openings: dict, segments: list[int], draw) -> dict:
    """A canary's openings: a real reveal's for its challenged `segments`, with one segment's update rescaled. Version 1
    planted a lazy worker's fake, whose isotropic noise shows without a replay: honest updates to the output layer's
    biases sum to ~0 (each softmax gradient row does), noise doesn't. A rescaled honest update keeps that structure.
    The forged checkpoint ends a challenged segment and starts none, so only one segment is off. No training, so it's
    cheap for the coordinator: a vector sum and one hash. `draw` is a crypto.Draw from the round secret."""
    ends = [s for s in segments if s + 1 not in segments]
    bad = ends[draw.below(len(ends))]
    forged = rescale_update(job.spec, openings[bad][0], openings[bad + 1][0], draw)
    proofs = _swap_leaf({i: openings[i][1] for s in segments for i in (s, s + 1)}, bad + 1, params_hash(forged), job.n_segments + 1)
    return {i: (forged if i == bad + 1 else openings[i][0], proof) for i, proof in proofs.items()}


def rescale_update(spec, before: np.ndarray, after: np.ndarray, draw) -> np.ndarray:
    """`after` moved so that the update from `before` is CANARY_SCALE bigger or smaller: a canary's forged checkpoint,
    for a whole unit's segment (plant_canary) or a sharded unit's cell (shards.plant_cell)."""
    lo, hi = CANARY_SCALE
    scale, sign = lo + (hi - lo) * draw.below(2**20) / 2**20, 1 if draw.chance(0.5) else -1
    after = np.asarray(after, dtype=np.float64)
    step = after - before  # in place from here on: models can be a GiB a checkpoint
    size = np.linalg.norm(step)
    if not size:  # it didn't move at all: any direction will do
        step, size = np.ones_like(after), np.sqrt(after.size)
    # A difference under a rounding step of the job's precision passes as rounding (reproduces()), so a segment that
    # barely moved the weights is pushed at least four steps, or an honest checker would approve it and be slashed.
    step *= sign * max(scale * size, 4 * np.finfo(spec.dtype).eps * np.linalg.norm(after)) / size
    step += after
    return spec.round(step)


def _swap_leaf(proofs: dict[int, list], index: int, leaf: bytes, n_leaves: int) -> dict[int, list]:
    """Merkle proofs of the same openings once leaf `index` is replaced by `leaf`: every sibling on that leaf's path to
    the root takes its new hash, so they all still prove against one root (a new one) and a checker who compares them
    sees nothing amiss. Proofs are bottom-up (crypto.merkle_proof); `n_leaves` tells where a level was padded."""
    def on_path(i: int, level: int, width: int) -> bool:  # is i's sibling at `level` the swapped leaf's ancestor?
        j = (i >> level) ^ 1
        if j >= width:  # merkle_root pads an odd level with a copy of its last node, which is then its own sibling
            j = i >> level
        return j == index >> level

    path, node, width = [], h256(b"leaf", leaf), n_leaves
    for level, (sibling, is_left) in enumerate(proofs[index]):
        path.append(node)
        sibling = node if on_path(index, level, width) else sibling
        node = h256(b"node", sibling, node) if is_left else h256(b"node", node, sibling)
        width = (width + 1) // 2
    out = {}
    for i, proof in proofs.items():
        new, width = [], n_leaves
        for level, (sibling, is_left) in enumerate(proof):
            new.append((path[level] if level < len(path) and on_path(i, level, width) else sibling, is_left))
            width = (width + 1) // 2
        out[i] = new
    return out


def replay_cost(job: TrainingJob, n_segments: int) -> int:
    return n_segments * job.checkpoint_every * job.spec.flops_per_step(job.batch_size)


def replay_openings(job: TrainingJob, unit: WorkUnit, openings: dict, segments: list[int],
                    behavior: str = "honest", backend=None) -> tuple[bool, int | None]:
    """A verifier's job. Returns (ok, first bad segment)."""
    if behavior == "lazy":
        return True, None
    if behavior == "false_accuser":
        return False, segments[0]
    for seg in segments:
        if not replay_segment(job, unit, seg, openings[seg][0], openings[seg + 1][0], backend):
            return False, seg
    return True, None


def full_replay(job: TrainingJob, unit: WorkUnit, start: np.ndarray, backend=None) -> np.ndarray:
    """Arbitration: recompute the entire unit, checkpoint by checkpoint as a worker does.
    Expensive, only used on disputes."""
    theta, c = start, job.checkpoint_every
    for seg in range(job.n_segments):
        theta = job.spec.round(train_segment(job.spec, job.data, theta, unit.seed, seg * c, c, job.lr, job.batch_size, backend))
    return theta


def detection_probability(fake_fraction: float, q: int) -> float:
    """Chance a replay spot-check of q segments hits at least one faked segment."""
    return 1 - (1 - fake_fraction) ** q
