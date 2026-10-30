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
