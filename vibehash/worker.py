"""The work a node does: train units, commit, reveal with openings, verify others.

Network-agnostic: the in-process simulation calls these directly and the HTTP
client (client.py) calls them between requests.

Behaviours other than "honest" exist to test the protocol's defences:
    lazy        skips training, fabricates checkpoints
    shortcut    trains the first half, extrapolates the rest
    freezer     skips training and resubmits the start weights as every checkpoint
    copycat    computes nothing, tries to copy another worker's reveal
    poisoner    trains honestly but reports a boosted final update
    withholder  commits to garbage, then refuses to reveal
Verifier behaviours:
    honest         replays challenged segments
    lazy           approves everything without replaying
    false_accuser  rejects honest work hoping to collect the bounty
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .crypto import KeyPair, merkle_proof, merkle_root
from .model import params_hash, train_segment
from .tasks import Commitment, Reveal, TrainingJob, WorkUnit, commit_digest
from .verification import replay_openings, required_openings


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    flops_per_sec: float
    memory_mb: int
    reliability: float = 0.99  # chance of finishing an assigned batch before the deadline (simulation only)


# Relative speeds roughly follow real fp32 throughput; absolute values are scaled for the simulation.
LOW_END_LAPTOP = HardwareProfile("Old laptop (iGPU)", 1.0e6, 4096, 0.95)
OFFICE_DESKTOP = HardwareProfile("Office desktop (CPU)", 2.0e6, 8192, 0.98)
MINI_PC = HardwareProfile("Mini PC / NUC", 1.5e6, 8192, 0.97)
GTX_1060 = HardwareProfile("GTX 1060 rig", 4.0e6, 6144, 0.99)
RTX_3060 = HardwareProfile("RTX 3060 rig", 9.0e6, 12288, 0.99)
NETBOOK = HardwareProfile("Netbook (512 MB)", 0.5e6, 512, 0.90)


@dataclass
class _Work:
    checkpoints: list[np.ndarray]
    leaves: list[bytes]
    root: bytes
    salt: bytes
    final: np.ndarray


@dataclass
class Worker:
    name: str
    keys: KeyPair
    hardware: HardwareProfile
    behavior: str = "honest"
    verifier_behavior: str = "honest"
    fp_noise: float = 1e-10  # emulates non-bit-exact hardware in simulations; 0 on real nodes
    backend: object = None  # compute backend (CPU or GPU); None means this process's active one
    blobs: object = None  # a BlobStore: keep checkpoints on disk (memory-mapped) instead of in RAM
    _work: dict[str, _Work] = field(default_factory=dict, repr=False)

    @property
    def address(self) -> str:
        return self.keys.address

    def compute(self, unit: WorkUnit, job: TrainingJob, start: np.ndarray) -> Commitment:
        rng = np.random.default_rng(int.from_bytes(self.keys.public[:8], "big") ^ unit.seed)
        c = job.checkpoint_every
        checkpoints = [start]
        theta = start

        for seg in range(job.n_segments):
            if self.behavior in ("lazy", "copycat", "withholder"):
                theta = theta + rng.normal(0, 0.01, theta.shape)
            elif self.behavior == "shortcut" and seg >= job.n_segments // 2:
                theta = theta + (checkpoints[-1] - checkpoints[-2])  # extrapolate the last real segment
            elif self.behavior == "freezer":
                pass  # passes any check that only asks "within tolerance?" once updates are smaller than it
            else:
                theta = train_segment(job.spec, job.data, theta, unit.seed, seg * c, c, job.lr, job.batch_size, self.backend)
                if self.fp_noise:
                    theta = theta * (1 + rng.normal(0, self.fp_noise, theta.shape))
            theta = job.spec.round(theta)
            if self.blobs is not None:  # big models: one checkpoint in RAM at a time
                bid = self.blobs.put_array(theta, job.wire_dtype)
                checkpoints.append(self.blobs.load(bid, job.wire_dtype, job.spec.n_params))
            else:
                checkpoints.append(theta)

        leaves = [params_hash(cp) for cp in checkpoints]
        root = merkle_root(leaves)
        salt = os.urandom(16)
        final = checkpoints[-1]
        if self.behavior == "poisoner":
            final = job.spec.round(start + 10 * (final - start))
        self._work[unit.unit_id] = _Work(checkpoints, leaves, root, salt, final)

        digest = commit_digest(unit.unit_id, self.keys.public, root, salt)
        if self.behavior == "copycat":
            digest = os.urandom(32)  # has nothing real to commit to yet
        unsigned = Commitment(unit.unit_id, self.address, self.keys.public, digest, b"")
        return Commitment(unit.unit_id, self.address, self.keys.public, digest, self.keys.sign(unsigned.message()))

    def reveal(self, unit: WorkUnit, job: TrainingJob, beacon: bytes, q: int, others: list[Reveal] = ()) -> Reveal | None:
        """Publish the result and the checkpoints the beacon picked. None means the worker withholds."""
        work = self._work.get(unit.unit_id)
        if work is None or self.behavior == "withholder":
            return None
        if self.behavior == "copycat" and others:
            src = others[0]
            return Reveal(unit.unit_id, self.address, src.merkle_root, src.salt, src.final_params.copy(), dict(src.openings))
        need = required_openings(job, beacon, unit.unit_id, self.address, q)
        openings = {i: (work.checkpoints[i], merkle_proof(work.leaves, i)) for i in need}
        return Reveal(unit.unit_id, self.address, work.root, work.salt, work.final, openings)

    def verify(self, job: TrainingJob, unit: WorkUnit, openings: dict, segments: list[int]) -> tuple[bool, int | None]:
        return replay_openings(job, unit, openings, segments, self.verifier_behavior, self.backend)

    def forget(self, unit_id: str) -> None:
        self._work.pop(unit_id, None)
