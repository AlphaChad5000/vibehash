"""In-process network driver for simulations and tests.

Runs the same Coordinator a real node runs, but with simulated workers called
directly and a simulated clock, so hundreds of rounds take seconds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .coordinator import Coordinator, CoordinatorConfig, RoundReport
from .crypto import KeyPair
from .ledger import SYSTEM, Chain, Tx
from .blobs import array_id
from .tasks import Reveal, TrainingJob
from .wire import blob_openings_out, seal
from .worker import Worker


class Checkpoints:
    """Every checkpoint revealed in simulated rounds, by blob id: what a node keeps for its challenge window,
    so simulated rounds can be rechecked like a node's (replay.py)."""

    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}

    def put(self, arr: np.ndarray, job: TrainingJob) -> str:
        bid = array_id(arr, job.wire_dtype)
        self.arrays[bid] = np.asarray(arr, dtype=np.float64)
        return bid

    def array(self, bid: str, job: TrainingJob) -> np.ndarray:
        return self.arrays[bid]

    def attach(self, job: TrainingJob) -> None:
        pass  # simulated jobs use synthetic data, which every replay rebuilds from the job's seed


class Protocol:
    def __init__(self, chain: Chain, workers: list[Worker], config: CoordinatorConfig | None = None, seed: int = 0):
        self.coord = Coordinator(chain, None, config)
        self.workers = {w.address: w for w in workers}
        self.rng = np.random.default_rng(seed + 1)  # simulated outages only; the coordinator draws from the chain
        self.checkpoints = Checkpoints()

    def _wire(self, rv: Reveal, job: TrainingJob) -> dict:
        """A reveal as a node records it: checkpoints by blob id."""
        put = lambda a: self.checkpoints.put(a, job)  # noqa: E731
        return {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(), "salt": rv.salt.hex(),
                "final": put(rv.final_params), "openings": blob_openings_out({i: (put(a), pr) for i, (a, pr) in rv.openings.items()})}

    def _signed(self, w: Worker, kind: str, body: dict) -> dict:
        """The envelope a real worker would send, so simulated rounds replay like a node's."""
        return seal(w.keys, kind, body, self.chain.chain_id, now=self.chain.tip.height + 1)

    @property
    def chain(self) -> Chain:
        return self.coord.chain

    @property
    def config(self) -> CoordinatorConfig:
        return self.coord.config

    @property
    def history(self) -> list[RoundReport]:
        return self.coord.history

    @property
    def reputation(self) -> dict[str, float]:
        return {a: self.chain.state.rep(a) for a in self.workers}

    def eligible(self, addr: str) -> bool:
        return self.coord.eligible(addr)

    def _proposer(self, seed: bytes) -> KeyPair:
        stakers = sorted(a for a in self.workers if self.chain.state.staked(a))  # not those still earning in
        weights = np.array([self.chain.state.stake(a) for a in stakers], dtype=float)
        pick = np.random.default_rng(int.from_bytes(seed[:8], "big")).choice(len(stakers), p=weights / weights.sum())
        return self.workers[stakers[pick]].keys

    def post_job(self, requester: KeyPair, job: TrainingJob) -> None:
        """Submit the job and seal it into a block straight away."""
        tx = Tx.signed(requester, "post_job", SYSTEM, job.budget, self.chain.next_nonce(requester.address),
                       job.spec_json(), self.chain.chain_id)
        self.coord.add_job(job)
        self.coord.submit_tx(tx)
        self.coord.keys = self._proposer(self.chain.tip.hash)
        self.coord.seal_pending(timestamp=self.chain.tip.height + 1)
        self.coord.keys = None

    def submit(self, tx: Tx) -> None:
        self.coord.submit_tx(tx)

    def run_round(self, job_id: str | None = None) -> RoundReport:
        c = self.coord
        if not c.open_round():
            raise RuntimeError("no active job with escrow")
        r = c.r
        job = c.job(r.job_id)
        for addr, w in self.workers.items():
            if self.eligible(addr):
                body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb}
                c.register(addr, w.hardware.flops_per_sec, w.hardware.memory_mb, envelope=self._signed(w, "register", body))

        c.start_compute()
        for addr, unit_ids in list(r.assignments.items()):
            w = self.workers[addr]
            if self.rng.random() > w.hardware.reliability:
                continue  # dropped offline / ran out of time
            for uid in unit_ids:
                c.commit(w.compute(r.units[uid], job, r.start))

        beacon = c.start_reveal()
        published: dict[str, list[Reveal]] = defaultdict(list)
        # Honest workers reveal first so a copycat has something to try copying.
        keys = sorted(r.commitments, key=lambda k: self.workers[k[1]].behavior == "copycat")
        for uid, addr in keys:
            rv = self.workers[addr].reveal(r.units[uid], job, beacon, self.config.challenge_segments, published[uid])
            if rv is not None:
                c.reveal(rv, wire=self._wire(rv, job))
                published[uid].append(rv)

        c.start_verify()
        for addr, w in self.workers.items():
            for t in c.tasks_for(addr):
                ok, bad = w.verify(job, t.unit, t.openings, t.segments)
                body = {"round": r.round_no, "task_id": t.task_id, "ok": bool(ok), "bad_segment": bad}
                c.submit_verdict(addr, t.task_id, ok, bad, envelope=self._signed(w, "verdict", body))

        report = c.finalize(proposer_keys=self._proposer(beacon), timestamp=report_ts(self.chain))
        for w in self.workers.values():
            for uid in list(w._work):
                w.forget(uid)
        return report


def report_ts(chain: Chain) -> int:
    return chain.tip.height + 1
