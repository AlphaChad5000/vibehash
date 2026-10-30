"""The round state machine: register → compute/commit → beacon/reveal → verify → finalize.

Transport-agnostic. The simulation drives it with in-process workers; node.py
drives it from a wall clock and exposes each step over HTTP. Callers are
responsible for authenticating *who* is calling (node.py checks signatures);
everything else — eligibility, phase, assignment, commitments, Merkle proofs —
is checked here.

Every choice it makes follows from the chain and the round's own signed
messages, never from its own clock: which job runs, how much work each worker
may take, which units each gets and which are secretly duplicated, who checks
whom, and the planted fakes. Its one private input is a secret the leader
commits to when it opens the round and reveals in the record, so checkers can't
work out which of their tasks are planted fakes. Each block commits to a
record of those messages, plus the results of the checks that need the model's
weights, so anyone can re-run the round with this same code (replay.py) and
must reach the same summary.
"""
from __future__ import annotations

import functools
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import economics as econ
from . import fees
from .crypto import Draw, KeyPair, address_of, h256
from .ledger import Block, Chain, InvalidBlock, RoundSummary, Tx, canonical
from .model import evaluate, params_hash
from .tasks import Commitment, InvalidJob, Reveal, TrainingJob, Verdict, WorkUnit
from .verification import (
    challenge_segments,
    check_commitment_signature,
    check_openings,
    check_reveal,
    check_sanity,
    full_replay,
    plant_canary,
    replay_cost,
    replay_openings,
    replay_segment,
    reproduces,
    required_openings,
)
from .wire import WireError, blob_openings_in, commitment_out
from .worker import HardwareProfile, Worker

# Records say which rules made them, and a replay follows the record's version, so every round still replays exactly as
# it was run. 1 (0.4.0-0.5.1): the beacon, who checks whom, task ids and canaries all came from public data. 2: they
# also hash in a secret the leader commits to when it opens the round, and canaries are copies of real work.
RECORD_VERSION = 2
RAMP = 2  # a worker's capacity can grow this much each round it delivers in full, up to what it says it can do
LOOKBACK = 50  # blocks searched for a worker's most recent round


def secret_commitment(secret: bytes) -> bytes:
    """What a leader publishes of its round secret when it opens the round (/status `beacon_commit`, and the record)."""
    return h256(b"round secret", secret)


class ProtocolError(Exception):
    """A message was rejected (wrong phase, not assigned, malformed). Not proof of fraud."""


class Phase(str, Enum):
    IDLE = "idle"
    REGISTER = "register"
    COMPUTE = "compute"
    REVEAL = "reveal"
    VERIFY = "verify"


@dataclass
class CoordinatorConfig:
    replication_rate: float = 0.2  # fraction of units secretly given to a second worker
    challenge_segments: int = 3  # q
    canaries_per_round: int = 2  # known-fake submissions slipped to verifiers
    agreement_tolerance: float = 1e-4  # whole-unit comparisons (noise compounds over a unit)
    compute_s: float = 10.0  # length of the compute phase; sizes each worker's capacity
    newcomer_units: int = 4  # cap for workers with no recent delivered work on chain, and for those still earning in
    max_newcomers: int = 64  # workers without stake (earning it in) admitted per round, so they can't crowd the round
    max_mempool: int = 10_000
    min_fee: int = 0  # syn a tx must pay to be relayed; `vibehash node` sets fees.DEFAULT_MIN_FEE

    def rules(self) -> dict:
        """The settings a round's outcome depends on, recorded with it so a replay uses the same ones."""
        return {"replication_rate": self.replication_rate, "challenge_segments": self.challenge_segments,
                "canaries_per_round": self.canaries_per_round, "agreement_tolerance": self.agreement_tolerance,
                "compute_s": self.compute_s, "newcomer_units": self.newcomer_units, "max_newcomers": self.max_newcomers}


@dataclass
class Registration:
    address: str
    flops_per_sec: float
    memory_mb: int


@dataclass
class VerifyTask:
    task_id: str
    verifier: str
    unit: WorkUnit
    submitter: str
    segments: list[int]
    openings: dict[int, tuple[np.ndarray, list]]
    canary: bool = False
    verdict: tuple[bool, int | None] | None = None


@dataclass
class UnitResult:
    unit: WorkUnit
    worker: str
    verdict: Verdict
    verifier: str | None = None
    detail: str = ""


@dataclass
class RoundReport:
    round: int
    job_id: str
    results: list[UnitResult]
    worker_flops: dict[str, int]
    verifier_flops: dict[str, int]
    slashes: list[list]
    unassigned: int
    val_loss: float
    val_acc: float
    proposer: str
    block: Block
    pool: int = 0
    burned: int = 0
    worker_pay: dict[str, int] = field(default_factory=dict)
    verifier_pay: dict[str, int] = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    params: np.ndarray | None = None  # the weights the round trained


@dataclass
class _Round:
    round_no: int
    job_id: str
    start: np.ndarray | None  # None while replaying: replays don't need the weights
    start_hash: bytes
    units: dict[str, WorkUnit]
    record: dict = field(default_factory=dict)
    regs: dict[str, Registration] = field(default_factory=dict)
    reg_entries: dict[str, dict] = field(default_factory=dict)  # each registration as the record holds it
    assignments: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    holders: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    commitments: dict[tuple[str, str], Commitment] = field(default_factory=dict)
    secret: bytes = b""  # record version 2's round secret (open_round); empty when replaying a version 1 record
    beacon: bytes = b""
    reveals: dict[tuple[str, str], Reveal] = field(default_factory=dict)
    results: dict[tuple[str, str], UnitResult] = field(default_factory=dict)
    slashes: list[list] = field(default_factory=list)
    forfeit: set[str] = field(default_factory=set)  # withheld a reveal: unpaid this round, not slashed
    tasks: dict[str, VerifyTask] = field(default_factory=dict)
    unassigned: int = 0


def _mean(arrays: list, job: TrainingJob) -> np.ndarray:
    """The new model: accepted results summed in float64 in round order, divided, rounded to the job's precision.
    Pinned down exactly, because anyone rechecking the round must get the same bits: numpy's own mean works in
    whatever precision the arrays arrived in (float32 from a node's disk, float64 elsewhere)."""
    total = np.zeros(job.spec.n_params, dtype=np.float64)
    for a in arrays:
        total += np.asarray(a, dtype=np.float64)
    return job.spec.round(total / len(arrays))


def _plain(ok, bad) -> list:
    """A replay's (passed, first bad segment) as JSON can hold it: numpy comparisons give numpy bools."""
    return [bool(ok), None if bad is None else int(bad)]


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)
    return wrapper


class Coordinator:
    def __init__(self, chain: Chain, keys: KeyPair | None = None, config: CoordinatorConfig | None = None,
                 store=None, replay: dict | None = None, recheck=None):
        """With `replay` (a round record), runs that recorded round instead of a live one: see replay.py. With
        `recheck` too (something with .array(blob_id, job) and .attach(job)), it also re-runs the checks that need
        the weights wherever their checkpoints are still available, and lists any result the record got wrong."""
        self.chain = chain
        self.keys = keys
        self.config = config or CoordinatorConfig()
        self.store = store
        self.replaying = replay
        self.recheck = recheck if replay else None
        self.timing = None  # job -> seconds of compute a live round of it gets (node.py); None: config.compute_s
        self.disputes: list[str] = []  # recorded results a recheck contradicted
        self.rechecked = self.unchecked = 0
        self.jobs: dict[str, TrainingJob] = {}
        self.params: dict[str, np.ndarray] = {}
        self.mempool: list[Tx] = []
        self.history: list[RoundReport] = []
        self.phase = Phase.IDLE
        self.r: _Round | None = None
        self.lock = threading.RLock()

    # --- jobs and transactions -------------------------------------------------

    def add_job(self, job: TrainingJob) -> None:
        """Register a job object (with its dataset attached) ahead of its post_job tx landing on chain."""
        self.jobs[job.job_id] = job

    def job(self, job_id: str) -> TrainingJob:
        if job_id not in self.jobs:
            job = self.chain.state.job(job_id)
            if job.data_source["kind"] == "npz" and self.recheck is not None:
                try:
                    self.recheck.attach(job)  # rechecking replays trains on the data
                except Exception:  # noqa: BLE001 - unavailable data only means those checks can't be redone
                    pass
            elif job.data_source["kind"] == "npz" and not self.replaying:  # plain replays never touch the data
                if self.store is None:
                    raise InvalidJob("no dataset store")
                self.store.attach_dataset(job)
            self.jobs[job_id] = job
        return self.jobs[job_id]

    def start_params(self, job_id: str) -> np.ndarray:
        if job_id not in self.params:
            saved = self.store.load_params(job_id) if self.store else None
            job = self.job(job_id)
            self.params[job_id] = saved if saved is not None else job.spec.init_params(job.init_seed)
        return self.params[job_id]

    def _summaries(self, lookback: int | None = LOOKBACK):
        """Round summaries on chain, newest first."""
        blocks = self.chain.blocks if lookback is None else self.chain.blocks[-lookback:]
        return (b.summary for b in reversed(blocks) if b.summary is not None)

    def active_job_id(self) -> str | None:
        """Jobs with budget left take turns a round at a time, so a new job never waits for an old one to run out.
        The turn order comes from the chain: the job after the one the last round trained."""
        ready = []
        for job_id, escrow in self.chain.state.escrow.items():
            if escrow > 0:
                try:
                    self.job(job_id)
                    ready.append(job_id)
                except (InvalidJob, ValueError, OSError):
                    continue  # dataset not available here yet
        if not ready:
            return None
        last = next((s.job_id for s in self._summaries()), None)
        return ready[(ready.index(last) + 1) % len(ready)] if last in ready else ready[0]

    def expected_start_hash(self, job: TrainingJob) -> bytes:
        """A round trains the job's model as its last round left it, or the starting weights the spec fixes."""
        s = next((s for s in self._summaries(None) if s.job_id == job.job_id), None)
        return bytes.fromhex(s.params_hash) if s else params_hash(job.spec.init_params(job.init_seed))

    def track_record(self, addr: str) -> tuple[int, bool] | None:
        """(FLOPs of accepted work, whether it missed a deadline) in the worker's most recent round on chain, or None
        if it hasn't taken part lately. Withholding or being slashed counts as delivering nothing."""
        for s in self._summaries():
            if addr in s.offenders:
                return 0, True
            if addr in s.worker_flops or addr in s.missed:
                return s.worker_flops.get(addr, 0), addr in s.missed
        return None

    def capacity(self, addr: str, claimed_flops: int, job: TrainingJob) -> int:
        """How much work a worker may take this round. Like network congestion control: up to RAMP times what it
        delivered last round if it delivered everything, exactly what it delivered if it missed a deadline, the
        newcomer cap if it's new (or withheld, or was slashed), and never more than it says it can do."""
        newcomer = self.config.newcomer_units * job.unit_flops
        record = self.track_record(addr)
        if record is None or not self.chain.state.staked(addr):  # new, or still earning its stake in
            return min(claimed_flops, newcomer)
        delivered, missed = record
        return min(claimed_flops, max(newcomer, delivered if missed else RAMP * delivered))

    @_locked
    def submit_tx(self, tx: Tx) -> None:
        """Admit a tx to the mempool, which is kept in fee order (fees.py). When it's full, a tx only gets in by
        paying more than the cheapest one that nothing else depends on, which it then replaces."""
        if tx.fee < self.config.min_fee:
            raise ProtocolError(f"fee too low: this node relays transactions paying at least {self.config.min_fee} syn")
        pool = self.mempool
        if len(pool) >= self.config.max_mempool:
            cheapest = fees.evictable(pool)
            if cheapest is None or tx.fee <= cheapest.fee:
                raise ProtocolError(f"mempool full: only transactions paying more than {cheapest.fee if cheapest else 0} syn get in")
            pool = [t for t in pool if t is not cheapest]
        try:
            self.chain.check_tx(tx, pool)
        except (InvalidBlock, InvalidJob) as e:
            raise ProtocolError(f"rejected transaction: {e}") from e
        self.mempool = fees.by_fee([*pool, tx])

    def eligible(self, addr: str) -> bool:
        """May take work: staked, or on a chain where workers earn their stake in."""
        return self.chain.state.staked(addr) or self.chain.state.earn_in

    def _need(self, phase: Phase) -> _Round:
        if self.phase != phase or self.r is None:
            raise ProtocolError(f"not in {phase.value} phase (now {self.phase.value})")
        return self.r

    def _claim(self, kind: str, key: str, compute):
        """Results of checks that need the model's weights: computed live and written to the record, read back
        from it on replay. A recheck computes them again and notes any disagreement, but carries on with the
        recorded result, so the rest of the replay still follows the block it's checking."""
        if self.replaying:
            try:
                value = self.replaying["checks"][kind][key]
            except (KeyError, TypeError):
                raise ProtocolError(f"the record has no {kind} result for {key}") from None
            if self.recheck is not None:
                try:
                    again = compute()
                except (TypeError, AttributeError, ValueError, InvalidJob):  # its checkpoints or data are gone
                    self.unchecked += 1
                else:
                    self.rechecked += 1
                    if again != value:
                        self.disputes.append(f"{kind} check for {key}: the record says {value!r}, re-running it gives {again!r}")
        else:
            value = compute()
        self.r.record["checks"][kind][key] = value
        return value

    @staticmethod
    def _weights(*arrays) -> None:
        """Checks that need the weights call this first: on a recheck, a pruned checkpoint means "can't redo it",
        never a failed check. (A live round always has them.)"""
        if any(a is None for a in arrays):
            raise ValueError("those checkpoints aren't available any more")

    def _fetch(self, blob_id: str | None, job: TrainingJob) -> np.ndarray | None:
        if not blob_id:
            return None
        try:
            return self.recheck.array(blob_id, job)
        except Exception:  # noqa: BLE001 - pruned, never uploaded, or the node is unreachable: all mean "gone"
            return None

    def _replay_start(self, job: TrainingJob, start_hash: bytes) -> np.ndarray | None:
        """The round's starting weights: every honest reveal's first checkpoint is them, and the record has their hash."""
        for d in self.replaying.get("reveals", []):
            arr = self._fetch(((d.get("openings") or {}).get("0") or [None])[0], job)
            if arr is not None and params_hash(arr) == start_hash:
                return arr
        return None

    def _replay_reveal(self, job: TrainingJob, wire: dict) -> Reveal | None:
        """A recorded reveal with its checkpoints loaded, or None if any of them is gone."""
        try:
            openings = blob_openings_in(wire["openings"], job.n_segments)
        except (KeyError, WireError):
            return None
        final = self._fetch(wire.get("final"), job)
        arrays = {i: (self._fetch(bid, job), proof) for i, (bid, proof) in openings.items()}
        if final is None or any(a is None for a, _ in arrays.values()):
            return None
        return Reveal(str(wire["unit_id"]), str(wire["worker"]), bytes.fromhex(wire["merkle_root"]), bytes.fromhex(wire["salt"]), final, arrays)

    # --- 1. register -------------------------------------------------------------

    @_locked
    def open_round(self) -> bool:
        if self.phase != Phase.IDLE:
            raise ProtocolError("round already open")
        round_no = self.chain.tip.height + 1
        if self.replaying:
            job_id = self.replaying["job_id"]
            if self.chain.state.escrow.get(job_id, 0) <= 0:
                raise ProtocolError(f"job {job_id} had no budget left")
            if self.replaying["round"] != round_no:
                raise ProtocolError(f"the record is for round {self.replaying['round']}, not {round_no}")
            if self.replaying.get("version", RECORD_VERSION) not in (1, 2):
                raise ProtocolError(f"the record is version {self.replaying.get('version')!r}, which this code doesn't know")
        else:
            job_id = self.active_job_id()
            if job_id is None:
                return False
        job = self.job(job_id)
        if self.replaying:
            start, start_hash = None, bytes.fromhex(self.replaying["start_hash"])
            if start_hash != self.expected_start_hash(job):
                raise ProtocolError("the round didn't start from the job's latest model")
            if self.recheck is not None:
                start = self._replay_start(job, start_hash)
        else:
            start = self.start_params(job_id)
            start_hash = params_hash(start)
        units = [WorkUnit.make(job, round_no, i, start_hash) for i in range(job.units_per_round)]
        self.r = _Round(round_no, job_id, start, start_hash, {u.unit_id: u for u in units})
        self.r.record = {"version": RECORD_VERSION, "round": round_no, "job_id": job_id, "start_hash": start_hash.hex(),
                         "rules": self.config.rules(), "registrations": [], "commitments": [], "reveals": [], "verdicts": [],
                         "checks": {"reveal": {}, "replay": {}, "arbitration": {}, "redundancy": {}, "model": {}}}
        # Names this attempt at the round, for receipts (receipts.py): a round abandoned and run again under the
        # same number gets a new label, so receipts from the abandoned attempt don't look like censorship.
        if self.replaying:
            self.r.record["version"] = self.replaying.get("version", RECORD_VERSION)
            if "attempt" in self.replaying:
                self.r.record["attempt"] = self.replaying["attempt"]
            # The rules exactly as recorded: a record made before a rule existed must hash the same on replay.
            self.r.record["rules"] = dict(self.replaying.get("rules", self.r.record["rules"]))
        else:
            self.r.record["attempt"] = os.urandom(8).hex()
            if self.timing is not None:  # the compute phase this job's round actually gets (node.py)
                self.r.record["rules"]["compute_s"] = float(self.timing(job))
        if self.r.record["version"] >= 2:
            # The round's secret: hashed into the beacon (start_reveal), who checks whom, the task ids and the canaries
            # (start_verify). Its hash is public from now on (/status `beacon_commit`), so the leader can't pick it
            # once it has seen the commitments. The secret itself goes into the record only when the round is settled
            # (_settle), and records are only served once their block is sealed.
            if self.replaying:
                secret = bytes.fromhex(self.replaying["secret"])
                if secret_commitment(secret).hex() != self.replaying.get("beacon_commit"):
                    raise ProtocolError("the record's round secret doesn't match the commitment the round opened with")
            else:
                secret = os.urandom(32)
            self.r.secret = secret
            self.r.record["beacon_commit"] = secret_commitment(secret).hex()
        self.phase = Phase.REGISTER
        return True

    @_locked
    def register(self, addr: str, flops_per_sec: float, memory_mb: int, envelope: dict | None = None) -> dict:
        """`envelope` is the worker's signed message, kept in the record as proof it registered.
        Returns the record entry that counts, which is the first one if the message was retried."""
        r = self._need(Phase.REGISTER)
        if addr in r.reg_entries:
            return r.reg_entries[addr]
        if not self.eligible(addr):
            raise ProtocolError(f"stake at least {econ.MIN_STAKE} syn to take work")
        if not (0 < flops_per_sec < 1e18 and 0 < memory_mb < 1e9):
            raise ProtocolError("implausible hardware report")
        staked = self.chain.state.staked
        if not staked(addr) and sum(not staked(a) for a in r.regs) >= self.config.max_newcomers:
            raise ProtocolError("this round has all the workers without stake it takes; try the next one")
        r.regs[addr] = Registration(addr, float(flops_per_sec), int(memory_mb))
        entry = envelope if envelope is not None else {"address": addr, "flops_per_sec": float(flops_per_sec), "memory_mb": int(memory_mb)}
        r.record["registrations"].append(entry)
        r.reg_entries[addr] = entry
        return entry

    # --- 2. compute + commit ----------------------------------------------------------

    @_locked
    def start_compute(self) -> None:
        r = self._need(Phase.REGISTER)
        job = self.job(r.job_id)
        # Seeded by the previous block and every signed registration: fixed once registration closes,
        # unknowable before it, and the same for anyone who replays the round.
        draw = Draw(b"assign", self.chain.tip.hash, r.round_no.to_bytes(8, "big"),
                    *sorted(canonical(e) for e in r.record["registrations"]))
        compute_s = r.record["rules"]["compute_s"]  # this round's, which a job sized for big models makes longer
        capacity = {a: self.capacity(a, int(reg.flops_per_sec * compute_s), job) for a, reg in r.regs.items()}
        used = {a: 0 for a in capacity}
        order = draw.shuffle(sorted(capacity))

        staked = self.chain.state.staked

        def pick(unit: WorkUnit, exclude: set[str]) -> str | None:
            # Staked workers first: those earning their stake in only get what's left, so they can't crowd them out.
            fits = [a for a in order if a not in exclude and r.regs[a].memory_mb >= unit.memory_mb
                    and used[a] + unit.flops <= capacity[a]]
            return min(fits, key=lambda a: (not staked(a), (used[a] + unit.flops) / capacity[a])) if fits else None

        for unit in r.units.values():
            holders = []
            first = pick(unit, set())
            if first is None:
                r.unassigned += 1
                continue
            holders.append(first)
            if draw.chance(self.config.replication_rate):
                second = pick(unit, {first})
                if second is not None:
                    holders.append(second)
            for a in holders:
                r.assignments[a].append(unit.unit_id)
                r.holders[unit.unit_id].append(a)
                used[a] += unit.flops
        self.phase = Phase.COMPUTE

    @_locked
    def assignment(self, addr: str) -> list[WorkUnit]:
        r = self._need(Phase.COMPUTE)
        return [r.units[u] for u in r.assignments.get(addr, [])]

    @_locked
    def commit(self, c: Commitment) -> None:
        r = self._need(Phase.COMPUTE)
        key = (c.unit_id, c.worker)
        if c.unit_id not in r.assignments.get(c.worker, []):
            raise ProtocolError("unit not assigned to this worker")
        if key in r.commitments:
            raise ProtocolError("already committed")
        if address_of(c.pubkey) != c.worker or not check_commitment_signature(c):
            raise ProtocolError("bad commitment signature")
        r.commitments[key] = c
        r.record["commitments"].append(commitment_out(c))

    # --- 3. beacon + reveal -------------------------------------------------------------

    @_locked
    def start_reveal(self) -> bytes:
        r = self._need(Phase.COMPUTE)
        for addr, unit_ids in r.assignments.items():
            for uid in unit_ids:
                if (uid, addr) not in r.commitments:
                    r.results[(uid, addr)] = UnitResult(r.units[uid], addr, Verdict.MISSED_DEADLINE)
        # Unknowable until every commitment is in, so nobody can pick work that dodges the check. Version 2 hashes in the
        # round secret too: a worker who knew every commitment (alone in a round, say) could otherwise re-salt its own
        # until the challenge missed the segments it faked. A leader that mines in its own rounds knows the secret, so
        # it still can; only other leaders' rounds are safe from it.
        secret = (r.secret,) if r.secret else ()
        r.beacon = h256(self.chain.tip.hash, r.round_no.to_bytes(8, "big"), *secret, *sorted(c.digest for c in r.commitments.values()))
        self.phase = Phase.REVEAL
        return r.beacon

    @_locked
    def reveal(self, rv: Reveal, wire: dict | None = None) -> Verdict | None:
        """Cheap checks happen immediately. Returns a fraud verdict, or None if it goes on to replay.
        `wire` is the reveal as sent (with its checkpoints' blob ids), kept in the record."""
        r = self._need(Phase.REVEAL)
        key = (rv.unit_id, rv.worker)
        if key not in r.commitments:
            raise ProtocolError("no commitment for this unit")
        if key in r.reveals or key in r.results:
            raise ProtocolError("already revealed")
        job = self.job(r.job_id)
        unit = r.units[rv.unit_id]
        r.record["reveals"].append(wire if wire is not None else
                                   {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(), "salt": rv.salt.hex()})
        if self.recheck is not None and wire is not None:  # rechecking: load the checkpoints it points to
            rv = self._replay_reveal(job, wire) or rv
        need = required_openings(job, r.beacon, rv.unit_id, rv.worker, self.config.challenge_segments)
        if not check_reveal(r.commitments[key], rv):
            return self._fraud(key, Verdict.BAD_REVEAL, "reveal doesn't match commitment")

        def checks() -> str:
            self._weights(rv.final_params)
            if not check_openings(job, rv, need):
                return "bad_openings"
            self._weights(r.start)
            return "ok" if check_sanity(job, unit, r.start, rv) else "failed_sanity"
        outcome = self._claim("reveal", f"{rv.unit_id}/{rv.worker}", checks)
        if outcome == "bad_openings":
            return self._fraud(key, Verdict.BAD_REVEAL, "missing or unproven checkpoint")
        if outcome == "failed_sanity":
            return self._fraud(key, Verdict.FAILED_SANITY, "wrong start/final or oversized update")
        if self.replaying and rv.final_params is None:  # the record vouches the challenged checkpoints were there
            rv = Reveal(rv.unit_id, rv.worker, rv.merkle_root, rv.salt, None, {i: (None, []) for i in need})
        r.reveals[key] = rv
        return None

    def _fraud(self, key, verdict: Verdict, detail: str, catcher: str | None = None, verifier: str | None = None) -> Verdict:
        r = self.r
        r.results[key] = UnitResult(r.units[key[0]], key[1], verdict, verifier, detail)
        r.slashes.append([key[1], catcher, verdict.value])
        return verdict

    # --- 4. verify ------------------------------------------------------------------------

    @_locked
    def start_verify(self) -> None:
        r = self._need(Phase.REVEAL)
        job = self.job(r.job_id)
        q = self.config.challenge_segments
        for key in r.commitments:
            if key not in r.reveals and key not in r.results:
                # Not slashed: honest rigs crash or drop offline mid-round. Withholding selectively
                # (after seeing the challenge) still loses: the whole round's pay is forfeited,
                # and P(caught) >= fraction faked, so it only "pays" if compute costs more than it earns.
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.WITHHELD, None, "committed but never revealed")
                r.forfeit.add(key[1])
        offenders = {s[0] for s in r.slashes}
        # Version 1 seeded who checks whom, the task ids and the canaries with the beacon alone. It's public, so a checker
        # could work out which of its tasks were canaries (their ids, keys and place in the shuffle) without replaying
        # anything. Version 2 adds the round secret, which nobody but the leader sees until the block is sealed.
        seed = (r.beacon, r.secret) if r.secret else (r.beacon,)
        draw = Draw(b"verify", *seed)

        def pool(exclude: set[str]) -> list[str]:
            # Only staked workers check others: a worker still earning in has little to lose by waving work
            # through, and a crowd of them could otherwise check each other's fakes.
            return [a for a, reg in sorted(r.regs.items()) if a not in exclude and a not in offenders
                    and reg.memory_mb >= job.memory_mb and self.chain.state.staked(a)]

        def task_id(uid, submitter, verifier) -> str:
            return h256(*seed, uid.encode(), submitter.encode(), verifier.encode()).hex()[:20]

        tasks, checkers = [], defaultdict(set)
        for (uid, addr), rv in r.reveals.items():
            segs = challenge_segments(r.beacon, uid, addr, job.n_segments, q)
            openings = {i: rv.openings[i] for s in segs for i in (s, s + 1)}
            candidates = pool(set(r.holders[uid]))
            verifier = candidates[draw.below(len(candidates))] if candidates else ""
            checkers[uid].add(verifier)
            tasks.append(VerifyTask(task_id(uid, addr, verifier), verifier, r.units[uid], addr, segs, openings))

        if r.secret:
            tasks += self._canaries(job, draw, pool, task_id, checkers)
        else:  # version 1's: a lazy worker's fake of a unit, which anyone who had the beacon could pick out
            candidates = pool(set())
            unit_list = list(r.units.values())
            for n in range(self.config.canaries_per_round if candidates else 0):
                fake = Worker("canary", KeyPair.generate(r.beacon + bytes([n])), HardwareProfile("canary", 1, 1), "lazy")
                unit = unit_list[n % len(unit_list)]
                segs = challenge_segments(r.beacon, unit.unit_id, fake.address, job.n_segments, q)
                openings = {}
                if r.start is not None and not self.replaying:  # a replay needs who checks the fake, not the fake itself
                    fake.compute(unit, job, r.start)
                    rv = fake.reveal(unit, job, r.beacon, q)
                    openings = {i: rv.openings[i] for s in segs for i in (s, s + 1)}
                verifier = candidates[draw.below(len(candidates))]
                tasks.append(VerifyTask(task_id(unit.unit_id, fake.address, verifier), verifier, unit, fake.address, segs, openings, canary=True))

        r.tasks = {tasks[i].task_id: tasks[i] for i in draw.shuffle(range(len(tasks)))}
        self.phase = Phase.VERIFY

    def _canaries(self, job: TrainingJob, draw: Draw, pool, task_id, checkers: dict[str, set[str]]) -> list[VerifyTask]:
        """Version 2's planted fakes, each a copy of a real reveal's task with one segment's update rescaled
        (verification.plant_canary): it fails a replay, but its checkpoints are honest training's, so it can't be
        picked out without one. Each also goes to a checker that could have been given the real task: never one that
        holds the unit (real tasks never go to their unit's holders) or checks a real task on it (it would see the same
        checkpoints twice, one set changed), and at most one canary a unit. So a round needs at least three staked
        workers for a canary: a holder, its checker and a third. A replay only needs who checks which canary and its
        segments, which follow from the record; the planted checkpoints are made live only, and never replayed.
        Residual tell: a checker given two tasks on one unit (both replicas of it) knows neither is a canary; those
        are the units whose replicas are compared with each other anyway."""
        r, q = self.r, self.config.challenge_segments
        planted, used = [], set()
        for uid, addr in draw.shuffle(sorted(r.reveals)):
            if len(planted) == self.config.canaries_per_round:
                break
            candidates = [] if uid in used else pool(set(r.holders[uid]) | checkers[uid])
            if not candidates:
                continue
            used.add(uid)
            n = len(planted).to_bytes(8, "big")
            fake = KeyPair.generate(r.secret + n).address
            segs = challenge_segments(r.beacon, uid, addr, job.n_segments, q)  # the real task's, as a checker would expect
            verifier = candidates[draw.below(len(candidates))]
            openings = {}
            if not self.replaying:  # no training, so cheap under the lock: a vector sum and one hash
                openings = plant_canary(job, r.reveals[(uid, addr)].openings, segs, Draw(b"canary", r.secret, n))
            planted.append(VerifyTask(task_id(uid, fake, verifier), verifier, r.units[uid], fake, segs, openings, canary=True))
        return planted

    @_locked
    def tasks_for(self, addr: str) -> list[VerifyTask]:
        r = self._need(Phase.VERIFY)
        return [t for t in r.tasks.values() if t.verifier == addr and t.verdict is None]

    @_locked
    def submit_verdict(self, addr: str, task_id: str, ok: bool, bad_segment: int | None, envelope: dict | None = None) -> None:
        """`envelope` is the verifier's signed message, kept in the record as proof of what it said."""
        r = self._need(Phase.VERIFY)
        t = r.tasks.get(task_id)
        if t is None or t.verifier != addr:
            raise ProtocolError("no such task for this verifier")
        if t.verdict is not None:
            raise ProtocolError("verdict already submitted")
        if not ok and bad_segment not in t.segments:
            raise ProtocolError("a rejection must name one of the challenged segments")
        t.verdict = (bool(ok), None if ok else int(bad_segment))
        r.record["verdicts"].append(envelope if envelope is not None else
                                    {"verifier": addr, "task_id": task_id, "ok": bool(ok), "bad_segment": t.verdict[1]})

    # --- 5. finalize ------------------------------------------------------------------------

    def _replay_task(self, job: TrainingJob, t: VerifyTask) -> list:
        """The coordinator's own check of a task its verifier never answered."""
        self._weights(*(a for a, _ in t.openings.values()))
        return _plain(*replay_openings(job, t.unit, t.openings, t.segments))

    def _arbitrate(self, job: TrainingJob, t: VerifyTask, bad: int) -> bool:
        """Whether the segment a verifier rejected actually reproduces (so the rejection was false)."""
        before, after = t.openings[bad][0], t.openings[bad + 1][0]
        self._weights(before, after)
        return bool(replay_segment(job, t.unit, bad, before, after))

    def _replica_losers(self, job: TrainingJob, uid: str, keys: list, accepted: dict) -> list[str]:
        """Replicas of a unit must agree; if they don't, a full replay of the unit decides who's wrong."""
        r, tol = self.r, self.config.agreement_tolerance
        self._weights(r.start, *(accepted[k].final_params for k in keys))
        if reproduces(accepted[keys[0]].final_params, accepted[keys[1]].final_params, r.start, tol, job.spec):
            return []
        truth = full_replay(job, r.units[uid], r.start)
        return sorted(k[1] for k in keys if not reproduces(accepted[k].final_params, truth, r.start, tol, job.spec))

    def _settle(self, proposer: str):
        """Everything the block will say about the round. Returns (summary, new weights or None on replay,
        worker FLOPs)."""
        r = self.r
        job = self.job(r.job_id)
        verifier_flops: dict[str, int] = defaultdict(int)
        missed: set[str] = set()

        for t in r.tasks.values():
            key = (t.unit.unit_id, t.submitter)
            if t.verdict is None:
                # Verifier didn't show up. The coordinator checks it itself so honest work isn't lost.
                if t.verifier:
                    missed.add(t.verifier)
                if t.canary:
                    continue
                ok, bad = self._claim("replay", t.task_id, lambda t=t: self._replay_task(job, t))
                if ok:
                    r.results[key] = UnitResult(t.unit, t.submitter, Verdict.ACCEPTED, None, "checked by coordinator")
                else:
                    self._fraud(key, Verdict.FAILED_REPLAY, f"segment {bad} mismatch (coordinator)")
                continue

            verifier_flops[t.verifier] += replay_cost(job, len(t.segments))
            ok, bad = t.verdict
            if t.canary:
                if ok:
                    r.slashes.append([t.verifier, None, "approved a canary"])
                continue
            if ok:
                r.results[key] = UnitResult(t.unit, t.submitter, Verdict.ACCEPTED, t.verifier)
            # A rejection is only a claim. Arbitration re-runs the disputed segment.
            elif self._claim("arbitration", t.task_id, lambda t=t, bad=bad: self._arbitrate(job, t, bad)):
                r.results[key] = UnitResult(t.unit, t.submitter, Verdict.ACCEPTED, t.verifier, "false accusation overturned")
                r.slashes.append([t.verifier, t.submitter, "false accusation"])
            else:
                self._fraud(key, Verdict.FAILED_REPLAY, f"segment {bad} mismatch", catcher=t.verifier, verifier=t.verifier)

        accepted = {k: r.reveals[k] for k, res in r.results.items() if res.verdict == Verdict.ACCEPTED}
        for uid, holders in r.holders.items():
            keys = [(uid, a) for a in holders if (uid, a) in accepted]
            if len(keys) < 2:
                continue
            losers = self._claim("redundancy", uid, lambda uid=uid, keys=keys: self._replica_losers(job, uid, keys, accepted))
            wrong = [k for k in keys if k[1] in losers]
            for k in wrong:
                del accepted[k]
                other = next((o[1] for o in keys if o not in wrong), None)
                self._fraud(k, Verdict.FAILED_REDUNDANCY, "disagreed with replica; lost full replay", catcher=other)

        # Fraud (and withholding) forfeits everything earned this round.
        offenders = {s[0] for s in r.slashes} | r.forfeit
        worker_flops: dict[str, int] = defaultdict(int)
        updates: dict[str, np.ndarray] = {}
        for (uid, addr), rv in accepted.items():
            if addr not in offenders:
                worker_flops[addr] += r.units[uid].flops
                updates.setdefault(uid, rv.final_params)  # one per unit so replicas don't double-count
        new_params = None

        def average() -> str:
            nonlocal new_params
            self._weights(*(updates.values() if updates else [r.start]))
            new_params = _mean(list(updates.values()), job) if updates else r.start
            return params_hash(new_params).hex()
        model_hash = self._claim("model", "params_hash", average)
        if r.secret:  # only now: every verdict is in, so it can't give a canary away any more
            r.record["secret"] = r.secret.hex()

        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(updates),
            worker_flops=dict(sorted(worker_flops.items())),
            verifier_flops={a: f for a, f in sorted(verifier_flops.items()) if a not in offenders},
            slashes=r.slashes,
            accepted=sorted(worker_flops),
            missed=sorted(missed | r.forfeit | {res.worker for res in r.results.values() if res.verdict == Verdict.MISSED_DEADLINE}),
            params_hash=model_hash,
            proposer=proposer,
            record_hash=h256(canonical(r.record)).hex(),
        )
        return summary, new_params, worker_flops

    @_locked
    def replayed_summary(self, proposer: str) -> RoundSummary:
        """The summary a replayed round arrives at; nothing is sealed or stored."""
        self._need(Phase.VERIFY)
        summary, _, _ = self._settle(proposer)
        self.phase, self.r = Phase.IDLE, None
        return summary

    @_locked
    def finalize(self, proposer_keys: KeyPair | None = None, timestamp: int | None = None, seal: bool = True) -> RoundReport:
        """Settle the round into a signed block. With `seal`, the block goes straight onto the chain (a chain one node
        runs); without, it's only built, for the validators to vote on (consensus.py), and adopt() adds it if they
        agree."""
        r = self._need(Phase.VERIFY)
        if self.replaying:
            raise ProtocolError("a replayed round isn't sealed; use replayed_summary")
        job = self.job(r.job_id)
        keys = proposer_keys or self.keys
        if keys is None:
            raise ProtocolError("no proposer key")
        summary, new_params, worker_flops = self._settle(keys.address)
        block = self._block(summary, keys, timestamp)
        val_loss, val_acc = evaluate(job.spec, new_params, job.data.x_val, job.data.y_val)
        report = RoundReport(r.round_no, r.job_id, list(r.results.values()), dict(worker_flops), summary.verifier_flops,
                             r.slashes, r.unassigned, val_loss, val_acc, keys.address, block, record=r.record, params=new_params)
        rewards = [t for t in block.txs if t.kind == "reward"]
        fee = sum(t.amount for t in block.txs if t.kind == "fee_release")
        report.pool = self.chain.state.subsidy(block.height) + fee  # the subsidy this block issues, plus released fees
        report.burned = report.pool - sum(t.amount for t in rewards)
        report.worker_pay = {t.recipient: t.amount for t in rewards if t.memo == "work"}
        report.verifier_pay = {t.recipient: t.amount for t in rewards if t.memo == "verify"}
        if seal:
            self.adopt(block, r.record, new_params, (val_loss, val_acc))
            self.history.append(report)
        self.phase, self.r = Phase.IDLE, None
        return report

    @_locked
    def seal_pending(self, timestamp: int | None = None) -> Block | None:
        """With no job to run, still put waiting transactions (like a post_job) into a block."""
        block = self.pending_block(timestamp)
        if block is not None:
            self.adopt(block)
        return block

    @_locked
    def pending_block(self, timestamp: int | None = None) -> Block | None:
        """A block of just the waiting transactions, signed but not added to the chain."""
        if self.phase != Phase.IDLE or not self.mempool or self.keys is None:
            return None
        return self._block(None, self.keys, timestamp)

    def _block(self, summary: RoundSummary | None, keys: KeyPair, timestamp: int | None) -> Block:
        block, _dropped = self.chain.build_block(summary, self.mempool, int(time.time()) if timestamp is None else timestamp)
        block.sign(keys)
        return block

    @_locked
    def adopt(self, block: Block, record: dict | None = None, params: np.ndarray | None = None,
              scores: tuple[float, float] | None = None, certificate: dict | None = None) -> None:
        """Add a finished block to the chain: one we sealed, or one the validators agreed on. `params` are the weights
        its round trained, when we have them, and `scores` their (loss, accuracy) on the job's validation data.
        Without `params` (another validator ran the round), the weights have to be fetched (node.py)."""
        self.chain.append(block)
        included = {t.txid for t in block.txs}
        kept = []
        for tx in self.mempool:  # what's left waits for the next block, if it's still valid after this one
            if tx.txid not in included:
                try:
                    self.chain.check_tx(tx, kept)
                    kept.append(tx)
                except (InvalidBlock, InvalidJob):
                    pass
        self.mempool = kept
        if self.store:
            self.store.save_block(block, certificate)
            if record is not None:
                self.store.save_record(block.height, record)
        s = block.summary
        if s is None:
            return
        if params is None:
            self.params.pop(s.job_id, None)
            return
        self.params[s.job_id] = params
        if self.store:
            self.store.save_params(s.job_id, block.height, params)
            if scores is not None:
                self.store.save_round(s.job_id, block.height, s.accepted_units, *scores)
