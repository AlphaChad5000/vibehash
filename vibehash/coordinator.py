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

A sharded job (SHARDING.md, shards.py) runs the same phases with teams in
place of workers: each unit goes to a team, one worker per shard; each member
commits to its own trace; the beacon challenges cells (one shard by one
segment) rather than segments of the whole model; neighbours' edge hashes are
compared here, and a signed edge message settles who lied when they differ;
and checkers each check one cell, holding one shard.
"""
from __future__ import annotations

import functools
import itertools
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import economics as econ
from . import fees
from . import shards as sh
from .crypto import Draw, KeyPair, address_of, h256, verify
from .ledger import Block, Chain, InvalidBlock, RoundSummary, Tx, canonical
from .model import evaluate, params_hash
from .tasks import Commitment, InvalidJob, Reveal, TrainingJob, Verdict, WorkUnit, commit_digest
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
# also hash in a secret the leader commits to when it opens the round, and canaries are copies of real work. 3: a
# sharded job's round (teams, cells, edges and edge evidence), with version 2's secret; only sharded jobs' rounds are
# recorded as 3, so whole units' rounds stay version 2 and keep replaying, and being voted for, as they are.
RECORD_VERSION = 2
SHARDED_RECORD_VERSION = 3
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
class CellTask(VerifyTask):
    """A sharded unit's verify task: one cell, one shard by one segment (`segments` is just that one), checked with
    shards.check_cell against the root its worker committed to. A checker needs that shard's memory and no more."""
    shard: sh.Shard | None = None
    root: bytes = b""
    cell: dict = field(default_factory=dict)


@dataclass
class Team:
    """Workers training a sharded job's units together, one per shard, in shard order."""
    members: list[str]
    units: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EdgeDispute:
    """Two neighbours whose commitments name different tensors at an edge: the first step each way where they do.
    `kind` is how the receiver recorded it ("h_in" or "g_out"); the receiver settles it by showing the sender's
    signed edge message (submit_edge_evidence)."""
    unit_id: str
    step: int
    kind: str
    receiver: str
    sender: str


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
    teams: list[list[str]] = field(default_factory=list)  # a sharded job's teams, each its members in shard order
    waiting: str = ""  # why a sharded job's units went untrained, for people (Coordinator.waiting), or ""


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
    # Sharded jobs only. `opened` holds every member's reveal whose commitment and edge proofs check out, which is all
    # comparing edges needs; `reveals` only those whose weights check out too, whose cells go to checkers.
    teams: list[Team] = field(default_factory=list)
    shard_of: dict[str, int] = field(default_factory=dict)
    team_of: dict[str, Team] = field(default_factory=dict)  # by unit
    opened: dict[tuple[str, str], sh.ShardReveal] = field(default_factory=dict)
    evidence: dict[tuple, tuple[bytes, bytes]] = field(default_factory=dict)  # (unit, receiver, step, kind) -> (hash, sig)
    waiting: str = ""


def _mean(arrays: list, job: TrainingJob) -> np.ndarray:
    """The new model: accepted results summed in float64 in round order, divided, rounded to the job's precision.
    Pinned down exactly, because anyone rechecking the round must get the same bits: numpy's own mean works in
    whatever precision the arrays arrived in (float32 from a node's disk, float64 elsewhere)."""
    total = np.zeros(job.spec.n_params, dtype=np.float64)
    for a in arrays:
        total += np.asarray(a, dtype=np.float64)
    return job.spec.round(total / len(arrays))


def _mean_by_shard(finals: list[list], shards: list[sh.Shard], job: TrainingJob) -> np.ndarray:
    """A sharded round's new model: each shard averaged across the units that passed (`finals`: per unit, its shards'
    final weights in shard order), and the model their concatenation. Averaging is per weight, so this is exactly
    _mean of the units' whole vectors, bit for bit, while needing only one shard in memory at a time."""
    theta = np.empty(job.spec.n_params)
    for k, s in enumerate(shards):
        total = np.zeros(s.size(job.spec), dtype=np.float64)
        for parts in finals:
            total += np.asarray(parts[k], dtype=np.float64)
        theta[s.span(job.spec)] = job.spec.round(total / len(finals))
    return theta


def _size(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


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
        # Whether whoever drives this coordinator can run a sharded job's teams. Members send each other activations
        # and gradients every step; protocol.Protocol does that in one process, but nodes' workers have no way to reach
        # each other yet, so a node leaves sharded jobs waiting (with a reason in `waiting`) instead of assigning units
        # nobody could train.
        self.runs_teams = False
        # job -> why it isn't training, in words for /status and the dashboards: a sharded job no team could be formed
        # for, or one this node can't run. Not in any record: nothing a round decides depends on it.
        self.waiting: dict[str, str] = {}

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
                    job = self.job(job_id)
                except (InvalidJob, ValueError, OSError):
                    continue  # dataset not available here yet
                if job.sharded and not self.runs_teams:
                    self.waiting[job_id] = (f"job {job_id} waits: it's split into {job.shards} shards, trained by teams "
                                            "whose workers send each other activations, and this node can't run teams yet")
                    continue
                ready.append(job_id)
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

    def _replay_team_start(self, job: TrainingJob, start_hash: bytes) -> np.ndarray | None:
        """A sharded round's starting weights, put together a shard at a time from the first checkpoints members
        revealed: every honest holder of a shard reveals the same bytes, so the commonest of each is tried first, and
        the record's start hash says whether a combination is right. A shard nobody revealed means unchecked."""
        parts = sh.split(job.spec, job.shards)
        try:
            shard_of = {a: k for team in self.replaying.get("teams", []) for k, a in enumerate(team["members"])}
            seen = [Counter() for _ in parts]
            for d in self.replaying.get("reveals", []):
                k, bid = shard_of.get(d["worker"]), ((d.get("ends") or {}).get("0") or [None])[0]
                if k is not None and isinstance(bid, str):
                    seen[k][bid] += 1
        except (KeyError, TypeError, ValueError, AttributeError):  # too mangled to say: the round's other checks will
            return None
        for combo in itertools.islice(itertools.product(*([b for b, _ in c.most_common()] for c in seen)), 16):
            arrays = [self._fetch(bid, job) for bid in combo]
            try:
                theta = sh.assemble(job.spec, parts, arrays)
            except (TypeError, ValueError):  # one is gone, or isn't its shard's size
                continue
            if params_hash(theta) == start_hash:
                return theta
        return None

    def _replay_shard_reveal(self, job: TrainingJob, wire: dict, shard: sh.Shard) -> sh.ShardReveal | None:
        """A recorded sharded reveal with its arrays loaded, or None if any of them is gone (an edge tensor that's None
        by rights, at the model's input or output, is no blob at all, so it's never mistaken for a missing one)."""
        gone = []

        def load(bid):
            a = self._fetch(bid, job)
            if a is None:
                gone.append(bid)
            return a
        try:
            rv = sh.reveal_in(wire, load, shard.shapes(job.spec, job.batch_size))
        except WireError:
            return None
        return None if gone else rv

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
            if self.replaying.get("version", RECORD_VERSION) not in (1, 2, SHARDED_RECORD_VERSION):
                raise ProtocolError(f"the record is version {self.replaying.get('version')!r}, which this code doesn't know")
        else:
            job_id = self.active_job_id()
            if job_id is None:
                return False
        job = self.job(job_id)
        if self.replaying and (self.replaying.get("version", RECORD_VERSION) == SHARDED_RECORD_VERSION) != job.sharded:
            raise ProtocolError(f"the record is version {self.replaying.get('version', RECORD_VERSION)}, but sharded jobs' "
                                f"rounds, and only theirs, are recorded as version {SHARDED_RECORD_VERSION}")
        if self.replaying:
            start, start_hash = None, bytes.fromhex(self.replaying["start_hash"])
            if start_hash != self.expected_start_hash(job):
                raise ProtocolError("the round didn't start from the job's latest model")
            if self.recheck is not None:
                start = self._replay_team_start(job, start_hash) if job.sharded else self._replay_start(job, start_hash)
        else:
            start = self.start_params(job_id)
            start_hash = params_hash(start)
        units = [WorkUnit.make(job, round_no, i, start_hash) for i in range(job.units_per_round)]
        self.r = _Round(round_no, job_id, start, start_hash, {u.unit_id: u for u in units})
        self.r.record = {"version": RECORD_VERSION, "round": round_no, "job_id": job_id, "start_hash": start_hash.hex(),
                         "rules": self.config.rules(), "registrations": [], "commitments": [], "reveals": [], "verdicts": [],
                         "checks": {"reveal": {}, "replay": {}, "arbitration": {}, "redundancy": {}, "model": {}}}
        if job.sharded:  # its teams (start_compute) and the signed edge messages that settled disputes (submit_edge_evidence)
            self.r.record |= {"version": SHARDED_RECORD_VERSION, "teams": [], "evidence": []}
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
        if job.sharded:
            self._form_teams(job, order, capacity)
            self.phase = Phase.COMPUTE
            return

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

    def _form_teams(self, job: TrainingJob, order: list[str], capacity: dict[str, int]) -> None:
        """A sharded job's teams, one worker per shard, and their units. Workers are taken in the round's drawn order,
        staked ones first as for whole units, each for the first shard (biggest first) its memory fits and its
        capacity can train a unit of; no worker is on two teams. Units then go one at a time to the team with the most
        room left, as they go to workers. A team needs every shard, since a pipeline missing a stage can't train at
        all, so workers left over form no team: they check cells instead. Too few for even one team leaves every unit
        untrained this round, as units nobody has room for are today; nothing is released from the job's escrow, and
        `waiting` says what it would take."""
        r, parts = self.r, sh.split(job.spec, job.shards)
        memory = [sh.memory_mb(job.memory_mb, job.spec, s) for s in parts]
        work = [sh.cell_flops(job.spec, s, job.batch_size, job.steps_per_unit) for s in parts]  # one unit's, per shard
        staked = self.chain.state.staked
        free = sorted(order, key=lambda a: not staked(a))  # a stable sort: the draw's order within each group
        teams: list[Team] = []
        while len(teams) < len(r.units):
            members: dict[int, str] = {}
            for k in sorted(range(len(parts)), key=lambda k: (-memory[k], k)):
                a = next((a for a in free if a not in members.values() and r.regs[a].memory_mb >= memory[k]
                          and capacity[a] >= work[k]), None)
                if a is None:
                    break
                members[k] = a
            if len(members) < len(parts):
                break
            teams.append(Team([members[k] for k in range(len(parts))]))
            free = [a for a in free if a not in members.values()]
        used = {a: 0 for t in teams for a in t.members}
        for uid, unit in r.units.items():
            fits = [t for t in teams if all(used[a] + work[k] <= capacity[a] for k, a in enumerate(t.members))]
            if not fits:
                r.unassigned += 1
                continue
            team = min(fits, key=lambda t: (any(not staked(a) for a in t.members),
                                            max((used[a] + work[k]) / capacity[a] for k, a in enumerate(t.members))))
            team.units.append(uid)
            r.team_of[uid] = team
            for k, a in enumerate(team.members):
                used[a] += work[k]
                r.assignments[a].append(uid)
                r.holders[uid].append(a)
        r.teams = [t for t in teams if t.units]
        r.shard_of = {a: k for t in r.teams for k, a in enumerate(t.members)}
        r.record["teams"] = [{"members": list(t.members), "units": list(t.units)} for t in r.teams]
        if self.replaying and self.replaying.get("teams") != r.record["teams"]:
            raise ProtocolError("the record's teams aren't the ones its registrations draw")
        r.waiting = "" if r.teams else self._why_no_team(job, memory, work, capacity)
        if r.waiting:
            self.waiting[job.job_id] = r.waiting
        else:
            self.waiting.pop(job.job_id, None)

    def _why_no_team(self, job: TrainingJob, memory: list[int], work: list[int], capacity: dict[str, int]) -> str:
        """Why not even one team could be formed, and what would change it, for whoever posted the job."""
        r, n = self.r, len(memory)
        roomy = [a for a in r.regs if r.regs[a].memory_mb >= max(memory)]
        fast = [a for a in roomy if capacity[a] >= max(work)]
        need = f"needs {n} workers with at least {_size(max(memory))} of memory for its {n} shards"
        if len(roomy) < n:
            return f"job {job.job_id} waits: it {need}; {len(roomy)} of the {len(r.regs)} registered this round have that much"
        return (f"job {job.job_id} waits: it {need}, each able to train its shard of a unit in the "
                f"{r.record['rules']['compute_s']:.0f} s compute phase; {len(fast)} of the {len(r.regs)} registered "
                f"this round can")

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
        if job.sharded:
            raise ProtocolError("this round's job is sharded: its members reveal cells (reveal_shard)")
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

    def _fraud(self, key, verdict: Verdict, detail: str, catcher: str | None = None, verifier: str | None = None,
               reason: str | None = None) -> Verdict:
        r = self.r
        r.results[key] = UnitResult(r.units[key[0]], key[1], verdict, verifier, detail)
        r.slashes.append([key[1], catcher, reason or verdict.value])
        return verdict

    # --- 3b. a sharded job's reveals and edges ----------------------------------------------------------

    def _asks(self, job: TrainingJob, uid: str, addr: str) -> tuple[list[int], list[int]]:
        """The cells the beacon challenges of a member's shard of a unit, and the steps it opens its edges at."""
        r = self.r
        return sh.challenged(r.beacon, uid, r.team_of[uid].members, addr, job.n_segments, job.checkpoint_every,
                             self.config.challenge_segments)

    @_locked
    def reveal_shard(self, rv: sh.ShardReveal, wire: dict | None = None) -> Verdict | None:
        """A team member's reveal for a unit (shards.ShardReveal). Hash checks happen here and now: it must open its
        commitment, every challenged cell and every edge hash the beacon asks of it, each proven against its root.
        Then the checks that need weights: every opening is what it committed to and shaped as its shard's are, its
        training starts from the round's model, and its update is finite and bounded. Returns a fraud verdict, or
        None if its cells go on to checkers. `wire` is the reveal as recorded (shards.reveal_out)."""
        r = self._need(Phase.REVEAL)
        key = (rv.unit_id, rv.worker)
        if key not in r.commitments:
            raise ProtocolError("no commitment for this unit")
        if key in r.reveals or key in r.results or key in r.opened:
            raise ProtocolError("already revealed")
        job = self.job(r.job_id)
        if not job.sharded:
            raise ProtocolError("this round's job isn't sharded")
        r.record["reveals"].append(wire if wire is not None else
                                   {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.root.hex(), "salt": rv.salt.hex()})
        shard = sh.split(job.spec, job.shards)[r.shard_of[rv.worker]]
        if self.recheck is not None and wire is not None:
            rv = self._replay_shard_reveal(job, wire, shard) or rv
        c, n = r.commitments[key], job.n_segments
        if c.digest != commit_digest(rv.unit_id, c.pubkey, rv.root, rv.salt):
            return self._fraud(key, Verdict.BAD_REVEAL, "reveal doesn't match commitment")
        segs, steps = self._asks(job, *key)
        leaf = lambda t, kind: sh.leaf_index(n + 1, t, kind)  # noqa: E731
        if (set(rv.cells) != set(segs) or set(rv.ends) != {0, n}
                or any((t, kind) not in rv.edges or not sh.proves(*rv.edges[(t, kind)], rv.root, leaf(t, kind))
                       for t in steps for kind in sh.KINDS)):
            return self._fraud(key, Verdict.BAD_REVEAL, "a challenged cell or edge hash missing or unproven")
        r.opened[key] = rv  # its edge hashes are proven: they can be compared with its neighbours'

        def checks() -> str:
            self._weights(*(a for a, _ in rv.ends.values()),
                          *(a for cell in rv.cells.values() for a in (cell["before"], cell["after"])))
            first, last = rv.ends[0][0], rv.ends[n][0]
            if (any(not isinstance(a, np.ndarray) or not sh.proves(sh.tensor_hash(a), proof, rv.root, i)
                    for i, (a, proof) in rv.ends.items())
                    or any(sh.cell_unproven(rv.root, s, cell, job.steps_per_unit, job.checkpoint_every)
                           or sh.misshapen(job.spec, shard, cell, job.batch_size) for s, cell in rv.cells.items())):
                return "bad_openings"
            self._weights(r.start)
            start = r.start[shard.span(job.spec)]
            # The job bounds a unit's update relative to the whole model. A shard's own can be far bigger relative to
            # itself (an output layer moves about half its size while the model moves a fifth), so each shard's update
            # is held to the bound on the whole model's, which it can't exceed when its unit's update is within it.
            sane = (first.shape == last.shape == start.shape and np.all(np.isfinite(last)) and np.array_equal(first, start)
                    and np.linalg.norm(last - start) <= job.max_update_ratio * (np.linalg.norm(r.start) + 1e-12))
            return "ok" if sane else "failed_sanity"
        outcome = self._claim("reveal", f"{rv.unit_id}/{rv.worker}", checks)
        if outcome == "bad_openings":
            return self._fraud(key, Verdict.BAD_REVEAL, "a cell's opening isn't what it committed to, or isn't its shard's")
        if outcome == "failed_sanity":
            return self._fraud(key, Verdict.FAILED_SANITY, "didn't start from the round's model, or an oversized update")
        r.reveals[key] = rv
        return None

    def _disputes(self) -> list[EdgeDispute]:
        """Every pair of neighbours, per unit, whose proven edge hashes disagree somewhere they were challenged."""
        r, job, out = self.r, self.job(self.r.job_id), []
        for team in r.teams:
            for uid in team.units:
                for lo, up in zip(team.members, team.members[1:]):
                    if (uid, lo) not in r.opened or (uid, up) not in r.opened:
                        continue  # one of them never revealed, or its hashes didn't check out: nothing to compare
                    steps = sorted(sh.segment_steps(self._asks(job, uid, lo)[0] + self._asks(job, uid, up)[0], job.checkpoint_every))
                    for step, kind in sh.disagreements(r.opened[(uid, lo)].edges, r.opened[(uid, up)].edges, steps):
                        receiver, sender = (up, lo) if kind == "h_in" else (lo, up)
                        out.append(EdgeDispute(uid, step, kind, receiver, sender))
        return out

    @_locked
    def edge_disputes(self) -> list[EdgeDispute]:
        """Disagreements at edges so far, for their receivers to settle with the edge messages they were sent."""
        self._need(Phase.REVEAL)
        return self._disputes()

    @_locked
    def submit_edge_evidence(self, addr: str, unit_id: str, step: int, kind: str, message: dict, envelope: dict | None = None) -> None:
        """A receiver's side of an edge dispute: the message its neighbour signed for the tensor it sent (shards.
        edge_message), {"hash", "sig"} in hex. Checked when the round moves on to verify (_settle_edges). `envelope` is
        the receiver's signed message, kept in the record."""
        r = self._need(Phase.REVEAL)
        if not any((d.unit_id, d.receiver, d.step, d.kind) == (unit_id, addr, step, kind) for d in self._disputes()):
            raise ProtocolError("no such edge dispute for this worker")
        if (unit_id, addr, step, kind) in r.evidence:
            raise ProtocolError("evidence already submitted")
        try:
            digest, sig = bytes.fromhex(message["hash"]), bytes.fromhex(message["sig"])
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"bad edge message: {e}") from e
        if len(digest) != 32 or len(sig) != 64:
            raise ProtocolError("an edge message is a SHA-256 and an Ed25519 signature")
        r.evidence[(unit_id, addr, step, kind)] = (digest, sig)
        r.record["evidence"].append(envelope if envelope is not None else
                                    {"receiver": addr, "unit_id": unit_id, "step": step, "kind": kind, "hash": digest.hex(), "sig": sig.hex()})

    def _settle_edges(self) -> None:
        """Who lied at each disputed edge. The receiver's evidence is its neighbour's signature over the hash of what
        it sent. A side whose commitment names anything else lied: the sender about what it sent, the receiver about
        what it took in (both, if both did). A receiver without a valid signature is to blame: it had to keep what it
        was sent, and an honest one always can, since it refuses unsigned tensors. Either way the honest side's edge
        counts as checked, so a liar can't cost its neighbour its pay."""
        r = self.r
        for d in self._disputes():
            sent = sh.SENT[d.kind]
            mine = r.opened[(d.unit_id, d.receiver)].edges[(d.step, d.kind)][0]
            theirs = r.opened[(d.unit_id, d.sender)].edges[(d.step, sent)][0]
            what = "activations" if d.kind == "h_in" else "gradient"
            ev = r.evidence.get((d.unit_id, d.receiver, d.step, d.kind))
            msg = ev and sh.edge_message(self.chain.chain_id, r.round_no, d.unit_id, d.step, sent, ev[0])
            if ev is not None and verify(r.commitments[(d.unit_id, d.sender)].pubkey, msg, ev[1]):
                liars = [(a, other, f"step {d.step}: committed to {what} other than it {how}")
                         for a, other, leaf, how in ((d.sender, d.receiver, theirs, "sent"), (d.receiver, d.sender, mine, "was sent"))
                         if leaf != ev[0]]
                if len(liars) == 2:  # neither earns a bounty from the other
                    liars = [(a, None, detail) for a, _, detail in liars]
            else:
                liars = [(d.receiver, None, f"step {d.step}: can't show the {what} it says it was sent")]
            for blamed, catcher, detail in liars:
                if (d.unit_id, blamed) not in r.results:  # one verdict a member a unit, as a whole unit gets one
                    self._fraud((d.unit_id, blamed), Verdict.BAD_REVEAL, detail, catcher, reason="lied at an edge")

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
        if job.sharded:
            self._cell_tasks(job)
            self.phase = Phase.VERIFY
            return
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

    def _cell_tasks(self, job: TrainingJob) -> None:
        """A sharded round's verify tasks: every challenged cell of every member whose reveal checked out, each to a
        staked checker with the memory for that shard who holds no shard of the unit, drawn from the beacon and the
        round secret as whole units' checkers are; then canaries made of cells. Edge disputes are settled first, so a
        member caught lying at an edge neither checks others nor has its own cells checked: its pay is forfeit and its
        team's update is out either way."""
        r = self.r
        self._settle_edges()
        offenders = {s[0] for s in r.slashes}
        parts = sh.split(job.spec, job.shards)
        memory = [sh.memory_mb(job.memory_mb, job.spec, s) for s in parts]
        seed = (r.beacon, r.secret)
        draw = Draw(b"verify", *seed)

        def pool(exclude: set[str], k: int) -> list[str]:
            return [a for a, reg in sorted(r.regs.items()) if a not in exclude and a not in offenders
                    and reg.memory_mb >= memory[k] and self.chain.state.staked(a)]

        def task_id(uid, submitter, verifier, seg) -> str:
            return h256(*seed, uid.encode(), submitter.encode(), verifier.encode(), seg.to_bytes(8, "big")).hex()[:20]

        tasks, checkers = [], defaultdict(set)
        for (uid, addr), rv in r.reveals.items():
            if addr in offenders:
                continue
            k = r.shard_of[addr]
            for seg in sorted(rv.cells):
                candidates = pool(set(r.holders[uid]), k)
                verifier = candidates[draw.below(len(candidates))] if candidates else ""
                checkers[uid].add(verifier)
                tasks.append(CellTask(task_id(uid, addr, verifier, seg), verifier, r.units[uid], addr, [seg], {},
                                      shard=parts[k], root=rv.root, cell=rv.cells[seg]))
        tasks += self._cell_canaries(job, draw, pool, task_id, checkers, offenders)
        r.tasks = {tasks[i].task_id: tasks[i] for i in draw.shuffle(range(len(tasks)))}

    def _cell_canaries(self, job: TrainingJob, draw: Draw, pool, task_id, checkers: dict[str, set[str]],
                       offenders: set[str]) -> list[CellTask]:
        """Record version 2's canaries, made of cells: a copy of a real challenged cell with its update rescaled
        (shards.plant_cell), so it fails a replay at the weights it ends at, and nowhere else, and can't be picked out
        without one. Its root is a fresh tree's, but nobody outside the team sees a member's root before the round is
        sealed. It goes to a checker that could have been given the real cell: one with the memory for that shard,
        holding no shard of the unit and checking no real cell of it (it could compare roots or checkpoints), and at
        most one canary a unit. So a sharded round has room for one once a unit's real cells leave a staked worker
        outside its team unasked. A replay needs only who checks which canary, which follows from the record."""
        r, parts, planted, used = self.r, sh.split(job.spec, job.shards), [], set()
        n_leaves = job.n_segments + 1 + 4 * job.steps_per_unit
        for uid, addr in draw.shuffle(sorted(r.reveals)):
            if len(planted) == self.config.canaries_per_round:
                break
            k = r.shard_of[addr]
            candidates = [] if uid in used or addr in offenders else pool(set(r.holders[uid]) | checkers[uid], k)
            if not candidates:
                continue
            used.add(uid)
            n = len(planted).to_bytes(8, "big")
            fake = KeyPair.generate(r.secret + n).address
            rv = r.reveals[(uid, addr)]
            segs = sorted(rv.cells)
            seg = segs[draw.below(len(segs))]
            verifier = candidates[draw.below(len(candidates))]
            cell, root = {}, b""
            if not self.replaying:  # no training, so cheap under the lock: a vector sum and a few hashes
                cell, root = sh.plant_cell(job.spec, rv.cells[seg], seg, n_leaves, Draw(b"canary", r.secret, n))
            planted.append(CellTask(task_id(uid, fake, verifier, seg), verifier, r.units[uid], fake, [seg], {}, canary=True,
                                    shard=parts[k], root=root, cell=cell))
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

    def _check_task(self, job: TrainingJob, t: CellTask) -> list:
        """The coordinator's own check of one cell (a task nobody answered, or arbitration): [passed, its segment if
        not]. Replays that one cell, so it needs one shard's memory, however big the model."""
        cell, seg = t.cell, t.segments[0]
        self._weights(cell.get("before"), cell.get("after"))  # a recorded reveal's arrays load all together or not at all
        try:
            bad = sh.check_cell(job.spec, t.shard, job.data, t.root, seg, cell, t.unit.seed, job.steps_per_unit,
                                job.checkpoint_every, job.lr, job.batch_size, job.tolerance)
        except (ValueError, TypeError, IndexError, KeyError):  # malformed beyond replaying: no honest cell is
            bad = "malformed"
        return _plain(bad is None, None if bad is None else seg)

    def _edge_checked(self, uid: str, addr: str, neighbour: str) -> bool:
        """Whether what crossed between a member and a neighbour on a unit is tied to both their commitments: both
        opened proven edge hashes (any disagreement was settled against one of them, _settle_edges), or the neighbour's
        reveal was itself fraud. A neighbour that never revealed leaves it unchecked: honest members on that edge go
        unpaid for the unit, though not blamed, since a member could otherwise fake the gradient a vanishing neighbour
        'sent' it and skip its backward pass (the cell would replay to the same faked result)."""
        r = self.r
        res = r.results.get((uid, neighbour))
        return (uid, addr) in r.opened and ((uid, neighbour) in r.opened or (res is not None and res.verdict.is_fraud))

    def _settle_teams(self, proposer: str):
        """A sharded round's _settle. Every cell task is settled as a whole unit's task is (the coordinator checks
        those nobody answered, arbitration replays a rejected cell, approving a canary is slashed). Then each member
        whose challenged cells all passed and whose edges were checked is paid for its cells' FLOPs, its whole shard of
        the unit, as a whole unit is paid once its challenged segments pass; a cheating teammate costs it nothing.
        A unit's update joins the new model only if every member's cells and edges passed and nobody on the team is
        an offender, and the model is averaged a shard at a time (_mean_by_shard)."""
        r = self.r
        job = self.job(r.job_id)
        parts = sh.split(job.spec, job.shards)
        verifier_flops: dict[str, int] = defaultdict(int)
        missed: set[str] = set()

        for t in r.tasks.values():
            key, seg = (t.unit.unit_id, t.submitter), t.segments[0]
            if t.verdict is None:
                if t.verifier:
                    missed.add(t.verifier)
                if t.canary:
                    continue
                ok, _ = self._claim("replay", t.task_id, lambda t=t: self._check_task(job, t))
                if not ok and key not in r.results:  # its first failed cell on a unit is its verdict on that unit
                    self._fraud(key, Verdict.FAILED_REPLAY, f"cell {r.shard_of[t.submitter]}/{seg} mismatch (coordinator)")
                continue
            verifier_flops[t.verifier] += sh.cell_flops(job.spec, t.shard, job.batch_size, job.checkpoint_every)
            ok, _ = t.verdict
            if t.canary:
                if ok:
                    r.slashes.append([t.verifier, None, "approved a canary"])
                continue
            if ok:
                continue
            if self._claim("arbitration", t.task_id, lambda t=t: self._check_task(job, t)[0]):
                r.slashes.append([t.verifier, t.submitter, "false accusation"])
            elif key not in r.results:
                self._fraud(key, Verdict.FAILED_REPLAY, f"cell {r.shard_of[t.submitter]}/{seg} mismatch",
                            catcher=t.verifier, verifier=t.verifier)

        offenders = {s[0] for s in r.slashes} | r.forfeit
        worker_flops: dict[str, int] = defaultdict(int)
        finals: dict[str, list] = {}
        for team in r.teams:
            for uid in team.units:
                passed = []
                for k, a in enumerate(team.members):
                    near = team.members[max(0, k - 1):k] + team.members[k + 1:k + 2]
                    passed.append((uid, a) in r.reveals and (uid, a) not in r.results
                                  and all(self._edge_checked(uid, a, b) for b in near))
                    if passed[-1] and a not in offenders:
                        worker_flops[a] += sh.cell_flops(job.spec, parts[k], job.batch_size, job.steps_per_unit)
                        r.results[(uid, a)] = UnitResult(r.units[uid], a, Verdict.ACCEPTED, None, f"shard {k}: cells and edges checked out")
                if all(passed) and not offenders & set(team.members):
                    finals[uid] = [r.reveals[(uid, a)].final for a in team.members]
        kept = [finals[uid] for uid in r.units if uid in finals]  # in unit order, so every recheck sums the same way
        new_params = None

        def average() -> str:
            nonlocal new_params
            self._weights(*(f for unit_finals in kept for f in unit_finals) if kept else [r.start])
            new_params = _mean_by_shard(kept, parts, job) if kept else r.start
            return params_hash(new_params).hex()
        model_hash = self._claim("model", "params_hash", average)
        r.record["secret"] = r.secret.hex()  # only now: every verdict is in, so it can't give a canary away any more

        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(kept),
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

    def _settle(self, proposer: str):
        """Everything the block will say about the round. Returns (summary, new weights or None on replay,
        worker FLOPs)."""
        r = self.r
        job = self.job(r.job_id)
        if job.sharded:
            return self._settle_teams(proposer)
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
                             r.slashes, r.unassigned, val_loss, val_acc, keys.address, block, record=r.record, params=new_params,
                             teams=[list(t.members) for t in r.teams], waiting=r.waiting)
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
