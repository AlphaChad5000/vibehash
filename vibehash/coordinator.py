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

A fine-tune (finetune.py) runs them so that the node never runs the model: its rounds are recorded as version 4.
Each claim goes to two checkers; scorers measure the round's starting model; and each rejection, and each pair of
replicas that disagree, becomes a dispute in an extra phase: the claim's worker and its accuser narrow their
disagreement down to one op of one step, and the node referees that one op with NumPy (dispute.py, referee.py;
start_disputes, _settle_finetune). Otherwise the node only checks hashes, Merkle proofs and bounds, and averages.

An evolution-strategies job (es.py) runs them as record version 5: its units are groups of antithetic seed pairs, the
seeds and the round's problems drawn when registration closes; a worker commits each member's rewards and its answers,
the node re-scores every answer itself at the reveal (no model needed), checkers teacher-force a beacon-chosen sample of
members through the perturbed model, and the node works the next state out of the accepted rewards with es.step
(_settle_es), so the record holds everything a replay needs to work it out again.

A sharded job (SHARDING.md, shards.py) runs the same phases with teams in
place of workers: each unit goes to a team, one worker per shard; each member
commits to its own trace; the beacon challenges cells (one shard by one
segment) rather than segments of the whole model; neighbours' edge hashes are
compared here, and a signed edge message settles who lied when they differ;
and checkers each check one cell, holding one shard.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import itertools
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import deepseek_graph as dsg
from . import deepseek_rounds as dsr
from . import dispute as dg
from . import economics as econ
from . import es
from . import es2, es2_rounds
from . import fees
from . import shards as sh
from .crypto import Draw, KeyPair, address_of, h256, verify
from .finetune import MAX_VAL, WHOLE_TOLERANCE
from .ledger import SYSTEM, Block, Chain, InvalidBlock, RoundSummary, Tx, canonical
from .model import params_hash
from .storage import NotSaved
from .tasks import Commitment, InvalidJob, Reveal, TrainingJob, Verdict, WorkUnit, commit_digest
from .verification import (
    FINETUNE_CHECK,
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
    replicas_agree,
    reproduces,
    required_openings,
)
from .wire import WireError, blob_openings_in, commitment_out
from .worker import HardwareProfile, Worker

log = logging.getLogger("vibehash.node")

# Records say which rules made them, and a replay follows the record's version, so every round still replays exactly as
# it was run. 1 (0.4.0-0.5.1): the beacon, who checks whom, task ids and canaries all came from public data. 2: they
# also hash in a secret the leader commits to when it opens the round, and canaries are copies of real work. 3: a
# sharded job's round (teams, cells, edges and edge evidence), with version 2's secret; only sharded jobs' rounds are
# recorded as 3, so whole units' rounds stay version 2 and keep replaying, and being voted for, as they are.
# 4: a fine-tune's round, where the node never runs the model: two checkers a claim, scorers, and a dispute phase
# (start_disputes) where the node referees one op of each disputed claim. Only fine-tunes' rounds are recorded as 4.
RECORD_VERSION = 2
PREP_MEMORY_MB = 6000  # a DeepSeek prep unit's GPU memory (deepseek_worker.NEEDS["prep"])
SHARDED_RECORD_VERSION = 3
FINETUNE_RECORD_VERSION = 4
# 5: an ES job's round (es.py): its seed, level, every claim's rewards and the update the node worked out. Only ES jobs'
# rounds are recorded as 5.
ES_RECORD_VERSION = 5
CHECKERS = 2  # checkers given each claim of a fine-tune, so one that doesn't answer leaves the other's verdict
SCORERS = 3  # workers who score a fine-tune round's starting model; the record keeps the median
MAX_GAMES = 4  # disputes one checker can open in a round (games_allowed); its further rejections are ignored
# The pilot's rule (DEEPSEEK.md §8): a rejected prep unit isn't refereed. It ends unchecked and is redone, nobody
# paid or slashed for it, until a prep game's moves are timed from its states' real size (one 4K-token chunk's state is
# 320 MB and more, and a move is timed for a LoRA adapter's) and the bf16 tolerance is measured across devices.
PREP_GAMES = False


def games_allowed(rep: float) -> int:
    """Disputes a checker with reputation `rep` may open in a round: MAX_GAMES from the reputation a newcomer starts
    with, one fewer for each round's worth of reputation it has lost since (economics.REP_LOSS_MISSED: a round in which
    it withdrew a rejection, or missed a deadline), and never fewer than one. So accusing again and again costs more:
    each round of it takes a dispute away, as well as that round's pay for checking and some of its pay rate."""
    lost = max(0, round((econ.REP_START - rep) / econ.REP_LOSS_MISSED))
    return max(1, MAX_GAMES - lost)
# Staked workers able to run a fine-tune that must register before its round assigns work: a holder and one checker.
# With the node refereeing disputes, a claim's verdict is right whenever one of its checkers is honest and online, so
# one checker is enough to start; from three workers on, each claim gets two, so one going quiet leaves the other.
# (Canaries, which catch checkers who approve without replaying, need a worker who neither holds nor checks either of
# two units, so they start from about five.)
FINETUNE_QUORUM = 2
# How close a scorer's (loss, accuracy) must be to the median to be paid: honest scorers agree to about 1e-6.
SCORE_AGREEMENT = 1e-3
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
    DISPUTE = "dispute"  # a fine-tune's disputes over claims, refereed by the node (start_disputes)


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
    # How a fine-tune's replicas are compared (verification.FINETUNE_CHECK). Only fine-tunes' rounds record it
    # (open_round), so other jobs' rules, and their records' hashes, stay as they were.
    finetune_check: int = FINETUNE_CHECK

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
    # Fine-tunes: "check" (replay the segments) or "score" (score the round's starting model on the held-back data; no
    # openings).
    kind: str = "check"
    score: tuple[float, float] | None = None  # what a scorer reported: (loss, accuracy)
    donor: str = ""  # a DeepSeek prep canary's: the unit whose claim it presents as `unit`'s


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
    # Fine-tunes only: the disputes (dispute.Game by id), and the checkers who approved a canary, whose verdicts this
    # round don't count.
    games: dict[str, dg.Game] = field(default_factory=dict)
    void: set[str] = field(default_factory=set)
    # A DeepSeek prep round's: each unit's graph unit (deepseek_rounds.py), and each claim's revealed commitment.
    prep: dict[str, str] = field(default_factory=dict)
    prep_commits: dict[tuple[str, str], dict] = field(default_factory=dict)
    prep_raw: dict[tuple[str, str], bytes] = field(default_factory=dict)  # each's bytes, which its checkers get
    es_claims: dict[tuple[str, str], tuple[str, list]] = field(default_factory=dict)  # an ES claim's (answers, rewards)

    @property
    def v4(self) -> bool:
        return self.record.get("version") == FINETUNE_RECORD_VERSION

    @property
    def es(self) -> bool:
        return self.record.get("version") == ES_RECORD_VERSION


def _mean(arrays: list, job: TrainingJob) -> np.ndarray:
    """The new model: accepted results summed in float64 in round order, divided, rounded to the job's precision.
    Pinned down exactly, because anyone rechecking the round must get the same bits: numpy's own mean works in
    whatever precision the arrays arrived in (float32 from a node's disk, float64 elsewhere)."""
    total = np.zeros(job.spec.n_params, dtype=np.float64)
    for a in arrays:
        total += np.asarray(a, dtype=np.float64)
    theta = job.spec.round(total / len(arrays))
    return job.spec.fresh(theta) if job.finetune else theta  # a fine-tune's units each start Adam afresh


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


def _pick(draw: Draw, candidates: list[str], k: int) -> list[str]:
    """Up to k of `candidates`, drawn without replacement."""
    candidates, out = list(candidates), []
    while candidates and len(out) < k:
        out.append(candidates.pop(draw.below(len(candidates))))
    return out


def _score(score) -> tuple[float, float]:
    """A scorer's report, [loss, accuracy]: a non-negative finite loss and an accuracy in [0, 1]."""
    try:
        loss, acc = (float(x) for x in score)
    except (TypeError, ValueError, OverflowError):
        raise ProtocolError("a score is [loss, accuracy]") from None
    if not (np.isfinite(loss) and loss >= 0 and 0 <= acc <= 1):
        raise ProtocolError("a score's loss is finite and not negative, and its accuracy in [0, 1]")
    return loss, acc


def _median(values: list[float]) -> float:
    v = sorted(values)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def score_cost(job: TrainingJob) -> int:
    """FLOPs to score a fine-tune's model on held-back data, from its spec alone (a replay may not have the data): a
    forward pass over MAX_VAL sequences, a third of what LoraSpec.flops_per_step counts for training them."""
    return job.spec.flops_per_step(MAX_VAL) // 3 if job.finetune else 0


def _piece(name: str) -> str:
    """The frozen layers a prep unit needs: a layers unit's piece ("layers-00-03" of "layers-00-03-c00012"), or a rows
    unit itself (it streams a table range and keeps nothing)."""
    return name.rsplit("-c", 1)[0] if name.startswith("layers") else name


def _spread(units: list) -> list:
    """Ready prep units taken a piece at a time, round robin, each piece's in its own order. Taking the graph's order
    as it is would fill rounds with the bottom piece's chunks, so every miner would fetch every piece in turn."""
    groups: dict[str, list] = {}
    for u in units:
        groups.setdefault(_piece(u.name), []).append(u)
    out, queues = [], list(groups.values())
    while queues:
        for q in queues:
            out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out


def _affinity(piece: str, addr: str) -> bytes:
    """A miner's standing for a piece (lower first): the same miners get a piece round after round, so each fetches
    the few pieces it's given once and keeps them, instead of every piece (rendezvous hashing)."""
    return h256(f"prep-piece|{piece}|{addr}".encode())


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
        # DeepSeek jobs' prep (deepseek_rounds.py): each job's PrepState, and the bucket a live node reads its units'
        # commitments from and writes the records' manifest to (node.py sets it); None without one.
        self.preps: dict = {}
        self.prep_store = None
        # Disputes: where states and evidence are kept (a BlobStore: the node's; None, a dict in memory), how the
        # referee reaches the base model's weights (job -> source; default dispute.HubWeights), and the seconds a
        # party has for each move.
        self.trace_blobs = None
        self.weight_source = None
        self.roots = None  # a replay's: where a dispute's checkpoints come from (.array(blob_id, job)), to check its states
        self.datasets = None  # a replay's: where a fine-tune's data comes from (.attach(job) or .attach_dataset(job))
        self.records = None  # a replay's: a DeepSeek job's bucket (job -> store), its disputes' records read from it
        self.move_s = 60.0
        # A replay's: an ES job's state the round started from ((job, start hash) -> state or None), so the replay works
        # the round's level and its update out again rather than taking the record's word (node._validate gives its own).
        self.es_states = None
        self._programs: dict = {}
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
            if job.data_source["kind"] in ("npz", "tokens", "deepseek") and self.recheck is not None:
                try:
                    self.recheck.attach(job)  # rechecking replays trains on the data
                except Exception:  # noqa: BLE001 - unavailable data only means those checks can't be redone
                    pass
            elif job.data_source["kind"] in ("npz", "tokens", "deepseek") and not self.replaying:  # plain replays never touch the data
                if self.store is None:
                    raise InvalidJob("no dataset store")
                self.store.attach_dataset(job)
            self.jobs[job_id] = job
        return self.jobs[job_id]

    def start_params(self, job_id: str) -> np.ndarray:
        if job_id not in self.params:
            saved = self.store.load_params(job_id) if self.store else None
            job = self.job(job_id)
            if job.es and job.spec.rule == es2.RULE2:
                saved = self._es2_rebuilt(job, saved)
            self.params[job_id] = saved if saved is not None else job.spec.init_params(job.init_seed)
        return self.params[job_id]

    def _es2_rebuilt(self, job: TrainingJob, saved):
        """A rule-2 Evo job's state (es2.py: its hash chain and control words, 24 numbers) worked out from the chain's
        blocks and this node's records of them (es2_rounds.rebuild_chain), each round checked against its block. A
        restart, or a node that didn't lead the job's last round, gets it this way; the saved copy only if that fails."""
        if self.store is None or not hasattr(self.store, "load_record"):
            return saved
        try:
            state, _, _ = es2_rounds.rebuild_chain(self.chain.blocks, self.store.load_record, job.job_id, job.spec)
        except (ValueError, KeyError, TypeError) as e:
            log.warning("couldn't rebuild Evo job %s's state from the chain: %s", job.job_id, e)
            return saved
        return state

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
                except Exception as e:  # noqa: BLE001 - one job's bad data must never stop every job's rounds
                    self.waiting[job_id] = f"job {job_id} can't run: its data or spec doesn't load ({type(e).__name__}: {e})"
                    continue
                if job.sharded and not self.runs_teams:
                    self.waiting[job_id] = (f"job {job_id} waits: it's split into {job.shards} shards, trained by teams "
                                            "whose workers send each other activations, and this node can't run teams yet")
                    continue
                ready.append(job_id)
        if not ready:
            return None
        # A starter job (in the genesis, owned by nobody: ledger.GENESIS_JOB) only fills rounds no paid job is ready for,
        # so it never takes turns away from jobs people pay for.
        paid = [j for j in ready if self.chain.state.job_owner.get(j) != SYSTEM]
        ready = paid or ready
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

    def _need(self, *phases: Phase) -> _Round:
        if self.phase not in phases or self.r is None:
            raise ProtocolError(f"not in {phases[0].value} phase (now {self.phase.value})")
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
            if self.replaying.get("version", RECORD_VERSION) not in (1, 2, SHARDED_RECORD_VERSION, FINETUNE_RECORD_VERSION,
                                                                     ES_RECORD_VERSION):
                raise ProtocolError(f"the record is version {self.replaying.get('version')!r}, which this code doesn't know")
        else:
            job_id = self.active_job_id()
            if job_id is None:
                return False
        job = self.job(job_id)
        if self.replaying and (self.replaying.get("version", RECORD_VERSION) == SHARDED_RECORD_VERSION) != job.sharded:
            raise ProtocolError(f"the record is version {self.replaying.get('version', RECORD_VERSION)}, but sharded jobs' "
                                f"rounds, and only theirs, are recorded as version {SHARDED_RECORD_VERSION}")
        if self.replaying and (self.replaying.get("version", RECORD_VERSION) == FINETUNE_RECORD_VERSION) != job.finetune:
            raise ProtocolError(f"the record is version {self.replaying.get('version', RECORD_VERSION)}, but fine-tunes' "
                                f"rounds, and only theirs, are recorded as version {FINETUNE_RECORD_VERSION}")
        if self.replaying and (self.replaying.get("version", RECORD_VERSION) == ES_RECORD_VERSION) != job.es:
            raise ProtocolError(f"the record is version {self.replaying.get('version', RECORD_VERSION)}, but ES jobs' "
                                f"rounds, and only theirs, are recorded as version {ES_RECORD_VERSION}")
        if self.replaying:
            start, start_hash = None, bytes.fromhex(self.replaying["start_hash"])
            if start_hash != self.expected_start_hash(job):
                raise ProtocolError("the round didn't start from the job's latest model")
            if job.es and self.es_states is not None:
                start = self.es_states(job.job_id, start_hash)
                if start is not None and params_hash(start) != start_hash:
                    start = None
            elif self.recheck is not None:
                start = self._replay_team_start(job, start_hash) if job.sharded else self._replay_start(job, start_hash)
        else:
            start = self.start_params(job_id)
            start_hash = params_hash(start)
        prep_units = {}
        if job.deepseek and (self.replaying.get("prep") is not None if self.replaying else not self.prep(job).done):
            # A prep round: its units are the work graph's ready ones (deepseek_rounds.py), or the record's, by id.
            recorded = self.replaying["prep"]["units"] if self.replaying else None
            names = [None] * len(recorded) if recorded is not None else \
                [u.name for u in _spread(self.prep(job).ready())[:min(dsr.MAX_PREP_UNITS, job.units_per_round)]]
            units = []
            for i, name in enumerate(names):  # each at the job's unit work: the ledger bounds a round's by it
                u = dataclasses.replace(WorkUnit.make(job, round_no, i, start_hash), memory_mb=PREP_MEMORY_MB)
                if recorded is not None:
                    if u.unit_id not in recorded:
                        raise ProtocolError("the record's prep units aren't the round's")
                    name = recorded[u.unit_id]
                units.append(u)
                prep_units[u.unit_id] = name
        else:
            units = [WorkUnit.make(job, round_no, i, start_hash) for i in range(job.units_per_round)]
        self.r = _Round(round_no, job_id, start, start_hash, {u.unit_id: u for u in units})
        self.r.prep = prep_units
        self.r.record = {"version": RECORD_VERSION, "round": round_no, "job_id": job_id, "start_hash": start_hash.hex(),
                         "rules": self.config.rules(), "registrations": [], "commitments": [], "reveals": [], "verdicts": [],
                         "checks": {"reveal": {}, "replay": {}, "arbitration": {}, "redundancy": {}, "model": {}}}
        if job.sharded:  # its teams (start_compute) and the signed edge messages that settled disputes (submit_edge_evidence)
            self.r.record |= {"version": SHARDED_RECORD_VERSION, "teams": [], "evidence": []}
        if job.finetune:  # its disputes (start_disputes): signed moves, evidence and the weights the referee used
            self.r.record |= {"version": FINETUNE_RECORD_VERSION, "games": []}
            self.r.record["checks"]["game"] = {}  # each dispute's root states, from the checkpoints
            # How its replicas are compared (verification.FINETUNE_CHECK). A replay takes the recorded rules instead
            # (below), and a record from before this has none: rule 1, which is recorded as it was, without one.
            if self.config.finetune_check != 1:
                self.r.record["rules"]["finetune_check"] = self.config.finetune_check
        if job.es and job.spec.rule == es2.RULE2:  # its step, its fitness (start_compute: its seed; _settle_es2)
            got = (self.replaying or {}).get("es", {})
            if start is not None:
                ctrl = es2.ctrl_of(start)
                step = int(ctrl[2])
                fitness = "replay" if job.spec.replay and ctrl[3] > 0 else "binary"
                if self.replaying and (got.get("step"), got.get("fitness")) != (step, fitness):
                    raise ProtocolError("the record's step or fitness isn't the one the round's starting state holds")
            else:
                step, fitness = got.get("step"), got.get("fitness")
            self.r.record |= {"version": ES_RECORD_VERSION,
                              "es": {"rule": es2.RULE2, "task": es2.task_at(job.spec, int(step or 0)), "step": step,
                                     "fitness": fitness}}
        elif job.es:  # its seed (start_compute), the level its problems are at, and the update (_settle_es)
            level = es.level_of(start) if start is not None else (self.replaying or {}).get("es", {}).get("level")
            if self.replaying and level != self.replaying.get("es", {}).get("level"):
                raise ProtocolError("the record's level isn't the one the round's starting state holds")
            self.r.record |= {"version": ES_RECORD_VERSION, "es": {"rule": job.spec.rule, "level": level}}
        if prep_units:  # a DeepSeek prep round: which unit of the work graph each of its units is
            self.r.record["prep"] = {"units": dict(prep_units)}
            self.r.record["checks"]["handoff"] = {}  # whether a rejected unit's outputs are its last state
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
        if r.es:
            # The round's seed, from what the assignments are drawn from: fixed when registration closes, unknown
            # before. Each unit's pairs' seeds and the round's problems follow from it (es.pair_seed, problems_seed).
            seed = int.from_bytes(h256(b"es-round", self.chain.tip.hash, r.round_no.to_bytes(8, "big"),
                                       *sorted(canonical(e) for e in r.record["registrations"]))[:8], "big")
            if self.replaying and f"{seed:016x}" != self.replaying.get("es", {}).get("seed"):
                raise ProtocolError("the record's ES seed isn't the one its registrations draw")
            r.record["es"]["seed"] = f"{seed:016x}"
            r.units = {uid: dataclasses.replace(u, seed=seed) for uid, u in r.units.items()}
        if r.v4 or r.es:
            able = [a for a, reg in r.regs.items() if staked(a) and reg.memory_mb >= job.memory_mb]
            if len(able) < FINETUNE_QUORUM:  # its work couldn't be checked: train nothing, waste nothing
                r.waiting = (f"job {r.job_id} waits: a fine-tune's work is checked by workers, so it needs "
                             f"{FINETUNE_QUORUM} staked workers with {_size(job.memory_mb)} of GPU memory each (one to "
                             f"train a unit, one to check it), and {len(able)} registered. Workers earn their stake on "
                             "other jobs first")
                self.waiting[r.job_id] = r.waiting
                r.unassigned = len(r.units)
                self.phase = Phase.COMPUTE
                return
            self.waiting.pop(r.job_id, None)

        def pick(unit: WorkUnit, exclude: set[str]) -> str | None:
            # Staked workers first: those earning their stake in only get what's left, so they can't crowd them out.
            # A prep unit goes to the miners its piece stands with (_affinity), so pieces stay where they're held.
            fits = [a for a in order if a not in exclude and r.regs[a].memory_mb >= unit.memory_mb
                    and used[a] + unit.flops <= capacity[a]]
            piece = _piece(r.prep[unit.unit_id]) if unit.unit_id in r.prep else None
            return min(fits, key=lambda a: (not staked(a), _affinity(piece, a) if piece else b"",
                                            (used[a] + unit.flops) / capacity[a])) if fits else None

        for unit in r.units.values():
            holders = []
            first = pick(unit, set())
            if first is None:
                r.unassigned += 1
                continue
            holders.append(first)
            # A prep unit has one holder: its objects in the bucket have one name, whoever writes them. An ES unit has
            # one too: its checkers re-check it, and the node re-scores every answer.
            if not r.prep and not r.es and draw.chance(self.config.replication_rate):
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

    @_locked
    def all_in(self) -> bool:
        """Whether everything this phase waits for is in, so the node can end it before its time runs out (but never
        sooner than its own length for the phase: Node._run_round). In compute, a commitment for every unit handed out
        (a worker commits once its unit's checkpoints, or a prep unit's outputs, are uploaded, so it has nothing left to
        do in the phase); in reveal, a reveal or a verdict for every commitment; in verify, a verdict on every task
        someone was given (one nobody could be given never gets one). Nothing the record could hold comes in after
        that: each is taken once, so a phase that ran its full time would record the same round. A phase with nothing to
        wait for is all in at once, as in a round nobody was handed work in: nobody registered who fits, or a
        fine-tune without its quorum (`waiting`). A sharded job's phases never are: its teams trade edges during
        compute and send edge evidence during the reveal (submit_edge_evidence)."""
        got = self.progress()
        return got is not None and got[0] == got[1]

    @_locked
    def progress(self) -> tuple[int, int] | None:
        """(in, of) of what this phase waits for (all_in): units committed of those handed out, commitments revealed or
        turned down, tasks answered of those given. /status shows it, so a dashboard can say how close an early end is.
        None where all_in never ends a phase: other phases, and a sharded job's."""
        r = self.r
        if r is None or self.job(r.job_id).sharded:
            return None
        if self.phase == Phase.COMPUTE:
            held = [(uid, a) for a, uids in r.assignments.items() for uid in uids]
            return sum(k in r.commitments for k in held), len(held)
        if self.phase == Phase.REVEAL:
            return sum(k in r.reveals or k in r.results for k in r.commitments), len(r.commitments)
        if self.phase == Phase.VERIFY:
            given = [t for t in r.tasks.values() if t.verifier]
            return sum(t.verdict is not None for t in given), len(given)
        return None

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
    def reveal(self, rv: Reveal, wire: dict | None = None, commitment: bytes | None = None) -> Verdict | None:
        """Cheap checks happen immediately. Returns a fraud verdict, or None if it goes on to replay.
        `wire` is the reveal as sent (with its checkpoints' blob ids), kept in the record. `commitment`: a prep
        unit's commitment, which the node read from the bucket (Node._prep_commitment; None if it couldn't)."""
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
        if rv.unit_id in r.prep:
            return self._prep_reveal(job, key, rv, wire, commitment)
        if r.es and job.spec.rule == es2.RULE2:
            return self._es2_reveal(job, key, rv, wire, commitment)
        if r.es:
            return self._es_reveal(job, key, rv, wire, commitment)
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

    def _prep_reveal(self, job: TrainingJob, key, rv: Reveal, wire: dict | None, raw: bytes | None) -> Verdict | None:
        """A prep unit's reveal: the root and salt its commitment hashed, the root being the sha256 of the unit's
        commitment in the bucket, which a live node has read (`raw`) and checks is that unit's
        (deepseek_rounds.check_commit)."""
        r = self.r
        r.record["reveals"].append(wire if wire is not None else
                                   {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(),
                                    "salt": rv.salt.hex(), "prep": True})
        if not check_reveal(r.commitments[key], rv):
            return self._fraud(key, Verdict.BAD_REVEAL, "reveal doesn't match commitment")

        def checks() -> str:
            if self.prep_store is None:
                return "ok"  # a replay: the record vouches for it
            gu = self.prep(job).graph.units[r.prep[rv.unit_id]]
            if raw is None:
                return "missing"
            why = self.prep(job).check_commit(gu, raw, rv.merkle_root)
            if why is None:
                r.prep_commits[key], r.prep_raw[key] = json.loads(raw), raw
            return "ok" if why is None else why
        outcome = self._claim("reveal", f"{rv.unit_id}/{rv.worker}", checks)
        if outcome == "missing":  # not slashed: an upload can fail; it's unpaid and the unit goes out again
            r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, "its commitment isn't in the bucket")
            return None
        if outcome != "ok":
            return self._fraud(key, Verdict.BAD_REVEAL, outcome)
        r.reveals[key] = rv
        return None

    def _es_reveal(self, job: TrainingJob, key, rv: Reveal, wire: dict | None, raw: bytes | None) -> Verdict | None:
        """An ES claim's reveal: the answers blob and the rewards its commitment's root binds (es.claim_root). The node
        re-scores every answer from the blob (`raw`, which a live node has read; es.rescore): rewards that aren't the
        answers' scores are fraud, slashed. A blob that never arrived leaves the claim unchecked: unpaid, unslashed."""
        r, spec = self.r, job.spec
        d = (wire or {}).get("es") or {}
        r.record["reveals"].append(wire if wire is not None else
                                   {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(),
                                    "salt": rv.salt.hex(), "es": d})
        rewards, answers = d.get("rewards"), d.get("answers")
        shaped = (isinstance(rewards, list) and len(rewards) == spec.pairs_per_unit
                  and all(isinstance(p, list) and len(p) == 2 and all(type(x) in (int, float) and 0 <= x <= 1 for x in p)
                          for p in rewards) and isinstance(answers, str) and len(answers) == 64)
        if not check_reveal(r.commitments[key], rv) or not shaped or rv.merkle_root != es.claim_root(answers, rewards):
            return self._fraud(key, Verdict.BAD_REVEAL, "reveal doesn't match commitment")
        rewards = [[float(a), float(b)] for a, b in rewards]

        def checks() -> str:
            if raw is None:
                if self.replaying:
                    raise ValueError("the answers aren't here")
                return "missing"
            got = es.rescore(spec, r.units[rv.unit_id].seed, r.record["es"]["level"], raw, rv.unit_id)
            return "bad_answers" if got is None else "ok" if got == rewards else "wrong_rewards"
        outcome = self._claim("reveal", f"{rv.unit_id}/{rv.worker}", checks)
        if outcome == "missing":
            r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, "its answers never arrived")
            return None
        if outcome != "ok":
            return self._fraud(key, Verdict.BAD_REVEAL, "its rewards aren't its answers' scores" if outcome == "wrong_rewards"
                               else "its answers aren't a unit's, in shape")
        r.reveals[key] = rv
        r.es_claims[key] = (answers, rewards)
        return None

    def _es2_reveal(self, job: TrainingJob, key, rv: Reveal, wire: dict | None, raw: bytes | None) -> Verdict | None:
        """A rule-2 Evo claim's reveal (es2_rounds.py): the answers blob, each member's right/wrong score and, when the
        round scores members on replay targets, its mean replay log-probability, all bound by the commitment's root.
        The node re-scores every answer from the blob (es2_rounds.rescore, the task's own check, no model): scores
        that aren't the answers' are fraud, slashed. The log-probabilities are the checkers' to recompute."""
        r, spec = self.r, job.spec
        d = (wire or {}).get("es") or {}
        r.record["reveals"].append(wire if wire is not None else
                                   {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(),
                                    "salt": rv.salt.hex(), "es": d})
        rewards, lps, answers = d.get("rewards"), d.get("lps"), d.get("answers")
        want_lps = r.record["es"]["fitness"] == "replay"
        if (not check_reveal(r.commitments[key], rv) or not isinstance(answers, str) or len(answers) != 64
                or not es2_rounds.shaped_claim(spec, rewards, lps, want_lps)
                or rv.merkle_root != es2_rounds.claim_root(answers, rewards, lps)):
            return self._fraud(key, Verdict.BAD_REVEAL, "reveal doesn't match commitment")
        rewards = [[float(a), float(b)] for a, b in rewards]
        lps = None if lps is None else [[float(a), float(b)] for a, b in lps]
        parsed = None

        def checks() -> str:
            nonlocal parsed
            if raw is None:
                if self.replaying:
                    raise ValueError("the answers aren't here")
                return "missing"
            parsed = es2_rounds.rescore(spec, r.units[rv.unit_id].seed, raw, rv.unit_id, int(r.record["es"]["step"] or 0))
            return "bad_answers" if parsed is None else "ok" if parsed[0] == rewards else "wrong_rewards"
        outcome = self._claim("reveal", f"{rv.unit_id}/{rv.worker}", checks)
        if outcome == "missing":
            r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, "its answers never arrived")
            return None
        if outcome != "ok":
            return self._fraud(key, Verdict.BAD_REVEAL, "its rewards aren't its answers' scores" if outcome == "wrong_rewards"
                               else "its answers aren't a unit's, in shape")
        r.reveals[key] = rv
        # (answers blob, rewards, lps, (answer ids, right or wrong) per member: the node's own reading, live only)
        r.es_claims[key] = (answers, rewards, lps, None if parsed is None else (parsed[1], parsed[2]))
        return None

    def prep(self, job: TrainingJob):
        """The job's prep state (deepseek_rounds.PrepState): from the node's store, then from each prep round."""
        if job.job_id not in self.preps:
            saved = self.store.load_meta(f"prep:{job.job_id}") if self.store and hasattr(self.store, "load_meta") else None
            self.preps[job.job_id] = dsr.PrepState.restore(job, saved)
        return self.preps[job.job_id]

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

        if r.prep:
            r.tasks = self._prep_tasks(job, draw, pool, task_id)
            self.phase = Phase.VERIFY
            return
        if r.es:
            r.tasks = self._es_tasks(job, draw, pool, task_id)
            self.phase = Phase.VERIFY
            return
        if r.v4:
            r.tasks = self._finetune_tasks(job, draw, pool, task_id)
            self.phase = Phase.VERIFY
            return
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

    def _finetune_tasks(self, job: TrainingJob, draw: Draw, pool, task_id) -> dict[str, VerifyTask]:
        """Version 4's verify tasks: each claim's challenged segments to CHECKERS checkers (never its unit's holders),
        canaries, and SCORERS scorers for the round's starting model when there's work to score. A claim nobody can
        check gets no task, and ends unchecked. Every draw is seeded by the round secret and what it's
        for (each claim's checkers by the claim), and claims are taken in sorted order, so the order a leader recorded
        the reveals in steers nothing."""
        r, q = self.r, self.config.challenge_segments
        tasks, checkers = [], defaultdict(set)
        for uid, addr in sorted(r.reveals):
            rv = r.reveals[(uid, addr)]
            segs = challenge_segments(r.beacon, uid, addr, job.n_segments, q)
            openings = {i: rv.openings[i] for s in segs for i in (s, s + 1)}
            mine = Draw(b"check", r.beacon, r.secret, uid.encode(), addr.encode())
            for verifier in _pick(mine, pool(set(r.holders[uid])), CHECKERS):
                checkers[uid].add(verifier)
                tasks.append(VerifyTask(task_id(uid, addr, verifier), verifier, r.units[uid], addr, segs, openings))
        tasks += self._finetune_canaries(job, Draw(b"canaries", r.beacon, r.secret), pool, checkers,
                                         self.config.canaries_per_round, "check")
        if r.reveals:  # (a round that trained nothing has nothing to score, and nobody to pay for scoring it)
            first = next(iter(r.units.values()))  # a score task names a unit only because every task has one
            for verifier in _pick(Draw(b"score", r.beacon, r.secret), pool(set()), SCORERS):
                tasks.append(VerifyTask(task_id("score", "", verifier), verifier, first, "", [], {}, kind="score"))
        tasks.sort(key=lambda t: t.task_id)
        return {tasks[i].task_id: tasks[i] for i in draw.shuffle(range(len(tasks)))}

    def _es_tasks(self, job: TrainingJob, draw: Draw, pool, task_id) -> dict[str, VerifyTask]:
        """An ES round's checks: each claim to CHECKERS staked checkers (never its unit's holder), who teacher-force the
        same beacon-chosen members of it (es.sample, the task's `segments`) and reject naming the first bad one. No
        canaries yet: approving without checking isn't caught (es.py's notes)."""
        r = self.r
        tasks = []
        for uid, addr in sorted(r.reveals):
            chosen = es.sample(job.spec, r.beacon, r.secret, uid, addr)
            mine = Draw(b"check", r.beacon, r.secret, uid.encode(), addr.encode())
            for verifier in _pick(mine, pool(set(r.holders[uid])), CHECKERS):
                tasks.append(VerifyTask(task_id(uid, addr, verifier), verifier, r.units[uid], addr, chosen, {}, kind="es"))
        tasks.sort(key=lambda t: t.task_id)
        return {tasks[i].task_id: tasks[i] for i in draw.shuffle(range(len(tasks)))}

    def es_task(self, t: VerifyTask) -> dict:
        """What a checker needs for an ES task besides its unit: the claim's answers blob and rewards, and the level
        (rule 1), or the claim's replay log-probabilities and the round's seed (rule 2)."""
        claim = self.r.es_claims[(t.unit.unit_id, t.submitter)]
        if self.r.record["es"].get("rule") == es2.RULE2:
            return {"answers": claim[0], "rewards": claim[1], "lps": claim[2], "seed": self.r.record["es"]["seed"]}
        answers, rewards = claim
        return {"answers": answers, "rewards": rewards, "level": self.r.record["es"]["level"]}

    def _prep_tasks(self, job: TrainingJob, draw: Draw, pool, task_id) -> dict[str, VerifyTask]:
        """A prep round's checks: each claim to CHECKERS staked checkers (never its unit's holders), who replay the unit
        and say whether its outputs hold, and canaries: an accepted-looking claim of one unit presented as another's
        (the donor's commitment under the target's unit id). An honest replay of the target makes other outputs and
        rejects it; approving it without replaying gets a checker slashed, as for a fine-tune's."""
        r = self.r
        tasks, checkers = [], defaultdict(set)
        for uid, addr in sorted(r.reveals):
            mine = Draw(b"check", r.beacon, r.secret, uid.encode(), addr.encode())
            for verifier in _pick(mine, pool(set(r.holders[uid])), CHECKERS):
                checkers[uid].add(verifier)
                tasks.append(VerifyTask(task_id(uid, addr, verifier), verifier, r.units[uid], addr, [], {}, kind="prep"))
        canaries, units = Draw(b"canaries", r.beacon, r.secret), sorted({uid for uid, _ in r.reveals})
        kinds = {uid: r.prep[uid].partition("-")[0] for uid in units}  # rows or layers (a replay has no graph)
        planted = 0
        for donor_uid, donor in canaries.shuffle(sorted(r.reveals)):
            if planted == self.config.canaries_per_round:
                break
            others = [u for u in units if u != donor_uid and kinds[u] == kinds[donor_uid]]
            if not others:
                continue
            target = others[canaries.below(len(others))]
            near = set(r.holders[target]) | set(r.holders[donor_uid]) | checkers[target] | checkers[donor_uid]
            candidates = pool(near)
            if not candidates:
                continue
            verifier = candidates[canaries.below(len(candidates))]
            tag = ("prep" + str(planted)).encode()
            tid = h256(b"canary", r.beacon, r.secret, tag, verifier.encode()).hex()[:20]
            tasks.append(VerifyTask(tid, verifier, r.units[target], donor, [], {}, canary=True, kind="prep",
                                    donor=donor_uid))
            planted += 1
        tasks.sort(key=lambda t: t.task_id)
        return {tasks[i].task_id: tasks[i] for i in draw.shuffle(range(len(tasks)))}

    def prep_task(self, t: VerifyTask) -> dict:
        """What a checker needs for a prep task: the unit to replay, and the commitment to check against (the
        submitter's, or a canary's donor's), its bytes and root. The bytes come inline, with no object name or donor
        id that would tell a canary from a claim (the fourth review's finding 6)."""
        r = self.r
        key = (t.donor or t.unit.unit_id, t.submitter)
        rv = r.reveals.get(key)
        return {"prep": r.prep[t.unit.unit_id], "commitment": r.prep_raw.get(key, b"").decode(),
                "root": rv.merkle_root.hex() if rv is not None else ""}

    def _finetune_canaries(self, job: TrainingJob, draw: Draw, pool, checkers: dict[str, set[str]], n: int,
                           kind: str) -> list[VerifyTask]:
        """A fine-tune's canaries: another unit's honest training, presented as this unit's. A donor claim's challenged
        openings (real Adam steps, so every invariant of Adam's moments holds, which a rescaled update breaks) go out
        under a target unit's id, and so its seed: an honest replay draws other batches and rejects them, and without
        a replay nothing gives it away. The checker neither holds nor checks either unit, so it can't have seen those
        checkpoints. Needs two units revealed."""
        r, q = self.r, self.config.challenge_segments
        units = sorted({uid for uid, _ in r.reveals})
        planted, used = [], set()
        if len(units) < 2:
            return planted
        for donor_uid, donor in draw.shuffle(sorted(r.reveals)):
            if len(planted) == n:
                break
            others = [u for u in units if u != donor_uid and u not in used]
            if not others or donor_uid in used:
                continue
            target = others[draw.below(len(others))]
            near = set(r.holders[target]) | set(r.holders[donor_uid]) | checkers[target] | checkers[donor_uid]
            candidates = pool(near)
            if not candidates:
                continue
            used.update((target, donor_uid))
            tag = (kind + str(len(planted))).encode()
            fake = KeyPair.generate(h256(b"canary", r.secret, tag)).address
            segs = challenge_segments(r.beacon, donor_uid, donor, job.n_segments, q)
            rv = r.reveals[(donor_uid, donor)]
            openings = {} if self.replaying else {i: rv.openings[i] for s in segs for i in (s, s + 1)}
            verifier = candidates[draw.below(len(candidates))]
            tid = h256(b"canary", r.beacon, r.secret, tag, verifier.encode()).hex()[:20]
            planted.append(VerifyTask(tid, verifier, r.units[target], fake, segs, openings, canary=True, kind=kind))
        return planted

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
        r = self.r
        if r is None or self.phase != Phase.VERIFY:
            raise ProtocolError(f"not in verify phase (now {self.phase.value})")
        return [t for t in r.tasks.values() if t.verifier == addr and t.verdict is None]

    @_locked
    def submit_verdict(self, addr: str, task_id: str, ok: bool, bad_segment: int | None, envelope: dict | None = None,
                       score=None) -> None:
        """`envelope` is the verifier's signed message, kept in the record as proof of what it said. A fine-tune's
        scorer answers with `score`, [loss, accuracy] of the round's starting model on the held-back data."""
        r = self.r
        if r is None or self.phase != Phase.VERIFY:
            raise ProtocolError(f"not in verify phase (now {self.phase.value})")
        t = r.tasks.get(task_id)
        if t is None or t.verifier != addr:
            raise ProtocolError("no such task for this verifier")
        if t.verdict is not None:
            raise ProtocolError("verdict already submitted")
        if t.kind == "score":
            t.score = _score(score)
            ok, bad_segment = True, None
        elif score is not None:
            raise ProtocolError("only a score task answers with a score")
        if t.kind == "prep":  # a prep unit is replayed whole: a rejection names no segment
            bad_segment = None
        elif not ok and bad_segment not in t.segments:
            raise ProtocolError("a rejection must name one of the challenged segments")
        t.verdict = (bool(ok), None if ok or bad_segment is None else int(bad_segment))
        entry = {"verifier": addr, "task_id": task_id, "ok": bool(ok), "bad_segment": t.verdict[1]}
        if t.score is not None:
            entry["score"] = list(t.score)
        r.record["verdicts"].append(envelope if envelope is not None else entry)

    @_locked
    def start_disputes(self, now: float | None = None) -> bool:
        """A fine-tune's round, after the verify phase: open a dispute (dispute.Game) for every rejection of a claim by
        a checker whose verdicts count, rooted at the segment it named, and, when replicas of a unit disagree, one each
        way between them, rooted at the whole unit. A checker that approved a canary is lazy: none of its verdicts
        count. Say whether there's a dispute phase; other rounds have none. With `now`, moves have deadlines."""
        r = self._need(Phase.VERIFY)
        if not r.v4:
            return False
        job = self.job(r.job_id)
        r.void = {t.verifier for t in r.tasks.values() if t.canary and t.verdict is not None and t.verdict[0]}
        opened = defaultdict(int)
        if r.prep and PREP_GAMES:
            self._prep_disputes(job, opened)
        for t in sorted(r.tasks.values(), key=lambda t: t.task_id):
            if t.kind == "check" and not t.canary and t.verdict is not None and not t.verdict[0] and t.verifier not in r.void:
                # One challenger can't load a round with disputes, fewer the more it has accused in vain: the rest of
                # its rejections are ignored (their claims stand on other verdicts).
                if opened[t.verifier] >= games_allowed(self.chain.state.rep(t.verifier)):
                    continue
                opened[t.verifier] += 1
                self._open_game(job, (t.unit.unit_id, t.submitter), t.verifier, "dispute", ["segment", t.verdict[1]])
        for uid, holders in sorted(r.holders.items()):
            keys = [(uid, a) for a in holders if (uid, a) in r.reveals]
            if len(keys) > 1 and not self._claim("redundancy", uid, lambda keys=keys: self._replicas_agree(job, keys)):
                for d, c in itertools.permutations(keys, 2):
                    self._open_game(job, d, c[1], "replicas", ["unit"])
        if not r.games:
            return False
        for g in r.games.values():
            g.deadline = 0.0 if now is None else now + self.move_s
        self.phase = Phase.DISPUTE
        return True

    def _program(self, job: TrainingJob, uid: str):
        from .program import Program

        key = (job.job_id, uid)
        if key not in self._programs:
            if getattr(job, "_data", None) is None:
                self._attach(job)
            if uid in self.r.prep:  # a prep unit's (deepseek_program.PrepProgram)
                from .deepseek_program import PrepProgram
                self._programs[key] = PrepProgram.of(job.spec, job.data, self.prep(job).graph.units[self.r.prep[uid]])
            else:
                self._programs[key] = Program.of(job, self.r.units[uid].seed)
        return self._programs[key]

    def _prep_disputes(self, job: TrainingJob, opened: dict) -> None:
        """A prep round's rejected layers units (rows units are copies of the tables: a rejected one ends unchecked).
        First the node checks a claim's outputs are its last committed state (deepseek_graph.handed_root, or
        records_root for the last piece: its objects decoded in NumPy): if not, the claim is fraud, its catcher the
        checker. If they are, the dispute is over its computation: a game rooted at the whole unit, from its first
        committed state to its last, layer by layer, then op by op."""
        r = self.r
        for t in sorted(r.tasks.values(), key=lambda t: t.task_id):
            if t.kind != "prep" or t.canary or t.verdict is None or t.verdict[0] or t.verifier in r.void:
                continue
            key = (t.unit.unit_id, t.submitter)
            if not r.prep[key[0]].startswith("layers") or key not in r.reveals or key in r.results:
                continue

            def check(key=key):  # (a replay takes the outcome from the record: it needn't read the job's text)
                return self._handoff(job, key, self.prep(job).graph.units[r.prep[key[0]]])
            outcome = self._claim("handoff", f"{key[0]}/{key[1]}", check)
            if outcome.startswith("unchecked") and key not in r.results:  # the bucket's fault as likely as the miner's
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, outcome)
            elif outcome != "ok" and key not in r.results:
                self._fraud(key, Verdict.FAILED_REPLAY, outcome, catcher=t.verifier, verifier=t.verifier)
            if outcome != "ok":
                continue
            if opened[t.verifier] >= games_allowed(self.chain.state.rep(t.verifier)):  # as for a training round's
                continue
            opened[t.verifier] += 1
            self._open_prep_game(job, key, t.verifier)

    def _handoff(self, job: TrainingJob, key, gu) -> str:
        """"ok" if a layers unit's outputs in the bucket are the last state it committed to, else why not: fraud if
        an object isn't the one committed to (Tampered) or the state isn't, "unchecked: ..." if they can't be read."""
        from .deepseek_graph import Missing, Tampered, handed_root, records_root
        from .deepseek_program import PrepProgram

        commit = self.r.prep_commits.get(key)
        if commit is None or self.prep_store is None:
            return "ok"
        state, store = self.prep(job), self.prep_store(job)
        want = {**state.want(), **commit["outputs"]}
        try:
            if gu.b < job.spec.top.first:
                above = next(u for u in state.graph.units.values()
                             if u.kind == "layers" and u.chunk == gu.chunk and u.a == gu.b)
                root = handed_root(store, PrepProgram.of(job.spec, job.data, above), want)
            else:
                root = records_root(store, PrepProgram.of(job.spec, job.data, gu), want)
        except Tampered as e:
            return f"its outputs aren't the objects it committed to: {e}"
        except (Missing, OSError, KeyError, ValueError) as e:
            return f"unchecked: its outputs can't be read: {e}"
        return "ok" if root == commit["roots"][-1] else "its outputs aren't the last state it committed to"

    def _open_prep_game(self, job: TrainingJob, key, challenger: str) -> None:
        r = self.r
        uid, defender = key
        gid = h256(b"game", r.beacon, r.secret, canonical([uid, defender, challenger, ["unit"]])).hex()[:20]
        if gid in r.games:
            return

        def ends():
            roots = r.prep_commits[key]["roots"]
            return [roots[0], roots[-1]]
        a, b = self._claim("game", gid, ends)
        g = dg.Game(gid, uid, defender, challenger, "dispute", ["unit"], a, b, seed=r.units[uid].seed)
        r.games[gid] = g.start(self._program(job, uid))

    def _attach(self, job: TrainingJob) -> None:
        """A replay needs a fine-tune's data to referee its disputes (which sequences a step trains on, and how long
        they are); a plain replay never needed it otherwise."""
        for src in (self.recheck, self.datasets, self.store):
            if src is None:
                continue
            try:
                (src.attach if hasattr(src, "attach") else src.attach_dataset)(job)
                return
            except Exception:  # noqa: BLE001 - try the next place
                continue
        raise ProtocolError(f"refereeing job {job.job_id}'s disputes needs its dataset, which isn't here")

    def _open_game(self, job: TrainingJob, key: tuple[str, str], challenger: str, why: str, root: list) -> None:
        r = self.r
        uid, defender = key
        gid = h256(b"game", r.beacon, r.secret, canonical([uid, defender, challenger, root])).hex()[:20]

        def ends():
            rv = r.reveals[key]
            if root[0] == "segment":
                s = int(root[1])
                before, after = rv.openings[s][0], rv.openings[s + 1][0]
            else:
                before, after = r.start, rv.final_params
            self._weights(before, after)
            out = []
            for theta in (before, after):
                state, manifest, blobs = dg.checkpoint_state(job.spec, theta)
                for raw in blobs.values():
                    self._put_trace(raw)
                self._put_trace(manifest)
                out.append(state)
            return out
        if self.replaying and self.roots is not None:  # a validator: the states are the checkpoints', or it refuses
            a, b = self._replay_ends(job, key, root)
            if self.replaying["checks"].get("game", {}).get(gid) != [a, b]:
                raise ProtocolError(f"dispute {gid}'s starting states in the record aren't the reveal's checkpoints'")
            r.record["checks"]["game"][gid] = [a, b]
        else:
            a, b = self._claim("game", gid, ends)
        g = dg.Game(gid, uid, defender, challenger, why, list(root), a, b, seed=r.units[uid].seed)
        if root == ["unit"] and not self.replaying:
            g.ready = self._checkpoint_states(job, r.reveals[key])
        r.games[gid] = g.start(self._program(job, uid))

    def _checkpoint_states(self, job: TrainingJob, rv) -> list[str]:
        """The states at a unit's segment boundaries that its reveal opened, put with the dispute's states: a split at
        the unit names exactly these (its children are the segments), so the defender needn't send again what it
        uploaded as checkpoints during compute. Round 7280 on the public network: a defender sending a unit's split
        tensor by tensor from home missed its move. Live rounds only: a replay rebuilds no states but the game's ends."""
        out = []
        for i in sorted(rv.openings):
            if 0 < i < job.n_segments:
                state, manifest, blobs = dg.checkpoint_state(job.spec, rv.openings[i][0])
                for raw in blobs.values():
                    self._put_trace(raw)
                self._put_trace(manifest)
                out.append(state)
        return out

    def _replay_bytes(self, blob_id: str) -> bytes | None:
        """A dispute's evidence, which the record names by id: from self.roots (the leader's blobs, for a validator).
        None when it can't be had: a validator then refuses the block (submit_move), a plain replay counts it
        unchecked."""
        get = getattr(self.roots, "bytes", None)
        if get is None:
            return None
        try:
            return get(blob_id)
        except Exception:  # noqa: BLE001 - pruned or unreachable: as a checkpoint that's gone
            return None

    def _replay_ends(self, job: TrainingJob, key: tuple[str, str], root: list) -> list[str]:
        """A game's starting and ending state roots, rebuilt from the checkpoints its claim's reveal names (fetched
        through self.roots: the leader's blobs, for a validator). Refuses when they can't be had."""
        uid, worker = key
        wire = next((d for d in self.replaying.get("reveals", []) if d.get("unit_id") == uid and d.get("worker") == worker),
                    None)
        if wire is None:
            raise ProtocolError(f"the record has no reveal for {uid} by {worker}")
        openings = blob_openings_in(wire["openings"], job.n_segments)
        if root[0] == "segment":
            s = int(root[1])
            ids = [openings[s][0], openings[s + 1][0]]
        else:
            ids = [openings[0][0], wire["final"]]
        out = []
        for bid in ids:
            try:
                theta = self.roots.array(bid, job)
            except Exception as e:  # noqa: BLE001 - the checkpoint couldn't be had: nothing to check the record by
                raise ProtocolError(f"checkpoint {str(bid)[:12]} for a dispute couldn't be fetched: {e}") from None
            out.append(dg.checkpoint_state(job.spec, theta)[0])
        return out

    # --- states and evidence on the node ---

    def _put_trace(self, raw: bytes) -> str:
        if self.trace_blobs is None:
            self.trace_blobs = _MemoryBlobs()
        return self.trace_blobs.put_bytes(raw)

    def _get_trace(self, blob_id: str) -> bytes | None:
        return None if self.trace_blobs is None else self.trace_blobs.get_bytes(blob_id)

    def _trace_size(self, blob_id: str) -> int | None:
        return None if self.trace_blobs is None else self.trace_blobs.size(blob_id)

    def _check_evidence_size(self, g, blob_id: str, outside) -> None:
        """Evidence no bigger than its tile can need (dispute.evidence_cap), checked before it's read."""
        size = self._trace_size(blob_id) if dg._hex64(blob_id) else None
        if size is None:
            raise ProtocolError("evidence is the id of an uploaded blob")
        if size > dg.MAX_EVIDENCE:
            try:
                cap = dg.evidence_cap(g.leaf, g.tile, outside)
            except dg.WeightsUnavailable:  # the referee will say so (unchecked), on evidence it then doesn't read
                return
            if size > cap:
                raise ProtocolError(f"evidence of {size} bytes is more than this tile can need ({cap})")

    def _check_states(self, roots: list[str]) -> None:
        """Every state a split names is on the node, with every tensor it names, each hashing as its manifest says:
        the challenger can fetch them all. Sizes are checked before anything is read or hashed."""
        from . import merkle

        for root in roots:
            size = self._trace_size(root)
            if size is None:
                raise dg.MoveRefused(f"state {root[:12]}… hasn't been uploaded")
            if size > merkle.MAX_MANIFEST:
                raise dg.MoveRefused(f"state {root[:12]}… is too big to be a manifest")
            m = self._get_trace(root)
            if m is None:
                raise dg.MoveRefused(f"state {root[:12]}… hasn't been uploaded")
            try:
                entries = merkle.parse_manifest(m)
            except merkle.MerkleError as e:
                raise dg.MoveRefused(f"state {root[:12]}…: {e}") from None
            for name, (troot, blob, rows, cols) in entries.items():
                if self.trace_blobs.tensor_root(blob, rows, cols) != troot:
                    raise dg.MoveRefused(f"state {root[:12]}…: {name} isn't uploaded, or doesn't hash as its manifest says")

    def _weight_source(self, job: TrainingJob):
        if job.deepseek:  # its weights, and its records as record.* (referee_deepseek.RecordSource)
            from . import referee_deepseek as rd
            from .deepseek_job import weight_source
            self._records(job)
            weights = self.weight_source(job) if self.weight_source is not None else weight_source(job.spec)
            return rd.Sources(weights, rd.RecordSource(job.data))
        if self.weight_source is not None:
            return self.weight_source(job)
        return dg.HubWeights(job.data.parsed_manifest())

    def _records(self, job: TrainingJob) -> None:
        """A DeepSeek job's records for the referee (referee_deepseek's ds_record), read from the bucket and checked
        against the root its prep pinned (deepseek_graph.RecordReader): set once, on a node with the bucket (or a
        replay given one, self.records)."""
        bucket = self.prep_store or self.records
        if not job.deepseek or getattr(job.data, "records", None) is not None or bucket is None:
            return
        state = self.prep(job)
        if state.root is None:
            return
        store = bucket(job)
        manifest = json.loads(store.get("manifest.json"))
        job.data.records = dsg.RecordReader(store, manifest, state.root)

    @_locked
    def game_views(self) -> list[dict]:
        r = self.r
        return [] if r is None else [g.view() for _, g in sorted(r.games.items())]

    @_locked
    def submit_move(self, addr: str, body: dict, envelope: dict | None = None, now: float | None = None,
                    evidence: bytes | None = None) -> dict:
        """One move of a dispute (dispute.play), `body` {"game", "n" (moves so far), "move"}. A replay passes the
        evidence from the record; a live node reads it from its blobs. Either way the referee reads the base model's
        weights it checked itself against the job's manifest (dispute.HubWeights): none come from the record."""
        r = self._need(Phase.DISPUTE)
        g = r.games.get(body.get("game")) if isinstance(body, dict) else None
        if g is None:
            raise ProtocolError("no such dispute in this round")
        if body.get("n") != len(g.moves):
            raise ProtocolError(f"this dispute has had {len(g.moves)} moves; that one was for another turn")
        if now is not None and g.deadline and now > g.deadline:
            raise ProtocolError("too late: that move's deadline has passed")
        job = self.job(r.job_id)
        move, outside = body.get("move"), None
        try:
            kind = dg.whose(g, addr, move)  # before reading anything the move names
        except dg.MoveRefused as e:
            raise ProtocolError(str(e)) from None
        if kind == "evidence":
            outside = dg.Outside(job.spec, job.data, self._weight_source(job))
            blob = str(move["evidence"])
            if not self.replaying:
                self._check_evidence_size(g, blob, outside)
            evidence = self._replay_bytes(blob) if self.replaying else self._get_trace(blob)
            if self.replaying and evidence is None:
                if getattr(self.roots, "bytes", None) is not None:  # a validator never takes an outcome on trust
                    raise ProtocolError(f"dispute {g.gid}'s evidence {blob[:12]}… can't be fetched to referee it again")
                # A plain replay with no blobs to fetch from (an old round, past its window): the record's outcome,
                # counted unchecked.
                entry = next((e for e in self.replaying.get("games", []) if e.get("id") == g.gid), {})
                g.end(str(entry.get("outcome", "unchecked")), str(entry.get("detail", "")))
                self.unchecked += 1
                g.moves.append(envelope if envelope is not None else {"from": addr, **body})
                return g.view()
        try:
            if not self.replaying and g.unit in r.prep and not g.moves:  # a prep game's ends: the defender's to post
                self._check_states([g.a, g.b])
            dg.play(g, self._program(job, g.unit), addr, move, evidence=evidence, outside=outside,
                    check_states=None if self.replaying else self._check_states)
        except dg.MoveRefused as e:
            raise ProtocolError(str(e)) from None
        g.moves.append(envelope if envelope is not None else {"from": addr, **body})
        if now is not None:
            g.deadline = now + self.move_s
        return g.view()

    @_locked
    def expire_moves(self, now: float) -> None:
        """A party whose move is past its deadline loses (dispute.silence)."""
        r = self.r
        if r is not None and self.phase == Phase.DISPUTE:
            for g in r.games.values():
                if g.turn != "done" and g.deadline and now > g.deadline:
                    dg.silence(g)

    @_locked
    def end_disputes(self, now: float) -> None:
        """The dispute phase is over: each unfinished game is judged by its own move's deadline (dispute.cut_short)."""
        r = self.r
        if r is not None and self.phase == Phase.DISPUTE:
            most = self.job(r.job_id).phase_lengths().get("moves", 0)
            for g in r.games.values():
                dg.cut_short(g, now, most)

    def disputes_done(self) -> bool:
        r = self.r
        return r is None or all(g.turn == "done" for g in r.games.values())

    def _replicas_agree(self, job: TrainingJob, keys: list) -> bool:
        """Whether replicas of a unit ended at the same weights, by the round's own rule (verification.FINETUNE_CHECK):
        rule 2's replicas_agree, or rule 1's agreement rule loosened to what a whole unit on other hardware measured
        (finetune.WHOLE_TOLERANCE). No training."""
        r = self.r
        finals = [r.reveals[k].final_params for k in keys]
        self._weights(r.start, *finals)
        if r.record["rules"].get("finetune_check", 1) >= 2:
            return all(replicas_agree(f, finals[0], r.start, job.spec, job.lr, job.steps_per_unit) for f in finals[1:])
        tol = max(self.config.agreement_tolerance, WHOLE_TOLERANCE)
        return all(reproduces(f, finals[0], r.start, tol, job.spec) for f in finals[1:])

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
        if r.es and job.spec.rule == es2.RULE2:
            return self._settle_es2(proposer)
        if r.es:
            return self._settle_es(proposer)
        if r.v4:
            return self._settle_finetune(proposer)
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

    def _settle_finetune(self, proposer: str):
        """Version 4's settlement, from verdicts and disputes: the node replays nothing. A claim is accepted when a
        checker whose verdicts count approved it, or when it won every dispute over it; found bad, when it lost one
        (its worker slashed, the winner its catcher). A dispute the referee couldn't decide (the Hub unreachable)
        leaves it unchecked: unpaid, out of the model, nobody slashed. So does a claim no counted checker answered.
        A challenger that lost is slashed for a false accusation, with no catcher: baiting one into a dispute earns
        nothing. A checker that approved a canary is slashed, and none of its verdicts count. Approving a claim found
        bad isn't slashed: the referee finds an op a few ulps out that a checker's replay rightly passes. Scorers'
        median goes into the record, for people: nothing is paid by it."""
        r = self.r
        job = self.job(r.job_id)
        most = job.phase_lengths().get("moves", 0)
        for g in r.games.values():
            # The leader's clock: its phase ended before the move was due, which it can be only near the game's end.
            if g.turn != "done" and self.replaying and len(g.moves) >= most - 1:
                entry = next((e for e in self.replaying.get("games", []) if e.get("id") == g.gid), {})
                if (entry.get("outcome"), entry.get("detail")) == ("unchecked", dg.PHASE_ENDED):
                    g.end("unchecked", dg.PHASE_ENDED)
            dg.silence(g)  # whoever's move it still is didn't make it
        verifier_flops: dict[str, int] = defaultdict(int)
        missed: set[str] = set()
        answered, scores = defaultdict(list), []
        for t in r.tasks.values():
            if t.verdict is None:
                if t.verifier:
                    missed.add(t.verifier)
                continue
            if t.kind == "score":
                scores.append((t.verifier, t.score))
                continue
            # A prep unit's checker replays all of it. Its rejection isn't refereed in the pilot (PREP_GAMES), so it's
            # never upheld, and unpaid: rejecting costs a griefer its pay (the fourth review's finding 8).
            if t.kind != "prep" or t.canary or t.verdict[0]:
                verifier_flops[t.verifier] += t.unit.flops if t.kind == "prep" else replay_cost(job, len(t.segments))
            if t.canary:
                if t.verdict[0]:
                    r.slashes.append([t.verifier, None, "approved a canary"])
                continue
            if t.verifier not in r.void:
                answered[(t.unit.unit_id, t.submitter)].append(t)
        games = defaultdict(list)
        for _, g in sorted(r.games.items()):
            games[(g.unit, g.defender)].append(g)
            if g.outcome == "defender":
                r.slashes.append([g.challenger, None, "false accusation"])
        # A checker that withdrew its rejection (it found no number the referee would refuse) isn't slashed, but isn't
        # paid for checking this round either: a rejection costs something even when it's taken back.
        withdrawn = {g.challenger for g in r.games.values() if g.outcome == "withdrawn"}
        for key in sorted(r.reveals):
            if r.prep and key in r.results:  # found bad already: its outputs aren't its state (_prep_disputes)
                continue
            lost = [g for g in games[key] if g.outcome == "challenger"]
            rejected = bool(r.prep) and (not PREP_GAMES or r.prep[key[0]].startswith("rows")) \
                and any(not t.verdict[0] for t in answered[key])
            if rejected:  # (no game settles it, so nobody is paid or slashed for it, and it's redone)
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None,
                                            "a checker rejected it, and prep units aren't refereed in the pilot")
            elif lost:
                g = lost[0]
                self._fraud(key, Verdict.FAILED_REDUNDANCY if g.why == "replicas" else Verdict.FAILED_REPLAY,
                            f"lost a dispute ({g.why}): {g.detail}", catcher=g.challenger, verifier=g.challenger)
            elif any(g.outcome == "unchecked" for g in games[key]):
                why = next(g.detail for g in games[key] if g.outcome == "unchecked")
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, f"not refereed: {why}")
            elif any(t.verdict[0] for t in answered[key]) or any(g.outcome == "defender" for g in games[key]):
                first = next((t.verifier for t in answered[key] if t.verdict[0]), None)
                won = sum(g.outcome == "defender" for g in games[key])
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.ACCEPTED, first,
                                            f"won {won} dispute{'s' * (won != 1)}" if won else "")
            else:
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None,
                                            "no checker whose verdicts count answered for it")
        r.record["games"] = [g.entry() for _, g in sorted(r.games.items())]
        # An offender pays one slash a round, its first (ledger._slash_txs): one with a catcher goes before a burn, or
        # a cheater that also accused falsely (or approved a canary) would keep its catcher from the bounty.
        r.slashes.sort(key=lambda s: s[1] is None)

        accepted = {k: r.reveals[k] for k, res in r.results.items() if res.verdict == Verdict.ACCEPTED}
        offenders = {s[0] for s in r.slashes} | r.forfeit
        worker_flops: dict[str, int] = defaultdict(int)
        updates: dict[str, np.ndarray] = {}
        if r.prep:  # a prep round: the work graph moves on, the model doesn't
            return self._settle_prep(job, accepted, offenders, verifier_flops, missed, proposer)
        for (uid, addr), rv in accepted.items():
            if addr not in offenders:
                worker_flops[addr] += r.units[uid].flops
                updates.setdefault(uid, rv.final_params)
        new_params = None

        def average() -> str:
            nonlocal new_params
            self._weights(*(updates.values() if updates else [r.start]))
            new_params = _mean(list(updates.values()), job) if updates else r.start
            return params_hash(new_params).hex()
        model_hash = self._claim("model", "params_hash", average)
        if scores:
            r.record["score"] = [_median([s[0] for _, s in scores]), _median([s[1] for _, s in scores])]
            # Paid only when SCORERS answered and this one agrees with their median: nothing checks a lone score.
            loss, acc = r.record["score"]
            for v, (l, a) in scores if len(scores) >= SCORERS else []:
                if abs(l - loss) <= SCORE_AGREEMENT * max(1.0, loss) and abs(a - acc) <= SCORE_AGREEMENT:
                    verifier_flops[v] += score_cost(job)
        r.record["secret"] = r.secret.hex()
        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(updates),
            worker_flops=dict(sorted(worker_flops.items())),
            verifier_flops={a: f for a, f in sorted(verifier_flops.items()) if a not in offenders | withdrawn},
            slashes=r.slashes,
            accepted=sorted(worker_flops),
            missed=sorted(missed | r.forfeit | withdrawn | {res.worker for res in r.results.values() if res.verdict == Verdict.MISSED_DEADLINE}),
            params_hash=model_hash,
            proposer=proposer,
            record_hash=h256(canonical(r.record)).hex(),
        )
        return summary, new_params, worker_flops

    def _settle_es(self, proposer: str):
        """An ES round's settlement. A claim is accepted when a checker approved it and none rejected it. A rejection
        isn't refereed yet (that needs the model on the node, or a dispute game over one token's logits), so a rejected
        claim is unchecked: unpaid, out of the update, nobody slashed, and the rejection unpaid, so rejecting honest
        work costs a griefer its pay. A claim nobody answered for is unchecked too. Rewards that aren't the answers'
        scores were slashed at the reveal. The update is the accepted claims' pairs, [j, r+, r−] sorted by j, which the
        record keeps, and the new state es.step's: worked out here, and again by any replay that has the state the
        round started from (es_states), never taken from the record."""
        r = self.r
        job = self.job(r.job_id)
        spec = job.spec
        verifier_flops: dict[str, int] = defaultdict(int)
        missed: set[str] = set()
        answered = defaultdict(list)
        for t in r.tasks.values():
            if t.verdict is None:
                if t.verifier:
                    missed.add(t.verifier)
                continue
            if t.verdict[0]:
                verifier_flops[t.verifier] += spec.check_flops()
            answered[(t.unit.unit_id, t.submitter)].append(t)
        for key in sorted(r.reveals):
            if key in r.results:
                continue
            got = answered[key]
            if any(not t.verdict[0] for t in got):
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None,
                                            "a checker found an answer that isn't its model's greedy choice (not refereed "
                                            "yet: unpaid, out of the update)")
            elif got:
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.ACCEPTED, got[0].verifier)
            else:
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, "no checker answered for it")
        offenders = {s[0] for s in r.slashes} | r.forfeit
        worker_flops: dict[str, int] = defaultdict(int)
        update, units = [], set()
        for key, res in sorted(r.results.items()):
            if res.verdict != Verdict.ACCEPTED or key[1] in offenders:
                continue
            unit = r.units[key[0]]
            worker_flops[key[1]] += unit.flops
            units.add(key[0])
            for k, (rp, rm) in enumerate(r.es_claims[key][1]):
                update.append([unit.index * spec.pairs_per_unit + k, rp, rm])
        update.sort(key=lambda u: u[0])
        seed = int(r.record["es"]["seed"], 16)
        r.record["es"]["update"] = update
        if update:
            mean = es.mean_reward([x for _, rp, rm in update for x in (rp, rm)])
            r.record["score"] = [1.0 - mean, mean]
        new_params = None

        def average() -> str:
            nonlocal new_params
            self._weights(r.start)
            new_params = es.step(spec, r.start, seed, update)
            return params_hash(new_params).hex()
        if r.start is not None:  # live, or a replay given the starting state: worked out, never read from the record
            model_hash = average()
            r.record["checks"]["model"]["params_hash"] = model_hash
        else:  # a replay without it: the record's (es.rebuild_chain works the states out from the chain itself)
            model_hash = self._claim("model", "params_hash", average)
        r.record["secret"] = r.secret.hex()
        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(units),
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

    def _settle_es2(self, proposer: str):
        """A rule-2 Evo round's settlement (es2.py). Claims are accepted, left unchecked or slashed as in rule 1
        (_settle_es). The accepted claims' members, in pair order, give the pair coefficients (es2.coefficients: z of
        their replay log-probabilities when the round scores them, else of their right/wrong scores), the step's scale
        is the cosine's lr over the members counted, and the round's targets are each problem's most common correct
        answer among the accepted members' (es2.replay_targets, from the node's own reading of their blobs; a replay
        takes the record's). The entry (es2.entry) moves the state's hash chain on (es2.step_state): the node keeps no
        weights, and the summary's params_hash is the new state's, worked out here and again by any replay that has the
        state the round started from (es_states)."""
        r = self.r
        job = self.job(r.job_id)
        spec = job.spec
        verifier_flops: dict[str, int] = defaultdict(int)
        missed: set[str] = set()
        answered = defaultdict(list)
        for t in r.tasks.values():
            if t.verdict is None:
                if t.verifier:
                    missed.add(t.verifier)
                continue
            if t.verdict[0]:
                verifier_flops[t.verifier] += spec.check_flops()
            answered[(t.unit.unit_id, t.submitter)].append(t)
        for key in sorted(r.reveals):
            if key in r.results:
                continue
            got = answered[key]
            if any(not t.verdict[0] for t in got):
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None,
                                            "a checker found an answer that isn't its model's greedy choice, or a "
                                            "replay log-probability off (not refereed yet: unpaid, out of the update)")
            elif got:
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.ACCEPTED, got[0].verifier)
            else:
                r.results[key] = UnitResult(r.units[key[0]], key[1], Verdict.UNCHECKED, None, "no checker answered for it")
        offenders = {s[0] for s in r.slashes} | r.forfeit
        worker_flops: dict[str, int] = defaultdict(int)
        rows, units, readings = [], set(), []
        for key, res in sorted(r.results.items()):
            if res.verdict != Verdict.ACCEPTED or key[1] in offenders:
                continue
            unit = r.units[key[0]]
            worker_flops[key[1]] += unit.flops
            units.add(key[0])
            _, rewards, lps, reading = r.es_claims[key]
            readings.append(reading)
            for k, (rp, rm) in enumerate(rewards):
                rows.append((unit.index * spec.pairs_per_unit + k, rp, rm, None if lps is None else lps[k]))
        rows.sort(key=lambda u: u[0])
        seed = int(r.record["es"]["seed"], 16)
        mean = None
        if rows:
            binary = [x for _, rp, rm, _ in rows for x in (rp, rm)]
            use_lps = r.record["es"]["fitness"] == "replay" and all(lp is not None for *_, lp in rows)
            lp = [x for *_, pair in rows for x in pair] if use_lps else None
            coefs, used = es2.coefficients(binary, lp)
            scale = es2.lr_at(spec, int(r.record["es"]["step"])) / len(binary)
            if self.replaying:  # the node's reading of the blobs isn't in a replay: the record's targets
                targets = (self.replaying.get("es") or {}).get("targets") or []
            else:
                answers = [a for rd in readings if rd is not None for a in rd[0]]
                ok = [o for rd in readings if rd is not None for o in rd[1]]
                targets = es2.replay_targets(spec.problems, answers, ok)
            ent = es2.entry(seed, [[j, c] for (j, *_), c in zip(rows, coefs)], scale, targets)
            mean = es.mean_reward(binary)
            r.record["es"] |= {"used": used, "update": ent["update"], "scale": ent["scale"],
                               "targets": ent["targets"], "reward": mean}
            r.record["score"] = [1.0 - mean, mean]
        else:
            ent = None
            r.record["es"]["update"] = []
        new_params = None

        def average() -> str:
            nonlocal new_params
            self._weights(r.start)
            new_params = r.start if ent is None else es2.step_state(r.start, ent, mean)
            return params_hash(new_params).hex()
        if r.start is not None:  # live, or a replay given the starting state: worked out, never read from the record
            model_hash = average()
            r.record["checks"]["model"]["params_hash"] = model_hash
        else:  # a replay without it: the record's (es2_rounds.rebuild_chain works the states out from the chain itself)
            model_hash = self._claim("model", "params_hash", average)
        r.record["secret"] = r.secret.hex()
        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(units),
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

    def _settle_prep(self, job: TrainingJob, accepted: dict, offenders: set[str], verifier_flops: dict, missed: set[str],
                     proposer: str):
        """A prep round's settlement: its accepted units, one claim each, go into the record with what later units
        and the records need of their commitments, and join the work graph when the block is adopted. If they
        finish the prep, the records' manifest goes to the bucket and its root into the record: the job trains from
        it from the next round. The model doesn't change, and the work is paid as any job's."""
        r = self.r
        # A replay has the record, not the job's text or the bucket: what the leader read from the bucket for each
        # accepted claim, and the root, are taken from it (a claim the replay doesn't accept can't be in it).
        recorded = (self.replaying or {}).get("prep") or {}
        state = None if self.replaying else self.prep(job)
        worker_flops: dict[str, int] = defaultdict(int)
        kept: dict[str, dict] = {}
        for (uid, addr), rv in sorted(accepted.items()):
            name = r.prep[uid]
            if addr in offenders or name in kept:
                continue
            commit = r.prep_commits.get((uid, addr))
            if state is not None and commit is not None:
                kept[name] = state.kept(state.graph.units[name], commit)
            elif state is None and name in (recorded.get("accepted") or {}):
                kept[name] = recorded["accepted"][name]
            else:
                continue
            worker_flops[addr] += r.units[uid].flops
        r.record["prep"]["accepted"] = kept
        if state is None:
            if recorded.get("root"):
                r.record["prep"]["root"] = recorded["root"]
        else:
            after = dsr.PrepState.restore(job, state.to_json())
            for name, k in kept.items():
                after._accept(name, k)
            if after.done and kept:
                m = after.manifest()
                if self.prep_store is not None:
                    self.prep_store(job).put("manifest.json", json.dumps(m, indent=1).encode())
                r.record["prep"]["root"] = m["root"]
        r.record["secret"] = r.secret.hex()
        summary = RoundSummary(
            job_id=r.job_id,
            accepted_units=len(kept),
            worker_flops=dict(sorted(worker_flops.items())),
            verifier_flops={a: f for a, f in sorted(verifier_flops.items()) if a not in offenders},
            slashes=r.slashes,
            accepted=sorted(worker_flops),
            missed=sorted(missed | r.forfeit | {res.worker for res in r.results.values() if res.verdict == Verdict.MISSED_DEADLINE}),
            params_hash=r.start_hash.hex(),
            proposer=proposer,
            record_hash=h256(canonical(r.record)).hex(),
        )
        return summary, r.start, worker_flops

    @_locked
    def may_sign(self, addr: str, method: str, name: str, query: dict) -> None:
        """Whether `addr` may have a link to do `method` on the current DeepSeek job's object `name`: anyone reads (GET,
        HEAD: the data is public, made from public text and weights); a prep round's units' holders write their own
        units' objects, during compute (PUT, and POST for a multipart upload). Raises ProtocolError if not."""
        r = self.r
        if r is None or not self.job(r.job_id).deepseek:
            raise ProtocolError("no DeepSeek job's round is open")
        if ".." in name or name.startswith("/") or not name:
            raise ProtocolError("not an object name")
        if method in ("GET", "HEAD"):
            return
        if method not in ("PUT", "POST") or set(query) - {"uploads", "uploadId", "partNumber"}:
            raise ProtocolError(f"links to {method} aren't given")
        if self.phase != Phase.COMPUTE or not r.prep:
            raise ProtocolError("objects are written in a prep round's compute phase")
        state = self.prep(self.job(r.job_id))
        mine = set()
        for uid in r.assignments.get(addr, []):
            mine |= state.objects(r.prep[uid])[1]
        if name not in mine:
            raise ProtocolError(f"{name} isn't one of your units' outputs this round")

    def prep_summary(self, job_id: str, now: float) -> dict | None:
        """The job summary's "prep" field (netstats), for a DeepSeek job: its work graph's progress."""
        try:
            job = self.job(job_id)
        except Exception:  # noqa: BLE001 - its data isn't here: no summary
            return None
        return self.prep(job).line.summary(now) if job.deepseek else None

    @_locked
    def replayed_summary(self, proposer: str) -> RoundSummary:
        """The summary a replayed round arrives at; nothing is sealed or stored."""
        self._need(Phase.VERIFY, Phase.DISPUTE)
        summary, _, _ = self._settle(proposer)
        self.phase, self.r = Phase.IDLE, None
        return summary

    @_locked
    def finalize(self, proposer_keys: KeyPair | None = None, timestamp: int | None = None, seal: bool = True) -> RoundReport:
        """Settle the round into a signed block. With `seal`, the block goes straight onto the chain (a chain one node
        runs); without, it's only built, for the validators to vote on (consensus.py), and adopt() adds it if they
        agree."""
        r = self._need(Phase.VERIFY, Phase.DISPUTE)
        if self.replaying:
            raise ProtocolError("a replayed round isn't sealed; use replayed_summary")
        job = self.job(r.job_id)
        keys = proposer_keys or self.keys
        if keys is None:
            raise ProtocolError("no proposer key")
        summary, new_params, worker_flops = self._settle(keys.address)
        block = self._block(summary, keys, timestamp)
        # A fine-tune's score is its scorers' (of the model this round started from): the node never runs the model.
        val_loss, val_acc = (r.record.get("score") or (None, None)) if r.v4 or r.es else job.evaluate(new_params)
        report = RoundReport(r.round_no, r.job_id, list(r.results.values()), dict(worker_flops), summary.verifier_flops,
                             r.slashes, r.unassigned, val_loss, val_acc, keys.address, block, record=r.record, params=new_params,
                             teams=[list(t.members) for t in r.teams], waiting=r.waiting)
        rewards = [t for t in block.txs if t.kind == "reward"]
        fee = sum(t.amount for t in block.txs if t.kind == "fee_release")
        report.pool = self.chain.state.round_subsidy(summary, block.height) + fee  # what this block mints, plus released fees
        report.burned = report.pool - sum(t.amount for t in rewards)
        report.worker_pay = {t.recipient: t.amount for t in rewards if t.memo == "work"}
        report.verifier_pay = {t.recipient: t.amount for t in rewards if t.memo == "verify"}
        if seal:  # (a fine-tune's scores are saved from its record)
            self.adopt(block, r.record, new_params, None if r.v4 or r.es else (val_loss, val_acc))
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
        Without `params` (another validator ran the round), the weights have to be fetched (node.py).

        The block, its certificate and its record go to disk first, in one transaction (Store.save_block), and only
        then into the chain in memory and the mempool. A block that couldn't be saved (NotSaved; RivalBlock when the
        database holds a different one at its height) leaves both as they were, and the error goes to the caller:
        node.py stops the node. Before, a node whose saves failed (2026-11-03: a second node on its folder had written
        those heights first) went on leading rounds on blocks only its memory held, served them, and would have lost
        them all at its next restart. The very block already on disk, put there by another writer, isn't an error."""
        self.chain.append(block, (lambda: self._save(block, certificate, record)) if self.store else None)
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
        s = block.summary
        if s is None:
            return
        if isinstance(record, dict) and record.get("prep") is not None and not self.replaying:
            try:  # whoever led the round: the work graph moves on here too, and is kept for a restart
                state = self.prep(self.job(s.job_id))
            except Exception:  # noqa: BLE001 - its data isn't on this node: nothing to keep
                state = None
            if state is not None:
                state.apply(record)
                if self.store and hasattr(self.store, "save_meta"):
                    self.store.save_meta(f"prep:{s.job_id}", state.to_json())
        if self.store and isinstance(record, dict) and record.get("version") in (FINETUNE_RECORD_VERSION, ES_RECORD_VERSION) \
                and record.get("score"):
            # Scores come with a fine-tune's record, whoever led its round: the scorers' median. A round no scorer
            # answered in has none, and isn't in the job's history of scores (the chain still has its units).
            self.store.save_round(s.job_id, block.height, s.accepted_units, *record["score"])
        if params is None:
            self.params.pop(s.job_id, None)
            return
        self.params[s.job_id] = params
        if self.store:
            self.store.save_params(s.job_id, block.height, params)
            fine_tune = isinstance(record, dict) and record.get("version") in (FINETUNE_RECORD_VERSION, ES_RECORD_VERSION)
            if scores is not None and not fine_tune:  # (a fine-tune's were saved from its record, above)
                self.store.save_round(s.job_id, block.height, s.accepted_units, *scores)

    def _save(self, block: Block, certificate: dict | None, record: dict | None) -> None:
        """adopt()'s write, before the chain changes: NotSaved however it fails, so the caller can tell a block that
        isn't on disk from one that doesn't apply (InvalidBlock, raised before anything is written)."""
        try:
            self.store.save_block(block, certificate, record)  # (False: this very block was there already; fine)
        except NotSaved:
            raise
        except Exception as e:  # noqa: BLE001 - a full disk, an I/O error, a locked database: it isn't saved, whatever it is
            raise NotSaved(block.height, e) from e


class _MemoryBlobs:
    """Disputes' states and evidence when no BlobStore is given (in-process networks and replays)."""

    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.roots: dict[tuple, str] = {}

    def put_bytes(self, raw: bytes) -> str:
        import hashlib

        bid = hashlib.sha256(raw).hexdigest()
        self.data[bid] = bytes(raw)
        return bid

    def get_bytes(self, bid: str) -> bytes | None:
        return self.data.get(bid)

    def size(self, bid: str) -> int | None:
        return len(self.data[bid]) if bid in self.data else None

    def tensor_root(self, bid: str, rows: int, cols: int) -> str | None:
        from . import merkle

        raw = self.data.get(bid)
        if raw is None or len(raw) != rows * cols * 4:
            return None
        if (bid, rows, cols) not in self.roots:  # by shape: the same bytes make another root in another shape
            self.roots[bid, rows, cols] = merkle.root_of(np.frombuffer(raw, "<f4").reshape(rows, cols))
        return self.roots[bid, rows, cols]
