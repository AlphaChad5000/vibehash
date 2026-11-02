"""A Vibe Hash node: HTTP API + a wall-clock driver for the round state machine.

On a chain with validators (see ledger.py), validators take turns running rounds: the round's leader runs it with
the workers, proposes the block, and the validators vote on it (consensus.py) before anyone adds it. Every node
gossips votes and transactions to its peers and catches up on blocks it missed (p2p.py). Workers follow the
current leader, which /status names. A chain without validators is run by one node, which seals each block itself.

Public (read-only) endpoints let anyone audit the chain:
    GET  /status                 phase, round, deadlines, beacon, chain tip; the consensus round and its leader
    GET  /genesis                genesis document
    GET  /blocks?from=N&to=M     raw blocks (max 100 per request), each with its commit certificate if it has one
    GET  /account/<address>      balance, stake, unbonding, nonce, reputation
    GET  /validators             each validator's address, stake, and whether it has a vote
    GET  /job/<id>               job spec, remaining escrow, validation score after each round, model size
    GET  /job/<id>/model?offset=N   one chunk of the job's latest trained weights (what its poster downloads)
    GET  /round/params           the current round's starting weights (as a blob id)
    GET  /round/<height>/record  the signed messages and check results a block's round was decided from (replay.py)
    GET  /round/games            a fine-tune round's disputes: whose move, and what's been posted (dispute.py)
Signed-envelope endpoints (workers):
    POST /register  /assignment  /commit  /reveal  /tasks  /verdict  /move
    (register, commit, reveal and verdict answer with a signed receipt: proof the message will be in the
    round's record, so a node that drops it can be caught; see receipts.py)
Binary transfer (chunked, resumable, content-addressed; no size cap from HTTP):
    GET  /blob/<sha256>/status   bytes received so far
    GET  /blob/<sha256>?offset=N one chunk of a checkpoint
    PUT  /blob/<sha256>          upload one chunk (signed headers; this round's workers, or a dispute's defender)
    ... and the same three under /dataset/<sha256> for training data
Other:
    POST /tx                     a signed user transaction (passed on to peers)
    POST /consensus              a validator's signed proposal or vote

Serve it over TLS on any network you don't control: natively with a certificate (tls.py), or behind a reverse
proxy. Every request passes per-client rate limits and connection caps first (ratelimit.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import ssl
import threading
import time
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import deepseek_graph as dsg
from . import economics as econ
from . import netstats, receipts, tasks
from .blobs import CHUNK, MAX_DATASET, STALE_PART_S, BlobError, BlobStore, check_id
from .client import NodeError
from .consensus import Engine, Timeouts, check_certificate
from .coordinator import (FINETUNE_RECORD_VERSION, RECORD_VERSION, SHARDED_RECORD_VERSION, Coordinator, CoordinatorConfig,
                          Phase, ProtocolError)
from .crypto import KeyPair, address_of, verify
from .ledger import VERSION, Block, InvalidBlock, Tx, apply_block, canonical
from .model import params_hash
from .p2p import Peers
from .ratelimit import GuardedHandler, GuardedServer, Limits, path_parts
from .replay import replay_block
from .storage import Store, record_blobs
from .tasks import SYN_PER_VIBE, InvalidJob, Reveal, TrainingJob
from .wire import MAX_CLOCK_SKEW_S, WireError, blob_openings_in, blob_openings_out, commitment_in, commitment_out, open_envelope

log = logging.getLogger("vibehash.node")
MAX_JSON = 4 * 2**20  # every JSON message is small now; weights travel as blobs
# ...except a proposal: a block and its round record, about 4 KB a unit at the default 8 segments, so this takes
# ~16,000 units (a job may ask for 10,000 a round, plus replicas) unless they have thousands of segments each. Only
# validators of a chain with others take one, and only MAX_BIG_BODIES at once, since each is parsed into a few times
# its size in memory.
MAX_PROPOSAL = 64 * 2**20
MAX_BIG_BODIES = 2
# A /blocks reply stops taking blocks once it's this big (it always has one). A block may carry 1 MiB of user
# transactions (ledger.MAX_BLOCK_TX_BYTES, on v2 blocks), so 100 of them could make 100 MiB.
MAX_BLOCKS_REPLY = 4 * 2**20
# A worker's signed messages, as wire.seal makes them, and the size an honest one can have: a few hundred bytes, or
# up to ~10 KB for a reveal's Merkle proofs. Registrations and verdicts go into the round record exactly as sent, so
# anything more would be junk that every validator stores and every proposal carries.
ENVELOPE = {"kind", "body", "ts", "chain_id", "pubkey", "sig"}
BODIES = {"register": {"round", "flops_per_sec", "memory_mb"}, "assignment": {"round"}, "commit": {"round", "commitment"},
          "sign": {"round", "requests"},
          "reveal": {"round", "reveal"}, "tasks": {"round"}, "verdict": {"round", "task_id", "ok", "bad_segment", "score"},
          "move": {"round", "game", "n", "move"}}
MAX_MESSAGE, MAX_REVEAL = 4 * 2**10, 64 * 2**10
MAX_MOVE = 16 * 2**10  # a dispute's split names up to program.K + 1 state roots
# What a dispute's defender may upload in a round: the states its splits name (a step's gradients after each sequence,
# a sequence's activations layer by layer, a layer's tensors op by op), and its evidence.
MAX_TRACE_BLOB, MAX_TRACE_UPLOADS = 2 * 2**30, 16 * 2**30
# The biggest model this node relays a post_job for (--max-job-mb), as one checkpoint at the job's precision. Measured:
# a float32 model's starting weights take ~7 times its checkpoint size to build (float64 copies), ~460 MB at 64 MB, and
# a round holds a few float64 copies (start, current, average: 135 MB each at 64 MB), so a 2 GB server running a
# ~0.5 GB node keeps its head above water. The 1 GiB jobs tasks.py allows would need ~7 GB.
MAX_JOB_BYTES = 64 * 2**20
# No phase is longer than this, whatever a job's size asks for (TrainingJob.phase_lengths): a spec can't stall a node
# for days. A job that would need more simply gets fewer units done a round.
MAX_PHASE_S = tasks.MAX_PHASE_S
MIN_UPLOADER_BALANCE = 1 * SYN_PER_VIBE


def bucket_from_env() -> dsg.Signer | None:
    """The bucket for DeepSeek jobs' hand-offs, from $VIBEHASH_BUCKET_HOST (my-bucket.nyc3.digitaloceanspaces.com),
    $VIBEHASH_BUCKET_REGION, $VIBEHASH_BUCKET_KEY and $VIBEHASH_BUCKET_SECRET (and $VIBEHASH_BUCKET_SCHEME, https
    unless set): the operator sets them on the node at launch. None if they aren't set."""
    import os
    env = {k: os.environ.get(f"VIBEHASH_BUCKET_{k}") for k in ("HOST", "REGION", "KEY", "SECRET", "SCHEME")}
    if not all(env[k] for k in ("HOST", "REGION", "KEY", "SECRET")):
        return None
    return dsg.Signer(env["HOST"], env["REGION"], env["KEY"], env["SECRET"], scheme=env["SCHEME"] or "https")


class Busy(Exception):
    """The node can't take this request now (503): the client waits and resends."""


@dataclass
class PhaseTimes:
    register_s: float = 3.0
    compute_s: float = 20.0
    reveal_s: float = 5.0
    verify_s: float = 10.0
    idle_s: float = 2.0
    dispute_s: float = 0.0  # a fine-tune's dispute phase, which its round has when a claim is disputed (times_for)
    move_s: float = 0.0  # how long a dispute's party has for each move

    @property
    def round_s(self) -> float:
        """The longest a round can take: validators wait for a proposal accordingly (_propose_wait)."""
        return self.register_s + self.compute_s + self.reveal_s + self.verify_s + self.dispute_s


class Node:
    def __init__(self, data_dir, genesis: dict, keys: KeyPair, host="127.0.0.1", port=8545,
                 times: PhaseTimes | None = None, config: CoordinatorConfig | None = None, keep_rounds: int = 10,
                 limits: Limits | None = None, tls: ssl.SSLContext | None = None,
                 peers: list[str] = (), announce: str | None = None, timeouts: Timeouts | None = None,
                 max_job_bytes: int = MAX_JOB_BYTES, bucket: dsg.Signer | None = None):
        """`peers`: other nodes to connect to (validators find each other from the chain without them).
        `announce`: the address other nodes reach this one at, if it isn't the one it listens on.
        `max_job_bytes`: the biggest model (one checkpoint) a job may have for this node to relay its post_job.
        `bucket`: the storage DeepSeek jobs hand off through (deepseek_graph.Signer, its keys this node's): workers get
        links it signs for their units' own objects. None: this node can't run a DeepSeek job's prep (bucket_from_env
        reads one from the environment)."""
        self.times = times or PhaseTimes()
        self.keep_rounds = keep_rounds  # rounds whose checkpoints stay available to re-check (the challenge window)
        self.max_job_bytes = max_job_bytes
        cfg = config or CoordinatorConfig()
        cfg.compute_s = self.times.compute_s * 0.8  # leave slack for network and polling
        self.keys = keys
        self.store = Store(data_dir)
        self.chain = self.store.open_chain(genesis)
        self.genesis = genesis
        self.coord = Coordinator(self.chain, keys, cfg, store=self.store)
        self.bucket = bucket
        if bucket is not None:
            self.coord.prep_store = self.prep_store
        self.coord.timing = lambda job: self.times_for(job).compute_s * 0.8  # capacity leaves slack, as above
        self.blobs: BlobStore = self.store.blobs
        self.coord.trace_blobs = TraceBlobs(self.blobs)
        self.trace_uploads: dict[str, int] = {}  # per-defender bytes uploaded for disputes this round
        self.start_blob: str | None = None
        self.uploads: dict[str, set[str]] = {}  # per-worker blob ids this round (quota)
        self._blob_of: dict[int, str] = {}  # id(array) -> blob id, for arrays this round loaded from blobs
        self.phase_ends = 0.0
        self._model_lock = threading.Lock()
        self._model_cache: tuple = (None, b"", "")
        self._network_lock = threading.Lock()
        self._network_cache: tuple = (None, 0.0, {})  # (tip hash, when, /network's answer)
        self.stopping = threading.Event()
        self.server = GuardedServer((host, port), _handler(self), limits, tls)  # limits=None: the defaults, which spare localhost
        self.threads: list[threading.Thread] = []
        self.announce = (announce or self.chain.state.validators.get(keys.address) or self.url).rstrip("/")
        self.peers = Peers(self.announce, keys.address, list(peers), lambda: self.chain.state.validators)
        # A leader runs a whole training round before it can propose, so the others wait that long for it, and more.
        self.engine = Engine(self.chain.chain_id, keys, timeouts or Timeouts(propose=1.5 * self.times.round_s + 5),
                             validators=lambda: self.chain.state.validator_set(), validate=self._validate,
                             commit=self._decided, broadcast=lambda m: self.peers.broadcast("/consensus", m),
                             has_work=self._has_work, behind=lambda height: self.wake_sync.set(),
                             persist=lambda state: self.store.save_meta("consensus", state), propose_wait=self._propose_wait)
        self.proposed: dict[str, tuple] = {}  # block id -> (weights, scores, record) for blocks we proposed at this height
        self.commit_lock = threading.Lock()
        self.wake_sync = threading.Event()
        self.leader_hint: str | None = None  # the current leader, as a validator told us (for nodes that don't vote)
        self.blobs.prune(self._evidence())  # keep recent rounds' evidence; the rest belongs to an abandoned round
        self.store.datasets.prune_partial(older_than=STALE_PART_S)  # uploads and downloads abandoned part way

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"{self.server.scheme}://{host}:{port}"

    def start(self) -> "Node":
        if self.bft:  # carrying on after the rounds we voted in before a restart, if it was mid-height
            self.engine.start(self.chain.tip.height + 1, resume=self.store.load_meta("consensus"))
        self.threads = [threading.Thread(target=self.server.serve_forever, daemon=True, name="http"),
                        threading.Thread(target=self._rounds, daemon=True, name="rounds"),
                        threading.Thread(target=self._consensus, daemon=True, name="consensus"),
                        threading.Thread(target=self._syncing, daemon=True, name="sync")]
        for t in self.threads:
            t.start()
        log.info("node up at %s  chain=%s  height=%d  genesis=%s%s", self.url, self.chain.chain_id,
                 self.chain.tip.height, self.chain.genesis_hash[:16], self._role())
        return self

    def stop(self) -> None:
        self.stopping.set()
        self.wake_sync.set()
        self.server.shutdown()
        self.server.server_close()
        for t in self.threads:
            t.join(timeout=30)
        self.peers.close()
        self.store.close()

    def _role(self) -> str:
        if not self.bft:
            return ""
        vals = self.chain.state.validator_set()
        return (f"  validator ({len(vals)} in the set)" if self.keys.address in vals
                else f"  following {len(vals)} validators")

    @property
    def bft(self) -> bool:
        """Whether validators run this chain (else this node seals blocks alone)."""
        return bool(self.chain.state.validators)

    def _wait(self, seconds: float, still=None) -> bool:
        """Sleep through a phase. False if we're stopping, or `still()` turns false (the round's no longer wanted)."""
        self.phase_ends = end = time.time() + seconds
        if still is None:
            return not self.stopping.wait(seconds)
        while (left := end - time.time()) > 0:
            if self.stopping.wait(min(left, 0.2)) or not still():
                return False
        return still()

    def _evidence(self) -> set[str]:
        """Checkpoints and dispute evidence recent rounds' records point to (storage.record_blobs), kept for
        `keep_rounds` rounds so anyone can re-run those rounds' checks and referees, plus those of blocks we've
        proposed and the validators are still voting on. Unopened checkpoints and older rounds' are deleted."""
        keep = self.store.evidence(self.chain.tip.height, self.keep_rounds)
        for _, _, record in list(self.proposed.values()):
            keep |= record_blobs(record)
        return keep

    def _end_round(self) -> None:
        with self.coord.lock:
            self.start_blob, self.uploads, self._blob_of, self.trace_uploads = None, {}, {}, {}
        self.blobs.prune(self._evidence())
        self.coord.trace_blobs.forget()
        self.store.datasets.prune_partial(older_than=STALE_PART_S)

    def _rounds(self) -> None:
        c, t = self.coord, self.times
        while not self.stopping.is_set():
            try:
                if self.bft:
                    self._lead()
                else:
                    self._solo()
            except Exception:
                log.exception("round failed; abandoning it")
                with c.lock:
                    c.phase, c.r = Phase.IDLE, None
                self._end_round()
                self._wait(t.idle_s)

    def _solo(self) -> None:
        """One round on a chain this node runs alone: it seals the block itself."""
        c = self.coord
        if not c.open_round():
            block = c.seal_pending()
            if block:
                log.info("block %d: %d txs (no active job)", block.height, len(block.txs))
            self._wait(self.times.idle_s)
            return
        if not self._run_round():
            return
        rep = c.finalize()
        self._end_round()
        fraud = sum(r.verdict.is_fraud for r in rep.results)
        acc = "unscored" if rep.val_acc is None else f"{100 * rep.val_acc:.1f}%"  # a fine-tune round nobody scored
        log.info("block %d sealed: %d units accepted, %d fraud, val acc %s, pool %.3f VIBE",
                 rep.block.height, rep.block.summary.accepted_units, fraud, acc, rep.pool / SYN_PER_VIBE)

    def prep_store(self, job) -> dsg.PresignedStore:
        """A DeepSeek job's objects in the bucket, under its id, as this node reaches them (signing its own links)."""
        return dsg.PresignedStore(lambda m, n, q=None: self.bucket(m, f"{job.job_id}/{n}", q))

    def times_for(self, job) -> PhaseTimes:
        """A round's phase lengths for `job`: this node's, or longer where the job's size needs it (or the job asks
        for a longer compute phase). See TrainingJob.phase_lengths."""
        t, need = self.times, job.phase_lengths()
        if job.deepseek:  # a prep unit's transfers at the slowest link the network takes (deepseek_graph.deadline)
            need = need | {"compute": max(need["compute"], job.spec.prep_seconds())}
        verify = min(MAX_PHASE_S, max(t.verify_s, need["verify"]))
        # A dispute has as many moves as the job's program is deep (need["moves"], program.dispute_size), and each as
        # long as its dearest takes (need["move"]: re-running a segment, and a split's states up and down), or verify
        # if that's longer. Most moves take a second or two, and the phase ends as soon as every dispute has; it's this
        # long only for a party that makes every move just in time (a DeepSeek job's prep rounds have no games yet:
        # their dispute phase ends at once). A job whose moves need more than MAX_PHASE_S in all gets less for each: a
        # defender too slow for it loses (DESIGN, "Time").
        moves = need.get("moves", 0)
        move = min(max(verify, need["move"]), MAX_PHASE_S / moves) if moves else 0.0
        return PhaseTimes(t.register_s, min(MAX_PHASE_S, max(t.compute_s, need["compute"])),
                          min(MAX_PHASE_S, max(t.reveal_s, need["reveal"])), verify, t.idle_s, moves * move, move)

    def _propose_wait(self) -> float:
        """How long validators wait for a proposal: the configured wait, plus however much longer than the node's
        own rounds the longest job with budget left runs, since its leader may be running that one."""
        s = self.chain.state
        longest = max((self.times_for(s.job(j)).round_s for j, v in s.escrow.items() if v > 0), default=0.0)
        return self.engine.timeouts.propose + 1.5 * max(0.0, longest - self.times.round_s)

    def _run_round(self, still=None) -> bool:
        """Drive an open round through its phases. False if it was given up: we're stopping, or `still()` says the
        round's no longer wanted (another block was decided at its height)."""
        c = self.coord
        with c.lock:
            job = c.job(c.r.job_id)
            self.start_blob = self.blobs.put_array(c.r.start, job.wire_dtype)
        t = self.times_for(job)
        longer = f" ({t.round_s:.0f} s rounds for its size)" if t.round_s > self.times.round_s + 0.5 else ""
        log.info("round %d open for job %s%s", c.r.round_no, c.r.job_id, longer)
        if not self._wait(t.register_s, still):
            return False
        c.start_compute()
        log.info("  %d workers registered; %d units assigned, %d unassigned", len(c.r.regs),
                 sum(len(v) for v in c.r.assignments.values()), c.r.unassigned)
        for step, wait in ((c.start_reveal, t.compute_s), (c.start_verify, t.reveal_s), (None, t.verify_s)):
            if not self._wait(wait, still):
                return False
            if step:
                step()
        c.move_s = t.move_s
        if c.start_disputes(time.time()):  # a fine-tune's disputed claims, refereed by this node
            log.info("  %d disputes", len(c.r.games))
            ends = self.phase_ends = time.time() + t.dispute_s
            while not c.disputes_done() and time.time() < ends:
                if self.stopping.wait(0.2) or (still is not None and not still()):
                    return False
                c.expire_moves(time.time())
            c.end_disputes(time.time())
            for g in sorted(c.r.games.values(), key=lambda g: g.gid):
                log.info("  dispute %s (%s, %s against %s): %s", g.gid, g.why, g.challenger[:10], g.defender[:10],
                         g.outcome or "unfinished")
        return True

    # --- running a chain with validators ------------------------------------------------------

    def _lead(self) -> None:
        """When it's our turn to propose, build the block: run a training round if a job has budget (and we have
        its data and latest weights), else take the waiting transactions."""
        c, e = self.coord, self.engine
        height = self.chain.tip.height + 1
        if e.height < height:  # validators just joined a chain this node ran alone, or a sync moved us on
            e.start(height)
        want = e.wanted
        if want is None or want[0] != height:
            self.stopping.wait(0.1)
            return
        rnd = want[1]
        still = lambda: e.wants(height, rnd) and self.chain.tip.height + 1 == height  # noqa: E731
        value, job_id = None, c.active_job_id()
        if job_id is not None and self._ensure_params(job_id) and c.open_round():
            where = f" at {self.announce}" if len(self.chain.state.validator_set()) > 1 else ""
            log.info("leading round %d%s (consensus round %d)", height, where, rnd)
            rep = c.finalize(seal=False) if self._run_round(still) else None
            if rep is None:
                with c.lock:
                    if c.r is not None and c.r.round_no == height:
                        c.phase, c.r = Phase.IDLE, None
                if not self.stopping.is_set():
                    log.info("round %d given up: the validators moved on", height)
            elif still():
                bid = rep.block.hash.hex()
                self.proposed[bid] = (rep.params, (rep.val_loss, rep.val_acc), rep.record)
                value = {"block": rep.block.to_dict(), "record": rep.record}
            self._end_round()
        elif c.mempool:
            block = c.pending_block()
            if block is not None:
                value = {"block": block.to_dict(), "record": None}
        if value is None:
            self.stopping.wait(self.times.idle_s)
            return
        if e.propose(height, rnd, value):
            log.debug("proposed block %d in consensus round %d", height, rnd)

    def _has_work(self) -> bool:
        return any(v > 0 for v in self.chain.state.escrow.values()) or bool(self.coord.mempool)

    def _validate(self, value: dict) -> bool:
        """Whether a proposed block may be the next one: it applies to the chain (signatures, balances, and the
        rewards its summary implies), and its summary follows from its round's signed messages."""
        try:
            block = Block.from_dict(value["block"])
            # Older blocks and records still apply and replay (syncing, chain verify), but a new block must follow
            # today's rules: a leader can't propose under the old ones to get round a fix.
            if block.version != VERSION:
                raise InvalidBlock(f"a new block must be version {VERSION}, not {block.version}")
            record = value.get("record")
            if block.summary is not None and isinstance(record, dict):
                job = self.chain.state.job(block.summary.job_id) if block.summary.job_id in self.chain.state.jobs else None
                want = (SHARDED_RECORD_VERSION if job and job.sharded else FINETUNE_RECORD_VERSION if job and job.finetune
                        else RECORD_VERSION)
                if record.get("version") != want:
                    raise InvalidBlock(f"a new round's record must be version {want}, not {record.get('version')!r}")
            with self.coord.lock:
                tip = self.chain.tip
                if block.height != tip.height + 1 or block.prev_hash != tip.hash:
                    return False
                if block.timestamp > time.time() + MAX_CLOCK_SKEW_S:
                    raise InvalidBlock("timestamp is in the future")
                apply_block(self.chain.state, block, self.chain.chain_id, tip)
                if block.summary is not None:
                    record = value.get("record")
                    problems = (["no round record"] if not isinstance(record, dict)
                                else replay_block(self.chain, block, record,
                                                  datasets=PeerDatasets(self, self.chain.state.validators.get(
                                                      address_of(block.proposer_pubkey))),
                                                  roots=LeaderBlobs(self, address_of(block.proposer_pubkey)),
                                                  records=self.prep_store if self.bucket is not None else None))
                    if problems:
                        raise InvalidBlock("; ".join(problems))
        except (InvalidBlock, InvalidJob, KeyError, TypeError, ValueError) as e:
            log.warning("voting against block %s: %s", value.get("block", {}).get("height"), e)
            return False
        return True

    def _decided(self, height: int, value: dict, cert: dict) -> None:
        """The validators agreed on block `height` (called by the engine)."""
        block = Block.from_dict(value["block"])
        mine = self.proposed.get(block.hash.hex())
        # A block without a round has no record; whatever a relay put in that unsigned field isn't kept.
        self._commit(block, value.get("record") if block.summary is not None else None, cert, mine)
        self.proposed.clear()

    def _commit(self, block: Block, record: dict | None, cert: dict | None, mine: tuple | None = None) -> bool:
        """Add a final block. `mine` is what we kept from proposing it, if we did. False if we already had it."""
        with self.commit_lock, self.coord.lock:
            if block.height != self.chain.tip.height + 1:
                return False
            # A round of ours still running for this height has lost out; _lead notices and gives it up.
            self.coord.adopt(block, record, mine[0] if mine else None, mine[1] if mine else None, cert)
        s = block.summary
        # One line per block across the network: its proposer's. The rest log it quietly.
        log.log(logging.INFO if mine or address_of(block.proposer_pubkey) == self.keys.address else logging.DEBUG,
                "block %d final%s: %s", block.height, f" (signed by {len(cert['votes'])} validators)" if cert else "",
                f"{s.accepted_units} units accepted" if s else f"{len(block.txs)} txs")
        if s is not None and mine is None:  # someone else's round: get the weights it trained
            threading.Thread(target=self._fetch_weights, args=(s.job_id,), daemon=True, name="weights").start()
        return True

    def _consensus(self) -> None:
        while not self.stopping.wait(0.1):
            if self.bft:
                try:
                    self.engine.tick()
                except Exception:  # noqa: BLE001
                    log.exception("consensus tick failed")

    def _syncing(self) -> None:
        while not self.stopping.is_set():
            self.wake_sync.wait(2.0)
            self.wake_sync.clear()
            if self.stopping.is_set() or not self.bft:
                continue
            try:
                self._sync()
                self._fetch_missing_data()
            except Exception:  # noqa: BLE001
                log.exception("sync failed")

    def _sync(self) -> None:
        """Fetch blocks we're missing from any peer that's ahead, trusting none of them: each one must apply,
        replay from its record, and carry a valid commit certificate. A peer whose replies make no sense, or that
        sends a block we refuse or nothing new, is left for the next."""
        for url in self.peers.urls():
            peer = self.peers.client(url)
            try:
                st = peer.status()
                self.peers.alive(url)
                if st.get("genesis_hash") != self.chain.genesis_hash:
                    continue
                if self.keys.address not in self.chain.state.validator_set() and "consensus" in st:
                    hint = st.get("leader") or url  # None: the peer is leading itself
                    # Our workers go wherever this points, so only to a validator the chain lists.
                    if isinstance(hint, str) and hint.rstrip("/") in {u.rstrip("/") for u in self.chain.state.validators.values()}:
                        self.leader_hint = hint
                while st["height"] > self.chain.tip.height:
                    lo = self.chain.tip.height + 1
                    try:
                        batch = peer.get(f"/blocks?from={lo}&to={lo + 99}")
                    except NodeError as e:
                        if e.status != 200:
                            raise
                        batch = peer.get(f"/blocks?from={lo}&to={lo}")  # too big to take 100 at once
                    if not batch or not all(self._take(peer, d) for d in batch) or self.chain.tip.height < lo:
                        break  # one we refused, or none we lacked (a peer serving old blocks forever)
            except (NodeError, InvalidBlock, KeyError, TypeError, ValueError, AttributeError) as e:
                log.debug("sync from %s: %s", url, e)
        if self.engine.height < self.chain.tip.height + 1:
            self.engine.start(self.chain.tip.height + 1)

    def _take(self, peer, d: dict) -> bool:
        block = Block.from_dict(d)
        if block.height != self.chain.tip.height + 1:
            return block.height <= self.chain.tip.height
        record = None
        if block.summary is not None and block.summary.record_hash:
            record = peer.get(f"/round/{block.height}/record")
        with self.coord.lock:
            problem = None
            if self.chain.state.validators:  # (blocks from before a chain had validators were sealed by one node)
                problem = check_certificate(d.get("certificate"), block.height, block.hash.hex(),
                                            self.chain.state.validator_set(), self.chain.chain_id)
            if problem is None and record is not None:
                problem = "; ".join(replay_block(self.chain, block, record,
                                                 datasets=PeerDatasets(self, peer.url))) or None
        if problem:
            log.error("refusing block %d from %s: %s", block.height, peer.url, problem)
            return False
        try:
            self._commit(block, record, d.get("certificate"))
        except (InvalidBlock, InvalidJob) as e:
            log.error("refusing block %d from %s: %s", block.height, peer.url, e)
            return False
        return True

    def _latest_round(self, job_id: str) -> int:
        return next((b.height for b in reversed(self.chain.blocks) if b.summary and b.summary.job_id == job_id), 0)

    def _ensure_params(self, job_id: str) -> bool:
        """Whether we have the weights the job's last round on chain trained (its next round starts from them),
        fetching them from a peer if not. They're checked against the hash the chain holds."""
        c, job = self.coord, self.chain.state.job(job_id)
        want = c.expected_start_hash(job)
        have = self.store.load_params(job_id)
        if params_hash(have if have is not None else job.spec.init_params(job.init_seed)) == want:
            return True
        for url in self.peers.urls():
            try:  # (the size is our job's; a peer that hasn't trained it has nothing we lack)
                _, raw = self.peers.client(url, timeout=60).download_model(job_id, job, untrained=False)
            except NodeError:
                continue
            weights = np.frombuffer(raw, dtype=job.wire_dtype).astype(np.float64)
            if weights.size != job.spec.n_params or params_hash(weights) != want:
                continue
            height = self._latest_round(job_id)
            with c.lock:
                if c.expected_start_hash(job) != want:
                    return False  # the job moved on again meanwhile
                self.store.save_params(job_id, height, weights)
                c.params.pop(job_id, None)
            if not job.finetune:  # a fine-tune's scores come with its records (Coordinator.adopt); nodes don't run models
                try:
                    full = c.job(job_id)  # with its data, to score the model as its poster will see it
                    summary = self.chain.blocks[height].summary
                    self.store.save_round(job_id, height, summary.accepted_units, *full.evaluate(weights))
                except (InvalidJob, OSError, ValueError, AttributeError):
                    pass
            return True
        return False

    def _fetch_weights(self, job_id: str) -> None:
        for _ in range(5):
            try:
                if self._ensure_params(job_id) or self.stopping.wait(2):
                    return
            except Exception:  # noqa: BLE001
                log.exception("fetching job %s's weights failed", job_id)
                return

    def _fetch_missing_data(self) -> None:
        """Datasets of jobs with budget left, from whichever peer the requester uploaded them to: without a job's
        data, a validator can't lead its rounds."""
        s = self.chain.state
        for job_id in [j for j, v in s.escrow.items() if v > 0]:
            src = s.job(job_id).data_source
            if src.get("kind") not in ("npz", "tokens") or self.store.datasets.has(src["sha256"]):
                continue
            for url in self.peers.urls():
                try:
                    peer = self.peers.client(url, timeout=60)
                    if not peer.get(f"/dataset/{src['sha256']}/status")["complete"]:
                        continue
                    peer.download("dataset", src["sha256"], self.store.datasets, check=self.store.check_dataset)
                    log.info("fetched job %s's dataset from %s", job_id, url)
                    break
                except (NodeError, InvalidJob, BlobError, KeyError, TypeError) as e:
                    log.debug("dataset %s from %s: %s", src["sha256"][:12], url, e)

    def submit_tx(self, tx: Tx) -> None:
        if tx.kind == "post_job":  # relay policy, not a rule: a block with a bigger job is as valid as before
            size = TrainingJob.from_spec_json(tx.memo, tx.sender, tx.amount).checkpoint_bytes
            if size > self.max_job_bytes:
                raise ProtocolError(f"this node relays jobs whose models are at most {self.max_job_bytes / 2**20:g} MB "
                                    f"a checkpoint (--max-job-mb); this one's are {size / 2**20:.1f} MB")
        self.coord.submit_tx(tx)
        # On to whoever proposes next (peers that have it already say no): as parsed, so only what the tx is goes on.
        self.peers.broadcast("/tx", tx.to_dict())

    def receive(self, msg: dict) -> None:
        if self.bft:
            self.engine.receive(msg)

    def proposal_limit(self) -> int:
        """The biggest /consensus body we read. Only a validator votes on proposals, and only another one sends it a
        big one: a chain's sole validator (like the public node), or a node that doesn't vote, never needs to."""
        vals = self.chain.state.validator_set()
        return MAX_PROPOSAL if self.keys.address in vals and len(vals) > 1 else MAX_JSON

    # --- read-only views ---------------------------------------------------------------

    def status(self) -> dict:
        c = self.coord
        consensus = self.engine.status() if self.bft else None  # before taking the round lock: the engine's comes first
        with c.lock:
            r = c.r
            out = {
                "chain_id": self.chain.chain_id, "genesis_hash": self.chain.genesis_hash,
                "height": self.chain.tip.height, "tip": self.chain.tip.hash.hex(),
                "phase": c.phase.value, "phase_ends": self.phase_ends,
                "round": r.round_no if r else None, "job_id": r.job_id if r else None,
                "attempt": r.record.get("attempt") if r else None,
                "beacon": r.beacon.hex() if r and r.beacon else None,
                # The hash of the round's secret, public from the moment the round opens: the leader can't pick the
                # secret (and with it the beacon) after seeing the commitments. The record reveals it (coordinator.py).
                "beacon_commit": r.record.get("beacon_commit") if r else None,
                "challenge_segments": c.config.challenge_segments, "min_stake": econ.MIN_STAKE,
                "mempool": len(c.mempool), "chunk_bytes": CHUNK, "keeps_checkpoints_for": self.keep_rounds,
                "economics": self.chain.state.economics.to_dict(), "subsidy": self.chain.state.subsidy(self.chain.tip.height + 1),
                "min_fee": c.config.min_fee, "earn_in": self.chain.state.earn_in,
                # Jobs with budget that aren't training, and why, in words (a fine-tune with too few workers to check it).
                "waiting": dict(c.waiting),
            }
        if consensus is not None:
            # Workers do the round with its leader, the validator whose turn it is to propose: None means this node.
            leader = consensus["proposer"]
            out["consensus"] = consensus
            if self.keys.address not in self.chain.state.validator_set():
                out["leader"] = self.leader_hint
            else:
                out["leader"] = None if leader == self.keys.address else self.chain.state.validators.get(leader)
        return out

    def account(self, addr: str) -> dict:
        s = self.chain.state
        return {"address": addr, "balance": s.balance(addr), "stake": s.stake(addr), "bootstrap": s.locked.get(addr, 0),
                "unbonding": s.unbonding.get(addr, []), "nonce": s.nonces.get(addr, 0), "reputation": s.rep(addr)}

    def job_info(self, job_id: str) -> dict:
        s = self.chain.state
        if job_id not in s.jobs:
            raise KeyError(job_id)
        if not self.store.params_height(job_id):  # untrained: its size is its spec's, and there's nothing to hash
            model = {"height": 0, "bytes": s.job(job_id).checkpoint_bytes, "sha256": None}
        else:
            height, data, sha = self._model_bytes(job_id)
            model = {"height": height, "bytes": len(data), "sha256": sha}
        out = {"spec": json.loads(s.jobs[job_id]), "owner": s.job_owner[job_id], "escrow": s.escrow.get(job_id, 0),
               "rounds": self.store.job_rounds(job_id), "model": model}
        if "deepseek" in out["spec"]:  # where its prep is, and the records' root once it's done (trainers check it)
            try:
                state = self.coord.prep(self.coord.job(job_id))
                out["prep"] = {"done": state.done, "root": state.root}
            except Exception:  # noqa: BLE001 - its data isn't on this node
                out["prep"] = {"done": False, "root": None}
        return out

    def _model_bytes(self, job_id: str) -> tuple[int, bytes, str]:
        """The job's latest trained weights at its precision, serialised once per round. Workers get the weights every
        round anyway, so serving them openly hides nothing. An untrained job's aren't served: they follow from its
        spec, so its poster can build them (client.download_model does), and building them here for anyone who asks
        took gigabytes."""
        height = self.store.params_height(job_id)
        with self._model_lock:
            if self._model_cache[0] != (job_id, height):
                job = self.chain.state.job(job_id)
                params = self.store.load_params(job_id)
                if params is None:
                    raise KeyError(f"job {job_id} hasn't been trained yet; its starting weights follow from its spec")
                data = np.asarray(params, dtype=job.wire_dtype).tobytes()
                self._model_cache = ((job_id, height), data, hashlib.sha256(data).hexdigest())
            return height, self._model_cache[1], self._model_cache[2]

    def network(self) -> dict:
        """The website's live panel: miners lately, every job and how it's going (netstats.py). Worked out at most once
        a block, or a minute, whichever comes first, however many pages ask."""
        tip, now = self.chain.tip, time.time()
        with self._network_lock:
            if self._network_cache[0] != tip.hash or now - self._network_cache[1] > 60:
                self._network_cache = (tip.hash, now, self._network(tip, now))
            return self._network_cache[2]

    def _network(self, tip: Block, now: float) -> dict:
        s, blocks = self.chain.state, self.chain.blocks[:tip.height + 1]
        recent = []  # the last day's blocks, and at least 50 to time them by
        for b in reversed(blocks):
            if now - b.timestamp > netstats.DAY_S and len(recent) >= 50:
                break
            recent.append({"height": b.height, "timestamp": b.timestamp, "summary": asdict(b.summary) if b.summary else None})
        recent.reverse()
        with self.coord.lock:
            r = self.coord.r
            current = r.job_id if r else None
            rnd = {"no": r.round_no if r else None, "phase": self.coord.phase.value, "job_id": current}
        jobs = [netstats.job_summary(j, json.loads(spec), s.escrow.get(j, 0), self.store.job_rounds(j), current,
                                     lambda h: blocks[h].timestamp if 0 <= h < len(blocks) else None)
                for j, spec in list(s.jobs.items())]
        for j in jobs:  # a DeepSeek job's prep, as the front page draws it (netstats: "prep")
            if "deepseek" in json.loads(s.jobs[j["id"]]):
                prep = self.coord.prep_summary(j["id"], now)
                if prep is not None:
                    j["prep"] = prep
        order = {"training": 0, "open": 1, "finished": 2}
        jobs.sort(key=lambda j: (order[j["status"]], -j["escrow"], j["id"]))
        return {"chain_id": self.chain.chain_id, "height": tip.height, "time": tip.timestamp, "now": now,
                "subsidy": s.subsidy(tip.height + 1), "validators": len(s.validator_set()), "round": rnd,
                "miners": netstats.summarize(recent, now), "jobs": jobs}

    def job_model(self, job_id: str, offset: int) -> bytes:
        if job_id not in self.chain.state.jobs:
            raise KeyError(job_id)
        _, data, _ = self._model_bytes(job_id)
        return data[offset:offset + CHUNK]

    def round_params(self) -> dict:
        with self.coord.lock:
            r = self.coord.r
            if r is None or self.start_blob is None:
                raise ProtocolError("no round open")
            job = self.coord.job(r.job_id)
            return {"round": r.round_no, "start_hash": r.start_hash.hex(), "dtype": job.wire_dtype,
                    "n": job.spec.n_params, "blob": self.start_blob}

    # --- worker messages ----------------------------------------------------------------

    def _load_blob(self, blob_id: str, job) -> np.ndarray:
        if not self.blobs.has(blob_id):
            raise ProtocolError(f"checkpoint {blob_id[:12]}… hasn't been uploaded yet")
        arr = self.blobs.load(blob_id, job.wire_dtype, job.spec.n_params)
        self._blob_of[id(arr)] = blob_id
        return arr

    def _blob_id(self, arr: np.ndarray, job) -> str:
        """Blob id for an opening we're about to hand a verifier (canaries are stored on demand)."""
        bid = self._blob_of.get(id(arr))
        if bid is None:
            bid = self.blobs.put_array(arr, job.wire_dtype)
            self._blob_of[id(arr)] = bid
        return bid

    def handle(self, kind: str, env: dict) -> dict:
        c = self.coord
        # Only what wire.seal sends, and no bigger than an honest message (see ENVELOPE): replay and receipts take a
        # registration or verdict exactly as sent, so nothing can be stripped from those, only refused.
        if not isinstance(env, dict) or not set(env) <= ENVELOPE or not isinstance(env.get("body"), dict):
            raise WireError("malformed envelope")
        if not set(env["body"]) <= BODIES.get(kind, set()):
            raise WireError(f"a {kind} message has only {', '.join(sorted(BODIES.get(kind, ())))}")
        if len(canonical(env)) > {"reveal": MAX_REVEAL, "move": MAX_MOVE, "sign": MAX_REVEAL}.get(kind, MAX_MESSAGE):
            raise WireError(f"{kind} message too large")
        addr, body = open_envelope(env, kind, self.chain.chain_id)
        with c.lock:
            r = c.r
            if r is None or body.get("round") != r.round_no:
                raise ProtocolError("no such round open")
            job = c.job(r.job_id)
            if kind == "register":
                entry = c.register(addr, float(body["flops_per_sec"]), int(body["memory_mb"]), envelope=env)
                return {"ok": True, "receipt": self._receipt(kind, entry)}
            if kind == "assignment":
                units = c.assignment(addr)
                out = {"units": [u.to_dict() for u in units]}
                if r.prep:  # a DeepSeek prep round: which unit of the work graph each is, and what its inputs must be
                    out["prep"] = {u.unit_id: r.prep[u.unit_id] for u in units}
                    out["want"] = c.prep(job).want()
                return out
            if kind == "commit":
                cm = commitment_in(body["commitment"])
                if cm.worker != addr:
                    raise ProtocolError("can only commit for yourself")
                if len(cm.digest) != 32:
                    raise WireError("a commitment's digest is a SHA-256")
                c.commit(cm)
                return {"ok": True, "receipt": self._receipt(kind, commitment_out(cm))}
            if kind == "sign":  # a DeepSeek unit's links to the bucket (deepseek_rounds; Coordinator.may_sign)
                reqs = body["requests"]
                if self.bucket is None:
                    raise ProtocolError("this node has no bucket for DeepSeek jobs' hand-offs")
                if not isinstance(reqs, list) or not 1 <= len(reqs) <= 64:
                    raise WireError("a sign message asks for 1-64 links")
                urls = []
                for req in reqs:
                    try:
                        method, name, query = str(req[0]), str(req[1]), dict(req[2] or {})
                    except (TypeError, IndexError, ValueError) as e:
                        raise WireError(f"a link request is [method, name, query]: {e}") from None
                    c.may_sign(addr, method, name, query)
                    urls.append(self.bucket(method, f"{r.job_id}/{name}", {k: str(v) for k, v in query.items()}))
                return {"urls": urls}
            if kind == "reveal" and isinstance(body.get("reveal"), dict) and body["reveal"].get("prep"):
                d = body["reveal"]  # a DeepSeek prep unit's: the root of its commitment in the bucket, and the salt
                try:
                    root, salt = bytes.fromhex(d["merkle_root"]), bytes.fromhex(d["salt"])
                    if len(root) != 32 or len(salt) > 64:
                        raise WireError("more in it than a reveal has")
                    rv = Reveal(str(d["unit_id"]), str(d["worker"]), root, salt, None, {})
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    raise WireError(f"bad reveal: {e}") from e
                if rv.worker != addr:
                    raise ProtocolError("can only reveal for yourself")
                wire = {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": root.hex(), "salt": salt.hex(), "prep": True}
                verdict = c.reveal(rv, wire=wire)
                return {"ok": verdict is None, "verdict": verdict.value if verdict else "pending", "receipt": self._receipt(kind, wire)}
            if kind == "reveal":
                d = body["reveal"]
                try:
                    openings = blob_openings_in(d["openings"], job.n_segments)
                    final, root, salt = check_id(d["final"]), bytes.fromhex(d["merkle_root"]), bytes.fromhex(d["salt"])
                    # No more than an honest reveal holds: a 32-byte root, a short salt, the first and last checkpoints
                    # and both ends of each challenged segment, each proven as deep as the unit's Merkle tree goes.
                    if (len(root) != 32 or len(salt) > 64 or len(openings) > 2 * c.config.challenge_segments + 2
                            or any(len(pr) > job.n_segments.bit_length() for _, pr in openings.values())):
                        raise WireError("more in it than a reveal has")
                    rv = Reveal(str(d["unit_id"]), str(d["worker"]), root, salt, self._load_blob(final, job),
                                {i: (self._load_blob(bid, job), pr) for i, (bid, pr) in openings.items()})
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    raise WireError(f"bad reveal: {e}") from e
                if rv.worker != addr:
                    raise ProtocolError("can only reveal for yourself")
                # Recorded as parsed, not as sent, so the record holds only what a reveal is (an honest worker's is the
                # same either way, so its receipt still vouches for what it sent).
                wire = {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(), "salt": rv.salt.hex(),
                        "final": final, "openings": blob_openings_out(openings)}
                verdict = c.reveal(rv, wire=wire)
                return {"ok": verdict is None, "verdict": verdict.value if verdict else "pending", "receipt": self._receipt(kind, wire)}
            if kind == "tasks":
                tasks = []
                for t in c.tasks_for(addr):
                    d = {"task_id": t.task_id, "unit": t.unit.to_dict(), "segments": t.segments, "dtype": job.wire_dtype,
                         "openings": blob_openings_out({i: (self._blob_id(a, job), pr) for i, (a, pr) in t.openings.items()})}
                    if t.kind == "prep":  # a DeepSeek prep unit's: what to replay, and what to check it against
                        d |= {"kind": "prep", **c.prep_task(t)}
                    elif t.kind != "check":  # fine-tunes': only new workers get these, so old ones see what they always did
                        d |= {"kind": t.kind}
                    tasks.append(d)
                return {"tasks": tasks}
            if kind == "verdict":
                bad = body.get("bad_segment")
                c.submit_verdict(addr, str(body["task_id"]), bool(body["ok"]), None if bad is None else int(bad), envelope=env,
                                 score=body.get("score"))
                return {"ok": True, "receipt": self._receipt(kind, env)}
            if kind == "move":
                view = c.submit_move(addr, {k: body.get(k) for k in ("game", "n", "move")}, envelope=env, now=time.time())
                return {**view, "receipt": self._receipt(kind, env)}
        raise ProtocolError(f"unknown message {kind}")

    def _receipt(self, kind: str, entry: dict) -> dict:
        """Our signed promise that `entry` will be in this round's record (receipts.py). Called holding the lock."""
        r = self.coord.r
        return receipts.make(self.coord.keys, self.chain.chain_id, r.round_no, r.record["attempt"], kind, entry)

    # --- binary uploads -------------------------------------------------------------------

    def _uploader(self, kind: str, blob_id: str, total: int, headers) -> str:
        try:
            pub, sig = bytes.fromhex(headers.get("X-VibeHash-Pubkey", "")), bytes.fromhex(headers.get("X-VibeHash-Sig", ""))
            ts = int(headers.get("X-VibeHash-Ts", "0"))
        except ValueError as e:
            raise WireError("missing or malformed upload signature") from e
        if abs(time.time() - ts) > MAX_CLOCK_SKEW_S:
            raise WireError("upload timestamp too far from node clock")
        if not verify(pub, canonical([self.chain.chain_id, kind, blob_id, total, ts]), sig):
            raise WireError("bad upload signature")
        return address_of(pub)

    def authorize_blob(self, blob_id: str, total: int, headers) -> str:
        """Only this round's workers may upload checkpoints, of exactly this job's size, within a quota.
        Returns the uploader's address."""
        addr = self._uploader("blob", blob_id, total, headers)
        c = self.coord
        with c.lock:
            r = c.r
            if r is not None and c.phase == Phase.DISPUTE:  # a defender's states and evidence
                if not any(g.defender == addr and g.turn != "done" for g in r.games.values()):
                    raise ProtocolError("only a dispute's defender uploads, while its dispute lasts")
                done = self.trace_uploads.get(addr, 0)
                if not 0 < total <= MAX_TRACE_BLOB or done + total > MAX_TRACE_UPLOADS:
                    raise ProtocolError("upload quota for this round's disputes used up")
                if not self.blobs.has(blob_id) and self.blobs.received(blob_id, addr) == 0:
                    self.trace_uploads[addr] = done + total
                return addr
            if r is None or c.phase not in (Phase.COMPUTE, Phase.REVEAL) or not r.assignments.get(addr):
                raise ProtocolError("checkpoints can only be uploaded by this round's workers, before the reveal deadline")
            job = c.job(r.job_id)
            if total != job.spec.n_params * np.dtype(job.wire_dtype).itemsize:
                raise ProtocolError("upload isn't the size of one checkpoint for this job")
            mine = self.uploads.setdefault(addr, set())
            if blob_id not in mine and len(mine) >= len(r.assignments[addr]) * (job.n_segments + 1):
                raise ProtocolError("checkpoint upload quota for this round used up")
            mine.add(blob_id)
        return addr

    def authorize_dataset(self, sha: str, total: int, headers) -> str:
        addr = self._uploader("dataset", sha, total, headers)
        if self.chain.state.balance(addr) < MIN_UPLOADER_BALANCE:
            raise ProtocolError("uploading a dataset needs at least 1 VIBE balance")
        if not 0 < total <= MAX_DATASET:
            raise ProtocolError(f"datasets can be at most {MAX_DATASET // 2**30} GB")
        return addr


def _number(text, what: str) -> int:
    """A count or offset from a query or header: a whole number that SQLite and seek() can take, else a 400."""
    try:
        n = int(text)
    except (TypeError, ValueError):
        n = -1
    if not 0 <= n < 2**63:
        raise WireError(f"bad {what}: expected a whole number from 0")
    return n


def _handler(node: Node):
    stores = {"blob": node.blobs, "dataset": node.store.datasets}
    big = threading.BoundedSemaphore(MAX_BIG_BODIES)

    class Handler(GuardedHandler):
        server_version = "vibehash-node/0.5"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            if self.command == "GET":  # everything a GET answers is public: any web page may read it
                self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _length(self) -> int:
            n = self.headers.get("Content-Length") or "0"
            if not (n.isascii() and n.isdigit()):  # int() takes "-1", and read(-1) reads until the client hangs up
                raise WireError("bad Content-Length")
            return int(n)

        def _body(self, limit: int) -> bytes:
            n = self._length()
            if n > limit:
                raise WireError("request too large")
            return self.rfile.read(n)

        def _json(self, limit: int):
            try:
                return json.loads(self._body(limit) or b"{}")
            except RecursionError:  # nested thousands deep
                raise WireError("malformed JSON") from None

        def _proposal(self):
            """A /consensus body, which can be a proposal up to node.proposal_limit(). Bigger than MAX_JSON, it's read
            only MAX_BIG_BODIES at a time; past that, the sender is told to come back (503), after its body is read
            and dropped, so it gets the answer instead of a reset."""
            n = self._length()
            if n <= MAX_JSON or n > node.proposal_limit():
                return self._json(MAX_JSON)
            if not big.acquire(blocking=False):
                while n > 0 and (got := len(self.rfile.read(min(n, 2**16)))):
                    n -= got
                raise Busy("busy with other large messages")
            try:
                return self._json(n)
            finally:
                big.release()

        def _guard(self, fn, raw: bool = False):
            try:
                result = fn()
                self._send(200, result, "application/octet-stream" if raw else "application/json")
            except KeyError as e:
                self._send(404, {"error": f"not found: {e}"})
            except (ProtocolError, WireError, InvalidJob, BlobError, ValueError, TypeError, OverflowError) as e:
                self._send(400, {"error": str(e)})
            except Busy as e:
                self._send(503, {"error": f"{e}; retry shortly"})
            except Exception as e:  # pragma: no cover
                log.exception("request failed")
                self._send(500, {"error": type(e).__name__})

        def do_GET(self):
            url = urlparse(self.path)
            parts = path_parts(self.path)  # as the rate limit sorted it
            q = parse_qs(url.query)

            if len(parts) in (2, 3) and parts[0] in stores:
                store, bid = stores[parts[0]], parts[1]
                if len(parts) == 3 and parts[2] == "status":
                    self._guard(lambda: {"received": store.received(check_id(bid), q.get("from", [""])[0]),
                                         "complete": store.has(bid), "size": store.size(bid)})
                else:
                    self._guard(lambda: store.read_chunk(check_id(bid), _number(q.get("offset", ["0"])[0], "offset"), CHUNK), raw=True)
                return
            if len(parts) == 3 and parts[0] == "job" and parts[2] == "model":
                self._guard(lambda: node.job_model(parts[1], _number(q.get("offset", ["0"])[0], "offset")), raw=True)
                return

            def route():
                if parts == ["status"]:
                    return node.status()
                if parts == ["genesis"]:
                    return node.genesis
                if parts == ["network"]:
                    return node.network()
                if parts == ["blocks"]:
                    lo = _number(q.get("from", ["0"])[0], "from")
                    hi = min(_number(q.get("to", [str(lo + 99)])[0], "to"), lo + 99, node.chain.tip.height)
                    certs = node.store.certificates(lo, hi)
                    page, size = [], 0
                    for b in node.chain.blocks[lo:hi + 1]:  # a short page just means another request
                        if size >= MAX_BLOCKS_REPLY:
                            break
                        page.append(b.to_dict() | ({"certificate": certs[b.height]} if b.height in certs else {}))
                        size += len(json.dumps(page[-1]))
                    return page
                if len(parts) == 2 and parts[0] == "account":
                    return node.account(parts[1])
                if parts == ["validators"]:
                    s = node.chain.state
                    voting = s.validator_set()
                    return {a: {"url": u, "stake": s.stake(a), "votes": a in voting} for a, u in s.validators.items()}
                if len(parts) == 2 and parts[0] == "job":
                    return node.job_info(parts[1])
                if parts == ["round", "params"]:
                    return node.round_params()
                if parts == ["round", "games"]:
                    return node.coord.game_views()
                if len(parts) == 3 and parts[0] == "round" and parts[2] == "record":
                    record = node.store.load_record(_number(parts[1], "height"))
                    if record is None:
                        raise KeyError(f"no record for block {parts[1]}")
                    return record
                raise KeyError(url.path)

            self._guard(route)

        def do_PUT(self):
            parts = path_parts(self.path)

            def route():
                if len(parts) != 2 or parts[0] not in stores:
                    raise KeyError(self.path)
                kind, bid = parts[0], check_id(parts[1])
                offset, total = _number(self.headers.get("X-Offset"), "X-Offset"), _number(self.headers.get("X-Total"), "X-Total")
                if kind == "blob":
                    uploader = node.authorize_blob(bid, total, self.headers)
                else:
                    uploader = node.authorize_dataset(bid, total, self.headers)
                chunk = self._body(CHUNK)
                # A dataset is checked by the write that completes it, before it's kept (not on a PUT to one that's
                # already complete, which would load it all again for anyone who asked).
                received = stores[kind].write_chunk(bid, offset, chunk, total, uploader,
                                                    check=node.store.check_dataset if kind == "dataset" else None)
                return {"received": received, "complete": stores[kind].has(bid)}

            self._guard(route)

        def do_POST(self):
            parts = path_parts(self.path)

            def route():
                if parts == ["consensus"]:
                    node.receive(self._proposal())
                    return {"ok": True}
                worker = len(parts) == 1 and parts[0] in BODIES
                body = self._json(MAX_REVEAL if worker else MAX_JSON)
                if not isinstance(body, dict):
                    raise WireError("expected a JSON object")
                if parts == ["tx"]:
                    tx = Tx.from_dict(body)
                    node.submit_tx(tx)
                    return {"ok": True, "txid": tx.txid.hex()}
                if worker:
                    return node.handle(parts[0], body)
                raise KeyError(self.path)

            self._guard(route)

    return Handler


class TraceBlobs:
    """A dispute's states and evidence, in the node's blob store: what Coordinator.trace_blobs needs."""

    def __init__(self, store: BlobStore):
        self.store, self.roots = store, {}

    def put_bytes(self, raw: bytes) -> str:
        return self.store.put_array(np.frombuffer(raw, dtype=np.uint8), "uint8")

    def get_bytes(self, blob_id: str) -> bytes | None:
        return self.store.path(blob_id).read_bytes() if self.store.has(blob_id) else None

    def size(self, blob_id: str) -> int | None:
        return self.store.size(blob_id) if self.store.has(blob_id) else None

    def tensor_root(self, blob_id: str, rows: int, cols: int) -> str | None:
        """The root of a blob read as a [rows, cols] tensor, or None when it isn't on the node at that size. Checked on
        every call, though the root is cached: a blob pruned since isn't on the node for a challenger to fetch. And
        cached by shape, since the same bytes (zeros, say) make another root in another shape."""
        from .merkle import root_of

        if not self.store.has(blob_id) or self.store.size(blob_id) != rows * cols * 4:
            return None
        key = (blob_id, rows, cols)
        if key not in self.roots:
            self.roots[key] = root_of(np.fromfile(self.store.path(blob_id), dtype="<f4").reshape(rows, cols))
        return self.roots[key]

    def forget(self) -> None:
        """At a round's end: its roots go with its blobs."""
        self.roots = {}


class PeerDatasets:
    """A job's dataset for a replay whose disputes need it (which sequences a step trains on): this node's own, else
    fetched (from `first`, the block's proposer or the peer that sent it, then any peer) and checked on arrival as an
    upload is. Without it a validator would refuse every block with a dispute over a job whose data it hadn't fetched
    yet, and a node syncing would stop at one whose job has since ended."""

    def __init__(self, node: Node, first: str | None = None):
        self.node, self.first = node, first

    def attach(self, job) -> None:
        store, sha = self.node.store, job.data_source.get("sha256", "")
        if not store.datasets.has(sha):
            urls = [self.first] if self.first else []
            for url in urls + [u for u in self.node.peers.urls() if u != self.first]:
                try:
                    self.node.peers.client(url, timeout=120).download("dataset", sha, store.datasets,
                                                                      check=store.check_dataset)
                    break
                except (NodeError, InvalidJob, BlobError, KeyError, TypeError, ValueError) as e:
                    log.debug("dataset %s from %s: %s", sha[:12], url, e)
        store.attach_dataset(job)


class LeaderBlobs:
    """A round's checkpoints, for a validator checking its disputes' starting states (replay_block's `roots`): this
    node's own when it has them, else fetched from the block's proposer and checked by their id."""

    def __init__(self, node: Node, proposer: str):
        self.node, self.proposer = node, proposer

    def array(self, blob_id: str, job) -> np.ndarray:
        store = self.node.blobs
        if not store.has(blob_id):
            url = self.node.chain.state.validators.get(self.proposer)
            if not url:
                raise ProtocolError("the proposer has no address to fetch its checkpoints from")
            self.node.peers.client(url, timeout=120).download("blob", blob_id, store, size=job.checkpoint_bytes)
        return np.asarray(store.load(blob_id, job.wire_dtype, job.spec.n_params), dtype=np.float64)

    def bytes(self, blob_id: str) -> bytes:
        """A dispute's evidence, by its id (checked on arrival)."""
        store = self.node.blobs
        if not store.has(blob_id):
            url = self.node.chain.state.validators.get(self.proposer)
            if not url:
                raise ProtocolError("the proposer has no address to fetch the evidence from")
            self.node.peers.client(url, timeout=120).download("blob", blob_id, store)
        return store.path(blob_id).read_bytes()
