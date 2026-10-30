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
Signed-envelope endpoints (workers):
    POST /register  /assignment  /commit  /reveal  /tasks  /verdict
    (register, commit, reveal and verdict answer with a signed receipt: proof the message will be in the
    round's record, so a node that drops it can be caught; see receipts.py)
Binary transfer (chunked, resumable, content-addressed; no size cap from HTTP):
    GET  /blob/<sha256>/status   bytes received so far
    GET  /blob/<sha256>?offset=N one chunk of a checkpoint
    PUT  /blob/<sha256>          upload one chunk (signed headers; registered workers only)
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
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import economics as econ
from . import receipts
from .blobs import CHUNK, BlobError, BlobStore, check_id
from .client import NodeError
from .consensus import Engine, Timeouts, check_certificate
from .coordinator import Coordinator, CoordinatorConfig, Phase, ProtocolError
from .crypto import KeyPair, address_of, verify
from .ledger import Block, InvalidBlock, Tx, apply_block, canonical
from .model import evaluate, params_hash
from .p2p import Peers
from .ratelimit import GuardedHandler, GuardedServer, Limits
from .replay import replay_block
from .storage import Store
from .tasks import SYN_PER_VIBE, InvalidJob, Reveal
from .wire import MAX_CLOCK_SKEW_S, WireError, blob_openings_in, blob_openings_out, commitment_in, commitment_out, open_envelope

log = logging.getLogger("vibehash.node")
MAX_JSON = 4 * 2**20  # every JSON message is small now; weights travel as blobs
MAX_DATASET = 20 * 2**30
MIN_UPLOADER_BALANCE = 1 * SYN_PER_VIBE


@dataclass
class PhaseTimes:
    register_s: float = 3.0
    compute_s: float = 20.0
    reveal_s: float = 5.0
    verify_s: float = 10.0
    idle_s: float = 2.0

    @property
    def round_s(self) -> float:
        return self.register_s + self.compute_s + self.reveal_s + self.verify_s


class Node:
    def __init__(self, data_dir, genesis: dict, keys: KeyPair, host="127.0.0.1", port=8545,
                 times: PhaseTimes | None = None, config: CoordinatorConfig | None = None, keep_rounds: int = 10,
                 limits: Limits | None = None, tls: ssl.SSLContext | None = None,
                 peers: list[str] = (), announce: str | None = None, timeouts: Timeouts | None = None):
        """`peers`: other nodes to connect to (validators find each other from the chain without them).
        `announce`: the address other nodes reach this one at, if it isn't the one it listens on."""
        self.times = times or PhaseTimes()
        self.keep_rounds = keep_rounds  # rounds whose checkpoints stay available to re-check (the challenge window)
        cfg = config or CoordinatorConfig()
        cfg.compute_s = self.times.compute_s * 0.8  # leave slack for network and polling
        self.keys = keys
        self.store = Store(data_dir)
        self.chain = self.store.open_chain(genesis)
        self.genesis = genesis
        self.coord = Coordinator(self.chain, keys, cfg, store=self.store)
        self.blobs: BlobStore = self.store.blobs
        self.start_blob: str | None = None
        self.uploads: dict[str, set[str]] = {}  # per-worker blob ids this round (quota)
        self._blob_of: dict[int, str] = {}  # id(array) -> blob id, for arrays this round loaded from blobs
        self.phase_ends = 0.0
        self._model_lock = threading.Lock()
        self._model_cache: tuple = (None, b"", "")
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
                             persist=lambda state: self.store.save_meta("consensus", state))
        self.proposed: dict[str, tuple] = {}  # block id -> (weights, scores, record) for blocks we proposed at this height
        self.commit_lock = threading.Lock()
        self.wake_sync = threading.Event()
        self.leader_hint: str | None = None  # the current leader, as a validator told us (for nodes that don't vote)
        self.blobs.prune(self._evidence())  # keep recent rounds' evidence; the rest belongs to an abandoned round

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
        """Checkpoints recent rounds' records point to, kept for `keep_rounds` rounds so anyone can re-run those
        rounds' weight-based checks, plus those of blocks we've proposed and the validators are still voting on.
        Unopened checkpoints and older rounds' are deleted."""
        keep = self.store.evidence(self.chain.tip.height, self.keep_rounds)
        for _, _, record in list(self.proposed.values()):
            for rv in record.get("reveals", []):
                keep.update(b for b in [rv.get("final"), *(o[0] for o in (rv.get("openings") or {}).values())] if b)
        return keep

    def _end_round(self) -> None:
        with self.coord.lock:
            self.start_blob, self.uploads, self._blob_of = None, {}, {}
        self.blobs.prune(self._evidence())

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
        log.info("block %d sealed: %d units accepted, %d fraud, val acc %.1f%%, pool %.3f VIBE",
                 rep.block.height, rep.block.summary.accepted_units, fraud, 100 * rep.val_acc, rep.pool / SYN_PER_VIBE)

    def _run_round(self, still=None) -> bool:
        """Drive an open round through its phases. False if it was given up: we're stopping, or `still()` says the
        round's no longer wanted (another block was decided at its height)."""
        c, t = self.coord, self.times
        with c.lock:
            job = c.job(c.r.job_id)
            self.start_blob = self.blobs.put_array(c.r.start, job.wire_dtype)
        log.info("round %d open for job %s", c.r.round_no, c.r.job_id)
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
            with self.coord.lock:
                tip = self.chain.tip
                if block.height != tip.height + 1 or block.prev_hash != tip.hash:
                    return False
                if block.timestamp > time.time() + MAX_CLOCK_SKEW_S:
                    raise InvalidBlock("timestamp is in the future")
                apply_block(self.chain.state, block, self.chain.chain_id, tip)
                if block.summary is not None:
                    record = value.get("record")
                    problems = ["no round record"] if not isinstance(record, dict) else replay_block(self.chain, block, record)
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
        self._commit(block, value.get("record"), cert, mine)
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
        replay from its record, and carry a valid commit certificate."""
        for url in self.peers.urls():
            peer = self.peers.client(url)
            try:
                st = peer.status()
                self.peers.alive(url)
                if st.get("genesis_hash") != self.chain.genesis_hash:
                    continue
                if self.keys.address not in self.chain.state.validator_set() and "consensus" in st:
                    self.leader_hint = st.get("leader") or url  # None: the peer is leading itself
                while st["height"] > self.chain.tip.height:
                    lo = self.chain.tip.height + 1
                    batch = peer.get(f"/blocks?from={lo}&to={lo + 99}")
                    if not batch:
                        break
                    for d in batch:
                        if not self._take(peer, d):
                            return
            except NodeError as e:
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
                problem = "; ".join(replay_block(self.chain, block, record)) or None
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
            try:
                _, raw = self.peers.client(url, timeout=60).download_model(job_id)
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
            try:
                full = c.job(job_id)  # with its data, to score the model as its poster will see it
                summary = self.chain.blocks[height].summary
                self.store.save_round(job_id, height, summary.accepted_units,
                                      *evaluate(full.spec, weights, full.data.x_val, full.data.y_val))
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
            if src.get("kind") != "npz" or self.store.datasets.has(src["sha256"]):
                continue
            for url in self.peers.urls():
                try:
                    peer = self.peers.client(url, timeout=60)
                    if not peer.get(f"/dataset/{src['sha256']}/status")["complete"]:
                        continue
                    peer.download("dataset", src["sha256"], self.store.datasets)
                    self.store.check_uploaded_dataset(src["sha256"])
                    log.info("fetched job %s's dataset from %s", job_id, url)
                    break
                except (NodeError, InvalidJob, BlobError) as e:
                    log.debug("dataset %s from %s: %s", src["sha256"][:12], url, e)

    def submit_tx(self, tx: Tx, body: dict) -> None:
        self.coord.submit_tx(tx)
        self.peers.broadcast("/tx", body)  # on to whoever proposes next; peers that have it already say no

    def receive(self, msg: dict) -> None:
        if self.bft:
            self.engine.receive(msg)

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
                "challenge_segments": c.config.challenge_segments, "min_stake": econ.MIN_STAKE,
                "mempool": len(c.mempool), "chunk_bytes": CHUNK, "keeps_checkpoints_for": self.keep_rounds,
                "economics": self.chain.state.economics.to_dict(), "subsidy": self.chain.state.subsidy(self.chain.tip.height + 1),
                "min_fee": c.config.min_fee,
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
        return {"address": addr, "balance": s.balance(addr), "stake": s.stake(addr),
                "unbonding": s.unbonding.get(addr, []), "nonce": s.nonces.get(addr, 0), "reputation": s.rep(addr)}

    def job_info(self, job_id: str) -> dict:
        s = self.chain.state
        if job_id not in s.jobs:
            raise KeyError(job_id)
        height, data, sha = self._model_bytes(job_id)
        return {"spec": json.loads(s.jobs[job_id]), "owner": s.job_owner[job_id], "escrow": s.escrow.get(job_id, 0),
                "rounds": self.store.job_rounds(job_id), "model": {"height": height, "bytes": len(data), "sha256": sha}}

    def _model_bytes(self, job_id: str) -> tuple[int, bytes, str]:
        """The job's latest weights at its precision, serialised once per round. Untrained jobs serve their
        starting weights. Workers get the weights every round anyway, so serving them openly hides nothing."""
        height = self.store.params_height(job_id)
        with self._model_lock:
            if self._model_cache[0] != (job_id, height):
                job = self.chain.state.job(job_id)
                params = self.store.load_params(job_id)
                params = job.spec.init_params(job.init_seed) if params is None else params
                data = np.asarray(params, dtype=job.wire_dtype).tobytes()
                self._model_cache = ((job_id, height), data, hashlib.sha256(data).hexdigest())
            return height, self._model_cache[1], self._model_cache[2]

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
                return {"units": [u.to_dict() for u in c.assignment(addr)]}
            if kind == "commit":
                cm = commitment_in(body["commitment"])
                if cm.worker != addr:
                    raise ProtocolError("can only commit for yourself")
                c.commit(cm)
                return {"ok": True, "receipt": self._receipt(kind, commitment_out(cm))}
            if kind == "reveal":
                d = body["reveal"]
                try:
                    openings = blob_openings_in(d["openings"], job.n_segments)
                    rv = Reveal(str(d["unit_id"]), str(d["worker"]), bytes.fromhex(d["merkle_root"]), bytes.fromhex(d["salt"]),
                                self._load_blob(check_id(d["final"]), job),
                                {i: (self._load_blob(bid, job), pr) for i, (bid, pr) in openings.items()})
                except (KeyError, TypeError, ValueError) as e:
                    raise WireError(f"bad reveal: {e}") from e
                if rv.worker != addr:
                    raise ProtocolError("can only reveal for yourself")
                verdict = c.reveal(rv, wire=d)
                return {"ok": verdict is None, "verdict": verdict.value if verdict else "pending", "receipt": self._receipt(kind, d)}
            if kind == "tasks":
                return {"tasks": [{"task_id": t.task_id, "unit": t.unit.to_dict(), "segments": t.segments, "dtype": job.wire_dtype,
                                   "openings": blob_openings_out({i: (self._blob_id(a, job), pr) for i, (a, pr) in t.openings.items()})}
                                  for t in c.tasks_for(addr)]}
            if kind == "verdict":
                bad = body.get("bad_segment")
                c.submit_verdict(addr, str(body["task_id"]), bool(body["ok"]), None if bad is None else int(bad), envelope=env)
                return {"ok": True, "receipt": self._receipt(kind, env)}
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


def _handler(node: Node):
    stores = {"blob": node.blobs, "dataset": node.store.datasets}

    class Handler(GuardedHandler):
        server_version = "vibehash-node/0.4"

        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self, limit: int) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise WireError("request too large")
            return self.rfile.read(n)

        def _guard(self, fn, raw: bool = False):
            try:
                result = fn()
                self._send(200, result, "application/octet-stream" if raw else "application/json")
            except KeyError as e:
                self._send(404, {"error": f"not found: {e}"})
            except (ProtocolError, WireError, InvalidJob, BlobError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # pragma: no cover
                log.exception("request failed")
                self._send(500, {"error": type(e).__name__})

        def do_GET(self):
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            q = parse_qs(url.query)

            if len(parts) in (2, 3) and parts[0] in stores:
                store, bid = stores[parts[0]], parts[1]
                if len(parts) == 3 and parts[2] == "status":
                    self._guard(lambda: {"received": store.received(check_id(bid), q.get("from", [""])[0]),
                                         "complete": store.has(bid), "size": store.size(bid)})
                else:
                    self._guard(lambda: store.read_chunk(check_id(bid), int(q.get("offset", ["0"])[0]), CHUNK), raw=True)
                return
            if len(parts) == 3 and parts[0] == "job" and parts[2] == "model":
                self._guard(lambda: node.job_model(parts[1], max(0, int(q.get("offset", ["0"])[0]))), raw=True)
                return

            def route():
                if parts == ["status"]:
                    return node.status()
                if parts == ["genesis"]:
                    return node.genesis
                if parts == ["blocks"]:
                    lo = int(q.get("from", ["0"])[0])
                    hi = min(int(q.get("to", [str(lo + 99)])[0]), lo + 99, node.chain.tip.height)
                    certs = node.store.certificates(lo, hi)
                    return [b.to_dict() | ({"certificate": certs[b.height]} if b.height in certs else {})
                            for b in node.chain.blocks[lo:hi + 1]]
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
                if len(parts) == 3 and parts[0] == "round" and parts[2] == "record":
                    record = node.store.load_record(int(parts[1]))
                    if record is None:
                        raise KeyError(f"no record for block {parts[1]}")
                    return record
                raise KeyError(url.path)

            self._guard(route)

        def do_PUT(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]

            def route():
                if len(parts) != 2 or parts[0] not in stores:
                    raise KeyError(self.path)
                kind, bid = parts[0], check_id(parts[1])
                offset, total = int(self.headers.get("X-Offset", "-1")), int(self.headers.get("X-Total", "0"))
                if kind == "blob":
                    uploader = node.authorize_blob(bid, total, self.headers)
                else:
                    uploader = node.authorize_dataset(bid, total, self.headers)
                chunk = self._body(CHUNK)
                received = stores[kind].write_chunk(bid, offset, chunk, total, uploader)
                if kind == "dataset" and received == total:
                    node.store.check_uploaded_dataset(bid)
                return {"received": received, "complete": received == total}

            self._guard(route)

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]

            def route():
                body = json.loads(self._body(MAX_JSON) or b"{}")
                if parts == ["tx"]:
                    tx = Tx.from_dict(body)
                    node.submit_tx(tx, body)
                    return {"ok": True, "txid": tx.txid.hex()}
                if parts == ["consensus"]:
                    node.receive(body)
                    return {"ok": True}
                if len(parts) == 1 and parts[0] in ("register", "assignment", "commit", "reveal", "tasks", "verdict"):
                    return node.handle(parts[0], body)
                raise KeyError(self.path)

            self._guard(route)

    return Handler
