"""HTTP client for a Vibe Hash node: a thin API wrapper plus the worker loop.

Model weights and datasets move as chunked, resumable, content-addressed
binary transfers (blobs.py), never inside JSON, so there's no request-size
ceiling on how big a model or dataset can be.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import finetune, hardware, receipts, settings, tls
from .blobs import CHUNK, MAX_DATASET, BlobStore, file_sha256
from .compute import get_backend
from .crypto import KeyPair
from .ledger import Block, Tx, canonical
from .model import params_hash
from .tasks import MAX_CHECKPOINT_BYTES, InvalidJob, TrainingJob, load_dataset
from .wire import blob_openings_in, blob_openings_out, commitment_out, seal, unit_in
from .worker import HardwareProfile, Worker

log = logging.getLogger("vibehash.worker")
RETRIES = 4
BUSY_RETRIES = 6  # for 429 (rate limited) and 503 (busy): the node turned the request away unprocessed
PARALLEL_REQUESTS = 4  # in flight at once for a round's reveals or a unit's uploads: plenty, and gentle on a small node
# The most of a reply we read, so a node can't feed us gigabytes: the biggest JSON a node sends is a round record or
# a batch of blocks, each at most a proposal's size (node.MAX_PROPOSAL), and binary replies are one chunk.
MAX_REPLY = 128 * 2**20
MAX_ERROR = 2**16  # of an error reply


class NodeError(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status  # HTTP status, or None if the node couldn't be reached

    @property
    def retryable(self) -> bool:
        """Connection trouble, a server fault, or the node asking us to slow down (429). Any other 4xx is the node
        saying no: retrying won't help."""
        return self.status is None or self.status >= 500 or self.status == 429


class NodeClient:
    def __init__(self, url: str, timeout: float = 120, *, ca_file: str | None = None, fingerprint: str | None = None):
        """For an https node, `fingerprint` pins its certificate (SHA-256) and `ca_file` names a CA to verify it with.
        Without either, the trust `setup` saved for this address applies, else the system's CAs (tls.py)."""
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._status: dict | None = None
        self._opener = tls.opener_for(self.url, ca_file, fingerprint)

    def _req(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None, limit: int | None = None) -> bytes:
        """The reply's body, if it's at most `limit` bytes (default MAX_REPLY; NodeError if it's more)."""
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers or {})
        limit = MAX_REPLY if limit is None else limit
        try:
            with self._open(req) as resp:
                body = resp.read(limit + 1)
                if len(body) > limit:
                    raise NodeError(f"{method} {path}: the node's reply is larger than {limit:,} bytes", resp.status)
                return body
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read(MAX_ERROR)).get("error", str(e))
            except Exception:
                msg = str(e)
            raise NodeError(f"{method} {path}: {msg}", e.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise NodeError(f"can't reach node at {self.url}: {getattr(e, 'reason', e)}") from None

    def _open(self, req):
        """urlopen, waiting and trying again while the node says it's busy: a 429 (rate limited) or 503 (at its
        connection limit) means it turned the request away before doing anything, so resending is safe."""
        for attempt in range(BUSY_RETRIES + 1):
            try:
                return (self._opener.open if self._opener else urllib.request.urlopen)(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or attempt == BUSY_RETRIES:
                    raise
                wait = _backoff(e.headers.get("Retry-After"), attempt)
                e.close()
                log.debug("%s %s: node busy (%d); retrying in %.1f s", req.get_method(), req.full_url, e.code, wait)
                time.sleep(wait)

    def get(self, path: str):
        return _json(self._req("GET", path), f"GET {path}")

    def post(self, path: str, body: dict):
        return _json(self._req("POST", path, json.dumps(body).encode(), {"Content-Type": "application/json"}), f"POST {path}")

    def status(self) -> dict:
        self._status = self.get("/status")
        return self._status

    @property
    def chain_id(self) -> str:
        return (self._status or self.status())["chain_id"]

    def account(self, addr: str) -> dict:
        return self.get(f"/account/{addr}")

    def min_fee(self) -> int:
        """The smallest fee the node relays (0 for nodes from before fees, which would reject a fee they can't read)."""
        return int((self._status or self.status()).get("min_fee", 0))

    def send_tx(self, keys: KeyPair, kind: str, recipient: str, amount: int, memo: str = "", fee: int | None = None) -> str:
        """`fee` defaults to the node's minimum, so a transaction always clears it."""
        nonce = self.account(keys.address)["nonce"]
        tx = Tx.signed(keys, kind, recipient, amount, nonce, memo, self.chain_id, self.min_fee() if fee is None else fee)
        return self.post("/tx", tx.to_dict())["txid"]

    def signed(self, keys: KeyPair, kind: str, body: dict):
        return self.post(f"/{kind}", seal(keys, kind, body, self.chain_id))

    # --- binary transfers -----------------------------------------------------------

    def upload(self, keys: KeyPair, kind: str, path: Path, blob_id: str) -> None:
        """Send a file in chunks, resuming from whatever the node already has. kind: 'blob' or 'dataset'."""
        total = path.stat().st_size
        for attempt in range(RETRIES):
            try:
                st = self.get(f"/{kind}/{blob_id}/status?from={keys.address}")  # resume our own upload, not another's
                if st["complete"]:
                    return
                offset = st["received"]
                ts = int(time.time())
                sig = keys.sign(canonical([self.chain_id, kind, blob_id, total, ts]))
                auth = {"X-VibeHash-Pubkey": keys.public.hex(), "X-VibeHash-Sig": sig.hex(), "X-VibeHash-Ts": str(ts),
                        "X-Total": str(total), "Content-Type": "application/octet-stream"}
                with open(path, "rb") as f:
                    f.seek(offset)
                    while offset < total:
                        chunk = f.read(CHUNK)
                        res = json.loads(self._req("PUT", f"/{kind}/{blob_id}", chunk, {**auth, "X-Offset": str(offset)}))
                        offset = res["received"]
                return
            except NodeError as e:
                if not e.retryable or attempt == RETRIES - 1:
                    raise
                time.sleep(1 + attempt)

    def download(self, kind: str, blob_id: str, store: BlobStore, size: int | None = None, check=None) -> Path:
        """Fetch a file in chunks into `store`, resuming a partial copy. The store checks the hash, and `check` (if
        given) what's in it, before keeping it. The node's word for the size isn't enough (one that lied could fill
        the disk): it must be `size` if the caller knows it (a checkpoint's is its job's), and at most what a node
        would store otherwise."""
        limit = MAX_DATASET if kind == "dataset" else MAX_CHECKPOINT_BYTES
        for attempt in range(RETRIES):
            try:
                if store.has(blob_id):
                    return store.path(blob_id)
                st = self.get(f"/{kind}/{blob_id}/status")
                if not st["complete"]:
                    raise NodeError(f"node doesn't have {kind} {blob_id[:12]}…")
                total, offset = st["size"], store.received(blob_id)
                if type(total) is not int or not 0 < total <= limit or size not in (None, total):  # (asking again won't help)
                    raise NodeError(f"the node says {kind} {blob_id[:12]}… is {total!r} bytes, which it can't be", 200)
                while offset < total:
                    chunk = self._req("GET", f"/{kind}/{blob_id}?offset={offset}", limit=min(CHUNK, total - offset))
                    if not chunk:
                        raise NodeError("download ended early")
                    offset = store.write_chunk(blob_id, offset, chunk, total, check=check)
                return store.path(blob_id)
            except NodeError as e:
                if not e.retryable or attempt == RETRIES - 1:
                    raise
                time.sleep(1 + attempt)
        raise NodeError("download failed")

    def download_model(self, job_id: str, job: TrainingJob | None = None, untrained: bool = True) -> tuple[dict, bytes]:
        """A job's latest trained weights, checked against the hash the node reports. Returns (job info, raw bytes).
        Their size is the job's: `job`'s, if the caller has it, else that of the spec the node reports. Nodes don't
        serve an untrained job's weights, since they follow from its spec: they're built here instead (or, with
        `untrained=False`, NodeError). Nodes from before 0.5.2 serve them, and they download as usual."""
        for _ in range(RETRIES):
            info = self.get(f"/job/{job_id}")
            try:
                job = job or TrainingJob.from_spec_json(json.dumps(info["spec"]), str(info["owner"]), 0)
                want, size = info["model"], job.checkpoint_bytes
                if want.get("sha256") is None:
                    if not untrained:
                        raise NodeError(f"job {job_id} hasn't been trained yet")
                    return info, np.asarray(job.spec.init_params(job.init_seed), dtype=job.wire_dtype).tobytes()
                if want["bytes"] != size:
                    raise NodeError(f"the node says job {job_id}'s model is {want['bytes']!r} bytes; its spec makes it {size}")
            except (InvalidJob, KeyError, TypeError, AttributeError) as e:
                raise NodeError(f"the node's description of job {job_id} is malformed: {e}") from None
            data = bytearray()
            while len(data) < size:
                chunk = self._req("GET", f"/job/{job_id}/model?offset={len(data)}", limit=min(CHUNK, size - len(data)))
                if not chunk:
                    raise NodeError("the model download ended early")
                data += chunk
            if hashlib.sha256(data).hexdigest() == want["sha256"]:
                return info, bytes(data)
            # a round finished mid-download and the model moved on: fetch the new one
        raise NodeError("the model kept changing during the download; try again")

    def upload_dataset(self, keys: KeyPair, path: str | Path) -> str:
        path = Path(path)
        load_dataset(path)  # catch a bad file here rather than after uploading gigabytes
        sha = file_sha256(path)
        self.upload(keys, "dataset", path, sha)
        return sha


def _json(raw: bytes, what: str):
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as e:  # (nested thousands deep)
        raise NodeError(f"{what}: the node's reply isn't JSON: {type(e).__name__}") from None


def _backoff(retry_after: str | None, attempt: int) -> float:
    """Exponential from 0.25 s, at least what the node asked for, at most 30 s, with jitter so clients that were
    turned away together don't all come back together."""
    try:
        asked = float(retry_after or 0)
    except ValueError:
        asked = 0.0
    return min(30.0, max(asked, 0.25 * 2**attempt)) * random.uniform(1.0, 1.25)


def default_cache_dir() -> Path:
    return settings.home() / "cache"


@dataclass
class WorkerStats:
    """What a worker is doing right now, for a live display. The worker loop writes it; a UI only reads."""
    round: int | None = None
    phase: str = "idle"
    phase_ends: float = 0.0
    height: int = 0
    job: TrainingJob | None = None
    flops_per_sec: float = 0.0
    activity: str = "connecting to the node"
    busy: bool = False  # training or checking right now
    assigned: int = 0  # this round
    trained: int = 0
    committed: int = 0
    checked: int = 0
    rounds: int = 0  # this session
    units: int = 0
    checks: int = 0
    training_flops: int = 0

    def new_round(self, rnd: int) -> None:
        if rnd != self.round:
            self.round, self.assigned, self.trained, self.committed, self.checked = rnd, 0, 0, 0, 0


class WorkerClient:
    """Polls the node and plays its part in each phase. One instance per wallet.

    On a chain with validators, each round runs at its leader (whichever validator's turn it is to propose), so the
    worker asks its own node who that is and does the round there. `node` is the leader; `home` is its own node."""

    def __init__(self, node_url: str, keys: KeyPair, memory_mb: int | None = None, flops_override: float | None = None,
                 poll_s: float = 0.5, behavior: str = "honest", verifier_behavior: str = "honest", device: str = "auto",
                 cache_dir: str | Path | None = None, deepseek: bool = False):
        """`deepseek`: take DeepSeek jobs' work, whose model files (~30 GB) it fetches into the model store
        (modelstore.py) without asking: the owner said yes (`worker --deepseek yes`, or the menu's item)."""
        self.home = self.node = NodeClient(node_url)
        self.deepseek = deepseek
        from .deepseek_client import DeepSeekWork
        self.ds = DeepSeekWork(self)  # a DeepSeek job's rounds (deepseek_client.py)
        self.backend = get_backend(device, log)
        self.memory_override = memory_mb
        self.flops_override = flops_override
        self.poll_s = poll_s
        cache = Path(cache_dir) if cache_dir else default_cache_dir() / keys.address
        self.blobs = BlobStore(cache / "blobs")  # checkpoints: this round only
        self.datasets = BlobStore(cache / "datasets")  # kept between rounds
        self.blobs.prune()
        self.worker = Worker("me", keys, HardwareProfile("local", 0, 0), behavior, verifier_behavior,
                             fp_noise=0, backend=self.backend, blobs=self.blobs)
        self.jobs: dict[str, TrainingJob] = {}
        self.speed: dict[str, float] = {}
        self.cannot: dict[str, str] = {}  # jobs this worker can't run, and why
        self.done: dict[tuple[int, str], bool] = {}
        self.units: dict[int, list] = {}
        self.start_blob: dict[int, str] = {}
        self.uploaded: dict[int, set[str]] = {}  # checkpoints the node confirmed it has, by round
        self.attempt_of: dict[int, str | None] = {}  # the attempt at each round those confirmations were for
        self.moved: set[tuple] = set()  # (round, dispute, move number) the node took our move for
        self.unsent: dict[tuple, dict] = {}  # moves worked out whose sending failed: sent again, not worked out again
        self.party: dict = {}  # this round's tracer.Party, for its disputes
        self.genesis_hash: str | None = None
        self.stats = WorkerStats()
        self.receipt_file = cache / "receipts.jsonl"  # every receipt, kept as evidence (see receipts.py)
        self.unaudited: list[dict] = []  # receipts for rounds not yet sealed
        self._receipts_lock = threading.Lock()  # reveals are sent from several threads at once
        self.censored: list[str] = []  # what the audit found: rounds the node sealed without our messages
        # The hash of each round's secret as /status showed it when the round opened, by (round, attempt). The record
        # reveals the secret when the round is sealed; a leader that swapped it (and so picked the beacon after seeing
        # the commitments) would have to change the commitment in the record too, and this is where that shows.
        self.secret_commits: dict[tuple[int, str], str] = {}

    @property
    def keys(self) -> KeyPair:
        return self.worker.keys

    def _signed(self, kind: str, body: dict, entry: dict | None = None) -> dict:
        """Send a signed message, and keep the node's receipt for it once it checks out. `entry` is the message as
        the round record holds it, when that isn't the envelope itself."""
        env = seal(self.keys, kind, body, self.node.chain_id)
        res = self.node.post(f"/{kind}", env)
        receipt = res.get("receipt")
        if receipt is not None:  # older nodes don't give receipts
            vouches = (receipts.valid(receipt, self.node.chain_id) and receipt["kind"] == kind and receipt["round"] == body["round"]
                       and receipt["digest"] == receipts.entry_digest(entry if entry is not None else env))
            if not vouches:
                log.warning("the node's receipt for our %s doesn't match what we sent, so it proves nothing", kind)
            else:
                with self._receipts_lock:
                    self.unaudited.append(receipt)
                    with open(self.receipt_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(receipt) + "\n")
        return res

    def audit_receipts(self, height: int) -> None:
        """Check every sealed round we have receipts for: the node must have recorded what it signed for."""
        due = sorted({r["round"] for r in self.unaudited if r["round"] <= height})
        for rnd in due:
            mine = [r for r in self.unaudited if r["round"] == rnd]
            self.unaudited = [r for r in self.unaudited if r["round"] != rnd]
            try:
                block = Block.from_dict(self.node.get(f"/blocks?from={rnd}&to={rnd}")[0])
            except (NodeError, IndexError):
                continue
            try:
                record = self.node.get(f"/round/{rnd}/record")
            except NodeError:
                record = None
            seen = self.secret_commits.pop((rnd, record.get("attempt")), None) if isinstance(record, dict) else None
            if seen and record.get("beacon_commit") != seen:
                detail = (f"round {rnd}: the leader's round secret isn't the one it committed to when the round "
                          "opened, so it could have picked the beacon")
                self.censored.append(detail)
                log.error("censored: %s", detail)
            for r in mine:
                status, detail = receipts.audit(r, block, record)
                if status == "censored":
                    self.censored.append(detail)
                    log.error("censored: %s (evidence in %s)", detail, self.receipt_file)
                elif status not in ("recorded", "pending"):
                    log.warning("%s", detail)

    def runs_on_gpu(self, job: TrainingJob) -> bool:
        return self.backend.name != "cpu" and job.spec.precision == "float32"

    def can_run(self, job: TrainingJob) -> str | None:
        """Why this worker can't take the job's units, or None if it can."""
        if job.finetune and not finetune.available():
            return f"job {job.job_id} fine-tunes {job.spec.repo}, which needs PyTorch: `vibehash install-gpu`"
        return None

    def memory_for(self, job: TrainingJob) -> int:
        """GPU memory for jobs the GPU runs, system RAM otherwise (float64 jobs always run on the CPU)."""
        if self.memory_override:
            return self.memory_override
        if self.runs_on_gpu(job):
            return self.backend.memory_mb() or hardware.total_memory_mb()
        return hardware.total_memory_mb()

    def job(self, job_id: str) -> TrainingJob:
        if job_id not in self.jobs:
            info = self.node.get(f"/job/{job_id}")
            job = TrainingJob.from_spec_json(json.dumps(info["spec"]), info["owner"], 0)
            why = self.ds.why_not(job) if job.deepseek else self.can_run(job)
            if why:
                self.jobs[job_id] = job
                self.cannot[job_id] = why
                log.warning("skipping it: %s", why)
                return job
            if job.data_source["kind"] in ("npz", "tokens", "deepseek"):
                sha = job.data_source["sha256"]
                self.datasets.prune_partial(keep={sha})  # other datasets' downloads that never finished
                path = self.node.download("dataset", sha, self.datasets)  # hash-checked on arrival
                job.attach_data(load_dataset(path), sha)
            if job.deepseek:  # its model files come unit by unit, through the model store (deepseek_client.py)
                self.jobs[job_id] = job
                self.speed[job_id] = self.flops_override or (5e12 if self.runs_on_gpu(job) else 1e10)
                log.info("job %s is a DeepSeek job: taking its work on %s", job_id, self.ds.device())
                return job
            if job.finetune:  # the base model, fetched once and kept on the GPU (finetune.runtime)
                self.stats.activity = f"fetching {job.spec.repo} to fine-tune it"
                log.info("job %s fine-tunes %s: loading it on %s", job_id, job.spec.repo, finetune.device_of(self.backend))
                def fetching(done: int, total: int, what: str) -> None:
                    self.stats.activity = f"{what}: {done / 1e9:.2f} / {total / 1e9:.2f} GB"
                finetune.runtime(job.spec, job.data, finetune.device_of(self.backend), progress=fetching)
            self.jobs[job_id] = job
            self.speed[job_id] = self.flops_override or hardware.benchmark(job, 0.5, self.backend)
            where = self.backend.describe() if self.runs_on_gpu(job) else "CPU"
            log.info("job %s (%s, %d weights): %.2f GFLOP/s on %s, %d MB", job_id, job.spec.precision, job.spec.n_params,
                     self.speed[job_id] / 1e9, where, self.memory_for(job))
        return self.jobs[job_id]

    def _upload(self, rnd: int, job: TrainingJob, arrays) -> dict[int, str]:
        """Upload checkpoints the node doesn't have yet. Returns {id(array): blob id}."""
        ids, todo = {}, []
        sent = self.uploaded.setdefault(rnd, set())
        for arr in arrays:
            if id(arr) not in ids:
                bid = self.blobs.put_array(arr, job.wire_dtype)  # already on disk for spilled checkpoints; this names it
                # The node made the start blob itself, and what went up this round needn't be asked about again: each
                # question is a round trip, and a reveal's six or so add up past the reveal phase on a distant link.
                if bid != self.start_blob.get(rnd) and bid not in sent and bid not in todo:
                    todo.append(bid)
                ids[id(arr)] = bid
        if todo:  # together: one after another, a unit's nine or so cost a CPU worker its last unit's commit deadline
            with ThreadPoolExecutor(min(PARALLEL_REQUESTS, len(todo))) as pool:
                for bid in pool.map(lambda b: (self.node.upload(self.keys, "blob", self.blobs.path(b), b), b)[1], todo):
                    sent.add(bid)
        return ids

    def _send_reveal(self, rnd: int, job: TrainingJob, rv) -> dict | None:
        """Checkpoints went up during compute, so this is one small message (plus any upload that failed)."""
        ids = self._upload(rnd, job, [rv.final_params, *(a for a, _ in rv.openings.values())])
        body = {"unit_id": rv.unit_id, "worker": rv.worker, "merkle_root": rv.merkle_root.hex(), "salt": rv.salt.hex(),
                "final": ids[id(rv.final_params)],
                "openings": blob_openings_out({i: (ids[id(a)], pr) for i, (a, pr) in rv.openings.items()})}
        # A reveal lost on the way forfeits the round, so a dropped connection is worth another try. Sending it twice is
        # harmless: if the first got through and only its reply was lost, the node says so.
        for attempt in range(RETRIES):
            try:
                return self._signed("reveal", {"round": rnd, "reveal": body}, entry=body)
            except NodeError as e:
                if "already revealed" in str(e):
                    return None
                if not e.retryable or attempt == RETRIES - 1:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def _follow_leader(self) -> dict:
        """Status from wherever this round runs: our own node, or the validator leading the round."""
        s = self.home.status()
        lead = (s.get("leader") or self.home.url).rstrip("/")
        if lead != self.node.url:
            self.node = self.home if lead == self.home.url else NodeClient(lead)
            log.debug("round leader is now %s", lead)
        if self.node is not self.home:
            s = self.node.status()
        return s

    def step(self) -> None:
        s = self._follow_leader()
        if self.genesis_hash is None:
            self.genesis_hash = s["genesis_hash"]
        elif s["genesis_hash"] != self.genesis_hash:
            raise NodeError("node switched networks (genesis changed); refusing to continue")
        if self.unaudited and min(r["round"] for r in self.unaudited) <= s["height"]:
            self.audit_receipts(s["height"])
        rnd, phase = s["round"], s["phase"]
        st = self.stats
        st.phase, st.phase_ends, st.height = phase, s["phase_ends"], s["height"]
        if rnd is None:
            st.activity = "waiting for someone to post a training job"
        key = (rnd, s.get("attempt"), phase)  # a round run again (by the next leader, say) is a new attempt to join
        if rnd is not None and self.attempt_of.get(rnd, s.get("attempt")) != s.get("attempt"):
            # Another leader, or the same node after a restart, has none of what went up for the abandoned attempt:
            # without this the reveal names checkpoints the node doesn't have, and the units count as withheld.
            self.uploaded.pop(rnd, None)
        if rnd is not None:
            self.attempt_of[rnd] = s.get("attempt")
            for old in [k for k in self.attempt_of if k < rnd - 100]:
                del self.attempt_of[old]
        if rnd is not None and s.get("beacon_commit"):
            self.secret_commits.setdefault((rnd, s.get("attempt")), s["beacon_commit"])
            for old in [k for k in self.secret_commits if k[0] < rnd - 100]:  # rounds we never sent anything in
                del self.secret_commits[old]
        if rnd is None or key in self.done:
            return
        if st.round != rnd:
            st.activity = "joining at the next round"
        st.new_round(rnd)
        job = self.job(s["job_id"])
        if job.job_id in self.cannot:
            st.activity = self.cannot[job.job_id]
            self.done[key] = True
            return
        st.job, st.flops_per_sec = job, self.speed[job.job_id]
        k = self.keys

        if phase == "register":
            self._signed("register", {"round": rnd, "flops_per_sec": self.speed[job.job_id], "memory_mb": self.memory_for(job)})
            st.rounds += 1
            st.activity = "registered; waiting for work"
            log.info("round %d: registered", rnd)
        elif phase == "compute":
            given = self.node.signed(k, "assignment", {"round": rnd})
            units = [unit_in(u) for u in given["units"]]
            st.assigned = len(units)
            committed = []
            if not units:
                why = (s.get("waiting") or {}).get(job.job_id)
                st.activity = why or ("no units for us this round" if st.rounds else "joining at the next round")
            if units and given.get("prep"):  # a DeepSeek prep round: units of the work graph, through the bucket
                st.busy = True
                try:
                    for c in self.ds.compute(rnd, job, units, given["prep"], given.get("want") or {}):
                        if self._try(self._signed, "commit", {"round": rnd, "commitment": commitment_out(c)}, commitment_out(c)):
                            st.committed += 1
                            st.units += 1
                finally:
                    st.busy = False
                st.activity = "prep units committed; waiting for the reveal"
                units = []
            elif units and job.deepseek:  # a DeepSeek training round: its records, once
                self.ds.records(rnd, job)
            if units:
                p = self.node.get("/round/params")
                self.node.download("blob", p["blob"], self.blobs, size=job.checkpoint_bytes)
                start = self.blobs.load(p["blob"], job.wire_dtype, job.spec.n_params)
                if params_hash(start) != units[0].start_hash:
                    raise NodeError("node served weights that don't match the unit's start hash")
                self.start_blob[rnd] = p["blob"]
                st.busy = True
                try:
                    for i, u in enumerate(units, 1):
                        st.activity = f"training unit {i} of {len(units)}"
                        c = self.worker.compute(u, job, start)
                        st.trained += 1
                        # Upload before committing: the reveal phase then only carries a small message, and uploads
                        # count toward the compute deadline, so a link too slow to deliver in time gets less work next round.
                        # Safe to send early: blob ids are content hashes nobody else can guess, and the commitment
                        # already binds the result before the beacon exists.
                        work = self.worker._work[u.unit_id]
                        st.activity = f"uploading unit {i} of {len(units)}"
                        if self._try(self._upload, rnd, job, [*work.checkpoints[1:], work.final]) is not None:
                            if self._try(self._signed, "commit", {"round": rnd, "commitment": commitment_out(c)}, commitment_out(c)):
                                committed.append(u)
                                st.committed += 1
                                st.units += 1
                                st.training_flops += u.flops
                finally:
                    st.busy = False
                st.activity = f"committed {len(committed)} units; waiting for the reveal"
                log.info("round %d: computed %d units, committed %d", rnd, len(units), len(committed))
            self.units[rnd] = committed  # only committed units get revealed
        elif phase == "reveal" and s["beacon"]:
            for body in self.ds.reveals(rnd):  # a DeepSeek prep round's: each unit's commitment root
                res = self._try(self._signed, "reveal", {"round": rnd, "reveal": body}, body)
                if res and not res["ok"]:
                    log.warning("round %d: prep unit %s rejected: %s", rnd, body["unit_id"], res["verdict"])
            beacon = bytes.fromhex(s["beacon"])
            if self.units.get(rnd):
                st.activity = "revealing the checkpoints the beacon picked"
            reveals = [rv for u in self.units.get(rnd, []) if (rv := self.worker.reveal(u, job, beacon, s["challenge_segments"]))]
            # All at once: a unit not revealed in time forfeits the whole round, and one at a time, a worker far from the
            # node with many units spends the phase waiting on round trips.
            with ThreadPoolExecutor(max(1, min(PARALLEL_REQUESTS, len(reveals)))) as pool:
                for rv, res in zip(reveals, pool.map(lambda rv: self._try(self._send_reveal, rnd, job, rv), reveals)):
                    if res and not res["ok"]:
                        log.warning("round %d: unit %s rejected: %s", rnd, rv.unit_id, res["verdict"])
        elif phase == "dispute":  # a fine-tune's disputes: a move in each that waits on us, every poll until it ends
            self._disputes(rnd, job)
            return
        elif phase == "verify":
            absent = self.worker.verifier_behavior == "absent"  # tests only: a checker that never answers
            tasks = [] if absent else self.node.signed(k, "tasks", {"round": rnd})["tasks"]
            self._answer(rnd, job, tasks)
            st.activity = "round done; waiting for the block"
            if tasks:
                log.info("round %d: verified %d tasks", rnd, len(tasks))
            for u in self.units.pop(rnd, []):
                self.worker.forget(u.unit_id)
            self.start_blob.pop(rnd, None)
            self.uploaded.pop(rnd, None)
            self.blobs.prune()
        else:
            return
        self.done[key] = True

    def _answer(self, rnd: int, job: TrainingJob, tasks: list[dict]) -> None:
        """Do this phase's tasks and send a verdict for each: replay segments, or score a fine-tune's starting model
        on its held-back data."""
        st = self.stats
        st.busy = bool(tasks)
        try:
            for i, t in enumerate(tasks, 1):
                if st.phase_ends and time.time() > st.phase_ends:
                    # The node refuses verdicts now, and a fine-tune's disputes may be waiting on this worker's move.
                    log.warning("round %d: the verify phase ended with %d tasks left unchecked", rnd, len(tasks) - i + 1)
                    break
                kind = t.get("kind", "check")
                body = {"round": rnd, "task_id": t["task_id"], "ok": True, "bad_segment": None}
                if kind == "prep":  # a DeepSeek prep unit: replayed, and its commitment held up against the replay
                    st.activity = f"checking another miner's prep unit ({i} of {len(tasks)})"
                    body["ok"] = bool(self.ds.check(rnd, job, t))
                    self._try(self._signed, "verdict", body)
                    st.checked += 1
                    st.checks += 1
                    continue
                if job.deepseek:  # a DeepSeek training round's checks and scores train on its records
                    self.ds.records(rnd, job)
                if kind == "score":
                    st.activity = "scoring the round's starting model on the held-back data"
                    body["score"] = list(self.worker.score(job, self._round_start(rnd, job)))
                else:
                    st.activity = f"checking other workers' results ({i} of {len(tasks)})"
                    openings = {}
                    for j, (bid, proof) in blob_openings_in(t["openings"], job.n_segments).items():
                        self.node.download("blob", bid, self.blobs, size=job.checkpoint_bytes)
                        openings[j] = (self.blobs.load(bid, job.wire_dtype, job.spec.n_params), proof)
                    ok, bad = self.worker.verify(job, unit_in(t["unit"]), openings, t["segments"])
                    body |= {"ok": bool(ok), "bad_segment": bad}
                self._try(self._signed, "verdict", body)
                st.checked += 1
                st.checks += 1
        finally:
            st.busy = False

    def _disputes(self, rnd: int, job: TrainingJob) -> None:
        """Make our move in each of this round's disputes that waits on us (dispute.py): as the defender, post the
        states between our training's ops, or the evidence for one op; as the challenger, name where they go wrong."""
        st, me = self.stats, self.keys.address
        games = [g for g in self.node.get("/round/games") if g["turn"] != "done"
                 and (g["defender"] if g["turn"] in ("split", "evidence") else g["challenger"]) == me
                 and (rnd, g["id"], g["n"]) not in self.moved]
        if not games:
            st.activity = "waiting on this round's disputes"
            return
        from .program import Program
        from .tracer import Party

        party = self.party.get(rnd)
        if party is None:
            from . import finetune
            from .worker import tamper

            cheat = tamper if self.worker.behavior == "tamper" else None
            if job.deepseek:  # its program and weights (deepseek_tracer), on its records
                from .deepseek_tracer import Party as DeepSeekParty
                from .deepseek_unit import runtime as deepseek_runtime
                self.ds.records(rnd, job)
                rt = deepseek_runtime(job.spec, finetune.device_of(self.backend))
                party = DeepSeekParty(rt, job.spec, job.data, _Remote(self), cheat)
            else:
                rt = finetune.runtime(job.spec, job.data, finetune.device_of(self.backend))
                party = Party(rt, job.spec, job.data, _Remote(self), cheat)
            self.party = {rnd: party}
        st.busy = True
        try:
            for g in games:
                if g["turn"] == "pick" and self.worker.verifier_behavior == "quitter":
                    continue  # tests only: a challenger that goes quiet
                st.activity = f"dispute {g['id'][:8]}: {g['turn']} ({'defending' if g['defender'] == me else 'challenging'})"
                key = (rnd, g["id"], g["n"])
                move = self.unsent.pop(key, None)
                if move is None:
                    prog = Program.of(job, int(g["seed"]))
                    move = (party.pick(prog, g) if g["turn"] == "pick" else {"split": party.split(prog, g)}
                            if g["turn"] == "split" else {"evidence": party.remote.put(party.evidence(prog, g))})
                # Made only once the node takes it: a send that fails is tried again next time round, until the game's
                # move count passes it (taken after all, or too late).
                if self._try(self._signed, "move", {"round": rnd, "game": g["id"], "n": g["n"], "move": move}) is None:
                    self.unsent = {k: v for k, v in self.unsent.items() if k[0] == rnd} | {key: move}
                else:
                    self.moved.add(key)
        finally:
            st.busy = False

    def _round_start(self, rnd: int, job: TrainingJob) -> np.ndarray:
        """The weights this round started from, fetched if it trained nothing this round."""
        if rnd not in self.start_blob:
            p = self.node.get("/round/params")
            self.node.download("blob", p["blob"], self.blobs, size=job.checkpoint_bytes)
            self.start_blob[rnd] = p["blob"]
        return self.blobs.load(self.start_blob[rnd], job.wire_dtype, job.spec.n_params)

    @staticmethod
    def _try(fn, *args):
        """One failed message (e.g. a phase that just closed) mustn't stop the rest being sent."""
        try:
            return fn(*args)
        except NodeError as e:
            log.warning("%s", e)
            return None

    def run(self, stop=None) -> None:
        log.info("worker %s starting on %s (%s)", self.keys.address, self.backend.describe(), hardware.describe())
        while stop is None or not stop.is_set():
            try:
                self.step()
            except NodeError as e:
                log.warning("%s", e)
                time.sleep(2)
            except Exception:  # one bad round (corrupt download, full disk...) mustn't stop the worker for good
                log.exception("unexpected error; skipping to the next phase")
                time.sleep(2)
            time.sleep(self.poll_s)


class _Remote:
    """A dispute's states and evidence, through the node's blob endpoints (tracer.Party's `remote`)."""

    def __init__(self, client: WorkerClient):
        self.client = client

    def put(self, raw: bytes) -> str:
        c = self.client
        bid = c.blobs.put_array(np.frombuffer(raw, dtype=np.uint8), "uint8")
        c.node.upload(c.keys, "blob", c.blobs.path(bid), bid)
        return bid

    def get(self, bid: str) -> bytes:
        c = self.client
        if not c.blobs.has(bid):
            c.node.download("blob", bid, c.blobs)
        return c.blobs.path(bid).read_bytes()
