"""On-disk node state: blocks and model weights in SQLite; datasets and checkpoints as
content-addressed files (see blobs.py).

On start-up the chain is rebuilt by re-validating every stored block from
genesis, so a corrupted or edited database is refused rather than trusted. On a
chain run by validators, each block must also carry a valid commit certificate.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .blobs import BlobStore, file_sha256
from .consensus import check_certificate
from .ledger import Block, Chain, canonical
from .model import Dataset
from .tasks import InvalidJob, TrainingJob


class Store:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.datasets = BlobStore(self.dir / "datasets")
        self.blobs = BlobStore(self.dir / "blobs")  # checkpoints: this round's, plus the challenge window's evidence
        self.db = sqlite3.connect(self.dir / "chain.sqlite3", check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS blocks (height INTEGER PRIMARY KEY, hash TEXT NOT NULL, body TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS params (job_id TEXT PRIMARY KEY, height INTEGER NOT NULL, blob BLOB NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS job_rounds (job_id TEXT NOT NULL, height INTEGER NOT NULL, "
                            "accepted INTEGER NOT NULL, val_loss REAL NOT NULL, val_acc REAL NOT NULL, PRIMARY KEY (job_id, height))")
            self.db.execute("CREATE TABLE IF NOT EXISTS round_records (height INTEGER PRIMARY KEY, record TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS certificates (height INTEGER PRIMARY KEY, cert TEXT NOT NULL)")

    def close(self) -> None:
        self.db.close()

    # --- chain ---------------------------------------------------------------------

    def open_chain(self, genesis: dict) -> Chain:
        """Rebuild the chain from disk, validating every block. Refuses a different genesis."""
        chain = Chain.from_genesis(genesis)
        with self.lock, self.db:
            row = self.db.execute("SELECT value FROM meta WHERE key='genesis_hash'").fetchone()
            if row is None:
                self.db.execute("INSERT INTO meta VALUES ('genesis_hash', ?)", (chain.genesis_hash,))
                self.db.execute("INSERT INTO meta VALUES ('genesis', ?)", (canonical(genesis).decode(),))
            elif row[0] != chain.genesis_hash:
                raise ValueError("this data directory belongs to a different network (genesis mismatch)")
            rows = self.db.execute("SELECT b.body, c.cert FROM blocks b LEFT JOIN certificates c ON c.height = b.height "
                                   "ORDER BY b.height").fetchall()
        for body, cert in rows:
            block = Block.from_dict(json.loads(body))
            if chain.state.validators:
                problem = check_certificate(json.loads(cert) if cert else None, block.height, block.hash.hex(),
                                            chain.state.validator_set(), chain.chain_id)
                if problem:
                    raise ValueError(f"block {block.height} in this data directory isn't final ({problem}); refusing it")
            chain.append(block)
        return chain

    def save_block(self, block: Block, certificate: dict | None = None) -> None:
        """`certificate`: the validators' signatures that made the block final (consensus.py), on chains that have them."""
        with self.lock, self.db:
            self.db.execute("INSERT INTO blocks VALUES (?, ?, ?)", (block.height, block.hash.hex(), json.dumps(block.to_dict())))
            if certificate is not None:
                self.db.execute("INSERT OR REPLACE INTO certificates VALUES (?, ?)", (block.height, canonical(certificate).decode()))

    def save_meta(self, key: str, value) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def load_meta(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        try:
            return None if row is None else json.loads(row[0])
        except ValueError:
            return None

    def certificates(self, lo: int, hi: int) -> dict[int, dict]:
        with self.lock:
            rows = self.db.execute("SELECT height, cert FROM certificates WHERE height BETWEEN ? AND ?", (lo, hi)).fetchall()
        return {h: json.loads(c) for h, c in rows}

    # --- model weights -----------------------------------------------------------------

    def save_params(self, job_id: str, height: int, params: np.ndarray) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO params VALUES (?, ?, ?)",
                            (job_id, height, np.ascontiguousarray(params, dtype="<f8").tobytes()))

    def load_params(self, job_id: str) -> np.ndarray | None:
        with self.lock:
            row = self.db.execute("SELECT blob FROM params WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype="<f8").astype(np.float64)

    def params_height(self, job_id: str) -> int:
        """The block whose round last updated this job's model (0: not trained yet)."""
        with self.lock:
            row = self.db.execute("SELECT height FROM params WHERE job_id=?", (job_id,)).fetchone()
        return 0 if row is None else int(row[0])

    def save_record(self, height: int, record: dict) -> None:
        """The round record a block commits to (coordinator.py), kept so anyone can replay the round."""
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO round_records VALUES (?, ?)", (height, canonical(record).decode()))

    def load_record(self, height: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT record FROM round_records WHERE height=?", (height,)).fetchone()
        return None if row is None else json.loads(row[0])

    def evidence(self, tip: int, rounds: int) -> set[str]:
        """Blob ids of the checkpoints the last `rounds` records point to: the opened checkpoints and finals
        anyone re-checking those rounds needs (replay.py). Everything else can go."""
        keep: set[str] = set()
        with self.lock:
            rows = self.db.execute("SELECT record FROM round_records WHERE height > ?", (tip - rounds,)).fetchall()
        for (text,) in rows:
            for rv in json.loads(text).get("reveals", []):
                keep.update(b for b in [rv.get("final"), *(o[0] for o in (rv.get("openings") or {}).values())] if b)
        return keep

    def save_round(self, job_id: str, height: int, accepted: int, val_loss: float, val_acc: float) -> None:
        """How a job's model scored on its validation data after a round: what a job poster follows."""
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO job_rounds VALUES (?, ?, ?, ?, ?)", (job_id, height, accepted, val_loss, val_acc))

    def job_rounds(self, job_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT height, accepted, val_loss, val_acc FROM job_rounds WHERE job_id=? ORDER BY height",
                                   (job_id,)).fetchall()
        return [{"height": h, "accepted_units": a, "val_loss": loss, "val_acc": acc} for h, a, loss, acc in rows]

    # --- datasets ----------------------------------------------------------------------

    def dataset_path(self, sha256: str) -> Path:
        try:
            return self.datasets.path(sha256)
        except ValueError as e:
            raise InvalidJob("bad dataset hash") from e

    def add_dataset(self, src: str | Path) -> str:
        """Copy a local dataset into the store, validating it first. Returns its sha256."""
        src = Path(src)
        Dataset.from_npz(src)  # refuse files that aren't a loadable, pickle-free dataset
        sha = file_sha256(src)
        dest = self.dataset_path(sha)
        if not dest.exists():
            tmp = dest.with_suffix(".tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
        return sha

    def check_dataset(self, path: Path) -> None:
        """Raises InvalidJob unless `path` is a real, pickle-free dataset (with no more data than its size allows, see
        Dataset.from_npz). Uploads and downloads run it on the finished file before it's kept (BlobStore.write_chunk),
        so a dataset in the store never needs deleting, even when a check fails for some other reason."""
        try:
            Dataset.from_npz(path)
        except Exception as e:
            raise InvalidJob(f"not a valid dataset: {e}") from e

    def attach_dataset(self, job: TrainingJob) -> None:
        path = self.dataset_path(job.data_source["sha256"])
        if not path.exists():
            raise InvalidJob(f"dataset {job.data_source['sha256'][:12]}… not available on this node")
        job.attach_data(Dataset.from_npz(path), file_sha256(path))
