"""A worker's part in a DeepSeek job's rounds (deepseek_rounds.py), as client.WorkerClient plays it: whether it takes
the job; in a prep round, running its prep units through the bucket (links the node signs) and committing to each
one's commitment, then revealing them, and checking other miners' by replaying them; in a training round, the records
its units train on, read from the bucket and checked against the job's pinned root.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from . import deepseek_graph as dsg
from . import modelstore
from .tasks import Commitment, TrainingJob, WorkUnit, commit_digest

log = logging.getLogger("vibehash.worker")


class DeepSeekWork:
    def __init__(self, client):
        self.client = client
        self.weights: dict = {}  # job id -> deepseek_worker.Weights
        self.committed: dict[int, list[tuple[WorkUnit, bytes, bytes]]] = {}  # round -> (unit, root, salt)
        self.round: int | None = None
        self.bend = None  # tests only: (store, unit, commitment) -> a cheat's commitment, its outputs bent to match

    # --- taking the job ---

    def why_not(self, job: TrainingJob) -> str | None:
        """Why this worker won't take the DeepSeek job's work, or None."""
        from .finetune import available
        if not available():
            return f"job {job.job_id} trains DeepSeek-V4.1, which needs PyTorch: `vibehash install-gpu`"
        if not self.client.deepseek:
            return (f"job {job.job_id} is a DeepSeek job: to mine it (about 30 GB of model files, deleted when it ends), "
                    "run `vibehash worker --deepseek yes`, or pick \"Mine the DeepSeek job\" in the menu. This PC keeps "
                    "earning on other jobs")
        if job.spec.config_name != "tiny":
            from .deepseek_worker import can_take
            ok, why = can_take("prep")
            if not ok:
                return why
        return None

    def weights_for(self, job: TrainingJob):
        from .deepseek_worker import Weights
        if job.job_id not in self.weights:
            store = None if job.spec.config_name == "tiny" else modelstore.Store.from_settings()
            self.weights[job.job_id] = Weights(job.spec, job.job_id, store, yes=True, say=log.info)
        return self.weights[job.job_id]

    def work(self, job: TrainingJob) -> Path:
        return self.client.blobs.root.parent / "deepseek" / job.job_id

    def device(self) -> str:
        from .finetune import device_of
        return device_of(self.client.backend)

    # --- the bucket, through links the node signs ---

    def store(self) -> dsg.PresignedStore:
        def sign(method, name, query=None):
            res = self.client.node.signed(self.client.keys, "sign",
                                          {"round": self.round, "requests": [[method, name, query or {}]]})
            return res["urls"][0]
        return dsg.PresignedStore(sign, part=dsg.PresignedStore.PART)

    # --- a prep round ---

    def compute(self, rnd: int, job: TrainingJob, units: list[WorkUnit], prep: dict, want: dict) -> list[Commitment]:
        """Run this round's prep units, each's outputs and commitment to the bucket. -> a signed Commitment each, to
        the sha256 of its commitment."""
        from .deepseek_worker import run_unit
        self.round = rnd
        graph = job.spec.graph(job.data.tokens)
        weights, store, out = self.weights_for(job), self.store(), []
        for u in units:
            gu = graph.units[prep[u.unit_id]]
            if not self._fetch(job, weights, gu):
                continue
            self.client.stats.activity = f"prep: {gu.name}"
            commit = run_unit(job.spec, job.data.tokens, gu, store, weights, self.work(job), want=want,
                              device=self.device())
            if self.bend is not None:
                commit = self.bend(store, gu, commit)
            root, salt = hashlib.sha256(json.dumps(commit).encode()).digest(), os.urandom(16)
            k = self.client.keys
            digest = commit_digest(u.unit_id, k.public, root, salt)
            unsigned = Commitment(u.unit_id, k.address, k.public, digest, b"")
            out.append(Commitment(u.unit_id, k.address, k.public, digest, k.sign(unsigned.message())))
            self.committed.setdefault(rnd, []).append((u, root, salt))
        return out

    def _fetch(self, job: TrainingJob, weights, gu) -> bool:
        """The model files a unit needs, through the model store (asked first: the owner said yes to DeepSeek work)."""
        if weights.tiny:
            return True
        spec = job.spec
        if gu.kind == "rows":
            files = weights.files([], tokenizer=True)
        else:
            layers = range(gu.a, gu.b)
            files = weights.files(layers, embed=gu.a == 0,
                                  tokenizer=any(i in spec.cfg.engram_layer_ids for i in layers))
        return weights.get(files, f"DeepSeek-V4.1-Flash's layers for {gu.name}")

    def reveals(self, rnd: int) -> list[dict]:
        """Each committed unit's reveal: its commitment's root, and the salt."""
        k = self.client.keys
        return [{"unit_id": u.unit_id, "worker": k.address, "merkle_root": root.hex(), "salt": salt.hex(), "prep": True}
                for u, root, salt in self.committed.pop(rnd, [])]

    def check(self, rnd: int, job: TrainingJob, task: dict) -> bool:
        """A prep task: replay the unit and say whether the commitment holds (that of its claim, or a canary's)."""
        from .deepseek_split import Missing, Tampered
        from .deepseek_worker import check_rows, check_unit
        self.round = rnd
        graph = job.spec.graph(job.data.tokens)
        gu, store, weights = graph.units[task["prep"]], self.store(), self.weights_for(job)
        try:
            raw = store.get(task["commit"])
        except OSError:
            return False
        if hashlib.sha256(raw).hexdigest() != task["root"]:
            return False
        commit = json.loads(raw)
        want = {}
        try:
            if not self._fetch(job, weights, gu):
                raise RuntimeError("no room for the unit's files")
            if gu.kind == "rows":
                return check_rows(job.spec, job.data.tokens, gu, commit, weights, self.work(job) / "check")
            ok, why = check_unit(job.spec, job.data.tokens, gu, commit, store, weights, self.work(job) / "check",
                                 want=want, device=self.device())
            log.info("checked %s: %s", gu.name, why)
            return ok
        except (Tampered, Missing, KeyError) as e:
            log.info("checked %s: rejected (%s)", gu.name, e)
            return False

    # --- a training round ---

    def records(self, rnd: int, job: TrainingJob) -> None:
        """Point the job's data at its records in the bucket, checked against the root the job pinned."""
        from .deepseek_unit import Records, runtime
        self.round = rnd
        if job.data.records is not None:
            return
        info = self.client.node.get(f"/job/{job.job_id}")
        root = (info.get("prep") or {}).get("root")
        if not root:
            raise RuntimeError(f"job {job.job_id}'s prep isn't done: no records yet")
        store = self.store()
        manifest = json.loads(store.get("manifest.json"))
        if manifest.get("root") != root:
            raise RuntimeError("the bucket's manifest isn't the dataset the job pinned")
        job.data.records = Records(store, manifest, root, self.work(job) / "records", job.spec.top)
        weights = self.weights_for(job)
        if not weights.tiny:
            if not weights.get(weights.files(job.spec.top.trained, head=True), "DeepSeek-V4.1-Flash's top layers"):
                raise RuntimeError("no room for the top layers")
            runtime(job.spec, self.device(), weights.trained())
