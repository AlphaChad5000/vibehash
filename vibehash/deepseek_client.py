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
import time
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
        self.names: dict[int, dict[str, str]] = {}  # round -> {unit id: its graph unit}, ours and those we check
        self._games: dict = {}
        self._prep_weights: dict = {}
        self.wants: dict[int, dict] = {}  # round -> the accepted objects' digests, as the assignment gave them
        # The latest prep round's units' own "stats" (seconds, tokens a second, peak GPU memory, transfers), as their
        # commitments carry them: what the worker's round log (records.RoundLog) keeps. {round: [stats, ...]}
        self.unit_stats: dict[int, list[dict]] = {}
        self._looked = 0.0  # when ended() last asked the node
        self.prep_cheat = None  # tests only: graph unit name -> a tamper (tracer.Executor's) for its run, or None

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

    def sweep(self) -> None:
        """When the worker starts: model files no job has used for a week, and downloads nobody resumed, go."""
        try:
            gone = modelstore.Store.from_settings().sweep()
        except OSError as e:
            log.warning("couldn't tidy the model files: %s", e)
            return
        if gone:
            log.info("deleted %d model file(s) no job used for a week", len(gone))

    def ended(self, current: str | None) -> None:
        """A DeepSeek job this worker took that has ended (its escrow spent): its model files go, unless kept
        (modelstore's job_ended). The node is asked at most once a minute, while rounds are for another job or none."""
        if time.time() - self._looked < 60:
            return
        self._looked = time.time()
        for job_id, w in list(self.weights.items()):
            if job_id == current:
                continue
            try:
                if self.client.node.get(f"/job/{job_id}")["escrow"] > 0:
                    continue
            except Exception as e:  # noqa: BLE001 - unreachable, or gone: the week's sweep has it
                log.debug("job %s: %s", job_id, e)
                continue
            if w.store is not None:
                gone = w.store.job_ended(job_id)
                log.info("job %s has ended: deleted its %d model file(s)", job_id, len(gone))
            del self.weights[job_id]

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
        self.wants[rnd] = dict(want)
        self.unit_stats = {rnd: []}
        for u in units:
            gu = graph.units[prep[u.unit_id]]
            self.names.setdefault(rnd, {})[u.unit_id] = gu.name
            if not self._fetch(job, weights, gu):
                continue
            self.client.stats.activity = f"prep: {gu.name}"
            cheat = self.prep_cheat(gu.name) if self.prep_cheat is not None else None
            commit = run_unit(job.spec, job.data.tokens, gu, store, weights, self.work(job), want=want,
                              device=self.device(), tamper=cheat)
            stats = commit.get("stats") if isinstance(commit.get("stats"), dict) else {}
            self.unit_stats[rnd].append({"unit": gu.name, **stats})
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
        self.names.setdefault(rnd, {})[task["unit"]["unit_id"]] = gu.name
        raw = str(task.get("commitment", "")).encode()
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

    # --- a prep unit's dispute ---

    def prep_game(self, rnd: int, job: TrainingJob, game: dict, remote):
        """(the program, our Party) for a game over a prep unit (deepseek_program.PrepProgram, deepseek_tracer.Party),
        made once a game. As its defender, before the first split, our unit's first and last states are run again
        from its inputs and put on the node, for the challenger to fetch."""
        import types

        from .deepseek_program import PrepProgram
        from .deepseek_tracer import Executor, Party
        from .finetune import device_of

        key = (rnd, game["id"])
        if key not in self._games:
            graph = job.spec.graph(job.data.tokens)
            gu = graph.units[self.names[rnd][game["unit"]]]
            prog = PrepProgram.of(job.spec, job.data, gu)
            weights = self.weights_for(job)
            rt = types.SimpleNamespace(model=None, device=device_of(self.client.backend), weights=self.prep_weights(job))
            cheat = self.prep_cheat(gu.name) if self.prep_cheat is not None else None
            party = Party(rt, job.spec, job.data, remote, cheat)
            if game["defender"] == self.client.keys.address:
                from . import deepseek_split as split
                self.round = rnd
                st = split._start(self.store(), prog, job.spec.top, gu, {}, self.wants.get(rnd), None)
                st = {k: v.to(rt.device) for k, v in st.items()}
                party.publish(st)
                ex = Executor(rt, job.spec, prog, cheat)
                ex.token_map = weights.token_map(self.work(job))
                for s in range(prog.segments):  # labelled as its run was (deepseek_split._compute)
                    node = prog.segment(s)
                    ex.run_atoms(node, 0, len(node.atoms), st, ())
                party.publish(st)
            self._games = {k: v for k, v in self._games.items() if k[0] == rnd}
            self._games[key] = (prog, party)
        return self._games[key]

    def prep_weights(self, job: TrainingJob):
        """The frozen layers' tensors by name, the embedding and the engram token map (and the tables, for the tiny
        config): what a prep unit's ops and its referee read."""
        if job.job_id not in self._prep_weights:
            from . import deepseek_split as split
            weights = self.weights_for(job)
            tmap = weights.token_map(self.work(job))
            if weights.tiny:
                from . import deepseek as ds
                self._prep_weights[job.job_id] = {**ds.random_weights(job.spec.cfg, 0), "engram.token_map": tmap}
            else:
                self._prep_weights[job.job_id] = split._Weights(weights.source(), self.work(job),
                                                                {"engram.token_map": tmap})
        return self._prep_weights[job.job_id]

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
