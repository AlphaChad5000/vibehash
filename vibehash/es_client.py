"""A worker's part in an evolution-strategies job's rounds (es.py), as client.WorkerClient plays it: the round's state,
rebuilt from the chain where it can be (else fetched from the node); in compute, each unit's members answering the
round's problems, the answers uploaded and the rewards committed to; in reveal, the answers' id and the rewards; in
verify, another worker's sampled members teacher-forced through their perturbed models.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np

from . import es
from .crypto import h256
from .ledger import Block, canonical
from .model import params_hash
from .tasks import Commitment, TrainingJob, WorkUnit, commit_digest
from .wire import commitment_out

log = logging.getLogger("vibehash.worker")


class EsWork:
    def __init__(self, client, max_rows: int = 256):
        self.client = client
        self.max_rows = max_rows  # sequences generated at once: members × problems a batch
        self.states: dict[str, tuple[int, np.ndarray]] = {}  # job -> (height it's as of, state), followed block by block
        self.claims: dict[int, list[dict]] = {}  # round -> each committed unit's reveal body
        self.cheat: str | None = None  # tests only: "liar" (claims rewards its answers didn't earn), "forger" (answers
        # the model didn't give: the problems' solutions)
        self.rebuilt = self.fetched = 0  # how each round's state was had: worked out from the chain, or downloaded
        self.timings: dict[int, dict] = {}  # round -> seconds answering, checking

    def why_not(self, job: TrainingJob) -> str | None:
        from .finetune import available
        if not available():
            return f"job {job.job_id} trains by evolution strategies on {job.spec.repo}, which needs PyTorch"
        try:
            self.runtime(job)
        except Exception as e:  # noqa: BLE001 - whatever stops it loading, this worker can't take the job
            return f"job {job.job_id}: {e}"
        return None

    def device(self) -> str:
        from .finetune import device_of
        return device_of(self.client.backend)

    def runtime(self, job: TrainingJob) -> es.Runtime:
        from .es_local import find_base
        find_base(job.spec)  # a local network's base, from this computer's Hugging Face cache (es_local.py)
        return es.runtime(job.spec, self.device())

    # --- the state ---

    def state(self, job: TrainingJob, start_hash: bytes) -> np.ndarray:
        """The state the round starts from: worked out from the chain's records (es.step, each block's summary checked),
        following on from the last one this worker had, or else downloaded from the node."""
        have = self.states.get(job.job_id)
        if have is not None and params_hash(have[1]) == start_hash:
            return have[1]
        try:
            got = self._follow(job)
            if got is not None and params_hash(got) == start_hash:
                self.rebuilt += 1
                return got
        except Exception as e:  # noqa: BLE001 - the chain couldn't be followed: fetch the state instead
            log.debug("couldn't rebuild job %s's state from the chain: %s", job.job_id, e)
        node = self.client.node
        p = node.get("/round/params")
        path = node.download("blob", p["blob"], self.client.blobs, size=job.checkpoint_bytes)
        state = np.fromfile(path, dtype=job.wire_dtype)
        if params_hash(state) != start_hash:
            raise ValueError("the node served a state that doesn't match the round's start hash")
        self.fetched += 1
        return state

    def _follow(self, job: TrainingJob) -> np.ndarray | None:
        node, spec = self.client.home, job.spec
        height, state = self.states.get(job.job_id, (0, None))
        tip = int(node.status()["height"])
        while height < tip:
            batch = node.get(f"/blocks?from={height + 1}&to={min(tip, height + 100)}")
            if not batch:
                break
            for d in batch:
                b = Block.from_dict(d)
                s = b.summary
                if s is not None and s.job_id == job.job_id:
                    record = node.get(f"/round/{b.height}/record")
                    if h256(canonical(record)).hex() != s.record_hash:
                        raise ValueError(f"block {b.height}'s record isn't the one it commits to")
                    h = record["es"]
                    state = es.step(spec, spec.init_params() if state is None else state, int(h["seed"], 16), h["update"])
                    if es.state_hash(state) != s.params_hash:
                        raise ValueError(f"block {b.height}: the state worked out isn't the one the block holds")
                height = b.height
        if state is None:
            state = spec.init_params()
        self.states[job.job_id] = (height, state)
        return state

    # --- compute ---

    def compute(self, rnd: int, job: TrainingJob, units: list[WorkUnit], given: dict):
        """Answer each unit, upload its answers and commit to them and its rewards. Yields each unit committed."""
        spec, keys, st = job.spec, self.client.keys, self.client.stats
        level = int(given["es"]["level"])
        state = self.state(job, units[0].start_hash)
        delta = es.delta_of(spec, state)
        rt = self.runtime(job)
        self.claims[rnd] = []
        t0 = time.perf_counter()
        for i, u in enumerate(units, 1):
            st.activity = f"answering unit {i} of {len(units)} ({spec.members} nudged models x {spec.problems} problems)"
            c = es.claim(rt, spec, delta, u.seed, level, u.index, self.max_rows)
            answers, rewards = c["answers"], c["rewards"]
            if self.cheat == "forger":  # the right answers, which the nudged model didn't give
                ps = es.problems(es.problems_seed(u.seed), spec.problems, level)
                forged = []
                for p in ps:
                    text = f"<answer>{p.solution}</answer>"
                    ids = rt.tokenizer.encode(text, add_special_tokens=False)[:spec.max_tokens - 1] + [rt.eos[0]]
                    forged.append({"ids": ids, "text": es.text_of(rt, ids)})
                answers = [forged] * spec.members
                rewards = es.rescore(spec, u.seed, level, es.claim_bytes(u.unit_id, answers), u.unit_id)
            if self.cheat == "liar":
                rewards = [[1.0, 1.0] for _ in rewards]
            raw = es.claim_bytes(u.unit_id, answers)
            bid = hashlib.sha256(raw).hexdigest()
            root, salt = es.claim_root(bid, rewards), os.urandom(16)
            digest = commit_digest(u.unit_id, keys.public, root, salt)
            unsigned = Commitment(u.unit_id, keys.address, keys.public, digest, b"")
            cm = Commitment(u.unit_id, keys.address, keys.public, digest, keys.sign(unsigned.message()))
            if self.client._try(lambda: self.client.node.put_blob(keys, raw, bid) or True) is None:
                continue  # not uploaded: committing would only get it called unchecked
            if self.client._try(self.client._signed, "commit", {"round": rnd, "commitment": commitment_out(cm)},
                                commitment_out(cm)) is not None:
                self.claims[rnd].append({"unit_id": u.unit_id, "worker": keys.address, "merkle_root": root.hex(),
                                         "salt": salt.hex(), "es": {"answers": bid, "rewards": rewards}})
                yield u
        self.timings.setdefault(rnd, {})["answer_s"] = time.perf_counter() - t0
        log.info("round %d: answered %d Evo units in %.1f s", rnd, len(units), time.perf_counter() - t0)

    def reveals(self, rnd: int) -> list[dict]:
        return self.claims.pop(rnd, [])

    # --- verify ---

    def check(self, rnd: int, job: TrainingJob, task: dict, unit: WorkUnit) -> tuple[bool, int | None]:
        """A verdict on another worker's claim: its sampled members' answers teacher-forced (es.check)."""
        t0 = time.perf_counter()
        raw = self.client.node.get_blob(task["answers"])
        chosen = [int(k) for k in task["segments"]]
        if raw is None:
            return False, chosen[0]
        state = self.state(job, unit.start_hash)
        ok, bad, why = es.check(self.runtime(job), job.spec, es.delta_of(job.spec, state), unit.seed, int(task["level"]),
                                unit.index, raw, chosen, unit.unit_id)
        if not ok:
            log.warning("round %d: unit %s's member %d fails the check: %s", rnd, unit.unit_id, bad, why)
        self.timings.setdefault(rnd, {})["check_s"] = self.timings.get(rnd, {}).get("check_s", 0.0) + time.perf_counter() - t0
        return ok, bad
