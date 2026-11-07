"""A worker's part in an evolution-strategies job's rounds (es.py), as client.WorkerClient plays it: the round's state,
rebuilt from the chain where it can be (else fetched from the node); in compute, each unit's members answering the
round's problems, the answers uploaded and the rewards committed to; in reveal, the answers' id and the rewards; in
verify, another worker's sampled members teacher-forced through their perturbed models.

A rule-2 job (es2.py, es2_rounds.py) goes to Es2Work: the weights a round starts from are worked out from the chain on
this computer (a float32 change from the base, kept per job in this process, es2.Change: the job's parents mixed in at
its start, then each round's entry applied as its block is sealed), loaded into a bf16 model on CUDA (float32 on the
CPU); every member of the worker's units answers in one batch under the shared-prompt rule (es_fast.answer_members),
is scored right or wrong by the task's own check and, once the job has targets, on its replay targets
(es_fast.score_items); a checker teacher-forces its sampled members (es_fast.gaps_items, spec.margin) and recomputes
their replay log-probabilities (spec.lp_tol). A round's twist factors are made once a process and shared by answering,
checking and the change's update.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import es, es2, es2_rounds
from .coordinator import MAX_GAP
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
        # the model didn't give: the problems' solutions), "nester" (an answers blob of JSON nested too deep to parse)
        self.rebuilt = self.fetched = 0  # how each round's state was had: worked out from the chain, or downloaded
        self.timings: dict[int, dict] = {}  # round -> seconds answering, checking
        # task id -> the worst gap a rule-2 check found, when nothing but the gaps could decide it (Es2Work.check_many):
        # sent with the verdict, and taken from here as it is (Client._answer)
        self.gaps: dict[str, float] = {}
        self.v2 = Es2Work(self)  # rule-2 jobs' (es2.py)

    def why_not(self, job: TrainingJob) -> str | None:
        from .finetune import available
        if not available():
            return f"job {job.job_id} trains by evolution strategies on {job.spec.repo}, which needs PyTorch"
        if job.spec.rule == es2.RULE2:
            return self.v2.why_not(job)
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
        if job.spec.rule == es2.RULE2:
            yield from self.v2.compute(rnd, job, units, given)
            return
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

    def check_many(self, rnd: int, job: TrainingJob, claims: list) -> list[tuple[bool, int | None]]:
        """Verdicts on several claims of one round [(task, unit)]: a rule-2 job's teacher-forced together
        (Es2Work.check_many), a rule-1 job's one at a time."""
        if job.spec.rule == es2.RULE2:
            return self.v2.check_many(rnd, job, claims)
        return [self.check(rnd, job, t, u) for t, u in claims]

    def check(self, rnd: int, job: TrainingJob, task: dict, unit: WorkUnit) -> tuple[bool, int | None]:
        """A verdict on another worker's claim: its sampled members' answers teacher-forced (es.check)."""
        if job.spec.rule == es2.RULE2:
            return self.v2.check(rnd, job, task, unit)
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


# --- rule 2 (es2.py) ------------------------------------------------------------------------------------------------

_rt2: dict = {}  # (manifest, device, tensors) -> es.Runtime with .linears, .base and .loaded: one model a process
_lock2 = threading.RLock()  # the lines, the blocks read and the factors below, shared by a process's workers
_chains: dict = {}  # (node address, genesis hash) -> that chain's blocks read, lines and starts (_Chain)
_factors: "OrderedDict[tuple, dict]" = OrderedDict()  # (round seed, pairs, device) -> every pair's twist factors
MAX_LINES = 4  # jobs' changes kept at once (each a float32 copy of the twisted weights: 1.4 GB for Qwen2.5-0.5B)


class _Chain:
    """What this process has read of one chain: blocks by height, each rule-2 job's line (newest used last), and
    starts (a start's key -> its parents' changes mixed: a cycle's jobs share one)."""

    def __init__(self):
        self.blocks: dict[int, Block] = {}
        self.lines: "OrderedDict[str, _Line]" = OrderedDict()
        self.starts: "OrderedDict[str, es2.Change]" = OrderedDict()


class _Line:
    """A rule-2 job as this process has followed it: the last block read, its state (es2.py), its round entries in
    order (the replay targets come from them) and its weight change from the base."""

    def __init__(self, job_id: str, spec, change):
        self.job_id, self.spec, self.change = job_id, spec, change
        self.height, self.state, self.history = 0, spec.init_params(), []
        self.last_round = self.saved = 0  # the height of its newest round, and of the one last saved to disk

    @property
    def params_hash(self) -> str:
        return es.state_hash(self.state)


class _Rebuild:
    """The model being worked out from the chain right now, for the dashboards and the log. A process's first round of
    a job (after a restart, say) replays every round of the job and its ancestors on the CPU, about 5 s a round for
    Qwen2.5-0.5B: half an hour by a few hundred rounds, with nothing else on screen moving. One at a time a process
    (under _lock2): `total` is counted when the outermost _line starts, `done` goes up a replayed round at a time."""

    SHOW = 3  # rounds to replay before it counts as a rebuild (each round's own update is one)
    LOG_EVERY = 30.0  # seconds between the log's progress lines

    def __init__(self):
        self.depth = 0  # _line calls in progress (a job's parents are followed inside its own)
        self.job, self.done, self.total, self.started, self.logged = "", 0, 0, 0.0, 0.0

    def begin(self, job_id: str, total: int) -> None:
        self.job, self.done, self.total = job_id, 0, total
        self.started = self.logged = time.monotonic()
        if self.active():
            log.info(f"rebuilding job {job_id}'s model from the chain on the CPU: {total:,} rounds to replay, its own "
                     f"and its ancestors' (saved to disk as it goes, so the next start is quick)")

    def step(self, job_id: str) -> None:
        self.job, self.done = job_id, self.done + 1
        if self.active() and time.monotonic() - self.logged >= self.LOG_EVERY:
            self.logged = time.monotonic()
            log.info(f"rebuilding the model from the chain: {self.words()}")

    def end(self, ok: bool) -> None:
        if ok and self.total >= self.SHOW and self.done:
            log.info(f"the model is rebuilt from the chain: {self.done:,} rounds in {_minutes(time.monotonic() - self.started)}")
        self.total = 0

    def active(self) -> bool:
        return self.depth > 0 and self.total >= self.SHOW

    def left_s(self) -> float | None:
        if not self.done:
            return None
        return (time.monotonic() - self.started) / self.done * max(0, self.total - self.done)

    def words(self) -> str:
        left = self.left_s()
        return (f"{min(self.done, self.total):,} of {self.total:,} rounds (job {self.job})"
                + (f", about {_minutes(left)} left" if left is not None else ""))


def _minutes(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min"


rebuild = _Rebuild()


def rebuilding() -> tuple[int, int, str, float | None] | None:
    """While this process works a model out from the chain: (rounds done, rounds in all, the job being replayed,
    seconds left or None); else None. For the dashboards (minerui)."""
    r = rebuild
    if not r.active():
        return None
    return min(r.done, r.total), r.total, r.job, r.left_s()


# A line's change saved to disk, so a restarted worker carries on from it instead of rebuilding every round since the
# job's first ancestor on the CPU (which grows with the chain: 15+ minutes by a few hundred rounds). Only a cache: a
# saved line is used only when its state is the one the chain's block at its height holds.
SAVE_EVERY = 8  # rounds between a line's saves
KEEP_SAVED = 3  # newest saved lines kept (each 1.4 GB for Qwen2.5-0.5B)


def _saved_dir() -> Path:
    from . import settings
    return settings.home() / "cache" / "evo2-lines"


def _save_line(ln: _Line) -> None:
    import torch
    d = _saved_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{ln.job_id}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        torch.save({"job": ln.job_id, "height": ln.height, "last_round": ln.last_round,
                    "state": np.asarray(ln.state, np.float32).tolist(), "history": json.dumps(ln.history, default=float), "d": ln.change.d}, tmp)
        os.replace(tmp, d / f"{ln.job_id}.pt")
        ln.saved = ln.last_round
        for old in sorted(d.glob("*.pt"), key=lambda p: p.stat().st_mtime)[:-KEEP_SAVED]:
            old.unlink(missing_ok=True)
    except OSError as e:  # a full disk: the line works on, unsaved
        log.warning(f"couldn't save job {ln.job_id}'s line to {d}: {e}")
        tmp.unlink(missing_ok=True)


def _load_line(c: _Chain, job_id: str, spec):
    """The saved line of `job_id`, if there is one and its state is what the chain's block at its height holds."""
    import torch
    f = _saved_dir() / f"{job_id}.pt"
    if not f.exists():
        return None
    try:
        saved = torch.load(f, map_location="cpu", weights_only=True)
        b = c.blocks.get(int(saved["last_round"]))
        state = np.asarray(saved["state"], np.float32)
        if saved["job"] != job_id or b is None or b.summary is None or b.summary.job_id != job_id or \
                b.summary.params_hash != es.state_hash(state):
            return None
    except Exception as e:  # noqa: BLE001 - a damaged or foreign file: rebuilt from the chain instead
        log.warning(f"ignoring {f}: {e}")
        return None
    ch = es2.Change.__new__(es2.Change)
    ch.d = saved["d"]
    ln = _Line(job_id, spec, ch)
    ln.height, ln.state, ln.history = int(saved["height"]), state, json.loads(saved["history"])
    ln.last_round = ln.saved = int(saved["last_round"])
    log.info(f"job {job_id}: carried on from its line saved at block {ln.last_round}")
    return ln


def _posted_at(c: _Chain, job_id: str) -> int | None:
    """The height of the block that posted `job_id` (its post_job's memo is the job's spec), among the blocks read;
    0 for a starter job, which the genesis brings."""
    for h, b in c.blocks.items():
        for tx in b.txs:
            if tx.kind == "genesis" and not h and f'"{job_id}"' in tx.memo:
                return 0
            if tx.kind == "post_job" and f'"{job_id}"' in tx.memo:
                try:
                    if json.loads(tx.memo).get("id") == job_id:
                        return h
                except (ValueError, AttributeError):
                    continue
    return None


def _saved_round(job_id: str) -> int | None:
    """The block a saved line of `job_id` was saved at, read without loading its weights; None without one."""
    f = _saved_dir() / f"{job_id}.pt"
    if not f.exists():
        return None
    try:
        import torch
        return int(torch.load(f, map_location="cpu", weights_only=True, mmap=True)["last_round"])
    except Exception:  # noqa: BLE001 - unreadable: _load_line won't use it either
        return None


def runtime2(spec, device: str) -> es.Runtime:
    """A rule-2 job's base on `device`, loaded once a process: bf16 on CUDA (es_fast's engine), float32 elsewhere; its
    twisted linears ([(module name, module)], sorted) checked against the spec, and their base weights in float32 on
    the CPU (for es2.Change.load_into)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (spec.manifest, device, spec.tensors, spec.n, spec.tensors_sha)
    with _lock2:
        rt = _rt2.get(key)
        if rt is not None:
            return rt
        folder = es.LOCAL.get(spec.manifest)
        if folder is None:
            raise RuntimeError(f"{spec.repo}'s weights aren't on this computer")
        cuda = device.startswith("cuda")
        model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16 if cuda else torch.float32,
                                                     attn_implementation="sdpa").to(device).eval()
        tok = AutoTokenizer.from_pretrained(folder)
        gen = Path(folder) / "generation_config.json"
        gen = json.loads(gen.read_text()) if gen.exists() else {}
        eos = gen.get("eos_token_id", tok.eos_token_id)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        pad = gen.get("pad_token_id", tok.pad_token_id if tok.pad_token_id is not None else eos[0])
        params = dict(model.named_parameters())
        chosen = es.select(((n, p.shape) for n, p in params.items()), spec.tensors)
        if es.tensors_digest(chosen) != (spec.n, spec.tensors_sha):
            raise ValueError("the spec's tensors aren't what its patterns pick from this model")
        rt = es.Runtime(model, tok, device, eos, pad)
        rt.linears = [(name.rsplit(".", 1)[0], model.get_submodule(name.rsplit(".", 1)[0])) for name, _ in chosen]
        rt.base = {mod: m.weight.detach().float().cpu().clone() for mod, m in rt.linears}
        rt.loaded = None  # (job, params_hash) whose weights the model holds
        _rt2[key] = rt
        return rt


def round_factors(rt: es.Runtime, spec, round_seed: int) -> dict:
    """Every pair's twist factors for a round (es_fast.lowrank_factors, pair j at index j), made once a process: the
    noise is about 550k numbers a pair."""
    import torch
    from .es_fast import lowrank_factors

    dtype = torch.bfloat16 if rt.device.startswith("cuda") else torch.float32
    key = (round_seed, spec.pairs, rt.device, str(dtype))
    with _lock2:
        fac = _factors.get(key)
        if fac is None:
            fac = lowrank_factors(rt.linears, es2.pair_seeds(round_seed, range(spec.pairs)), rt.device, dtype)
            _factors[key] = fac
            while len(_factors) > 2:
                _factors.popitem(last=False)
        else:
            _factors.move_to_end(key)
        return fac


def apply_round(rt: es.Runtime, spec, ch: es2.Change, ent: dict) -> None:
    """A round's entry onto a change (es2.apply_entry's arithmetic, with the round's shared factors): the job's anchor
    gives back spec.decay of the change, then the round's step."""
    import torch
    if not ent["update"]:
        return
    ch.shrink(getattr(spec, "decay", 0.0))
    fac = round_factors(rt, spec, int(ent["seed"], 16))
    idx = torch.tensor([j for j, _ in ent["update"]], device=rt.device)
    ch.apply(rt.linears, {n: (A[idx], B[idx]) for n, (A, B) in fac.items()}, [c for _, c in ent["update"]],
             ent["scale"])


def prompt_rows(rt: es.Runtime, items) -> list[list[int]]:
    """Each item's chat prompt as token ids. Thinking off: Qwen3.5-4B's template opens a <think> block by default,
    which a 64-token answer never gets out of; the templates that don't know the switch (Qwen2.5's, and
    Qwen3.5-0.8B's, which already closes the block) give the same ids with it as without."""
    out = []
    for it in items:
        ids = rt.tokenizer.apply_chat_template(it.messages, add_generation_prompt=True, tokenize=True,
                                               enable_thinking=False)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        out.append(list(ids))
    return out


class Es2Work:
    """A worker's part in rule-2 Evo rounds (the module's docstring)."""

    def __init__(self, work: EsWork):
        self.work = work

    @property
    def client(self):
        return self.work.client

    def why_not(self, job: TrainingJob) -> str | None:
        from .es_local import find_base
        try:
            from . import es_fast  # noqa: F401 - PyTorch
            from . import es_tasks
            for task in job.spec.tasks or (job.spec.task,):  # their datasets' downloads, once (TaskUnavailable without)
                es_tasks.get(task).items(0, 1)
            find_base(job.spec, download=True)  # a public Evo network's base, fetched once (es_local.find_base)
            runtime2(job.spec, self.work.device())
        except Exception as e:  # noqa: BLE001 - whatever stops it, this worker can't take the job
            return f"job {job.job_id}: {e}"
        return None

    def runtime(self, job: TrainingJob) -> es.Runtime:
        from .es_local import find_base
        find_base(job.spec)
        return runtime2(job.spec, self.work.device())

    # --- the weights, from the chain ---

    def _chain(self) -> tuple[_Chain, int]:
        """This chain's cache (by the node's address and genesis), its blocks read up to the node's tip; and the tip."""
        node = self.client.home
        st = node.status()
        tip = int(st["height"])
        c = _chains.setdefault((getattr(node, "url", ""), st.get("genesis_hash")), _Chain())
        if c.blocks and max(c.blocks) > tip:  # not the chain it was (a testnet started over): read it afresh
            c.__init__()
        while (max(c.blocks) if c.blocks else 0) < tip:
            lo = max(c.blocks) + 1 if c.blocks else 1
            batch = node.get(f"/blocks?from={lo}&to={min(tip, lo + 99)}")
            if not batch:
                break
            for d in batch:
                b = Block.from_dict(d)
                c.blocks[b.height] = b
        return c, tip

    def _spec_of(self, job_id: str):
        if job_id in self.client.jobs:
            return self.client.jobs[job_id].spec
        info = self.client.node.get(f"/job/{job_id}")
        return TrainingJob.from_spec_json(json.dumps(info["spec"]), info["owner"], 0).spec

    def _start(self, rt: es.Runtime, spec) -> es2.Change:
        """A job's change before its first round: its parents' changes, each times its weight (a fresh zero change
        for a job from the base). A cycle's jobs share their start, so it's worked out once."""
        key = canonical([spec.manifest, [list(s) for s in spec.start]]).decode()
        ch = es2.Change(rt.linears, "cpu")
        if not spec.start:
            return ch
        c, _ = self._chain()
        mixed = c.starts.get(key)
        if mixed is None:
            mixed = es2.Change(rt.linears, "cpu")
            for pj, w, pin in spec.start:
                parent = self._line(rt, pj, self._spec_of(pj), pin)
                mixed.add(parent.change, w)
            c.starts[key] = mixed
            while len(c.starts) > 2:
                c.starts.popitem(last=False)
        ch.add(mixed, 1.0)
        return ch

    def _line(self, rt: es.Runtime, job_id: str, spec, pin: str = "") -> _Line:
        """Job `job_id` followed up to the chain's tip (or, with `pin`, until its state's hash chain or params_hash is
        that): every round's record checked against its block, its entry applied to the change, the state worked out
        and checked against the params_hash the block holds. A long replay shows its progress (`rebuild`)."""
        with _lock2:
            c, tip = self._chain()
            rebuild.depth += 1
            ok = False
            try:
                if rebuild.depth == 1:
                    rebuild.begin(job_id, self._todo(c, job_id, spec, set()))
                ln = self._follow(rt, c, tip, job_id, spec, pin)
                ok = True
                return ln
            finally:
                rebuild.depth -= 1
                if rebuild.depth == 0:
                    rebuild.end(ok)

    def _todo(self, c: _Chain, job_id: str, spec, seen: set, before: float = float("inf")) -> int:
        """About how many rounds _line(job_id) will replay: the job's rounds with accepted units past what this process
        has of its line (in memory, or saved to disk) and below `before` (for a parent, its child's posting: the
        child's start pins the parent's state then); and, with neither line, its parents' too unless their mix is kept."""
        if job_id in seen:
            return 0
        seen.add(job_id)
        ln = c.lines.get(job_id)
        after = ln.height if ln is not None else _saved_round(job_id)
        n = sum(1 for h, b in c.blocks.items() if (after or 0) < h < before and b.summary is not None
                and b.summary.job_id == job_id and b.summary.accepted_units)
        if after is None and spec.start:
            key = canonical([spec.manifest, [list(s) for s in spec.start]]).decode()
            if key not in c.starts:
                posted = _posted_at(c, job_id)
                n += sum(self._todo(c, pj, self._spec_of(pj), seen, float("inf") if posted is None else posted)
                         for pj, _, _ in spec.start)
        return n

    def _follow(self, rt: es.Runtime, c: _Chain, tip: int, job_id: str, spec, pin: str) -> _Line:
        ln = c.lines.get(job_id)
        if ln is None:
            ln = _load_line(c, job_id, spec) or _Line(job_id, spec, self._start(rt, spec))
            c.lines[job_id] = ln
            while len(c.lines) > MAX_LINES:
                c.lines.popitem(last=False)
        c.lines.move_to_end(job_id)

        def pinned() -> bool:
            return bool(pin) and pin in (es2.hash_of(ln.state), ln.params_hash)
        if pinned():
            return ln
        node = self.client.home
        for h in range(ln.height + 1, tip + 1):
            b = c.blocks.get(h)
            if b is None:
                break
            s = b.summary
            if s is not None and s.job_id == job_id:
                record = node.get(f"/round/{h}/record")
                if h256(canonical(record)).hex() != s.record_hash:
                    raise ValueError(f"block {h}'s record isn't the one it commits to")
                e = record["es"]
                if e.get("update"):
                    ent = es2_rounds.entry_of(e)
                    apply_round(rt, spec, ln.change, ent)
                    ln.state = es2.step_state(ln.state, ent, float(e["reward"]))
                    ln.history.append(ent)
                    rebuild.step(job_id)
                if ln.params_hash != s.params_hash:
                    c.lines.pop(job_id, None)  # (whatever it holds is wrong now: worked out again next time)
                    raise ValueError(f"block {h}: the state worked out isn't the one the block holds")
                ln.last_round = h
            ln.height = h
            if pinned():
                break
        if ln.last_round - ln.saved >= SAVE_EVERY:  # once caught up, not along the way
            _save_line(ln)
        if pin and not pinned():
            raise ValueError(f"job {job_id} never reached the state {pin[:16]}… its child starts from")
        return ln

    def at(self, job: TrainingJob, start_hash: bytes) -> tuple[es.Runtime, _Line]:
        """The runtime, and the job's line as the round starts (its state's params_hash is the round's start hash),
        its weights loaded into the model; the caller holds rt.busy."""
        rt = self.runtime(job)
        ln = self._line(rt, job.job_id, job.spec)
        if ln.params_hash != start_hash.hex():
            raise ValueError(f"job {job.job_id}: the chain's state {ln.params_hash[:16]}… isn't the round's start")
        self.work.rebuilt += 1
        return rt, ln

    @staticmethod
    def load(rt: es.Runtime, ln: _Line) -> None:
        """The line's weights in the model (base + change, in the model's dtype), unless it holds them already."""
        if rt.loaded != (ln.job_id, ln.params_hash):
            ln.change.load_into(rt.linears, rt.base)
            rt.loaded = (ln.job_id, ln.params_hash)

    def replay_items(self, rt: es.Runtime, ln: _Line) -> list[tuple[list[int], list[int]]]:
        """The round's replay targets as (prompt ids, answer ids): the last `replay` of the job's targets so far. A
        target with ids the model doesn't have, or none, is left out (worker and checker alike): nodes before 0.5.27
        took one from a miner's blob, and scoring it ends the round (IndexError, or a device-side assert) every round."""
        spec, out, rows = ln.spec, [], {}
        n_vocab = rt.model.get_input_embeddings().num_embeddings
        for seed, i, ids, step in es2_rounds.targets_of(ln.history, spec.replay):
            if not ids or not all(type(t) is int and 0 <= t < n_vocab for t in ids):
                continue
            if seed not in rows:
                rows[seed] = prompt_rows(rt, es2_rounds.items(spec, seed, step))
            out.append((rows[seed][i], ids))
        return out

    # --- compute ---

    def compute(self, rnd: int, job: TrainingJob, units: list[WorkUnit], given: dict):
        """Every unit's members answering in one batch, each member's right/wrong score and (once the job has targets)
        its replay log-probability; each unit's answers uploaded and committed to with both lists. Yields each unit
        committed."""
        import torch
        from . import es_fast

        spec, keys, st, work = job.spec, self.client.keys, self.client.stats, self.work
        t0 = time.perf_counter()
        rt, ln = self.at(job, units[0].start_hash)
        seed = units[0].seed
        step = int(es2.ctrl_of(ln.state)[2])  # the round's step: its task (es2.task_at)
        ps = es2_rounds.items(spec, seed, step)
        rows = prompt_rows(rt, ps)
        want_lps = bool(spec.replay) and es2.ctrl_of(ln.state)[3] > 0
        replay = self.replay_items(rt, ln) if want_lps else []
        fac = round_factors(rt, spec, seed)
        members = [(j, sign) for u in units for j, _, sign in es.unit_members(spec, seed, u.index)]
        work.claims[rnd] = []
        st.activity = (f"answering {len(units)} Evo unit{'s' * (len(units) > 1)} ({len(members)} twisted models x "
                       f"{spec.problems} {es2.task_at(spec, step)} problems)")
        with rt.busy, torch.no_grad():
            self.load(rt, ln)
            got = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank,
                                         max(work.max_rows, len(rows)), loop=spec.loop_stop)
            lp = None
            if replay:
                st.activity = f"scoring {len(members)} twisted models on {len(replay)} earlier right answers"
                lp = es_fast.score_items(rt, rt.linears, fac, members, replay, spec.sigma, spec.rank).mean(1).tolist()
        M = spec.members
        for n, u in enumerate(units):
            mine = got[n * M:(n + 1) * M]
            answers = [[{"ids": list(a), "text": es.text_of(rt, a)} for a in member] for member in mine]
            if work.cheat == "forger":  # the right answers, which the twisted models didn't give
                forged = []
                for p in ps:
                    text = f"<answer>{p.gold}</answer>"
                    ids = rt.tokenizer.encode(text, add_special_tokens=False)[:spec.max_tokens - 1] + [rt.eos[0]]
                    forged.append({"ids": ids, "text": es.text_of(rt, ids)})
                answers = [forged] * M
            scores = [es.mean_reward([1.0 if p.check(a["text"]) else 0.0 for p, a in zip(ps, member)])
                      for member in answers]
            rewards = [[scores[2 * k], scores[2 * k + 1]] for k in range(spec.pairs_per_unit)]
            lps = None if lp is None else [[lp[n * M + 2 * k], lp[n * M + 2 * k + 1]] for k in range(spec.pairs_per_unit)]
            if work.cheat == "liar":
                rewards = [[1.0, 1.0] for _ in rewards]
            if work.cheat == "lp-liar" and lps is not None:  # log-probabilities its models didn't give
                lps = [[a + 0.5, b - 0.5] for a, b in lps]
            raw = es.claim_bytes(u.unit_id, answers)
            if work.cheat == "nester":  # what the node's parser gives up on with a RecursionError, not a ValueError
                raw = b"[" * 200_000
            bid = hashlib.sha256(raw).hexdigest()
            root, salt = es2_rounds.claim_root(bid, rewards, lps), os.urandom(16)
            digest = commit_digest(u.unit_id, keys.public, root, salt)
            unsigned = Commitment(u.unit_id, keys.address, keys.public, digest, b"")
            cm = Commitment(u.unit_id, keys.address, keys.public, digest, keys.sign(unsigned.message()))
            if self.client._try(lambda: self.client.node.put_blob(keys, raw, bid) or True) is None:
                continue  # not uploaded: committing would only get it called unchecked
            if self.client._try(self.client._signed, "commit", {"round": rnd, "commitment": commitment_out(cm)},
                                commitment_out(cm)) is not None:
                work.claims[rnd].append({"unit_id": u.unit_id, "worker": keys.address, "merkle_root": root.hex(),
                                         "salt": salt.hex(), "es": {"answers": bid, "rewards": rewards, "lps": lps}})
                yield u
        took = time.perf_counter() - t0
        work.timings.setdefault(rnd, {})["answer_s"] = took
        log.info("round %d: answered %d Evo units (%d members) in %.1f s", rnd, len(units), len(members), took)

    # --- verify ---

    def check(self, rnd: int, job: TrainingJob, task: dict, unit: WorkUnit) -> tuple[bool, int | None]:
        """A verdict on another worker's claim: each sampled member's answers token ids that decode to its texts and
        end as answers do, each token its twisted model's greedy choice within spec.margin (teacher-forced under the
        shared-prompt rule), and, when the claim has replay log-probabilities, those members' within spec.lp_tol of
        what this computer works out. (check_many, for one claim.)"""
        return self.check_many(rnd, job, [(task, unit)])[0]

    def check_many(self, rnd: int, job: TrainingJob, claims: list) -> list[tuple[bool, int | None]]:
        """Verdicts on several claims of one round [(task, unit)], each as check() gives it (its first bad member, in
        the task's order: a member's answers that aren't one a problem, don't decode to its texts or don't end as
        answers do; then a token more than spec.margin below the greedy choice; then a replay log-probability more
        than spec.lp_tol off), with every claim's sampled members teacher-forced together (es_fast.gaps_members,
        score_items). A round's claims share its state, seed, problems and twist factors, and the engine's passes
        always hold the same number of rows, so a member's gaps are what it gets alone: one call fills the passes a
        member at a time left three quarters empty (16 problems in a 64-row pass on Qwen3.5-0.8B)."""
        import torch
        from . import es_fast

        if not claims:
            return []
        if len({u.seed for _, u in claims}) > 1 or len({u.start_hash for _, u in claims}) > 1:
            return [v for t, u in claims for v in self.check_many(rnd, job, [(t, u)])]  # (never: one round's claims)
        t0 = time.perf_counter()
        spec = job.spec
        verdicts: list = [None] * len(claims)
        parsed = []  # (claim, chosen members, their answers' ids, their texts, the claim's replay log-probabilities)
        for c, (task, unit) in enumerate(claims):
            chosen = [int(k) for k in task["segments"]]
            raw = self.client.node.get_blob(task["answers"])
            if raw is None:
                verdicts[c] = (False, chosen[0])
                continue
            try:
                answers = json.loads(raw)["answers"]
                got = {k: [list(map(int, a["ids"])) for a in answers[k]] for k in chosen}
                texts = {k: [a["text"] for a in answers[k]] for k in chosen}
            except (ValueError, KeyError, TypeError, IndexError, RecursionError):  # (RecursionError: JSON nested too deep)
                verdicts[c] = (False, chosen[0])
                continue
            parsed.append((c, chosen, got, texts, task.get("lps")))
        if parsed:
            unit0 = claims[parsed[0][0]][1]
            rt, ln = self.at(job, unit0.start_hash)
            seed = unit0.seed
            rows = prompt_rows(rt, es2_rounds.items(spec, seed, int(es2.ctrl_of(ln.state)[2])))
            replay = self.replay_items(rt, ln) if any(p[4] is not None for p in parsed) else []
            fac = round_factors(rt, spec, seed)
            why: dict = {}  # (claim, member) -> what's wrong before any teacher forcing
            todo = []  # (claim, member, (pair, sign), its answers): every member that gets that far
            n_vocab = rt.model.get_input_embeddings().num_embeddings
            for c, chosen, got, texts, _ in parsed:
                members = es.unit_members(spec, seed, claims[c][1].index)
                for k in chosen:
                    if len(got[k]) != spec.problems:
                        why[(c, k)] = "a member's answers aren't one a problem"
                    # Ids the model has no row for: decoding 10**12 or -5 raises, and past the embedding the GPU
                    # itself stops (a device-side assert), so another worker's blob could end this checker's round.
                    elif any(not 0 <= t < n_vocab for a in got[k] for t in a):
                        why[(c, k)] = "an answer has token ids the model doesn't have"
                    elif any(t != es.text_of(rt, a) for a, t in zip(got[k], texts[k])):
                        why[(c, k)] = "an answer's text isn't its tokens' decoding"
                    elif not all(es2.ends_right(a, spec, rt.eos) for a in got[k]):
                        why[(c, k)] = ("an answer stops before an end-of-turn token or max_tokens"
                                       + (", or runs on past where it loops" if spec.loop_stop else ""))
                    else:
                        j, _, sign = members[k]
                        todo.append((c, k, (j, sign), got[k]))
            worst, mine = {}, {}
            lps_of = {p[0]: p[4] for p in parsed}
            with rt.busy, torch.no_grad():
                self.load(rt, ln)
                if todo:
                    gaps = es_fast.gaps_members(rt, rt.linears, fac, [m for _, _, m, _ in todo], rows,
                                                [a for _, _, _, a in todo], spec.sigma, spec.rank,
                                                max_tokens=spec.max_tokens)
                    for (c, k, _, _), g in zip(todo, gaps):
                        worst[(c, k)] = float(g.max()) if g.numel() else 0.0
                    lp_todo = [(c, k, m) for c, k, m, _ in todo if lps_of[c] is not None]
                    if lp_todo and replay:
                        lps = es_fast.score_items(rt, rt.linears, fac, [m for _, _, m in lp_todo], replay, spec.sigma,
                                                  spec.rank).mean(1)
                        for (c, k, _), v in zip(lp_todo, lps):
                            mine[(c, k)] = float(v)
            def lp_wrong(c, k, lps) -> str:
                """Why member k's replay log-probability is off ("" when it isn't, or the claim has none)."""
                if lps is None:
                    return ""
                try:
                    claimed = float(lps[k // 2][k % 2])
                except (TypeError, IndexError, ValueError):
                    return "its replay log-probabilities aren't a unit's"
                got_lp = mine.get((c, k), float("nan"))
                return "" if abs(got_lp - claimed) <= spec.lp_tol else                     f"replay log-probability {claimed:.4f}, worked out {got_lp:.4f}"

            for c, chosen, _, _, lps in parsed:
                ok, bad, reason = True, None, ""
                lp_why = {k: lp_wrong(c, k, lps) for k in chosen if (c, k) in worst}
                tid = claims[c][0].get("task_id")
                if tid is not None and all((c, k) in worst and not lp_why[k] for k in chosen):
                    # Every member teacher-forced and nothing else wrong: the worst gap alone decides, and goes with the
                    # verdict (EsWork.gaps), for the census of how far apart honest machines are (netstats.gap_census).
                    # Only a number the node takes (coordinator.valid_gap): a logit of -inf makes a gap of inf and an
                    # overflow NaN, and a verdict whose gap the node refused would be lost, rejection and all.
                    g = max((worst[(c, k)] for k in chosen), default=0.0)
                    if math.isfinite(g) or g == float("inf"):
                        self.work.gaps[tid] = round(min(max(g, 0.0), MAX_GAP), 4)
                for k in chosen:
                    if (c, k) in why:
                        ok, bad, reason = False, k, why[(c, k)]
                    elif worst[(c, k)] > spec.margin:
                        ok, bad, reason = False, k, f"a token {worst[(c, k)]:.4f} below the greedy choice's logit"
                    elif lp_why[k]:
                        ok, bad, reason = False, k, lp_why[k]
                    if not ok:
                        log.warning("round %d: unit %s's member %d fails the check: %s", rnd, claims[c][1].unit_id,
                                    bad, reason)
                        break
                verdicts[c] = (ok, bad)
        timings = self.work.timings.setdefault(rnd, {})
        timings["check_s"] = timings.get("check_s", 0.0) + time.perf_counter() - t0
        return verdicts
