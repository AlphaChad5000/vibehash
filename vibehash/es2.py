"""Evo v2's core, ES rule 2, with no protocol code: the job spec, the learning-rate schedule, the pair coefficients, the
canonical round entry and the hash chain that commits to it (packed into a tiny state), the running weight change a
worker keeps, the canonical weights anyone can compute, and the replay targets the node records.

What changes from rule 1 (es.py):
* **Twists** are rank 1 on every linear but the head and embeddings (es_fast.lowrank_factors): pair j's factors come
  from one es.noise stream, about 550k numbers a pair in Qwen2.5-0.5B, the same bits everywhere.
* **The shared-prompt rule** for answering, checking and scoring (es_fast).
* **The node keeps no weights.** A job's state is 16 + es.CTRL float32 words: the 32-byte hash chain as 16 uint16
  values (exact in float32), then the control words. The chain starts at `start_hash` (the base, the twisted tensors
  and the parents) and each round moves it on with `next_hash` over the round's entry (`entry`: seed, update, scale,
  targets). The weights are a function of the chain: base + the change, which starts as the sum over parents of weight
  x their change and, each round with an update, becomes (1 - decay) x itself + scale x sum_j c_j A_j B_j^T. Workers
  keep that change on their device (`Change`, `rebuild`); `canonical_weights` computes it in a fixed float64 order on
  the CPU, for anyone who wants the exact numbers. Updates use the factors rounded to bf16 (what the workers' forward
  uses), so every machine applies the same values.
* **Fitness**: each member's mean per-token log-probability of earlier rounds' correct answers (the targets the node
  records, `replay_targets`), z-scored; right/wrong only while no target exists yet (`coefficients`).
* **Averaging**: a job may start from a weighted sum of earlier jobs' changes (`start`: [[job id, weight, the
  parent's params_hash], ...]), how a cycle of tunes is averaged into the next cycle's starting point.
* **The anchor** (`decay`): before each round's step the change gives back that share of itself, so it stays near the
  base. Without it the change is a random walk: the z-scored step is the same size whether or not a round found
  anything, and the noise adds up. Measured on Qwen2.5-0.5B-Instruct (2026-11-05, HANDOFF "The Evo collapse: cause
  and fix"): the v2 step with random coefficients alone broke the model by round 100 (13% from the base; the testnet's
  line was 24% after ~400 rounds and worse on every task); 60 real svamp rounds at lr 4e-3 took svamp 14% to 29% and
  ag_news 75% to 20%; at lr 1e-3 with decay 1/64, svamp 14% to 25%, the other tasks unchanged, 0.9% from the base.
  Random noise then levels off at about step / sqrt(2 decay); a push the rounds agree on still adds up, to about
  step / decay. 0 (the default, left out of the spec) is every job before it.
"""
from __future__ import annotations

import collections
import hashlib
import re
from dataclasses import dataclass

import numpy as np

from . import es, es_tasks

RULE2 = 2
WORDS = 16  # the chain hash's 32 bytes as 16 uint16 values, ahead of es.CTRL control words
TASKS = es_tasks.NAMES  # the tasks a rule-2 job may name (es_tasks.py: problems and their checks)
TENSORS = ("model.layers.*_proj.weight",)  # every attention and MLP projection's weight (es.select keeps out the head)
_JOB = re.compile(r"^[\w.-]{1,64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Spec2(es.EsSpec):
    """A rule-2 Evo job: es.EsSpec's fields and interface (tensors, n and tensors_sha pin the twisted linears: n their
    weights' count, as es.tensors_digest gives), plus the task, the cosine and the parents. Rule 1's level, retarget,
    reward, optimizer, betas, eps and alpha are unused."""
    task: str = "countdown"
    pairs: int = 64
    pairs_per_unit: int = 8
    problems: int = 16
    max_tokens: int = 64
    lr: float = 4e-3
    lr_floor: float = 0.25  # the cosine's end, as a share of lr
    rounds: int = 50  # the rounds the cosine spans (the job's planned length)
    replay: int = 16
    start: tuple = ()  # ((job id, weight, the parent's params_hash or ""), ...)
    lp_tol: float = 0.02  # how far a claimed replay log-probability (mean per token) may be from a checker's
    rank: int = 1
    rule: int = RULE2
    # Tasks taken in turn, `rounds` rounds each, the cosine starting again with each (task_at, lr_at), for a job that
    # never ends, like a genesis starter job: the testnet's short jobs one task after another, inside one job and one
    # line. Empty: `task` alone, as every job before it (the field is left out of its spec, so its hash is the same).
    tasks: tuple = ()
    # The change's share given back to the base before each round's step (the anchor; see the module's docstring). A
    # power of two's reciprocal (1/64) keeps 1 - decay exact in float32. 0: left out of the spec, as before it.
    decay: float = 0.0
    # Answers scored by es_tasks.lenient: a right answer counts with or without its <answer> tags, so the reward is for
    # being right, not for the format. False: left out of the spec, as before it.
    lenient: bool = False
    # Every round's problems drawn across `tasks` (es2_rounds.items: problem i of step s is task (s * problems + i) mod
    # the count), at a constant lr, instead of `rounds` rounds of one task at a time. Taken in turn, the tasks trained
    # long ago fade: with the anchor (decay) the change mostly holds its last 1 / decay rounds, a 17-task cycle is
    # about 200, and at the end of the owner's second cycle (2026-11-05) the tasks trained first had given back most of
    # what the first cycle taught them. Mixed, every task is always recent. False: left out of the spec, as before it.
    mix: bool = False

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "tasks", tuple(str(t) for t in self.tasks))
        object.__setattr__(self, "start", tuple((str(s[0]), float(s[1]), str(s[2]) if len(s) > 2 else "")
                                                for s in self.start))
        if isinstance(self.decay, int) and not isinstance(self.decay, bool):
            object.__setattr__(self, "decay", float(self.decay))

    @property
    def n_params(self) -> int:
        return WORDS + es.CTRL

    def problems_list(self) -> list[str]:
        out = []
        if not isinstance(self.base, str) or not es._BASE.match(self.base):
            out.append("base is hf:org/name@<40-hex commit>")
        for name in ("manifest", "tensors_sha"):
            if not isinstance(getattr(self, name), str) or not es._SHA.match(getattr(self, name)):
                out.append(f"{name} is a sha256")
        if not (self.tensors and len(self.tensors) <= 16 and isinstance(self.n, int) and self.n > 0):
            out.append("tensors is 1-16 patterns, n the weights they pick")
        if self.task not in TASKS:
            out.append(f"task is one of {', '.join(TASKS)}")
        if len(self.tasks) > 64 or any(t not in TASKS for t in self.tasks) or (self.tasks and self.tasks[0] != self.task):
            out.append("tasks is at most 64 of the tasks, the first being `task`")
        if not (1 <= self.pairs_per_unit <= self.pairs <= 4096 and self.pairs % self.pairs_per_unit == 0):
            out.append("pairs is 1-4096, a multiple of pairs_per_unit")
        if not (1 <= self.problems <= 512 and 1 <= self.max_tokens <= 2048):
            out.append("problems is 1-512 and max_tokens 1-2048")
        if not (0 < self.sigma <= 1 and 0 < self.lr <= 10 and 0 <= self.lr_floor <= 1 and 1 <= self.rounds <= 10**6):
            out.append("sigma in (0, 1], lr in (0, 10], lr_floor in [0, 1], rounds 1-1,000,000")
        if not (0 <= self.replay <= 1024 and 1 <= self.check <= self.members and self.margin >= 0 and self.lp_tol > 0):
            out.append("replay 0-1024, check 1 to a unit's members, margin and lp_tol positive")
        if len(self.start) > 64 or any(not _JOB.match(j) or not -4 <= w <= 4 or (h and not _HEX64.match(h))
                                       for j, w, h in self.start):
            out.append("start is at most 64 [job id, weight in -4..4, params_hash]")
        if not (isinstance(self.decay, float) and 0.0 <= self.decay <= 0.5):
            out.append("decay is 0 to 0.5")
        if not isinstance(self.lenient, bool):
            out.append("lenient is true or false")
        if not isinstance(self.mix, bool) or (self.mix and not self.tasks):
            out.append("mix is true or false, and true only with tasks")
        if self.rank != 1 or self.rule != RULE2 or self.precision != "float32":
            out.append("rank is 1, rule 2, precision float32")
        return out

    def init_params(self, seed: int = 0) -> np.ndarray:
        """The state before any round: the start hash, then control words [RULE2, 0, steps 0, targets 0, 0, 0, 0, 0]."""
        ctrl = np.zeros(es.CTRL, np.float32)
        ctrl[0] = RULE2
        return pack(start_hash(self), ctrl)

    def flops_per_step(self, batch_size: int = 1) -> int:
        """A unit's nominal FLOPs: every member writing every answer and being scored on the replay targets."""
        return 2 * self.base_params * (self.max_tokens + 2 * self.replay) * self.problems * self.members * batch_size

    def to_json(self) -> dict:
        return {"base": self.base, "manifest": self.manifest, "model": self.model,
                "es": {"rule": RULE2, "tensors": list(self.tensors), "n": self.n, "tensors_sha": self.tensors_sha,
                       "task": self.task, "pairs": self.pairs, "pairs_per_unit": self.pairs_per_unit,
                       "problems": self.problems, "max_tokens": self.max_tokens, "sigma": self.sigma, "lr": self.lr,
                       "lr_floor": self.lr_floor, "rounds": self.rounds, "replay": self.replay,
                       "start": [[j, w, h] for j, w, h in self.start], "check": self.check, "margin": self.margin,
                       "lp_tol": self.lp_tol, "rank": self.rank, **({"tasks": list(self.tasks)} if self.tasks else {}),
                       **({"decay": self.decay} if self.decay else {}),
                       **({"lenient": True} if self.lenient else {}), **({"mix": True} if self.mix else {})}}

    @classmethod
    def from_json(cls, d: dict) -> "Spec2":
        e = d["es"]
        known = {"rule", "tensors", "n", "tensors_sha", "task", "pairs", "pairs_per_unit", "problems", "max_tokens",
                 "sigma", "lr", "lr_floor", "rounds", "replay", "start", "check", "margin", "lp_tol", "rank", "tasks",
                 "decay", "lenient", "mix"}
        if set(e) - known:
            raise ValueError(f"a rule-2 ES spec has no field {sorted(set(e) - known)[0]!r}")
        if not isinstance(e["tensors"], list) or not all(isinstance(e.get(k, []), list) for k in ("start", "tasks")):
            raise TypeError("tensors, start and tasks are lists")
        return cls(base=str(d["base"]), manifest=str(d["manifest"]), dims=dict(d["model"]),
                   tensors=tuple(str(t) for t in e["tensors"]), n=int(e["n"]), tensors_sha=str(e["tensors_sha"]),
                   task=str(e["task"]), pairs=int(e["pairs"]), pairs_per_unit=int(e["pairs_per_unit"]),
                   problems=int(e["problems"]), max_tokens=int(e["max_tokens"]), sigma=float(e["sigma"]),
                   lr=float(e["lr"]), lr_floor=float(e["lr_floor"]), rounds=int(e["rounds"]),
                   replay=int(e["replay"]), start=tuple(tuple(s) for s in e.get("start", [])),
                   check=int(e["check"]), margin=float(e["margin"]), lp_tol=float(e["lp_tol"]),
                   rank=int(e.get("rank", 1)), rule=int(e["rule"]), tasks=tuple(str(t) for t in e.get("tasks", [])),
                   decay=float(e.get("decay", 0.0)), lenient=e.get("lenient", False), mix=e.get("mix", False))


def is_v2(d) -> bool:
    """Whether a parsed job spec is a rule-2 Evo job's."""
    return isinstance(d, dict) and isinstance(d.get("es"), dict) and d["es"].get("rule") == RULE2


# --- the state: the hash chain and the control words ------------------------------------------------------------------

def start_hash(spec: Spec2) -> str:
    """h0 = sha256(b"vibehash-es2-start" + canonical({base, tensors_sha, start})): the chain's start, fixed by the
    spec alone (a parent is pinned by its params_hash)."""
    from .ledger import canonical
    body = {"base": spec.base, "tensors_sha": spec.tensors_sha, "start": [[j, w, h] for j, w, h in spec.start]}
    return hashlib.sha256(b"vibehash-es2-start" + canonical(body)).hexdigest()


def pack(hash_hex: str, ctrl) -> np.ndarray:
    """The state: the hash's 32 bytes as 16 big-endian uint16 values in float32 (exact), then es.CTRL control words."""
    raw = bytes.fromhex(hash_hex)
    out = np.zeros(WORDS + es.CTRL, np.float32)
    out[:WORDS] = np.frombuffer(raw, ">u2").astype(np.float32)
    out[WORDS:] = np.asarray(ctrl, np.float32)
    return out


def hash_of(state) -> str:
    words = np.asarray(state[:WORDS], np.float64)
    if not np.all((words >= 0) & (words < 65536) & (words == np.floor(words))):
        raise ValueError("not a rule-2 state: its first 16 words aren't uint16 values")
    return words.astype(">u2").tobytes().hex()


def ctrl_of(state) -> np.ndarray:
    return np.asarray(state[WORDS:], np.float32)


def steps_of(state) -> int:
    return int(ctrl_of(state)[2])


def entry(round_seed: int, update: list, scale: float, targets: list) -> dict:
    """A round's canonical entry, what the chain hash covers: its seed, update [[pair index, c_j], ...] sorted by
    index (its accepted pairs), the step's scale, and its targets [[problem index, [token ids]], ...] (the correct
    answers later rounds' fitness is scored on)."""
    update = sorted([[int(j), float(c)] for j, c in update])
    if len({j for j, _ in update}) != len(update):
        raise ValueError("a pair is in an update once")
    targets = [[int(i), [int(t) for t in toks]] for i, toks in targets]
    return {"seed": f"{round_seed:016x}", "update": update, "scale": float(scale), "targets": targets}


def next_hash(prev_hex: str, ent: dict) -> str:
    """The params_hash after a round: sha256(previous hash bytes || canonical(entry))."""
    from .ledger import canonical
    return hashlib.sha256(bytes.fromhex(prev_hex) + canonical(ent)).hexdigest()


def step_state(state, ent: dict, mean_reward: float) -> np.ndarray:
    """The state after a round: the hash moved on over its entry; steps + 1, targets so far + its targets, its mean
    right/wrong reward."""
    ctrl = ctrl_of(state).copy()
    ctrl[2] += 1
    ctrl[3] += len(ent["targets"])
    ctrl[5] = mean_reward
    return pack(next_hash(hash_of(state), ent), ctrl)


# --- a round's arithmetic ---------------------------------------------------------------------------------------------

def pair_seeds(round_seed: int, js) -> list[int]:
    return [es.pair_seed(round_seed, int(j)) for j in js]


def cos_pi(x: float) -> float:
    """cos(πx) for x in [0, 1], the same bits on every machine: es._sincos's series (libm's cos can differ by an ulp
    between machines, and the step's scale goes into the hash chain)."""
    return float(es._sincos(np.array([x / 2.0], np.float64))[1][0])


def task_at(spec: Spec2, round_index: int) -> str:
    """The task of the job's round `round_index` (its step, from 0): `task`, or with `tasks`, each in turn for `rounds`
    rounds, then round again."""
    if not spec.tasks:
        return spec.task
    if getattr(spec, "mix", False):
        return "mixed"  # every task, problem by problem (es2_rounds.items)
    return spec.tasks[(round_index // max(1, spec.rounds)) % len(spec.tasks)]


def lr_at(spec: Spec2, round_index: int) -> float:
    """The cosine from lr down to lr * lr_floor over the job's planned rounds (round_index from 0), then flat. With
    `tasks`, over each task's `rounds`, starting again with the next task. Mixed (`mix`): lr, constant."""
    if getattr(spec, "mix", False):
        return spec.lr
    if spec.tasks:
        round_index %= max(1, spec.rounds)
    x = min(1.0, round_index / max(1, spec.rounds))
    return spec.lr * (spec.lr_floor + (1 - spec.lr_floor) * 0.5 * (1 + cos_pi(x)))


def coefficients(binary: list, replay_lp: list | None) -> tuple[list[float], str]:
    """Pair coefficients c_j = z+_j - z-_j from members in order [+0, -0, +1, -1, ...]: z-scores of the replay
    log-probabilities when given, else of the right/wrong scores. Returns (coefficients, "replay" or "binary").
    The step is then scale = lr_at(...) / N (N members) times sum_j c_j A_j B_j^T."""
    use = replay_lp if replay_lp is not None and len(replay_lp) == len(binary) else None
    z = es.shaped([float(x) for x in (use if use is not None else binary)])
    return [z[2 * k] - z[2 * k + 1] for k in range(len(z) // 2)], ("replay" if use is not None else "binary")


def replay_targets(n_problems: int, answers: list, ok: list) -> list[tuple[int, list[int]]]:
    """(problem index, token ids) of each problem's most common correct answer among all members' answers (answers
    [member][problem] = token ids; ok [member][problem]); ties: the shortest, then the smallest."""
    out = []
    for i in range(n_problems):
        sols = collections.Counter(tuple(answers[k][i]) for k in range(len(answers)) if ok[k][i])
        if sols:
            out.append((i, list(min(sols, key=lambda s: (-sols[s], len(s), s)))))
    return out


# --- the weights --------------------------------------------------------------------------------------------------

class Change:
    """A job's running weight change from the base, float32 on a device: {linear name: [out, in]}."""

    def __init__(self, linears, device):
        import torch
        self.d = {n: torch.zeros(m.out_features, m.in_features, dtype=torch.float32, device=device)
                  for n, m in linears}

    def apply(self, linears, fac: dict, coefs: list, scale: float) -> None:
        """change += scale * sum_p coefs[p] * A_p B_p^T, fac's pairs in coefs' order, the factors as bf16 (as the
        forward used them)."""
        import torch
        if not coefs:
            return
        dev = next(iter(self.d.values())).device
        c = torch.tensor(coefs, dtype=torch.float32, device=dev)
        for n, _ in linears:
            A, B = fac[n]
            A = A.to(dev, torch.bfloat16).float()
            B = B.to(dev, torch.bfloat16).float()
            self.d[n] += scale * torch.einsum("pmr,pnr->mn", A * c[:, None, None], B)

    def shrink(self, decay: float) -> None:
        """change *= 1 - decay: the anchor, before a round's step (float32 elementwise, the same bits everywhere)."""
        if decay:
            for v in self.d.values():
                v.mul_(1.0 - decay)

    def add(self, other: "Change", weight: float) -> None:
        for n, v in other.d.items():
            self.d[n].add_(v, alpha=float(weight))

    def load_into(self, linears, base: dict) -> None:
        """Each linear's weight = base + change, in the model's dtype (base: {name: float32 tensor})."""
        import torch
        with torch.no_grad():
            for n, m in linears:
                m.weight.data.copy_(base[n].to(self.d[n].device, torch.float32) + self.d[n])


def apply_entry(ch: Change, linears, ent: dict, device, decay: float = 0.0) -> None:
    """One round's entry onto a change (its accepted pairs' factors, drawn again from the round seed), after the
    job's anchor gives back `decay` of it. A round with no update is no step: no decay either."""
    import torch
    from .es_fast import lowrank_factors
    if not ent["update"]:
        return
    ch.shrink(decay)
    seed = int(ent["seed"], 16)
    js = [j for j, _ in ent["update"]]
    fac = lowrank_factors(linears, pair_seeds(seed, js), device, torch.bfloat16)
    ch.apply(linears, fac, [c for _, c in ent["update"]], ent["scale"])


def rebuild(linears, history: list, parents: list = (), device="cuda", decay: float = 0.0) -> Change:
    """A job's change: its parents' [(weight, their Change)], then its rounds' entries in order, each after the job's
    anchor (`decay`, the spec's)."""
    ch = Change(linears, device)
    for w, pc in parents:
        ch.add(pc, w)
    for ent in history:
        apply_entry(ch, linears, ent, device, decay)
    return ch


def canonical_weights(layout: list, base: dict, history: list, parents: list = (), decay: float = 0.0) -> dict:
    """The exact weights, in a fixed order on the CPU, for anyone who wants the numbers rather than a worker's float32
    approximation. layout: es_fast.stream_layout's [(name, out, in)]; base: {name: float32 array}; history: the job's
    entries in order; parents: [(weight, their history, their parents[, their decay])], applied first, recursively;
    decay: the job's anchor (its spec's). Per element, float64: base, then each parent's rounds times its weight, then
    this job's rounds, pairs in entry order, factors rounded to bf16; float32 once at the end. With an anchor anywhere,
    each job's change is worked out on its own (its parents' times their weights, then per round with an update
    (1 - decay) x change + the round's step) and added to the base at the end. Slow: every pair is an outer product
    over every linear."""
    import torch

    def bf16(x: np.ndarray) -> np.ndarray:
        return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).to(torch.float32).numpy().astype(np.float64)

    total = sum(o + i for _, o, i in layout)

    def anchored(decay_, parents_) -> bool:
        return bool(decay_) or any(anchored(p[3] if len(p) > 3 else 0.0, p[2]) for p in parents_)

    if anchored(decay, parents):
        def change_of(hist, parents_, d) -> dict:
            ch = {n: np.zeros((o, i), np.float64) for n, o, i in layout}
            for p in parents_:
                pc = change_of(p[1], p[2], p[3] if len(p) > 3 else 0.0)
                for n in ch:
                    ch[n] += p[0] * pc[n]
            for ent in hist:
                if not ent["update"]:
                    continue
                if d:
                    for n in ch:
                        ch[n] *= 1.0 - d
                seed = int(ent["seed"], 16)
                for j, c in ent["update"]:
                    z = es.noise(es.pair_seed(seed, j), total)
                    at = 0
                    for n, o, i in layout:
                        ch[n] += (ent["scale"] * c) * np.outer(bf16(z[at:at + o]), bf16(z[at + o:at + o + i]))
                        at += o + i
            return ch
        change = change_of(history, parents, decay)
        return {n: (np.asarray(base[n], np.float64) + change[n]).astype(np.float32) for n, _, _ in layout}
    acc = {n: np.asarray(base[n], np.float64) for n, _, _ in layout}

    def add_history(hist, weight):
        for ent in hist:
            seed = int(ent["seed"], 16)
            for j, c in ent["update"]:
                z = es.noise(es.pair_seed(seed, j), total)
                at = 0
                for n, o, i in layout:
                    acc[n] += (weight * ent["scale"] * c) * np.outer(bf16(z[at:at + o]), bf16(z[at + o:at + o + i]))
                    at += o + i

    def add_job(hist, parents_, weight):
        for w, p_hist, p_parents in parents_:
            add_job(p_hist, p_parents, weight * w)
        add_history(hist, weight)

    add_job(history, parents, 1.0)
    return {n: v.astype(np.float32) for n, v in acc.items()}
