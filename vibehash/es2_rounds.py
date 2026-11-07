"""A rule-2 Evo round's protocol pieces that the node and the workers share, with no model (no PyTorch): the round's
problems, the commitment root, the node's re-score of a claim's answers, the replay targets a round scores members on,
and the job's state rebuilt from the chain's blocks and records alone (es2.py has the arithmetic).

Wire format: a claim's blob is es.claim_bytes(unit_id, answers), as rule 1's; its reveal's `es` is {"answers": the
blob's sha256, "rewards": [[r+, r-] per pair], "lps": [[lp+, lp-] per pair] or null}, rewards each member's mean
right/wrong over the round's problems and lps its mean replay log-probability (null while the job has no targets). The
commitment root is h256(b"es2-claim", answers sha, json(rewards), json(lps)).
"""
from __future__ import annotations

import json
import math

from . import es, es2
from .crypto import h256


def items(spec, round_seed: int, step: int = 0) -> list:
    """The round's problems (es_tasks.Item), drawn from its seed, of the task its step has (es2.task_at); with the
    spec's `lenient`, each checked by es_tasks.lenient (a right answer counts without its tags)."""
    from . import es_tasks
    if getattr(spec, "mix", False):  # problem i of step s: task (s * problems + i) mod the count, its own draw
        n = len(spec.tasks)
        got = [es_tasks.get(spec.tasks[(step * spec.problems + i) % n]).items(mix_seed(round_seed, i), 1)[0]
               for i in range(spec.problems)]
    else:
        got = es_tasks.get(es2.task_at(spec, step)).items(round_seed, spec.problems)
    return [es_tasks.lenient(p) for p in got] if getattr(spec, "lenient", False) else got


def mix_seed(round_seed: int, i: int) -> int:
    """A mixed round's problem i is drawn from its task with this seed: the round's, told apart by the problem."""
    import hashlib
    return int.from_bytes(hashlib.sha256(b"evo2 mix %d %d" % (round_seed, i)).digest()[:8], "big")


def claim_root(answers_sha: str, rewards: list, lps: list | None) -> bytes:
    return h256(b"es2-claim", bytes.fromhex(answers_sha), json.dumps(rewards, separators=(",", ":")).encode(),
                json.dumps(lps, separators=(",", ":")).encode())


def shaped_claim(spec, rewards, lps, want_lps: bool) -> bool:
    """Whether a reveal's numbers are a unit's in shape: a pair of right/wrong scores in [0, 1] for each pair, and a
    pair of finite log-probabilities (at most 0, give or take rounding) for each when the round scores replays."""
    def pairs(v, ok):
        return (isinstance(v, list) and len(v) == spec.pairs_per_unit
                and all(isinstance(p, list) and len(p) == 2 and all(type(x) in (int, float) and ok(x) for x in p)
                        for p in v))
    if not pairs(rewards, lambda x: 0 <= x <= 1):
        return False
    if not want_lps:
        return lps is None
    return pairs(lps, lambda x: math.isfinite(x) and -1e4 <= x <= 1e-3)


def rescore(spec, round_seed: int, raw: bytes, unit_id: str, step: int = 0):
    """(rewards [[r+, r-] per pair], answers [member][problem] token ids, ok [member][problem]) from a claim's blob,
    or None if it isn't a unit's answers in shape: the node's own check of every claim's right/wrong scores."""
    try:
        d = json.loads(raw)
        if not isinstance(d, dict) or d.get("unit") != unit_id:
            return None
        answers = d["answers"]
        if not (isinstance(answers, list) and len(answers) == spec.members
                and all(isinstance(m, list) and len(m) == spec.problems for m in answers)):
            return None
        # No answer has a negative id or none at all (each ends at its end-of-turn token, or has max_tokens): an
        # accepted member's right answer can become a replay target (es2.replay_targets), which every later round
        # scores, and checkers sample only some members, so a [-1] in another one ended every worker's and checker's
        # scoring for good. (Ids past the model's vocabulary are left out of targets instead, by the spec's vocab:
        # a spec that said too few mustn't make honest answers a fraud.)
        for member in answers:
            for a in member:
                if not (isinstance(a, dict) and isinstance(a.get("text"), str) and isinstance(a.get("ids"), list)
                        and 0 < len(a["ids"]) <= spec.max_tokens and all(type(t) is int and t >= 0 for t in a["ids"])):
                    return None
        ps = items(spec, round_seed, step)
        ok = [[bool(p.check(a["text"])) for p, a in zip(ps, member)] for member in answers]
    except (ValueError, KeyError, TypeError, RecursionError):  # (RecursionError: JSON nested too deep)
        return None
    scores = [es.mean_reward([1.0 if x else 0.0 for x in row]) for row in ok]
    return ([[scores[2 * k], scores[2 * k + 1]] for k in range(spec.pairs_per_unit)],
            [[list(a["ids"]) for a in member] for member in answers], ok)


def entry_of(es_record: dict) -> dict:
    """The es2.entry a round's record holds (the hash chain covers exactly it)."""
    return es2.entry(int(es_record["seed"], 16), es_record.get("update") or [], float(es_record.get("scale", 0.0)),
                     es_record.get("targets") or [])


def targets_of(history: list, replay: int) -> list[tuple[int, int, list[int], int]]:
    """The replay targets a round scores members on: the last `replay` of the job's earlier rounds' targets, newest
    last, each (round seed, problem index, answer token ids, that round's step). history: those rounds' entries in
    chain order, from the job's first round (each round is one entry and one step, es2.step_state), so an entry's
    place is its round's step, which says its task (es2.task_at)."""
    out = []
    for step, ent in enumerate(history):
        seed = int(ent["seed"], 16)
        out += [(seed, int(i), list(ids), step) for i, ids in ent["targets"]]
    return out[-replay:] if replay else []


def job_spec_in(block, job_id: str):
    """`job_id`'s spec if `block` brings the job: a post_job, or in the genesis block a starter job (ledger.GENESIS_JOB),
    which no post_job ever names. None otherwise."""
    from .ledger import GENESIS_JOB, SYSTEM
    from .tasks import TrainingJob

    for tx in block.txs:
        if block.height and tx.kind == "post_job":
            memo, owner = tx.memo, tx.sender
        elif not block.height and tx.kind == "genesis" and tx.memo.startswith(GENESIS_JOB):
            memo, owner = tx.memo[len(GENESIS_JOB):], SYSTEM
        else:
            continue
        try:
            job = TrainingJob.from_spec_json(memo, owner, tx.amount)
        except ValueError:
            continue
        if job.job_id == job_id:
            return job.spec
    return None


def rebuild_chain(blocks, record_of, job_id: str, spec=None):
    """A rule-2 job's state from the chain alone: its blocks in order and `record_of(height)`. Every record is checked
    against the record_hash its block commits to and every round's state against the params_hash its block holds
    (ValueError at the first that differs). Returns (state, the job's round entries in order, rounds)."""
    from .ledger import canonical
    from .tasks import TrainingJob

    state, history, rounds = None, [], 0
    for b in blocks:
        s = b.summary
        if s is None or s.job_id != job_id:
            if spec is None:
                spec = job_spec_in(b, job_id)
            continue
        if spec is None:
            raise ValueError(f"no post_job or starter job for {job_id} before block {b.height}")
        if state is None:
            state = spec.init_params()
        record = record_of(b.height)
        if record is None or h256(canonical(record)).hex() != s.record_hash:
            raise ValueError(f"block {b.height}'s record isn't the one it commits to")
        h = record["es"]
        if h.get("rule") != es2.RULE2:
            raise ValueError(f"block {b.height}'s round isn't a rule-2 one")
        if h.get("update"):
            ent = entry_of(h)
            state = es2.step_state(state, ent, float(h["reward"]))
            history.append(ent)
        if es.state_hash(state) != s.params_hash:
            raise ValueError(f"block {b.height}: rebuilt state {es.state_hash(state)[:16]}… but the block says "
                             f"{s.params_hash[:16]}…")
        rounds += 1
    if state is None and spec is not None:
        state = spec.init_params()
    return state, history, rounds
