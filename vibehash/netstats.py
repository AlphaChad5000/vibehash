"""What the website's live panel shows, worked out from the chain's own records: who's been mining lately, what's
training and how well. GET /network serves it (node.py); a page can read it from any origin, since all of it is public.

Kept apart from the node so it takes plain data (block summaries as dicts, a job's rounds as stored), which a page or a
script can build from the public API as well.

What the front page reads of a job (docs/site/index.html), beyond the fields job_summary fills in:
- "prep", for a job whose frozen bottom layers are prepared in pieces (DEEPSEEK.md §8): {"layers": how many,
  "done": [the fraction of each layer's pieces accepted, bottom first], "pieces_done", "pieces_total", "miners": who
  ran pieces lately}. The page draws it as the assembly line, and shows "Preparing" until every layer is done.
"""
import math
from typing import Callable

ONLINE_S = 600  # a miner whose work was in a block in the last 10 minutes is online
DAY_S = 86400
HISTORY_POINTS = 48  # a job's accuracy curve, thinned to this many points
TOP_MINERS = 5


def _miners(summary: dict) -> set[str]:
    return {*summary.get("worker_flops", {}), *summary.get("verifier_flops", {}), *summary.get("accepted", []),
            *summary.get("missed", [])}


def _number(x) -> float | None:
    """A score as JSON can carry it: a fine-tune round nobody scored has none, and NaN isn't JSON."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def _model(spec: dict) -> str:
    if "base" in spec:  # a fine-tune of a pretrained model
        return f"fine-tune of {spec['base']}"
    m = spec.get("model", {})
    if "n_in" in m and "n_out" in m:
        return f"{m['n_in']} inputs to {m['n_out']} classes"
    return "model"


def _thin(points: list, n: int) -> list:
    """At most n of points, evenly spread, always keeping the first and the last."""
    if len(points) <= n:
        return points
    step = (len(points) - 1) / (n - 1)
    return [points[round(i * step)] for i in range(n)]


def job_summary(job_id: str, spec: dict, escrow: int, rounds: list[dict], current: str | None,
                time_of: Callable[[int], float | None]) -> dict:
    """One job: what it trains, its budget, the rounds that did work, and how its held-back score moved over them:
    accuracy for a model trained from scratch, loss for a fine-tune (whose rounds carry the scorers' median loss)."""
    worked = [r for r in rounds if r["accepted_units"] > 0]
    metric = "loss" if "base" in spec else "accuracy"
    key = "val_loss" if metric == "loss" else "val_acc"
    if metric == "accuracy":
        # Where the model started (the first round scored, before any work) and every round that changed it: the node
        # scores the model a round leaves.
        curve = ([rounds[0]] if rounds and rounds[0] not in worked else []) + worked
    else:
        # A fine-tune's scorers score the model its round started from, so what a round that worked left is the next
        # scored round's: on test night a job's loss went 5.13 to 3.65 after its one unit, and this said 5.13.
        after, scored = [], None
        for r in reversed(rounds):
            if r["accepted_units"] > 0 and scored is not None and (not after or after[-1] is not scored):
                after.append(scored)
            if _number(r[key]) is not None:
                scored = r
        curve = ([rounds[0]] if rounds else []) + after[::-1]
    last = worked[-1] if worked else None
    per_unit = spec.get("batch_size", 0) * spec.get("steps_per_unit", 0)  # examples a unit trains on (with repeats)
    units = sum(r["accepted_units"] for r in worked)
    return {
        "id": job_id, "model": _model(spec), "status": "training" if job_id == current else "open" if escrow > 0 else "finished",
        "escrow": escrow, "fee_per_round": spec.get("fee_per_round"), "rounds_trained": len(worked),
        "units": units, "examples": units * per_unit if isinstance(per_unit, int) and per_unit > 0 else None,
        "last_trained": None if last is None else {"height": last["height"], "time": time_of(last["height"])},
        "metric": metric, "start": _number(curve[0][key]) if curve else None, "now": _number(curve[-1][key]) if curve else None,
        "history": [[r["height"], _number(r[key])] for r in _thin(curve, HISTORY_POINTS)],
    }


def summarize(blocks: list[dict], now: float) -> dict:
    """Miners and work from recent blocks ({"height", "timestamp", "summary"}, oldest first, covering the last day)."""
    online, day, flops, rounds, by_job = set(), set(), {}, {}, {}
    units = work = 0
    for b in blocks:
        s, age = b.get("summary"), now - b["timestamp"]
        if not s or age > DAY_S:
            continue
        who = _miners(s)
        day |= who
        if age <= ONLINE_S:
            online |= who
            by_job.setdefault(s.get("job_id"), set()).update(who)
        units += s.get("accepted_units", 0)
        for f in ("worker_flops", "verifier_flops"):
            for addr, n in s.get(f, {}).items():
                flops[addr] = flops.get(addr, 0) + n
                work += n
        for addr in who:
            rounds[addr] = rounds.get(addr, 0) + 1
    top = sorted(flops, key=lambda a: (-flops[a], a))[:TOP_MINERS]
    times = [b["timestamp"] for b in blocks[-50:]]
    return {
        "online": len(online), "day": len(day), "units_day": units, "flops_day": work,
        "online_by_job": {j: len(who) for j, who in by_job.items() if j},
        "top_miners": [{"address": a, "flops": flops[a], "rounds": rounds[a]} for a in top],
        "block_time": (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
    }
