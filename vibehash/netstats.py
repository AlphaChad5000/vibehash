"""What the website's live panel shows, worked out from the chain's own records: who's been mining lately, what's
training and how well. GET /network serves it (node.py); a page can read it from any origin, since all of it is public.

Kept apart from the node so it takes plain data (block summaries as dicts, a job's rounds as stored), which a page or a
script can build from the public API as well.

What the front page reads of a job (docs/site/index.html), beyond the fields job_summary fills in:
- "prep", for a job whose frozen bottom layers are prepared in pieces (DEEPSEEK.md §8): {"layers": how many,
  "done": [the fraction of each layer's pieces accepted, bottom first], "pieces_done", "pieces_total", "miners": who
  ran pieces lately}. The page draws it as the assembly line, and shows "Preparing" until every layer is done.
"""
import collections
import math
import threading
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


CENSUS_BLOCKS = 300  # the records a node reads for the census, at most: a day of ~5-minute rounds


def census(rounds: list[dict], now: float) -> dict:
    """What kinds of machines registered lately, counted and nothing more: no addresses. `rounds` is
    [{"timestamp", "workers": {who: hardware class or None}}] from the day's round records (node._registrations); "who"
    is anything that tells workers apart (the node uses the public key), never shown.

    Returns {"online": workers registered in the last 10 minutes, "day": today, "reported": how many of today's said
    what they run on, "classes": [{"label", "vendor", "arch", "memory_gb", "online", "day"}], most online first,
    "vendors": {vendor: online}}. A worker counts once however many rounds it registered for, under the class it last
    gave."""
    seen: dict = {}  # who -> (class or None, last registration time)
    for rnd in rounds:
        age = now - rnd["timestamp"]
        if age > DAY_S:
            continue
        for who, klass in rnd.get("workers", {}).items():
            last = seen.get(who)
            if last is None or rnd["timestamp"] >= last[1]:
                seen[who] = (klass if isinstance(klass, dict) else None, rnd["timestamp"])
    classes: dict[tuple, dict] = {}
    online = day = reported = 0
    vendors: dict[str, int] = {}
    for klass, t in seen.values():
        is_online = now - t <= ONLINE_S
        day += 1
        online += is_online
        if klass is None:
            continue
        reported += 1
        key = (klass.get("vendor"), klass.get("arch"), klass.get("memory_gb"), klass.get("label"))
        c = classes.setdefault(key, {"label": key[3], "vendor": key[0], "arch": key[1], "memory_gb": key[2], "online": 0, "day": 0})
        c["day"] += 1
        if is_online:
            c["online"] += 1
            vendors[key[0]] = vendors.get(key[0], 0) + 1
    ordered = sorted(classes.values(), key=lambda c: (-c["online"], -c["day"], c["label"], c["memory_gb"]))
    return {"online": online, "day": day, "reported": reported, "classes": ordered, "vendors": vendors}


# --- how far apart honest machines are ---------------------------------------------------------------------------
#
# An Evo checker teacher-forces another worker's answers and finds, for each token, how far below its own greedy choice
# the answer's token is (a gap, in logits: es2.Spec2.margin). Two machines with the same number setup
# (hardware.numerics) give gaps of 0; others give small honest gaps, measured at up to 0.375 between kinds of GPU, and
# a cheat gives 3.5 or more (docs/GPUS.md). Each verdict a checker sends with its worst gap (client.Client) is counted
# here, by the pair of setups: worker's and checker's. Whichever setups' gaps creep toward the margin show up before
# an honest worker on them is turned down: a new driver, PyTorch or card that rounds differently.
#
# It's this node's own view: the verdicts sent to it while it ran rounds (a validator sees the rounds it led), kept in
# memory for a day. Nothing in it names a worker.
GAP_KEEP = 20000  # observations kept at most, newest last
GAP_NEAR = 0.5  # a gap this fraction of its job's margin, or more, counts as near it


class GapCensus:
    def __init__(self, keep: int = GAP_KEEP):
        self._seen = collections.deque(maxlen=keep)  # (time, worker's setup, checker's setup, gap, margin)
        self._lock = threading.Lock()

    def add(self, now: float, worker: str | None, checker: str | None, gap: float, margin: float) -> None:
        with self._lock:
            self._seen.append((now, worker or "unknown", checker or "unknown", float(gap), float(margin)))

    def summary(self, now: float) -> dict:
        """{"checks": how many in the last day, "pairs": [{"worker", "checker", "same", "checks", "max", "p99",
        "near", "over"}]}: per pair of setups, the worst gap, the 99th percentile, how many came within GAP_NEAR of
        their job's margin, and how many went over it. Worst pairs first."""
        with self._lock:
            seen = [s for s in self._seen if now - s[0] <= DAY_S]
        pairs: dict[tuple, list] = {}
        for _, w, c, gap, margin in seen:
            pairs.setdefault((w, c), []).append((gap, margin))
        out = []
        for (w, c), got in pairs.items():
            gaps = sorted(g for g, _ in got)
            out.append({"worker": w, "checker": c, "same": w == c and w != "unknown", "checks": len(gaps),
                        "max": gaps[-1], "p99": gaps[min(len(gaps) - 1, math.ceil(0.99 * len(gaps)) - 1)],
                        "near": sum(g >= GAP_NEAR * m for g, m in got), "over": sum(g > m for g, m in got)})
        out.sort(key=lambda p: (-p["max"], -p["checks"], p["worker"], p["checker"]))
        return {"checks": len(seen), "pairs": out}
