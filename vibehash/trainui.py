"""The live training dashboard that `vibehash train` and `vibehash job watch` show a job's poster.

A forge in the same mine the workers dig: the model is a blade on an anvil, longer and brighter the more accurate it
gets. While the network trains it, the furnace roars and a trip hammer strikes it; when a round is sealed the blade
glows white-hot, grows to its new length, and the change in accuracy flashes above it. While other jobs take their
turn the fire banks down to embers, and once the budget is spent it goes out and the finished blade gleams.

Below it: the network's round and phase and whether this job is the one training, accuracy and loss per round as a
chart that climbs, the units accepted each round, the budget spent and left in escrow with the rounds it still pays
for, and the latest events.

It asks the node about once a second (`/status`, plus `/job/<id>` when something changed), backs off when the node
is busy or unreachable, and says so on screen rather than stopping. The `/job` reply is read defensively: any field
may be missing, and the `model` field isn't used at all. Same conventions as the mining dashboard (termui.py):
plain ANSI, an ASCII fallback, NO_COLOR, the alternate screen; `--plain` or piped output gets log lines instead
(`follow_plain`).
"""
from __future__ import annotations

import collections
import http.client
import logging
import math
import random
import shutil
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import termui
from .client import NodeClient, NodeError
from .termui import (DIM, GOLD, GREEN, LAMP, PHASES, RED, SPARKS, TEXT, TIMBER, WHITE, YELLOW, _cells, _line, _n,
                     _noise, _vibe, rgb, supported)

__all__ = ["JobWatch", "TrainDashboard", "follow_plain", "parse_job", "parse_status", "supported"]

GOOD = logging.INFO + 1  # good news: gold in the feed
POLL_S = 1.5  # between /status reads; the phase bar animates by itself in between
JOB_EVERY_S = 10.0  # between /job reads while /status shows nothing new (a follower node scores rounds a little late)
MISSING_EVERY_S = 3.0  # between /job reads while the node doesn't know the job yet
MISSING_WARN_S = 90.0
BUSY_S = 5.0  # the first wait after the node says it's busy (429 or 503); errors start from POLL_S; both double
BACKOFF_MAX_S = 30.0
TIMEOUT_S = 8.0
LINGER_S = 8.0  # a job that finishes while watched stays on screen this long (and its last round can still land)
HEAT_S = 4.0  # how long a newly sealed round glows

LOSS = rgb(200, 125, 255, True)  # the reveal phase's violet
GOLD_RGB = (255, 200, 50)
IRON, HAMMER, ASH = rgb(138, 146, 162), rgb(182, 188, 202, True), rgb(70, 64, 62)
FIRE = [rgb(*c, bold=i > 2) for i, c in enumerate([(80, 30, 22), (150, 45, 25), (220, 90, 30), (255, 150, 45),
                                                    (255, 210, 105), (255, 245, 210)])]
AREA_RGB = [(38, 112, 80), (52, 142, 96), (68, 172, 110), (82, 200, 122), (95, 225, 130)]  # the chart's rows, bottom up
AREA = [rgb(*c, bold=i == 4) for i, c in enumerate(AREA_RGB)]
STEEL_DULL, STEEL_BRIGHT = (88, 100, 116), (188, 208, 230)
HOT = [(0.3, (190, 55, 30)), (0.55, (250, 120, 40)), (0.8, (255, 195, 80)), (1.0, (255, 245, 210))]  # heat -> colour
DOING = {"register": "workers are signing up", "compute": "workers are training it", "reveal": "workers are "
         "revealing their work", "verify": "their work is being checked", "idle": "its block is being sealed"}

GLYPHS = {**termui.GLYPHS, "hammer": "▐█▌", "loss": "•", "dot": "·", "eighths": " ▁▂▃▄▅▆▇█", "axis": "┤",
          "rule": "│", "arrow": "→"}
ASCII = {**termui.ASCII, "hammer": "[#]", "loss": "o", "dot": ".", "eighths": " ..__==##", "axis": "+",
         "rule": "|", "arrow": "->"}
BAND = 7  # rows in the forge picture: rock above, the chamber in rows 1-4, the floor
HANDLE, BLADE, HAMMER_X = 9, 11, 13  # columns: the blade's handle, where its body starts, the hammer over the anvil


# --- reading the node's replies, trusting none of their fields ---------------------------------------------------

def _num(v, kind=float, default=None):
    if isinstance(v, bool):
        return default
    try:
        x = kind(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return x if kind is not float or math.isfinite(x) else default


@dataclass(frozen=True)
class Round:
    height: int  # the block that sealed it: rounds are numbered by block height
    accepted: int  # units of training that passed their checks
    loss: float | None  # on the job's held-back validation rows
    acc: float | None


@dataclass(frozen=True)
class Job:
    """What `GET /job/<id>` said about a job."""
    fee: int = 0  # syn a round, paid in full when every unit is accepted
    units: int = 0  # units of work a round
    classes: int = 0
    widths: tuple = ()
    escrow: int | None = None  # budget left
    rounds: tuple = ()

    @property
    def scored(self) -> list[Round]:
        return [r for r in self.rounds if r.acc is not None]

    @property
    def spent(self) -> int:
        """Paid out so far, as the ledger works it out: the fee pro rata to the units accepted."""
        return sum(self.fee * min(r.accepted, self.units) // self.units for r in self.rounds) if self.units else 0

    @property
    def rounds_left(self) -> int:
        """About how many more rounds the escrow pays for, if they're all fully accepted."""
        return -(-self.escrow // self.fee) if self.fee and self.escrow else 0

    @property
    def chance(self) -> float:
        """The accuracy of a blind guess: where the blade starts."""
        return 1 / self.classes if self.classes >= 2 else 0.0


def parse_job(reply) -> Job:
    d = reply if isinstance(reply, dict) else {}
    spec = d.get("spec") if isinstance(d.get("spec"), dict) else {}
    m = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    hidden = m.get("n_hidden")
    hidden = hidden if isinstance(hidden, list) else [hidden]
    widths = tuple(w for w in (_num(v, int) for v in [m.get("n_in"), *hidden, m.get("n_out")]) if w and w > 0)
    rounds = {}
    for r in d.get("rounds") if isinstance(d.get("rounds"), list) else []:
        h = _num(r.get("height"), int) if isinstance(r, dict) else None
        if h is not None:
            acc = _num(r.get("val_acc"))
            rounds[h] = Round(h, max(0, _num(r.get("accepted_units"), int, 0)), _num(r.get("val_loss")),
                              None if acc is None else max(0.0, min(1.0, acc)))
    escrow = _num(d.get("escrow"), int)
    return Job(fee=max(0, _num(spec.get("fee_per_round"), int, 0)), units=max(0, _num(spec.get("units_per_round"), int, 0)),
               classes=max(0, _num(m.get("n_out"), int, 0)), widths=widths, escrow=None if escrow is None else max(0, escrow),
               rounds=tuple(rounds[h] for h in sorted(rounds)))


@dataclass(frozen=True)
class Network:
    """What `GET /status` said: the round the network is on, its phase, and which job it trains."""
    height: int | None = None
    round: int | None = None
    phase: str = "idle"
    phase_ends: float = 0.0
    job_id: str | None = None
    chain_id: str = ""


def parse_status(reply) -> Network:
    d = reply if isinstance(reply, dict) else {}
    phase, job = d.get("phase"), d.get("job_id")
    return Network(_num(d.get("height"), int), _num(d.get("round"), int), phase if phase in PHASES else "idle",
                   _num(d.get("phase_ends"), float, 0.0), job if isinstance(job, str) else None,
                   d.get("chain_id")[:64] if isinstance(d.get("chain_id"), str) else "")


_UNSET = object()


class Reader(NodeClient):
    """A NodeClient for watching: one try per request with a short timeout. NodeClient waits and retries by itself
    while a node is busy; here JobWatch does the backing off, so the dashboard can say what's going on meanwhile."""

    def __init__(self, url: str, timeout: float = TIMEOUT_S):
        super().__init__(url, timeout)

    def _open(self, req):
        opener = getattr(self, "_opener", _UNSET)
        if opener is _UNSET:  # a NodeClient that works differently: let it do things its own way
            return super()._open(req)
        return (opener.open if opener else urllib.request.urlopen)(req, timeout=self.timeout)


# --- following a job ---------------------------------------------------------------------------------------------

class JobWatch:
    """One job followed on one node: polled politely, read defensively, and what changes said as events.

    `poll()` makes one round of requests and says how long to wait before the next; `follow()` loops it on a thread
    for the dashboard. Events go to `events`, as (time, level, text), and to each of `listeners`."""

    def __init__(self, job_id: str, node_url: str, budget: int | None = None, posted: bool = False, api=None):
        self.job_id, self.url, self.posted = job_id, node_url.rstrip("/"), posted
        self.api = api if api is not None else Reader(node_url)
        self.budget_hint = budget if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0 else None
        self.job: Job | None = None
        self.net: Network | None = None
        self.missing_since: float | None = None  # when the node first said it has no such job (yet)
        self.error: str | None = None  # why the node isn't answering, while it isn't
        self.busy = False  # ...because it asked us to slow down
        self.fails, self.retry_at, self.heard = 0, 0.0, 0.0
        self.landed: tuple[float, Round, float | None] | None = None  # the newest round: when it landed, its accuracy change
        self.finished_at: float | None = None
        self.was_finished = False  # it was already finished when we started watching
        self.events: collections.deque = collections.deque(maxlen=60)
        self.listeners: list = []
        self._said: set[str] = set()
        self._job_at, self._job_height, self._open_round, self._trouble = 0.0, None, None, None

    def say(self, text: str, level: int = logging.INFO) -> None:
        t = time.time()
        self.events.append((t, level, text))
        for fn in self.listeners:
            fn(t, level, text)

    @property
    def budget(self) -> int | None:
        """The job's whole budget: what `train` posted it with, else what's left plus what its rounds were paid."""
        if self.budget_hint:
            return self.budget_hint
        job = self.job
        return None if job is None or job.escrow is None else job.escrow + job.spent

    def lingering(self, now: float) -> float | None:
        """Seconds before a job that finished while watched leaves the screen; None if it hasn't."""
        if self.finished_at is None or self.was_finished:
            return None
        return max(0.0, LINGER_S - (now - self.finished_at))

    # --- polling ---------------------------------------------------------------------------------------------

    def _get(self, path: str):
        try:
            return self.api.get(path)
        except NodeError:
            raise
        except (ValueError, OSError, http.client.HTTPException) as e:  # not JSON, or the connection broke mid-reply
            raise NodeError(f"GET {path}: unreadable reply ({e})", 502) from None

    def poll(self) -> float:
        """One round of requests: /status, and /job when it's due. Returns the seconds to wait before the next."""
        now = time.time()
        try:
            reply = self._get("/status")
            if self.error is not None:
                self.say("the node is answering again")
            self.error, self.busy, self.fails, self.heard, self._trouble = None, False, 0, now, None
            self._read_status(reply)
            if self._job_due(now):
                try:
                    reply = self._get(f"/job/{urllib.parse.quote(self.job_id, safe='')}")
                except NodeError as e:
                    if e.status != 404 or self.job is not None:
                        raise
                    self._read_missing(now)
                else:
                    self._read_job(reply, now)
                self._job_at, self._job_height = now, self.net.height if self.net else None
        except NodeError as e:
            return self._failed(e, now)
        return POLL_S

    def follow(self, stop: threading.Event) -> None:
        """Poll until `stop` is set. Nothing a node says ends it: a reply that can't be read is waited out."""
        while not stop.is_set():
            try:
                delay = self.poll()
            except Exception as e:  # noqa: BLE001  (the dashboard must outlive any reply)
                self.say(f"couldn't read the node's reply: {e}", logging.WARNING)
                delay = BACKOFF_MAX_S
            stop.wait(delay)

    def _job_due(self, now: float) -> bool:
        if self.job is None:
            return now - self._job_at >= (MISSING_EVERY_S if self.missing_since is not None else 0)
        if self.net is not None and self.net.height != self._job_height:
            return True  # a block landed: maybe one of this job's rounds
        return now - self._job_at >= JOB_EVERY_S

    def _failed(self, e: NodeError, now: float) -> float:
        self.fails += 1
        busy = e.status in (429, 503)
        reason = str(e).split(": ", 1)[-1] if e.status is None else str(e)  # "can't reach node at URL: refused"
        kind = "busy" if busy else "down" if e.status is None else "error"
        if kind != self._trouble:  # said once per kind of trouble, not once per try
            self._trouble = kind
            self.say("the node is busy (rate limited): asking less often" if busy else
                     f"can't reach the node: {reason}; retrying" if e.status is None else
                     f"the node answered {reason}; retrying", logging.WARNING)
        self.error, self.busy = reason, busy
        delay = min(BACKOFF_MAX_S, (BUSY_S if busy else POLL_S) * 2 ** (self.fails - 1)) * random.uniform(1.0, 1.25)
        self.retry_at = now + delay
        return delay

    # --- what changed ----------------------------------------------------------------------------------------

    def _read_status(self, reply) -> None:
        net = parse_status(reply)
        if (net.job_id == self.job_id and net.round and net.round != self._open_round
                and self.finished_at is None and self.job is not None):
            self._open_round = net.round
            self.say(f"round {net.round} open: training {self.job_id} now")
        self.net = net

    def _read_missing(self, now: float) -> None:
        if self.missing_since is None:
            self.missing_since = now
            self.say(f"waiting for {self.job_id} to reach a block" if self.posted else
                     f"the node doesn't know {self.job_id} yet: waiting for it to reach a block")
        elif now - self.missing_since > MISSING_WARN_S and "missing" not in self._said:
            self._said.add("missing")
            self.say(f"still no job called {self.job_id} after {MISSING_WARN_S:.0f}s: check its name, or --node",
                     logging.WARNING)

    def _read_job(self, reply, now: float) -> None:
        job, prev = parse_job(reply), self.job
        self.missing_since = None
        if prev is None:
            if job.escrow != 0:
                acc = job.scored[-1].acc if job.scored else None
                left = job.rounds_left
                self.say(f"{self.job_id} so far: {_n(len(job.rounds), 'round')} trained"
                         + (f", accuracy {acc:.1%}" if acc is not None else "") if job.rounds else
                         f"{self.job_id} is on chain: {_vibe(job.escrow or 0)} budget"
                         + (f", about {_n(left, 'round')} at {_vibe(job.fee)} each" if left else ""))
        else:
            last = prev.rounds[-1].height if prev.rounds else -1
            before = list(prev.rounds)
            for r in job.rounds:
                if r.height > last:
                    self._sealed(r, before, now)
                    before.append(r)
        self.job = job
        self._budget_news(job, now, first=prev is None)

    def _sealed(self, r: Round, before: list[Round], now: float) -> None:
        accs = [x.acc for x in before if x.acc is not None]
        delta = r.acc - accs[-1] if r.acc is not None and accs else None
        best = delta is not None and r.acc > max(accs)
        self.landed = (now, r, delta)
        units = f"{_n(r.accepted, 'unit')} accepted"
        if r.acc is None:
            self.say(f"round {r.height} sealed: {units}")
            return
        change = f" ({delta * 100:+.1f} pts{', best yet' if best else ''})" if delta is not None else ""
        loss = f", loss {r.loss:.4f}" if r.loss is not None else ""
        self.say(f"round {r.height}: accuracy {r.acc:.1%}{change}{loss}, {units}", GOOD if best else logging.INFO)

    def _budget_news(self, job: Job, now: float, first: bool) -> None:
        if job.escrow is None:
            return
        if job.escrow == 0:
            if self.finished_at is None:
                self.finished_at, self.was_finished = now, first
                scored = job.scored
                if not job.rounds:
                    self.say(f"{self.job_id} has no budget left and trained no rounds: it was cancelled", logging.WARNING)
                else:
                    self.say(f"finished: {_n(len(job.rounds), 'round')} trained"
                             + (f", accuracy {scored[-1].acc:.1%} (best {max(r.acc for r in scored):.1%})" if scored else "")
                             + (f"; {_vibe(self.budget)} spent" if self.budget else ""), GOOD)
            return
        budget, low = self.budget or 0, max(2 * job.fee, (self.budget or 0) // 5)
        if "low" not in self._said and job.fee and budget > 3 * job.fee and job.escrow <= low:
            self._said.add("low")
            self.say(f"budget low: {_vibe(job.escrow)} left, about {_n(job.rounds_left, 'more round')}", logging.WARNING)

    def summary(self) -> str:
        """Where the job stands, in a sentence, for after the dashboard closes."""
        job = self.job
        if job is None:
            return (f"{self.job_id} hasn't reached a block yet." if self.missing_since is not None else
                    f"No word from {self.url} about {self.job_id}" + (f" ({self.error})." if self.error else "."))
        scored = job.scored
        acc = (f", accuracy {scored[-1].acc:.1%} on held-back rows (started at {scored[0].acc:.1%}, "
               f"best {max(r.acc for r in scored):.1%})") if scored else ""
        if self.finished_at is not None:
            if not job.rounds:
                return f"{self.job_id} was cancelled before it trained."
            return f"{self.job_id} finished: {_n(len(job.rounds), 'round')} trained{acc}."
        left = job.rounds_left
        return (f"{self.job_id} is still training: {_n(len(job.rounds), 'round')} so far{acc}; "
                f"{_vibe(job.escrow or 0)} of budget left" + (f", about {_n(left, 'more round')}." if left else "."))


def follow_plain(watch: JobWatch, write=None, stop: threading.Event | None = None) -> None:
    """The classic log: a line per event, until the job is finished (or Ctrl+C, or `stop`). For `--plain` and
    output that isn't a terminal."""
    write = write or (lambda text: print(text, flush=True))
    watch.listeners.append(lambda t, level, text: write(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {text}"))
    write(f"following job {watch.job_id} on {watch.url}; Ctrl+C to stop")

    def done() -> bool:
        return (stop is not None and stop.is_set()) or watch.was_finished or watch.lingering(time.time()) == 0

    while not done():
        try:
            delay = watch.poll()
        except Exception as e:  # noqa: BLE001
            watch.say(f"couldn't read the node's reply: {e}", logging.WARNING)
            delay = BACKOFF_MAX_S
        until = time.time() + delay
        while time.time() < until and not done():
            time.sleep(min(0.25, max(0.0, until - time.time())))  # short naps: Ctrl+C lands at once, on Windows too


# --- drawing -----------------------------------------------------------------------------------------------------

def _on(fg: str, back: tuple) -> str:
    """`fg` on a background colour: a loss dot inside the accuracy area, which stays filled around it."""
    return f"{fg}\x1b[48;2;{back[0]};{back[1]};{back[2]}m" if termui.COLOR else fg


def _mix(a, b, t: float) -> tuple:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _metal(base: tuple, heat: float) -> str:
    """Steel at `heat` 0 (cold) to 1 (white-hot), in a few steps so a frame doesn't carry a colour per cell."""
    heat = round(max(0.0, min(1.0, heat)) * 12) / 12
    lo_t, lo_c = 0.0, base
    for t, c in HOT:
        if heat <= t:
            return rgb(*_mix(lo_c, c, (heat - lo_t) / (t - lo_t)), heat > 0.5)
        lo_t, lo_c = t, c
    return rgb(*HOT[-1][1], True)


def _pick(width: int, *variants):
    """The first variant, a list of (text, style), that fits in `width`; else the last, to be cut."""
    return next((v for v in variants if sum(len(t) for t, _ in v) <= width), variants[-1])


def _runs(cells) -> list:
    """[(char, style), ...] as [(text, style), ...], with runs of one style merged: what _line takes."""
    out = []
    for ch, st in cells:
        if out and out[-1][1] == st:
            out[-1] = (out[-1][0] + ch, st)
        else:
            out.append((ch, st))
    return out


def _loss(x: float) -> str:
    return f"{x:.4f}" if x < 10 else f"{x:.1f}" if x < 1e5 else f"{x:.0e}"


class TrainDashboard:
    """A job's live view. `run()` draws until the job finishes (and has lingered), Ctrl+C, or `stop`."""

    def __init__(self, watch: JobWatch):
        self.watch = watch
        self.g = termui.glyphs_for(GLYPHS, ASCII)
        self.frame_no, self.phase_seen = 0, {}
        self.blade = 2.0  # the blade's drawn length, which grows toward its accuracy's

    def state(self, now: float) -> str:
        w = self.watch
        if w.finished_at is not None:
            return "finished"
        if w.error is not None:
            return "offline"
        if w.job is None:
            return "pending"
        net = w.net
        return "training" if net is not None and net.round and net.job_id == w.job_id else "waiting"

    # --- the forge ---------------------------------------------------------------------------------------------

    def _forge(self, width: int, now: float) -> list[str]:
        g, f, watch = self.g, self.frame_no, self.watch
        job, net, state = watch.job, watch.net, self.state(now)
        landed = watch.landed
        heat = max(0.0, 1 - (now - landed[0]) / HEAT_S) if landed else 0.0
        phase = net.phase if net else "idle"
        working = state == "training" and phase != "idle"
        fire = 0.0 if state == "finished" else 1.0 if working else 0.75 if state == "training" else 0.35
        fire = max(fire, 0.8 * heat if state != "finished" else 0.0)
        right = width - 3  # where the chamber ends; its walls are a little ragged
        rows = [[(" ", None)] * width for _ in range(BAND)]
        for r in range(BAND):
            for x in range(width):
                if 1 <= r <= 4 and 1 <= x <= right - (r in (1, 4)) - _noise(r, 7) % 2:
                    if r < 4 and x % 12 == 0 and 24 <= x <= right - 3:
                        rows[r][x] = (g["post"], TIMBER)  # pit props, as in the mine
                    elif r == 1 and x >= 22 and termui.lamp_distance(x) == 0:
                        rows[r][x] = (g["lamp"], LAMP)
                    continue
                if r == 0 and 24 <= x <= right - 2 and x % 12 in (11, 0, 1):
                    rows[r][x] = (g["beam"], TIMBER)
                    continue
                near = termui.lit_by(termui.lamp_distance(x)) if x >= 20 else 1
                lit = 0 if x <= 9 and fire > 0.6 and r in (0, 5) else near if r in (0, 5) else 1  # glow of the fire
                rows[r][x] = termui.rock(x, r, g, lit, ore=1)

        def put(r, c, text, style):
            termui.put(rows, r, c, text, style)

        # The furnace, built into the left wall: a stone hood and base, and the fire between.
        stone = termui.ROCK[(3, 0 if fire > 0.6 else 1)]
        put(1, 1, g["top"] + g["full"] * 4 + g["top"], stone)
        put(4, 1, g["full"] * 6, stone)
        for r in (2, 3):
            put(r, 1, g["full"], stone)
            put(r, 6, g["full"], stone)
            for x in range(2, 6):
                n = _noise(x * 31 + f // 2, r * 17 + f // 2)  # flickers six times a second
                level = fire * (0.55 + 0.45 * (n % 100) / 100) * (0.8 if r == 2 else 1.0)
                put(r, x, g["shades"][(n >> 8) % 3], FIRE[min(5, int(level * 6))] if fire else ASH)
        # The anvil: a face (and horn) under the blade, on a block standing on the floor.
        put(4, 8, g["bottom"] * 3 + g["full"] * 5 + g["bottom"] * 2, IRON)
        # The hammer strikes twice a second while workers train the job, every second in its other phases.
        period = 6 if phase == "compute" else 12
        down = working and f % period < 2
        head = 2 if down else 1
        for r in range(head):
            put(r, HAMMER_X, g["post"], TIMBER)
        put(head, HAMMER_X - 1, g["hammer"], HAMMER)
        # The blade: as long as the model is accurate, from a blind guess (short) to every row right (full length).
        scored = job.scored if job else []
        tip = None
        if job is not None:
            acc = max(0.0, min(1.0, scored[-1].acc)) if scored else None
            span = max(1e-9, 1 - job.chance)
            frac = 0.12 + 0.88 * max(0.0, min(1.0, (acc - job.chance) / span)) if acc is not None else 0.0
            longest = max(4, min(60, width - BLADE - 12))
            target = max(2.0, frac * longest)
            step = target - self.blade
            self.blade = target if abs(step) < 0.5 else self.blade + step * 0.25  # grows to its new length
            length = round(self.blade)
            base = _mix(STEEL_DULL, STEEL_BRIGHT, frac)
            glint = f % (length + 30) if state == "finished" else -9
            put(3, HANDLE, g["beam"] * 2, TIMBER)
            for i in range(length + (acc is not None)):  # an untrained lump has no point yet
                x = BLADE + i
                worked = 0.7 * max(0.0, 1 - abs(x - HAMMER_X) / 6) * (1.0 if down else 0.8) if working else 0.0
                cell = g["tips"][0] if i == length else g["full"]
                style = rgb(240, 248, 255, True) if i in (glint, glint - 1) else _metal(base, max(heat, worked))
                put(3, x, cell, style)
            tip = BLADE + length
            if acc is not None:
                put(3, tip + 2, f"{acc:.1%}", GOLD if state == "finished" or heat > 0.3 else WHITE)
            else:
                put(3, tip + 2, "untrained", DIM)
        # Sparks off the hammer while it works, and off the blade as a round lands.
        if down or (working and f % period == 2):
            for i in range(6):
                n = _noise(f, i)
                r, c = 1 + n % 2, HAMMER_X - 4 + (n >> 4) % 9
                if not (r == head and HAMMER_X - 1 <= c <= HAMMER_X + 1):
                    put(r, c, g["sparks"][(n >> 8) % 5], SPARKS[(n >> 12) % 4])
        if heat > 0.5 and tip is not None:
            for i in range(5):
                n = _noise(f, 40 + i)
                put(1 + (n >> 5) % 2, BLADE + n % max(1, tip - BLADE + 1), g["sparks"][(n >> 8) % 5], SPARKS[(n >> 12) % 4])
        # What the round did, above the blade's point.
        if landed and now - landed[0] < 3 and tip is not None:
            _, r, delta = landed
            if delta is not None:
                label, style = f" {delta * 100:+.1f} pts ", GREEN if delta > 0 else RED if delta < 0 else TEXT
            else:
                label, style = (f" {r.acc:.1%} " if r.acc is not None else " sealed "), GOLD
            put(0, max(HAMMER_X + 3, min(width - len(label) - 1, tip - len(label) // 2)), label, style)
        return [_cells(row) for row in rows]

    # --- the chart ---------------------------------------------------------------------------------------------

    def _chart(self, plot: int, height: int, now: float) -> list[list]:
        """Accuracy as a green area that climbs, loss as violet dots, one round per column (or several columns per
        round when there are few), scaled to fill the chart; then the units accepted each round, and the rounds."""
        g, watch = self.g, self.watch
        job = watch.job
        scored = job.scored if job else []
        future = job.rounds_left if job and watch.finished_at is None else 0
        per = max(1, min(12, plot // max(1, len(scored) + future)))
        shown = scored[-max(1, plot // per):] if scored else []
        grid = [[(" ", None)] * plot for _ in range(height)]
        for s in range(len(shown), min(len(shown) + future, plot // per)):  # a dot for each round the budget pays for
            grid[-1][s * per + per - 1] = (g["dot"], DIM)
        left, right = [" " * 5 + g["rule"]] * height, [""] * height
        if shown:
            accs = [r.acc for r in shown]
            lo = max(0.0, math.floor(min(accs) * 10 - 0.5) / 10)
            hi = min(1.0, math.ceil(max(accs) * 10 + 0.5) / 10)
            hi = max(hi, lo + 0.1)
            losses = [r.loss for r in shown if r.loss is not None]
            llo, lhi = (min(losses), max(losses)) if losses else (0.0, 1.0)
            flash = watch.landed and now - watch.landed[0] < HEAT_S and watch.landed[1].height == shown[-1].height
            full = [[None] * plot for _ in range(height)]  # the colour of each completely filled cell
            for i, r in enumerate(shown):
                prev = shown[i - 1].acc if i else r.acc
                for j in range(per):
                    v = prev + (r.acc - prev) * (j + 1) / per  # a slope between rounds, not steps
                    eighths = max(1, min(height * 8, round((v - lo) / (hi - lo) * height * 8)))
                    top = -(-eighths // 8)
                    for k in range(top):
                        part = min(8, eighths - 8 * k)
                        back = AREA_RGB[k * len(AREA_RGB) // height]
                        if flash and i == len(shown) - 1:  # the round just sealed: bright, with a gold cap
                            back = GOLD_RGB if k == top - 1 else AREA_RGB[-1]
                        grid[height - 1 - k][i * per + j] = (g["eighths"][part], rgb(*back, bold=back == AREA_RGB[-1]))
                        full[height - 1 - k][i * per + j] = back if part == 8 else None
                if r.loss is not None:
                    k = round((r.loss - llo) / (lhi - llo) * (height - 1)) if lhi > llo else height // 2
                    back = full[height - 1 - k][i * per + per - 1]
                    grid[height - 1 - k][i * per + per - 1] = (g["loss"], _on(LOSS, back) if back else LOSS)
            left[0], left[-1] = f"{hi:>4.0%} {g['axis']}", f"{lo:>4.0%} {g['axis']}"
            if losses:
                right[0], right[-1] = f" {_loss(lhi)}", f" {_loss(llo)}"
        else:
            note = "the first round's score lands here" if job is not None else "waiting for the job"
            note = note if len(note) <= plot else note[:plot]
            for i, ch in enumerate(note):
                grid[(height - 1) // 2][(plot - len(note)) // 2 + i] = (ch, DIM)
        lines = [[(left[r], DIM), *_runs(grid[r]), (right[r], LOSS)] for r in range(height)]
        # Units accepted per round, under the rounds that took them; then where the rounds start and end.
        cap = max([job.units] + [r.accepted for r in shown]) if job else 1
        units = [(" ", None)] * plot
        for i, r in enumerate(shown):
            bar = g["eighths"][max(1 if r.accepted else 0, round(8 * r.accepted / max(1, cap)))]
            for j in range(per - (per >= 3)):
                units[i * per + j] = (bar, GOLD)
        last = f" {shown[-1].accepted}/{job.units}" if shown and job.units else ""
        lines.append([("units ", DIM), *_runs(units), (last, DIM)])
        axis = ""
        if shown:
            first, end = f"round {shown[0].height}", f"{shown[-1].height}"
            axis = first.ljust(len(shown) * per - len(end)) + end if len(shown) * per >= len(first) + len(end) + 2 else first
        lines.append([(" " * 6, None), (axis, DIM)])
        return lines

    # --- the frame -----------------------------------------------------------------------------------------------

    def frame(self) -> str:
        width, height = shutil.get_terminal_size((100, 32))
        w, now = max(40, width - 1), time.time()
        g, watch = self.g, self.watch
        job, net, state = watch.job, watch.net, self.state(now)
        self.frame_no += 1
        lines: list[str] = []
        add = lambda *segs: lines.append(_line([(" ", None), *segs], w))  # noqa: E731
        content = w - 1
        room = w - 1 - len(termui.BANNER[0]) - 3
        shape = f" {g['arrow']} ".join(str(x) for x in job.widths) if job and job.widths else ""
        chain = net.chain_id if net else ""
        tails = [termui._fit(room, f"model forge  ·  {watch.job_id}", watch.job_id),
                 termui._fit(room, f"{shape}  ·  {chain}" if shape and chain else shape or chain, shape, chain)]
        termui.header(add, g, ASCII, tails)
        head = len(lines)
        # Fixed rows: header, forge, two status lines, the headline, units and axis rows, two budget lines, footer.
        spare = height - 1 - (head + BAND + 2 + 3 + 2 + 1)
        gaps = spare >= 8  # blank lines between sections, when there's room
        spare -= 5 * gaps
        rows = max(3, min(10, height // 5, spare - 3))
        gap = (lambda: lines.append("")) if gaps else (lambda: None)  # noqa: E731

        gap()
        lines += [" " + row for row in self._forge(w - 2, now)]
        gap()
        # The network: its round and phase, and whose turn it is.
        mine = net is not None and net.round and net.job_id == watch.job_id
        if watch.error is not None:
            add(("NODE BUSY" if watch.busy else "NODE UNREACHABLE", YELLOW if watch.busy else RED),
                (f"   retrying in {max(0.0, watch.retry_at - now):.0f}s", DIM),
                (f"   last at block {net.height}" if net and net.height is not None else "", DIM))
        elif net is None:
            add(("CONNECTING", DIM), (f"   {watch.url}", DIM))
        else:
            tail = ([(f"    block {net.height}", DIM)] if not net.round or state == "finished" else
                    [("    your job's turn", GREEN)] if mine else [("    another job's turn", DIM)])
            add(*termui.phase_bar(g, self.phase_seen, net.round, net.phase, net.phase_ends,
                                  w - max(0, len(tail[0][0]) - 12), tail))
        mark = (g["tips"][self.frame_no % 3] if state == "training" else g["tips"][0]) + " "
        add((mark, PHASES.get(net.phase, DIM) if mine else DIM), (self._doing(state, now), TEXT))
        gap()
        # Accuracy and loss, then the chart.
        if job and job.scored:
            last, best = job.scored[-1], max(r.acc for r in job.scored)
            delta = last.acc - job.scored[-2].acc if len(job.scored) > 1 else None
            change = ([(f" {delta * 100:+.1f} pts", GREEN if delta > 0 else RED if delta < 0 else DIM)]
                      if delta is not None else [])
            acc = [(g["eighths"][6] + " ", AREA[-1]), ("accuracy ", DIM), (f"{last.acc:.1%}", WHITE), *change]
            loss = [("   " + g["loss"] + " ", LOSS), ("loss ", DIM), (_loss(last.loss), TEXT)] if last.loss is not None else []
            add(*_pick(content, [*acc, ("  best ", DIM), (f"{best:.1%}", TEXT), *loss, (f"   after round {last.height}", DIM)],
                       [*acc, ("  best ", DIM), (f"{best:.1%}", TEXT), *loss], [*acc, *loss], acc))
        else:
            add((g["eighths"][6] + " ", AREA[-1]), ("accuracy -", DIM), ("   " + g["loss"] + " ", LOSS), ("loss -", DIM),
                ("   no rounds yet" if job is not None else "", DIM))
        chart = self._chart(max(10, content - 13), rows, now)
        for segs in chart[:-1] + ([chart[-1]] if spare - 3 >= rows else []):  # the axis row, when there's room
            add(*segs)
        gap()
        # The money: escrow spent and left, and the rounds it still pays for.
        budget = watch.budget
        if job is None or job.escrow is None or not budget:
            add(("budget    ", DIM), ("waiting for the node..." if job is None else "unknown", DIM))
            add(("rounds    ", DIM), ("-", DIM))
        else:
            spent = max(0, budget - job.escrow)
            words = [(_vibe(spent)[:-5], GOLD), (" spent", DIM), (" · ", DIM), (_vibe(job.escrow), WHITE), (" left", DIM)]
            n = max(8, min(24, content - 12 - sum(len(t) for t, _ in words)))  # the bar takes the room the words leave
            fill = round(n * spent / budget)
            bar = [(g["bar"] * fill, GOLD), (g["empty"] * (n - fill), DIM), ("  ", None)]
            add(*_pick(content, [("budget    ", DIM), *bar, *words],
                       [("budget    ", DIM), (_vibe(job.escrow), WHITE), (f" left of {_vibe(budget)[:-5]}", DIM)]))
            trained = [(f"{len(job.rounds)}", WHITE), (" trained", DIM)]
            units = sum(r.accepted for r in job.rounds)
            if watch.finished_at is not None:
                add(*_pick(content, [("rounds    ", DIM), *trained, (" · ", DIM), (f"{units:,}", WHITE),
                                     (" units accepted · budget spent", DIM)], [("rounds    ", DIM), *trained]))
            else:
                more = [(" · about ", DIM), (f"{job.rounds_left}", WHITE), (" more", DIM)]
                add(*_pick(content, [("rounds    ", DIM), *trained, *more, (f" at {_vibe(job.fee)} each", DIM), (" · ", DIM),
                                     (f"{units:,}", WHITE), (" units accepted", DIM)],
                           [("rounds    ", DIM), *trained, *more, (f" at {_vibe(job.fee)} each", DIM)],
                           [("rounds    ", DIM), *trained, *more]))
        gap()
        termui.feed_lines(add, watch.events, max(0, height - 2 - len(lines)))
        while len(lines) < height - 2:
            lines.append("")
        lingering = watch.lingering(now)
        stop = ([(f"finished · closing in {lingering:.0f}s", GOLD), ("   ·   Ctrl+C to close now", DIM)]
                if lingering is not None else [("finished · Ctrl+C to close", DIM)] if state == "finished"
                else [("Ctrl+C to stop watching", DIM)])
        quiet = now - watch.heard if watch.heard and watch.error is None else 0
        add(*stop, (f"   ·   node {watch.url}", DIM), (f"   ·   last heard from {quiet:.0f}s ago" if quiet > 6 else "", YELLOW))
        return termui.finish(lines, height, g is ASCII)

    def _doing(self, state: str, now: float) -> str:
        w, net = self.watch, self.watch.net
        if state == "finished":
            job = w.job
            return (f"finished: {_n(len(job.rounds), 'round')} trained, the budget is spent" if job and job.rounds
                    else "cancelled: its budget went back to its owner")
        if state == "offline":
            return ("the node asked for fewer requests; waiting before the next" if w.busy
                    else f"can't reach the node: {w.error}")
        if state == "pending":
            return (f"waiting for {w.job_id} to reach a block" if w.missing_since is not None
                    else f"asking the node about {w.job_id}...")
        if state == "training":
            return f"round {net.round} is training {w.job_id}: {DOING.get(net.phase, net.phase)}"
        if net is not None and net.round:
            return f"round {net.round} trains {net.job_id or 'another job'}; {w.job_id} waits its turn"
        return "waiting for the next round to open"

    def run(self, stop: threading.Event | None = None) -> None:
        """Draw until the job finishes (after lingering), `stop` is set, or Ctrl+C. The node is polled on a thread
        meanwhile, so a slow reply never stalls the picture; the terminal is put back as it was however it ends."""
        stop = stop or threading.Event()
        threading.Thread(target=self.watch.follow, args=(stop,), daemon=True, name="job-watch").start()
        try:
            with termui.alternate_screen() as out:
                termui.animate(out, self.frame, lambda: not stop.is_set() and self.watch.lingering(time.time()) != 0)
        finally:
            stop.set()
