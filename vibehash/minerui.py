"""The live mining dashboard that `vibehash worker` and `vibehash testnet` show in a terminal.

A drill bores through rock while the worker trains, slows down while it checks other
workers' results, fills its ore cart with every unit it commits, and throws gold sparks
when VIBE lands. Below it: the round and phase, what the worker is doing, its speed, the
work mined this session, earnings and stake, and the latest events.

Plain ANSI escape codes, nothing to install. Only used on a real terminal; `--plain` (or
piping the output) keeps the classic log. NO_COLOR turns the colours off.
"""
from __future__ import annotations

import collections
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass

from .client import NodeClient, NodeError, WorkerClient
from .economics import REP_MAX, REP_MIN
from .tasks import SYN_PER_VIBE

FPS = 12
COLOR = not os.environ.get("NO_COLOR")
RESET = "\x1b[0m" if COLOR else ""


def rgb(r: int, g: int, b: int, bold: bool = False) -> str:
    return f"\x1b[{'1;' if bold else ''}38;2;{r};{g};{b}m" if COLOR else ""


DIM, TEXT, WHITE = rgb(125, 135, 148), rgb(205, 212, 220), rgb(240, 244, 248, True)
GOLD, GREEN, YELLOW, RED = rgb(255, 200, 50, True), rgb(95, 225, 130, True), rgb(240, 200, 80), rgb(240, 95, 85)
STRATA = [(96, 66, 50), (128, 90, 62), (158, 118, 82), (116, 104, 94), (82, 76, 74)]  # earth and stone layers
LIGHT = [1.1, 0.95, 0.78]  # rock brightness in the tunnel: lit by a lantern, near one, between them
ROCK = {(i, lit): rgb(*(min(255, int(v * LIGHT[lit])) for v in STRATA[i])) for i in range(5) for lit in range(3)}
RAIL, TIE, CART, TIMBER = rgb(150, 152, 165), rgb(110, 86, 62), rgb(176, 116, 60), rgb(168, 112, 58)
LAMP = rgb(255, 214, 120, True)
DRILL, DRILL_DIM, TIP = rgb(80, 200, 255, True), rgb(60, 110, 140), rgb(225, 245, 255, True)
SPARKS = [rgb(255, 225, 130, True), rgb(255, 160, 60, True), rgb(255, 250, 225, True), rgb(255, 110, 40)]
PHASES = {"register": rgb(120, 170, 255, True), "compute": rgb(255, 180, 60, True), "reveal": rgb(200, 125, 255, True),
          "verify": rgb(90, 220, 180, True), "idle": DIM}

BANNER = ["█ █ █ █▄▄ █▀▀   █ █ ▄▀█ █▀ █ █",
          "▀▄▀ █ █▄█ ██▄   █▀█ █▀█ ▄█ █▀█"]
GLYPHS = {"shades": "█▓▒", "ore": "♦", "rail": "═", "tie": "╪", "full": "█", "top": "▄", "bottom": "▀",
          "stripe": "▓", "tips": "►»>", "sparks": "*+·'`", "coin": "●", "bar": "█", "empty": "░", "cart": "▐▌",
          "post": "║", "beam": "═", "lamp": "¤"}
ASCII = {"shades": "#%=", "ore": "$", "rail": "=", "tie": "+", "full": "#", "top": "_", "bottom": "~",
         "stripe": "=", "tips": ">>-", "sparks": "*+.'`", "coin": "o", "bar": "#", "empty": "-", "cart": "[]",
         "post": "|", "beam": "=", "lamp": "*"}
BAND, TUNNEL, RAILS = 7, (2, 3, 4), 4  # rows in the mine picture; the drill bores rows 2-4


def _noise(x: int, y: int) -> int:
    n = (x * 374761393 + y * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return n ^ (n >> 16)


def _vibe(syn: int, sign: bool = False) -> str:
    return f"{syn / SYN_PER_VIBE:{'+' if sign else ''},.2f} VIBE"


def _n(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _fit(width: int, *options: str) -> str:
    """The first (longest) option that fits."""
    return next((o for o in options if len(o) <= width), options[-1][:max(0, width)])


def _si(n: float, unit: str) -> str:
    for scale, prefix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= scale:
            return f"{n / scale:.2f} {prefix}{unit}"
    return f"{n:.0f} {unit}"


def _line(segs, width: int) -> str:
    """[(text, style), ...] cut to `width` visible characters."""
    out, used = [], 0
    for text, style in segs:
        if used >= width:
            break
        text = text[:width - used]
        out.append(f"{style}{text}{RESET}" if style else text)
        used += len(text)
    return "".join(out)


def _cells(cells) -> str:
    """[(char, style), ...] with runs of one style merged, so a frame stays small."""
    out, style = [], None
    for ch, st in cells:
        if st != style:
            out.append(RESET + (st or ""))
            style = st
        out.append(ch)
    return "".join(out) + RESET


def enable_ansi() -> bool:
    """Turn on escape-code handling in a Windows console (Windows Terminal has it already)."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        k = ctypes.windll.kernel32
        handle, mode = k.GetStdHandle(-11), ctypes.c_uint32()
        return bool(k.GetConsoleMode(handle, ctypes.byref(mode)) and k.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError):
        return False


def supported(plain: bool = False) -> bool:
    return not plain and sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and enable_ansi()


@dataclass
class Rig:
    name: str
    client: WorkerClient
    start: int | None = None  # balance + stake when the dashboard started
    total: int | None = None
    balance: int = 0
    stake: int = 0
    reputation: float = 0.0

    @property
    def earned(self) -> int:
        return 0 if self.start is None or self.total is None else self.total - self.start


class _Feed(logging.Handler):
    def __init__(self, dash: "MinerDashboard"):
        super().__init__()
        self.dash = dash

    def emit(self, record):
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            msg += f": {record.exc_info[1]}"
        self.dash.note(msg, record.levelno)


class MinerDashboard:
    """One or more workers' live view. `run()` draws until `stop` is set; the workers run on their own threads."""

    def __init__(self, rigs: list[tuple[str, WorkerClient]], node_url: str, footer: str = ""):
        self.rigs = [Rig(name, c) for name, c in rigs]
        self.api, self.node_url, self.footer = NodeClient(node_url), node_url, footer
        self.feed: collections.deque = collections.deque(maxlen=40)
        self.g = GLYPHS if self._can_encode("".join(GLYPHS.values()) + "".join(BANNER)) else ASCII
        self.scroll, self.frame_no = 0.0, 0
        self.popup: tuple[float, int] | None = None  # (time, syn) of the last payout
        self.phase_seen: dict[tuple, float] = {}
        self.chain_id = ""
        self.started = time.time()

    @staticmethod
    def _can_encode(text: str) -> bool:
        try:
            text.encode(sys.stdout.encoding or "ascii")
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    def note(self, text: str, level: int = logging.INFO) -> None:
        self.feed.append((time.time(), level, text))

    # --- background polling: balances arrive over HTTP, which mustn't stall the animation ------------

    def _poll(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.chain_id = self.chain_id or self.api.status()["chain_id"]
            except NodeError:
                pass
            paid = 0
            for rig in self.rigs:
                try:
                    acct = self.api.account(rig.client.keys.address)
                except NodeError:
                    continue
                total = acct["balance"] + acct["stake"]
                if rig.start is None:
                    rig.start = total
                elif rig.total is not None and total != rig.total:
                    if total > rig.total:
                        paid += total - rig.total
                    else:
                        self.note(f"{rig.name} lost {_vibe(rig.total - total)}: slashed", logging.ERROR)
                rig.total, rig.balance, rig.stake, rig.reputation = total, acct["balance"], acct["stake"], acct["reputation"]
            if paid:
                self.popup = (time.time(), paid)
                height = self.rigs[0].client.stats.height
                self.note(f"paid {_vibe(paid, sign=True)} in block {height}" if height else f"paid {_vibe(paid, sign=True)}",
                          logging.INFO + 1)
            stop.wait(1.5)

    # --- drawing -----------------------------------------------------------------------------------------

    def _band(self, w: int) -> list[str]:
        """The mine: layered rock, and a timbered tunnel the drill has bored, with rails and an ore cart."""
        g, f, off = self.g, self.frame_no, int(self.scroll)
        stats = [r.client.stats for r in self.rigs]
        busy = any(s.busy for s in stats)
        dx = max(16, int(w * 0.45))  # the drill tip's column; everything left of it is already mined
        rows = [[(" ", None)] * w for _ in range(BAND)]
        for r in range(BAND):
            for c in range(w):
                x, mined = off + c, c <= dx
                lamp = min((x - 6) % 24, 24 - (x - 6) % 24)  # columns to the nearest lantern
                if mined and r in TUNNEL:
                    if r == RAILS:
                        rows[r][c] = (g["tie"], TIE) if x % 4 == 0 else (g["rail"], RAIL)
                    elif x % 12 == 0:
                        rows[r][c] = (g["post"], TIMBER)  # pit props holding the roof up
                    elif r == TUNNEL[0] and lamp == 0:
                        rows[r][c] = (g["lamp"], LAMP)  # lanterns hung from the roof
                    continue
                if mined and r == TUNNEL[0] - 1 and x % 12 in (11, 0, 1):
                    rows[r][c] = (g["beam"], TIMBER)
                    continue
                v = _noise(x, r) % 100
                if not mined and v < (4 if r in TUNNEL else 1):
                    rows[r][c] = (g["ore"], GOLD)  # ore is only left in rock the drill hasn't reached
                    continue
                layer = (r + _noise(x // 9, 99) % 3 + (_noise(x // 4, r) % 7 == 0)) % 5  # strata that drift and break
                lit = (0 if lamp <= 3 else 1 if lamp <= 7 else 2) if mined else 1  # tunnel rock glows near lanterns
                rows[r][c] = (g["shades"][0 if v < 60 else 1 if v < 88 else 2], ROCK[(layer, lit)])

        def put(r, c, text, style):
            for i, ch in enumerate(text):
                if 0 <= c + i < w:
                    rows[r][c + i] = (ch, style)

        top, mid, low = TUNNEL
        # Ore cart: one nugget per unit committed this round.
        committed = sum(s.committed for s in stats)
        load = (g["ore"] * min(committed, 4)).ljust(4, "·" if g is GLYPHS else ".")
        cx = dx - 13
        put(mid, cx, g["cart"][0], CART)
        put(mid, cx + 1, load[:4], GOLD if committed else DIM)
        put(mid, cx + 5, g["cart"][1], CART)
        put(mid, cx + 6, "───" if g is GLYPHS else "---", RAIL)
        wheel = "oO"[(int(self.scroll) // 2) % 2] if busy else "o"
        put(low, cx + 1, wheel, TEXT)
        put(low, cx + 4, wheel, TEXT)
        # The drill: a body whose stripes turn while it works, and a spinning tip.
        style = DRILL if busy or any(s.round for s in stats) else DRILL_DIM
        put(top, dx - 4, g["top"] * 4, style)
        put(low, dx - 4, g["bottom"] * 4, style)
        body = [g["full"] if (i + (f if busy else 0)) % 3 else g["stripe"] for i in range(4)]
        put(mid, dx - 4, "".join(body), style)
        put(mid, dx, g["tips"][f % 3] if busy else g["tips"][0], TIP if busy else style)
        # Sparks off the rock face, gold coins after a payout.
        paid = self.popup and time.time() - self.popup[0] < 2.5
        if busy or paid:
            for i in range(7 if busy else 4):
                n = _noise(f, i)
                r, c = top - 1 + n % 5, dx + 1 + (n >> 4) % 4
                if paid and i % 2:
                    put(r, c, g["coin"], GOLD)
                else:
                    put(r, c, g["sparks"][(n >> 8) % 5], SPARKS[(n >> 12) % 4])
        if paid:
            label = f" {_vibe(self.popup[1], sign=True)} "
            put(0, max(0, dx - len(label) // 2), label, GOLD)
        return [_cells(row) for row in rows]

    def _phase_bar(self, s, width: int):
        key, now = (s.round, s.phase), time.time()
        first = self.phase_seen.setdefault(key, now)
        left = max(0.0, s.phase_ends - now)
        total = max(s.phase_ends - first, 0.1)
        done = 0.0 if s.phase == "idle" else min(1.0, 1 - left / total)
        n = max(10, min(32, width - 46))
        fill = int(done * n)
        color = PHASES.get(s.phase, DIM)
        if not s.round:
            return [("IDLE", DIM), (f"    block {s.height}", DIM)]
        return [(f"ROUND {s.round} ", WHITE), (f" {s.phase.upper():<8} ", color),
                (self.g["bar"] * fill, color), (self.g["empty"] * (n - fill), DIM),
                (f" {left:4.0f}s left" if s.phase != "idle" else "", TEXT), (f"    block {s.height}", DIM)]

    def frame(self) -> str:
        width, height = shutil.get_terminal_size((100, 32))
        w = max(40, width - 1)
        stats = [r.client.stats for r in self.rigs]
        lead = stats[0]
        busy_training = any(s.busy and s.activity.startswith(("training", "uploading")) for s in stats)
        self.scroll += 1.0 if busy_training else 0.35 if any(s.busy for s in stats) else 0.0
        self.frame_no += 1
        lines: list[str] = []
        add = lambda *segs: lines.append(_line([(" ", None), *segs], w))  # noqa: E731
        room = w - 1 - len(BANNER[0]) - 3
        addr = self.rigs[0].client.keys.address
        tails = [_fit(room, f"useful-work miner  ·  {self.chain_id}", self.chain_id or "useful-work miner"),
                 _fit(room, addr, f"{addr[:10]}{'…' if self.g is GLYPHS else '..'}{addr[-4:]}") if len(self.rigs) == 1
                 else f"{len(self.rigs)} workers"]
        if self.g is GLYPHS:
            grad = [(255, 205, 60), (255, 150, 70), (255, 95, 150), (180, 110, 255)]
            for text, tail in zip(BANNER, tails):
                segs = [(ch, rgb(*grad[min(3, j * 4 // len(text))], True)) for j, ch in enumerate(text)]
                add(*segs, ("   ", None), (tail, DIM))
        else:
            add(("VIBE HASH", WHITE), ("   ", None), (tails[0], DIM))
        lines.append("")
        lines += [" " + row for row in self._band(w - 2)]
        lines.append("")
        add(*self._phase_bar(lead, w))
        mark = (self.g["tips"][self.frame_no % 3] if lead.busy else self.g["tips"][0]) + " "
        add((mark, PHASES.get(lead.phase, DIM)), (lead.activity if len(self.rigs) == 1 else
             f"{sum(s.busy for s in stats)} of {len(stats)} workers busy · {lead.activity}", TEXT))
        lines.append("")
        job = lead.job
        what = f"{job.job_id} ({job.spec.n_params:,} {job.spec.precision} weights)" if job else "no job yet"
        if len(self.rigs) == 1:
            c = self.rigs[0].client
            add(("rig       ", DIM), (c.backend.describe(), WHITE))
            add(("speed     ", DIM), (_si(lead.flops_per_sec, "FLOP/s") if lead.flops_per_sec else "measuring...", WHITE),
                (f"  on {what}", DIM))
            add(("round     ", DIM), (f"{lead.assigned}", WHITE), (" assigned · ", DIM), (f"{lead.trained}", WHITE),
                (" trained · ", DIM), (f"{lead.committed}", WHITE), (" committed · ", DIM), (f"{lead.checked}", WHITE),
                (" checked", DIM))
            add(("session   ", DIM), (_n(lead.units, "unit"), WHITE), (" · ", DIM), (_n(lead.checks, "check"), WHITE),
                (" · ", DIM), (_n(lead.rounds, "round"), WHITE), (" · ", DIM),
                (_si(lead.training_flops, "FLOP"), WHITE), (" trained", DIM))
        else:
            dev = max(8, min(34, w - 2 - 11 - 7 - 8 - 15))  # the device column gives way on narrow terminals
            add((f"{'worker':<11}{'device':<{dev}}{'units':>7}{'checks':>8}{'earned':>15}", DIM))
            for rig in self.rigs:
                s = rig.client.stats
                add((f"{rig.name:<11}", WHITE), (f"{rig.client.backend.describe()[:dev - 2]:<{dev}}", TEXT),
                    (f"{s.units:>7}{s.checks:>8}", WHITE), (f"{_vibe(rig.earned, sign=True):>15}", GREEN if rig.earned else DIM))
            add(("job       ", DIM), (what, TEXT), ("   speed ", DIM), (_si(sum(s.flops_per_sec for s in stats), "FLOP/s"), WHITE))
        lines.append("")
        earned, rig = sum(r.earned for r in self.rigs), self.rigs[0]
        rep = max(0.0, min(1.0, (rig.reputation - REP_MIN) / (REP_MAX - REP_MIN))) if rig.total is not None else 0.0
        if rig.total is None:
            add(("wallet    ", DIM), ("checking balance...", DIM))
        else:
            add(("wallet    ", DIM), (_vibe(sum(r.balance for r in self.rigs)), GOLD), ("   ", None),
                (f"{_vibe(earned, sign=True)} this session", GREEN if earned > 0 else DIM))
            add(("stake     ", DIM), (_vibe(sum(r.stake for r in self.rigs)), WHITE), ("   reputation ", DIM),
                (self.g["bar"] * round(rep * 10), GREEN), (self.g["empty"] * (10 - round(rep * 10)), DIM),
                (f" {rig.reputation:.2f}", TEXT))
        lines.append("")
        room = max(0, height - len(lines) - 3)
        for t, level, text in list(self.feed)[-room:] if room else []:
            style = RED if level >= logging.ERROR else YELLOW if level >= logging.WARNING else GOLD if level > logging.INFO else TEXT
            add((time.strftime("%H:%M:%S  ", time.localtime(t)), DIM), (text, style))
        while len(lines) < height - 2:
            lines.append("")
        add(("Ctrl+C to stop", DIM), (f"   ·   node {self.node_url}", DIM), (f"   ·   {self.footer}" if self.footer else "", DIM))
        frame = "\n".join(line + "\x1b[K" for line in lines[:height - 1])
        if self.g is ASCII:  # a terminal that can't show Unicode must never be sent any, log lines included
            frame = frame.replace("·", "-").encode("ascii", "replace").decode()
        return frame

    def run(self, stop: threading.Event, tick=None, on_start=None) -> None:
        """Draw until `stop` is set or Ctrl+C, calling `tick()` every frame. Worker and node log lines go to the
        event feed meanwhile; start the workers in `on_start()` so their first lines land there too."""
        loggers = [logging.getLogger(n) for n in ("vibehash.worker", "vibehash.node")]
        handler, saved = _Feed(self), [(lg, lg.propagate) for lg in loggers]
        for lg in loggers:
            lg.addHandler(handler)
            lg.propagate = False
        threading.Thread(target=self._poll, args=(stop,), daemon=True, name="dashboard-poll").start()
        out = sys.stdout
        out.write("\x1b[?1049h\x1b[?25l\x1b[2J")  # the alternate screen: the terminal comes back as it was
        try:
            if on_start:
                on_start()
            while not stop.is_set():
                if tick:
                    tick()
                out.write("\x1b[H" + self.frame() + "\x1b[J")
                out.flush()
                time.sleep(1 / FPS)  # interruptible by Ctrl+C on Windows, unlike Event.wait
        finally:
            out.write(f"{RESET}\x1b[?25h\x1b[?1049l")
            out.flush()
            for lg, propagate in saved:
                lg.removeHandler(handler)
                lg.propagate = propagate

    def summary(self) -> str:
        units, checks = sum(r.client.stats.units for r in self.rigs), sum(r.client.stats.checks for r in self.rigs)
        rounds = max(r.client.stats.rounds for r in self.rigs)
        mins = (time.time() - self.started) / 60
        return (f"Mined {units} units and checked {checks} results over {rounds} rounds in {mins:.0f} min; "
                f"earned {_vibe(sum(r.earned for r in self.rigs), sign=True)} this session.")
