"""What the terminal dashboards share, so the mine (minerui.py) and the forge (trainui.py) look like one product.

The palette, the rock and lanterns of the mine, the banner, text cut to fit a line, the round and phase bar, the
event feed's colours, and the screen: the alternate screen, which puts the terminal back as it was on exit or Ctrl+C.

Plain ANSI escape codes, nothing to install. NO_COLOR turns the colours off, and a terminal that can't show Unicode
gets pure ASCII. The dashboards only run on a real terminal; `supported()` says whether this is one.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from .tasks import SYN_PER_VIBE

FPS = 12
COLOR = not os.environ.get("NO_COLOR")
RESET = "\x1b[0m" if COLOR else ""


def rgb(r: int, g: int, b: int, bold: bool = False) -> str:
    return f"\x1b[{'1;' if bold else ''}38;2;{r};{g};{b}m" if COLOR else ""


DIM, TEXT, WHITE = rgb(125, 135, 148), rgb(205, 212, 220), rgb(240, 244, 248, True)
GOLD, GREEN, YELLOW, RED = rgb(255, 200, 50, True), rgb(95, 225, 130, True), rgb(240, 200, 80), rgb(240, 95, 85)
STRATA = [(96, 66, 50), (128, 90, 62), (158, 118, 82), (116, 104, 94), (82, 76, 74)]  # earth and stone layers
LIGHT = [1.1, 0.95, 0.78]  # rock brightness: lit by a lantern, near one, between them
ROCK = {(i, lit): rgb(*(min(255, int(v * LIGHT[lit])) for v in STRATA[i])) for i in range(5) for lit in range(3)}
TIMBER, LAMP = rgb(168, 112, 58), rgb(255, 214, 120, True)
SPARKS = [rgb(255, 225, 130, True), rgb(255, 160, 60, True), rgb(255, 250, 225, True), rgb(255, 110, 40)]
PHASES = {"register": rgb(120, 170, 255, True), "compute": rgb(255, 180, 60, True), "reveal": rgb(200, 125, 255, True),
          "verify": rgb(90, 220, 180, True), "idle": DIM}

BANNER = ["█ █ █ █▄▄ █▀▀   █ █ ▄▀█ █▀ █ █",
          "▀▄▀ █ █▄█ ██▄   █▀█ █▀█ ▄█ █▀█"]
BANNER_COLORS = [(255, 205, 60), (255, 150, 70), (255, 95, 150), (180, 110, 255)]  # gold to violet, left to right
# The glyphs both scenes draw with; each dashboard adds its own. ASCII has the same keys, for terminals without Unicode.
GLYPHS = {"shades": "█▓▒", "ore": "♦", "full": "█", "top": "▄", "bottom": "▀", "tips": "►»>", "sparks": "*+·'`",
          "coin": "●", "bar": "█", "empty": "░", "post": "║", "beam": "═", "lamp": "¤", "more": "…"}
ASCII = {"shades": "#%=", "ore": "$", "full": "#", "top": "_", "bottom": "~", "tips": ">>-", "sparks": "*+.'`",
         "coin": "o", "bar": "#", "empty": "-", "post": "|", "beam": "=", "lamp": "*", "more": ".."}


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


# --- the mine's rock, shared by both scenes ---------------------------------------------------------------

def lamp_distance(x: int) -> int:
    """Columns from `x` to the nearest lantern: one hangs every 24 columns."""
    return min((x - 6) % 24, 24 - (x - 6) % 24)


def lit_by(distance: int) -> int:
    """How brightly a lantern `distance` columns away lights the rock: 0 brightest, 2 dark."""
    return 0 if distance <= 3 else 1 if distance <= 7 else 2


def rock(x: int, r: int, g: dict, lit: int = 1, ore: int = 0) -> tuple[str, str]:
    """The rock at column `x`, row `r`: strata that drift and break, with a gold nugget `ore`% of the time."""
    v = _noise(x, r) % 100
    if v < ore:
        return g["ore"], GOLD
    layer = (r + _noise(x // 9, 99) % 3 + (_noise(x // 4, r) % 7 == 0)) % 5
    return g["shades"][0 if v < 60 else 1 if v < 88 else 2], ROCK[(layer, lit)]


def put(rows: list[list], r: int, c: int, text: str, style: str | None) -> None:
    """Write `text` into a grid of (char, style) cells, clipped at its edges."""
    if 0 <= r < len(rows):
        for i, ch in enumerate(text):
            if 0 <= c + i < len(rows[r]):
                rows[r][c + i] = (ch, style)


# --- text around the scene ---------------------------------------------------------------------------------

def can_encode(text: str) -> bool:
    try:
        text.encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def glyphs_for(unicode: dict, ascii: dict) -> dict:
    """`unicode` if this terminal can show every one of its glyphs and the banner, else `ascii`."""
    return unicode if can_encode("".join(unicode.values()) + "".join(BANNER)) else ascii


def header(add, g: dict, ascii: dict, tails: list[str]) -> None:
    """The VIBE HASH banner in its gold-to-violet gradient, each line followed by a tail (dim); one line in ASCII."""
    if g is not ascii:
        for text, tail in zip(BANNER, tails):
            segs = [(ch, rgb(*BANNER_COLORS[min(3, j * 4 // len(text))], True)) for j, ch in enumerate(text)]
            add(*segs, ("   ", None), (tail, DIM))
    else:
        add(("VIBE HASH", WHITE), ("   ", None), (tails[0], DIM))


def phase_bar(g: dict, seen: dict, rnd, phase: str, phase_ends: float, width: int, tail=None):
    """ROUND n, its phase, a bar running down to the phase's end, and `tail` (default: nothing). `seen` remembers
    when each (round, phase) was first shown, which is where its bar starts."""
    key, now = (rnd, phase), time.time()
    first = seen.setdefault(key, now)
    left = max(0.0, phase_ends - now)
    total = max(phase_ends - first, 0.1)
    done = 0.0 if phase == "idle" else min(1.0, 1 - left / total)
    n = max(10, min(32, width - 46))
    fill = int(done * n)
    color = PHASES.get(phase, DIM)
    tail = list(tail or [])
    if not rnd:
        return [("IDLE", DIM), *tail]
    return [(f"ROUND {rnd} ", WHITE), (f" {phase.upper():<8} ", color),
            (g["bar"] * fill, color), (g["empty"] * (n - fill), DIM),
            (f" {left:4.0f}s left" if phase != "idle" else "", TEXT), *tail]


def feed_style(level: int) -> str:
    """Errors red, warnings yellow, good news (a level just above INFO) gold, the rest plain."""
    return RED if level >= logging.ERROR else YELLOW if level >= logging.WARNING else GOLD if level > logging.INFO else TEXT


def feed_lines(add, feed, room: int) -> None:
    """The latest `room` events of a feed of (time, level, text), oldest first."""
    for t, level, text in list(feed)[-room:] if room else []:
        add((time.strftime("%H:%M:%S  ", time.localtime(t)), DIM), (text, feed_style(level)))


def finish(lines: list[str], height: int, ascii_only: bool) -> str:
    """The lines as one frame that fills the screen, each clearing the rest of its row."""
    frame = "\n".join(line + "\x1b[K" for line in lines[:height - 1])
    if ascii_only:  # a terminal that can't show Unicode must never be sent any, log lines included
        frame = frame.replace("·", "-").encode("ascii", "replace").decode()
    return frame


# --- the terminal itself -----------------------------------------------------------------------------------

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


class Feed(logging.Handler):
    """Log records into a dashboard's event feed, through its `note(text, level)`."""

    def __init__(self, dash):
        super().__init__()
        self.dash = dash

    def emit(self, record):
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            msg += f": {record.exc_info[1]}"
        self.dash.note(msg, record.levelno)


@contextlib.contextmanager
def alternate_screen(out=None):
    """Draw on the alternate screen with the cursor hidden; the terminal comes back as it was, on Ctrl+C too."""
    out = out or sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        yield out
    finally:
        out.write(f"{RESET}\x1b[?25h\x1b[?1049l")
        out.flush()


def animate(out, frame, keep_going, tick=None) -> None:
    """Draw `frame()` FPS times a second while `keep_going()`, calling `tick()` before each."""
    while keep_going():
        if tick:
            tick()
        out.write("\x1b[H" + frame() + "\x1b[J")
        out.flush()
        time.sleep(1 / FPS)  # interruptible by Ctrl+C on Windows, unlike Event.wait
