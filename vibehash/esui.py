"""The Evo scene: what the dashboards show for an Evo job (es.py) in place of the drill (minerui.py) and the forge
(trainui.py). Every other job keeps those.

A row of blocks joined by chain links, the sealed ones on the left and the open one, the round being worked, on the
right. Something incubates in the open block as the round goes through its phases: workers sign up and the eggs are
laid, one twin pair for each antithetic pair (the +σε and −σε members); in compute they grow as the round's answers
come in; in reveal their shells crack as the scores are revealed; in verify a checker's glint passes over them. When
the block is sealed they hatch, the fitter twin of each accepted pair glowing, and the block locks into the chain with
a link while a new empty one opens. Each sealed block keeps a trace of what grew in it, one block a generation, so the
chain reads as the model's history: a hatchling as big and bright as its generation's mean reward, a bar of that
reward, and under the block its generation, its difficulty level (an arrow when it moved) and the bytes it put on the
chain, the seeds and scores a rebuild works the model out from.

`Lineage` follows a job's generations from what any node serves: its blocks, and each round's record (the ES entry:
seed, level, the accepted pairs' scores). Read defensively, since a node could send anything. Same conventions as the
other dashboards (termui.py): plain ANSI in a green and teal palette, a pure-ASCII fallback, NO_COLOR.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from . import es, termui
from .es_local import chain_bytes
from .termui import DIM, GOLD, GREEN, PHASES, TEXT, WHITE, YELLOW, _cells, _noise, rgb

GOOD = logging.INFO + 1  # good news: gold in a feed
HATCH_S = 3.2  # how long a sealed generation hatches in its block before it's one of the chain's
LOCK_S = 1.9  # when, into that, the block locks into the chain
BAND = 7  # rows in the scene: the open block is rows 0-5 and its label row 6; sealed blocks rows 1-4, labels 5-6
BLOCK, LINK = 8, 3  # a sealed block's width, and a link's
MAX_ROW = 16  # twin pairs on one row of eggs at most; two rows at most
MIN_INNER = 15  # the open block's narrowest inside

TEAL, TEAL_DIM, MINT = rgb(64, 200, 186, True), rgb(40, 116, 110), rgb(160, 246, 214, True)
LEAF, GLOW, LINKS = rgb(70, 170, 110), rgb(215, 255, 160, True), rgb(92, 172, 160)
PLUS, MINUS = rgb(186, 238, 228), rgb(206, 236, 186)  # the twins' shells: + a little teal, − a little green
PLUS_DIM, MINUS_DIM = rgb(92, 128, 122), rgb(104, 128, 94)
NEST, DUST = rgb(118, 126, 72), rgb(40, 70, 66)
FIT_RGB = [(70, 96, 90), (56, 150, 122), (76, 200, 142), (150, 236, 150), (215, 255, 160)]  # a hatchling, by reward
FIT = [rgb(*c, bold=i >= 3) for i, c in enumerate(FIT_RGB)]

# A box's corners and sides, as "top-left, across, top-right, down, bottom-left, bottom-right, link in from the left,
# link out to the right".
GLYPHS = {**termui.GLYPHS, "sealed": "╔═╗║╚╝╣╠", "open": "┌─┐│└┘╡│", "chain": "═", "link": "∞", "eggs": "·∙o○●",
          "cracked": "Θ", "hatched": "☻", "twin": "☺", "lost": "·", "heat": "≈", "creatures": "·∙o☺☻", "half": "▌",
          "nest": "▀", "up": "▲", "down": "▼", "eighths": " ▁▂▃▄▅▆▇█", "ops": {"+": "+", "-": "−", "*": "×", "/": "÷"}}
ASCII = {**termui.ASCII, "sealed": "+=+|++++", "open": "+-+|+++|", "chain": "=", "link": "o", "eggs": "..oO0",
         "cracked": "Q", "hatched": "@", "twin": "o", "lost": ".", "heat": "~", "creatures": "..oO@", "half": "+",
         "nest": "~", "up": "^", "down": "v", "eighths": " ..__==##", "ops": {"+": "+", "-": "-", "*": "*", "/": "/"}}


def glyphs() -> dict:
    """GLYPHS if this terminal can show them all, else ASCII."""
    flat = {k: "".join(v.values()) if isinstance(v, dict) else v for k, v in GLYPHS.items()}
    return GLYPHS if termui.can_encode("".join(flat.values()) + "".join(termui.BANNER)) else ASCII


# --- words -------------------------------------------------------------------------------------------------------

def size(n: int) -> str:
    return f"{n:,} B" if n < 1000 else f"{n / 1000:.1f} KB" if n < 10**6 else f"{n / 10**6:.1f} MB"


def level_words(level: int | None, g: dict) -> str:
    """A Countdown level (es.LEVELS) in words: "3 numbers 1-20 with + − ×"."""
    if level is None or not 0 <= level < len(es.LEVELS):
        return "level unknown"
    lv = es.LEVELS[level]
    return f"{lv.count} numbers 1-{lv.top} with {' '.join(g['ops'].get(o, o) for o in lv.ops)}"


def pick(width: int, *variants):
    """The first variant, a list of (text, style), that fits in `width`; else the last, to be cut."""
    return next((v for v in variants if sum(len(t) for t, _ in v) <= width), variants[-1])


# --- a job's generations, from the node ----------------------------------------------------------------------------

def _int(v, lo=0, hi=None):
    return v if type(v) is int and v >= lo and (hi is None or v <= hi) else None


def _score(v):
    return float(v) if type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 else None


@dataclass(frozen=True)
class Generation:
    """One sealed round of an Evo job: a generation of its population."""
    number: int  # 1 for the job's first round
    height: int  # the block that sealed it
    level: int | None  # the Countdown level its puzzles were at
    update: tuple  # ((pair, reward of the + twin, of the − twin), ...): its accepted pairs, as its record keeps them
    bytes: int  # what it put on the chain to train the model (es_local.chain_bytes)

    @property
    def pairs(self) -> int:
        return len(self.update)

    @property
    def mean(self) -> float | None:
        """The accepted members' mean reward (the record's score), or None when none was accepted."""
        return es.mean_reward([x for _, rp, rm in self.update for x in (rp, rm)]) if self.update else None


def parse_generation(number: int, height: int, record) -> Generation:
    entry = record.get("es") if isinstance(record, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    update = []
    rows = entry.get("update") if isinstance(entry.get("update"), list) else []
    if rows and all(isinstance(u, list) and len(u) == 2 for u in rows):
        # Evo v2 (es2.entry): [pair, coefficient] rows; the generation's mean right/wrong reward is the record's score.
        sc = record.get("score") if isinstance(record, dict) else None
        m = _score(sc[1]) if isinstance(sc, list) and len(sc) == 2 else None
        update = [(u[0], m, m) for u in rows if m is not None and _int(u[0], 0, 4095) is not None]
    for u in rows:
        if isinstance(u, list) and len(u) == 3 and _int(u[0], 0, 4095) is not None:
            rp, rm = _score(u[1]), _score(u[2])
            if rp is not None and rm is not None:
                update.append((u[0], rp, rm))
    return Generation(number, height, _int(entry.get("level"), 0, len(es.LEVELS) - 1), tuple(update[:4096]),
                      chain_bytes(entry))


class Lineage:
    """An Evo job's generations, followed block by block from a node: `poll(api, tip)` reads the blocks after the last
    one it read, up to `tip`, and each of the job's rounds' records, and returns the generations it found. The newest
    to land while followed (not the ones found catching up at the start) is `landed`, with when."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.gens: list[Generation] = []
        self.height = 0  # the last block read
        self.caught_up = False
        self.landed: tuple[float, Generation] | None = None

    def poll(self, api, tip) -> list[Generation]:
        new = []
        tip = _int(tip)
        while tip is not None and self.height < tip:
            batch = api.get(f"/blocks?from={self.height + 1}&to={min(tip, self.height + 100)}")
            heights = []
            for d in batch if isinstance(batch, list) else []:
                h = _int(d.get("height") if isinstance(d, dict) else None, self.height + 1, tip)
                if h is None or (heights and h <= heights[-1]):
                    continue
                s = d.get("summary")
                if isinstance(s, dict) and s.get("job_id") == self.job_id:
                    try:
                        record = api.get(f"/round/{h}/record")
                    except Exception as e:  # noqa: BLE001 - a node without the record still has the block
                        if getattr(e, "status", None) != 404:
                            raise
                        record = None
                    gen = parse_generation(len(self.gens) + 1, h, record)
                    self.gens.append(gen)
                    new.append(gen)
                heights.append(h)
                self.height = h
            if not heights:
                break
        if new and self.caught_up:
            self.landed = (time.time(), new[-1])
        self.caught_up = self.caught_up or (tip is not None and self.height >= tip)
        return new

    @property
    def bytes(self) -> int:
        return sum(x.bytes for x in self.gens)

    def best(self) -> float | None:
        means = [x.mean for x in self.gens if x.mean is not None]
        return max(means) if means else None

    def news(self, gen: Generation, g: dict) -> list[tuple[str, int]]:
        """What a newly sealed generation says in a feed: its reward, and the level when it moved."""
        before = [x for x in self.gens if x.number < gen.number]
        last = next((x.mean for x in reversed(before) if x.mean is not None), None)
        best = max([x.mean for x in before if x.mean is not None], default=None)
        if gen.mean is None:
            out = [(f"generation {gen.number} sealed in block {gen.height}: no pair accepted, the model stays as it was",
                    logging.INFO)]
        else:
            change = f" ({(gen.mean - last) * 100:+.1f} pts)" if last is not None else ""
            top = best is not None and gen.mean > best
            out = [(f"generation {gen.number} hatched in block {gen.height}: reward {gen.mean:.1%}{change}"
                    f"{', best yet' if top else ''}, {gen.pairs} pairs, {size(gen.bytes)} on chain", GOOD if top else logging.INFO)]
        prev = before[-1].level if before else None
        if gen.level is not None and prev is not None and gen.level != prev:
            out.append((f"difficulty {'up' if gen.level > prev else 'down'} to level {gen.level}: "
                        f"{level_words(gen.level, g)}", GOOD if gen.level > prev else logging.WARNING))
        return out


# --- the scene ---------------------------------------------------------------------------------------------------

def creature(mean: float | None) -> int:
    """How big a generation's hatchling is: 0 (none accepted) to 4, by its mean reward."""
    if mean is None:
        return 0
    return 1 if mean < 0.1 else 2 if mean < 0.3 else 3 if mean < 0.55 else 4


def _fraction(progress, frac: float) -> float:
    if progress is not None and progress[1]:
        return max(0.0, min(1.0, progress[0] / progress[1]))
    return max(0.0, min(1.0, frac))


def scene(g: dict, width: int, gens: list[Generation], now: float, frame: int, *, number: int | None = None,
          phase: str = "idle", progress=None, frac: float = 0.0, pairs: int = 8,
          landed: tuple[float, Generation] | None = None, busy: bool = False, waiting: str = "") -> list[str]:
    """BAND rows, `width` columns each. `gens`: the sealed generations, oldest first. The open block holds round
    `number`'s population of `pairs` twin pairs (None: no round of this job is open, so it lies dormant and says
    `waiting`), in the network's `phase` ("next": the round's block is sealed and this one is new and empty),
    `progress` (in, of) of what it waits for (else `frac` of its time gone).
    `landed`: the newest generation and when it was sealed, which hatches in the open block for HATCH_S seconds and
    then takes its place in the chain."""
    rows = [[(" ", None)] * width for _ in range(BAND)]

    def put(r, c, text, style):
        termui.put(rows, r, c, text, style)

    hatch_t = now - landed[0] if landed and gens and gens[-1] is landed[1] else None
    hatching = hatch_t is not None and 0 <= hatch_t < HATCH_S
    locking = hatching and hatch_t >= LOCK_S
    sealed = gens[:-1] if hatching else gens
    pairs = max(1, pairs)
    # The open block: every pair on one row of eggs, a nest under them, when two sealed blocks still fit beside it.
    one = max(MIN_INNER, 3 * min(pairs, MAX_ROW) + 1)
    two = max(MIN_INNER, 3 * -(-min(pairs, 2 * MAX_ROW) // 2) + 1)
    inner = one if width - one - 3 - LINK >= 2 * (BLOCK + LINK) else two
    inner = max(5, min(inner, width - 3))
    per_row = max(1, (inner - 1) // 3)
    shown = min(pairs, 2 * per_row) if inner == two or pairs > per_row else min(pairs, per_row)
    egg_rows = -(-shown // per_row)
    ox = width - inner - 3  # the open block's left side
    # Dust drifting behind the chain, faster while the round's answers are being worked out.
    drift = frame // (2 if busy else 6)
    for r in range(BAND):
        for c in range(width):
            if _noise(c + drift, r * 31 + 7) % 53 == 0:
                rows[r][c] = (g["lost"], DUST)

    # The sealed blocks, newest next to the open one; when they don't all fit, the chain runs in from the left edge.
    last_x = ox - LINK - BLOCK
    fit = min(len(sealed), last_x // (BLOCK + LINK) + 1) if last_x >= 0 else 0
    hidden = len(sealed) - fit
    if hidden:
        fit = min(len(sealed), (last_x - LINK) // (BLOCK + LINK) + 1) if last_x >= LINK else 0
        hidden = len(sealed) - fit
    shown_gens = sealed[len(sealed) - fit:] if fit else []
    for i, gen in enumerate(shown_gens):
        bx = last_x - (len(shown_gens) - 1 - i) * (BLOCK + LINK)
        prev = sealed[len(sealed) - len(shown_gens) + i - 1] if len(sealed) - len(shown_gens) + i > 0 else None
        _sealed_block(g, put, bx, gen, prev, linked_in=i > 0 or hidden > 0)
        put(2, bx + BLOCK, g["chain"] + g["link"] + g["chain"], LINKS)
    if hidden and shown_gens:
        put(2, last_x - (len(shown_gens) - 1) * (BLOCK + LINK) - LINK, g["chain"] + g["link"] + g["chain"], LINKS)
    if not sealed and not hatching and ox - LINK - 8 >= 0:
        put(2, ox - LINK - 8, "genesis ", DIM)
    if sealed or hatching or ox - LINK - 8 >= 0:
        put(2, ox - LINK, g["chain"] + g["link"] + g["chain"], MINT if locking else LINKS)

    # The open block.
    box = g["sealed"] if locking else g["open"]
    dormant = number is None and not hatching
    edge = MINT if locking else TEAL_DIM if dormant else (MINT if busy and frame % 12 < 6 else TEAL)
    put(0, ox, box[0] + box[1] * inner + box[2], edge)
    for r in range(1, 5):
        put(r, ox, box[6] if r == 2 else box[3], edge)
        put(r, ox + inner + 1, box[3], edge)
    put(5, ox, box[4] + box[1] * inner + box[5], edge)
    for r in range(1, 5):
        put(r, ox + 1, " " * inner, None)
    gen_no = landed[1].number if hatching else number
    if gen_no is not None:
        title = termui._fit(inner - 2, f" generation {gen_no} ", f" gen {gen_no} ", f" {gen_no} ")
        put(0, ox + 2, title, WHITE if not dormant else DIM)
    start = ox + 1 + max(0, (inner - (3 * min(shown, per_row) - 1)) // 2)

    def egg_at(k):
        return 2 + k // per_row, start + 3 * (k % per_row)

    caption, style = "", DIM
    if hatching:
        gen = landed[1]
        got = {j: (rp, rm) for j, rp, rm in gen.update}
        for k in range(shown):
            r, c = egg_at(k)
            if k not in got:
                put(r, c, g["lost"] * 2, DUST)
                continue
            rp, rm = got[k]
            for side, (mine, other) in enumerate(((rp, rm), (rm, rp))):
                if mine > other:
                    cell = (g["hatched"], GLOW if not locking else FIT[3])
                elif mine < other:
                    cell = (g["eggs"][1], PLUS_DIM if side == 0 else MINUS_DIM)
                else:
                    cell = (g["twin"], FIT[2]) if mine > 0 else (g["eggs"][1], DUST)
                put(r, c + side, *cell)
        if not locking:
            for i in range(5):  # sparks off the hatching
                n = _noise(frame, 50 + i)
                put(1, ox + 1 + n % inner, g["sparks"][(n >> 8) % 5], GLOW if n & 1 else GOLD)
        caption = (f"sealed {size(gen.bytes)}" if locking else
                   f"hatched {gen.mean:.0%}" if gen.mean is not None else "none hatched")
        style = MINT if locking else GLOW
    elif dormant:
        caption = waiting or "waiting for a round"
    elif phase == "next":  # the last round's block is sealed: this one is new, and empty until the next round
        caption = "a new block opens"
    else:
        f = _fraction(progress, frac)
        at = f * shown
        for k in range(shown):
            r, c = egg_at(k)
            for side in (0, 1):
                hue, dim = (PLUS, PLUS_DIM) if side == 0 else (MINUS, MINUS_DIM)
                if phase == "register":
                    cell = (g["eggs"][0], dim)
                elif phase == "compute":
                    if k < int(at) or f >= 1:
                        cell = (g["eggs"][4], hue)
                    elif k == int(at):
                        cell = (g["eggs"][1 + (frame // 3 + side) % 3], hue)  # growing
                    else:
                        cell = (g["eggs"][1], dim)
                elif phase == "reveal":
                    cell = (g["cracked"], hue) if k < int(at) or f >= 1 else (g["eggs"][4], hue)
                elif phase == "verify":
                    checked = k < int(at) or f >= 1
                    cell = (g["cracked"], WHITE if k == (frame // 2) % shown else MINT if checked else hue)
                else:  # idle: the round is over and its block being sealed
                    cell = (g["cracked"], MINT if frame % 8 < 4 else hue)
                put(r, c + side, *cell)
            if egg_rows == 1:
                put(3, c, g["nest"] * 2, NEST)
        if phase == "compute":  # the incubator's warmth
            for i in range(inner):
                n = _noise(i * 7 + frame // 2, 13)
                if n % (4 if busy else 9) == 0:
                    put(1, ox + 1 + i, g["heat"], TEAL if n & 2 else TEAL_DIM)
        got = f" {progress[0]}/{progress[1]}" if progress is not None and progress[1] else ""
        caption = {"register": "laying: workers sign up", "compute": f"incubating{got}", "reveal": f"cracking{got}",
                   "verify": f"checking{got}", "idle": "sealing the block"}.get(phase, phase)
        style = PHASES.get(phase, DIM)
        if pairs > shown:
            caption += f" +{pairs - shown} pairs"
    caption = caption[:inner]
    put(4, ox + 1 + max(0, (inner - len(caption)) // 2), caption, style)
    legend = termui._fit(inner + 2, f"{pairs} twin pairs, + and -", f"{pairs} twin pairs", f"{pairs} pairs", "")
    if not dormant:
        put(6, ox + max(0, (inner + 2 - len(legend)) // 2), legend, DIM)
    return [_cells(row) for row in rows]


def _sealed_block(g: dict, put, bx: int, gen: Generation, prev: Generation | None, linked_in: bool) -> None:
    box, size_ = g["sealed"], BLOCK - 2
    mean = gen.mean
    fit = creature(mean)
    put(1, bx, box[0] + box[1] * size_ + box[2], TEAL_DIM)
    for r in (2, 3):
        put(r, bx, box[6] if r == 2 and linked_in else box[3], TEAL_DIM)
        put(r, bx + BLOCK - 1, box[7] if r == 2 else box[3], TEAL_DIM)
    put(4, bx, box[4] + box[1] * size_ + box[5], TEAL_DIM)
    for r in (2, 3):
        put(r, bx + 1, " " * size_, None)
    put(2, bx + 1 + (size_ - 1) // 2, g["creatures"][fit], FIT[fit])  # what grew in it
    halves = round((mean or 0.0) * 2 * size_)
    bar = g["full"] * (halves // 2) + g["half"] * (halves % 2)
    put(3, bx + 1, bar.ljust(size_, " "), FIT[max(1, fit)])
    label = f"g{gen.number}"
    if gen.level is not None:
        moved = prev is not None and prev.level is not None and gen.level != prev.level
        mark = (g["up"] if gen.level > prev.level else g["down"]) if moved else "L"
        tail, tail_style = f" {mark}{gen.level}", (GREEN if gen.level > prev.level else YELLOW) if moved else DIM
    else:
        tail, tail_style = "", DIM
    text = (label + tail)[:BLOCK]
    at = bx + max(0, (BLOCK - len(text)) // 2)
    put(5, at, label[:BLOCK], TEXT)
    put(5, at + len(label), tail[:max(0, BLOCK - len(label))], tail_style)
    b = size(gen.bytes)
    put(6, bx + max(0, (BLOCK - len(b)) // 2), b[:BLOCK], DIM)
