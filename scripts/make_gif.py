"""Record the mining dashboard (minerui.py) as an animated GIF, for sharing.

It runs the real MinerDashboard on a scripted ten-second session (training units, filling the ore cart, checking
other workers' results, a payout) on a fake clock, and draws each frame's colored text into an image: a
terminal's colors and cells, with block characters painted as solid cells the way a terminal shows them.

    python scripts/make_gif.py docs/brand/vibehash-mining.gif

Needs Pillow (`pip install pillow`); Vibe Hash itself doesn't.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

os.environ.pop("NO_COLOR", None)  # the dashboard honours it, but a recording is always in color
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import minerui  # noqa: E402
from vibehash.client import WorkerStats  # noqa: E402
from vibehash.model import ModelSpec  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE, TrainingJob  # noqa: E402

COLS, ROWS, FPS, SECONDS = 92, 27, 12, 10
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
REGULAR, BOLD = ImageFont.truetype(str(FONT_DIR / "consola.ttf"), 18), ImageFont.truetype(str(FONT_DIR / "consolab.ttf"), 18)
CW, CH = round(REGULAR.getlength("M")), 22  # one character cell
PAD, BAR = 18, 34  # margin around the text; the window's title bar
BG, BAR_BG, BAR_INK, DEFAULT = (12, 11, 10), (27, 25, 22), (127, 137, 149), (205, 212, 220)
SHADE = {"█": 1.0, "▓": 0.75, "▒": 0.5, "░": 0.25}  # painted as solid cells: font glyphs leave seams
HALF = {"▀": (0, 0, 1, 0.5), "▄": (0, 0.5, 1, 1), "▌": (0, 0, 0.5, 1), "▐": (0.5, 0, 1, 1)}
SGR = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


def cells(frame: str):
    """Each character with its (color, bold)."""
    color, bold = DEFAULT, False
    for row in frame.split("\n")[:ROWS]:
        out, pos = [], 0
        for m in SGR.finditer(row):
            out += [(ch, color, bold) for ch in row[pos:m.start()]]
            pos = m.end()
            if m.group(2) == "m":
                p = [int(v) for v in m.group(1).split(";") if v]
                if not p or p == [0]:
                    color, bold = DEFAULT, False
                if 1 in p:
                    bold = True
                if 38 in p:
                    i = p.index(38)
                    color = tuple(p[i + 2:i + 5])
        out += [(ch, color, bold) for ch in row[pos:]]
        yield out


def draw(frame: str) -> Image.Image:
    img = Image.new("RGB", (COLS * CW + 2 * PAD, ROWS * CH + 2 * PAD + BAR), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, BAR), fill=BAR_BG)
    for i, dot in enumerate([(240, 95, 85), (240, 200, 80), (95, 225, 130)]):
        d.ellipse((14 + i * 20, 11, 26 + i * 20, 23), fill=dot)
    d.text((80, 7), "> vibehash worker", font=BOLD, fill=BAR_INK)
    for r, row in enumerate(cells(frame)):
        for c, (ch, color, bold) in enumerate(row[:COLS]):
            if ch == " ":
                continue
            x, y = PAD + c * CW, BAR + PAD + r * CH
            if ch in SHADE:
                a = SHADE[ch]
                d.rectangle((x, y, x + CW - 1, y + CH - 1), fill=tuple(int(b + (v - b) * a) for v, b in zip(color, BG)))
            elif ch in HALF:
                x0, y0, x1, y1 = HALF[ch]
                d.rectangle((x + x0 * CW, y + y0 * CH, x + x1 * CW - 1, y + y1 * CH - 1), fill=color)
            else:
                d.text((x, y + 1), ch, font=BOLD if bold else REGULAR, fill=color)
    return img


def session():
    """A worker on the public network: training, filling the cart, checking others' work, then paid."""
    clock = [1_793_400_000.0]
    job = TrainingJob("fruit-demo", "req", 0, SYN_PER_VIBE, spec=ModelSpec(5, (128, 128), 5, "float32"))
    stats = WorkerStats(round=213, phase="compute", height=212, job=job, flops_per_sec=2.9e11, busy=True, assigned=6,
                        rounds=14, units=81, checks=37, training_flops=6.1e12)
    rig = SimpleNamespace(stats=stats, keys=SimpleNamespace(address="vibe17b0d8bce54f6eab50df24fd44e5a408a9525"),
                          backend=SimpleNamespace(describe=lambda: "NVIDIA GeForce RTX 4070 SUPER (CUDA)"))
    with mock.patch("time.time", lambda: clock[0]), \
            mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((COLS + 1, ROWS + 1))):
        dash = minerui.MinerDashboard([("you", rig)], "https://node.vibehash.org")
        dash.g, dash.chain_id = minerui.GLYPHS, "vibehash-public-1"
        r = dash.rigs[0]
        r.start, r.total, r.balance, r.stake, r.reputation = 0, 41_76 * SYN_PER_VIBE // 100, 36_76 * SYN_PER_VIBE // 100, 5 * SYN_PER_VIBE, 0.92
        for back, text, level in ((40, "round 211: registered", 20), (24, "round 211: computed 6 units, committed 6", 20),
                                  (9, "paid +3.18 VIBE in block 211", 21), (6, "round 212: registered", 20)):
            dash.feed.append((clock[0] - back, level, text))
        start = clock[0]
        stats.phase_ends = start + 7.5
        frames = []
        for n in range(FPS * SECONDS):
            t = n / FPS
            if t < 6:  # compute: training, then uploading and committing each unit
                stats.phase, stats.busy = "compute", True
                stats.trained = min(6, 2 + int(t / 1.25))
                stats.committed = min(6, 1 + int(t / 1.15))
                stats.activity = (f"uploading unit {stats.trained} of 6" if t % 1.25 > 0.95 or t >= 5
                                  else f"training unit {stats.trained + 1} of 6")
            elif t < 8.4:  # verify: checking other workers' results
                if stats.phase != "verify":
                    stats.phase, stats.phase_ends = "verify", clock[0] + 2.6
                    dash.note("round 213: computed 6 units, committed 6")
                stats.busy = True
                stats.checked = min(3, 1 + int((t - 6) / 0.8))
                stats.activity = f"checking other workers' results ({stats.checked} of 3)"
            else:  # paid: the block lands
                if dash.popup is None:
                    paid = 321 * SYN_PER_VIBE // 100
                    r.total += paid
                    r.balance += paid
                    dash.popup = (clock[0], paid)
                    dash.note("paid +3.21 VIBE in block 213", 21)
                    stats.phase, stats.busy, stats.height, stats.activity = "idle", False, 213, "round done; waiting for the next"
            frames.append(draw(dash.frame()))
            clock[0] += 1 / FPS
    return frames


def main(out: str) -> None:
    frames = session()
    # One palette for every frame, so colors don't flicker from frame to frame.
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 4))
    for i, f in enumerate(frames[:: max(1, len(frames) // 4)][:4]):
        sample.paste(f, (0, i * frames[0].height))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    paletted[0].save(out, save_all=True, append_images=paletted[1:], duration=round(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {frames[0].width}x{frames[0].height}, {Path(out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
