"""The app icons, written with nothing but the standard library (an .ico may hold PNG images; this writes them with
zlib): the brand mark (the pickaxe of three gold blocks, docs/site) on a rounded square, at the sizes Windows asks for.

    launcher/VibeHash.ico                          the miner's app: the mark on dark
    vibehash/data/admin.ico, vibehash/data/admin.png   the admin app (admin.py's shortcut and window): the mark on violet

Run it after changing the mark: python scripts/make_icon.py"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

GOLD, DARK, VIOLET = (241, 178, 60, 255), (23, 26, 33, 255), (52, 33, 92, 255)
MARK = ((0, 0, 1, 3), (2, 0, 1, 3), (1, 3, 1, 1))  # on a 3 by 4 grid: two uprights and the head's block
SIZES = (16, 24, 32, 48, 64, 128, 256)


def pixel(x: float, y: float, n: int, ground=DARK) -> tuple[int, int, int, int]:
    """The colour at (x, y) of an n by n icon: transparent outside the rounded square, the mark in gold inside."""
    r = n * 0.22
    dx = max(r - x, x - (n - r), 0)
    dy = max(r - y, y - (n - r), 0)
    if dx * dx + dy * dy > r * r:
        return (0, 0, 0, 0)
    pad = n * 0.2
    cell = (n - 2 * pad) / 4  # the grid is 3 wide, 4 tall: fit the height, centre the width
    gx, gy = (x - pad - cell / 2) / cell, (y - pad) / cell
    for bx, by, bw, bh in MARK:
        if bx <= gx < bx + bw and by <= gy < by + bh:
            return GOLD
    return ground


def png(n: int, ground=DARK) -> bytes:
    rows = bytearray()
    for y in range(n):
        rows.append(0)  # filter: none
        for x in range(n):
            # 4 samples per pixel for soft edges
            acc = [0, 0, 0, 0]
            for sx, sy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
                p = pixel(x + sx, y + sy, n, ground)
                for i in range(4):
                    acc[i] += p[i]
            rows.extend(v // 4 for v in acc)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def ico(sizes=SIZES, ground=DARK) -> bytes:
    images = [png(n, ground) for n in sizes]
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(head) + 16 * len(sizes)
    entries = b""
    for n, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return head + entries + b"".join(images)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    outs = {root / "launcher" / "VibeHash.ico": ico(), root / "vibehash" / "data" / "admin.ico": ico(ground=VIOLET),
            root / "vibehash" / "data" / "admin.png": png(256, VIOLET)}
    if len(sys.argv) > 1:  # one file, named: the miner's icon
        outs = {Path(sys.argv[1]): ico()}
    for out, data in outs.items():
        out.write_bytes(data)
        print(f"wrote {out} ({out.stat().st_size} bytes)")
