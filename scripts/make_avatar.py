"""The Vibe Hash profile picture: the site's three-block mark as beveled gold ore on the dashboard's dark band.
Everything sits inside the circle X crops avatars to. Plain NumPy and a hand-written PNG: no image library needed.

    python scripts/make_avatar.py docs/brand/vibehash-x-profile.png [SIZE]
"""
import struct
import sys
import zlib

import numpy as np

S = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255  # noqa: E731
BAND, EDGE, GOLD = hexrgb("#1c1712"), hexrgb("#0a0908"), hexrgb("#f1b23c")
LIGHT, DARK = hexrgb("#ffd987"), hexrgb("#a86d0e")

y, x = np.mgrid[0:S, 0:S].astype(float) + 0.5
c = S / 2
d = np.hypot(x - c, y - c) / (S / 2)  # 0 at the centre, 1 at the edge

img = BAND + (EDGE - BAND) * np.clip(d, 0, 1)[..., None] ** 1.4  # the dark band, darker towards the rim
img += GOLD * (0.22 * np.exp(-(d / 0.42) ** 2))[..., None]  # a soft glow behind the mark


def block(x0, y0, x1, y1, bevel, a=1.0):
    """A gold block with a pixel-art bevel: lit from the top left."""
    xs, ys = slice(int(x0), int(x1)), slice(int(y0), int(y1))
    tile = np.empty((int(y1) - int(y0), int(x1) - int(x0), 3))
    tile[:] = GOLD
    b = int(bevel)
    tile[:b, :] = LIGHT
    tile[:, :b] = LIGHT
    tile[-b:, :] = DARK
    tile[:, -b:] = DARK
    tile[-b:, :b] = (LIGHT + DARK) / 2  # the two corners where light meets shade
    tile[:b, -b:] = (LIGHT + DARK) / 2
    img[ys, xs] = img[ys, xs] * (1 - a) + tile * a


# The mark: viewBox 0 0 3 4, rects (0,0)-(1,3), (2,0)-(3,3), (1,3)-(2,4).
u = S * 0.145
ox, oy = c - 1.5 * u, c - 2 * u
for rx, ry, rw, rh in ((0, 0, 1, 3), (2, 0, 1, 3), (1, 3, 1, 1)):
    block(ox + rx * u, oy + ry * u, ox + (rx + rw) * u, oy + (ry + rh) * u, u * 0.12)

# Flecks of ore around it, inside the crop circle.
for fx, fy, size, a in ((0.19, 0.25, 0.026, 0.85), (0.80, 0.31, 0.020, 0.7), (0.76, 0.79, 0.030, 0.8),
                        (0.24, 0.74, 0.018, 0.6), (0.85, 0.56, 0.014, 0.55), (0.14, 0.50, 0.012, 0.45),
                        (0.62, 0.15, 0.012, 0.5), (0.40, 0.87, 0.014, 0.5)):
    h = size * S
    block(fx * S - h, fy * S - h, fx * S + h, fy * S + h, max(2, h * 0.25), a)

px = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
raw = b"".join(b"\x00" + row.tobytes() for row in px)  # filter type 0 on every row


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", S, S, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
with open(sys.argv[1], "wb") as f:
    f.write(png)
print(f"wrote {sys.argv[1]}: {S}x{S}, {len(png):,} bytes")
