"""vibehash/data/globe.json: the land of the Earth as a field of evenly spaced dots, what the admin app's globe
(data/admin.html) is drawn from. Made from Natural Earth's 110m land polygons (public domain), as the world-atlas
package publishes them in TopoJSON (land-110m.json), with nothing but the standard library.

    python scripts/make_globe.py [land-110m.json]   # downloads the TopoJSON when no file is given

Dots sit on rows of latitude STEP degrees apart, spaced STEP / cos(latitude) along each row so they are about as far
apart everywhere (an equal-area look: no crowding at the poles), and only where the row crosses land (even-odd rule
over every ring, so lakes stay empty). The file holds integer tenths of a degree: {"step": STEP, "rows": [[lat,
[lon, lon, ...]], ...]}, about 8,000 dots at STEP 1.2."""
from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

SOURCE = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json"
STEP = 1.2  # degrees between rows, and between dots along the equator
OUT = Path(__file__).resolve().parent.parent / "vibehash" / "data" / "globe.json"


def rings(topology: dict) -> list[list[tuple[float, float]]]:
    """Every ring of every land polygon as (lon, lat) points, from the TopoJSON's delta-encoded, quantised arcs."""
    scale, translate = topology["transform"]["scale"], topology["transform"]["translate"]
    arcs = []
    for arc in topology["arcs"]:
        x = y = 0
        points = []
        for dx, dy in arc:
            x += dx
            y += dy
            points.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        arcs.append(points)
    out = []
    for geometry in topology["objects"]["land"]["geometries"]:
        polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
        for polygon in polygons:
            for ring in polygon:
                points: list[tuple[float, float]] = []
                for index in ring:
                    arc = arcs[index] if index >= 0 else arcs[~index][::-1]
                    points.extend(arc if not points else arc[1:])
                out.append(points)
    return out


def crossings(ring: list[tuple[float, float]], lat: float) -> list[float]:
    """The longitudes where a ring crosses a latitude (half-open on each edge, so a vertex counts once)."""
    out = []
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 <= lat) != (y1 <= lat):
            out.append(x0 + (lat - y0) * (x1 - x0) / (y1 - y0))
    return out


def dots(land: list[list[tuple[float, float]]], step: float = STEP) -> list[tuple[int, list[int]]]:
    rows = []
    n_rows = int(round(180 / step))
    for k in range(n_rows):
        lat = -90 + (k + 0.5) * (180 / n_rows)
        xs = sorted(x for ring in land for x in crossings(ring, lat))
        inside = list(zip(xs[0::2], xs[1::2]))  # even-odd: between the 1st and 2nd crossing is land, 2nd to 3rd isn't
        if not inside:
            continue
        spacing = min(step / max(math.cos(math.radians(lat)), 1e-6), 60.0)
        offset = (k % 2) * spacing / 2  # staggered rows
        lons = []
        lon = -180 + offset
        while lon < 180:
            if any(a <= lon < b for a, b in inside):
                lons.append(int(round(lon * 10)))
            lon += spacing
        if lons:
            rows.append((int(round(lat * 10)), lons))
    return rows


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        topology = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as resp:  # noqa: S310 (a fixed https address)
            topology = json.loads(resp.read())
    rows = dots(rings(topology))
    OUT.write_text(json.dumps({"step": STEP, "source": "Natural Earth 110m land (public domain) via world-atlas 2.0.2",
                               "rows": [[lat, lons] for lat, lons in rows]}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {sum(len(l) for _, l in rows)} dots on {len(rows)} rows ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv)
