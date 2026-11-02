"""The split prep's numbers (DEEPSEEK.md §8, "The split prep"), from deepseek_split's work graph for
DeepSeek-V4.1-Flash and a few measured or stated rates (below). Prints Markdown tables.

    python scripts/deepseek_split_plan.py [--cut 36] [--seq 4096] [--chunk 65536]

- What crosses each cut, bytes a token: the stream and side channels as stored (exact) and with lossless compression.
- Traffic for pieces of G layers at 100K, 1M and 10M tokens: weights from Hugging Face, the bucket's egress, and
  what miners upload; the bucket's bill.
- One miner's share: a unit's GPU time and transfers for a chunk, and its download and upload per million tokens.
- End to end for 1M tokens with N miners: a simulation of the assembly line, units network-bound, with and without
  waiting for each unit's challenge window.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibehash import deepseek_split as split  # noqa: E402
from vibehash import deepseek_top as top  # noqa: E402
from vibehash.deepseek import Config  # noqa: E402

LAYER_GB = 266.1 / 36  # a frozen layer's weight file (measured sizes: 7.39-7.40 GB)
EMBED_GB, TABLES_GB = 1.32, 2 * 101.4  # the embedding; the two engram tables, streamed by the rows units
STREAM_ZSTD = 0.68  # the bf16 stream's byte planes through zstd level 3, measured on real layers' output
ROWS_PER_TOKEN = 3439  # engram rows a token needs per table, in 64K-token chunks (measured on real chat text)
GPU_MS = 0.30  # a frozen layer's forward per token on a miner's GPU (§7: 0.24 on a 4070 SUPER, experts in RAM)
SPACES = dict(base=5.0, storage_gib=250, egress_gib=1024, per_gib_stored=0.02, per_gib_out=0.01)
LINKS = {"cable, 100/20 Mbit/s": (12.5e6, 2.5e6), "fiber, 300/300 Mbit/s": (37.5e6, 37.5e6)}


def picks_bytes(cfg: Config, ratio: int, seq: int) -> float:
    """Picks of an index source as a bitmap over the compressed tokens a query sees, a token's mean over a sequence:
    nothing while it sees no more than index_topk (it picks them all)."""
    total = 0.0
    for t in range(seq):
        n = (t + 1) // ratio
        total += n / 8 if n > cfg.index_topk else 0
    return total / seq


def sizes(cfg: Config, seq: int) -> dict:
    """Bytes a token of each kind of object, (as stored, losslessly compressed)."""
    width = 2 * cfg.hc_mult * cfg.dim + 4 * cfg.hc_mult  # bf16 stream, float32 pre_mix
    out = {"stream": (width, width * STREAM_ZSTD), "rows": (ROWS_PER_TOKEN, ROWS_PER_TOKEN)}
    for r in (1, 2):
        out[f"compress_kv/{r}"] = (2 * cfg.head_dim / r, 0.5625 * cfg.head_dim / r)  # FP4 values, e4m3 scale per 16
        out[f"index_k/{r}"] = (2 * cfg.index_head_dim / r, (0.5 + 1 / 32) * cfg.index_head_dim / r)
        out[f"picks/{r}"] = (2 * cfg.index_topk, picks_bytes(cfg, r, seq))
    out["candidates"] = (0.0, 0.0)  # only past 16K tokens at ratio 1
    return out


def object_bytes(name: str, cfg: Config, size: dict, packed: bool) -> float:
    """Bytes a token of an object of the plan."""
    leaf = name.split("/", 1)[1]
    k = int(packed)
    if leaf.startswith("stream"):
        return size["stream"][k]
    if leaf.startswith("rows"):
        return size["rows"][k] / 15  # one of a table's 15 ranges
    kind, src = leaf.rsplit("-", 1)
    if kind == "candidates":
        return 0.0
    return size[f"{kind}/{cfg.ratio(int(src))}"][k]


def traffic(cut: top.Cut, g: int, tokens: float, seq: int, packed: bool = True) -> dict:
    """GB moved by a split prep in pieces of g layers (one chunk stands for all: every object scales with tokens)."""
    cfg, size = cut.cfg, sizes(cut.cfg, seq)
    graph = split.plan(cut, [[0]], split.pieces(cut.first, g))
    egress = upload = 0.0
    for name, writer in graph.producer.items():
        if name.endswith(".json") or name.startswith("block-"):
            continue
        b = object_bytes(name, cfg, size, packed) * tokens / 1e9
        upload += b
        egress += b * len(graph.readers[name])
    records = top.Cut(cfg, cut.first).bytes_per_token(seq_len=seq)["total"] * tokens / 1e9
    weights = cut.first * LAYER_GB + EMBED_GB + TABLES_GB
    return dict(weights=weights, egress=egress, upload=upload + records, records=records,
                pieces=len(split.pieces(cut.first, g)))


def bill(egress_gb: float, stored_gb: float) -> float:
    gib = 1e9 / 2**30
    return SPACES["base"] + max(0, egress_gb * gib - SPACES["egress_gib"]) * SPACES["per_gib_out"] + \
        max(0, stored_gb * gib - SPACES["storage_gib"]) * SPACES["per_gib_stored"]


def simulate(cut: top.Cut, g: int, tokens: int, chunk: int, n: int, link: tuple, seq: int, window: float = 0.0,
             cached: bool = False) -> float:
    """Hours until the last record is written, n miners shared out evenly over the pieces (the rows units run
    first, on other miners). Each miner downloads its layers (unless cached), then takes chunks as their inputs
    arrive: a chunk's download, its GPU work and its upload overlap with other chunks', not with each other.
    `window`: hours a unit's output waits to be accepted before the next piece reads it (0: read at once)."""
    cfg, size = cut.cfg, sizes(cut.cfg, seq)
    down, up = link
    graph = split.plan(cut, [[0]], split.pieces(cut.first, g))
    per_piece = max(1, n // len(split.pieces(cut.first, g)))
    chunks = -(-tokens // chunk)
    rows_h = (TABLES_GB / 2 / 15 * 1e9 / down + ROWS_PER_TOKEN * tokens / 15 / up) / 3600
    ready = [rows_h] * chunks
    for a, b in split.pieces(cut.first, g):
        unit = graph.units[f"layers-{a:02d}-{b:02d}-c00000"]
        inp = sum(object_bytes(i, cfg, size, True) for i in unit.inputs if not i.endswith(".json")) * chunk
        out = sum(object_bytes(o, cfg, size, True) for o in unit.outputs
                  if not o.endswith(".json") and not o.startswith("block-")) * chunk
        if b == cut.first:
            out = top.Cut(cfg, cut.first).bytes_per_token(seq_len=seq)["total"] * chunk
        if a == 0:
            inp += 4 * chunk  # the token ids
        start = 0.0 if cached else ((b - a) * LAYER_GB + (EMBED_GB if a == 0 else 0)) * 1e9 / down / 3600
        miners = [[start, start, start] for _ in range(per_piece)]  # when its downlink, GPU and uplink are free
        nxt = []
        for c in sorted(range(chunks), key=lambda c: ready[c]):
            best = None
            for m in miners:
                d_end = max(ready[c], m[0]) + inp / down / 3600
                g_end = max(d_end, m[1]) + (b - a) * chunk * GPU_MS / 1e3 / 3600
                u_end = max(g_end, m[2]) + out / up / 3600
                if best is None or u_end < best[0]:
                    best = (u_end, m, d_end, g_end)
            u_end, m, d_end, g_end = best
            m[:] = [d_end, g_end, u_end]
            nxt.append((c, u_end + window))
        ready = [t for _, t in sorted(nxt)]
    return max(ready)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cut", type=int, default=36)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=65536)
    args = ap.parse_args()
    cfg = Config.flash()
    cut = top.Cut(cfg, args.cut)
    size = sizes(cfg, args.seq)

    print(f"## What crosses each cut, bytes a token (sequences of {args.seq})\n")
    print("| cut | side channels crossing it (source layer) | stored | compressed |")
    print("|---|---|---|---|")
    for c in (2, 4, 8, 14, 16, 20, 21, 24, 28, 32, 36):
        cross = {k: v for k, v in top.Cut(cfg, c).crossing().items() if k not in ("engram", "candidates")}
        side = [size[f"{k}/{cfg.ratio(v)}"] for k, v in cross.items()]
        stored = size["stream"][0] + sum(s[0] for s in side)
        packed = size["stream"][1] + sum(s[1] for s in side)
        names = ", ".join(f"{k} ({v})" for k, v in cross.items()) or "none"
        print(f"| {c} | {names} | {stored:,.0f} | {packed:,.0f} |")
    print(f"\nthe stream alone: {size['stream'][0]:,.0f} stored, {size['stream'][1]:,.0f} compressed; engram rows "
          f"{ROWS_PER_TOKEN:,} a token a table, read by the piece holding layer 1 or 14\n")

    print("## Traffic, GB (pieces of G layers; one holder a piece; hand-offs compressed)\n")
    print("| G | pieces | tokens | weights from HF (once) | hand-offs: bucket egress | miners' upload | "
          "Spaces bill |")
    print("|---|---|---|---|---|---|---|")
    for g in (1, 2, 4, 9, 36):
        for tokens in (1e5, 1e6, 1e7):
            t = traffic(cut, g, tokens, args.seq)
            print(f"| {g} | {t['pieces']} | {tokens / 1e6:g}M | {t['weights']:,.0f} | {t['egress']:,.0f} | "
                  f"{t['upload']:,.0f} | ${bill(t['egress'] + t['records'], t['records']):,.2f} |")
    print("\nminers' upload includes the records (the dataset the trainers read), stored at "
          f"{top.Cut(cfg, args.cut).bytes_per_token(seq_len=args.seq)['total']:,.0f} bytes a token; the bill counts the "
          "trainers reading them once, and storing them a month")

    print(f"\n## A unit: one piece over a {args.chunk // 1024}K-token chunk\n")
    print("| G | GPU | download | upload | cable 100/20: down, up | fiber 300/300 |")
    print("|---|---|---|---|---|---|")
    graph = split.plan(cut, [[0]], split.pieces(args.cut, 4))
    mid = graph.units["layers-24-28-c00000"]
    inp = sum(object_bytes(i, cfg, size, True) for i in mid.inputs if not i.endswith(".json")) * args.chunk / 1e9
    out = size["stream"][1] * args.chunk / 1e9
    for g in (1, 2, 4, 9):
        gpu = g * args.chunk * GPU_MS / 1e3
        print(f"| {g} | {gpu:.0f} s | {inp:.2f} GB | {out:.2f} GB | {inp * 1e9 / 12.5e6 / 60:.1f} min, "
              f"{out * 1e9 / 2.5e6 / 60:.1f} min | {inp * 1e9 / 37.5e6 / 60:.1f} min, {out * 1e9 / 37.5e6 / 60:.1f} min |")
    print(f"\nper million tokens through a piece: download {inp * 1e6 / args.chunk:.1f} GB, upload "
          f"{out * 1e6 / args.chunk:.1f} GB, whatever G is; plus the piece's weights once, G x {LAYER_GB:.2f} GB\n")

    print("## End to end, 1M tokens, hours (weights not cached; in 64K-token chunks)\n")
    for label, link in LINKS.items():
        print(f"{label}:\n")
        print("| G | miners | read at once | wait 1 h a unit | weights cached, read at once |")
        print("|---|---|---|---|---|")
        for g in (1, 2, 4, 9):
            p = len(split.pieces(args.cut, g))
            for n in sorted({p, 2 * p, 36}):
                if n < p:
                    continue
                times = [simulate(cut, g, 10**6, args.chunk, n, link, args.seq),
                         simulate(cut, g, 10**6, args.chunk, n, link, args.seq, window=1.0),
                         simulate(cut, g, 10**6, args.chunk, n, link, args.seq, cached=True)]
                print(f"| {g} | {n} | {times[0]:.1f} | {times[1]:.1f} | {times[2]:.1f} |")
        print()


if __name__ == "__main__":
    main()
