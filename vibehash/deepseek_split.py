"""The split prep (DEEPSEEK.md §8, "The split prep"): the frozen layers' one-time pass over a job's text, cut into
small units that ordinary miners run, connected by stored hand-offs. Nobody has to be online with anybody else: the
bottom is frozen, so a unit's output depends only on its inputs, and each output waits in storage for the unit that
reads it. An asynchronous assembly line.

The units of the work graph `plan` makes:
- rows (engram layer L, row range r): rows [lo, hi) of L's 101 GB table streamed once (about 7 GB), keeping, for
  each chunk of the text, the rows its tokens' n-grams hash to in that range. Needs no other unit.
- layers (piece [a, b), chunk c): layers a..b-1 over chunk c's sequences. It reads the stream entering layer a, the
  side channels its layers reuse from below a (each stored once, by the piece that made it, and read by every piece
  that reuses it: never relayed), and an engram layer's rows. It writes the stream entering layer b, the side
  channels it made that later layers reuse, and its commitment: every sequence's hash after each of its layers
  (deepseek_prep.layer_hash, so a dispute bisects over them). The last piece writes the job's dataset records
  instead of a stream, in deepseek_prep's format, with deepseek_prep's Merkle root.

Hand-offs are exact. The stream stays bf16 between frozen layers, so the records are the model's own, as the one-PC
prep makes them; what FP8 at every cut would cost the model isn't measured (scripts/deepseek_stream_stats.py: its
stand-in chain of real layers is thrown as far by a 0.4% nudge as by FP8). It's compressed losslessly instead: byte
planes, then zstd, to about 0.68 of its size. A commitment hashes what was decoded, not the compressed bytes, so the
codec can change without changing a hash.

Storage is a Store (put, get, has, delete). PresignedStore uses URLs the node signs (`presign`: S3's query-string
signature, which DigitalOcean Spaces takes), so miners need no storage account of their own; LocalStore is a folder.
`Line` is the node's side: which units can be claimed, optimistic consumption of outputs not yet past their
challenge window, rollback when a challenge succeeds, and which objects can be deleted.

Wiring into the node waits for PR #5's work units; run_local runs a whole plan on one machine (tests, dev).
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F

from . import deepseek as ds
from . import deepseek_prep as prep
from . import deepseek_top as top
from .crypto import merkle_root
from .deepseek import Choices, Side

try:
    from compression import zstd as _zstd  # Python 3.14's standard library
except ImportError:
    try:
        import zstandard

        class _zstd:  # the zstandard package (pip install zstandard), for older Pythons
            compress = staticmethod(lambda data, level: zstandard.ZstdCompressor(level=level).compress(data))
            decompress = staticmethod(lambda data: zstandard.ZstdDecompressor().decompress(data))
    except ImportError:
        _zstd = None

FORMAT = "vibehash-deepseek-split/1"
CODEC = "zstd" if _zstd is not None else "zlib"
ROWS_PER_PIECE = 2**18  # deepseek_prep.extract_rows' piece of a table: row ranges start and end on these
SIDE = ("compress_kv", "index_k", "picks")


# --- the plan ----------------------------------------------------------------------------------------------------------

def pieces(cut: int, size: int | None = None, bounds=None) -> list[tuple[int, int]]:
    """The frozen layers [0, cut) as pieces [a, b): `size` layers each (the last may be shorter), or starting at
    `bounds`."""
    starts = sorted({0, *(range(0, cut, size) if bounds is None else bounds)})
    if starts[-1] >= cut:
        raise ValueError(f"a piece starts at or above the cut ({cut})")
    return list(zip(starts, starts[1:] + [cut]))


def needs(cut: top.Cut, a: int, b: int) -> dict:
    """What piece [a, b) reads besides the stream: Cut.crossing's {name: source layer} for its layers, and
    "engram": its engram layers (their rows). The last piece also reads what the trained layers reuse, to write it
    into the records."""
    out = {} if a == 0 else top.Cut(cut.cfg, a).crossing(None if b == cut.first else b)
    out.pop("engram", None)
    engram = [layer for layer in cut.cfg.engram_layer_ids if a <= layer < b]
    return {**out, "engram": engram} if engram else out


def emits(cut: top.Cut, a: int, b: int) -> dict[str, int]:
    """The side channels piece [a, b) makes that layers at or above b reuse: {name: source layer}."""
    if b == cut.first:
        return {}  # the last piece writes them into the records
    return {k: v for k, v in top.Cut(cut.cfg, b).crossing().items() if k != "engram" and a <= v < b}


def row_ranges(cfg: ds.Config, layer: int, n: int, align: int = ROWS_PER_PIECE) -> list[tuple[int, int]]:
    """Engram table `layer` as at most n ranges of rows [lo, hi), each a whole number of `align` rows."""
    rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(layer)]
    step = -(-rows // n // align) * align or align
    return [(lo, min(lo + step, rows)) for lo in range(0, rows, step)]


def stream_name(c: int, layer: int) -> str:
    return f"c{c:05d}/stream-{layer:02d}"


def side_name(c: int, kind: str, src: int) -> str:
    return f"c{c:05d}/{kind}-{src:02d}"


def rows_name(c: int, layer: int, r: int) -> str:
    return f"c{c:05d}/rows-{layer:02d}-{r:03d}"


def commit_name(c: int, a: int) -> str:
    return f"c{c:05d}/commit-{a:02d}.json"


def block_names(c: int) -> tuple[str, str]:
    return f"block-{c:05d}.bin", f"block-{c:05d}.json"


@dataclass(frozen=True)
class Unit:
    kind: str  # "rows" or "layers"
    a: int  # rows: the engram layer; layers: the piece's first layer
    b: int  # rows: the row range's index; layers: the layer after the piece's last
    chunk: int = -1  # layers: the chunk; rows: -1 (a rows unit serves every chunk)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return f"rows-{self.a:02d}-{self.b:03d}" if self.kind == "rows" else \
            f"layers-{self.a:02d}-{self.b:02d}-c{self.chunk:05d}"


class Graph:
    """Units, and who reads what another writes. Objects no unit writes (none, for a plan) count as given."""

    def __init__(self, units: list[Unit]):
        self.units = {u.name: u for u in units}
        self.producer = {o: u.name for u in units for o in u.outputs}
        self.readers: dict[str, list[str]] = defaultdict(list)
        for u in units:
            for i in u.inputs:
                self.readers[i].append(u.name)

    def deps(self, name: str) -> set[str]:
        return {self.producer[i] for i in self.units[name].inputs if i in self.producer}

    def downstream(self, name: str) -> list[str]:
        """Every unit that reads, directly or not, what `name` writes."""
        out, todo = [], [name]
        while todo:
            for o in self.units[todo.pop()].outputs:
                for r in self.readers.get(o, []):
                    if r not in out:
                        out.append(r)
                        todo.append(r)
        return out

    def order(self) -> list[Unit]:
        done, out = set(), []
        while len(out) < len(self.units):
            for name, u in self.units.items():
                if name not in done and self.deps(name) <= done:
                    done.add(name)
                    out.append(u)
        return out


def plan(cut: top.Cut, chunks: list[list[int]], bounds: list[tuple[int, int]], ranges: int = 15,
         align: int = ROWS_PER_PIECE) -> Graph:
    """The work graph of a split prep: rows units for each engram table below the cut (`ranges` row ranges each),
    and a layers unit for each piece in `bounds` and each chunk (deepseek_prep.blocks_of's blocks of sequences)."""
    cfg, units = cut.cfg, []
    for layer in (layer for layer in cfg.engram_layer_ids if layer < cut.first):
        for r in range(len(row_ranges(cfg, layer, ranges, align))):
            units.append(Unit("rows", layer, r, -1, (), tuple(rows_name(c, layer, r) for c in range(len(chunks)))))
    for a, b in bounds:
        need = needs(cut, a, b)
        for c in range(len(chunks)):
            inputs = [stream_name(c, a)] if a else []
            inputs += [side_name(c, k, src) for k, src in need.items() if k != "engram"]
            inputs += [rows_name(c, layer, r) for layer in need.get("engram", [])
                       for r in range(len(row_ranges(cfg, layer, ranges, align)))]
            if b == cut.first:
                inputs += [commit_name(c, a2) for a2, b2 in bounds if b2 <= a]  # their hashes go into the records
                outputs = [*block_names(c), commit_name(c, a)]
            else:
                outputs = [stream_name(c, b), *(side_name(c, k, src) for k, src in emits(cut, a, b).items()),
                           commit_name(c, a)]
            units.append(Unit("layers", a, b, c, tuple(inputs), tuple(outputs)))
    return Graph(units)


# --- the node's side: claims, optimistic consumption, rollback, deletion ----------------------------------------------

class Line:
    """A split prep's progress, as the node keeps it. A unit is pending, claimed, done (committed: what it wrote can
    be read at once, before its challenge window closes) or accepted (the window passed, or a check agreed). A
    challenge that succeeds sends the unit and everything downstream of it back to pending: what they did was right
    for the inputs they were given, so they are paid, and the cheater's stake pays for it. An object is deleted once
    every unit that reads it is accepted, so storage holds only the frontier, and the dataset's records."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.state = {name: "pending" for name in graph.units}

    def claimable(self, holds=None) -> list[Unit]:
        """Pending units whose inputs are all written, for a miner holding layers `holds` (None: any)."""
        out = []
        for name, u in self.graph.units.items():
            if self.state[name] != "pending" or any(self.state[d] not in ("done", "accepted")
                                                    for d in self.graph.deps(name)):
                continue
            layers = {u.a} if u.kind == "rows" else set(range(u.a, u.b))
            if holds is None or layers <= set(holds):
                out.append(u)
        return out

    def claim(self, name: str) -> None:
        assert self.state[name] == "pending", (name, self.state[name])
        self.state[name] = "claimed"

    def done(self, name: str) -> None:
        self.state[name] = "done"

    def accept(self, name: str) -> list[str]:
        """The unit is past its window: -> the objects that can now be deleted."""
        self.state[name] = "accepted"
        gone = []
        for i in self.graph.units[name].inputs:
            producer = self.graph.producer.get(i)
            if (producer is None or self.state[producer] == "accepted") and \
                    all(self.state[r] == "accepted" for r in self.graph.readers[i]):
                gone.append(i)
        return gone

    def overturn(self, name: str) -> list[str]:
        """A challenge against the unit succeeded: it and everything downstream go back to pending. -> them."""
        back = [name] + self.graph.downstream(name)
        for n in back:
            self.state[n] = "pending"
        return back


# --- hand-offs: one object per (what, chunk), lossless ------------------------------------------------------------------

def _compress(data: bytes) -> bytes:
    return _zstd.compress(data, 3) if CODEC == "zstd" else zlib.compress(data, 6)


def _decompress(data: bytes, codec: str) -> bytes:
    if codec == "zstd":
        if _zstd is None:
            raise RuntimeError("this hand-off is zstd-compressed: it needs Python 3.14, or `pip install zstandard`")
        return _zstd.decompress(data)
    return zlib.decompress(data)


def encode(tensors: dict[str, torch.Tensor]) -> bytes:
    """Tensors as one object: a JSON line (names, dtypes, shapes, the codec), then each tensor's bytes in byte
    planes (every value's first byte, then every second byte, ...) and compressed. bf16 activations' high bytes (sign
    and exponent) repeat a lot and their low bytes hardly at all, so planes compress better than the values do."""
    parts, blobs = [], []
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu()
        raw = t.reshape(-1).view(torch.uint8).numpy().reshape(-1, t.element_size())
        blobs.append(_compress(np.ascontiguousarray(raw.T).tobytes()))
        parts.append([name, str(t.dtype).removeprefix("torch."), list(t.shape), len(blobs[-1])])
    head = {"format": FORMAT, "codec": CODEC, "tensors": parts}
    return json.dumps(head, separators=(",", ":")).encode() + b"\n" + b"".join(blobs)


def decode(data: bytes) -> dict[str, torch.Tensor]:
    line, blob = data.split(b"\n", 1)
    head = json.loads(line)
    if head.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} object")
    out, at = {}, 0
    for name, dtype, shape, n in head["tensors"]:
        dt = getattr(torch, dtype)
        size = torch.empty((), dtype=dt).element_size()
        planes = np.frombuffer(_decompress(blob[at:at + n], head["codec"]), np.uint8).reshape(size, -1)
        raw = np.ascontiguousarray(planes.T).reshape(-1)
        out[name] = torch.from_numpy(raw.copy()).view(dt).reshape(shape)
        at += n
    return out


def digest(tensors: dict[str, torch.Tensor]) -> str:
    """What a commitment says an object holds: its tensors' names, dtypes, shapes and bytes, not its compression."""
    h = hashlib.sha256(FORMAT.encode())
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# --- storage -----------------------------------------------------------------------------------------------------------

class LocalStore:
    """A folder standing in for the bucket (tests, one machine)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def put(self, name: str, data: bytes) -> None:
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get(self, name: str) -> bytes:
        return (self.folder / name).read_bytes()

    def has(self, name: str) -> bool:
        return (self.folder / name).exists()

    def delete(self, name: str) -> None:
        (self.folder / name).unlink(missing_ok=True)


class PresignedStore:
    """A bucket reached through URLs the node signs for each object and method (`sign(method, name) -> url`), so a
    miner needs no account with the storage provider. Retries with growing waits: home links drop."""

    def __init__(self, sign, tries: int = 8):
        self.sign, self.tries = sign, tries

    def _call(self, method: str, name: str, data: bytes | None = None) -> bytes:
        for attempt in range(self.tries):
            req = urllib.request.Request(self.sign(method, name), data=data, method=method)
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404 or attempt == self.tries - 1:
                    raise
            except OSError:
                if attempt == self.tries - 1:
                    raise
            time.sleep(min(600, 15 * 2 ** attempt))
        raise AssertionError("unreachable")

    def put(self, name: str, data: bytes) -> None:
        self._call("PUT", name, data)

    def get(self, name: str) -> bytes:
        return self._call("GET", name)

    def has(self, name: str) -> bool:
        try:
            self._call("HEAD", name)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def delete(self, name: str) -> None:
        self._call("DELETE", name)


def presign(method: str, host: str, key: str, region: str, access_key: str, secret_key: str, expires: int = 3600,
            when: datetime.datetime | None = None) -> str:
    """A URL that lets its holder do `method` on object `key` of the bucket at `host` (say
    my-bucket.nyc3.digitaloceanspaces.com, region nyc3) for `expires` seconds: S3's Signature Version 4 in the
    query string. The node signs, holding the key; the miner only sees the URL."""
    when = when or datetime.datetime.now(datetime.timezone.utc)
    stamp, day = when.strftime("%Y%m%dT%H%M%SZ"), when.strftime("%Y%m%d")
    scope = f"{day}/{region}/s3/aws4_request"
    query = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access_key}/{scope}",
             "X-Amz-Date": stamp, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": "host"}
    q = "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(query.items()))
    path = "/" + quote(key, safe="/-_.~")
    canonical = f"{method}\n{path}\n{q}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    k = ("AWS4" + secret_key).encode()
    for part in (day, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return f"https://{host}{path}?{q}&X-Amz-Signature={hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()}"


# --- running units -------------------------------------------------------------------------------------------------------

def _get(store, name: str, digests: dict) -> dict[str, torch.Tensor]:
    got = decode(store.get(name))
    digests[name] = digest(got)
    return got


def _put(store, name: str, tensors: dict, digests: dict) -> None:
    store.put(name, encode(tensors))
    digests[name] = digest(tensors)


def run_rows(source, store, seqs, chunks, cut: top.Cut, layer: int, r: int, token_map, work: Path,
             ranges: int = 15, align: int = ROWS_PER_PIECE) -> dict:
    """A rows unit: range r of engram table `layer`, streamed once by `source` (deepseek_prep's HubSource or
    LocalSource), and for each chunk the rows of the range its tokens read. -> its commitment."""
    cfg = cut.cfg
    lo, hi = row_ranges(cfg, layer, ranges, align)[r]
    col = cfg.engram_layer_ids.index(layer)
    per_chunk = []
    for members in chunks:
        h = torch.cat([ds.engram_hashes(seqs[i][0][None], token_map, cfg)[0, :, col].reshape(-1) for i in members])
        h = torch.unique(h).numpy()
        per_chunk.append(h[(h >= lo) & (h < hi)])
    wanted = np.unique(np.concatenate(per_chunk))
    row_bytes = cfg.engram_head_dim, cfg.engram_head_dim // ds.BLOCK
    sl = source.slice(layer, wanted, work / f"rows-{layer}-{r}") if len(wanted) else \
        prep.Slice(wanted, np.zeros((0, row_bytes[0]), np.uint8), np.zeros((0, row_bytes[1]), np.uint8))
    outputs = {}
    for c, ids in enumerate(per_chunk):
        at = np.searchsorted(sl.ids, ids)
        _put(store, rows_name(c, layer, r), {"ids": torch.from_numpy(np.ascontiguousarray(ids, np.int64)),
                                             "weight": torch.from_numpy(np.ascontiguousarray(sl.weight[at])),
                                             "scale": torch.from_numpy(np.ascontiguousarray(sl.scale[at]))}, outputs)
    return {"format": FORMAT, "unit": ["rows", layer, r], "range": [lo, hi], "inputs": {}, "outputs": outputs}


def run_piece(source, store, seqs, chunks, cut: top.Cut, unit: Unit, token_map=None, work: Path | None = None,
              device="cpu", whole_layer_gb: float = 14.0, fp8: bool = False, ranges: int = 15,
              align: int = ROWS_PER_PIECE) -> dict:
    """A layers unit (piece [a, b) over chunk c, as `plan` made it), reading its inputs from `store` and writing its
    outputs there, the arithmetic deepseek_prep's (run_layer), to the bit. -> its commitment, also stored."""
    a, b, c = unit.a, unit.b, unit.chunk
    cfg, num, members = cut.cfg, ds.Numerics(), chunks[c]
    need, inputs, outputs = needs(cut, a, b), {}, {}
    if a == 0:
        embed = source.embed()
        states = []
        for i in members:
            ids = seqs[i][0][None]
            x = F.embedding(ids, embed).to(num.dtype)
            pre = torch.zeros(1, ids.size(1), cfg.hc_mult, dtype=num.hi)
            pre[..., 0] = 1
            states.append((x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1), pre, Side(), Choices()))
    else:
        got = _get(store, stream_name(c, a), inputs)
        states = [(got[f"{j}.x"], got[f"{j}.pre"], Side(), Choices()) for j in range(len(members))]
        for kind, src in need.items():
            if kind == "engram":
                continue
            got = _get(store, side_name(c, kind, src), inputs)
            for j, (_, _, side, choices) in enumerate(states):
                t = got.get(str(j))  # absent: the sequence had none (candidate blocks of a short one)
                if kind == "candidates":
                    choices.candidates = t
                else:
                    setattr(side, kind, t)
    slices = {}
    for layer in need.get("engram", []):
        parts = [_get(store, rows_name(c, layer, r), inputs) for r in range(len(row_ranges(cfg, layer, ranges, align)))]
        slices[layer] = prep.Slice(*(torch.cat([p[k] for p in parts]).numpy() for k in ("ids", "weight", "scale")))
    hashes = [[] for _ in members]
    for layer in range(a, b):
        block = prep._load_block(cfg, layer, source.layer(layer, work), device, whole_layer_gb)
        rows = None
        if layer in slices:
            col = cfg.engram_layer_ids.index(layer)
            rows = [slices[layer].rows(ds.engram_hashes(seqs[i][0][None], token_map, cfg)[:, :, col], num.dtype)
                    for i in members]
        prep.run_layer(block, states, device, rows=rows)
        for j, (x, pre, side, choices) in enumerate(states):
            hashes[j].append(prep.layer_hash(x, pre, side, choices, layer, cfg).hex())
        del block
    commit = {"format": FORMAT, "unit": ["layers", a, b, c], "inputs": inputs, "outputs": outputs,
              "layer_hashes": hashes}
    if b < cut.first:
        stream = {}
        for j, (x, pre, _, _) in enumerate(states):
            stream[f"{j}.x"], stream[f"{j}.pre"] = x, pre
        _put(store, stream_name(c, b), stream, outputs)
        for kind, src in emits(cut, a, b).items():
            made = {str(j): (choices.candidates if kind == "candidates" else getattr(side, kind))
                    for j, (_, _, side, choices) in enumerate(states)}
            _put(store, side_name(c, kind, src), {k: v for k, v in made.items() if v is not None}, outputs)
    else:  # the last piece: the job's records, as deepseek_prep writes a block
        earlier = []
        for name in (i for i in unit.inputs if i.endswith(".json")):  # the earlier pieces' commitments, in order
            data = store.get(name)
            inputs[name] = hashlib.sha256(data).hexdigest()
            earlier.append(json.loads(data)["layer_hashes"])
        every = [[h for part in earlier for h in part[j]] + hashes[j] for j in range(len(members))]
        records, leaves = [], []
        for j, i in enumerate(members):
            x, pre, side, choices = states[j]
            kept, cand = top.crossing_of(cut, side, choices, seqs[i][0].numel())
            ex = top.Example(seqs[i][0][None], seqs[i][1][None], x.cpu(), pre.cpu(), kept, cand)
            rec = top.pack(ex, 0, fp8_stream=fp8)
            records.append(rec)
            leaves.append(hashlib.sha256(hashlib.sha256(rec).digest()
                                         + b"".join(bytes.fromhex(h) for h in every[j])).hexdigest())
        bin_name, meta_name = block_names(c)
        store.put(bin_name, b"".join(records))
        meta = json.dumps({"sequences": list(members), "offsets": np.cumsum([0] + [len(r) for r in records]).tolist(),
                           "layer_hashes": every}).encode()
        store.put(meta_name, meta)
        outputs[bin_name], outputs[meta_name] = hashlib.sha256(b"".join(records)).hexdigest(), \
            hashlib.sha256(meta).hexdigest()
        commit["leaves"] = leaves
    store.put(commit_name(c, a), json.dumps(commit).encode())
    return commit


def run_local(source, store, seqs, chunks, cut: top.Cut, bounds, token_map=None, work: Path | None = None,
              device="cpu", fp8: bool = False, ranges: int = 15, align: int = ROWS_PER_PIECE) -> dict:
    """Every unit of a plan, in order, on this machine; then the manifest, as deepseek_prep.run's. -> it."""
    graph = plan(cut, chunks, bounds, ranges, align)
    work = Path(work) if work else ds.settings.home() / "split"
    commits = {}
    for u in graph.order():
        if u.kind == "rows":
            commits[u.name] = run_rows(source, store, seqs, chunks, cut, u.a, u.b, token_map, work, ranges, align)
        else:
            commits[u.name] = run_piece(source, store, seqs, chunks, cut, u, token_map, work, device, fp8=fp8,
                                        ranges=ranges, align=align)
    return manifest(cut, seqs, chunks, [commits[u.name] for u in graph.order()
                                        if u.kind == "layers" and u.b == cut.first], fp8)


def manifest(cut: top.Cut, seqs, chunks, last: list[dict], fp8: bool = False) -> dict:
    """The dataset's manifest from the last piece's commitments, chunk by chunk: deepseek_prep's root."""
    last = sorted(last, key=lambda m: m["unit"][3])
    leaves = [bytes.fromhex(leaf) for m in last for leaf in m["leaves"]]
    return {"format": prep.FORMAT, "repo": ds.REPO, "commit": ds.COMMIT, "cut": cut.first, "fp8_stream": fp8,
            "sequences": len(seqs), "tokens": sum(len(seqs[i][0]) for ch in chunks for i in ch),
            "blocks": [{"file": block_names(c)[0], "first": ch[0], "count": len(ch), "leaves": m["leaves"]}
                       for c, (ch, m) in enumerate(zip(chunks, last))],
            "root": merkle_root(leaves).hex()}
