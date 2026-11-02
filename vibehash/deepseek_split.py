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

There is no preparer: the protocol knows only units, and a big machine simply takes many of them. Anyone who can
hold a piece's layers can run a unit, replay it to check it, or challenge it; the node only schedules and referees.

Storage is a Store (put, get, has, delete), and nothing about a host is trusted. A reader checks every object
against the digest its writer committed to (`want`, from the node), so a host that alters one is caught
(Tampered). Mirrors reads from the first of several hosts with the committed copy: the bucket, the writer's own
storage, peers. So a host can only withhold, and a withheld object can be made again by anyone holding its piece,
from its inputs, to the bit (Line.redo). v1's host is a bucket the operator pays for: PresignedStore uses URLs the
node signs (`presign`: S3's query-string signature, which DigitalOcean Spaces takes), so miners need no storage
account of their own. LocalStore is a folder. Big objects go up in parts and come down in ranges, each retried on its own, and an
upload resumes after a crash: a 1.8 GB hand-off survives a home link. `Line` is the node's side: which units can be
claimed, deadlines that a slow miner can meet, optimistic consumption of outputs not yet past their challenge
window, rollback when a challenge succeeds, redoing a unit whose output went missing, and which objects can be
deleted.

Checking: `replay` runs a committed unit again from its inputs and says where it disagrees, the first layer for each
sequence; `state_before` serves what enters that layer, which both sides of a dispute hand the referee. Each unit's
commitment also carries what it cost (seconds, tokens a second, peak GPU memory, bytes moved): the first live job
measures the real numbers.

Wiring into the node waits for PR #5's work units. run_local runs a whole plan on one machine, in order: a dev and
test tool, not a role.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import http.client
import json
import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
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
        self.due: dict[str, float | None] = {}
        self.ran: dict[str, tuple[str | None, float]] = {}  # unit -> (who finished it, when)

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

    def claim(self, name: str, now: float = 0.0, seconds: float | None = None) -> None:
        """A miner takes the unit, with `seconds` to finish it (see `deadline`; None: no limit)."""
        assert self.state[name] == "pending", (name, self.state[name])
        self.state[name] = "claimed"
        self.due[name] = None if seconds is None else now + seconds

    def expire(self, now: float) -> list[str]:
        """Claimed units past their deadline go back to pending, for another miner. -> them."""
        late = [n for n, st in self.state.items() if st == "claimed" and self.due.get(n) is not None
                and now > self.due[n]]
        for n in late:
            self.state[n] = "pending"
        return late

    def done(self, name: str, miner: str | None = None, now: float = 0.0) -> None:
        self.state[name] = "done"
        self.ran[name] = (miner, now)

    def summary(self, now: float, recent: float = 600.0) -> dict:
        """The job summary's "prep" field, which the front page draws as the assembly line (netstats documents it):
        {"layers": how many frozen layers, "done": the fraction of each one's units accepted, bottom first,
        "pieces_done" and "pieces_total" (layers units accepted, and all of them), "miners": how many miners
        finished a unit in the last `recent` seconds}."""
        pieces = [u for u in self.graph.units.values() if u.kind == "layers"]
        layers = max(u.b for u in pieces)
        done = []
        for layer in range(layers):
            mine = [u for u in pieces if u.a <= layer < u.b]
            done.append(sum(self.state[u.name] == "accepted" for u in mine) / len(mine))
        lately = {m for name, (m, t) in self.ran.items() if m is not None and now - t <= recent
                  and self.state[name] in ("done", "accepted")}
        return {"layers": layers, "done": done, "pieces_done": sum(self.state[u.name] == "accepted" for u in pieces),
                "pieces_total": len(pieces), "miners": len(lately)}

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

    def redo(self, name: str) -> None:
        """What the unit wrote went missing (withheld, or lost) before every reader had it: it runs again. Units are
        deterministic, so it writes the same objects to the bit, and nothing downstream changes. If its own inputs
        are gone too, their writers are redone first, back to the token ids and the public weights if need be."""
        self.state[name] = "pending"

    def overturn(self, name: str) -> list[str]:
        """A challenge against the unit succeeded: it and everything downstream go back to pending. -> them."""
        back = [name] + self.graph.downstream(name)
        for n in back:
            self.state[n] = "pending"
        return back


def deadline(read_bytes: float, write_bytes: float, layer_tokens: float, slack: float = 2.0, down: float = 3e6,
             up: float = 1.25e6, gpu_ms: float = 1.0, floor: float = 600.0) -> float:
    """Seconds a miner gets for a unit: its transfers at the slowest link the network takes (25 Mbit/s down, 10 up),
    its compute (layers x tokens) at 1 ms a token a layer (about 4x §7's estimate for a 4070 SUPER), times `slack`,
    and at least `floor`. Generous on purpose: one too short throws a slow miner's work away and stalls the line;
    one too long only delays reclaiming a vanished miner's unit. Units report what they really took (their
    commitment's "stats"), which says how far the first live job's numbers let later jobs tighten it."""
    return max(floor, slack * (read_bytes / down + write_bytes / up + layer_tokens * gpu_ms / 1e3))


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

class Tampered(Exception):
    """An object isn't what its writer committed to: every host that had it altered it."""


class Missing(LookupError):
    """No host has an object: the node redoes the unit that wrote it (Line.redo)."""


class Mirrors:
    """Several hosts for the same objects, tried in order: the bucket, the writer's own storage, peers. A unit
    writes to the first. Reading (`_get`) takes the first copy that matches its committed digest, skipping hosts
    that don't have it or altered it."""

    def __init__(self, *stores):
        self.stores = stores

    def put(self, name: str, data: bytes) -> None:
        self.stores[0].put(name, data)

    def get(self, name: str) -> bytes:
        for s in self.stores:
            try:
                return s.get(name)
            except OSError:
                continue
        raise Missing(name)

    def has(self, name: str) -> bool:
        return any(s.has(name) for s in self.stores)

    def delete(self, name: str) -> None:
        self.stores[0].delete(name)


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
    """A bucket reached through URLs the node signs, `sign(method, name, query) -> url` (query: S3's sub-resource
    parameters, as for a part of a multipart upload; {} for a plain object), so a miner needs no account with the
    storage provider.

    Home links drop, so nothing big travels in one request. An object over `part` bytes goes up as S3's multipart
    upload, each part retried on its own with growing waits, and its progress kept in `journal` (a folder): run
    again after a crash, it sends only the parts not yet acknowledged. An object comes down in ranges of `part`
    bytes, each retried on its own."""

    PART = 64 * 2**20  # S3 takes parts of 5 MiB to 5 GiB, at most 10,000: up to 640 GB at this size

    def __init__(self, sign, tries: int = 8, part: int = PART, journal: Path | None = None, wait=time.sleep):
        self.sign, self.tries, self.part, self.wait = sign, tries, part, wait
        self.journal = Path(journal) if journal else None

    def _call(self, method: str, name: str, query: dict | None = None, data: bytes | None = None,
              headers: dict | None = None):
        """-> (body, response headers). A fresh URL each try; a 404 isn't retried."""
        for attempt in range(self.tries):
            req = urllib.request.Request(self.sign(method, name, query or {}), data=data, method=method,
                                         headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read(), r.headers
            except urllib.error.HTTPError as e:
                if e.code == 404 or attempt == self.tries - 1:
                    raise
                e.close()
            except (OSError, http.client.HTTPException):  # dropped, reset, cut short
                if attempt == self.tries - 1:
                    raise
            self.wait(min(600, 15 * 2 ** attempt))
        raise AssertionError("unreachable")

    def put(self, name: str, data: bytes) -> None:
        if len(data) <= self.part:
            self._call("PUT", name, data=data)
        else:
            self._multipart(name, data)

    def _multipart(self, name: str, data: bytes, fresh: bool = False) -> None:
        tag = hashlib.sha256(name.encode() + b"\0" + hashlib.sha256(data).digest()).hexdigest()[:40]
        path = self.journal / f"{tag}.json" if self.journal else None
        state = None if fresh or path is None or not path.exists() else json.loads(path.read_text())
        if state is None:
            body, _ = self._call("POST", name, {"uploads": ""})
            state = {"name": name, "upload": _xml_text(body, "UploadId"), "parts": {}}
            self._save(path, state)
        n = -(-len(data) // self.part)
        for i in range(1, n + 1):
            if str(i) in state["parts"]:
                continue
            try:
                _, headers = self._call("PUT", name, {"partNumber": str(i), "uploadId": state["upload"]},
                                        data[(i - 1) * self.part:i * self.part])
            except urllib.error.HTTPError as e:
                if e.code == 404 and not fresh:  # the upload expired or was aborted: start it again
                    e.close()
                    return self._multipart(name, data, fresh=True)
                raise
            state["parts"][str(i)] = headers["ETag"]
            self._save(path, state)
        parts = "".join(f"<Part><PartNumber>{i}</PartNumber><ETag>{state['parts'][str(i)]}</ETag></Part>"
                        for i in range(1, n + 1))
        try:
            body, _ = self._call("POST", name, {"uploadId": state["upload"]},
                                 f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>".encode())
        except urllib.error.HTTPError as e:  # a 404 here can mean the completion went through and its answer didn't
            if e.code != 404 or self._size(name) != len(data):
                raise
            e.close()
            body = b""
        if b"<Error>" in body:  # S3 can answer 200 and still fail a completion
            raise OSError(f"completing the upload of {name}: {body[:300]!r}")
        if path is not None:
            path.unlink(missing_ok=True)

    def _save(self, path: Path | None, state: dict) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)

    def _size(self, name: str) -> int | None:
        try:
            _, headers = self._call("HEAD", name)
        except urllib.error.HTTPError as e:
            e.close()
            return None
        return int(headers.get("Content-Length", -1))

    def get(self, name: str) -> bytes:
        first, headers = self._call("GET", name, headers={"Range": f"bytes=0-{self.part - 1}"})
        total = int(headers["Content-Range"].rsplit("/", 1)[1]) if headers.get("Content-Range") else len(first)
        pieces, at = [first], len(first)
        while at < total:
            piece, _ = self._call("GET", name, headers={"Range": f"bytes={at}-{min(at + self.part, total) - 1}"})
            if not piece:
                raise OSError(f"{name}: an empty range at byte {at} of {total}")
            pieces.append(piece)
            at += len(piece)
        return b"".join(pieces)

    def has(self, name: str) -> bool:
        return self._size(name) is not None

    def delete(self, name: str) -> None:
        try:
            self._call("DELETE", name)
        except urllib.error.HTTPError as e:
            if e.code != 404:  # already gone: what was wanted
                raise
            e.close()


def _xml_text(body: bytes, tag: str) -> str:
    """The text of the first element named `tag` in an S3 answer, whatever its namespace."""
    for el in ElementTree.fromstring(body).iter():
        if el.tag.rsplit("}", 1)[-1] == tag:
            return el.text or ""
    raise OSError(f"no <{tag}> in {body[:300]!r}")


class Signer:
    """The node's side of PresignedStore, for a bucket whose key it holds: `signer(method, name, query)` -> a URL
    for object prefix + name. The node checks that a unit may read or write a name before it signs for it."""

    def __init__(self, host: str, region: str, access_key: str, secret_key: str, prefix: str = "",
                 expires: int = 3600, scheme: str = "https"):
        self.host, self.region, self.access_key, self.secret_key = host, region, access_key, secret_key
        self.prefix, self.expires, self.scheme = prefix, expires, scheme

    def __call__(self, method: str, name: str, query: dict | None = None) -> str:
        return presign(method, self.host, self.prefix + name, self.region, self.access_key, self.secret_key,
                       self.expires, query=query, scheme=self.scheme)


def presign(method: str, host: str, key: str, region: str, access_key: str, secret_key: str, expires: int = 3600,
            when: datetime.datetime | None = None, query: dict | None = None, scheme: str = "https") -> str:
    """A URL that lets its holder do `method` on object `key` of the bucket at `host` (say
    my-bucket.nyc3.digitaloceanspaces.com, region nyc3) for `expires` seconds: S3's Signature Version 4 in the
    query string. `query`: sub-resource parameters signed with it ({"uploads": ""} starts a multipart upload,
    {"partNumber": "3", "uploadId": ...} is its third part). The node signs, holding the key; the miner only sees
    the URL, and can't change its method, object or parameters without breaking the signature."""
    when = when or datetime.datetime.now(datetime.timezone.utc)
    stamp, day = when.strftime("%Y%m%dT%H%M%SZ"), when.strftime("%Y%m%d")
    scope = f"{day}/{region}/s3/aws4_request"
    query = {**{k: str(v) for k, v in (query or {}).items()},
             "X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access_key}/{scope}",
             "X-Amz-Date": stamp, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": "host"}
    q = "&".join(f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(query.items()))
    path = "/" + quote(key, safe="/-_.~")
    canonical = f"{method}\n{path}\n{q}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    k = ("AWS4" + secret_key).encode()
    for part in (day, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return f"{scheme}://{host}{path}?{q}&X-Amz-Signature={hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()}"


# --- running units -------------------------------------------------------------------------------------------------------

def _get(store, name: str, digests: dict, want: dict | None = None, io: dict | None = None) -> dict[str, torch.Tensor]:
    """An object's tensors from the first host whose copy matches the digest its writer committed to (want[name];
    none: any copy)."""
    altered = False
    for host in (store.stores if isinstance(store, Mirrors) else (store,)):
        t0 = time.perf_counter()
        try:
            data = host.get(name)
            got = decode(data)
        except OSError:  # this host hasn't got it (a 404 is an OSError too)
            continue
        except ValueError:  # not even an object
            altered = True
            continue
        if io is not None:
            io["read_bytes"] += len(data)
            io["read_seconds"] += time.perf_counter() - t0
        d = digest(got)
        if want is None or name not in want or d == want[name]:
            digests[name] = d
            return got
        altered = True
    raise Tampered(name) if altered else Missing(name)


def _put(store, name: str, obj, digests: dict, io: dict | None = None) -> None:
    """Write an object (tensors, encoded; or bytes as they are) and note its digest."""
    data = encode(obj) if isinstance(obj, dict) else obj
    t0 = time.perf_counter()
    store.put(name, data)
    if io is not None:
        io["wrote_bytes"] += len(data)
        io["write_seconds"] += time.perf_counter() - t0
    digests[name] = _digest_of(obj)


def _digest_of(obj) -> str:
    return digest(obj) if isinstance(obj, dict) else hashlib.sha256(obj).hexdigest()


def _io() -> dict:
    return {"read_bytes": 0, "read_seconds": 0.0, "wrote_bytes": 0, "write_seconds": 0.0}


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


def _compute(source, store, seqs, chunks, cut: top.Cut, unit: Unit, token_map, work, device, whole_layer_gb: float,
             ranges: int, align: int, want: dict | None, only=None, stop: int | None = None, io: dict | None = None):
    """A layers unit's arithmetic: its inputs read (and checked against `want`), then its layers a..stop-1 (stop:
    the piece's end) over its chunk's sequences, or over those at positions `only` in the chunk (each sequence's
    arithmetic is its own, so a sequence replayed alone hashes as it did in its chunk). -> (states, each sequence's
    hash after each layer, the inputs' digests, the positions)."""
    a, b, c = unit.a, unit.b, unit.chunk
    stop = b if stop is None else stop
    cfg, num = cut.cfg, ds.Numerics()
    pos = list(range(len(chunks[c]))) if only is None else list(only)
    members = [chunks[c][j] for j in pos]
    need, inputs = needs(cut, a, b), {}
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
        got = _get(store, stream_name(c, a), inputs, want, io)
        states = [(got[f"{j}.x"], got[f"{j}.pre"], Side(), Choices()) for j in pos]
        for kind, src in need.items():
            if kind == "engram":
                continue
            got = _get(store, side_name(c, kind, src), inputs, want, io)
            for (_, _, side, choices), j in zip(states, pos):
                t = got.get(str(j))  # absent: the sequence had none (candidate blocks of a short one)
                if kind == "candidates":
                    choices.candidates = t
                else:
                    setattr(side, kind, t)
    slices = {}
    for layer in (layer for layer in need.get("engram", []) if layer < stop):
        parts = [_get(store, rows_name(c, layer, r), inputs, want, io)
                 for r in range(len(row_ranges(cfg, layer, ranges, align)))]
        slices[layer] = prep.Slice(*(torch.cat([p[k] for p in parts]).numpy() for k in ("ids", "weight", "scale")))
    hashes = [[] for _ in pos]
    for layer in range(a, stop):
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
    return states, hashes, inputs, pos


def _outputs(store, seqs, chunks, cut: top.Cut, unit: Unit, states, hashes, inputs: dict, fp8: bool = False):
    """What a layers unit writes, {name: tensors, or bytes}, and for the last piece its records' leaves."""
    a, b, c = unit.a, unit.b, unit.chunk
    members, out = chunks[c], {}
    if b < cut.first:
        stream = {}
        for j, (x, pre, _, _) in enumerate(states):
            stream[f"{j}.x"], stream[f"{j}.pre"] = x, pre
        out[stream_name(c, b)] = stream
        for kind, src in emits(cut, a, b).items():
            made = {str(j): (choices.candidates if kind == "candidates" else getattr(side, kind))
                    for j, (_, _, side, choices) in enumerate(states)}
            out[side_name(c, kind, src)] = {k: v for k, v in made.items() if v is not None}
        return out, None
    earlier = []  # the last piece: the job's records, as deepseek_prep writes a block
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
    out[bin_name] = b"".join(records)
    out[meta_name] = json.dumps({"sequences": list(members),
                                 "offsets": np.cumsum([0] + [len(r) for r in records]).tolist(),
                                 "layer_hashes": every}).encode()
    return out, leaves


def run_piece(source, store, seqs, chunks, cut: top.Cut, unit: Unit, token_map=None, work: Path | None = None,
              device="cpu", whole_layer_gb: float = 14.0, fp8: bool = False, ranges: int = 15,
              align: int = ROWS_PER_PIECE, want: dict | None = None) -> dict:
    """A layers unit (piece [a, b) over chunk c, as `plan` made it), reading its inputs from `store` (a host, or
    Mirrors of several) and writing its outputs there, the arithmetic deepseek_prep's (run_layer), to the bit. want:
    {object: digest} from its inputs' writers' commitments, which the node holds; an input that doesn't match is
    Tampered. -> its commitment, also stored. Its "stats" (seconds, tokens a second, peak GPU memory, bytes and
    seconds of transfer) aren't hashed by anything: they're the run's measurements."""
    io = _io()
    with top.Meter(device) as meter:
        states, hashes, inputs, _ = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device,
                                             whole_layer_gb, ranges, align, want, io=io)
    objects, leaves = _outputs(store, seqs, chunks, cut, unit, states, hashes, inputs, fp8)
    outputs = {}
    for name, obj in objects.items():
        _put(store, name, obj, outputs, io)
    commit = {"format": FORMAT, "unit": ["layers", unit.a, unit.b, unit.chunk], "inputs": inputs, "outputs": outputs,
              "layer_hashes": hashes}
    if leaves is not None:
        commit["leaves"] = leaves
    tokens = sum(len(seqs[i][0]) for i in chunks[unit.chunk]) * (unit.b - unit.a)
    commit["stats"] = {**meter.stats(tokens), **{k: round(v, 3) for k, v in io.items()}}
    store.put(commit_name(unit.chunk, unit.a), json.dumps(commit).encode())
    return commit


def replay(source, store, seqs, chunks, cut: top.Cut, unit: Unit, commit: dict, token_map=None,
           work: Path | None = None, device="cpu", whole_layer_gb: float = 14.0, fp8: bool = False, ranges: int = 15,
           align: int = ROWS_PER_PIECE, want: dict | None = None) -> dict:
    """Check a committed layers unit: run it again from the same inputs (checked against `want`), writing nothing.
    Anyone holding the piece's layers can. -> {"agrees", "disagree": [(sequence's position in the chunk, the first
    layer whose hash differs)], "outputs": [objects whose digest differs]}. A disagreement is where a challenge
    starts: what enters that layer is agreed (the hash before it matches, or it's the unit's checked input), what
    leaves it isn't, and state_before gives both sides' referee what enters it."""
    states, hashes, inputs, _ = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device,
                                         whole_layer_gb, ranges, align, want)
    objects, leaves = _outputs(store, seqs, chunks, cut, unit, states, hashes, inputs, fp8)
    disagree = []
    for j, (mine, theirs) in enumerate(zip(hashes, commit["layer_hashes"])):
        for k, h in enumerate(mine):
            if k >= len(theirs) or theirs[k] != h:
                disagree.append((j, unit.a + k))
                break
    outs = [n for n, obj in objects.items() if commit["outputs"].get(n) != _digest_of(obj)]
    if leaves is not None and leaves != commit.get("leaves"):
        outs.append("leaves")
    return {"agrees": not disagree and not outs, "disagree": disagree, "outputs": outs}


def state_before(source, store, seqs, chunks, cut: top.Cut, unit: Unit, j: int, layer: int, token_map=None,
                 work: Path | None = None, device="cpu", whole_layer_gb: float = 14.0, ranges: int = 15,
                 align: int = ROWS_PER_PIECE, want: dict | None = None) -> tuple[dict, str | None]:
    """What enters `layer` for the sequence at position j of a layers unit's chunk, replayed from the unit's inputs:
    -> ({"x", "pre", and the side channels and candidate blocks it reuses}, its hash after layer - 1 as the unit
    committed it, or None at the piece's first layer, whose input is a checked object). Either side of a dispute at
    (j, layer) can serve it, so the bisection inside the layer starts from an agreed state."""
    states, hashes, _, _ = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device,
                                    whole_layer_gb, ranges, align, want, only=[j], stop=layer)
    x, pre, side, choices = states[0]
    out = {"x": x, "pre": pre}
    for kind in SIDE:
        if getattr(side, kind) is not None:
            out[kind] = getattr(side, kind)
    if choices.candidates is not None:
        out["candidates"] = choices.candidates
    return out, (hashes[0][-1] if hashes[0] else None)


def run_local(source, store, seqs, chunks, cut: top.Cut, bounds, token_map=None, work: Path | None = None,
              device="cpu", fp8: bool = False, ranges: int = 15, align: int = ROWS_PER_PIECE) -> dict:
    """Every unit of a plan, in order, on this machine, each checking its inputs against their writers'
    commitments; then the manifest, as deepseek_prep.run's. -> it. A dev and test tool: on the network, a machine
    that can take every unit takes them as units like anyone else's."""
    graph = plan(cut, chunks, bounds, ranges, align)
    work = Path(work) if work else ds.settings.home() / "split"
    commits, committed = {}, {}
    for u in graph.order():
        if u.kind == "rows":
            commits[u.name] = run_rows(source, store, seqs, chunks, cut, u.a, u.b, token_map, work, ranges, align)
        else:
            commits[u.name] = run_piece(source, store, seqs, chunks, cut, u, token_map, work, device, fp8=fp8,
                                        ranges=ranges, align=align, want=committed)
        committed.update(commits[u.name]["outputs"])
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
