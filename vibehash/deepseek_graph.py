"""The split prep's work graph and storage, without PyTorch (DEEPSEEK.md §8, "The split prep"): the node plans a job's
units, hands them out and keeps their state here, and signs storage links, with no model in sight; the units
themselves run in deepseek_split, which re-exports all of this.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .deepseek_config import Config, Cut

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


def needs(cut: Cut, a: int, b: int) -> dict:
    """What piece [a, b) reads besides the stream: Cut.crossing's {name: source layer} for its layers, and
    "engram": its engram layers (their rows). The last piece also reads what the trained layers reuse, to write it
    into the records."""
    out = {} if a == 0 else Cut(cut.cfg, a).crossing(None if b == cut.first else b)
    out.pop("engram", None)
    engram = [layer for layer in cut.cfg.engram_layer_ids if a <= layer < b]
    return {**out, "engram": engram} if engram else out


def emits(cut: Cut, a: int, b: int) -> dict[str, int]:
    """The side channels piece [a, b) makes that layers at or above b reuse: {name: source layer}."""
    if b == cut.first:
        return {}  # the last piece writes them into the records
    return {k: v for k, v in Cut(cut.cfg, b).crossing().items() if k != "engram" and a <= v < b}


def row_ranges(cfg: Config, layer: int, n: int, align: int = ROWS_PER_PIECE) -> list[tuple[int, int]]:
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


def rows_commit_name(layer: int, r: int) -> str:
    """Where a rows unit's commitment goes (a layers unit's is commit_name)."""
    return f"rows-{layer:02d}-{r:03d}.json"


def unit_commit(unit) -> str:
    """The object holding a unit's commitment, which its reveal names by hash."""
    return rows_commit_name(unit.a, unit.b) if unit.kind == "rows" else commit_name(unit.chunk, unit.a)


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


def plan(cut: Cut, chunks: list[list[int]], bounds: list[tuple[int, int]], ranges: int = 15,
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
