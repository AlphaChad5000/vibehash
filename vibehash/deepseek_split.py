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

import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import deepseek as ds
from . import deepseek_prep as prep
from . import deepseek_top as top
from .crypto import merkle_root
from .deepseek import Choices, Side
from .deepseek_graph import (  # noqa: F401 - re-exported: the graph and storage, without PyTorch
    ROWS_PER_PIECE, SIDE, pieces, needs, emits, row_ranges, stream_name, side_name, rows_name, commit_name,
    block_names, Unit, Graph, plan, Line, deadline, Tampered, Missing, Mirrors, LocalStore, PresignedStore, Signer,
    presign, rows_commit_name, unit_commit, check_block, unpack_np, RecordReader)

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
    commit = {"format": FORMAT, "unit": ["rows", layer, r], "range": [lo, hi], "inputs": {}, "outputs": outputs}
    store.put(rows_commit_name(layer, r), json.dumps(commit).encode())
    return commit


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


def distance(mine: torch.Tensor | None, theirs: torch.Tensor | None) -> float:
    """How far a replayed output is from a committed one: the relative RMS of numbers (the stream, keys), the share
    that differ of ids and flags (picks, candidate blocks); infinite if the shapes differ or only one exists."""
    if mine is None or theirs is None:
        return 0.0 if mine is None and theirs is None else float("inf")
    if mine.shape != theirs.shape:
        return float("inf")
    if mine.dtype.is_floating_point and mine.dtype.itemsize > 1:
        a, b = mine.float(), theirs.float()
        return float((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-30)) if a.numel() else 0.0
    return float((mine.view(torch.uint8) != theirs.view(torch.uint8)).float().mean()) if mine.numel() else 0.0


def replay_close(source, store, seqs, chunks, cut: top.Cut, unit: Unit, commit: dict, tolerance: float,
                 token_map=None, work: Path | None = None, device="cpu", whole_layer_gb: float = 14.0,
                 fp8: bool = False, ranges: int = 15, align: int = ROWS_PER_PIECE, want: dict | None = None) -> dict:
    """A check across hardware: the unit run again from its checked inputs, its outputs compared with the ones it
    committed to (read from `store`, checked against the commitment) within `tolerance` (distance), since the
    reference's arithmetic differs a little between devices. On the same hardware they agree to the bit, and this
    says so without comparing. -> {"agrees", "worst": the largest distance, "where": which output, and which
    tensor of it}."""
    states, hashes, inputs, _ = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device,
                                         whole_layer_gb, ranges, align, want)
    objects, leaves = _outputs(store, seqs, chunks, cut, unit, states, hashes, inputs, fp8)
    if hashes == commit["layer_hashes"] and all(commit["outputs"].get(n) == _digest_of(o) for n, o in objects.items()):
        return {"agrees": True, "worst": 0.0, "where": ""}
    worst, where = 0.0, ""

    def note(d: float, label: str) -> None:
        nonlocal worst, where
        if d >= worst:
            worst, where = d, label

    for name, obj in objects.items():
        if isinstance(obj, dict):
            theirs = _get(store, name, {}, {name: commit["outputs"].get(name, "missing")})
            for k in sorted(set(obj) | set(theirs)):
                note(distance(obj.get(k), theirs.get(k)), f"{name}:{k}")
        elif name.endswith(".bin"):
            meta_name = name[:-4] + ".json"
            data, meta = store.get(name), store.get(meta_name)
            if hashlib.sha256(data).hexdigest() != commit["outputs"].get(name) or                     hashlib.sha256(meta).hexdigest() != commit["outputs"].get(meta_name):
                raise Tampered(f"{name} isn't what the unit committed to")
            ours_meta, their_meta = json.loads(objects[meta_name]), json.loads(meta)
            for j in range(len(ours_meta["sequences"])):
                mine = top.unpack(obj[ours_meta["offsets"][j]:ours_meta["offsets"][j + 1]])
                theirs = top.unpack(data[their_meta["offsets"][j]:their_meta["offsets"][j + 1]])
                for k in sorted(set(mine) | set(theirs)):
                    note(distance(mine.get(k), theirs.get(k)), f"{name}:{j}:{k}")
    return {"agrees": worst <= tolerance, "worst": worst, "where": where}


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


def read_block(store, manifest: dict, c: int, root: str | None = None) -> list[tuple[int, dict]]:
    """Chunk c's records, as a trainer or a scorer reads its shard from the bucket, checked (check_block). ->
    [(the sequence's index in the dataset, its record's tensors)]."""
    bin_name, meta_name = block_names(c)
    return [(i, top.unpack(rec)) for i, rec in check_block(manifest, c, store.get(bin_name), store.get(meta_name), root)]
