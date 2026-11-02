"""The split prep (DEEPSEEK.md §8, "The split prep"): the frozen layers' one-time pass over a job's text, cut into
small units that ordinary miners run, connected by stored hand-offs. Nobody has to be online with anybody else: the
bottom is frozen, so a unit's output depends only on its inputs, and each output waits in storage for the unit that
reads it. An asynchronous assembly line.

The units of the work graph `plan` makes:
- rows (engram layer L, row range r): rows [lo, hi) of L's 101 GB table streamed once (about 7 GB), keeping, for
  each chunk of the text, the rows its tokens' n-grams hash to in that range. Needs no other unit.
- layers (piece [a, b), chunk c): layers a..b-1 over chunk c's sequences. It reads the stream entering layer a, the
  side channels that cross the cut at a (each stored once, by the piece that made it, and read by every piece that
  reuses it: never relayed), and an engram layer's rows. It computes op by op (deepseek_program.PrepProgram, run by
  deepseek_tracer), writes the stream entering layer b and the side channels it made that later layers reuse, and
  commits to them and to the state root before it and after each of its layers, so a dispute bisects a unit layer
  by layer, then op by op (referee_deepseek.py). A state between layers holds only what crosses the cut above it, so
  its root doesn't depend on where the pieces end. The last piece writes the job's dataset records instead of a
  stream, in deepseek_prep's format; each record's leaf hashes it with the chunk's roots.

Hand-offs are exact. The stream stays bf16 between frozen layers, so the records are the model's own; what FP8 at every cut would cost the model isn't measured (scripts/deepseek_stream_stats.py: its
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

Checking: `replay_close` runs a committed unit again from its inputs: on the same hardware its roots are the
committed ones, to the bit; across hardware its outputs agree within a tolerance. Its first root must be the state
its inputs make, and its last the state its outputs hand on, exactly. Each unit's commitment also carries what it cost (seconds, tokens a second, peak GPU memory, bytes moved): the first live job
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


class _Weights:
    """The weights a prep unit's ops read, by checkpoint name, from `source` (deepseek_prep's LocalSource or
    HubSource): one layer's at a time (the one its ops are on), the embedding, and `extra` (the engram token map)."""

    def __init__(self, source, work, extra: dict | None = None):
        self.source, self.work, self.extra = source, work, dict(extra or {})
        self.layer, self.held = None, {}

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name) -> bool:
        return self.get(name) is not None

    def __getitem__(self, name):
        if name in self.extra:
            return self.extra[name]
        if name == "embed.weight":
            return self.source.embed()
        if not name.startswith("layers."):
            raise KeyError(name)
        layer = int(name.split(".")[1])
        if layer != self.layer:
            self.held, self.layer = {}, None
            self.held, self.layer = self.source.layer(layer, self.work), layer
        return self.held[name]


def tokens_of(seqs):
    """The token file deepseek_prep's sequences [(ids, learn mask)] make (deepseek_job.TokenFile), for the program."""
    from .deepseek_job import TokenFile
    ids = torch.cat([s[0].reshape(-1) for s in seqs]).numpy().astype(np.int64)
    lengths = np.array([s[0].numel() for s in seqs], dtype=np.int64)
    learn = torch.cat([torch.cat((torch.zeros(1, dtype=torch.bool), s[1].reshape(-1)[:-1].bool())) for s in seqs])
    return TokenFile(ids, lengths, learn.numpy().astype(np.uint8))


def state_root(st: dict) -> str:
    """The merkle root of a program state (dispute.state_manifest's), as a commitment names it."""
    from . import dispute, merkle
    m, _ = dispute.state_manifest({k: v.detach().cpu().numpy() for k, v in st.items()})
    return merkle.state_root(m)


def _program(seqs, chunks, cut: top.Cut, unit: Unit):
    import types
    from .deepseek_program import PrepProgram
    return PrepProgram(cut, types.SimpleNamespace(tokens=tokens_of(seqs)), chunks[unit.chunk], unit)


def _start(store, prog, cut: top.Cut, unit: Unit, inputs: dict, want: dict | None, io) -> dict:
    """The state a layers unit starts from: empty at layer 0, else the stream and side channels the unit below handed
    on, read from `store` (checked against `want`)."""
    from .deepseek_program import SIDE_PREFIX
    if unit.a == 0:
        return {}
    c = unit.chunk
    got = _get(store, stream_name(c, unit.a), inputs, want, io)
    n = len(prog.T)
    st = {"h": torch.cat([got[f"{j}.x"].reshape(prog.T[j], -1).float() for j in range(n)]),
          "p": torch.cat([got[f"{j}.pre"].reshape(prog.T[j], -1).float() for j in range(n)])}
    for kind, src in prog.side_kinds(unit.a).items():
        obj = _get(store, side_name(c, kind, src), inputs, want, io)
        for j in range(n):
            if prog.n_comp(j, src):
                st[f"{SIDE_PREFIX[kind]}{j}"] = obj[str(j)].reshape(prog.side_shape(kind, j, src)).float()
    return st


def _compute(source, store, seqs, chunks, cut: top.Cut, unit: Unit, token_map, work, device, ranges: int,
             align: int, want: dict | None, io: dict | None = None, tamper=None):
    """A layers unit's arithmetic: deepseek_program.PrepProgram's ops (deepseek_tracer), from the state the unit
    below handed on (checked against `want`), its engram layers' rows from the rows units. -> (the final state,
    the state root before the unit and after each of its segments, the inputs' digests)."""
    import types
    from .deepseek_tracer import Executor
    prog = _program(seqs, chunks, cut, unit)
    inputs = {}
    st = {k: v.to(device) for k, v in _start(store, prog, cut, unit, inputs, want, io).items()}
    slices = {}
    for layer in (layer for layer in cut.cfg.engram_layer_ids if unit.a <= layer < unit.b):
        parts = [_get(store, rows_name(unit.chunk, layer, r), inputs, want, io)
                 for r in range(len(row_ranges(cut.cfg, layer, ranges, align)))]
        slices[layer] = prep.Slice(*(torch.cat([p[k] for p in parts]).numpy() for k in ("ids", "weight", "scale")))
    rt = types.SimpleNamespace(model=None, device=device, weights=_Weights(source, work))
    ex = Executor(rt, types.SimpleNamespace(cfg=cut.cfg, lora_a=lambda: []), prog, tamper)
    ex.slices, ex.token_map = slices, token_map
    roots = [state_root(st)]
    with torch.no_grad():
        for s in range(prog.segments):
            node = prog.segment(s)
            ex.run_atoms(node, 0, len(node.atoms), st, (node.label,))
            roots.append(state_root(st))
    return prog, st, roots, inputs


def _side_tensor(kind: str, t: torch.Tensor) -> torch.Tensor:
    """A side channel as the hand-offs and records keep it, from the state's float32."""
    if kind == "picks":
        return t.long()[None]
    if kind == "candidates":
        return (t != 0)[None]
    return t.to(torch.bfloat16)[None]


def _outputs(store, seqs, chunks, cut: top.Cut, unit: Unit, prog, st: dict, roots: list, inputs: dict,
             fp8: bool = False):
    """What a layers unit writes, {name: tensors, or bytes}, from its final state, and for the last piece its
    records' leaves: each the sha256 of the record and the chunk's state roots after every layer below the cut."""
    from .deepseek_program import SIDE_PREFIX
    a, b, c = unit.a, unit.b, unit.chunk
    cfg, members, out = cut.cfg, chunks[c], {}
    st = {k: v.cpu() for k, v in st.items()}
    n, offs = len(prog.T), prog.offs

    def rows(name, j):
        return st[name][offs[j]:offs[j + 1]]
    if b < cut.first:
        stream = {}
        for j in range(n):
            stream[f"{j}.x"] = rows("h", j).reshape(1, prog.T[j], cfg.hc_mult, cfg.dim).to(torch.bfloat16)
            stream[f"{j}.pre"] = rows("p", j).reshape(1, prog.T[j], cfg.hc_mult)
        out[stream_name(c, b)] = stream
        for kind, src in emits(cut, a, b).items():
            out[side_name(c, kind, src)] = {str(j): _side_tensor(kind, st[f"{SIDE_PREFIX[kind]}{j}"])
                                            for j in range(n) if f"{SIDE_PREFIX[kind]}{j}" in st}
        return out, None
    chain = []  # the last piece: the job's records, as deepseek_prep writes a block
    for name in (i for i in unit.inputs if i.endswith(".json") and "/commit-" in i):  # earlier pieces, in order
        data = store.get(name)
        inputs[name] = hashlib.sha256(data).hexdigest()
        chain += json.loads(data)["roots"][1:]
    chain += roots[1:]
    records, leaves = [], []
    for j, i in enumerate(members):
        side = Side(*(_side_tensor(k, st[f"{SIDE_PREFIX[k]}{j}"]) if f"{SIDE_PREFIX[k]}{j}" in st else None
                      for k in ("compress_kv", "index_k", "picks")))
        cand = st.get(f"cand{j}")
        choices = Choices(candidates=None if cand is None else _side_tensor("candidates", cand))
        kept, cand = top.crossing_of(cut, side, choices, seqs[i][0].numel())
        x = rows("h", j).reshape(1, prog.T[j], cfg.hc_mult, cfg.dim).to(torch.bfloat16)
        ex = top.Example(seqs[i][0][None], seqs[i][1][None], x, rows("p", j).reshape(1, prog.T[j], cfg.hc_mult),
                         kept, cand)
        rec = top.pack(ex, 0, fp8_stream=fp8)
        records.append(rec)
        leaves.append(hashlib.sha256(hashlib.sha256(rec).digest()
                                     + b"".join(bytes.fromhex(h) for h in chain)).hexdigest())
    bin_name, meta_name = block_names(c)
    out[bin_name] = b"".join(records)
    out[meta_name] = json.dumps({"sequences": list(members),
                                 "offsets": np.cumsum([0] + [len(r) for r in records]).tolist(),
                                 "layer_hashes": [chain] * len(members)}).encode()
    return out, leaves


def run_piece(source, store, seqs, chunks, cut: top.Cut, unit: Unit, token_map=None, work: Path | None = None,
              device="cpu", whole_layer_gb: float = 14.0, fp8: bool = False, ranges: int = 15,
              align: int = ROWS_PER_PIECE, want: dict | None = None, tamper=None) -> dict:
    """A layers unit (piece [a, b) over chunk c, as `plan` made it), reading its inputs from `store` (a host, or
    Mirrors of several) and writing its outputs there, op by op (deepseek_program.PrepProgram). want: {object:
    digest} from its inputs' writers' commitments, which the node holds; an input that doesn't match is Tampered.
    -> its commitment, also stored: its outputs' digests, and the state root before it and after each of its
    segments ("roots"), which a dispute starts from. Its "stats" (seconds, tokens a second, peak GPU memory, bytes and
    seconds of transfer) aren't hashed by anything: they're the run's measurements. `tamper`: tests' cheat."""
    io = _io()
    with top.Meter(device) as meter:
        prog, st, roots, inputs = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device, ranges,
                                           align, want, io=io, tamper=tamper)
    objects, leaves = _outputs(store, seqs, chunks, cut, unit, prog, st, roots, inputs, fp8)
    outputs = {}
    for name, obj in objects.items():
        _put(store, name, obj, outputs, io)
    commit = {"format": FORMAT, "unit": ["layers", unit.a, unit.b, unit.chunk], "inputs": inputs, "outputs": outputs,
              "roots": roots}
    if leaves is not None:
        commit["leaves"] = leaves
    tokens = sum(len(seqs[i][0]) for i in chunks[unit.chunk]) * (unit.b - unit.a)
    commit["stats"] = {**meter.stats(tokens), **{k: round(v, 3) for k, v in io.items()}}
    store.put(commit_name(unit.chunk, unit.a), json.dumps(commit).encode())
    return commit


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
    """A check: the unit run again from its checked inputs. On the same hardware its state roots are the committed
    ones, to the bit. Across hardware, the reference's arithmetic differs a little, so its outputs are compared with
    the committed ones (read from `store`, checked against the commitment) within `tolerance` (distance). A unit
    whose first root isn't the state its inputs make committed to something else than its work. -> {"agrees",
    "worst": the largest distance, "where": which output, and which tensor of it}."""
    prog, st, roots, inputs = _compute(source, store, seqs, chunks, cut, unit, token_map, work, device, ranges,
                                       align, want)
    if roots[0] != (commit.get("roots") or [None])[0]:
        return {"agrees": False, "worst": float("inf"), "where": "the state it starts from"}
    if unit.b < cut.first:  # the state it hands on is its outputs, exactly: its root can't be anything else
        nxt = Unit("layers", unit.b, cut.first, unit.chunk, [], [])
        handed = _start(store, _program(seqs, chunks, cut, nxt), cut, nxt, {}, {**(want or {}), **commit["outputs"]},
                        None)
        if state_root(handed) != commit["roots"][-1]:
            return {"agrees": False, "worst": float("inf"), "where": "the state it hands on"}
    objects, leaves = _outputs(store, seqs, chunks, cut, unit, prog, st, roots, inputs, fp8)
    if roots == commit["roots"] and all(commit["outputs"].get(n) == _digest_of(o) for n, o in objects.items()):
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
            if hashlib.sha256(data).hexdigest() != commit["outputs"].get(name) or \
                    hashlib.sha256(meta).hexdigest() != commit["outputs"].get(meta_name):
                raise Tampered(f"{name} isn't what the unit committed to")
            ours_meta, their_meta = json.loads(objects[meta_name]), json.loads(meta)
            for j in range(len(ours_meta["sequences"])):
                mine = top.unpack(obj[ours_meta["offsets"][j]:ours_meta["offsets"][j + 1]])
                theirs = top.unpack(data[their_meta["offsets"][j]:their_meta["offsets"][j + 1]])
                for k in sorted(set(mine) | set(theirs)):
                    note(distance(mine.get(k), theirs.get(k)), f"{name}:{j}:{k}")
    return {"agrees": worst <= tolerance, "worst": worst, "where": where}


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
