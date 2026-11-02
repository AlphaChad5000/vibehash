"""Disputes over a fine-tune's claims (record version 4): a bisection game between a claim's worker and a checker who
rejected it, refereed by the node on one tiny calculation, with NumPy on its CPU.

The defender (the claim's worker) and the challenger (a checker that rejected it, or a replica that disagrees) narrow
their disagreement down program.py's tree, unit → segment → step → sequence → layer → op, and then to one tile of that
op's output (merkle.LEAF numbers):
  1. split: the defender posts the state roots at the boundaries of up to program.K children of the current range.
     The first must be the range's start and the last its end, or the defender loses: children that don't chain to
     their parent can't be what it computed. Each state it names must be on the node (its manifest and tensors, which
     the node checks hash as the manifest says), so the challenger can fetch them.
  2. pick: the challenger names the child where, recomputing from the defender's own state before it, it doesn't get
     the defender's state after it. A child that is one op comes with the tile of its output to check.
  3. evidence: for that op, the defender shows both states' manifests and, with Merkle proofs, the numbers of its
     inputs the tile needs and the tile itself. The node checks the op changed exactly its output (and dropped what
     it should), then referee.judge recomputes the tile. Inside the bounds, the defender wins; outside, it loses.
A party silent past its move's deadline loses. A challenger may concede instead of picking: a checker that does
loses (it accused), but a replica challenging the other replica that concedes (or never moves) before its first pick
goes without blame: it never chose to accuse. Every move is a signed message and goes in the record with the evidence and
the weights the referee used, so a validator replays the game and re-runs the referee and gets the same verdict.

The base model's weights the referee needs come from the pinned commit on the Hub by byte range (HubWeights), or from
the node's tensor cache: never from either party. The job's manifest pins each tensor's sha256, but only whole: a slice
of a tensor has nothing to be proven against, so a party offering one would have to send the whole tensor (up to
hundreds of MB), and it would be checked against the same manifest the node can check the Hub's copy with. If the Hub
can't be reached, the claim ends unchecked and nobody is slashed.
"""
from __future__ import annotations

import hashlib
import math
import types
from dataclasses import dataclass, field

import numpy as np

from . import merkle, referee
from .merkle import LEAF, MerkleError

MAX_EVIDENCE = 32 * 2**20  # a tile's inputs: a few MB at most (an attention column over every query position)
MAX_WEIGHTS = 256 * 2**20  # base-model weights one tile can need (a whole projection, for its backward)
MAX_RECORDED = 64 * 2**20  # weights a record keeps for one game (the column slices used, not whole tensors)
MAX_MOVES = 40


class WeightsUnavailable(Exception):
    """The base model's weights couldn't be fetched: the claim ends unchecked."""


class RecordIncomplete(Exception):
    """A replayed game's record lacks something its referee used."""


# --- the base model's weights -----------------------------------------------------------------------------------------

def decode(dtype: str, raw: bytes, cols: int) -> np.ndarray:
    """Stored weights as float64, exactly: bf16 and f16 widen without rounding, and DeepSeek's FP8 (E4M3), ue8m0
    scales and FP4 (E2M1 codes packed two to an I8 byte, low nibble first: `cols` packed bytes give 2·cols values)
    decode from their bits (referee_deepseek)."""
    if dtype in ("F8_E4M3", "F8_E8M0", "I8"):
        from . import referee_deepseek as rd

        b = np.frombuffer(raw, np.uint8).reshape(-1, cols)
        if dtype == "F8_E4M3":
            return rd.E4M3[b]
        if dtype == "F8_E8M0":
            return rd.decode_e8m0(b)
        return np.stack([rd.FP4[b & 15], rd.FP4[b >> 4]], -1).reshape(b.shape[0], 2 * cols)
    if dtype in ("I32", "I64"):  # integer tables (DeepSeek's engram token map), exact up to 2^53
        return np.frombuffer(raw, "<i4" if dtype == "I32" else "<i8").astype(np.float64).reshape(-1, cols)
    if dtype == "BF16":
        a = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        a = np.frombuffer(raw, "<f2")
    elif dtype == "F32":
        a = np.frombuffer(raw, "<f4")
    else:
        raise WeightsUnavailable(f"weights stored as {dtype} aren't supported by the referee yet")
    return a.astype(np.float64).reshape(-1, cols)


def _cols(shape) -> int:
    return int(math.prod(shape[1:])) if len(shape) > 1 else int(shape[0])


def _rows(shape) -> int:
    return int(shape[0]) if len(shape) > 1 else 1


_ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "I32": 4, "I64": 8}


class HubWeights:
    """Rows of the job's pinned base model, as the node's referee reads them: from its tensor cache when it has the
    whole tensor (hash-checked when it was fetched), else by byte range from the pinned commit on the Hub. Ranges
    fetched are kept in memory, up to `keep` bytes."""

    def __init__(self, manifest, hub=None, cache=None, keep: int = 256 * 2**20):
        from . import hub as hubmod

        self.manifest, self.hub, self.cache = manifest, hub, cache or hubmod.TensorCache()
        self.keep, self.kept = keep, {}

    def shape(self, name):
        e = self.manifest.tensors.get(name)
        if e is None:
            raise WeightsUnavailable(f"the base model has no {name}")
        return e.dtype, tuple(e.shape)

    def read(self, name, r0, r1) -> bytes:
        from . import hub as hubmod

        dtype, shape = self.shape(name)
        width = _cols(shape) * _ITEM.get(dtype, 0)
        key = (name, r0, r1)
        if key in self.kept:
            return self.kept[key]
        e = self.manifest.tensors[name]
        path = self.cache.path(self.manifest.commit, e.sha256)
        try:
            if path.exists() and path.stat().st_size == e.length:
                with open(path, "rb") as f:
                    f.seek(r0 * width)
                    raw = f.read((r1 - r0) * width)
            else:
                size = self.manifest.files[e.file]["size"]
                f = (self.hub or hubmod.Hub()).open(self.manifest.repo, self.manifest.commit, e.file, size)
                raw = f.read(e.offset + r0 * width, (r1 - r0) * width)
        except (hubmod.HubError, OSError) as ex:
            raise WeightsUnavailable(f"couldn't fetch {name} from the Hub: {ex}") from None
        if len(raw) != (r1 - r0) * width:
            raise WeightsUnavailable(f"{name}: the Hub sent {len(raw)} bytes, not {(r1 - r0) * width}")
        while self.kept and sum(map(len, self.kept.values())) + len(raw) > self.keep:
            self.kept.pop(next(iter(self.kept)))
        self.kept[key] = raw
        return raw


class Recorded:
    """The weights a game's referee used, as the record keeps them (pack/unpack)."""

    def __init__(self, blob: bytes | None = None):
        self.items: dict[str, tuple[str, tuple, bytes]] = {}
        if blob is not None:
            header, chunks = merkle.unpack(blob, MAX_RECORDED + 2**20)
            for key, (dtype, shape) in header.items():
                if key not in chunks or dtype not in _ITEM or not isinstance(shape, list):
                    raise RecordIncomplete("the record's weights are malformed")
                self.items[key] = (dtype, tuple(int(x) for x in shape), chunks[key])

    @staticmethod
    def key(name, r0, r1) -> str:
        return f"{name}[{r0}:{r1}]"

    def shape(self, name):
        for k, (dtype, shape, _) in self.items.items():
            if k.rsplit("[", 1)[0] == name:
                return dtype, shape
        raise RecordIncomplete(f"the record has none of {name}")

    def read(self, name, r0, r1) -> bytes:
        try:
            return self.items[self.key(name, r0, r1)][2]
        except KeyError:
            raise RecordIncomplete(f"the record lacks rows {r0}-{r1} of {name}") from None

    def pack(self) -> bytes:
        return merkle.pack({k: [d, list(s)] for k, (d, s, _) in self.items.items()}, {k: raw for k, (_, _, raw) in self.items.items()})


class Outside:
    """What the referee reads besides the evidence: base-model weights (`source`: something with .shape(name) and
    .read(name, r0, r1)), the job's data, its LoRA A matrices and the rotary tables. With `keep`, every weight read is
    kept (Recorded) for the record."""

    def __init__(self, spec, data, source, keep: bool = True, limit: int = MAX_WEIGHTS):
        self.spec, self.data, self.source, self.limit, self.used = spec, data, source, limit, 0
        self.kept = Recorded() if keep else None
        self._a: dict[int, np.ndarray] = {}
        self._rope: dict = {}
        self._got: dict[str, np.ndarray] = {}

    def weight_rows(self, name) -> int:
        return _rows(self.source.shape(name)[1])

    def weight(self, name, r0, r1) -> np.ndarray:
        key = Recorded.key(name, r0, r1)
        if key in self._got:
            return self._got[key]
        dtype, shape = self.source.shape(name)
        if not 0 <= r0 < r1 <= _rows(shape):
            raise referee.Missing(f"rows {r0}-{r1} of {name}")
        raw = self.source.read(name, r0, r1)
        self.used += len(raw)
        if self.used > self.limit:
            raise WeightsUnavailable("the tile needs more of the base model than the referee fetches for one")
        if self.kept is not None:
            self.kept.items[key] = (dtype, shape, raw)
        self._got[key] = decode(dtype, raw, _cols(shape))
        return self._got[key]

    def lora_a(self, k) -> np.ndarray:
        if k not in self._a:
            self._a[k] = lora_a_one(self.spec, k).astype(np.float64)
        return self._a[k]

    def tokens(self, seq):
        n = int(self.data.train_len[seq])
        return self.data.train_ids[seq, :n].astype(np.int64), self.data.train_mask[seq, :n]

    def rope(self, op, rows, r):
        key = (op.attrs["hd"], op.attrs["theta"], tuple(op.attrs["scaling"]), rows)
        if key not in self._rope:
            arch = types.SimpleNamespace(head_dim=op.attrs["hd"], rope_theta=op.attrs["theta"],
                                         rope_scaling=tuple(op.attrs["scaling"]))
            self._rope = {key: referee.rope_tables(arch, rows)}
        cos, sin = self._rope[key]
        return cos[r], sin[r]


def lora_a_one(spec, k: int) -> np.ndarray:
    """Module k's A, exactly as LoraSpec.lora_a draws it, without drawing the others: the stream skipped ahead."""
    import numpy.random as npr

    mods = spec.modules()
    rng = npr.Generator(npr.PCG64([spec.seed, 0x10A]))
    rng.bit_generator.advance(sum(spec.rank * inp for _, _, inp in mods[:k]))
    inp = mods[k][2]
    u = (rng.bit_generator.random_raw(spec.rank * inp) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
    return ((2 * u - 1) * (1 / math.sqrt(inp))).astype(np.float32).reshape(spec.rank, inp)


class Inputs:
    """The referee's view of a state: numbers of its tensors, either proven leaves (`leaves`) or, for a party building
    its evidence, whole arrays (`full`), in which case every leaf the referee reads is noted in `used`."""

    def __init__(self, shapes: dict, outside: Outside, leaves: dict | None = None, full: dict | None = None):
        self.shapes, self.outside, self.leaves, self.full = shapes, outside, leaves or {}, full
        self.used: dict[str, set[int]] = {}
        self.after: "Inputs | None" = None  # the op's output, for the ops judged by whole rows (referee.CHOICES)

    def out_row(self, name, r):
        """Row r of the op's output, as the state after it holds it (proven leaves, or the party's array)."""
        if self.after is None:
            raise referee.Missing(f"{name}'s row {r}")
        return self.after.row(name, r)

    def shape(self, name):
        if name not in self.shapes:
            raise referee.Missing(name)
        return self.shapes[name]

    def get(self, name, r0, r1, c0=0, c1=None) -> np.ndarray:
        rows, cols = self.shape(name)
        c1 = cols if c1 is None else c1
        if not (0 <= r0 <= r1 <= rows and 0 <= c0 <= c1 <= cols):
            raise referee.Missing(f"{name}[{r0}:{r1}, {c0}:{c1}]")
        if self.full is not None:
            if c1 > c0:
                used = self.used.setdefault(name, set())
                for r in range(r0, r1):
                    used.update(range((r * cols + c0) // LEAF, (r * cols + c1 - 1) // LEAF + 1))
            return np.asarray(self.full[name][r0:r1, c0:c1], dtype=np.float64)
        out = np.empty((r1 - r0, c1 - c0))
        have = self.leaves.get(name, {})
        for r in range(r0, r1):
            s, e = r * cols + c0, r * cols + c1
            if e <= s:
                continue
            first, last = s // LEAF, (e - 1) // LEAF
            try:
                flat = np.concatenate([have[i] for i in range(first, last + 1)])
            except KeyError:
                raise referee.Missing(f"{name}[{r}, {c0}:{c1}]") from None
            out[r - r0] = flat[s - first * LEAF:e - first * LEAF]
        return out

    def row(self, name, r):
        return self.get(name, r, r + 1)[0]

    def weight(self, name, r0, r1):
        return self.outside.weight(name, r0, r1)

    def weight_rows(self, name):
        return self.outside.weight_rows(name)

    def lora_a(self, k):
        return self.outside.lora_a(k)

    def tokens(self, seq):
        return self.outside.tokens(seq)

    def rope(self, op, r):
        return self.outside.rope(op, self.shape(op.ins[0])[0], r)


# --- states and evidence ----------------------------------------------------------------------------------------------

def state_manifest(arrays: dict[str, np.ndarray], known: dict | None = None) -> tuple[bytes, dict[str, bytes]]:
    """A state's manifest and the bytes of each of its tensors, by blob id. `known` (array id -> (root, blob)) saves
    rehashing tensors already hashed."""
    entries, blobs = {}, {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a, dtype="<f4")
        hit = known.get(id(a)) if known is not None else None
        root, blob = hit if hit else (merkle.root_of(a), merkle.blob_of(a))
        entries[name] = (root, blob, a.shape[0], a.shape[1])
        blobs[blob] = a.tobytes()
    return merkle.manifest_bytes(entries), blobs


def checkpoint_state(spec, theta: np.ndarray) -> tuple[str, bytes, dict[str, bytes]]:
    """A checkpoint as a state: (root, manifest, tensor bytes by blob id)."""
    from .program import split_theta

    manifest, blobs = state_manifest(split_theta(spec, theta))
    return merkle.state_root(manifest), manifest, blobs


def diff_problem(op, before: dict, after: dict) -> str | None:
    """Why the state after `op` isn't the state before it with only its output written and its drops gone (and its
    inputs the shapes it reads), or None."""
    for n, s in zip(op.ins, op.ishapes):
        if n not in before or tuple(before[n][2:]) != tuple(s):
            return f"the state before {op.describe()} has no {n} of shape {tuple(s)}"
    want = dict(before)
    for d in op.drops:
        if d not in want:
            return f"{op.describe()} drops {d}, which the state before it doesn't have"
        del want[d]
    if op.out:
        if op.out not in after or tuple(after[op.out][2:]) != tuple(op.shape):
            return f"the state after {op.describe()} has no {op.out} of shape {tuple(op.shape)}"
        want[op.out] = after[op.out]
    if want != after:
        changed = sorted(set(want) ^ set(after) | {n for n in set(want) & set(after) if want[n] != after[n]})
        return f"{op.describe()} changed more of the state than its output: {', '.join(changed[:3])}"
    return None


def _proof(arr: np.ndarray, idx) -> tuple[dict, bytes]:
    tree = merkle.levels(merkle.leaf_hashes(arr))
    idx = sorted(set(idx))
    flat = np.ascontiguousarray(arr, dtype="<f4").reshape(-1)
    data = b"".join(flat[i * LEAF:(i + 1) * LEAF].tobytes() for i in idx)
    return {"top": tree[-1][0].hex(), "leaves": idx, "proof": [h.hex() for h in merkle.multiproof(tree, idx)]}, data


def make_evidence(op, tile, before: dict, after: dict, manifests: tuple[bytes, bytes], outside: Outside,
                  shapes: dict | None = None) -> tuple[bytes, bool, str]:
    """A defender's evidence for one tile of `op` (states as {name: float32 array}; `shapes` of `before`'s tensors,
    if reading them all to find out would be a waste): (the evidence, whether the referee will pass it, why). The
    leaves it proves are the ones the referee reads, found by running it."""
    header, chunks = {"tensors": {}}, {"manifest:in": manifests[0], "manifest:out": manifests[1]}
    ok, why = True, f"{op.describe()} only drops tensors"
    if op.out:
        x = Inputs(shapes or {n: a.shape for n, a in before.items()}, outside, full=before)
        x.after = Inputs({op.out: tuple(after[op.out].shape)}, outside, full={op.out: after[op.out]})
        said = np.ascontiguousarray(after[op.out], dtype="<f4").reshape(-1)[tile * LEAF:(tile + 1) * LEAF]
        ok, why = referee.judge(op, x, tile, said)
        for name, idx in sorted(x.used.items()):
            header["tensors"][f"in:{name}"], chunks[f"in:{name}"] = _proof(before[name], idx)
        out_idx = {tile} | x.after.used.get(op.out, set())  # (whole rows, for a choice)
        header["tensors"][f"out:{op.out}"], chunks[f"out:{op.out}"] = _proof(after[op.out], out_idx)
    return merkle.pack(header, chunks), ok, why


def judge_evidence(op, tile, a: str, b: str, evidence: bytes, outside: Outside) -> tuple[str, str]:
    """The referee's verdict on a game's last move: ("defender" | "challenger" | "unchecked", why)."""
    try:
        header, chunks = merkle.unpack(evidence, MAX_EVIDENCE)
        before_m, after_m = chunks.get("manifest:in", b""), chunks.get("manifest:out", b"")
        if merkle.state_root(before_m) != a or merkle.state_root(after_m) != b:
            return "challenger", "the evidence is for other states than the ones the defender posted around the op"
        before, after = merkle.parse_manifest(before_m), merkle.parse_manifest(after_m)
        problem = diff_problem(op, before, after)
        if problem:
            return "challenger", problem
        if not op.out:
            return "defender", f"{op.describe()} only drops tensors, as it should"
        if not isinstance(header, dict) or not isinstance(header.get("tensors"), dict):
            raise MerkleError("no tensors")
        leaves: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        for key, t in header["tensors"].items():
            side, name = key.split(":", 1) if ":" in key else ("", key)
            entry = (before if side == "in" else after if side == "out" else {}).get(name)
            if entry is None or key not in chunks or not isinstance(t, dict):
                raise MerkleError(f"proof for {key} names no tensor of the state")
            rows, cols = entry[2], entry[3]
            top = bytes.fromhex(t["top"])
            idx = t["leaves"]
            count = merkle.n_leaves(rows, cols)
            if merkle.tensor_root(rows, cols, top) != entry[0] or idx != sorted(set(idx)) \
                    or not all(type(i) is int and 0 <= i < count for i in idx):
                raise MerkleError(f"proof for {key} doesn't match the state")
            sizes = [4 * (min(rows * cols, (i + 1) * LEAF) - i * LEAF) for i in idx]
            raw = chunks[key]
            if sum(sizes) != len(raw):
                raise MerkleError(f"proof for {key} has the wrong number of bytes")
            got, at = {}, 0
            for i, n in zip(idx, sizes):
                got[i] = raw[at:at + n]
                at += n
            proof = [bytes.fromhex(h) for h in t["proof"]]
            if not merkle.verify(top, count, {i: merkle._leaf(v) for i, v in got.items()}, proof):
                raise MerkleError(f"proof for {key} doesn't check out")
            leaves[(side, name)] = {i: np.frombuffer(v, "<f4").astype(np.float64) for i, v in got.items()}
        said = leaves.get(("out", op.out), {}).get(tile)
        if said is None:
            return "challenger", f"the evidence doesn't show tile {tile} of {op.out}"
        shapes = {n: (e[2], e[3]) for n, e in before.items()}
        x = Inputs(shapes, outside, leaves={n: v for (s, n), v in leaves.items() if s == "in"})
        x.after = Inputs({op.out: (after[op.out][2], after[op.out][3])}, outside,
                         leaves={op.out: leaves.get(("out", op.out), {})})
        ok, why = referee.judge(op, x, tile, said)
        return ("defender" if ok else "challenger"), why
    except referee.Missing as e:
        return "challenger", f"the evidence doesn't prove {e}"
    except (MerkleError, ValueError, TypeError, KeyError) as e:
        return "challenger", f"the evidence is malformed: {e}"
    except WeightsUnavailable as e:
        return "unchecked", str(e)


# --- the game ---------------------------------------------------------------------------------------------------------

class MoveRefused(ValueError):
    """A move that isn't taken (wrong turn, malformed, states not on the node): the sender may still make a good one."""


class Unavailable:
    """A weight source that never has anything: a replayed game whose referee couldn't reach the Hub."""

    def shape(self, name):
        raise WeightsUnavailable("the Hub couldn't be reached when this game was refereed")

    read = shape


@dataclass
class Game:
    gid: str
    unit: str
    defender: str
    challenger: str
    why: str  # "dispute" (a checker rejected the claim) or "replicas" (the other replica of its unit disagrees)
    root: list  # ["segment", s] (a rejected segment) or ["unit"] (a replica's whole unit)
    a: str  # the current range's state roots, before and after
    b: str
    origin: tuple = ()  # the root's (a, b)
    lo: int = 0
    hi: int = 0
    path: list = field(default_factory=list)  # atoms descended into, from the root
    turn: str = "split"  # whose move: "split" and "evidence" the defender's, "pick" the challenger's; "done"
    roots: list | None = None  # the split being picked from
    tile: int | None = None
    moves: list = field(default_factory=list)  # the signed messages, as the record keeps them
    played: bool = False  # the challenger has picked at least once
    outcome: str | None = None  # "defender" (it was right), "challenger", "unchecked" or "conceded"
    detail: str = ""
    deadline: float = 0.0
    evidence: bytes | None = None
    weights: bytes | None = None
    leaf: object = None  # the op picked, once it is one
    seed: int = 0  # the unit's, which the parties' program.Program needs

    def node(self, prog):
        n = prog.segment(int(self.root[1])) if self.root[0] == "segment" else prog.unit()
        for i in self.path:
            n = n.atoms[i]
        return n

    def start(self, prog) -> "Game":
        self.origin, self.lo, self.hi = (self.a, self.b), 0, len(self.node(prog).atoms)
        return self

    @property
    def loser(self) -> str | None:
        return {"challenger": self.defender, "defender": self.challenger}.get(self.outcome)

    def end(self, outcome: str, detail: str) -> None:
        self.turn, self.outcome, self.detail, self.roots = "done", outcome, detail, None

    def view(self) -> dict:
        """What the parties (and anyone) see of it: GET /round/games."""
        from .program import groups

        return {"id": self.gid, "unit": self.unit, "defender": self.defender, "challenger": self.challenger,
                "why": self.why, "root": list(self.root), "path": list(self.path), "lo": self.lo, "hi": self.hi,
                "a": self.a, "b": self.b, "turn": self.turn, "roots": self.roots, "tile": self.tile,
                "groups": [list(g) for g in groups(self.lo, self.hi)] if self.turn in ("split", "pick") else None,
                "n": len(self.moves), "deadline": self.deadline, "outcome": self.outcome, "detail": self.detail,
                "seed": self.seed, "played": self.played}

    def entry(self) -> dict:
        """The game as the round's record keeps it."""
        return {"id": self.gid, "unit": self.unit, "defender": self.defender, "challenger": self.challenger,
                "why": self.why, "root": list(self.root), "a": self.origin[0], "b": self.origin[1],
                "moves": list(self.moves), "evidence": merkle.b64(self.evidence) if self.evidence is not None else None,
                "weights": merkle.b64(self.weights) if self.weights is not None else None,
                "outcome": self.outcome, "detail": self.detail}


def _hex64(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def play(game: Game, prog, who: str, move: dict, *, check_states=None, evidence: bytes | None = None,
         outside: Outside | None = None) -> None:
    """Apply one move `who` made. Raises MoveRefused for one that isn't taken. `check_states(roots)` (a live node)
    raises MoveRefused unless every state a split names is on the node; `evidence` is the evidence move's bytes
    (whose sha256 the move names) and `outside` what the referee reads besides it."""
    from .program import Op, groups

    if game.turn == "done":
        raise MoveRefused("this game is over")
    if len(game.moves) >= MAX_MOVES or not isinstance(move, dict):
        raise MoveRefused("not a move")
    kind = next((k for k in ("split", "pick", "evidence", "concede") if k in move), None)
    if kind is None or set(move) - {kind, "tile"} or ("tile" in move and kind != "pick"):
        raise MoveRefused("a move is a split, a pick (with a tile, for an op), evidence, or conceding")
    party = game.defender if kind in ("split", "evidence") else game.challenger
    if who != party or game.turn != ("pick" if kind == "concede" else kind):
        whose = "defender" if game.turn in ("split", "evidence") else "challenger"
        raise MoveRefused(f"it's the {whose}'s turn to {game.turn}")
    if kind == "concede":
        if move["concede"] is not True:
            raise MoveRefused("conceding is {\"concede\": true}")
        blameless = game.why == "replicas" and not game.played
        game.end("conceded" if blameless else "defender", "the challenger conceded")
    elif kind == "split":
        roots = move["split"]
        n = len(groups(game.lo, game.hi))
        if not isinstance(roots, list) or len(roots) != n + 1 or not all(_hex64(r) for r in roots):
            raise MoveRefused(f"a split here is {n + 1} state roots, the ends of {n} children")
        if roots[0] != game.a or roots[-1] != game.b:
            game.end("challenger", "the defender's children don't chain to the states it is defending")
            return
        if check_states is not None:
            check_states(roots[1:-1])
        game.roots, game.turn = list(roots), "pick"
    elif kind == "pick":
        child = move["pick"]
        g = groups(game.lo, game.hi)
        if type(child) is not int or not 0 <= child < len(g):
            raise MoveRefused(f"pick one of the {len(g)} children")
        (g0, g1), a, b = g[child], game.roots[child], game.roots[child + 1]
        atom = game.node(prog).atoms[g0] if g1 - g0 == 1 else None
        tile = move.get("tile")
        if isinstance(atom, Op):
            count = merkle.n_leaves(*atom.shape) if atom.out else 0
            if atom.out and (type(tile) is not int or not 0 <= tile < count):
                raise MoveRefused(f"name a tile of the op's output, 0-{count - 1}")
            if not atom.out and tile is not None:
                raise MoveRefused("this op has no output to name a tile of")
            game.leaf, game.tile, game.turn, game.lo, game.hi = atom, tile, "evidence", g0, g1
        else:
            if tile is not None:
                raise MoveRefused("that child isn't one op")
            if atom is not None:
                game.path.append(g0)
                game.lo, game.hi = 0, len(atom.atoms)
            else:
                game.lo, game.hi = g0, g1
            game.turn = "split"
        game.a, game.b, game.roots, game.played = a, b, None, True
    else:
        blob = move["evidence"]
        if not _hex64(blob) or evidence is None or hashlib.sha256(evidence).hexdigest() != blob:
            raise MoveRefused("evidence is the id of an uploaded blob")
        outcome, detail = judge_evidence(game.leaf, game.tile, game.a, game.b, evidence, outside)
        game.evidence = evidence
        if outcome != "unchecked" and outside is not None and outside.kept is not None:
            game.weights = outside.kept.pack()
        game.end(outcome, detail)


def silence(game: Game) -> None:
    """The party whose move it is didn't make it: it loses. A replica challenging that never picked concedes."""
    if game.turn == "done":
        return
    if game.turn == "pick":
        if game.why == "replicas" and not game.played:
            game.end("conceded", "the other replica didn't challenge it")
        else:
            game.end("defender", "the challenger didn't pick in time")
    else:
        game.end("challenger", f"the defender didn't {game.turn} in time")
