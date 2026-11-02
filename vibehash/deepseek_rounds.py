"""A DeepSeek job's prep on the node, round by round (DEEPSEEK.md §8, "The split prep"), without PyTorch: the
coordinator asks this which of the job's units a round hands out, checks a reveal against what the unit must have
made, settles the round into the work graph, and, once every unit is accepted, pins the records' Merkle root, after
which the job's rounds are a fine-tune's (training on the records).

A prep round is a fine-tune round (record version 4) whose units are the work graph's ready units, named in the
record's "prep": {"units": {unit id: graph unit}}. A worker commits to the sha256 of its unit's commitment (the
JSON deepseek_split writes to the bucket); its reveal is that root and salt; two staked checkers replay the unit
(deepseek_worker.check_unit) and say whether it holds. Settling adds "accepted": {graph unit: what later units and
the records need of its commitment}, and "root" when it's the last.

Disputes. A training round's are refereed as a fine-tune's (deepseek_program.py, referee_deepseek.py). A prep
unit has no program in the referee yet, so for now a prep claim any counted checker rejects ends unchecked (unpaid,
not slashed) and its unit goes out again in a later round: a cheat gains nothing, and is never caught either.
DEEPSEEK.md §8 lists it as a known weakness, until prep units are refereed (before launch).
"""
from __future__ import annotations

import hashlib
import json

from . import deepseek_graph as g
from .crypto import merkle_root

MAX_PREP_UNITS = 64  # units a prep round hands out at most
PREP_FORMAT = "vibehash-deepseek-prep/1"  # deepseek_prep's manifest format: the records are the same


class PrepState:
    """Where a DeepSeek job's prep is: its work graph (from the spec and the job's token file, the same on every
    node and worker), each accepted unit's outputs, and the root once the records are all in."""

    def __init__(self, job):
        self.job, self.spec, self.tokens = job, job.spec, job.data.tokens
        self.graph = self.spec.graph(self.tokens)
        self.line = g.Line(self.graph)
        self.commits: dict[str, dict] = {}  # graph unit -> {"outputs": {object: digest}, ["leaves": [...]]}
        self.root: str | None = None

    def to_json(self) -> dict:
        return {"commits": self.commits, "root": self.root}

    @classmethod
    def restore(cls, job, saved: dict | None) -> "PrepState":
        state = cls(job)
        for name, kept in sorted(((saved or {}).get("commits") or {}).items()):
            state._accept(name, kept)
        state.root = (saved or {}).get("root")
        return state

    def apply(self, record: dict) -> None:
        """A settled prep round's outcome, from its record (ours, or another validator's)."""
        prep = record.get("prep") or {}
        for name, kept in sorted((prep.get("accepted") or {}).items()):
            self._accept(name, kept)
        if prep.get("root"):
            self.root = prep["root"]

    def _accept(self, name: str, kept: dict) -> None:
        if name in self.graph.units:
            self.line.state[name] = "accepted"
            self.commits[name] = kept

    @property
    def done(self) -> bool:
        return all(st == "accepted" for st in self.line.state.values())

    def ready(self) -> list[g.Unit]:
        """Units whose inputs are all accepted, in the graph's order (rows units first, then pieces bottom first)."""
        out = []
        for u in self.graph.order():
            if self.line.state[u.name] == "pending" and all(self.line.state[d] == "accepted"
                                                           for d in self.graph.deps(u.name)):
                out.append(u)
        return out

    def want(self) -> dict[str, str]:
        """Every accepted object's digest: what a unit checks its inputs against."""
        return {o: d for kept in self.commits.values() for o, d in kept["outputs"].items()}

    def flops(self, unit: g.Unit) -> int:
        """Nominal FLOPs a unit is worth, for pay and capacity: a layers unit's layer-tokens at the trained layers'
        rate per token a layer (forward only: a third of training's), a rows unit's table range at a token's share."""
        tokens = sum(int(self.tokens.lengths[i]) for i in self.tokens.chunks(self.spec.chunk_tokens)[unit.chunk]) \
            if unit.kind == "layers" else int(self.tokens.lengths.sum())
        layers = unit.b - unit.a if unit.kind == "layers" else 1
        return 2 * self.spec.active_weights() * tokens * layers

    def check_commit(self, unit: g.Unit, raw: bytes, root: bytes) -> str | None:
        """Why a revealed commitment isn't this unit's (None if it is): its bytes must hash to the committed root, name
        the unit, and give a digest for exactly the objects the unit makes (and the records' leaves, for the last
        piece)."""
        if hashlib.sha256(raw).digest() != root:
            return "the commitment in the bucket isn't the one committed to"
        try:
            c = json.loads(raw)
        except ValueError:
            return "the commitment isn't JSON"
        want = ["rows", unit.a, unit.b] if unit.kind == "rows" else ["layers", unit.a, unit.b, unit.chunk]
        if c.get("unit") != want:
            return f"the commitment is for {c.get('unit')}, not {want}"
        made = set(unit.outputs) - {g.unit_commit(unit)}
        if set(c.get("outputs") or {}) != made:
            return "the commitment doesn't name exactly the objects the unit makes"
        if unit.kind == "layers" and unit.b == self.spec.cut:
            members = self.tokens.chunks(self.spec.chunk_tokens)[unit.chunk]
            if not isinstance(c.get("leaves"), list) or len(c["leaves"]) != len(members):
                return "the last piece's commitment must give a leaf a record"
        return None

    def kept(self, unit: g.Unit, commit: dict) -> dict:
        """What the record keeps of an accepted commitment: its outputs' digests, and the last piece's leaves."""
        out = {"outputs": dict(commit["outputs"])}
        if unit.kind == "layers" and unit.b == self.spec.cut:
            out["leaves"] = list(commit["leaves"])
        return out

    def manifest(self) -> dict:
        """The records' manifest, deepseek_split.manifest's (and so deepseek_prep's root), from the last pieces'
        accepted commitments."""
        cut, chunks = self.spec.cut, self.tokens.chunks(self.spec.chunk_tokens)
        last = {self.graph.units[n].chunk: kept for n, kept in self.commits.items()
                if self.graph.units[n].kind == "layers" and self.graph.units[n].b == cut}
        leaves = [bytes.fromhex(leaf) for c in range(len(chunks)) for leaf in last[c]["leaves"]]
        from .deepseek_config import COMMIT, REPO
        return {"format": PREP_FORMAT, "repo": REPO, "commit": COMMIT, "cut": cut, "fp8_stream": False,
                "sequences": len(self.tokens.lengths), "tokens": int(self.tokens.lengths.sum()),
                "blocks": [{"file": g.block_names(c)[0], "first": ch[0], "count": len(ch), "leaves": last[c]["leaves"]}
                           for c, ch in enumerate(chunks)],
                "root": merkle_root(leaves).hex()}

    def objects(self, name: str) -> tuple[set[str], set[str]]:
        """(what a unit reads, what it writes, its commitment included): what the node signs links for."""
        u = self.graph.units[name]
        return set(u.inputs), set(u.outputs) | {g.unit_commit(u)}
