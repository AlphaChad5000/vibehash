"""Hashes of a fine-tune's training state that a referee can check a few numbers of without the rest (dispute.py).

A state is a set of named float32 tensors, each 2-D [rows, cols]: the adapters and Adam's moments while training, plus
a step's gradients and one sequence's activations in the middle of one. Its root is the sha256 of its manifest, which
names each tensor's Merkle root, its blob (the sha256 of its bytes, as blobs.py stores it) and its shape. A tensor's
Merkle tree is over LEAF-element chunks of it flattened, so any few of its numbers come with a proof a few hundred bytes
long, and a node can check them against a state's root without the tensor.

The manifest's own bytes are canonical JSON, so equal states have equal roots on every machine, and a node stores a
manifest as a blob whose id is the root.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct

import numpy as np

LEAF = 256  # float32s a leaf: 1 KB
FORMAT = "vibehash-state/1"
MAX_MANIFEST = 4 * 2**20  # bytes; a state of a few thousand tensors is a few hundred KB
MAX_TENSORS = 20_000


class MerkleError(ValueError):
    """A proof, manifest or container that doesn't check out."""


def _leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + a + b).digest()


def leaf_hashes(arr: np.ndarray) -> list[bytes]:
    """The leaves of a tensor: its float32 bytes, little-endian, row after row, LEAF numbers at a time."""
    raw = memoryview(np.ascontiguousarray(arr, dtype="<f4")).cast("B")
    step = LEAF * 4
    return [_leaf(raw[i:i + step]) for i in range(0, len(raw), step)]


def levels(leaves: list[bytes]) -> list[list[bytes]]:
    """Every level of the tree, leaves first. Pairs are hashed together and an odd one out goes up unchanged."""
    out = [list(leaves) or [hashlib.sha256(b"\x02").digest()]]
    while len(out[-1]) > 1:
        lv = out[-1]
        out.append([_node(lv[i], lv[i + 1]) if i + 1 < len(lv) else lv[i] for i in range(0, len(lv), 2)])
    return out


def tensor_root(rows: int, cols: int, top: bytes) -> str:
    """What a manifest names a tensor by: its tree's top and its shape, together."""
    return hashlib.sha256(b"T" + struct.pack(">QQ", rows, cols) + top).hexdigest()


def root_of(arr: np.ndarray) -> str:
    rows, cols = arr.shape
    return tensor_root(rows, cols, levels(leaf_hashes(arr))[-1][0])


def blob_of(arr: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(arr, dtype="<f4")).cast("B")).hexdigest()


def n_leaves(rows: int, cols: int) -> int:
    return -(-rows * cols // LEAF)


def multiproof(tree: list[list[bytes]], idxs) -> list[bytes]:
    """The hashes that, with the leaves `idxs`, rebuild the root: level by level, in the order verify() asks."""
    proof, known = [], sorted(set(idxs))
    for lv in tree[:-1]:
        have, up = set(known), []
        for i in known:
            if up and up[-1] == i // 2:
                continue
            sib = i ^ 1
            if sib < len(lv) and sib not in have:
                proof.append(lv[sib])
            up.append(i // 2)
        known = up
    return proof


def verify(top: bytes, count: int, leaves: dict[int, bytes], proof: list[bytes]) -> bool:
    """Whether leaf hashes {index: hash} and a multiproof rebuild `top` for a tree of `count` leaves."""
    if not leaves or any(not 0 <= i < count for i in leaves):
        return False
    level, width, it = dict(leaves), count, iter(proof)
    try:
        while width > 1:
            up = {}
            for i in sorted(level):
                if i // 2 in up:
                    continue
                sib = i ^ 1
                if sib >= width:
                    up[i // 2] = level[i]
                    continue
                other = level[sib] if sib in level else next(it)
                up[i // 2] = _node(level[i], other) if i % 2 == 0 else _node(other, level[i])
            level, width = up, (width + 1) // 2
    except StopIteration:
        return False
    return next(it, None) is None and level.get(0) == top


# --- manifests -------------------------------------------------------------------------------------------------------

def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def manifest_bytes(entries: dict[str, tuple[str, str, int, int]]) -> bytes:
    """A state's manifest: {name: (Merkle root, blob id, rows, cols)}."""
    return canonical({"format": FORMAT, "tensors": {n: [r, b, int(R), int(C)] for n, (r, b, R, C) in entries.items()}})


def state_root(manifest: bytes) -> str:
    return hashlib.sha256(manifest).hexdigest()


def parse_manifest(data: bytes) -> dict[str, tuple[str, str, int, int]]:
    """A manifest's entries, checked: canonical, hex hashes, sensible shapes, not too big."""
    if len(data) > MAX_MANIFEST:
        raise MerkleError("a state manifest is too big")
    try:
        d = json.loads(data)
    except (ValueError, RecursionError):
        raise MerkleError("a state manifest isn't JSON") from None
    if not isinstance(d, dict) or set(d) != {"format", "tensors"} or d["format"] != FORMAT \
            or not isinstance(d["tensors"], dict) or len(d["tensors"]) > MAX_TENSORS:
        raise MerkleError(f"not a {FORMAT} manifest")
    out = {}
    for name, e in d["tensors"].items():
        if not (isinstance(name, str) and 0 < len(name) <= 200 and isinstance(e, list) and len(e) == 4
                and all(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in e[:2])
                and all(type(x) is int and 0 < x < 2**31 for x in e[2:])):
            raise MerkleError(f"a state manifest's entry for {name!r} is malformed")
        out[name] = (e[0], e[1], e[2], e[3])
    if manifest_bytes(out) != bytes(data):
        raise MerkleError("a state manifest isn't in canonical form")
    return out


# --- containers: a JSON header and raw bytes, for evidence and a referee's inputs ---------------------------------

def pack(header: dict, chunks: dict[str, bytes]) -> bytes:
    """header + named byte strings, as 8-byte header length | canonical JSON | the bytes back to back."""
    spans, at = {}, 0
    for k in sorted(chunks):
        spans[k] = [at, len(chunks[k])]
        at += len(chunks[k])
    head = canonical({"header": header, "spans": spans})
    return struct.pack("<Q", len(head)) + head + b"".join(chunks[k] for k in sorted(chunks))


def unpack(data: bytes, limit: int) -> tuple[dict, dict[str, bytes]]:
    if len(data) > limit or len(data) < 8:
        raise MerkleError("a container is too big, or empty")
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8:
        raise MerkleError("a container's header runs past its end")
    try:
        d = json.loads(data[8:8 + n])
    except (ValueError, RecursionError):
        raise MerkleError("a container's header isn't JSON") from None
    if not isinstance(d, dict) or set(d) != {"header", "spans"} or not isinstance(d["spans"], dict) \
            or canonical(d) != bytes(data[8:8 + n]):
        raise MerkleError("a container's header is malformed")
    body, out, at = data[8 + n:], {}, 0
    for k in sorted(d["spans"]):
        s = d["spans"][k]
        if not (isinstance(s, list) and len(s) == 2 and s[0] == at and type(s[1]) is int and 0 <= s[1] <= len(body) - at):
            raise MerkleError("a container's spans are malformed")
        out[k] = bytes(body[at:at + s[1]])
        at += s[1]
    if at != len(body):
        raise MerkleError("a container has bytes it doesn't name")
    return d["header"], out


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def unb64(text: str, limit: int) -> bytes:
    if not isinstance(text, str) or len(text) > limit * 4 // 3 + 8:
        raise MerkleError("too big")
    try:
        return base64.b64decode(text, validate=True)
    except ValueError:
        raise MerkleError("not base64") from None
