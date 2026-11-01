"""Content-addressed files for model weights, so checkpoints never travel inside JSON.

A blob's id is the SHA-256 of its bytes: raw little-endian floats at the job's
precision. Whoever receives a blob re-hashes it, so a node or a worker can't
be handed different bytes than the id promises. Uploads arrive in sequential
chunks and resume from whatever already arrived. Loaded arrays are
memory-mapped, so a 1 GB checkpoint costs disk space, not RAM.
"""
from __future__ import annotations

import gc
import hashlib
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

CHUNK = 8 * 2**20  # transfer unit, not a size limit
MAX_DATASET = 20 * 2**30  # the biggest dataset a node takes (so the biggest a download can be)
STALE_PART_S = 24 * 3600  # a partial transfer nothing has written to for this long was abandoned
_ID = re.compile(r"^[0-9a-f]{64}$")
_OWNER = re.compile(r"^[0-9a-z]{0,64}$")


class BlobError(ValueError):
    pass


def check_id(blob_id: str) -> str:
    if not isinstance(blob_id, str) or not _ID.match(blob_id):
        raise BlobError("bad blob id")
    return blob_id


def array_bytes(arr: np.ndarray, dtype: str) -> np.ndarray:
    """The canonical byte layout of an array at `dtype` (a contiguous array, no copy if already right)."""
    return np.ascontiguousarray(arr, dtype=dtype)


def array_id(arr: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(memoryview(array_bytes(arr, dtype)).cast("B")).hexdigest()


class BlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._finishing = threading.Lock()  # uploads of one blob can complete at the same moment

    def path(self, blob_id: str) -> Path:
        return self.root / check_id(blob_id)

    def _part(self, blob_id: str, owner: str = "") -> Path:
        """A partial upload. Each uploader (`owner`) gets its own, so two uploads of one blob never mix: two
        workers holding a replicated unit produce identical checkpoints, and either may finish first."""
        if not _OWNER.match(owner):
            raise BlobError("bad uploader")
        return self.root / (f"{check_id(blob_id)}.{owner}.part" if owner else f"{check_id(blob_id)}.part")

    def has(self, blob_id: str) -> bool:
        return self.path(blob_id).exists()

    def size(self, blob_id: str) -> int | None:
        p = self.path(blob_id)
        return p.stat().st_size if p.exists() else None

    def received(self, blob_id: str, owner: str = "") -> int:
        """Bytes already stored: the full size if complete, else how far `owner`'s partial upload got."""
        done = self.size(blob_id)
        if done is not None:
            return done
        part = self._part(blob_id, owner)
        return part.stat().st_size if part.exists() else 0

    # --- writing ------------------------------------------------------------------

    def put_array(self, arr: np.ndarray, dtype: str) -> str:
        data = array_bytes(arr, dtype)
        blob_id = hashlib.sha256(memoryview(data).cast("B")).hexdigest()
        if not self.has(blob_id):
            tmp = self.root / f"{blob_id}.{os.getpid()}.tmp"
            data.tofile(tmp)
            os.replace(tmp, self.path(blob_id))
        return blob_id

    def write_chunk(self, blob_id: str, offset: int, chunk: bytes, total: int, owner: str = "", check=None) -> int:
        """Append one chunk of `owner`'s upload. Returns bytes received so far (a finished file's real size, whatever
        `total` says).

        Chunks must arrive in order (offset == bytes already received), which makes
        retries safe: a repeated chunk is ignored. The finished file must hash to its id,
        and pass `check(path)` if given (a dataset must load) before it's kept: so only the
        write that completes a file vets it, and a file once kept is never thrown away.
        """
        done = self.size(blob_id)
        if done is not None:
            return done
        part = self._part(blob_id, owner)
        have = part.stat().st_size if part.exists() else 0
        if offset < have and offset + len(chunk) <= have:
            return have  # duplicate of something we already have
        if offset != have:
            raise BlobError(f"expected offset {have}, got {offset}")
        if have + len(chunk) > total:
            raise BlobError("upload longer than declared")
        with open(part, "ab") as f:
            f.write(chunk)
        have += len(chunk)
        if have == total:
            if file_sha256(part) != blob_id:
                part.unlink(missing_ok=True)
                raise BlobError("upload doesn't match its id; discarded")
            if check is not None:
                try:
                    check(part)
                except BaseException:
                    part.unlink(missing_ok=True)
                    raise
            with self._finishing:
                if self.has(blob_id):  # another uploader finished first; same bytes, and the file may be in use
                    part.unlink(missing_ok=True)
                else:
                    os.replace(part, self.path(blob_id))
        return have

    # --- reading ------------------------------------------------------------------

    def read_chunk(self, blob_id: str, offset: int, length: int) -> bytes:
        p = self.path(blob_id)
        if not p.exists():
            raise KeyError(blob_id)
        with open(p, "rb") as f:
            f.seek(offset)
            return f.read(min(length, CHUNK))

    def load(self, blob_id: str, dtype: str, n: int) -> np.ndarray:
        """Memory-mapped, read-only view. Checks the length matches the model."""
        p = self.path(blob_id)
        if not p.exists():
            raise KeyError(blob_id)
        if p.stat().st_size != np.dtype(dtype).itemsize * n:
            raise BlobError("blob is the wrong size for this model")
        return np.memmap(p, dtype=dtype, mode="r", shape=(n,))

    # --- housekeeping -------------------------------------------------------------

    def prune(self, keep: set[str] = frozenset()) -> int:
        """Delete every blob (and partial upload) except `keep`. Returns how many were removed."""
        gc.collect()  # release memory maps so Windows lets the files go
        removed = 0
        for p in self.root.iterdir():
            if p.name.split(".")[0] in keep:
                continue
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass  # still mapped somewhere; the next prune gets it
        return removed

    def prune_partial(self, keep: set[str] = frozenset(), older_than: float = 0.0) -> int:
        """Delete partial transfers, except those of blobs in `keep` and any written to in the last `older_than`
        seconds: one that stopped part way (a node that went away, or lied about the size) otherwise stays for good."""
        now, removed = time.time(), 0
        for p in self.root.glob("*.part"):
            try:
                if p.name.split(".")[0] not in keep and now - p.stat().st_mtime >= older_than:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        return removed


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
