"""Locks between processes on one computer, through a lock file the operating system holds for its owner: two workers
on one machine (one per GPU) mustn't fetch the same model into the same files at once, or both run with one wallet's
cache. The lock goes when its process does, however it ends (a crash, Ctrl+C, a pod stopped), so a lock file left
behind never blocks anyone.

`hold(path)` waits for the lock; `try_hold(path)` takes it only if it's free. Standard library only: msvcrt on
Windows, fcntl elsewhere. Within one process, two holders of one path conflict too (each opens the file itself).
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path


def _lock(fd: int) -> bool:
    """Take the lock on an open lock file without waiting. -> whether it was free."""
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


class Lock:
    """A held lock: `release()` (or the end of its process) lets the next one in."""

    def __init__(self, path: Path, fd: int):
        self.path, self.fd = path, fd

    def release(self) -> None:
        if self.fd is not None:
            _unlock(self.fd)
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "Lock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _open(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)


def try_hold(path: str | Path, note: str = "") -> Lock | None:
    """The lock at `path` if nobody holds it, else None. `note` is written into it (who holds it)."""
    path = Path(path)
    fd = _open(path)
    if not _lock(fd):
        os.close(fd)
        return None
    if note:
        data = note.encode()[:200]
        try:  # after the locked byte: on Windows nobody else can read or write byte 0 while it's held
            os.ftruncate(fd, 1)
            os.lseek(fd, 1, os.SEEK_SET)
            os.write(fd, data)
        except OSError:
            pass
    return Lock(path, fd)


def holder(path: str | Path) -> str:
    """What the lock's holder noted in it, or ''."""
    try:
        with open(path, "rb") as f:
            f.seek(1)
            return f.read(200).decode(errors="replace").strip("\0 \n")
    except OSError:
        return ""


@contextlib.contextmanager
def hold(path: str | Path, wait_s: float = 3600.0, poll_s: float = 0.25):
    """Hold the lock at `path` for a `with` block, waiting up to `wait_s` for another holder to finish (TimeoutError
    after that)."""
    path, deadline = Path(path), time.monotonic() + wait_s
    while True:
        lock = try_hold(path)
        if lock is not None:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} stayed locked by another process for {wait_s:.0f} s")
        time.sleep(poll_s)
    try:
        yield lock
    finally:
        lock.release()
