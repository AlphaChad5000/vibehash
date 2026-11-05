"""Running with nobody at the keyboard: `nohup`, a tmux session left alone, a Docker CMD, a RunPod pod, a scheduled
task. Nothing may wait for an answer that can't come, and whatever needs one says which flag or variable gives it.

- `interactive()`: whether a question can be asked and answered here. Windows calls NUL (`< NUL`, Git Bash's
  /dev/null, a service's stdin) a terminal, and getpass then waits on the console for ever; this asks the console.
- `need(what, how)`: the one-line error for an answer that's needed when there's no terminal.
- `save_words(...)`: a new wallet's 24 recovery words when nobody is watching the screen. They'd otherwise land in a
  log (nohup.out, `docker logs`, a pod's web log), so they go to a file only this user can read, and the log gets a
  loud note saying where it is and to move the words offline.
- `WorkerLock`: one worker per wallet on a computer. Two workers with one wallet share its cache (each start prunes
  the other's checkpoints) and look like one worker to the node, so the second is stopped with what to do instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from . import filelock, settings
from .crypto import checksummed


def interactive(stream=None) -> bool:
    """Whether `stream` (stdin by default) is a terminal someone can type into."""
    stream = sys.stdin if stream is None else stream
    try:
        if stream is None or not stream.isatty():
            return False
    except (AttributeError, ValueError, OSError):  # closed, or not a real file
        return False
    if os.name != "nt":
        return True
    try:  # NUL and other character devices say isatty(); only a console has a console mode
        import ctypes
        import msvcrt

        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(stream.fileno()), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return True  # can't tell: assume a terminal, as isatty() does


def need(what: str, how: str) -> SystemExit:
    """`raise need("a wallet password", "set VIBEHASH_PASSWORD")`: no terminal to ask on, and what gives the answer."""
    return SystemExit(f"error: {what} is needed and there's no terminal to ask on (nohup, Docker, a pod's log?): {how}")


def words_file(address: str) -> Path:
    return settings.home() / f"recovery-words-{address[:12]}.txt"


def save_words(words: str, address: str, wallet: Path) -> Path:
    """Write a new wallet's recovery words to a file only this user can read (on Windows, the user's own folder keeps
    them private), never overwriting one. -> its path."""
    path = words_file(address)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = words.split()
    text = (f"Vibe Hash recovery words for {checksummed(address)}\nwallet file: {wallet}\n\n"
            + "".join(f"{i + 1:>2}. {word}\n" for i, word in enumerate(w))
            + "\nWrite them down or keep them in a password manager, offline, then delete this file. They restore the "
              "wallet if its file or password is lost (vibehash wallet restore), and anyone who reads them can take "
              "its coins.\n")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def words_note(path: Path, indent: str = "") -> str:
    """What the log says instead of the words."""
    remove = f'del "{path}"' if os.name == "nt" else f"rm '{path}'"
    return (f"\n{indent}!!! IMPORTANT: this wallet's 24 recovery words were NOT printed here, because this output may be\n"
            f"{indent}!!! kept in a log others can read. They are in {path}\n"
            f"{indent}!!! (readable only by you). Copy them somewhere offline now, then delete the file: {remove}\n"
            f"{indent}!!! Without them a lost wallet file or password can't be undone; with them anyone can take its coins.\n")


class WorkerLock:
    """Held while a worker runs, in its cache folder (one per wallet unless --cache-dir says otherwise)."""

    def __init__(self, cache: Path):
        self.path = Path(cache) / "worker.lock"
        self.lock = filelock.try_hold(self.path, note=f"process {os.getpid()}")

    @property
    def held(self) -> bool:
        return self.lock is not None

    def why_not(self, address: str, prog: str = "vibehash") -> str:
        who = filelock.holder(self.path)
        return (f"error: another worker ({who or 'still running'}) already mines with wallet "
                f"{checksummed(address)[:14]}... on this computer, and two with one wallet would undo each other's "
                f"work. For a second GPU, give it its own wallet: `{prog} wallet create --out gpu1.json`, then "
                f"`{prog} worker --device cuda:1 --wallet gpu1.json`")

    def release(self) -> None:
        if self.lock is not None:
            self.lock.release()
            self.lock = None
