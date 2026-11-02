"""`vibehash` on its own: a menu of what people do most, picked with the arrow keys or a number, so nobody has to
remember a command. Every item runs the command `vibehash ...` would, and the menu shows that command, so people
learn them as they go.

Plain ANSI and the keyboard, nothing to install: msvcrt on Windows, termios elsewhere. A terminal that can't draw gets
a numbered list read with input() instead.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from . import settings, termui

GUIDE = "https://vibehash.org/guide/"


@dataclass(frozen=True)
class Item:
    label: str
    hint: str  # what it does, in a few words
    argv: tuple[str, ...] = ()  # the command it runs, as `vibehash` takes it
    ask: Callable[[], list[list[str]] | None] | None = None  # asks first, then gives the commands to run (None: cancel)
    action: Callable[[], None] | None = None  # for the few items that aren't a command


def _train() -> list[list[str]] | None:
    path = input("CSV file to train on (Enter for the sample, fruit.csv): ").strip().strip('"')
    budget = input("Budget in VIBE (Enter for 100): ").strip() or "100"
    if path:
        return [["train", path, "--budget", budget]]
    return [["example"], ["train", "fruit.csv", "--budget", budget]]  # writes fruit.csv here unless it's already here


def _update() -> None:
    print(f"> {sys.executable} -m pip install --upgrade vibehash")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "vibehash"])
    print("Updated. Quit and start `vibehash` again to use it." if code == 0 else f"pip stopped with exit code {code}.")


SECTIONS = [
    ("Mine", [
        Item("Start mining", "earn VIBE with this computer", ("worker",)),
        Item("My balance", "what you've earned and staked", ("balance",)),
        Item("What can this computer train on?", "its graphics cards and processor", ("devices",)),
    ]),
    ("Train", [
        Item("Train a model on a CSV file", "post it as a job; the network trains it", ask=_train),
        Item("Watch my job", "follow it live, round by round", ("job", "watch")),
        Item("My job's status", "rounds, accuracy, budget left", ("job", "status")),
        Item("Download my trained model", "save it to use with predict", ("job", "download")),
        Item("Models to fine-tune", "popular open-weight models and what they take", ("models",)),
    ]),
    ("Settings", [
        Item("Set up again", "network, wallet, graphics card, stake", ("setup",)),
        Item("My wallet", "its address and where its file is", ("wallet", "show")),
        Item("Update Vibe Hash", "install the latest release", action=_update),
        Item("Open the guide", "vibehash.org/guide, in your browser", action=lambda: webbrowser.open(GUIDE)),
    ]),
]
QUIT = Item("Quit", "")


def items() -> list[Item]:
    return [i for _, group in SECTIONS for i in group] + [QUIT]


# --- who's using it: the wallet and its balance, fetched in the background so the menu shows at once ------------------

class Status:
    def __init__(self):
        self.node = settings.node_url()
        path = settings.wallet_path()
        try:
            self.address = json.loads(path.read_text())["address"] if path and path.exists() else None
        except (OSError, ValueError, KeyError):
            self.address = None
        self.account: dict | None = None
        self.failed = False
        if self.address:
            threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self) -> None:
        from .client import NodeClient
        try:
            self.account = NodeClient(self.node).account(self.address)
        except Exception:  # offline, or a node that won't say: the menu works anyway
            self.failed = True

    def line(self) -> str:
        if not self.address:
            return "No wallet yet: choose Set up again to make one."
        who = f"{self.address[:9]}…{self.address[-4:]}"
        if self.account:
            fmt = termui._vibe
            return f"{who} · {fmt(self.account['balance'])} · staked {fmt(self.account['stake'])}"
        return f"{who} · {'balance unavailable (offline?)' if self.failed else 'checking balance…'}"


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("vibehash")
    except Exception:
        return "dev"


# --- the keyboard ------------------------------------------------------------------------------------------------

class Keys:
    """Single keypresses as names: up, down, home, end, enter, quit, a digit, or None when `timeout` passes."""

    def __enter__(self):
        if os.name != "nt":
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, timeout: float) -> str | None:
        return self._windows(timeout) if os.name == "nt" else self._posix(timeout)

    @staticmethod
    def name(ch: str) -> str | None:
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("q", "Q", "\x1b"):
            return "quit"
        if ch in ("k", "w"):
            return "up"
        if ch in ("j", "s"):
            return "down"
        return ch if ch.isdigit() else None

    def _windows(self, timeout: float) -> str | None:
        import msvcrt
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrows and the like come as two
                    return {"H": "up", "P": "down", "G": "home", "O": "end"}.get(msvcrt.getwch())
                return self.name(ch)
            time.sleep(0.02)
        return None

    def _posix(self, timeout: float) -> str | None:
        import select
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        ch = os.read(self.fd, 1).decode(errors="ignore")
        if ch == "\x1b" and select.select([self.fd], [], [], 0.05)[0]:
            seq = os.read(self.fd, 2).decode(errors="ignore")
            return {"[A": "up", "[B": "down", "[H": "home", "[F": "end", "OA": "up", "OB": "down"}.get(seq)
        return self.name(ch)


# --- drawing -----------------------------------------------------------------------------------------------------

def frame(choice: int, status: Status, prog: str, width: int, glyphs: dict) -> str:
    lines: list[str] = []
    add = lambda *segs: lines.append(termui._line([(" ", None), *segs], width))  # noqa: E731
    termui.header(add, glyphs, termui.ASCII, ["mining that trains real AI models",
                                             f"{_version()} · {urlsplit(status.node).hostname or status.node}"])
    add()
    add(("  " + status.line(), termui.TEXT))
    n, label_w = 0, max(len(i.label) for i in items()) + 4
    for title, group in SECTIONS + [("", [QUIT])]:
        add()
        if title:
            add(("  " + title.upper(), termui.DIM))
        for item in group:
            here = n == choice
            mark = f"› {n + 1 if n < 9 else ' '} " if here else f"  {n + 1 if n < 9 else ' '} "
            add(("  " + mark, termui.GOLD if here else termui.DIM),
                (item.label.ljust(label_w), termui.WHITE if here else termui.TEXT),
                (item.hint, termui.DIM))
            n += 1
    add()
    chosen = items()[choice]
    same = f"{prog} {' '.join(chosen.argv)}" if chosen.argv else ""
    add(("  same as:  ", termui.DIM), (same or "—", termui.YELLOW if same else termui.DIM))
    add(("  ↑↓ move · Enter choose · 1-9 jump · q quit" if glyphs is not termui.ASCII else
         "  up/down move, Enter choose, 1-9 jump, q quit", termui.DIM))
    return "\n".join(lines)


def pick(choice: int, status: Status, prog: str) -> int | None:
    """Draw the menu until something is chosen: its index, or None to quit."""
    glyphs = termui.glyphs_for(termui.GLYPHS, termui.ASCII)
    count = len(items())
    with termui.alternate_screen() as out, Keys() as keys:
        while True:
            width = max(40, os.get_terminal_size().columns) if sys.stdout.isatty() else 80
            out.write("\x1b[H" + frame(choice, status, prog, width, glyphs) + termui.RESET + "\x1b[J")
            out.flush()
            key = keys.read(0.5)  # redraw now and then, so the balance shows up once it's in
            if key == "up":
                choice = (choice - 1) % count
            elif key == "down":
                choice = (choice + 1) % count
            elif key == "home":
                choice = 0
            elif key == "end":
                choice = count - 1
            elif key and key.isdigit() and 1 <= int(key) <= min(9, count):
                choice = int(key) - 1
            elif key == "enter":
                return None if items()[choice] is QUIT else choice
            elif key == "quit":
                return None


def numbered(status: Status) -> int | None:
    """The menu as a numbered list, for a terminal that can't draw it."""
    print(f"Vibe Hash {_version()}. {status.line()}")
    n = 0
    for title, group in SECTIONS:
        print(f"\n{title}")
        for item in group:
            n += 1
            print(f"  {n:2}. {item.label} ({item.hint})")
    answer = input("\nNumber to run, or Enter to quit: ").strip()
    return int(answer) - 1 if answer.isdigit() and 1 <= int(answer) <= n else None


def run(execute: Callable[[list[str]], None], prog: str) -> None:
    """The menu, until Quit. `execute(argv)` runs one command line as `vibehash ARGS` would."""
    status, choice = Status(), 0
    while True:
        try:
            picked = pick(choice, status, prog) if termui.supported() else numbered(status)
        except (KeyboardInterrupt, EOFError):
            picked = None
        if picked is None:
            return
        choice, item = picked, items()[picked]
        try:
            if item.action:
                item.action()
            else:
                for argv in (item.ask() if item.ask else [list(item.argv)]) or []:
                    print(f"\n> {prog} {' '.join(argv)}")
                    execute(argv)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped")
        try:
            input("\nPress Enter to go back to the menu.")
        except (KeyboardInterrupt, EOFError):
            return
        status = Status()  # the balance may have changed
