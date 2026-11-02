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
    usage: str = ""  # the command an asking item runs, with its blanks named, for "same as"
    submenu: tuple = ()  # (title, [items]) sections of a page of its own


def _train() -> list[list[str]] | None:
    path = input("CSV file to train on (Enter for the sample, fruit.csv): ").strip().strip('"')
    budget = input("Budget in VIBE (Enter for 100): ").strip() or "100"
    if path:
        return [["train", path, "--budget", budget]]
    return [["example"], ["train", "fruit.csv", "--budget", budget]]  # writes fruit.csv here unless it's already here


def _finetune() -> list[list[str]] | None:
    data = input("Chats file, one conversation per line (the guide shows the format): ").strip().strip('"')
    if not data:
        return None
    model = input("Model to fine-tune (Enter for qwen3-0.6b; Models to fine-tune lists them): ").strip() or "qwen3-0.6b"
    budget = input("Budget in VIBE (Enter for 100): ").strip() or "100"
    return [["train", data, "--base", model, "--budget", budget]]


def _yes(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _send() -> list[list[str]] | None:
    to = input("Send to (an address starting vibe1): ").strip()
    amount = input("How many VIBE: ").strip()
    if to and amount and _yes(f"Send {amount} VIBE to {to}? It can't be undone."):
        return [["send", "--to", to, "--amount", amount]]
    return None


def _amount(cmd: str, question: str) -> Callable[[], list[list[str]] | None]:
    def ask() -> list[list[str]] | None:
        amount = input(question).strip()
        return [[cmd, "--amount", amount]] if amount else None
    return ask


def _predict() -> list[list[str]] | None:
    data = input("CSV of new rows (Enter for the sample, fruit-new.csv): ").strip().strip('"')
    return [["predict", data or "fruit-new.csv"]]


def _post_job() -> list[list[str]] | None:
    spec = input("Job file (Enter for job.json): ").strip().strip('"') or "job.json"
    budget = input("Budget in VIBE: ").strip()
    return [["post-job", "--spec", spec, "--budget", budget]] if budget else None


def _deepseek() -> list[list[str]] | None:
    print("The DeepSeek job, think shorter's pilot, fine-tunes DeepSeek-V4.1-Flash's top layers. Your PC needs an\n"
          "NVIDIA graphics card with 8 GB or more (RTX 20-series or newer), 16 GB of memory and about 40 GB free on an\n"
          "SSD. It downloads about 30 GB of the model into the folder setup chose, and deletes it when the job ends,\n"
          "unless you chose to keep it (`vibehash cache` shows what's there). A PC that can't take it keeps mining\n"
          "other jobs.")
    return [["worker", "--deepseek", "yes"]] if _yes("Mine it?") else None


def _cancel() -> list[list[str]] | None:
    return [["cancel-job"]] if _yes("Cancel your last training job?") else None


def _update() -> None:
    print(f"> {sys.executable} -m pip install --upgrade vibehash")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "vibehash"])
    print("Updated. Quit and start `vibehash` again to use it." if code == 0 else f"pip stopped with exit code {code}.")


MORE = (
    ("Wallet", [
        Item("Send VIBE", "to another address", ask=_send, usage="send --to ADDRESS --amount VIBE"),
        Item("Stake more", "lock VIBE as your stake", ask=_amount("stake", "How many VIBE to stake: "),
             usage="stake --amount VIBE"),
        Item("Unstake", "unlock stake; it waits 20 blocks", ask=_amount("unstake", "How many VIBE to unstake: "),
             usage="unstake --amount VIBE"),
        Item("Make a new wallet", "with 24 new recovery words", ("wallet", "create")),
        Item("Restore a wallet", "from its 24 recovery words", ("wallet", "restore")),
    ]),
    ("Train", [
        Item("Predict with my model", "run it on new rows of a CSV", ask=_predict, usage="predict FILE.csv"),
        Item("Write a starter job file", "for full control over a job", ("job", "template")),
        Item("Post a job file", "a job written by hand", ask=_post_job, usage="post-job --spec FILE --budget VIBE"),
        Item("Cancel my job", "stop it and get back what's unspent", ask=_cancel, usage="cancel-job"),
    ]),
    ("Network", [
        Item("Validators", "who runs the network, and their stake", ("validator", "list")),
        Item("Verify the whole chain", "re-check every block from the start", ("chain", "verify")),
        Item("Install GPU support", "PyTorch for your graphics card", ("install-gpu",)),
        Item("Run a test network here", "a node, workers and a job, on this computer", ("testnet",)),
    ]),
    ("Help", [
        Item("Every command", "and all of their options", ("--help",)),
    ]),
)

SECTIONS = [
    ("Mine", [
        Item("Start mining", "earn VIBE with this computer", ("worker",)),
        Item("My balance", "what you've earned and staked", ("balance",)),
        Item("What can this computer train on?", "its graphics cards and processor", ("devices",)),
        Item("Check this computer", "a real training test: is it ready, what can it take", ("selftest",)),
        Item("Mine the DeepSeek job", "think shorter's pilot: ~30 GB, asked first", ask=_deepseek,
             usage="worker --deepseek yes"),
    ]),
    ("Train", [
        Item("Train a model on a CSV file", "post it as a job; the network trains it", ask=_train,
             usage="train FILE.csv --budget VIBE"),
        Item("Fine-tune a model on my chats", "teach an open model your own text", ask=_finetune,
             usage="train CHATS.jsonl --base MODEL --budget VIBE"),
        Item("Watch my job", "follow it live, round by round", ("job", "watch")),
        Item("My job's status", "rounds, accuracy, budget left", ("job", "status")),
        Item("Download my trained model", "save it to use with predict", ("job", "download")),
        Item("Models to fine-tune", "popular open-weight models and what they take", ("models",)),
    ]),
    ("Settings", [
        Item("Set up again", "network, wallet, graphics card, stake", ("setup",)),
        Item("My wallet", "its address and where its file is", ("wallet", "show")),
        Item("Model files", "what big models take on this PC; clear them", ("cache",)),
        Item("Update Vibe Hash", "install the latest release", action=_update),
        Item("Open the guide", "vibehash.org/guide, in your browser", action=lambda: webbrowser.open(GUIDE)),
        Item("More commands", "send, stake, predict, validators and the rest", submenu=MORE),
    ]),
]
QUIT, BACK = Item("Quit", ""), Item("Back", "to the main menu")


def items(sections=None, last: Item = QUIT) -> list[Item]:
    return [i for _, group in (SECTIONS if sections is None else sections) for i in group] + [last]


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

def frame(choice: int, status: Status, prog: str, width: int, glyphs: dict, sections=None, last: Item = QUIT) -> str:
    sections = SECTIONS if sections is None else sections
    lines: list[str] = []
    add = lambda *segs: lines.append(termui._line([(" ", None), *segs], width))  # noqa: E731
    termui.header(add, glyphs, termui.ASCII, ["mining that trains real AI models",
                                             f"{_version()} · {urlsplit(status.node).hostname or status.node}"])
    add()
    add(("  " + status.line(), termui.TEXT))
    page = items(sections, last)
    n, label_w = 0, max(len(i.label) for i in page) + 4
    for title, group in list(sections) + [("", [last])]:
        add()
        if title:
            add(("  " + title.upper(), termui.DIM))
        for item in group:
            here = n == choice
            mark = f"› {n + 1 if n < 9 else ' '} " if here else f"  {n + 1 if n < 9 else ' '} "
            add(("  " + mark, termui.GOLD if here else termui.DIM),
                ((item.label + (" …" if item.submenu else "")).ljust(label_w), termui.WHITE if here else termui.TEXT),
                (item.hint, termui.DIM))
            n += 1
    add()
    chosen = page[choice]
    same = f"{prog} {' '.join(chosen.argv) or chosen.usage}" if chosen.argv or chosen.usage else ""
    add(("  same as:  ", termui.DIM), (same or "—", termui.YELLOW if same else termui.DIM))
    leave = "q quit" if last is QUIT else "q back"
    add((f"  ↑↓ move · Enter choose · 1-9 jump · {leave}" if glyphs is not termui.ASCII else
         f"  up/down move, Enter choose, 1-9 jump, {leave}", termui.DIM))
    return "\n".join(lines)


def pick(choice: int, status: Status, prog: str, sections=None, last: Item = QUIT) -> int | None:
    """Draw a page of the menu until something is chosen: its index, or None to quit (or go back)."""
    glyphs = termui.glyphs_for(termui.GLYPHS, termui.ASCII)
    page = items(sections, last)
    count = len(page)
    with termui.alternate_screen() as out, Keys() as keys:
        while True:
            width = max(40, os.get_terminal_size().columns) if sys.stdout.isatty() else 80
            out.write("\x1b[H" + frame(choice, status, prog, width, glyphs, sections, last) + termui.RESET + "\x1b[J")
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
                return None if page[choice] is last else choice
            elif key == "quit":
                return None


def numbered(status: Status, sections=None, last: Item = QUIT) -> int | None:
    """A page of the menu as a numbered list, for a terminal that can't draw it."""
    print(f"Vibe Hash {_version()}. {status.line()}")
    n = 0
    for title, group in SECTIONS if sections is None else sections:
        print(f"\n{title}")
        for item in group:
            n += 1
            print(f"  {n:2}. {item.label} ({item.hint})")
    answer = input(f"\nNumber to run, or Enter to {'quit' if last is QUIT else 'go back'}: ").strip()
    return int(answer) - 1 if answer.isdigit() and 1 <= int(answer) <= n else None


def run(execute: Callable[[list[str]], None], prog: str) -> None:
    """The menu, until Quit. `execute(argv)` runs one command line as `vibehash ARGS` would."""
    status = Status()
    pages = [[SECTIONS, QUIT, 0]]  # the main menu, and "More commands" on top of it while it's open
    while pages:
        sections, last, choice = pages[-1]
        try:
            picked = pick(choice, status, prog, sections, last) if termui.supported() else numbered(status, sections, last)
        except (KeyboardInterrupt, EOFError):
            picked = None
        if picked is None:
            pages.pop()  # Quit, or Back to the main menu
            continue
        pages[-1][2], item = picked, items(sections, last)[picked]
        if item.submenu:
            pages.append([item.submenu, BACK, 0])
            continue
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
