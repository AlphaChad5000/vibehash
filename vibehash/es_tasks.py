"""Evo v2's tasks: problems a rule-2 ES job trains on, each a chat prompt asking for a short answer between <answer>
tags and a check of the reply that is pure Python (the node rescores every answer with it, no model needed).

Every draw goes through crypto.Draw (hash-based), so a round's problems are the same on every machine and Python
version, which the node and every worker have to agree on.

* countdown: Countdown at levels 3, 4 and 5 (es.LEVELS), a third each: mixed operations, where adding every number
  solves 19%, 14% and 5%.
* arithmetic: "a op b op c" with two-digit numbers and + - *, answered in one number.
* arc_easy: ARC-Easy multiple-choice science questions (allenai/ai2_arc through `datasets`, loaded on first use; the
  training questions are those whose question and choices fit in MAX_CHARS, which keeps a worker's caches small).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import es
from .crypto import Draw

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)
EVAL_N = 200
MAX_CHARS = 320
NAMES = ("countdown", "arithmetic", "arc_easy")


class TaskUnavailable(RuntimeError):
    """The task needs something this computer doesn't have (ARC: the `datasets` library or the download)."""


def last_answer(text: str):
    found = _ANS.findall(text)
    return found[-1].strip() if found else None


@dataclass(frozen=True)
class Item:
    messages: list = field(hash=False)  # the chat: [{"role", "content"}]
    check: Callable[[str], bool] = field(hash=False)
    gold: str = ""  # one right answer's text between the tags


def _draw(*parts) -> Draw:
    return Draw(*[p if isinstance(p, bytes) else str(p).encode() for p in parts])


def _number_check(gold):
    def ok(text: str) -> bool:
        a = last_answer(text)
        if a is None:
            return False
        a = a.replace(",", "").replace("$", "").strip().rstrip(".")
        try:
            return abs(float(a) - float(gold)) < 1e-6
        except ValueError:
            return False
    return ok


def _letter_check(key: str):
    def ok(text: str) -> bool:
        a = last_answer(text)
        return a is not None and a[:1].upper() == key.upper()
    return ok


# --- countdown ---

def _countdown_item(p) -> Item:
    return Item(es.messages(p), lambda t, p=p: es.reward(p, t) == 1.0, p.solution)


class Countdown:
    name = "countdown"
    LEVELS = (3, 4, 5)

    def items(self, seed: int, n: int) -> list[Item]:
        out = []
        for k, lv in enumerate(self.LEVELS):
            count = n // len(self.LEVELS) + (k < n % len(self.LEVELS))
            out += [_countdown_item(p) for p in es.problems(es.derive(b"evo2 countdown", seed, lv), count, lv)]
        return out

    def eval_items(self) -> list[Item]:
        return [_countdown_item(p) for p in es.heldout(67, 3) + es.heldout(67, 4) + es.heldout(66, 5)]

    def val_items(self, n: int) -> list[Item]:
        return self.items(2**62, n)  # a seed no round's comes near (round seeds are 64-bit hashes; this is fixed)


# --- arithmetic ---

def _arith(d: Draw) -> Item:
    a, b, c = 2 + d.below(98), 2 + d.below(98), 2 + d.below(29)
    o1, o2 = "+-*"[d.below(3)], "+-*"[d.below(3)]
    expr = f"{a} {o1} {b} {o2} {c}"
    value = es.parse(expr)[0]  # exact, with * before + and -
    q = f"What is {expr}? Reply with only the number between <answer> and </answer>."
    return Item([{"role": "user", "content": q}], _number_check(value), str(int(value)))


class Arithmetic:
    name = "arithmetic"

    def items(self, seed: int, n: int) -> list[Item]:
        d = _draw(b"evo2 arithmetic", seed)
        return [_arith(d) for _ in range(n)]

    def eval_items(self) -> list[Item]:
        d = _draw(b"evo2 arithmetic eval")
        return [_arith(d) for _ in range(EVAL_N)]

    def val_items(self, n: int) -> list[Item]:
        d = _draw(b"evo2 arithmetic validation")
        return [_arith(d) for _ in range(n)]


# --- ARC-Easy ---

def _arc_item(row) -> Item:
    choices = "\n".join(f"{lab}. {txt}" for lab, txt in zip(row["choices"]["label"], row["choices"]["text"]))
    q = (f"{row['question']}\n{choices}\nReply with only the letter of the correct choice between <answer> and "
         "</answer>.")
    return Item([{"role": "user", "content": q}], _letter_check(row["answerKey"]), row["answerKey"])


class ArcEasy:
    name = "arc_easy"

    def __init__(self):
        self._d = None
        self._short = None

    def _data(self):
        if self._d is None:
            try:
                from datasets import load_dataset
                self._d = load_dataset("allenai/ai2_arc", "ARC-Easy")
            except Exception as e:  # noqa: BLE001 - no library, no network: the task can't run here
                raise TaskUnavailable(f"ARC-Easy needs the `datasets` library and its download: {e}") from None
            train = self._d["train"]
            self._short = [i for i in range(len(train)) if len(train[i]["question"]) +
                           sum(len(c) for c in train[i]["choices"]["text"]) <= MAX_CHARS]
        return self._d

    def items(self, seed: int, n: int) -> list[Item]:
        train = self._data()["train"]
        d = _draw(b"evo2 arc_easy", seed)
        return [_arc_item(train[self._short[d.below(len(self._short))]]) for _ in range(n)]

    def _pick(self, split: str, tag: bytes, n: int) -> list[Item]:
        rows = self._data()[split]
        order = _draw(b"evo2 arc_easy", tag).shuffle(range(len(rows)))
        return [_arc_item(rows[i]) for i in order[:n]]

    def eval_items(self) -> list[Item]:
        return self._pick("test", b"eval", EVAL_N)

    def val_items(self, n: int) -> list[Item]:
        return self._pick("validation", b"validation", n)


_TASKS: dict = {}


def get(name: str):
    """The task called `name` (NAMES), made once per process."""
    if name not in NAMES:
        raise ValueError(f"no Evo task {name!r}: one of {', '.join(NAMES)}")
    if name not in _TASKS:
        _TASKS[name] = {"countdown": Countdown, "arithmetic": Arithmetic, "arc_easy": ArcEasy}[name]()
    return _TASKS[name]
