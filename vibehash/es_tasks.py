"""Evo v2's tasks: problems a rule-2 ES job trains on, each a chat prompt asking for a short answer between <answer>
tags and a check of the reply that is pure Python (the node rescores every answer with it, no model needed).

Every draw goes through crypto.Draw (hash-based), so a round's problems are the same on every machine and Python
version, which the node and every worker have to agree on.

* countdown: Countdown at levels 3, 4 and 5 (es.LEVELS), a third each: mixed operations, where adding every number
  solves 19%, 14% and 5%.
* arithmetic: "a op b op c" with two-digit numbers and + - *, answered in one number.
* arc_easy: ARC-Easy multiple-choice science questions (allenai/ai2_arc through `datasets`, loaded on first use; the
  training questions are those whose question and choices fit in MAX_CHARS, which keeps a worker's caches small).
* sixteen more from Hugging Face datasets (HubTask, HUB), each pinned at a commit so every machine reads the same rows:
  science, commonsense, reading, word-problem math, trivia, sentiment, news topics and emotions. Multiple choice is
  answered with a letter, math with a number, trivia with a few words checked against the dataset's accepted answers.

NAMES is every task a job may name (the first nineteen in the order Evo Hash first trained them; ARC-Challenge, PIQA
and Social IQa came later). CURATED is what Evo Hash trains on now: the tasks a small model can learn from, measured on
Qwen2.5-0.5B-Instruct (2026-11-05, 200 held-out items each): trivia_qa (0.5% right), nq_open (0%) and web_questions
(0.5%) ask for facts the model doesn't hold, which reward-driven training can't add; countdown and arithmetic are
puzzles no one benchmarks. GSM8K and MMLU stay out of every task, so a tuned model can be scored on them as never seen.

`lenient(item)`: the same problem, its check also taking a right answer given without the <answer> tags ("B", "(B)",
"The answer is B", the last number, a leading yes or no). A job whose spec says `lenient` (es2.Spec2) is scored that
way, so its reward is for being right, not for writing the tags: the base model leaves them out of a quarter to a half
of its replies on some tasks (boolq 49.5% in format), and a strict check pays a model for learning the tags, which
reads as a gain that standard benchmarks don't see.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Callable

from . import es
from .crypto import Draw

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)
EVAL_N = 200
MAX_CHARS = 320
NAMES = ("countdown", "arithmetic", "arc_easy", "sciq", "svamp", "trivia_qa", "commonsense_qa", "sst2", "openbookqa",
         "asdiv", "nq_open", "winogrande", "ag_news", "qasc", "aqua_rat", "web_questions", "hellaswag", "emotion",
         "boolq", "arc_challenge", "piqa", "social_iqa")
# What Evo Hash trains on (see the docstring): the knowledge benchmarks first, then commonsense, math, classification.
CURATED = ("arc_easy", "sciq", "commonsense_qa", "openbookqa", "arc_challenge", "piqa", "social_iqa", "qasc",
           "winogrande", "hellaswag", "boolq", "aqua_rat", "svamp", "asdiv", "sst2", "ag_news", "emotion")
MAX_LONG = 520  # a Hugging Face task's longest prompt, in characters (about 150 tokens)
POOL = 20_000  # the most rows a task keeps the indices of for each part


class TaskUnavailable(RuntimeError):
    """The task needs something this computer doesn't have (ARC: the `datasets` library or the download)."""


def last_answer(text: str):
    found = _ANS.findall(text)
    return found[-1].strip() if found else None


_LETTER_ALONE = re.compile(r"^\(?([A-H])\)?(?:[.):]|$)")  # "B", "B.", "B)", "(B)", "B: ..." (not "A dog ...")
_LETTER_SAID = re.compile(r"\banswer(?:\s+is)?\s*[:\-]?\s*\(?([A-H])\b", re.I)  # "the answer is B", "Answer: (C)"
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_YES_NO = re.compile(r"^(yes|no)\b", re.I)


def salvage(text: str, gold: str) -> str | None:
    """A reply's answer when it has no <answer> tags, read the way the item's gold is written: a letter (alone at the
    start, or after "answer is"), a number (the last one), yes or no (the first word), or a short reply as a whole.
    None if there's nothing to read."""
    t = text.strip()
    if len(gold) == 1 and gold in LETTERS:
        said = [m.group(1).upper() for m in _LETTER_SAID.finditer(t)]
        if said:
            return said[-1]
        m = _LETTER_ALONE.match(t)
        return m.group(1) if m else None
    try:
        float(gold)
        nums = _NUMBER.findall(t)
        return nums[-1].replace(",", "") if nums else None
    except ValueError:
        pass
    if gold.lower() in ("yes", "no"):
        m = _YES_NO.match(t)
        return m.group(1).lower() if m else None
    return t.rstrip(".") if 0 < len(t) <= 60 and "\n" not in t else None


def lenient(item: "Item") -> "Item":
    """The item, its check also taking a right answer given without tags (`salvage`); a reply with tags is checked as
    before."""
    strict = item.check

    def ok(text: str) -> bool:
        if last_answer(text) is not None:
            return strict(text)
        a = salvage(text, item.gold)
        return a is not None and strict(f"<answer>{a}</answer>")
    return Item(item.messages, ok, item.gold)


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


# --- tasks from Hugging Face datasets ---

LETTERS = "ABCDEFGH"
_ASK_LETTER = "Reply with only the letter of the correct choice between <answer> and </answer>."
_ASK_NUMBER = "Reply with only the number between <answer> and </answer>."
_ASK_SHORT = "Reply with only the answer, in a few words, between <answer> and </answer>."
_ASK_YES_NO = "Reply with only yes or no between <answer> and </answer>."


def _fits(text: str, limit: int = MAX_LONG):
    return [{"role": "user", "content": text}] if len(text) <= limit else None


def _norm(s: str) -> str:
    """SQuAD's answer normalisation: lower case, no punctuation or articles, single spaces."""
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def _words_check(accepted):
    ok_set = {_norm(a) for a in accepted} - {""}

    def ok(text: str) -> bool:
        a = last_answer(text)
        return a is not None and _norm(a) in ok_set
    return ok


def _mc(question: str, texts: list, key: int):
    """A multiple-choice item, choices lettered A, B, ...; None if it's too long or `key` isn't one of them."""
    if not 0 <= key < len(texts) <= len(LETTERS):
        return None
    m = _fits(question.strip() + "\n" + "\n".join(f"{LETTERS[i]}. {str(t).strip()}" for i, t in enumerate(texts)) +
              "\n" + _ASK_LETTER)
    return Item(m, _letter_check(LETTERS[key]), LETTERS[key]) if m else None


def _labelled(row, question: str):
    """A multiple-choice item from a row with choices {text, label} and an answerKey."""
    labels = list(row["choices"]["label"])
    return _mc(question, row["choices"]["text"], labels.index(row["answerKey"]) if row["answerKey"] in labels else -1)


def _short(question: str, accepted: list):
    accepted = [a for a in accepted if isinstance(a, str) and _norm(a)]
    m = _fits(question.strip() + "\n" + _ASK_SHORT) if accepted else None
    return Item(m, _words_check(accepted), accepted[0]) if m else None


def _number(question: str, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    gold = str(int(v)) if v.is_integer() else repr(v)
    m = _fits(question.strip() + "\n" + _ASK_NUMBER)
    return Item(m, _number_check(gold), gold) if m else None


def _story(body: str, question: str) -> str:
    body = body.strip()
    return f"{body}{'' if body[-1:] in '.!?' else '.'} {question.strip()}"


def _sciq(row):
    order = _draw(b"evo2 sciq choices", row["question"]).shuffle([0, 1, 2, 3])
    texts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    return _mc(row["question"], [texts[i] for i in order], order.index(0))


def _aqua(row):
    opts = [o.strip() for o in row["options"]]
    if not all(len(o) > 2 and o[1] == ")" for o in opts):
        return None
    labels = [o[0] for o in opts]
    return _mc(row["question"], [o[2:] for o in opts], labels.index(row["correct"]) if row["correct"] in labels else -1)


def _asdiv(row):
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(\([^()]*\))?\s*", row["answer"])
    return _number(_story(row["body"], row["question"]), m.group(1)) if m else None


def _boolq(row):
    m = _fits(f"{row['passage'].strip()}\nQuestion: {row['question'].strip()}?\n{_ASK_YES_NO}", 600)
    gold = "yes" if row["answer"] else "no"
    return Item(m, _words_check([gold]), gold) if m else None


def _hellaswag(row):
    if not str(row["label"]).isdigit():
        return None
    return _mc(f"Which ending fits best?\n{row['activity_label']}: {row['ctx']}", row["endings"], int(row["label"]))


def _winogrande(row):
    key = {"1": 0, "2": 1}.get(row["answer"], -1)
    return _mc(f"Which option fits the blank (_)?\n{row['sentence']}", [row["option1"], row["option2"]], key)


def _nq(row):
    q = row["question"].strip()
    return _short(q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?"), list(row["answer"]))


class HubTask:
    """A task from a Hugging Face dataset at a pinned commit, loaded through `datasets` on first use. `make(row)` turns
    a row into an Item, or None to leave it out (too long, or no usable answer). A part is a split's name, or (split,
    lo, hi): that share of the split in a fixed shuffled order, for a dataset with one labelled split. Training draws
    from `train`; eval_items and val_items come from parts the training never sees."""

    def __init__(self, name, repo, config, revision, train, test, val, make):
        self.name, self.repo, self.config, self.revision = name, repo, config, revision
        self.train, self.test, self.val, self.make = train, test, val or test, make
        self._d = None
        self._pools: dict = {}

    def _data(self):
        if self._d is None:
            try:
                from datasets import load_dataset
                kw = self.config if isinstance(self.config, dict) else {"name": self.config}
                self._d = load_dataset(self.repo, revision=self.revision, **kw)
            except Exception as e:  # noqa: BLE001 - no library, no network: the task can't run here
                raise TaskUnavailable(f"{self.name} needs the `datasets` library and its download: {e}") from None
        return self._d

    def _pool(self, part):
        """`part`'s split, and the indices of its rows that make an item (at most POOL), in a fixed shuffled order."""
        split, lo, hi = (part, 0.0, 1.0) if isinstance(part, str) else part
        rows = self._data()[split]
        if part not in self._pools:
            order = _draw(b"evo2 order", self.name, split).shuffle(list(range(len(rows))))
            keep = []
            for i in order[int(lo * len(order)):int(hi * len(order))]:
                if self.make(rows[i]) is not None:
                    keep.append(i)
                    if len(keep) >= POOL:
                        break
            if not keep:
                raise TaskUnavailable(f"{self.name}: no usable rows in {part}")
            self._pools[part] = keep
        return rows, self._pools[part]

    def items(self, seed: int, n: int) -> list[Item]:
        rows, pool = self._pool(self.train)
        d = _draw(b"evo2 " + self.name.encode(), seed)
        return [self.make(rows[pool[d.below(len(pool))]]) for _ in range(n)]

    def eval_items(self) -> list[Item]:
        rows, pool = self._pool(self.test)
        return [self.make(rows[i]) for i in pool[:EVAL_N]]

    def val_items(self, n: int) -> list[Item]:
        rows, pool = self._pool(self.val)
        return [self.make(rows[i]) for i in pool[-n:]]


def _half(split: str, first: bool):
    return (split, 0.0, 0.5) if first else (split, 0.5, 1.0)


# name: (repo, config, revision, train, test, val, make)
HUB = {
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815", "train", "test", "validation", _sciq),
    "svamp": ("ChilleD/SVAMP", None, "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef", "train", "test", None,
              lambda r: _number(_story(r["Body"], r["Question"]), r["Answer"])),
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc.nocontext", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "train",
                  _half("validation", True), _half("validation", False),
                  lambda r: _short(r["question"], [r["answer"]["value"], *r["answer"]["aliases"]])),
    "commonsense_qa": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155", "train",
                       _half("validation", True), _half("validation", False), lambda r: _labelled(r, r["question"])),
    "sst2": ("nyu-mll/glue", "sst2", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "train", _half("validation", True),
             _half("validation", False),
             lambda r: _mc(f'What is the sentiment of this movie review?\n"{r["sentence"].strip()}"',
                           ["negative", "positive"], r["label"])),
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454", "train", "test",
                   "validation", lambda r: _labelled(r, r["question_stem"])),
    "asdiv": ("EleutherAI/asdiv", None, "8f95807222d87b4c688c3c22a6ba2801e1fa03e2", ("validation", 0.0, 0.8),
              ("validation", 0.8, 0.9), ("validation", 0.9, 1.0), _asdiv),
    "nq_open": ("google-research-datasets/nq_open", None, "5dd9790a83002ad084ddeb7c420dc716852c6f28", "train",
                _half("validation", True), _half("validation", False), _nq),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "train",
                   _half("validation", True), _half("validation", False), _winogrande),
    "ag_news": ("fancyzhx/ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4", "train", "test", None,
                lambda r: _mc("What is this news story about?\n" + r["text"].replace("\\", " "),
                              ["World", "Sports", "Business", "Science and technology"], r["label"])),
    "qasc": ("allenai/qasc", None, "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "train", _half("validation", True),
             _half("validation", False), lambda r: _labelled(r, r["question"])),
    "aqua_rat": ("deepmind/aqua_rat", "raw", "33301c6a050c96af81f63cad5562cb5363e88971", "train", "test", "validation",
                 _aqua),
    "web_questions": ("stanfordnlp/web_questions", None, "0e473cbe21d1e91ec18da343644498be6a3f5454", "train", "test",
                      None, lambda r: _short(r["question"], list(r["answers"]))),
    "hellaswag": ("Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", "train",
                  _half("validation", True), _half("validation", False), _hellaswag),
    "emotion": ("dair-ai/emotion", "split", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "train", "test", "validation",
                lambda r: _mc(f'Which emotion does this message show?\n"{r["text"].strip()}"',
                              ["sadness", "joy", "love", "anger", "fear", "surprise"], r["label"])),
    "boolq": ("google/boolq", None, "35b264d03638db9f4ce671b711558bf7ff0f80d5", "train", _half("validation", True),
              _half("validation", False), _boolq),
    # Added 2026-11-05 for the curated mix. PIQA and Social IQa are loading scripts upstream, which `datasets` no
    # longer runs: these are Hugging Face's parquet conversions, pinned at their commits. PIQA's test answers are
    # hidden, so its held-out items are half of validation, as Social IQa's (it has no test split).
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "train", "test",
                      "validation", lambda r: _labelled(r, r["question"])),
    "piqa": ("ybisk/piqa", {"data_dir": "plain_text"}, "142c51238b3ca2bc61e9a075913871b8b600e8e1", "train",
             _half("validation", True), _half("validation", False),
             lambda r: _mc(f"Which is the better way to do this?\n{r['goal']}", [r["sol1"], r["sol2"]],
                           int(r["label"]))),
    "social_iqa": ("allenai/social_i_qa", None, "537a2ec8ec565adc0b70b70752893e59e024df26", "train",
                   _half("validation", True), _half("validation", False),
                   lambda r: _mc(f"{r['context'].strip()}\n{r['question'].strip()}",
                                 [r["answerA"], r["answerB"], r["answerC"]], int(r["label"]) - 1)),
}

_TASKS: dict = {}


def get(name: str):
    """The task called `name` (NAMES), made once per process."""
    if name not in NAMES:
        raise ValueError(f"no Evo task {name!r}: one of {', '.join(NAMES)}")
    if name not in _TASKS:
        made = {"countdown": Countdown, "arithmetic": Arithmetic, "arc_easy": ArcEasy}.get(name)
        _TASKS[name] = made() if made else HubTask(name, *HUB[name])
    return _TASKS[name]
