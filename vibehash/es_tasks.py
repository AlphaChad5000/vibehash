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

* four for a job with room to think (max_tokens about 384-512), added for the Qwen3.5-0.8B job (VIBE):
  - ifeval: instruction following, allenai/RLVR-IFeval's prompts, each with one constraint checked by es_ifeval (no
    <answer> tags: the whole reply is the answer);
  - gsm8k: grade-school word problems (openai/gsm8k's train split), reasoned, the final number in the tags;
  - math: competition problems (Hendrycks MATH's train split, every subject), reasoned, the final answer in the tags
    (or \\boxed{}), compared after `math_norm`;
  - medmcqa: medical-exam multiple choice, four options (openlifescienceai/medmcqa).

NAMES is every task a job may name (the first nineteen in the order Evo Hash first trained them; ARC-Challenge, PIQA
and Social IQa came later, then the four above). CURATED is what Evo Hash trains on now: the tasks a small model can
learn from, measured on Qwen2.5-0.5B-Instruct (2026-11-05, 200 held-out items each): trivia_qa (0.5% right), nq_open
(0%) and web_questions (0.5%) ask for facts the model doesn't hold, which reward-driven training can't add; countdown
and arithmetic are puzzles no one benchmarks. VIBE is the Qwen3.5-0.8B job's mix (at its definition, why).

Kept out of every task, so a tuned model can be scored on them as never seen: GSM8K's test split (the gsm8k task trains
on its train split and holds out a share of that), MMLU, MMLU-Pro, MMLU-Redux, GPQA, SuperGPQA, C-Eval, MMMLU, google's
IFEval prompts and MATH's test split (MATH-500 is drawn from it).

`lenient(item)`: the same problem, its check also taking a right answer given without the <answer> tags ("B", "(B)",
"The answer is B", the last number, a leading yes or no). A job whose spec says `lenient` (es2.Spec2) is scored that
way, so its reward is for being right, not for writing the tags: the base model leaves them out of a quarter to a half
of its replies on some tasks (boolq 49.5% in format), and a strict check pays a model for learning the tags, which
reads as a gain that standard benchmarks don't see. An item whose answer isn't read that way carries its own lenient
check (`Item.loose`): a reasoned answer in \\boxed{}, or an ifeval reply (checked the same either way).
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
         "boolq", "arc_challenge", "piqa", "social_iqa", "ifeval", "gsm8k", "math", "medmcqa")
# What Evo Hash trains on (see the docstring): the knowledge benchmarks first, then commonsense, math, classification.
CURATED = ("arc_easy", "sciq", "commonsense_qa", "openbookqa", "arc_challenge", "piqa", "social_iqa", "qasc",
           "winogrande", "hellaswag", "boolq", "aqua_rat", "svamp", "asdiv", "sst2", "ag_news", "emotion")
# The mix for the job on Qwen/Qwen3.5-0.8B (non-thinking), aiming past its published MMLU-Pro, MMLU-Redux, SuperGPQA,
# IFEval, C-Eval and MMMLU, at max_tokens 384-512. Five of the six are knowledge multiple choice, so most of the mix is:
# medmcqa (the hardest, MMLU-Pro-like knowledge), the science sets (arc_challenge, arc_easy, sciq, openbookqa, qasc)
# and commonsense MC (commonsense_qa, social_iqa, piqa, winogrande, hellaswag, boolq), which earlier runs found teach a
# small model to say what it knows (MMLU 25% to 37.5% untrained, HANDOFF). ifeval is IFEval's own skill; gsm8k, math
# and aqua_rat train the step-by-step reasoning MMLU-Pro and SuperGPQA reward (both are scored with reasoning). Left
# out: sst2, ag_news and emotion (classification no listed benchmark asks for), svamp and asdiv (easy word problems
# gsm8k covers), and the five CURATED already drops. Sixteen tasks: with 16 problems a round, every round has one of
# each (es2_rounds.items).
VIBE = ("ifeval", "gsm8k", "math", "medmcqa", "arc_challenge", "arc_easy", "sciq", "openbookqa", "commonsense_qa",
        "qasc", "social_iqa", "piqa", "winogrande", "hellaswag", "boolq", "aqua_rat")
VIBE_MAX_TOKENS = 512  # VIBE's jobs' longest answer: gsm8k, math and ifeval reason or write at length
VIBE_LOOP_STOP = (32, 48)  # VIBE's jobs' es2 loop_stop: 279 of 1,832 answers at 512 tokens were loops (es2.Spec2)
MAX_LONG = 520  # a Hugging Face task's longest prompt, in characters (about 150 tokens)
MAX_REASON = 1000  # the longest prompt of the tasks with room to think (ifeval, gsm8k, math), about 300 tokens
POOL = 20_000  # the most rows a task keeps the indices of for each part


class TaskUnavailable(RuntimeError):
    """The task needs something this computer doesn't have (ARC: the `datasets` library or the download)."""


def last_answer(text: str):
    found = es.tagged(text)  # (_ANS.findall's answers, in linear time: es.tagged)
    return found[-1].strip() if found else None


_LETTER_ALONE = re.compile(r"^\(?([A-H])\)?(?:[.):]|$)")  # "B", "B.", "B)", "(B)", "B: ..." (not "A dog ...")
# "the answer is B", "Answer: (C)". (Written \s*(?:[:\-]\s*)? and not \s*[:\-]?\s*, which matches the same text but
# tries every split of a run of spaces between its two \s*: "answer", 40,000 spaces and a letter took 22 s.)
_LETTER_SAID = re.compile(r"\banswer(?:\s+is)?\s*(?:[:\-]\s*)?\(?([A-H])\b", re.I)
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
    before. An item with its own lenient check (`loose`) takes that instead."""
    if item.loose is not None:
        return Item(item.messages, item.loose, item.gold)
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
    gold: str = ""  # one right answer's text between the tags ("" for ifeval: no one answer)
    # The check `lenient` uses when `salvage` can't read the answer (a reasoned one in \boxed{}, a whole reply); None:
    # salvage's, as every task before these.
    loose: Callable[[str], bool] | None = field(default=None, hash=False, compare=False)


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


# --- the tasks with room to think (VIBE): ifeval, gsm8k, math, and medmcqa's multiple choice ---

_ASK_REASON_NUMBER = "Reason step by step, then give the final number between <answer> and </answer>."
_ASK_REASON_MATH = ("Reason step by step, then give the final answer, in simplest form, between <answer> and "
                    "</answer>.")


def last_boxed(text: str) -> str | None:
    """What the last \\boxed{...} (or \\fbox{...}) in `text` holds, braces matched; "\\boxed 5" gives "5". None if
    there's none, or its braces don't close."""
    at = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if at < 0:
        return None
    rest = text[at + (6 if text.startswith("\\boxed", at) else 5):]
    if rest.startswith(" "):
        found = rest.strip().split("$")[0].split()
        return found[0] if found else None
    if not rest.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(rest):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return rest[1:i]
    return None


def _last_number(text: str) -> str | None:
    nums = _NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _same_number(a: str | None, gold: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(gold)) < 1e-6
    except ValueError:
        return False


def _reasoned_number_check(gold: str, loose: bool = False):
    """A reasoned reply's final number: the last number between its last <answer> tags ("<answer>$18.00</answer>",
    "<answer>18 apples</answer>"). Loose (es_tasks.lenient), without tags: the last \\boxed{}'s, else the reply's last
    number."""
    def ok(text: str) -> bool:
        a = last_answer(text)
        if a is None:
            if not loose:
                return False
            boxed = last_boxed(text)
            a = text if boxed is None else boxed
        return _same_number(_last_number(a), gold)
    return ok


def _gsm8k(row):
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    if "####" not in row["answer"] or not re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
        return None
    m = _fits(row["question"].strip() + "\n" + _ASK_REASON_NUMBER, MAX_REASON)
    return Item(m, _reasoned_number_check(gold), gold, _reasoned_number_check(gold, True)) if m else None


# MATH's answer normalisation, ported from Hendrycks et al.'s MATH code (github.com/hendrycks/math, MIT: is_equiv's
# strip_string and its helpers), with a few more of the usual steps: \text{} and \mbox{} unwrapped, thin spaces,
# thousands commas and a trailing ".0" dropped.

def _fix_fracs(s: str) -> str:  # \frac12 -> \frac{1}{2}, \frac1{72} -> \frac{1}{72}
    parts = s.split("\\frac")
    out = parts[0]
    for sub in parts[1:]:
        out += "\\frac"
        if sub[:1] == "{":
            out += sub
        elif len(sub) < 2:
            return s
        elif sub[1] != "{":
            out += "{" + sub[0] + "}{" + sub[1] + "}" + sub[2:]
        else:
            out += "{" + sub[0] + "}" + sub[1:]
    return out


def _fix_a_slash_b(s: str) -> str:  # 3/4 -> \frac{3}{4}, whole numbers only
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return "\\frac{" + m.group(1) + "}{" + m.group(2) + "}" if m else s


def _fix_sqrt(s: str) -> str:  # \sqrt3 -> \sqrt{3}
    parts = s.split("\\sqrt")
    out = parts[0]
    for sub in parts[1:]:
        out += "\\sqrt" + (sub if sub[:1] in ("{", "") else "{" + sub[0] + "}" + sub[1:])
    return out


def math_norm(s: str) -> str:
    """A MATH answer in a standard form, so two ways of writing it compare equal as strings."""
    s = s.strip().replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "").replace("$", "")
    for space in ("\\,", "\\;", "\\:"):
        s = s.replace(space, "")
    if "\\text{ " in s:  # a unit after the answer: "5\text{ cm}"
        s = s.split("\\text{ ")[0]
    s = re.sub(r"\\(?:text|textbf|mbox|mathrm)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:  # "x = 5", "k=5"
        s = s.split("=")[1]
    s = _fix_sqrt(s).replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    s = re.sub(r"^(-?\d+)\.0+$", r"\1", s)
    return s.rstrip(".")


def _math_clean(norm: str) -> bool:
    """Whether a normalised gold answer is one a string match can judge: short, braces balanced, no text or layout."""
    depth = 0
    for ch in norm:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return (0 < len(norm) <= 40 and depth == 0 and "\\text" not in norm and "\\begin" not in norm
            and "\\mbox" not in norm and "\\quad" not in norm and "\\boxed" not in norm)


def _math_equal(a: str, gold_norm: str) -> bool:
    if a == gold_norm:
        return True
    try:
        return abs(float(a) - float(gold_norm)) < 1e-6
    except ValueError:
        return False


def _math_check(gold_norm: str, loose: bool = False):
    """The answer between the last <answer> tags (a \\boxed{} inside them is unwrapped), normalised (math_norm) and
    compared with the gold's. Loose (es_tasks.lenient), without tags: the last \\boxed{}'s, else for a numeric gold the
    reply's last number."""
    def ok(text: str) -> bool:
        a = last_answer(text)
        if a is not None:
            boxed = last_boxed(a)
            a = a if boxed is None else boxed
        elif loose:
            a = last_boxed(text)
            if a is None and re.fullmatch(r"-?\d+(?:\.\d+)?", gold_norm):
                a = _last_number(text)
        return a is not None and _math_equal(math_norm(a), gold_norm)
    return ok


def _math(row):
    gold = last_boxed(row["solution"])
    if gold is None or not _math_clean(math_norm(gold)):
        return None
    norm = math_norm(gold)
    m = _fits(row["problem"].strip() + "\n" + _ASK_REASON_MATH, MAX_REASON)
    return Item(m, _math_check(norm), gold.strip(), _math_check(norm, True)) if m else None


def _ifeval(row):
    from .es_ifeval import constraint
    msgs = row["messages"]
    if len(msgs) != 1 or msgs[0]["role"] != "user":
        return None
    check = constraint(row["ground_truth"])
    m = _fits(msgs[0]["content"].strip(), MAX_REASON) if check else None
    return Item(m, check, "", check) if m else None


def _medmcqa(row):
    opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
    if not (isinstance(row["question"], str) and row["question"].strip()
            and all(isinstance(o, str) and o.strip() for o in opts) and isinstance(row["cop"], int)):
        return None
    return _mc(row["question"], opts, row["cop"])


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


MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")

# name: (repo, config, revision, train, test, val, make). Each entry's license is its Hugging Face card's (2026-11-05;
# "unclear" where the card names none or says unknown). ARC-Easy (ArcEasy above, allenai/ai2_arc) is CC BY-SA 4.0.
HUB = {
    # CC BY-NC 3.0: non-commercial
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815", "train", "test", "validation", _sciq),
    # MIT
    "svamp": ("ChilleD/SVAMP", None, "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef", "train", "test", None,
              lambda r: _number(_story(r["Body"], r["Question"]), r["Answer"])),
    # unclear: "unknown" (the University of Washington doesn't own the questions' copyright, the card says)
    "trivia_qa": ("mandarjoshi/trivia_qa", "rc.nocontext", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f", "train",
                  _half("validation", True), _half("validation", False),
                  lambda r: _short(r["question"], [r["answer"]["value"], *r["answer"]["aliases"]])),
    # MIT
    "commonsense_qa": ("tau/commonsense_qa", None, "94630fe30dad47192a8546eb75f094926d47e155", "train",
                       _half("validation", True), _half("validation", False), lambda r: _labelled(r, r["question"])),
    # unclear: "other" (GLUE points to each source dataset's own license)
    "sst2": ("nyu-mll/glue", "sst2", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "train", _half("validation", True),
             _half("validation", False),
             lambda r: _mc(f'What is the sentiment of this movie review?\n"{r["sentence"].strip()}"',
                           ["negative", "positive"], r["label"])),
    # unclear: "unknown"
    "openbookqa": ("allenai/openbookqa", "main", "388097ea7776314e93a529163e0fea805b8a6454", "train", "test",
                   "validation", lambda r: _labelled(r, r["question_stem"])),
    # CC BY-NC 4.0: non-commercial
    "asdiv": ("EleutherAI/asdiv", None, "8f95807222d87b4c688c3c22a6ba2801e1fa03e2", ("validation", 0.0, 0.8),
              ("validation", 0.8, 0.9), ("validation", 0.9, 1.0), _asdiv),
    # CC BY-SA 3.0
    "nq_open": ("google-research-datasets/nq_open", None, "5dd9790a83002ad084ddeb7c420dc716852c6f28", "train",
                _half("validation", True), _half("validation", False), _nq),
    # unclear: the card names none
    "winogrande": ("allenai/winogrande", "winogrande_xl", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "train",
                   _half("validation", True), _half("validation", False), _winogrande),
    # unclear: "unknown"
    "ag_news": ("fancyzhx/ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4", "train", "test", None,
                lambda r: _mc("What is this news story about?\n" + r["text"].replace("\\", " "),
                              ["World", "Sports", "Business", "Science and technology"], r["label"])),
    # CC BY 4.0
    "qasc": ("allenai/qasc", None, "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "train", _half("validation", True),
             _half("validation", False), lambda r: _labelled(r, r["question"])),
    # Apache-2.0
    "aqua_rat": ("deepmind/aqua_rat", "raw", "33301c6a050c96af81f63cad5562cb5363e88971", "train", "test", "validation",
                 _aqua),
    # unclear: "unknown"
    "web_questions": ("stanfordnlp/web_questions", None, "0e473cbe21d1e91ec18da343644498be6a3f5454", "train", "test",
                      None, lambda r: _short(r["question"], list(r["answers"]))),
    # MIT (the card's licensing section)
    "hellaswag": ("Rowan/hellaswag", None, "218ec52e09a7e7462a5400043bb9a69a41d06b76", "train",
                  _half("validation", True), _half("validation", False), _hellaswag),
    # "other": "for educational and research purposes only", the card says (in effect non-commercial)
    "emotion": ("dair-ai/emotion", "split", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "train", "test", "validation",
                lambda r: _mc(f'Which emotion does this message show?\n"{r["text"].strip()}"',
                              ["sadness", "joy", "love", "anger", "fear", "surprise"], r["label"])),
    # CC BY-SA 3.0
    "boolq": ("google/boolq", None, "35b264d03638db9f4ce671b711558bf7ff0f80d5", "train", _half("validation", True),
              _half("validation", False), _boolq),
    # Added 2026-11-05 for the curated mix. PIQA and Social IQa are loading scripts upstream, which `datasets` no
    # longer runs: these are Hugging Face's parquet conversions, pinned at their commits. PIQA's test answers are
    # hidden, so its held-out items are half of validation, as Social IQa's (it has no test split).
    # CC BY-SA 4.0
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "210d026faf9955653af8916fad021475a3f00453", "train", "test",
                      "validation", lambda r: _labelled(r, r["question"])),
    # unclear: "unknown"
    "piqa": ("ybisk/piqa", {"data_dir": "plain_text"}, "142c51238b3ca2bc61e9a075913871b8b600e8e1", "train",
             _half("validation", True), _half("validation", False),
             lambda r: _mc(f"Which is the better way to do this?\n{r['goal']}", [r["sol1"], r["sol2"]],
                           int(r["label"]))),
    # CC BY 4.0 (the card's licensing section)
    "social_iqa": ("allenai/social_i_qa", None, "537a2ec8ec565adc0b70b70752893e59e024df26", "train",
                   _half("validation", True), _half("validation", False),
                   lambda r: _mc(f"{r['context'].strip()}\n{r['question'].strip()}",
                                 [r["answerA"], r["answerB"], r["answerC"]], int(r["label"]) - 1)),
    # Added 2026-11-05 for the Qwen3.5-0.8B job (VIBE). RLVR-IFeval and MATH have one split to train on, GSM8K's test
    # is the benchmark: each holds out the last tenth of its train split (in the fixed shuffled order), half for
    # eval_items and half for val_items. MATH reads only its train files, so its test split (MATH-500's source) is
    # never even loaded. MedMCQA's test answers are hidden: its held-out items are halves of validation.
    # ODC-BY (prompts from Tülu 3's SFT sources, each under its own terms); none of google/IFEval's 541 prompts is in
    # it (checked 2026-11-05 on each prompt's first 80 characters)
    "ifeval": ("allenai/RLVR-IFeval", None, "47c03c73621c4aab2b824b7818681117d662770e", ("train", 0.0, 0.9),
               ("train", 0.9, 0.95), ("train", 0.95, 1.0), _ifeval),
    # MIT
    "gsm8k": ("openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", ("train", 0.0, 0.9),
              ("train", 0.9, 0.95), ("train", 0.95, 1.0), _gsm8k),
    # MIT
    "math": ("EleutherAI/hendrycks_math",
             {"data_files": {"train": [f"{s}/train-00000-of-00001.parquet" for s in MATH_SUBJECTS]}},
             "21a5633873b6a120296cce3e2df9d5550074f4a3", ("train", 0.0, 0.9), ("train", 0.9, 0.95),
             ("train", 0.95, 1.0), _math),
    # Apache-2.0
    "medmcqa": ("openlifescienceai/medmcqa", None, "91c6572c454088bf71b679ad90aa8dffcd0d5868", "train",
                _half("validation", True), _half("validation", False), _medmcqa),
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
