"""Instruction following for Evo Hash (es_tasks' `ifeval`): each prompt carries one constraint a reply is checked
against with the standard library alone. The prompts and constraints are allenai/RLVR-IFeval's (Tülu 3's RL data: a
chat prompt, and a ground truth {"func_name": ..., its arguments}); the reply as a whole is the answer.

The checks are ported from two Apache-2.0 sources: open-instruct's open_instruct/if_functions.py (allenai/open-instruct,
the reward RLVR-IFeval was made for) and google-research's instruction_following_eval (the IFEval benchmark's own
checker, also IFEvalG's base). Where they differ this follows the benchmark, which is what the job is measured on, or
the prompt's own words where those are more exact (they say "should appear N times", the benchmark "at least"). Each
difference is noted at its check. Left out:
* the response-language constraint (langdetect), and the benchmark's "and in English" on all-caps/all-lowercase replies
  (langdetect too): RLVR-IFeval has no language rows, and casing is checked without it;
* sentence counts by nltk's punkt (the benchmark's): open-instruct's regex split is used instead;
* rows a reply of a few hundred tokens can't satisfy (LIMITS: "39 paragraphs" with max_tokens 512), and rows whose
  arguments don't parse (`constraint` returns None for them).

Every check is pure, deterministic and never raises (anything unexpected is a wrong answer): the node re-runs it.
"""
from __future__ import annotations

import json
import re
from typing import Callable

# The most a constraint may ask for (its N) and still fit in a reply of ~400 tokens; "at most" counts always fit.
LIMITS = {"verify_paragraph_count": 8, "validate_paragraphs": 8, "validate_sections": 8, "verify_bullet_points": 15,
          "validate_highlighted_sections": 15, "validate_placeholders": 15, "verify_keyword_frequency": 10,
          "verify_sentence_constraint": 20}
REPEAT = ("First, repeat the request without change, then give your answer (do not say anything before repeating the "
          "request; the request you need to repeat does not include this sentence)")


def _quantified(count: int, n: int, quantifier: str, around: int) -> bool:
    if quantifier == "at least":
        return count >= n
    if quantifier == "at most":
        return count <= n
    if quantifier == "around":
        return abs(count - n) <= around
    return False


def _tolerance(n: int) -> int:
    """open-instruct's ±10% for "around" (max(round(N * 0.1), 1)), in integers: N // 10, at least 1."""
    return max(1, n // 10)


def keywords(t, a):  # "Include keywords X, Y in your response" (both sources: case-insensitive substring)
    low = t.lower()
    return all(str(k).lower() in low for k in a["keyword_list"])


def keyword_frequency(t, a):  # "the word X should appear N times": exactly N (open-instruct, the prompt's words)
    return re.findall(r"\b\w+\b", t.lower()).count(str(a["word"]).lower()) == a["N"]


def forbidden_words(t, a):  # "Do not include keywords X, Y": as whole words (the benchmark; open-instruct: substrings)
    return not any(re.search(r"\b" + re.escape(str(w)) + r"\b", t, re.I) for w in a["forbidden_words"])


def letter_frequency(t, a):  # "the letter x should appear N times": exactly N (the prompt), either case (the benchmark)
    return len(a["letter"]) == 1 and t.lower().count(a["letter"].lower()) == a["N"]


def paragraph_count(t, a):  # N paragraphs split by "* * *" (the benchmark: empty ones only at the ends)
    parts = re.split(r"\s?\*\s?\*\s?\*\s?", t)
    n = len(parts)
    for i, p in enumerate(parts):
        if not p.strip():
            if i in (0, len(parts) - 1):
                n -= 1
            else:
                return False
    return n == a["N"]


def word_constraint(t, a):  # at least / around / at most N words, counted as the benchmark does (runs of \w)
    return _quantified(len(re.findall(r"\w+", t)), a["N"], a["quantifier"], _tolerance(a["N"]))


def sentence_constraint(t, a):  # at least / around (±1) / at most N sentences, open-instruct's split (no nltk)
    parts = [s for s in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", t.strip()) if s.strip()]
    return _quantified(len(parts), a["N"], a["quantifier"], 1)


def paragraphs_first_word(t, a):  # N paragraphs split by blank lines, the i-th starting with a word (the benchmark)
    parts = re.split(r"\n\n", t)
    n = len(parts) - sum(1 for p in parts if not p.strip())
    i = a["i"]
    if not 1 <= i <= n:
        return False
    p = parts[i - 1].strip()
    if not p:
        return False
    word = p.split()[0].strip().lstrip("'").lstrip('"')
    first = ""
    for ch in word:
        if ch in ".,":
            break
        first += ch.lower()
    return n == a["N"] and first == str(a["first_word"]).lower()


def postscript(t, a):  # a postscript starting with the marker: its last use, with words after it (open-instruct)
    marker = str(a["postscript_marker"]).lower()
    at = t.lower().rfind(marker)
    return at >= 0 and bool(t[at + len(marker):].strip())


# The node scores every claim's texts with these while it holds its lock, and a text is whatever a miner writes, so
# each check takes time in proportion to the text: the benchmark's regexes `\[.*?\]`, `^\s*\*...` (re.M, \s spanning
# lines) and `<<[^\n]+>>` take time in its square on a run of "[", "\n" or "<" (8 MB of "<": hours, 2026-11-07).
# These count exactly what they do (tests/test_evo_hostile.py compares them on random texts).

def placeholders(t, a):  # at least N [placeholders] (both sources): \[.*?\] matches, "." not crossing a line
    n = 0
    for line in t.split("\n"):
        at = 0
        while (i := line.find("[", at)) >= 0 and (j := line.find("]", i + 1)) >= 0:
            n, at = n + 1, j + 1
    return n >= a["N"]


def bullet_points(t, a):  # exactly N markdown bullets, "* " or "- " lines (the benchmark: "**bold**" lines aren't)
    # (a match's leading \s* may start on blank lines above its bullet, so whitespace short of a newline counts the same)
    return len(re.findall(r"^[^\S\n]*\*[^\*].*$", t, re.M)) + len(re.findall(r"^[^\S\n]*-.*$", t, re.M)) == a["N"]


def title(t, a):  # a title in <<double angular brackets>>, not empty (the benchmark): <<[^\n]+>> matches, greedy
    for line in t.split("\n"):  # a line's one match: its first "<<" to its last ">>", with something between
        i, j = line.find("<<"), line.rfind(">>")
        if i >= 0 and j >= i + 3 and line[i:j + 2].lstrip("<").rstrip(">").strip():
            return True
    return False


def _options(s) -> list[str]:
    """RLVR-IFeval's options, a string: "yes/no/maybe", "I know or I don't know", "a), b), c), d)"."""
    s = str(s).strip()
    for sep in (", ", " or ", "/"):
        if sep in s:
            return [o.strip() for o in s.split(sep) if o.strip()]
    return [s] if s else []


def choice(t, a):
    """The reply opens with one of the options (then a non-word character or the end), either case. Stricter than
    both sources, which look for an option anywhere (and open-instruct's iterates the options string's characters, so
    any reply with an "a" passes): an option must be the answer, not a word inside it."""
    reply = t.strip().lstrip("\"'*").lower()
    for o in _options(a["options"]):
        o = o.lower()
        if reply.startswith(o) and (len(reply) == len(o) or not (reply[len(o)].isalnum() or reply[len(o)] == "_")):
            return True
    return False


def highlighted(t, a):  # at least N *highlighted* (or **bold**) sections, not empty (the benchmark)
    n = sum(1 for m in re.findall(r"\*[^\n\*]*\*", t) if m.strip("*").strip())
    n += sum(1 for m in re.findall(r"\*\*[^\n\*]*\*\*", t) if m.removeprefix("**").removesuffix("**").strip())
    return n >= a["N"]


def sections(t, a):  # N sections, each opened by the splitter and a number ("Section: 1"); the benchmark's split, and
    # exactly N as the prompt says (the benchmark: at least N)
    return len(re.split(r"\s?" + re.escape(str(a["section_splitter"])) + r"\s?\d+\s?", t)) - 1 == a["N"]


def json_format(t, a):  # the whole reply is JSON, a ``` fence allowed around it (the benchmark)
    s = t.strip()
    for p in ("```json", "```Json", "```JSON", "```"):
        s = s.removeprefix(p)
    s = s.removesuffix("```").strip()
    try:
        json.loads(s)
    except (ValueError, RecursionError):
        return False
    return True


def repeat_prompt(t, a):
    """The reply starts with the request, either case (the benchmark), runs of white space read as one space. The
    ground truth's original_prompt is the whole prompt, the instruction to repeat included, which the instruction
    itself says not to repeat: it's taken out."""
    return " ".join(t.split()).lower().startswith(a["request"].lower())


def two_responses(t, a):  # two different replies split by ******, empty parts only at the ends (the benchmark)
    kept = []
    parts = t.split("******")
    for i, p in enumerate(parts):
        if not p.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            kept.append(p)
    return len(kept) == 2 and kept[0].strip() != kept[1].strip()


def uppercase(t, a):  # capitals only, with at least one letter (the benchmark's isupper, without its langdetect)
    return t.isupper()


def lowercase(t, a):  # lower case only, with at least one letter (likewise)
    return t.islower()


def capital_words(t, a):  # at least / around / at most N all-capital words (open-instruct's regex; the benchmark: nltk)
    return _quantified(len(re.findall(r"\b[A-Z]+\b", t)), a["N"], a["quantifier"], _tolerance(a["N"]))


def end_phrase(t, a):  # ends with the phrase, either case, trailing space aside (the benchmark)
    return t.strip().lower().endswith(str(a["end_phrase"]).strip().lower())


def quotation(t, a):  # wrapped in double quotes (the benchmark)
    s = t.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


def no_commas(t, a):
    return "," not in t


# RLVR-IFeval's func_name: (its check, the arguments it takes)
CHECKS: dict[str, tuple[Callable[[str, dict], bool], tuple]] = {
    "verify_keywords": (keywords, ("keyword_list",)),
    "verify_keyword_frequency": (keyword_frequency, ("word", "N")),
    "validate_forbidden_words": (forbidden_words, ("forbidden_words",)),
    "verify_letter_frequency": (letter_frequency, ("letter", "N")),
    "verify_paragraph_count": (paragraph_count, ("N",)),
    "validate_word_constraint": (word_constraint, ("N", "quantifier")),
    "verify_sentence_constraint": (sentence_constraint, ("N", "quantifier")),
    "validate_paragraphs": (paragraphs_first_word, ("N", "first_word", "i")),
    "verify_postscript": (postscript, ("postscript_marker",)),
    "validate_placeholders": (placeholders, ("N",)),
    "verify_bullet_points": (bullet_points, ("N",)),
    "validate_title": (title, ()),
    "validate_choice": (choice, ("options",)),
    "validate_highlighted_sections": (highlighted, ("N",)),
    "validate_sections": (sections, ("N", "section_splitter")),
    "validate_json_format": (json_format, ()),
    "validate_repeat_prompt": (repeat_prompt, ("original_prompt",)),
    "validate_two_responses": (two_responses, ()),
    "validate_uppercase": (uppercase, ()),
    "validate_lowercase": (lowercase, ()),
    "validate_frequency_capital_words": (capital_words, ("N", "quantifier")),
    "validate_end": (end_phrase, ("end_phrase",)),
    "validate_quotation": (quotation, ()),
    "validate_no_commas": (no_commas, ()),
}


def _args(name: str, gt: dict) -> dict | None:
    """The check's arguments from a ground truth, typed and in range, or None to leave the row out."""
    a = {k: gt.get(k) for k in CHECKS[name][1]}
    if any(v is None for v in a.values()):
        return None
    for k in ("N", "i"):
        if k in a:
            if type(a[k]) is not int or a[k] < 0:
                return None
    if "N" in a and a.get("quantifier") != "at most" and a["N"] > LIMITS.get(name, a["N"]):
        return None
    if "quantifier" in a and a["quantifier"] not in ("at least", "around", "at most"):
        return None
    for k in ("keyword_list", "forbidden_words"):
        if k in a and not (isinstance(a[k], list) and a[k] and all(isinstance(w, str) and w.strip() for w in a[k])):
            return None
    for k in ("word", "letter", "first_word", "postscript_marker", "section_splitter", "end_phrase"):
        if k in a and not (isinstance(a[k], str) and a[k].strip()):
            return None
    if name == "validate_paragraphs" and not 1 <= a["i"] <= a["N"]:
        return None
    if name == "validate_choice" and len(_options(a["options"])) < 2:
        return None
    if name == "validate_repeat_prompt":
        if not isinstance(a["original_prompt"], str):
            return None
        a = {"request": " ".join(a.pop("original_prompt").replace(REPEAT, " ").split())}
        if not a["request"]:
            return None
    return a


def constraint(ground_truth) -> Callable[[str], bool] | None:
    """The check of a reply against one RLVR-IFeval row's ground truth (its JSON text, or the parsed dict), or None if
    the row is left out (a constraint this module doesn't check, arguments that don't parse, an N out of LIMITS)."""
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        name = gt["func_name"]
    except (ValueError, TypeError, KeyError):
        return None
    if name not in CHECKS:
        return None
    a = _args(name, gt)
    if a is None:
        return None
    fn = CHECKS[name][0]

    def ok(text: str) -> bool:
        try:
            return isinstance(text, str) and bool(fn(text, a))
        except Exception:  # noqa: BLE001 - a reply the check can't read is a wrong one, never an error on the node
            return False
    return ok
