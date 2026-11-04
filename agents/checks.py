"""Rules a reply either meets or doesn't, scored by code: the part of a goal no judge has to weigh.

The Job Architect picks a goal's rules from this list (never writes its own: a rule it invented would be one more
thing to trust). Every training chat has to pass them before any GPU time is spent, every judged reply is scored on
them, and a Standard or Strict request isn't published until the fine-tune passes them on most of its test prompts.

    >>> passes({"check": "exact_sentences", "arg": 2}, "Yes. No.")
    True
"""
import json
import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+", re.M)
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|```)", re.M)


def _json(reply: str):
    try:
        value = json.loads(reply.strip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _sentences(reply: str) -> int:
    return len([s for s in _SENTENCE.split(reply.strip()) if s.strip()])


def _lines(reply: str) -> list[str]:
    return [ln for ln in reply.strip().splitlines() if ln.strip()]


def _balanced(reply: str) -> bool:
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for ch in reply:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


# name: (what the argument is, or None; the test). Arguments are checked by `valid` before a plan is accepted.
CHECKS = {
    "json": (None, lambda r, a: _json(r) is not None and bool(_json(r))
             and all(v not in ("", None, [], {}) for v in _json(r).values())),
    "json_keys": ("list of the exact keys", lambda r, a: _json(r) is not None and set(_json(r)) == set(a)),
    "json_enum": ("list: a key, then every value it may take",
                  lambda r, a: _json(r) is not None and a[0] in _json(r) and _json(r)[a[0]] in a[1:]),
    "one_line": (None, lambda r, a: len(_lines(r)) == 1),
    "starts_with": ("text", lambda r, a: r.strip().startswith(a)),
    "ends_with": ("text", lambda r, a: r.strip().endswith(a)),
    "exact_sentences": ("number", lambda r, a: _sentences(r) == a),
    "max_sentences": ("number", lambda r, a: _sentences(r) <= a),
    "max_words": ("number", lambda r, a: len(r.split()) <= a),
    "min_words": ("number", lambda r, a: len(r.split()) >= a),
    "no_bullets": (None, lambda r, a: not _BULLET.search(r)),
    "no_markdown": (None, lambda r, a: not _MARKDOWN.search(r)),
    "balanced_brackets": (None, lambda r, a: _balanced(r)),
    "lines": ("number of non-empty lines", lambda r, a: len(_lines(r)) == a),
    "line_prefixes": ("list: what each line starts with, in order",
                      lambda r, a: len(_lines(r)) == len(a) and all(ln.startswith(p) for ln, p in zip(_lines(r), a))),
    "contains": ("text that must appear (case doesn't matter)", lambda r, a: a.lower() in r.lower()),
    "not_contains": ("text that must not appear (case doesn't matter)", lambda r, a: a.lower() not in r.lower()),
}
MAX_RULES = 5

MENU = "\n".join(f"  {name}" + (f" (arg: {arg})" if arg else "") for name, (arg, _) in CHECKS.items())


def valid(rules) -> list[str]:
    """What's wrong with a plan's rules: a name not in the list, a missing or wrong-typed argument, too many."""
    if rules is None:
        return []
    if not isinstance(rules, list):
        return ["checks must be a list"]
    if len(rules) > MAX_RULES:
        return [f"{len(rules)} checks; pick the {MAX_RULES} that matter most"]
    problems = []
    for rule in rules:
        name = rule.get("check") if isinstance(rule, dict) else None
        if name not in CHECKS:
            problems.append(f"check {name!r} isn't one of: {', '.join(CHECKS)}")
            continue
        kind, arg = CHECKS[name][0], rule.get("arg")
        ok = (arg is None if kind is None else
              isinstance(arg, int) and not isinstance(arg, bool) and 0 < arg < 10000 if kind.startswith("number") else
              isinstance(arg, list) and arg and all(isinstance(x, str) and x for x in arg) if kind.startswith("list") else
              isinstance(arg, str) and 0 < len(arg) <= 200)
        if not ok:
            problems.append(f"check {name!r} needs {kind or 'no'} argument, not {arg!r}")
    return problems


def passes(rule: dict, reply) -> bool:
    try:
        return bool(CHECKS[rule["check"]][1](str(reply or ""), rule.get("arg")))
    except Exception:  # a rule can't crash a run; a reply it can't be applied to fails it
        return False


def failed(rules: list, reply, exempt=None) -> list[str]:
    """The rules a reply breaks, named, for the repair loop and the results page. A reply `exempt` accepts (a narrow
    goal's decline sentence, which no formatting rule is meant for) breaks none."""
    if exempt and exempt(reply):
        return []
    return [name(r) for r in rules if not passes(r, reply)]


def name(rule: dict) -> str:
    return rule["check"] + (f" {json.dumps(rule['arg'])}" if rule.get("arg") is not None else "")


def score(rules: list, pairs: list[dict], sides=("tuned", "prompted", "base"), exempt=None) -> dict | None:
    """Each rule's pass rate per model over the pairs, and the share of replies passing every rule."""
    if not rules or not pairs:
        return None
    ok = lambda reply: bool(exempt and exempt(reply))
    rate = lambda test, side: round(sum(test(p.get(side)) for p in pairs) / len(pairs), 3)
    return {"prompts": len(pairs),
            "rules": [{"rule": name(r)} | {s: rate(lambda reply, r=r: ok(reply) or passes(r, reply), s) for s in sides}
                      for r in rules],
            "all": {s: rate(lambda reply: not failed(rules, reply, exempt), s) for s in sides}}
