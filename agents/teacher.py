"""The fine-tune against the model that trained it.

Muse Spark wrote the chats the fine-tune learned from. Here it answers the same test prompts itself, told the goal
(and the decline sentence) the way the base model told the goal is, and the same blind judge compares each pair
both ways. Code scores its replies on the rules and the declines too. The question it answers: how much of a
frontier model's skill at this one task fits in a 0.6B model that runs offline, with no goal in the prompt.

This is a measurement beside the bar, never part of it: a run can't pass or fail on it, and the repair loop never
sees it.

    python agents/teacher.py agent-jobs/<job>.eval.answers.graded.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from anywhere: this checkout's agents
from agents import checks, judge  # noqa: E402
from agents.llm_call import chat, model_name  # noqa: E402

TEACHER = "teacher"
WORKERS = 6


def reply(told: str, prompt: str) -> str:
    """Muse Spark's own answer, told the goal in its system prompt exactly as the base model told the goal is."""
    text = chat([{"role": "system", "content": told}, {"role": "user", "content": prompt}])
    return text.split("</think>")[-1].strip()


def tally(pairs: list[dict]) -> dict:
    """Wins, ties and losses against the teacher over some pairs, with the same interval and luck as the other scores."""
    n = len(pairs)
    wins = sum(p["vs_teacher"]["winner"] == "tuned" for p in pairs)
    ties = sum(p["vs_teacher"]["winner"] == "tie" for p in pairs)
    neither = sum(p["vs_teacher"]["winner"] == "neither" for p in pairs)  # no reply did the goal: not a tie
    return {"prompts": n, "tuned_wins": wins, "ties": ties, "neither": neither, "teacher_wins": n - wins - ties - neither,
            "interval": judge.interval(wins, n), "chance": judge.chance(wins, n - wins - ties),
            "held": round((wins + ties) / n, 3) if n else None}  # the share the fine-tune matched or beat it on


def measure(graded_file: Path, report=print) -> dict:
    """Add the teacher's replies and the judge's verdicts on them to a graded answers file, and a summary under
    summary["teacher"]. Returns the summary."""
    data = json.loads(graded_file.read_text(encoding="utf-8"))
    decline = (data.get("scope") or {}).get("decline")
    told = judge.told(data["goal"], decline)
    goal = data["goal"] + (f"\n(This model does only that task. To a message that isn't such input it should reply, "
                           f"in effect: {decline})" if decline else "")
    pairs = data["pairs"]
    report(f"{model_name()} is answering the {len(pairs)} test prompts itself, told the goal")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for p, r in zip(pairs, pool.map(lambda p: reply(told, p["prompt"]), pairs)):
            p[TEACHER] = r
    report(f"the judge is comparing the fine-tune with {model_name()}, blind, twice each")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for p, v in zip(pairs, pool.map(lambda p: judge.verdict(goal, p, TEACHER), pairs)):
            p["vs_teacher"] = v
            report(f"{v['winner']:>7}  {v['reason']}")

    out = {"model": model_name(), **tally([p for p in pairs if not p.get("set")])}
    for name in [*judge.SETS, judge.FRESH, judge.OFF]:
        if some := [p for p in pairs if p.get("set") == name]:
            out[name] = tally(some)
    on_task = [p for p in pairs if p.get("set") != judge.OFF]
    exempt = (lambda r: judge.norm(decline) in judge.norm(r or "")) if decline else None
    if scored := checks.score(data.get("checks") or [], on_task, sides=("tuned", TEACHER), exempt=exempt):
        out["checks"] = scored
    if decline and (off := [p for p in pairs if p.get("set") == judge.OFF]):
        rate = lambda side: round(sum(judge.norm(decline) in judge.norm(p.get(side) or "") for p in off) / len(off), 3)
        out["scope"] = {"prompts": len(off), "tuned": rate("tuned"), TEACHER: rate(TEACHER)}
    data["summary"][TEACHER] = out
    graded_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    report(f"against {model_name()} told the goal: fine-tune preferred on {out['tuned_wins']} of {out['prompts']}, "
           f"{out['ties']} ties, {out['teacher_wins']} losses")
    return out


if __name__ == "__main__":
    measure(Path(sys.argv[1]))
