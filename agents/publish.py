"""Publish a finished job to the results page: the plan, the training numbers and the judge's verdicts in one file.

    python agents/publish.py support-bot                     # -> web/public/results/support-bot.json
    python agents/publish.py support-bot --title "Support replies"
    python agents/publish.py car-noises-only-4bc0eb pirate-captain-two-sentences-0352d5   # several at once

Each run gets its own file, and runs.json lists every published run, best first, for the page's run picker.

It reads what the other steps left in agent-jobs/:
  <job>.plan.json                       the Job Architect's plan
  adapters/<job>/train_report.json      held-back loss and accuracy after every epoch (train_local.py)
  <job>.eval.answers.graded.json        the judge's verdicts (judge.py grade)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from anywhere: this checkout's agents
from agents.models import BASES  # noqa: E402

TITLES = {"qwen3-0.6b": "Qwen3 0.6B", "llama-3.2-1b": "Llama 3.2 1B"}
RESULTS = Path(__file__).parent.parent / "web" / "public" / "results"  # what the results page reads


def build(job: str, jobs: Path, title: str | None = None) -> dict:
    plan = json.loads((jobs / f"{job}.plan.json").read_text())
    report = json.loads((jobs / "adapters" / job / "train_report.json").read_text())
    graded = json.loads((jobs / f"{job}.eval.answers.graded.json").read_text())
    base = plan["base"]
    epochs = [{"epoch": h["epoch"], "loss": h["loss"], "acc": h["acc"]} for h in report["history"]]
    pairs = [{"prompt": p["prompt"], "base": p["base"], "tuned": p["tuned"], "tuned_won": p["verdict"]["tuned_won"],
              "winner": p["verdict"].get("winner"), "reason": p["verdict"].get("reason"),
              "flagged": p["verdict"].get("flagged", False) or p.get("vs_prompted", {}).get("flagged", False)}
             | ({"prompted": p["prompted"], "vs_prompted": p["vs_prompted"]["winner"],
                 "prompted_reason": p["vs_prompted"]["reason"]} if "vs_prompted" in p else {})
             | {k: p[k] for k in ("set", "level") if p.get(k)}
             | ({"second": p["second"]["winner"], "second_reason": p["second"]["reason"]} if "second" in p else {})
             | ({"teacher": p["teacher"], "vs_teacher": p["vs_teacher"]["winner"],
                 "teacher_reason": p["vs_teacher"]["reason"]} if "vs_teacher" in p else {})
             | {"readings": p["verdict"].get("readings")}
             | ({"fails": p["fails"]} if p.get("fails") else {})
             | ({"prompted_readings": p["vs_prompted"].get("readings")} if "vs_prompted" in p else {})
             for p in graded["pairs"]]
    return {"title": title or job.replace("-", " ").capitalize(), "job": job, "goal": plan["goal"],
            "base_title": TITLES.get(base, base), "base_repo": BASES[base],
            "date": datetime.date.today().isoformat(),
            "source": (f"{report['training_chats']} chats written by the Job Architect, {report['heldout_chats']} held "
                       f"back, trained on {report['device']} in {report['minutes']} minutes"),
            "settings": (f"LoRA rank {plan['rank']} on every linear layer · {report['trainable_weights']:,} trained "
                         f"weights · {report['epochs']} epochs"),
            "reasoning": plan["reasoning"], "caveat": plan.get("caveat") or None,
            "agent_model": plan.get("model"), "judge_model": graded.get("model"),
            "audit": plan.get("audit"), "checks": plan.get("checks") or [],
            # what a reader needs to check this run is the one trained, and what training it took
            "receipt": {k: report[k] for k in ("trainable_weights", "frozen_weights", "rank", "lora_alpha", "lr",
                                               "peak_vram_mb", "device", "minutes", "dataset_sha256", "adapter_sha256")
                        if report.get(k) is not None},
            "base_loss": epochs[0]["loss"], "epochs": epochs, "kept_epoch": report.get("kept_epoch", epochs[-1]["epoch"]), "judge": {**graded["summary"], "pairs": pairs}}


def score(run: dict) -> tuple[int, int, str]:
    """The run's headline verdict: against the base model told the goal when it was judged that way, else untrained."""
    j = run["judge"]
    if j.get("vs_prompted"):
        return j["vs_prompted"]["tuned_wins"], j["prompts"], "told the goal"
    return j["tuned_wins"], j["prompts"], "untrained"


def save(run: dict, results: Path = RESULTS) -> Path:
    """Write the run as results/<job>.json and rebuild results/runs.json from every run there, best first."""
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"{run['job']}.json"
    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in results.glob("*.json") if f.name != "runs.json"]
    index = []
    for r in runs:
        wins, prompts, against = score(r)
        index.append({"job": r["job"], "title": r["title"], "base_title": r["base_title"], "date": r["date"],
                      "wins": wins, "prompts": prompts, "against": against})
    # the honest comparison first, then the share of prompts won, then the newest
    index.sort(key=lambda r: (r["against"] == "told the goal", r["wins"] / r["prompts"], r["date"]), reverse=True)
    (results / "runs.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description="Write a finished job's results for the results page.")
    p.add_argument("jobs_", nargs="+", metavar="job")
    p.add_argument("--title", help="the run's name on the page (default: from the job id; one job only)")
    p.add_argument("--jobs", type=Path, default=Path("agent-jobs"))
    p.add_argument("--results", type=Path, default=RESULTS)
    a = p.parse_args()
    if a.title and len(a.jobs_) > 1:
        p.error("--title names one run; publish the others separately")
    for job in a.jobs_:
        run = build(job, a.jobs, a.title)
        out = save(run, a.results)
        wins, prompts, against = score(run)
        print(f"wrote {out}: held-back loss {run['base_loss']} -> {run['epochs'][run['kept_epoch']]['loss']}, "
              f"fine-tune preferred over the {'base model ' if against == 'told the goal' else ''}{against}"
              f"{'' if against == 'told the goal' else ' model'} on {wins} of {prompts}")


if __name__ == "__main__":
    main()
