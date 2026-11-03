"""Publish a finished job to the results page: the plan, the training numbers and the judge's verdicts in one file.

    python agents/publish.py support-bot                     # -> web/public/results/run.json
    python agents/publish.py support-bot --title "Support replies"

It reads what the other steps left in agent-jobs/:
  <job>.plan.json                       the Job Architect's plan
  adapters/<job>/train_report.json      held-back loss and accuracy after every epoch (train_local.py)
  <job>.eval.answers.graded.json        the judge's verdicts (judge.py grade)
"""
import argparse
import datetime
import json
from pathlib import Path

from models import BASES

TITLES = {"qwen3-0.6b": "Qwen3 0.6B", "llama-3.2-1b": "Llama 3.2 1B"}
OUT = Path(__file__).parent.parent / "web" / "public" / "results" / "run.json"  # what the results page reads


def build(job: str, jobs: Path, title: str | None = None) -> dict:
    plan = json.loads((jobs / f"{job}.plan.json").read_text())
    report = json.loads((jobs / "adapters" / job / "train_report.json").read_text())
    graded = json.loads((jobs / f"{job}.eval.answers.graded.json").read_text())
    base = plan["base"]
    epochs = [{"epoch": h["epoch"], "loss": h["loss"], "acc": h["acc"]} for h in report["history"]]
    pairs = [{"prompt": p["prompt"], "base": p["base"], "tuned": p["tuned"], "tuned_won": p["verdict"]["tuned_won"],
              "winner": p["verdict"].get("winner"), "reason": p["verdict"].get("reason")} for p in graded["pairs"]]
    return {"title": title or job.replace("-", " ").capitalize(), "goal": plan["goal"],
            "base_title": TITLES.get(base, base), "base_repo": BASES[base],
            "date": datetime.date.today().isoformat(),
            "source": (f"{report['training_chats']} chats written by the Job Architect, {report['heldout_chats']} held "
                       f"back, trained on {report['device']} in {report['minutes']} minutes"),
            "settings": (f"LoRA rank {plan['rank']} on every linear layer · {report['trainable_weights']:,} trained "
                         f"weights · {report['epochs']} epochs"),
            "reasoning": plan["reasoning"], "caveat": plan.get("caveat") or None,
            "agent_model": plan.get("model"), "judge_model": graded.get("model"),
            "base_loss": epochs[0]["loss"], "epochs": epochs, "kept_epoch": report.get("kept_epoch", epochs[-1]["epoch"]), "judge": {**graded["summary"], "pairs": pairs}}


def main():
    p = argparse.ArgumentParser(description="Write a finished job's results for the results page.")
    p.add_argument("job")
    p.add_argument("--title", help="the run's name on the page (default: from the job id)")
    p.add_argument("--jobs", type=Path, default=Path("agent-jobs"))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    run = build(a.job, a.jobs, a.title)
    a.out.write_text(json.dumps(run, indent=2), encoding="utf-8")
    j = run["judge"]
    print(f"wrote {a.out}: held-back loss {run['base_loss']} -> {run['epochs'][run['kept_epoch']]['loss']}, "
          f"fine-tune preferred on {j['tuned_wins']} of {j['prompts']}")


if __name__ == "__main__":
    main()
