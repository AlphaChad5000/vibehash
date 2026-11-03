"""Agent 1, the Job Architect: a plain-English goal in, a fine-tuning job out.

    python agents/architect.py "answer customer-support questions in two friendly sentences"
    python agents/architect.py "..." --post      # also trains it (otherwise it only prints the command)

The split of responsibilities is the whole design:
  - The LLM does the fuzzy part: understand the goal, write example chats, pick sensible settings.
  - This code does the strict part: check every value against hard limits before any GPU time is spent.
  - agents/train_local.py does the training: a LoRA adapter, on this machine's GPU.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from llm_call import chat_json, model_name
from models import BASES

# Qwen3-0.6B unless .env.local says otherwise (ALLOWED_BASES=qwen3-0.6b,llama-3.2-1b): Llama is gated on Hugging Face,
# so it needs a logged-in account that accepted its licence
ALLOWED_BASES = set(os.environ.get("ALLOWED_BASES", "qwen3-0.6b").split(",")) & set(BASES)
EPOCHS = range(2, 6)  # passes over the chats: fewer under-learns a style, more memorises the examples

SYSTEM = f"""You design fine-tuning jobs. Given a goal, reply with ONE JSON object:
{{
  "job_id": "short-kebab-case-name",
  "base": one of {sorted(ALLOWED_BASES)},
  "rank": LoRA rank, 8 or 16,
  "epochs": passes over the examples, {EPOCHS.start} to {EPOCHS.stop - 1} (with about 40 examples, 2 or 3: more
            passes memorise the examples' wording),
  "reasoning": "one or two sentences on why these settings",
  "caveat": "" if a small fine-tune can do what the goal asks; otherwise one plain sentence for the requester on what
            it won't do. A fine-tune of a small model learns a style, format, tone or persona well, but it doesn't
            learn many new facts from a few dozen chats: for a goal that depends on facts (lore, a catalogue, prices,
            dates), say it will copy the expert's voice and likely get facts wrong,
  "examples": [{{"user": "...", "assistant": "..."}}, ...]   (at least 40, varied, in exactly the style wanted),
  "eval_prompts": ["...", ...]   (10 NEW user messages, not in examples, to test the result with)
}}"""


def design(goal: str) -> dict:
    return chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": goal}])


def check(plan: dict) -> list[str]:
    """Every rule the LLM was told, enforced. An LLM's output is a suggestion until code has checked it."""
    problems = []
    if plan.get("base") not in ALLOWED_BASES:
        problems.append(f"base {plan.get('base')!r} isn't one of {sorted(ALLOWED_BASES)}")
    if plan.get("rank") not in (8, 16):
        problems.append(f"rank {plan.get('rank')!r} isn't 8 or 16")
    if plan.get("epochs") not in EPOCHS:
        problems.append(f"epochs {plan.get('epochs')!r} isn't {EPOCHS.start} to {EPOCHS.stop - 1}")
    examples = plan.get("examples") or []
    if len(examples) < 40:
        problems.append(f"only {len(examples)} examples; need at least 40")
    if any(not isinstance(e, dict) or not e.get("user") or not e.get("assistant") for e in examples):
        problems.append("an example is missing its user or assistant text")
    if not isinstance(plan.get("caveat", ""), str):
        problems.append("caveat isn't a string")
    if len(plan.get("eval_prompts") or []) < 5:
        problems.append("fewer than 5 eval prompts")
    return problems


def write_files(plan: dict, out: Path) -> tuple[Path, Path]:
    """The chats file train_local.py reads, and the eval prompts for Agent 2."""
    out.mkdir(parents=True, exist_ok=True)
    chats, evals = out / f"{plan['job_id']}.jsonl", out / f"{plan['job_id']}.eval.json"
    with chats.open("w", encoding="utf-8") as f:
        for e in plan["examples"]:
            f.write(json.dumps({"messages": [{"role": "user", "content": e["user"]},
                                             {"role": "assistant", "content": e["assistant"]}]}) + "\n")
    evals.write_text(json.dumps({"goal": plan["goal"], "base": plan["base"], "prompts": plan["eval_prompts"]}, indent=2))
    return chats, evals


def plan_job(goal: str, log=print) -> dict:
    """Design a job and check it, giving the model two chances to fix what the checks reject."""
    plan = design(goal)
    for attempt in range(2):  # up to two retries: tell the model what was wrong and let it fix it
        problems = check(plan)
        if not problems:
            break
        log("plan rejected:\n  " + "\n  ".join(problems))
        plan = chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": goal},
                          {"role": "assistant", "content": json.dumps(plan)},
                          {"role": "user", "content": "Fix these and reply with the whole JSON again:\n" + "\n".join(problems)}])
    if problems := check(plan):
        raise ValueError("still invalid after a retry:\n  " + "\n  ".join(problems))
    plan["goal"] = goal
    return plan


def save(plan: dict, out: Path) -> tuple[Path, Path]:
    """The chats and eval files, and what publish.py needs later: the goal and the settings."""
    chats, evals = write_files(plan, out)
    (out / f"{plan['job_id']}.plan.json").write_text(json.dumps(
        {k: plan[k] for k in ("job_id", "goal", "base", "rank", "epochs", "reasoning")}
        | {"caveat": plan.get("caveat") or "", "examples": len(plan["examples"]), "model": model_name()}, indent=2))
    return chats, evals


def train_command(plan: dict, chats: Path, out: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).with_name("train_local.py")), str(chats), "--base", plan["base"],
            "--rank", str(plan["rank"]), "--epochs", str(plan["epochs"]), "--out", str(out / "adapters")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("goal")
    p.add_argument("--post", action="store_true", help="train it now, not just print the command")
    p.add_argument("--out", default="agent-jobs")
    a = p.parse_args()

    try:
        plan = plan_job(a.goal)
    except ValueError as e:
        sys.exit(str(e))
    out = Path(a.out)
    chats, evals = save(plan, out)
    cmd = train_command(plan, chats, out)
    print(f"\n{plan['job_id']}: {len(plan['examples'])} examples on {plan['base']}\n  why: {plan['reasoning']}")
    if plan.get("caveat"):
        print(f"  caveat: {plan['caveat']}")
    print(f"  wrote {chats} and {evals}\n  command: {' '.join(cmd)}")
    if a.post:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
