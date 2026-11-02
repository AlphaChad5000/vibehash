"""Agent 1, the Job Architect: a plain-English goal in, a Vibe Hash fine-tuning job out.

    python agents/architect.py "make Llama answer customer-support questions in two friendly sentences"
    python agents/architect.py "..." --post      # also posts it (otherwise it only prints the command)

The split of responsibilities is the whole design:
  - The LLM does the fuzzy part: understand the goal, write example chats, pick sensible settings.
  - This code does the strict part: check every value against hard limits before anything is spent.
  - `vibehash train` does the protocol part: tokenizing, the held-back split, dataset hashes, pinning the model,
    posting the job. The agent never touches those, so it can't get them wrong.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere in the repo
from vibehash import catalog  # noqa: E402
from llm_call import chat_json  # noqa: E402

# Small models only, so a demo trains on gaming GPUs in minutes (FINETUNING.md's sizes).
ALLOWED_BASES = {"llama-3.2-1b", "qwen3-0.6b"}
MAX_BUDGET = 200  # VIBE: the most one agent-written job may spend

SYSTEM = f"""You design fine-tuning jobs. Given a goal, reply with ONE JSON object:
{{
  "job_id": "short-kebab-case-name",
  "base": one of {sorted(ALLOWED_BASES)},
  "rank": LoRA rank, 8 or 16,
  "budget": VIBE to spend, 50 to {MAX_BUDGET},
  "reasoning": "one or two sentences on why these settings",
  "examples": [{{"user": "...", "assistant": "..."}}, ...]   (at least 40, varied, in exactly the style wanted),
  "eval_prompts": ["...", ...]   (10 NEW user messages, not in examples, to test the result with)
}}"""


def design(goal: str) -> dict:
    return chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": goal}])


def check(plan: dict) -> list[str]:
    """Every rule the LLM was told, enforced. An LLM's output is a suggestion until code has checked it."""
    problems = []
    if plan.get("base") not in ALLOWED_BASES or catalog.find(plan.get("base", "")) is None:
        problems.append(f"base {plan.get('base')!r} isn't one of {sorted(ALLOWED_BASES)}")
    if plan.get("rank") not in (8, 16):
        problems.append(f"rank {plan.get('rank')!r} isn't 8 or 16")
    if not isinstance(plan.get("budget"), (int, float)) or not 0 < plan["budget"] <= MAX_BUDGET:
        problems.append(f"budget {plan.get('budget')!r} isn't between 1 and {MAX_BUDGET}")
    examples = plan.get("examples") or []
    if len(examples) < 40:
        problems.append(f"only {len(examples)} examples; need at least 40")
    if any(not isinstance(e, dict) or not e.get("user") or not e.get("assistant") for e in examples):
        problems.append("an example is missing its user or assistant text")
    if len(plan.get("eval_prompts") or []) < 5:
        problems.append("fewer than 5 eval prompts")
    return problems


def write_files(plan: dict, out: Path) -> tuple[Path, Path]:
    """The chats file in the format `vibehash train` reads (FINETUNING.md), and the eval prompts for Agent 2."""
    out.mkdir(parents=True, exist_ok=True)
    chats, evals = out / f"{plan['job_id']}.jsonl", out / f"{plan['job_id']}.eval.json"
    with chats.open("w", encoding="utf-8") as f:
        for e in plan["examples"]:
            f.write(json.dumps({"messages": [{"role": "user", "content": e["user"]},
                                             {"role": "assistant", "content": e["assistant"]}]}) + "\n")
    evals.write_text(json.dumps({"goal": plan["goal"], "base": plan["base"], "prompts": plan["eval_prompts"]}, indent=2))
    return chats, evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("goal")
    p.add_argument("--post", action="store_true", help="run `vibehash train` (it still shows its pre-flight first)")
    p.add_argument("--out", default="agent-jobs")
    p.add_argument("--node"); p.add_argument("--wallet")
    a = p.parse_args()

    plan = design(a.goal)
    for attempt in range(2):  # up to two retries: tell the model what was wrong and let it fix it
        problems = check(plan)
        if not problems:
            break
        print("plan rejected:", *problems, sep="\n  ")
        plan = chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.goal},
                          {"role": "assistant", "content": json.dumps(plan)},
                          {"role": "user", "content": "Fix these and reply with the whole JSON again:\n" + "\n".join(problems)}])
    if problems := check(plan):
        sys.exit("still invalid after a retry:\n  " + "\n  ".join(problems))
    plan["goal"] = a.goal

    chats, evals = write_files(plan, Path(a.out))
    cmd = ["vibehash", "train", str(chats), "--base", plan["base"], "--rank", str(plan["rank"]),
           "--budget", str(plan["budget"]), "--id", plan["job_id"]]
    cmd += ["--node", a.node] if a.node else []
    cmd += ["--wallet", a.wallet] if a.wallet else []
    print(f"\n{plan['job_id']}: {len(plan['examples'])} examples on {plan['base']}\n  why: {plan['reasoning']}")
    print(f"  wrote {chats} and {evals}\n  command: {' '.join(cmd)}")
    if a.post:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
