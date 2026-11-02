"""Agent 1, the Job Architect: a plain-English goal in, a Vibe Hash fine-tuning job out.

    python agents/architect.py "answer customer-support questions in two friendly sentences"
    python agents/architect.py "..." --post --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json

Without --post it writes the files and prints the `vibehash train` command, and no VIBE is spent. With --post it runs
that command, which shows its pre-flight and asks "Post it?" first (unless --yes), so there's time to read the chats
file; once posted, the job's budget is spent on the node it went to: the public network's unless --node or
`vibehash setup` says otherwise. Each design is a paid call to the LLM either way (agents/README.md, "What it costs").

The split of responsibilities is the whole design:
  - The LLM does the fuzzy part: understand the goal, write example chats, pick sensible settings.
  - This code does the strict part: check every value against what `vibehash train` and the chain would refuse, and
    against what would quietly make a bad job (duplicates, cut-off replies, test prompts it trained on), before
    anything is spent.
  - `vibehash train` does the protocol part: tokenizing, the held-back split, dataset hashes, pinning the model,
    posting the job. The agent never touches those, so it can't get them wrong.

It writes AGENT_JOBS/<job>.jsonl (the chats, in the format FINETUNING.md gives) and <job>.eval.json (the goal and
prompts it didn't train on, for the Semantic Judge).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # run from anywhere: this checkout's vibehash and agents
from agents.llm_call import LLMError, chat_json, describe  # noqa: E402
from vibehash import catalog, settings  # noqa: E402
from vibehash.tasks import _JOB_ID  # noqa: E402  (what the chain takes as a job id)

# Small models only, so a demo trains on gaming GPUs in minutes (FINETUNING.md's sizes). Llama is gated on Hugging
# Face: `vibehash train` and the judge need a token from an account that accepted Meta's licence.
BASES = {"qwen3-0.6b": False, "llama-3.2-1b": True}  # name -> gated
RANKS = (4, 8, 16, 32)  # LoRA ranks; each passes the chain's limits on both bases at MAX_LEN (tests/test_agents.py)
FEE = Decimal(10)  # VIBE a round: `vibehash train`'s --fee default. The chain wants budget >= one round's fee
MAX_BUDGET = 200  # VIBE: the most one agent-written job may spend
MAX_LEN = 512  # train's --max-len default; longer sequences make these bases' disputes too big for the chain
MIN_EXAMPLES, MAX_EXAMPLES = 40, 400
MIN_EVAL, MAX_EVAL = 5, 50
TEMPLATE_TOKENS = 64  # what a chat template wraps around one exchange (Llama 3.2's dated system header is ~30)
RETRIES = 2  # times the LLM is shown what was wrong and asked again (each one is a paid call)
WINDOWS_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}
_SPECIAL = re.compile(r"<\|[^|<>\s]{1,40}\|>|</?think>", re.IGNORECASE)  # chat-template tokens, Qwen3's think tags

SYSTEM = f"""You design fine-tuning jobs for small open chat models. Given a goal, reply with ONE JSON object and
nothing else:
{{
  "job_id": "short-kebab-case-name (lowercase letters, digits and dashes, under 40 characters)",
  "base": one of {json.dumps(sorted(BASES))} (qwen3-0.6b unless the goal needs Llama),
  "rank": LoRA rank, one of {list(RANKS)} (8 or 16 suit most goals),
  "budget": whole VIBE to spend, {FEE} to {MAX_BUDGET}. Each round costs {FEE} VIBE and is about 16 training steps
            of 8 examples; 4 to 8 rounds suit 60-150 examples,
  "reasoning": "one or two sentences on why these settings",
  "examples": [{{"user": "...", "assistant": "..."}}, ...],
  "eval_prompts": ["...", ...]
}}
Rules for examples: {MIN_EXAMPLES} to 150 of them, each one user message and the assistant reply in exactly the style
the goal wants; varied topics and wording; no two with the same user message; each exchange under about 300 words;
plain text only, with no chat-template tokens such as <|im_start|> and no <think> tags.
Rules for eval_prompts: {MIN_EVAL} to 20 NEW user messages, none of them the same as an example's, to test the result
with."""


def design(goal: str, base: str | None = None, env=None) -> dict:
    """The LLM's plan for the goal (unchecked)."""
    ask = goal if base is None else f"{goal}\n\n(Use base {base}.)"
    return chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}], env=env)


def fix(goal: str, plan, problems: list[str], env=None) -> dict:
    """The LLM's plan again, told what the checks refused."""
    return chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": goal},
                      {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                      {"role": "user", "content": "These problems must be fixed. Reply with the whole JSON object "
                                                  "again:\n- " + "\n- ".join(problems)}], env=env)


def job_id_for(name) -> str:
    """A job id from the LLM's name for it, as `vibehash train` makes one from a file name: lowercase, a-z 0-9 . _ -,
    and four random hex digits so it's new on the chain."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(name or "").lower()).strip("-._")[:40] or "job"
    return f"{slug}-{secrets.token_hex(2)}"


def id_problems(job_id, out: Path | None = None) -> list[str]:
    """Why `job_id` can't be a job's name: the chain's rule (tasks._JOB_ID), safe as a file name on every system,
    and not one `vibehash train` already has a receipt for on this computer."""
    if not isinstance(job_id, str) or not _JOB_ID.match(job_id):
        return [f"job_id {job_id!r} must be 1-64 characters of a-z, 0-9, '.', '_' and '-', starting with a letter or "
                "digit (the chain's rule)"]
    if job_id.endswith("."):
        return [f"job_id {job_id!r} ends in a dot, which Windows drops from folder names"]
    if job_id.split(".")[0] in WINDOWS_NAMES:
        return [f"job_id {job_id!r} is a device name on Windows, which can't be a file name"]
    if (settings.home() / "jobs" / job_id).exists():
        return [f"you already have a job called {job_id} ({settings.home() / 'jobs' / job_id}); pick another name"]
    if out is not None and (out / f"{job_id}.jsonl").exists():
        return [f"{out / (job_id + '.jsonl')} already exists; pick another name or --out folder"]
    return []


def tokens_about(text: str) -> int:
    """A generous guess at a text's length in tokens, without the model's tokenizer: a token every 3 characters of
    ASCII (English averages about 4), and one a character for the rest (Chinese, emoji)."""
    ascii_ = sum(1 for c in text if ord(c) < 128)
    return math.ceil(ascii_ / 3) + len(text) - ascii_


def _norm(text: str) -> str:
    """Text compared for duplicates: lowercase, letters and digits only, single spaces."""
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


def _which(numbers: list[int]) -> str:
    shown = ", ".join(str(n) for n in numbers[:6])
    return shown + (f" and {len(numbers) - 6} more" if len(numbers) > 6 else "")


def _number(v):
    """A JSON number as a Decimal, or None for anything else (true and false included)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return Decimal(str(v))


def check(plan, *, fee=FEE, max_len: int = MAX_LEN, out: Path | None = None) -> list[str]:
    """Every rule the LLM was told, enforced, plus what `vibehash train` and the chain would refuse after the poster
    has waited for the weights check. An LLM's output is a suggestion until code has checked it. [] means it's fine."""
    if not isinstance(plan, dict):
        return ["the reply must be one JSON object"]
    problems = id_problems(plan.get("job_id"), out)
    base = plan.get("base")
    if base not in BASES or catalog.find(base) is None:
        problems.append(f"base {base!r} must be one of {sorted(BASES)}")
    elif BASES[base] and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        problems.append(f"base {base} is gated on Hugging Face: set HF_TOKEN from an account that accepted Meta's "
                        "licence, or use qwen3-0.6b")
    if type(plan.get("rank")) is not int or plan["rank"] not in RANKS:
        problems.append(f"rank {plan.get('rank')!r} must be one of {list(RANKS)}")
    budget, fee = _number(plan.get("budget")), Decimal(str(fee))
    if budget is None or budget != budget.to_integral_value():
        problems.append(f"budget {plan.get('budget')!r} must be a whole number of VIBE")
    elif not fee <= budget <= MAX_BUDGET:
        problems.append(f"budget {budget} must be from {fee} (one round's fee) to {MAX_BUDGET} VIBE")

    examples = plan.get("examples")
    if not isinstance(examples, list):
        problems.append("examples must be a list of {\"user\": ..., \"assistant\": ...} objects")
        examples = []
    elif not MIN_EXAMPLES <= len(examples) <= MAX_EXAMPLES:
        problems.append(f"there are {len(examples)} examples; it takes {MIN_EXAMPLES} to {MAX_EXAMPLES}")
    shape, empty, special, long, seen, dupes = [], [], [], [], {}, []
    for n, e in enumerate(examples, 1):
        if not isinstance(e, dict) or not isinstance(e.get("user"), str) or not isinstance(e.get("assistant"), str):
            shape.append(n)
            continue
        user, reply = e["user"].strip(), e["assistant"].strip()
        if not user or not reply:
            empty.append(n)
        if _SPECIAL.search(user) or _SPECIAL.search(reply):
            special.append(n)
        if tokens_about(user) + tokens_about(reply) + TEMPLATE_TOKENS > max_len:
            long.append(n)
        key = _norm(user)
        if key and key in seen:
            dupes.append(f"{seen[key]} and {n}")
        seen.setdefault(key, n)
    if shape:
        problems.append(f"examples {_which(shape)} aren't {{\"user\": \"text\", \"assistant\": \"text\"}} objects")
    if empty:
        problems.append(f"examples {_which(empty)} have an empty user message or reply")
    if special:
        problems.append(f"examples {_which(special)} contain chat-template tokens or <think> tags; use plain text")
    if long:
        problems.append(f"examples {_which(long)} may be longer than {max_len} tokens, where `vibehash train` cuts "
                        "chats short (and the reply's end is lost); shorten them")
    if dupes:
        problems.append(f"examples {_which(dupes)} have the same user message; make each one different")

    prompts = plan.get("eval_prompts")
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        problems.append("eval_prompts must be a list of non-empty strings")
    else:
        if not MIN_EVAL <= len(prompts) <= MAX_EVAL:
            problems.append(f"there are {len(prompts)} eval prompts; it takes {MIN_EVAL} to {MAX_EVAL}")
        keys = [_norm(p) for p in prompts]
        repeated = [n for n, k in enumerate(keys, 1) if k in keys[:n - 1]]
        trained = [n for n, k in enumerate(keys, 1) if k in seen]
        if repeated:
            problems.append(f"eval prompts {_which(repeated)} repeat earlier eval prompts")
        if trained:
            problems.append(f"eval prompts {_which(trained)} are also examples' user messages; the test must be new")
        if any(_SPECIAL.search(p) for p in prompts):
            problems.append("eval prompts contain chat-template tokens or <think> tags; use plain text")
    return problems


def write_files(plan: dict, out: Path, goal: str) -> tuple[Path, Path]:
    """The chats file in the format `vibehash train` reads (FINETUNING.md), and the eval prompts for the judge."""
    out.mkdir(parents=True, exist_ok=True)
    chats, evals = out / f"{plan['job_id']}.jsonl", out / f"{plan['job_id']}.eval.json"
    with chats.open("w", encoding="utf-8", newline="\n") as f:
        for e in plan["examples"]:
            f.write(json.dumps({"messages": [{"role": "user", "content": e["user"].strip()},
                                             {"role": "assistant", "content": e["assistant"].strip()}]},
                               ensure_ascii=False) + "\n")
    evals.write_text(json.dumps({
        "goal": goal, "job_id": plan["job_id"], "base": plan["base"], "prompts": [p.strip() for p in plan["eval_prompts"]],
        "settings": {"rank": plan["rank"], "budget": int(plan["budget"]), "examples": len(plan["examples"])},
        "reasoning": plan.get("reasoning"), "designed_by": describe()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    return chats, evals


def train_command(plan: dict, chats: Path, *, max_len: int = MAX_LEN, fee: str | None = None, node: str | None = None,
                  wallet: str | None = None, yes: bool = False) -> list[str]:
    """The `vibehash train` command line for the plan, run by this Python. --plain: it posts and returns, rather than
    open the training dashboard (`vibehash job watch` does that)."""
    cmd = [sys.executable, "-m", "vibehash", "train", str(Path(chats).resolve()), "--base", plan["base"],
           "--rank", str(plan["rank"]), "--budget", str(int(plan["budget"])), "--max-len", str(max_len),
           "--id", plan["job_id"], "--plain"]
    cmd += ["--fee", fee] if fee is not None else []
    cmd += ["--node", node] if node else []
    cmd += ["--wallet", wallet] if wallet else []
    cmd += ["--yes"] if yes else []
    return cmd


def shown(cmd: list[str]) -> str:
    """A command line to paste into this system's shell."""
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def post(cmd: list[str], job_id: str) -> bool:
    """Run `vibehash train`, with this checkout's vibehash first on the path. True if it posted the job: train keeps a
    receipt in ~/.vibehash/jobs/<job>/job.json once it has, and says "Nothing was posted" (exit 0) when told no."""
    path = os.environ.get("PYTHONPATH")
    env = {**os.environ, "PYTHONPATH": str(REPO) + (os.pathsep + path if path else "")}
    code = subprocess.run(cmd, env=env).returncode
    if code:
        sys.exit(f"`vibehash train` stopped with exit code {code}; nothing more was done")
    return (settings.home() / "jobs" / job_id / "job.json").exists()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="architect.py", description="A plain-English goal in, a fine-tuning job out.")
    p.add_argument("goal", help="what the fine-tuned model should do, in plain English")
    p.add_argument("--base", choices=sorted(BASES), help="the model to fine-tune (default: the LLM picks)")
    p.add_argument("--id", help="the job's name (default: the LLM's, plus four random hex digits)")
    p.add_argument("--out", default="agent-jobs", help="folder for the files it writes (default agent-jobs)")
    p.add_argument("--max-len", type=int, default=MAX_LEN, help=f"longest chat in tokens, 64-{MAX_LEN} (default "
                   f"{MAX_LEN}, train's)")
    p.add_argument("--fee", help=f"VIBE a round, passed to train (default: train's, {FEE})")
    p.add_argument("--post", action="store_true", help="run `vibehash train`, which spends the budget once it posts")
    p.add_argument("--yes", action="store_true", help="with --post: train asks nothing (for scripts)")
    p.add_argument("--node", help="the node to post to (default: train's: the one setup saved, else the public one)")
    p.add_argument("--wallet", help="the wallet that pays (default: train's)")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    out = Path(a.out)
    if not 64 <= a.max_len <= MAX_LEN:
        sys.exit(f"--max-len must be 64-{MAX_LEN}: the chain refuses longer sequences on these models' disputes")
    try:
        fee = FEE if a.fee is None else Decimal(a.fee)
    except InvalidOperation:
        sys.exit(f"--fee {a.fee!r} isn't an amount of VIBE")
    if fee <= 0 or fee > MAX_BUDGET:
        sys.exit(f"--fee must be above 0 and at most {MAX_BUDGET} VIBE")
    if a.id and (problems := id_problems(a.id, out)):  # before anything is paid for
        sys.exit("error: " + problems[0])
    if a.base and (problems := [p for p in check({"base": a.base}) if p.startswith("base ")]):
        sys.exit("error: " + problems[0])

    job_id = a.id
    try:
        plan = design(a.goal, a.base)
        for attempt in range(RETRIES + 1):
            if isinstance(plan, dict):
                job_id = job_id or job_id_for(plan.get("job_id"))
                plan["job_id"] = job_id
                if a.base:
                    plan["base"] = a.base
            problems = check(plan, fee=fee, max_len=a.max_len, out=out)
            if not problems:
                break
            print("plan rejected:", *problems, sep="\n  ", file=sys.stderr)
            if attempt == RETRIES:
                sys.exit(f"still not right after {RETRIES} retries; nothing was written, and no VIBE spent")
            plan = fix(a.goal, plan, problems)
    except LLMError as e:
        sys.exit(f"error: {e}")

    chats, evals = write_files(plan, out, a.goal)
    wallet = os.path.expanduser(a.wallet) if a.wallet else None  # ~/... from a shell that leaves it alone
    cmd = train_command(plan, chats, max_len=a.max_len, fee=a.fee, node=a.node, wallet=wallet, yes=a.yes)
    rounds = int(Decimal(int(plan["budget"])) // fee)
    print(f"\n{job_id}: {len(plan['examples'])} examples on {plan['base']}, LoRA rank {plan['rank']}, "
          f"{int(plan['budget'])} VIBE (up to {rounds} rounds at {fee} VIBE)\n"
          f"  why:   {plan.get('reasoning') or '(not given)'}\n"
          f"  wrote: {chats}\n         {evals} ({len(plan['eval_prompts'])} prompts it didn't train on)\n"
          f"  train: {shown(cmd)}")
    if not a.post:
        print(f"\nNothing was posted, and no VIBE spent. To post this job, run that command from {REPO} (so it uses "
              "this checkout's vibehash). Running the architect again designs a new one.")
        return 0
    print(f"\nPosting to {settings.node_url(a.node)}: `vibehash train` shows its pre-flight first"
          + ("." if a.yes else ", and asks before it spends anything."), flush=True)
    if not post(cmd, job_id):
        print("Nothing was posted, so nothing was spent.")
        return 0
    adapter = f"{job_id}-lora"
    print(f"\nPosted {job_id}. Next, from this folder:\n"
          f"  python -m vibehash job watch {job_id}\n"
          f"      follow it as the network trains it\n"
          f"  python -m vibehash job download {job_id} --out {adapter}\n"
          f"      the LoRA adapter, once rounds have trained\n"
          f"  python agents/judge.py answer {shown([str(evals)])} {adapter}\n"
          f"      the base model and the fine-tune answer the new prompts (a GPU helps)\n"
          f"  python agents/judge.py grade {shown([str(evals).replace('.eval.json', '.answers.json')])}\n"
          "      an LLM compares each pair of answers blind")
    return 0


if __name__ == "__main__":
    sys.exit(main())
