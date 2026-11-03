"""Agent 2, the Semantic Judge: did the fine-tune actually get better at the goal, not just at lowering loss?

Two steps, because they need different machines:

    # 1. On a machine with a GPU, torch, transformers and peft: both models answer the same unseen prompts.
    python agents/judge.py answer agent-jobs/support-bot.eval.json path/to/support-bot-lora

    # 2. Anywhere with an API key: a stronger model compares each pair of answers.
    python agents/judge.py grade agent-jobs/support-bot.eval.answers.json

Training reports held-back loss, which says the model fits the style of its examples. It doesn't say whether the
answers are *good*. That's this agent's job, and the trainer's numbers and this one complement each other.

Judging is pairwise and blind: the judge sees "A" and "B", never which is the fine-tune. Every pair is judged twice,
once in each order, because LLM judges tend to favour whichever answer comes first; a win needs both.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_call import chat_json, model_name
from models import BASES, chat_prompt


def answer(eval_file: Path, adapter_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = json.loads(eval_file.read_text())
    repo = BASES[spec["base"]]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

    def reply(m, prompt):
        ids = tok(chat_prompt(tok, prompt), add_special_tokens=False, return_tensors="pt").input_ids.to(m.device)
        out = m.generate(ids, max_new_tokens=200, do_sample=False)  # greedy: the same answer every run
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    base = [reply(model, p) for p in spec["prompts"]]
    tuned_model = PeftModel.from_pretrained(model, str(adapter_dir))  # the same weights plus the LoRA adapter
    tuned = [reply(tuned_model, p) for p in spec["prompts"]]

    out = eval_file.with_suffix(".answers.json")
    out.write_text(json.dumps({"goal": spec["goal"], "pairs": [
        {"prompt": p, "base": b, "tuned": t} for p, b, t in zip(spec["prompts"], base, tuned)]}, indent=2))
    print(f"wrote {out}")
    return out


JUDGE = """You compare two assistant replies against a goal. Reply with ONE JSON object:
{"winner": "A" or "B" or "tie", "reason": "one sentence"}
Judge only how well each reply meets the goal. Ignore which is longer unless the goal is about length."""


def compare(goal: str, pair: dict, tuned_is_a: bool) -> tuple[str, str]:
    """One blind reading: which reply the judge prefers ("tuned", "base" or "tie"), and its reason."""
    a, b = (pair["tuned"], pair["base"]) if tuned_is_a else (pair["base"], pair["tuned"])
    verdict = chat_json([{"role": "system", "content": JUDGE},
                         {"role": "user", "content": f"Goal: {goal}\n\nUser: {pair['prompt']}\n\n"
                                                     f"Reply A:\n{a}\n\nReply B:\n{b}"}])
    winner = verdict.get("winner")
    side = "tie" if winner not in ("A", "B") else "tuned" if (winner == "A") == tuned_is_a else "base"
    return side, unblind(verdict.get("reason") or "", tuned_is_a)


def unblind(reason: str, tuned_is_a: bool) -> str:
    """The judge's reason names "Reply A" and "Reply B"; once it's decided, say which model each was, since the
    letters swap between the two readings and mean nothing to whoever reads the verdict."""
    name = {"A": "the fine-tune", "B": "the base model"} if tuned_is_a else {"A": "the base model", "B": "the fine-tune"}
    reason = re.sub(r"\b[Rr]eply ([AB])\b", lambda m: name[m[1]], reason)
    reason = re.sub(r"(^|[.!?]\s+)(the )", lambda m: m[1] + "The ", reason)  # sentence starts
    return reason


def verdict(goal: str, pair: dict) -> dict:
    """Both readings of one pair, and the verdict they make together."""
    one, two = compare(goal, pair, True), compare(goal, pair, False)
    side = one[0] if one[0] == two[0] else "tie"
    reason = one[1] if side == one[0] else two[1] if side == two[0] else \
        f"The two readings disagreed. With the fine-tune first: {one[1]} With it second: {two[1]}"
    return {"tuned_won": side == "tuned", "winner": side, "reason": reason, "readings": [one[0], two[0]]}


def grade(answers_file: Path, on_verdict=None) -> None:
    """Each pair is read twice, once with each reply first, and only a verdict both readings agree on counts: a
    single reading of a borderline pair can go either way, and the judge can lean towards whichever reply it sees
    first. A split is a tie. `on_verdict(i, pair)` hears each one as it's decided, for the site's live view."""
    data = json.loads(answers_file.read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:  # two pairs, so four calls, at a time; rate limits are retried
        pending = {pool.submit(verdict, data["goal"], pair): i for i, pair in enumerate(data["pairs"])}
        for done in as_completed(pending):
            i = pending[done]
            pair = data["pairs"][i]
            pair["verdict"] = done.result()
            print(f"{pair['verdict']['winner']:>5}  {pair['verdict']['reason']}")
            if on_verdict:
                on_verdict(i, pair)
    wins = sum(p["verdict"]["winner"] == "tuned" for p in data["pairs"])
    ties = sum(p["verdict"]["winner"] == "tie" for p in data["pairs"])
    n = len(data["pairs"])
    data["model"] = model_name()
    data["summary"] = {"prompts": n, "tuned_wins": wins, "ties": ties, "base_wins": n - wins - ties}
    answers_file.with_suffix(".graded.json").write_text(json.dumps(data, indent=2))
    print(f"\nfine-tune preferred on {wins} of {n} prompts ({ties} ties)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "answer":
        answer(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "grade":
        grade(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
