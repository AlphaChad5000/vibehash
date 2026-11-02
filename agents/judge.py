"""Agent 2, the Semantic Judge: did the fine-tune actually get better at the goal, not just at lowering loss?

Two steps, because they need different machines:

    # 1. On a machine with a GPU, torch, transformers and peft: both models answer the same unseen prompts.
    python agents/judge.py answer agent-jobs/support-bot.eval.json path/to/support-bot-lora

    # 2. Anywhere with an API key: a stronger model compares each pair of answers.
    python agents/judge.py grade agent-jobs/support-bot.eval.answers.json

Vibe Hash already proves the training was done honestly (its referees and jury check the arithmetic) and reports
held-back loss. Neither says whether the answers are *good*. That's this agent's job, and the network's numbers and
this one complement each other.

Judging is pairwise and blind: the judge sees "A" and "B", never which is the fine-tune, and the order is shuffled
per prompt, because LLM judges tend to favour whichever answer comes first.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere in the repo
from llm_call import chat_json  # noqa: E402


def answer(eval_file: Path, adapter_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vibehash import catalog

    spec = json.loads(eval_file.read_text())
    repo = catalog.find(spec["base"]).repo
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

    def reply(m, prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt").to(m.device)
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


def grade(answers_file: Path, seed: int = 0) -> None:
    data = json.loads(answers_file.read_text())
    rng = random.Random(seed)  # seeded, so a rerun shuffles identically and the result can be reproduced
    wins = ties = 0
    for pair in data["pairs"]:
        tuned_is_a = rng.random() < 0.5
        a, b = (pair["tuned"], pair["base"]) if tuned_is_a else (pair["base"], pair["tuned"])
        verdict = chat_json([{"role": "system", "content": JUDGE},
                             {"role": "user", "content": f"Goal: {data['goal']}\n\nUser: {pair['prompt']}\n\n"
                                                         f"Reply A:\n{a}\n\nReply B:\n{b}"}])
        winner = verdict.get("winner")
        if winner == "tie":
            ties += 1
        elif (winner == "A") == tuned_is_a:
            wins += 1
        pair["verdict"] = {"tuned_won": winner != "tie" and (winner == "A") == tuned_is_a, **verdict}
        print(f"{'tuned' if pair['verdict']['tuned_won'] else 'tie' if winner == 'tie' else 'base':>5}  {verdict.get('reason')}")
    n = len(data["pairs"])
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
