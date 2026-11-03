"""The local trainer: the Job Architect's chats file in, a LoRA adapter out, on one machine.

    python agents/train_local.py agent-jobs/support-bot.jsonl --base qwen3-0.6b --rank 16

Needs PyTorch, transformers and PEFT (requirements-train.txt); a GPU makes it minutes instead of an hour.

LoRA in one paragraph: the base model's weights stay frozen. Next to each weight matrix W, LoRA adds two thin
matrices A and B (rank r, e.g. 16) and the layer computes W·x + B·A·x. Only A and B train, a fraction of a percent of
the parameters, so it fits on a small GPU and the result is a few-megabyte adapter instead of a whole new model.

The loop is written out rather than hidden in a trainer library, so every step is visible:
  1. Hold back a fifth of the chats. The model never trains on them; their loss before and after is the honest
     "did it learn the style or memorise the examples" number.
  2. Turn each chat into tokens, and mark the user's half -100 so the loss only scores the reply.
  3. A few passes over the training chats, an optimizer step every few chats.
  4. Keep the epoch with the lowest held-back loss. With a few dozen chats, a later pass often starts memorising the
     examples' wording, and the held-back loss rises again.
"""
import argparse
import json
import random
import time
from pathlib import Path

from models import BASES, chat_prompt

IGNORE = -100  # PyTorch's cross-entropy skips labels with this value


def load_chats(path: Path) -> list[tuple[str, str]]:
    """(user, assistant) pairs from the chats file the Job Architect writes."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)["messages"]
            pairs.append((m[0]["content"], m[1]["content"]))
    return pairs


def encode(tok, user: str, reply: str, max_len: int) -> dict:
    """Tokens for one chat, with labels that score only the reply (and the end-of-turn token, so it learns to stop)."""
    prompt = tok(chat_prompt(tok, user), add_special_tokens=False).input_ids
    answer = tok(reply + tok.eos_token, add_special_tokens=False).input_ids
    ids = (prompt + answer)[:max_len]
    return {"input_ids": ids, "labels": ([IGNORE] * len(prompt) + answer)[:max_len]}


def score(model, batch: list[dict]) -> tuple[float, float]:
    """Mean loss over the chats, and the share of reply tokens the model predicts exactly: the two numbers the results
    page charts epoch by epoch."""
    import torch
    model.eval()
    losses, right, total = [], 0, 0
    with torch.no_grad():
        for ex in batch:
            t = tensors(model, ex)
            out = model(**t)
            losses.append(out.loss.item())
            guess, labels = out.logits[0, :-1].argmax(-1), t["labels"][0, 1:]  # token i predicts token i + 1
            scored = labels != IGNORE
            right += (guess[scored] == labels[scored]).sum().item()
            total += scored.sum().item()
    model.train()
    return sum(losses) / len(losses), right / max(total, 1)


def tensors(model, ex: dict) -> dict:
    import torch
    return {k: torch.tensor([v], device=model.device) for k, v in ex.items()}


def train(chats: Path, base: str, rank: int, out: Path, epochs: int = 3, lr: float = 2e-4, accumulate: int = 4,
          max_len: int = 512, seed: int = 0) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    torch.manual_seed(seed)
    pairs = load_chats(chats)
    random.Random(seed).shuffle(pairs)
    held = max(2, len(pairs) // 5)
    heldout, training = pairs[:held], pairs[held:]

    gpu = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(BASES[base])
    model = AutoModelForCausalLM.from_pretrained(BASES[base], torch_dtype=torch.bfloat16 if gpu else torch.float32,
                                                 device_map="cuda" if gpu else None)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_set = [encode(tok, u, r, max_len) for u, r in training]
    held_set = [encode(tok, u, r, max_len) for u, r in heldout]
    before, acc = score(model, held_set)
    history = [{"epoch": 0, "loss": round(before, 4), "acc": round(acc, 4)}]
    print(f"held-back loss before: {before:.3f}  ({len(training)} training chats, {held} held back)")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    best, kept, best_weights = float("inf"), 0, None
    for epoch in range(epochs):
        order = random.Random(seed + epoch).sample(train_set, len(train_set))
        for i, ex in enumerate(order, 1):
            (model(**tensors(model, ex)).loss / accumulate).backward()  # gradients add up over `accumulate` chats
            if i % accumulate == 0 or i == len(order):
                optimizer.step()
                optimizer.zero_grad()
        loss, acc = score(model, held_set)
        history.append({"epoch": epoch + 1, "loss": round(loss, 4), "acc": round(acc, 4)})
        print(f"epoch {epoch + 1}/{epochs}: held-back loss {loss:.3f}, {acc:.1%} of reply tokens right")
        if loss < best:
            best, kept = loss, epoch + 1
            best_weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    if kept < epochs:  # a later epoch did worse: put the best one's LoRA weights back
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, w in best_weights.items():
                params[n].copy_(w)
    after = history[kept]["loss"]
    adapter = out / chats.stem
    model.save_pretrained(str(adapter))  # only the LoRA matrices: a few megabytes
    report = {"chats": str(chats), "base": base, "repo": BASES[base], "rank": rank, "epochs": epochs, "kept_epoch": kept, "lr": lr,
              "training_chats": len(training), "heldout_chats": held, "heldout_loss_before": round(before, 4),
              "heldout_loss_after": after, "history": history,
              "trainable_weights": sum(p.numel() for p in model.parameters() if p.requires_grad), "device": torch.cuda.get_device_name() if gpu else "cpu",
              "minutes": round((time.time() - started) / 60, 1)}
    (adapter / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"held-back loss {before:.3f} -> {after:.3f}, keeping epoch {kept} of {epochs}; adapter in {adapter}")
    return adapter


def main():
    p = argparse.ArgumentParser(description="Fine-tune a LoRA adapter on one machine.")
    p.add_argument("chats", type=Path)
    p.add_argument("--base", choices=sorted(BASES), required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("agent-jobs/adapters"))
    a = p.parse_args()
    train(a.chats, a.base, a.rank, a.out, epochs=a.epochs)


if __name__ == "__main__":
    main()
