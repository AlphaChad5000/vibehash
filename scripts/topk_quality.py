"""What training only a model's top layers costs: held-back loss after LoRA on all layers, the top 20% or the top 10%.

    python scripts/topk_quality.py [--model DIR] [--data everyday.jsonl] [--epochs 2] [--lrs 2e-4,1e-3]

DEEPSEEK.md §8's route fine-tunes DeepSeek-V4.1-Flash's top layers only, since then the frozen layers' output can be
computed once and shipped as the job's data. This measures the price on a model small enough to train whole. It uses
Hugging Face transformers and PEFT, as a fine-tune's users would, and Vibe Hash's own recipe (finetune.py): LoRA rank 16,
alpha 32 on q, k, v and o, A frozen and only B trained, Adam, loss on the assistant's replies only. The same steps and
data for every run; held-back chats scored on their replies before training and every half epoch. Prints a table and
writes it as JSON next to the data.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HOME = Path.home() / ".vibehash" / "cache"


def chats(path: Path, tok, seq_len: int):
    """Token ids and a mask of the assistant's reply tokens (each reply's text and its <|im_end|>), per chat."""
    start = tok.convert_tokens_to_ids("<|im_start|>")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    assistant = tok.encode("assistant\n", add_special_tokens=False)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(text, add_special_tokens=False)[:seq_len]
        mask, inside, i = [0] * len(ids), False, 0
        while i < len(ids):
            if ids[i] == start and ids[i + 1:i + 1 + len(assistant)] == assistant:
                inside, i = True, i + 1 + len(assistant)
                continue
            if inside:
                mask[i] = 1
                if ids[i] == end:
                    inside = False
            i += 1
        if sum(mask):
            out.append((ids, mask))
    return out


def batches(rows, batch: int, pad: int, rng=None):
    order = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(order)
    for s in range(0, len(order), batch):
        part = [rows[i] for i in order[s:s + batch]]
        n = max(len(ids) for ids, _ in part)
        ids = torch.full((len(part), n), pad)
        mask = torch.zeros(len(part), n, dtype=torch.bool)
        for j, (t, m) in enumerate(part):
            ids[j, :len(t)] = torch.tensor(t)
            mask[j, :len(m)] = torch.tensor(m, dtype=torch.bool)
        yield ids, mask


def loss_of(model, ids, mask):
    """Summed cross-entropy over the reply tokens (each predicted from what precedes it), and their count."""
    logits = model(input_ids=ids).logits[:, :-1].float()
    want = mask[:, 1:]
    lse = torch.logsumexp(logits[want], -1)
    picked = logits[want].gather(1, ids[:, 1:][want][:, None])[:, 0]
    return (lse - picked).sum(), int(want.sum())


@torch.no_grad()
def evaluate(model, rows, pad, dev):
    total = count = 0
    for ids, mask in batches(rows, 8, pad):
        s, c = loss_of(model, ids.to(dev), mask.to(dev))
        total, count = total + float(s), count + c
    return total / count


def run(load, tok, train, val, top: int | None, lr: float, epochs: int, batch: int, seed: int, dev):
    from peft import LoraConfig, inject_adapter_in_model
    base = load()
    layers = base.config.num_hidden_layers
    chosen = list(range(layers)) if top is None else list(range(layers - top, layers))
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0,
                     layers_to_transform=chosen, layers_pattern="layers", init_lora_weights=True)
    torch.manual_seed(seed)
    model = inject_adapter_in_model(cfg, base)
    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad_("lora_B" in name)  # LoRA-FA, as finetune.py: A frozen, B trained
        if p.requires_grad:
            trainable.append(p)
    opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    rng = random.Random(seed)
    curve = [(0, evaluate(model, val, tok.pad_token_id, dev))]
    steps_per_epoch = math.ceil(len(train) / batch)
    t0, step = time.perf_counter(), 0
    for _ in range(epochs):
        for ids, mask in batches(train, batch, tok.pad_token_id, rng):
            total, count = loss_of(model, ids.to(dev), mask.to(dev))
            (total / count).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % (steps_per_epoch // 2) == 0:
                curve.append((step, evaluate(model, val, tok.pad_token_id, dev)))
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    n_b = sum(p.numel() for p in trainable)
    del model, base, opt
    torch.cuda.empty_cache()
    return {"layers": len(chosen), "top": top, "lr": lr, "trained_weights": n_b, "seconds": round(secs, 1),
            "steps": step, "curve": curve}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(HOME / "hf-models" / "Qwen3-0.6B"))
    ap.add_argument("--data", default=str(HOME / "datasets-src" / "everyday.jsonl"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lrs", default="2e-4,1e-3")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = chats(Path(args.data), tok, args.seq_len)
    random.Random(1).shuffle(rows)
    val, train = rows[:len(rows) // 10], rows[len(rows) // 10:]
    def load():
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
        return m.requires_grad_(False)

    from transformers import AutoConfig
    layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    tops = [None, max(1, round(layers * 0.2)), max(1, round(layers * 0.1))]
    print(f"{Path(args.model).name}: {layers} layers; {len(train)} chats to train on, {len(val)} held back; "
          f"{args.epochs} epochs of batch {args.batch}; GPU {torch.cuda.get_device_name()}", flush=True)
    results = []
    for lr in (float(x) for x in args.lrs.split(",")):
        for top in tops:
            r = run(load, tok, train, val, top, lr, args.epochs, args.batch, args.seed, dev)
            results.append(r)
            print(json.dumps(r), flush=True)
    print(f"\n{'lr':>7} {'layers':>7} {'B weights':>10} {'start':>7} " + " ".join(
        f"{'@' + str(s):>7}" for s, _ in results[0]["curve"][1:]) + f" {'seconds':>8}")
    for r in results:
        print(f"{r['lr']:>7.0e} {r['layers']:>7} {r['trained_weights']:>10} {r['curve'][0][1]:>7.4f} " + " ".join(
            f"{v:>7.4f}" for _, v in r["curve"][1:]) + f" {r['seconds']:>8.1f}")
    out = Path(args.data).with_name(f"topk_quality_{Path(args.model).name}.json")
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
