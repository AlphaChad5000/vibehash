"""Score an Evo v2 line against its base model: did the training help or hurt?

Loads a line a miner saved (`~/.vibehash/cache/evo2-lines/JOB.pt`, es_client._save_line) and scores the base model and
base + the line's change the way scripts/evo2_progress.py does: greedy, 64 tokens, 200 items a set, every es_tasks task
plus GSM8K and ARC-Challenge test items, which no Evo job trains on. Prints a table and writes JSON. Reads the line
only; needs no node.

  python scripts/evo2_score_line.py ~/.vibehash/cache/evo2-lines/evo2-c2-emotion.pt [--model Qwen/Qwen2.5-0.5B-Instruct]
      [--out scores.json]

First run, 2026-11-04 (HANDOFF, "Two findings that block an Evo starter job"): the owner's evo2-c2-emotion line (block
650, ~400 rounds) was worse than the base on every set and answered with repeated <answer> tags.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibehash import es, es2, es_local  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("line", help="a saved line, ~/.vibehash/cache/evo2-lines/JOB.pt")
    p.add_argument("--model", default=es_local.MODEL, help=f"the line's base model (default {es_local.MODEL})")
    p.add_argument("--out", help="write the scores here as JSON")
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("evo2_progress", ROOT / "scripts" / "evo2_progress.py")
    prog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prog)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    found = es_local.find(a.model, download=False)
    model = AutoModelForCausalLM.from_pretrained(found.folder, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(found.folder)
    gen = json.loads((found.folder / "generation_config.json").read_text())
    eos = tuple(gen["eos_token_id"]) if isinstance(gen["eos_token_id"], list) else (gen["eos_token_id"],)
    rt = es.Runtime(model, tok, "cuda", eos, gen.get("pad_token_id", eos[0]))
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    base = {n: m.weight.detach().float().cpu() for n, m in linears}
    saved = torch.load(Path(a.line).expanduser(), map_location="cpu", weights_only=True)
    ch = es2.Change.__new__(es2.Change)
    ch.d = saved["d"]
    moved = sorted(float(ch.d[n].norm() / base[n].norm()) for n in ch.d if n in base)
    print(f"line {saved['job']} at block {saved['last_round']}: each weight moved a median {moved[len(moved) // 2]:.1%} "
          f"of its size (largest {moved[-1]:.1%})", flush=True)
    sets = prog.eval_sets()

    def ids(item):
        x = tok.apply_chat_template(item.messages, add_generation_prompt=True, tokenize=True)
        return list(x["input_ids"] if hasattr(x, "keys") else x)

    def score(label, change):
        if change is None:
            for n, m in linears:
                m.weight.data.copy_(base[n])
        else:
            change.load_into(linears, base)
        res = {}
        with torch.no_grad():
            for name, items in sets.items():
                rows = [ids(i) for i in items]
                right = 0
                for lo in range(0, len(rows), 256):
                    got = es.generate(rt, rows[lo:lo + 256], 64)
                    right += sum(i.check(es.text_of(rt, x)) for i, x in zip(items[lo:lo + 256], got))
                res[name] = round(right / len(items), 4)
                print(f"  {label:8s} {name:32s} {res[name]:.1%}  ({time.time() - t0:.0f}s)", flush=True)
        return res

    results = {"line": saved["job"], "block": int(saved["last_round"]), "base": score("base", None),
               "trained": score("trained", ch)}
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\n{'set':32s}   base  trained  change")
    for name in sets:
        b, t = results["base"][name], results["trained"][name]
        print(f"{name:32s} {b:6.1%}  {t:6.1%}  {(t - b) * 100:+5.1f}")


if __name__ == "__main__":
    main()
