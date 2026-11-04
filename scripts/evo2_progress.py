"""Watch an Evo v2 testnet's model get better: after each cycle, rebuild its averaged model from the chain alone and
score it on the trained tasks and two it never trains on.

For every finished rule-2 job the node has, the job's change is rebuilt from its parents (es2.Spec2.start: [[job,
weight, params_hash], ...]) and its rounds' entries (es2.rebuild), as a worker rebuilds it; each job's change is kept
on disk (bf16) so it is rebuilt once. A cycle's model is the base plus the mean of its jobs' changes, the start the next
cycle's jobs train from. Scores: greedy, 64 tokens, 200 items a task (es_tasks' eval items; GSM8K and ARC-Challenge
test items for the never-trained pair), in bf16 on the GPU, beside the miners.

  python scripts/evo2_progress.py [--node http://127.0.0.1:8545] [--dir ~/.vibehash/testnet-evo2] [--watch 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibehash import es, es2, es_local, es_tasks  # noqa: E402
from vibehash.client import NodeClient  # noqa: E402

N_EVAL = 200


def _gsm8k_items():
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main")["test"]
    order = es_tasks._draw(b"evo2 gsm8k eval").shuffle(range(len(rows)))[:N_EVAL]
    out = []
    for i in order:
        r = rows[i]
        gold = float(r["answer"].split("####")[-1].replace(",", "").strip())
        q = f"{r['question']}\nReply with only the final number between <answer> and </answer>."
        out.append(es_tasks.Item([{"role": "user", "content": q}], es_tasks._number_check(gold), f"{gold:g}"))
    return out


def _arc_challenge_items():
    from datasets import load_dataset
    rows = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    order = es_tasks._draw(b"evo2 arc_challenge eval").shuffle(range(len(rows)))[:N_EVAL]
    return [es_tasks._arc_item(rows[i]) for i in order]


def eval_sets() -> dict:
    sets = {}
    for name in es_tasks.NAMES:
        try:
            sets[name] = es_tasks.get(name).eval_items()
        except es_tasks.TaskUnavailable as e:
            print(f"note: no {name} scores: {e}", file=sys.stderr)
    for name, make in (("gsm8k (never trained)", _gsm8k_items), ("arc_challenge (never trained)", _arc_challenge_items)):
        try:
            sets[name] = make()
        except Exception as e:  # noqa: BLE001 - no datasets library or no network: score without it
            print(f"note: no {name} scores: {e}", file=sys.stderr)
    return sets


class Chain:
    """The node's rule-2 jobs: each one's spec and its rounds' entries in chain order."""

    def __init__(self, url: str):
        self.api, self.height = NodeClient(url), 0
        self.specs: dict = {}
        self.entries: dict = {}

    def poll(self) -> None:
        tip = int(self.api.status()["height"])
        while self.height < tip:
            batch = self.api.get(f"/blocks?from={self.height + 1}&to={min(tip, self.height + 100)}")
            if not batch:
                break
            for b in batch:
                for tx in b.get("txs", []):
                    if tx.get("kind") == "post_job":
                        try:
                            d = json.loads(tx["memo"])
                        except (ValueError, KeyError, TypeError):
                            continue
                        if es2.is_v2(d):
                            self.specs[d["id"]] = es2.Spec2.from_json(d)
                s = b.get("summary")
                if isinstance(s, dict) and s.get("job_id") in self.specs:
                    rec = self.api.get(f"/round/{b['height']}/record")
                    e = (rec or {}).get("es") or {}
                    e = e.get("entry", e)  # the entry, wherever the round's record keeps it
                    if isinstance(e, dict) and "update" in e and "seed" in e:
                        self.entries.setdefault(s["job_id"], []).append(e)
                self.height = int(b["height"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", default="http://127.0.0.1:8545")
    p.add_argument("--dir", default=str(Path.home() / ".vibehash" / "testnet-evo2"))
    p.add_argument("--watch", type=float, default=0, help="seconds between checks for a newly finished cycle (0: once)")
    a = p.parse_args()
    d = Path(a.dir)
    found = es_local.find(json.loads((d / "evo.json").read_text())["model"] if (d / "evo.json").exists()
                          else es_local.MODEL, download=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(found.folder, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(found.folder)
    gen = json.loads((found.folder / "generation_config.json").read_text())
    eos = tuple(gen["eos_token_id"]) if isinstance(gen["eos_token_id"], list) else (gen["eos_token_id"],)
    rt = es.Runtime(model, tok, "cuda", eos, gen.get("pad_token_id", eos[0]))
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    base = {n: m.weight.detach().float().cpu() for n, m in linears}
    sets = eval_sets()
    cache_dir = d / "progress"
    cache_dir.mkdir(parents=True, exist_ok=True)
    changes: dict = {}
    chain = Chain(a.node)
    out_path = cache_dir / "progress.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    def ids(item):
        x = tok.apply_chat_template(item.messages, add_generation_prompt=True, tokenize=True)
        return list(x["input_ids"] if hasattr(x, "keys") else x)

    def change_of(job_id: str) -> es2.Change:
        """A finished job's change: from disk, or rebuilt from its parents and its entries (then kept)."""
        if job_id in changes:
            return changes[job_id]
        f = cache_dir / f"{job_id}.pt"
        ch = es2.Change(linears, "cpu")  # changes live in RAM: the miners have the GPU
        if f.exists():
            for n, v in torch.load(f).items():
                ch.d[n].copy_(v)
        else:
            spec = chain.specs[job_id]
            parents = [(w, change_of(j)) for j, w, _ in spec.start]
            ch = es2.rebuild(linears, chain.entries.get(job_id, []), parents, "cpu")
            torch.save({n: v.to(torch.bfloat16).cpu() for n, v in ch.d.items()}, f)
        changes[job_id] = ch
        return ch

    def score(label: str, ch: es2.Change | None) -> dict:
        if ch is None:
            for n, m in linears:
                m.weight.data.copy_(base[n])
        else:
            ch.load_into(linears, base)
        res = {}
        with torch.no_grad():
            for name, items in sets.items():
                rows = [ids(i) for i in items]
                right = 0
                for lo in range(0, len(rows), 256):
                    got = es.generate(rt, rows[lo:lo + 256], 64)
                    right += sum(i.check(es.text_of(rt, x)) for i, x in zip(items[lo:lo + 256], got))
                res[name] = round(right / len(items), 4)
        print(f"{label:28s} " + "  ".join(f"{k} {v:.1%}" for k, v in res.items()), flush=True)
        return res

    if "base" not in results:
        results["base"] = score("base model", None)
        out_path.write_text(json.dumps(results, indent=1))
    while True:
        chain.poll()
        cyc = json.loads((d / "evo2.json").read_text()) if (d / "evo2.json").exists() else {"cycle": 1, "jobs": []}
        # Cycle c's jobs are those named evo2-c{c}-*; a cycle is finished once the next one has started.
        done_cycles = {}
        for job_id in chain.specs:
            if job_id.startswith("evo2-c"):
                done_cycles.setdefault(int(job_id.split("-")[1][1:]), []).append(job_id)
        for c in sorted(done_cycles):
            label = f"cycle {c} average"
            if label in results or c >= cyc["cycle"]:
                continue  # scored already, or not finished
            jobs = sorted(done_cycles[c])
            mean = es2.Change(linears, "cpu")
            for j in jobs:
                mean.add(change_of(j), 1 / len(jobs))
            results[label] = {"jobs": jobs, "scores": score(label, mean)}
            out_path.write_text(json.dumps(results, indent=1))
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
