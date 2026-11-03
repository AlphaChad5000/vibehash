"""Train a model by evolution strategies on a local testnet (es.py): a node and two workers in this process sharing
one GPU, then the state rebuilt from the chain alone and the model scored on held-out Countdown problems.

    python scripts/es_local_run.py MODEL_DIR OUT_DIR [--preset mlp --layer 23 --sigma 1e-3 --optimizer sgd ...]

MODEL_DIR is a Hugging Face folder (config.json, model.safetensors, tokenizer files, generation_config.json); the run
pins it as Qwen2.5-0.5B-Instruct's commit 7ae5576 unless --base says otherwise. Everything goes under OUT_DIR, which is
also the run's VIBEHASH_HOME, so nothing touches ~/.vibehash. OUT_DIR/results.json has the numbers: held-out pass@1
before and after by level, each round's level, accepted pairs, mean reward and seconds, the rebuild (its time, and
whether it reached the chain's params_hash), generation speed, and held-out level 0 after every round.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("model", help="the base model's folder")
p.add_argument("out", help="where the run's node, workers and results go")
p.add_argument("--base", default="hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775")
p.add_argument("--preset", default="mlp", choices=["mlp", "norms+biases", "attn-out"])
p.add_argument("--layer", type=int, default=23, help="the mlp preset's layer")
p.add_argument("--sigma", type=float, default=1e-3)
p.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum", "adam"])
p.add_argument("--alpha", type=float, default=None, help="default: the paper's step, sigma**2 / 2, for sgd")
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--level", type=int, default=2)
p.add_argument("--retarget", type=int, default=2, help="move the level every K rounds toward 0.5 solved (0: never)")
p.add_argument("--pairs", type=int, default=8)
p.add_argument("--problems", type=int, default=32)
p.add_argument("--max-tokens", type=int, default=64)
p.add_argument("--reward", default="binary", choices=["binary", "partial"])
p.add_argument("--device", default="cuda")
a = p.parse_args()
OUT = Path(a.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["VIBEHASH_HOME"] = str(OUT / "home")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from vibehash import es  # noqa: E402
from vibehash.client import NodeClient, WorkerClient  # noqa: E402
from vibehash.coordinator import CoordinatorConfig  # noqa: E402
from vibehash.crypto import KeyPair  # noqa: E402
from vibehash.ledger import Block  # noqa: E402
from vibehash.node import Node, PhaseTimes  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE, TrainingJob  # noqa: E402

logging.basicConfig(filename=OUT / "run.log", level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
EVAL_LEVELS = (0, 1, 2, 6)
results = {"argv": sys.argv[1:]}


def save():
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


gen_stats = []
_generate = es.generate


def timed_generate(rt, rows, max_tokens):
    """es.generate, timed: its answers, tokens and seconds (the GPU synchronised either side)."""
    if rt.device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = _generate(rt, rows, max_tokens)
    if rt.device.startswith("cuda"):
        torch.cuda.synchronize()
    gen_stats.append({"rows": len(rows), "tokens": sum(len(x) for x in out), "s": time.perf_counter() - t})
    return out


es.generate = timed_generate

model = AutoModelForCausalLM.from_pretrained(a.model)
pats = es.preset(a.preset, a.layer)
n, sha = es.tensors_digest(es.select(((name, q.shape) for name, q in model.named_parameters()), pats))
del model
c = AutoConfig.from_pretrained(a.model)
# Stands in for hub's manifest of the pinned commit: the folder is registered under it (es.register).
manifest = hashlib.sha256(f"es local run {a.base}".encode()).hexdigest()
es.register(manifest, a.model)
es.cache_noise(2000)  # the node and both workers share one process: each pair's noise is made once a round
spec = es.EsSpec(base=a.base, manifest=manifest,
                 dims={"family": c.model_type, "vocab": c.vocab_size, "hidden": c.hidden_size,
                       "layers": c.num_hidden_layers, "heads": c.num_attention_heads, "kv_heads": c.num_key_value_heads,
                       "head_dim": c.hidden_size // c.num_attention_heads, "ffn": c.intermediate_size,
                       "tied": bool(c.tie_word_embeddings)},
                 tensors=pats, n=n, tensors_sha=sha, sigma=a.sigma,
                 alpha=a.alpha if a.alpha is not None else a.sigma * a.sigma / 2, pairs=a.pairs, pairs_per_unit=2,
                 problems=a.problems, max_tokens=a.max_tokens, reward=a.reward, level=a.level,
                 retarget=(a.retarget, 0.5, 0.1), optimizer=a.optimizer)
results["spec"] = spec.to_json()
print(f"{a.preset}: {n:,} numbers perturbed, sigma {spec.sigma}, {spec.optimizer} alpha {spec.alpha}", flush=True)
rt = es.runtime(spec, a.device)
results["before"] = {lv: es.evaluate(rt, spec, np.zeros(spec.n, np.float32), es.heldout(200, lv)) for lv in EVAL_LEVELS}
print("held-out pass@1 before, by level:", results["before"], flush=True)
save()

VIBE = SYN_PER_VIBE
node_keys, req = KeyPair.generate(b"es local node"), KeyPair.generate(b"es local requester")
workers = [KeyPair.generate(f"es local worker {i}".encode()) for i in range(2)]
genesis = {"chain_id": "vibehash-es-local", "timestamp": 0, "allocations": {req.address: 10_000 * VIBE},
           "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True}
data = Path(tempfile.mkdtemp(prefix="es-node-", dir=OUT))
# The node's own phases are short (a phase is never shorter than them); the job's compute_s is the longest one gets.
node = Node(data / "node", genesis, node_keys, port=0,
            times=PhaseTimes(register_s=2.0, compute_s=5.0, reveal_s=3.0, verify_s=5.0, idle_s=0.3),
            config=CoordinatorConfig(newcomer_units=4), max_job_bytes=512 * 2**20).start()
api = NodeClient(node.url)
clients = [WorkerClient(node.url, k, memory_mb=12000, flops_override=2e12, poll_s=0.2, device=a.device,
                        cache_dir=data / f"cache{i}") for i, k in enumerate(workers)]
job = TrainingJob.from_spec_json(json.dumps({"id": "es-local", "fee_per_round": 1 * VIBE, **spec.to_json(), "init_seed": 0,
                                             "units_per_round": spec.units, "memory_mb": 4000, "compute_s": 300}),
                                 req.address, 1000 * VIBE)
api.send_tx(req, "post_job", "system", 1000 * VIBE, job.spec_json())
stop = threading.Event()
threads = [threading.Thread(target=w.run, args=(stop,), daemon=True) for w in clients]
t0 = time.time()
for th in threads:
    th.start()
seen = 0
while node.chain.tip.height < a.rounds + 1 and time.time() - t0 < 3 * 3600:
    time.sleep(1)
    if node.chain.tip.height != seen:
        seen = node.chain.tip.height
        print(f"  block {seen} at {time.time() - t0:.0f} s", flush=True)
stop.set()
for th in threads:
    th.join(timeout=120)
results["wall_s"] = time.time() - t0
blocks = [Block.from_dict(d) for d in api.get(f"/blocks?from=0&to={node.chain.tip.height}")]
records = {b.height: api.get(f"/round/{b.height}/record") for b in blocks if b.summary is not None}
node_state = node.coord.params[job.job_id].copy()
node.stop()
mine = [b for b in blocks if b.summary is not None and b.summary.job_id == job.job_id]
results["curve"] = [{"height": b.height, "level": records[b.height]["es"]["level"],
                     "pairs": len(records[b.height]["es"]["update"]),
                     "mean_reward": (records[b.height].get("score") or [None, None])[1],
                     "round_s": None if i == 0 else b.timestamp - mine[i - 1].timestamp} for i, b in enumerate(mine)]
save()

es.cache_noise(0)  # a fresh node's rebuild: no noise kept from the run
t = time.perf_counter()
state, rounds = es.rebuild_chain(blocks, records.__getitem__, job.job_id)
results["rebuild_s"] = time.perf_counter() - t
results["rebuild"] = {"rounds": rounds, "hash": es.state_hash(state), "chain_hash": mine[-1].summary.params_hash,
                      "matches": es.state_hash(state) == mine[-1].summary.params_hash,
                      "node_state_equal": bool(np.array_equal(state, node_state))}
print("rebuilt from the chain alone:", results["rebuild"], f"in {results['rebuild_s']:.0f} s", flush=True)
results["workers"] = [{"rebuilt": w.es.rebuilt, "fetched": w.es.fetched} for w in clients]
delta = es.delta_of(spec, state)
results["after"] = {lv: es.evaluate(rt, spec, delta, es.heldout(200, lv)) for lv in EVAL_LEVELS}
print("held-out pass@1 after, by level:", results["after"], flush=True)
s, along = spec.init_params(), []
for b in mine:
    h = records[b.height]["es"]
    s = es.step(spec, s, int(h["seed"], 16), h["update"])
    along.append(es.evaluate(rt, spec, es.delta_of(spec, s), es.heldout(200, 0)))
results["heldout_level0_by_round"] = along
g = [x for x in gen_stats if x["rows"] >= 64]
results["generation"] = {"answers": sum(x["rows"] for x in g), "tokens": sum(x["tokens"] for x in g),
                         "s": sum(x["s"] for x in g)}
save()
print(json.dumps({k: results[k] for k in ("before", "after", "rebuild", "rebuild_s", "generation")}), flush=True)
os._exit(0)  # (worker threads may still hold the GPU)
