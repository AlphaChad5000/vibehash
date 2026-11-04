"""An Evo Hash job's model as a folder anyone can load (`vibehash evo export`, and `job download` for an Evo job).

A rule-2 Evo job keeps no weights anywhere: its blocks hold each round's seed and how its twists scored, and the model
is a function of them (es2.py). Exporting works it out the way a worker does, with the workers' own code
(es_client.Es2Work._line): every block of the job and of its parents read from the node, every round's record checked
against the hash its block commits to and every state against the block's params_hash, the job's change from the base
built round by round (or carried on from a line a worker on this computer saved to disk). Then base + change goes into
the base model, and the model is saved in Hugging Face's layout (safetensors, its tokenizer and chat template), with an
EVO.md and evo.json saying which chain, job and block it is.

The weights are saved in bfloat16 by default: what a worker's model holds on a GPU when it answers a round (the base's
own dtype too). float32 keeps the change unrounded, as es2.canonical_weights would give it to about 1e-5.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from . import es, es2, es_client, es_local
from .tasks import TrainingJob

LOCAL_NODE = "http://127.0.0.1:8545"  # `vibehash testnet --evo2`'s node: Evo jobs run on local networks for now


def job_of(node, job_id: str) -> TrainingJob:
    info = node.get(f"/job/{job_id}")
    return TrainingJob.from_spec_json(json.dumps(info["spec"]), info["owner"], 0)


def is_evo2(job: TrainingJob) -> bool:
    return isinstance(job.spec, es2.Spec2)


def newest_job(node) -> str | None:
    """The newest rule-2 Evo job with a round that learned something (accepted units), from the chain's tip down."""
    tip = int(node.status()["height"])
    seen: dict[str, bool] = {}
    hi = tip
    while hi >= 1:
        lo = max(1, hi - 99)
        for d in reversed(node.get(f"/blocks?from={lo}&to={hi}") or []):
            s = d.get("summary")
            if not s or not s.get("accepted_units"):
                continue
            j = s["job_id"]
            if j not in seen:
                try:
                    seen[j] = is_evo2(job_of(node, j))
                except Exception:  # noqa: BLE001 - a job the node no longer describes, or not one we can read
                    seen[j] = False
            if seen[j]:
                return j
        hi = lo - 1
    return None


def _lineage(node, spec, seen: set) -> list[dict]:
    """The job's ancestors, nearest first: each parent's id, weight in its child's start, and task."""
    out = []
    for pj, w, _ in spec.start:
        if pj in seen:
            continue
        seen.add(pj)
        try:
            ps = job_of(node, pj).spec
        except Exception:  # noqa: BLE001 - named in the spec but not described by the node: listed without a task
            out.append({"job": pj, "weight": w, "task": None})
            continue
        out.append({"job": pj, "weight": w, "task": ps.task})
        out += _lineage(node, ps, seen)
    return out


def export(node, job_id: str, out: Path, device: str | None = None, dtype: str = "bfloat16") -> dict:
    """Job `job_id`'s model, worked out from the chain up to the node's tip, saved to the folder `out` (made, or
    emptied of an earlier export's files). Returns what EVO.md and evo.json say."""
    import torch

    job = job_of(node, job_id)
    if not is_evo2(job):
        raise ValueError(f"{job_id} isn't an Evo Hash job (testnet --evo2's)" + (
            ": the first Evo mode's jobs (testnet --evo) can't be exported yet" if job.es else ""))
    spec = job.spec
    if not es_local.find_base(spec):
        raise FileNotFoundError(f"{spec.repo} at commit {spec.commit[:12]} (the job's base) isn't in this computer's "
                                f"Hugging Face cache: export on a computer that mined the job, or fetch it first with "
                                f"`huggingface-cli download {spec.repo} --revision {spec.commit}`")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / "evo.json").exists():
        raise FileExistsError(f"{out} has files in it that aren't an earlier export: pick another --out")

    t0 = time.monotonic()
    rt = es_client.runtime2(spec, device)
    client = SimpleNamespace(home=node, node=node, jobs={job_id: job})
    v2 = es_client.Es2Work(SimpleNamespace(client=client, rebuilt=0, device=lambda: device))
    ln = v2._line(rt, job_id, spec)  # every round checked against its block; a saved line used if it's the chain's
    with torch.no_grad():
        v2.load(rt, ln)
    want = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    was = next(rt.model.parameters()).dtype
    try:
        rt.model.to(want)  # in place, so tied weights (Qwen's embeddings and head) stay one tensor in the file
        rt.model.save_pretrained(tmp, safe_serialization=True)
    finally:
        rt.model.to(was)
        rt.loaded = None  # (rounded through `want`: the next load copies base + change in again)
    rt.tokenizer.save_pretrained(tmp)
    base = Path(es.LOCAL[spec.manifest])
    if (base / "generation_config.json").is_file():  # the base's own stop tokens and sampling defaults
        shutil.copyfile(base / "generation_config.json", tmp / "generation_config.json")
    cfg_path = tmp / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["dtype" if "dtype" in cfg else "torch_dtype"] = dtype  # (newer transformers name it dtype)
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    st = node.status()
    info = {
        "job": job_id, "task": spec.task, "base": spec.base, "chain_id": st.get("chain_id"),
        "genesis_hash": st.get("genesis_hash"), "node": getattr(node, "url", None),
        "block": ln.last_round, "params_hash": ln.params_hash, "rounds_learned": len(ln.history),
        "lineage": _lineage(node, spec, {job_id}), "dtype": dtype, "device": device,
        "seconds": round(time.monotonic() - t0, 1), "exported": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (tmp / "evo.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (tmp / "EVO.md").write_text(_report(info, spec), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    return info


def _report(info: dict, spec) -> str:
    tasks = [info["task"]] + [p["task"] for p in info["lineage"] if p["task"]]
    lines = [
        f"# {info['job']}: an Evo Hash model",
        "",
        f"{spec.repo} (commit {spec.commit}) trained by evolution on the chain `{info['chain_id']}`, worked out from "
        f"its blocks up to block {info['block']} and checked against every block's params_hash on the way.",
        "",
        f"- **Job:** {info['job']}, task {info['task']}, {info['rounds_learned']} rounds that changed the weights",
        "- **Started from:** " + (", ".join(f"{pj} (weight {w:g})" for pj, w, _ in spec.start)
                                    or f"the base, {spec.repo}"),
        f"- **Tasks along its line:** {', '.join(dict.fromkeys(tasks))}",
        f"- **State at block {info['block']}:** params_hash `{info['params_hash']}`",
        f"- **Weights:** {info['dtype']} safetensors, the base's tokenizer and chat template",
        "",
        "## Use it",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        'tok = AutoTokenizer.from_pretrained("THIS_FOLDER")',
        'model = AutoModelForCausalLM.from_pretrained("THIS_FOLDER", dtype="auto")',
        'msgs = [{"role": "user", "content": "What is 17 + 4 x 6?"}]',
        "ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors=\"pt\")",
        "print(tok.decode(model.generate(ids, max_new_tokens=64)[0][ids.shape[1]:], skip_special_tokens=True))",
        "```",
        "",
        "It's an ordinary Qwen2 folder, so llama.cpp's convert_hf_to_gguf.py turns it into a GGUF for Ollama or LM Studio.",
        "",
    ]
    return "\n".join(lines)
