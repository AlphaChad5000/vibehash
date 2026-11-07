"""Evo's logit margin across GPU kinds: does a checker on one card accept honest answers from another, and still
reject skipped work? (PR #43 set the margin to 0.5 from an RTX 4070 SUPER checking RTX 4090 answers; this measures
any pair.) docs/evo-margin-measurement.md is the plan this serves.

Three steps, each on whatever machine it names:

  answer   on machine A: the twisted models answer a fixed round's problems, exactly as a worker does
           (es_fast.answer_members), and the answers go to a dump file (gzip JSON, a few MB).
  check    on machine B (and on A itself, the baseline): every member's answers teacher-forced exactly as a checker
           does (es_fast.gaps_items), honest ones and twin-swapped ones, and the worst gap below the greedy logit
           for each (member, problem) goes to a result file.
  report   anywhere: every result file in one table, honest gaps (max, p99) and, at each margin, false rejections
           of honest members and false acceptances of cheating ones.

  python scripts/evo_margin_xgpu.py answer --out answers-4060.json.gz
  python scripts/evo_margin_xgpu.py check answers-4060.json.gz --out check-4060-by-2060.json
  python scripts/evo_margin_xgpu.py report check-*.json

What a "member" is: one twisted model (pair j, sign +1 or -1), answering every problem of the round. A checker
rejects a member when its worst token, over all of the member's answers, sits more than `margin` below the greedy
choice; so the decision, and the counts below, are per member, as the network's are (es_client.Es2Work.check).

The cheat measured is the one PR #43 measured, the one that pays: skipping the work by claiming the twin's answers
(member (j, +1) commits what (j, -1) answered, and the other way round). A twin pair whose answers are identical
on every problem can't be told apart by any margin and isn't skipped work anyone gains from; those are counted
apart ("identical twins"), not as cheats.

The model is the job's base with no training applied (the line's change is zero): the live job's weights drift a
few percent from it, which moves where the logits sit but not how two GPUs round them. Pin the base's commit with
--model REPO@COMMIT on every machine; `check` refuses a dump made from another commit.

Needs: this repository installed with the evo extra (pip install -e .[evo]), PyTorch with CUDA (vibehash
install-gpu), and for the default task mix the `datasets` library (the problems come from Hugging Face datasets).
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # this checkout's vibehash, not an installed one

MARGINS = (0.1, 0.375, 0.5, 0.75, 1.0)
DEFAULT_SEED = "5eed0e7a1c3b2d4f"  # any fixed round seed: it draws the problems and every pair's noise
FORMAT = "vibehash-evo-margin/1"


def _gpu_facts(device: str) -> dict:
    import torch
    out = {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
           "os": platform.platform(), "hip": getattr(torch.version, "hip", None)}  # (hip: an AMD card's ROCm build)
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        from vibehash import __version__
        out["vibehash"] = __version__
    except ImportError:
        pass
    # transformers reads the shared prompt through Qwen3.5's linear-attention layers with these kernels when they're
    # installed, and with its reference PyTorch code when not: either may round differently, so it's recorded.
    from importlib.util import find_spec
    out["kernels"] = {k: find_spec(k) is not None for k in ("causal_conv1d", "fla")}
    if device.startswith("cuda"):
        i = torch.device(device).index or 0
        major, minor = torch.cuda.get_device_capability(i)
        try:  # Turing (sm_75) has no native bf16: PyTorch emulates it, which is part of what this measures
            native = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # an older PyTorch without the argument
            native = torch.cuda.is_bf16_supported()
        out.update(gpu=torch.cuda.get_device_name(i), capability=f"sm_{major}{minor}",
                   memory_gb=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                   bf16_native=bool(native))
    else:
        out["gpu"] = platform.processor() or "cpu"
    return out


def _label(facts: dict) -> str:
    return f"{facts.get('gpu', '?')} ({facts.get('capability', facts.get('device', '?'))})"


def _setup(model: str, tasks: tuple, pairs: int, problems: int, max_tokens: int, device: str):
    """The spec, runtime and twist factors a worker or checker would have for this job, built off the chain."""
    from vibehash import compute, es_client, es_local
    if device != "cpu":
        # The worker's PyTorch settings (compute.get_backend makes one): TF32 off and deterministic kernels, which
        # also keeps Hopper and Blackwell's attention off cuDNN's kernel, as on a worker
        compute.TorchBackend(device)
    found = es_local.find(model)
    es_local.register(found)
    per_unit = min(8, pairs)
    spec = es_local.spec2_for(found, tasks[0], tasks=tasks, mix=len(tasks) > 1, max_tokens=max_tokens, pairs=pairs,
                              pairs_per_unit=per_unit, problems=problems)
    rt = es_client.runtime2(spec, device)
    return found, spec, rt


def _rows(rt, spec, seed: int) -> list[list[int]]:
    from vibehash import es2_rounds, es_client
    return es_client.prompt_rows(rt, es2_rounds.items(spec, seed, 0))


def _members(spec) -> list[tuple[int, int]]:
    return [(j, s) for j in range(spec.pairs) for s in (1, -1)]


# --- answer ----------------------------------------------------------------------------------------------------------

def cmd_answer(a) -> None:
    import torch
    from vibehash import es_client, es_fast, es_tasks
    tasks = tuple(es_tasks.VIBE) if a.tasks == "vibe" else tuple(t.strip() for t in a.tasks.split(","))
    found, spec, rt = _setup(a.model, tasks, a.pairs, a.problems, a.max_tokens, a.device)
    facts = _gpu_facts(a.device)
    print(f"answering on {_label(facts)}: {spec.pairs * 2} members x {spec.problems} problems, up to {spec.max_tokens} "
          f"tokens, model {found.base}", flush=True)
    rounds = []
    for seed_hex in a.seed:
        seed = int(seed_hex, 16)
        rows = _rows(rt, spec, seed)
        fac = es_client.round_factors(rt, spec, seed)
        members = _members(spec)
        t0 = time.perf_counter()
        with torch.no_grad():
            got = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank,
                                         max(512, len(rows)))
        took = time.perf_counter() - t0
        lengths = [len(x) for m in got for x in m]
        print(f"  seed {seed_hex}: {len(members)} members answered in {took:.0f} s (answers {min(lengths)}-"
              f"{max(lengths)} tokens, mean {sum(lengths) / len(lengths):.0f})", flush=True)
        rounds.append({"seed": seed_hex, "rows": rows, "members": members,
                       "answers": [[list(map(int, x)) for x in m] for m in got], "answer_s": round(took, 1)})
    dump = {"format": FORMAT, "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "answered_on": facts,
            "model": found.base, "tasks": list(tasks), "pairs": spec.pairs, "problems": spec.problems,
            "max_tokens": spec.max_tokens, "sigma": spec.sigma, "rounds": rounds}
    out = Path(a.out)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(dump, f)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB). Next, on each checking machine (this one too):\n"
          f"  python scripts/evo_margin_xgpu.py check {out.name} --out check-<answered>-by-<checker>.json")


# --- check -----------------------------------------------------------------------------------------------------------

def _gaps(rt, spec, fac, member, rows, answers) -> list[float]:
    """The checker's numbers for one member (es_client.Es2Work.check): its worst gap on each of its answers."""
    from vibehash import es_fast
    g = es_fast.gaps_items(rt, rt.linears, fac, [tuple(member)], list(zip(rows, answers)), spec.sigma, spec.rank,
                           max_tokens=spec.max_tokens)
    return [round(float(x), 6) for x in g[0].tolist()]


def cmd_check(a) -> None:
    import torch
    from vibehash import es_client
    with gzip.open(a.dump, "rt", encoding="utf-8") as f:
        dump = json.load(f)
    if dump.get("format") != FORMAT:
        sys.exit(f"{a.dump} isn't an answers dump from this script")
    model = dump["model"].removeprefix("hf:")
    found, spec, rt = _setup(model, tuple(dump["tasks"]), dump["pairs"], dump["problems"], dump["max_tokens"],
                             a.device)
    if found.base != dump["model"]:
        sys.exit(f"this machine found {found.base}, the dump was answered with {dump['model']}")
    facts = _gpu_facts(a.device)
    print(f"checking answers from {_label(dump['answered_on'])} on {_label(facts)}", flush=True)
    rounds = []
    for r in dump["rounds"]:
        seed = int(r["seed"], 16)
        rows = r["rows"]
        if _rows(rt, spec, seed) != rows:
            print(f"  note: seed {r['seed']}'s prompts come out differently here (another tokenizer or dataset "
                  "version?); checking against the dump's prompts", flush=True)
        fac = es_client.round_factors(rt, spec, seed)
        members, answers = [tuple(m) for m in r["members"]], r["answers"]
        twin = {m: i for i, m in enumerate(members)}
        honest, cheat, identical = [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i, (j, s) in enumerate(members):
                print(f"\r  seed {r['seed']}: member {i + 1}/{len(members)}", end="", flush=True)
                honest.append(_gaps(rt, spec, fac, (j, s), rows, answers[i]))
                k = twin.get((j, -s))
                if k is None:
                    continue
                if answers[k] == answers[i]:  # the twin answered the same everywhere: no cheat to catch
                    identical.append([j, s])
                    continue
                cheat.append({"member": [j, s], "claims": [j, -s], "gaps": _gaps(rt, spec, fac, (j, s), rows,
                                                                                  answers[k])})
        print(f"\r  seed {r['seed']}: {len(members)} members, {len(cheat)} twin swaps, {len(identical)} identical "
              f"twins, in {time.perf_counter() - t0:.0f} s", flush=True)
        rounds.append({"seed": r["seed"], "members": [list(m) for m in members], "honest": honest, "cheat": cheat,
                       "identical_twins": identical})
    res = {"format": FORMAT, "answered_on": dump["answered_on"], "checked_on": facts, "model": dump["model"],
           "max_tokens": dump["max_tokens"], "rounds": rounds}
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print(f"wrote {a.out}")
    _print_table([res])


# --- report ----------------------------------------------------------------------------------------------------------

def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(res: dict) -> dict:
    honest_member = [max(g) for r in res["rounds"] for g in r["honest"] if g]
    honest_answer = [x for r in res["rounds"] for g in r["honest"] for x in g]
    cheat_member = [max(c["gaps"]) for r in res["rounds"] for c in r["cheat"] if c["gaps"]]
    return {"pair": f"{_label(res['answered_on'])} -> {_label(res['checked_on'])}",
            "members": len(honest_member), "cheats": len(cheat_member),
            "identical": sum(len(r["identical_twins"]) for r in res["rounds"]),
            "max": max(honest_member, default=float("nan")), "p99_member": _pct(honest_member, 0.99),
            "p99_answer": _pct(honest_answer, 0.99), "cheat_min": min(cheat_member, default=float("nan")),
            "false_reject": {m: sum(x > m for x in honest_member) for m in MARGINS},
            "false_accept": {m: sum(x <= m for x in cheat_member) for m in MARGINS}}


def _print_table(results: list[dict]) -> None:
    rows = [summarise(r) for r in results]
    ms = " / ".join(f"{m:g}" for m in MARGINS)
    print("\n| answered on -> checked on | honest members | worst gap | p99 (member) | p99 (answer) | false rejections "
          f"at {ms} | cheats | smallest cheat gap | false acceptances at {ms} | identical twins |")
    print("|---" * 10 + "|")
    for s in rows:
        fr = " / ".join(str(s["false_reject"][m]) for m in MARGINS)
        fa = " / ".join(str(s["false_accept"][m]) for m in MARGINS)
        print(f"| {s['pair']} | {s['members']} | {s['max']:.4f} | {s['p99_member']:.4f} | {s['p99_answer']:.4f} | {fr} "
              f"| {s['cheats']} | {s['cheat_min']:.4f} | {fa} | {s['identical']} |")


def cmd_report(a) -> None:
    results = []
    for p in a.results:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("format") != FORMAT:
            sys.exit(f"{p} isn't a result file from this script")
        results.append(d)
    _print_table(results)
    every = [summarise(r) for r in results]
    worst, least = max(s["max"] for s in every), min(s["cheat_min"] for s in every)
    print(f"\nAcross all {len(every)} pairs: worst honest gap {worst:.4f}, smallest cheat gap {least:.4f}.")
    if worst < least:
        print(f"Any margin from {worst:.4f} up to (not including) {least:.4f} rejects no honest member and accepts no "
              "cheat in any pair measured.")
    else:
        print("No margin separates them: some honest member sits further below the greedy choice than some cheat.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("answer", help="answer a fixed round on this GPU and dump the answers")
    an.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="REPO or REPO@COMMIT (pin the commit everywhere)")
    an.add_argument("--tasks", default="vibe", help="'vibe' (the live job's 16-task mix) or a comma-separated list")
    an.add_argument("--pairs", type=int, default=16, help="antithetic pairs (members = 2 x pairs; default 16)")
    an.add_argument("--problems", type=int, default=16)
    an.add_argument("--max-tokens", type=int, default=512)
    an.add_argument("--seed", nargs="+", default=[DEFAULT_SEED], help="round seeds in hex (each one a round)")
    an.add_argument("--device", default="cuda", help="cuda, cuda:1, ... (or cpu, slowly)")
    an.add_argument("--out", required=True, help="the dump to write (.json.gz)")
    ck = sub.add_parser("check", help="teacher-force a dump's answers on this GPU, honest and twin-swapped")
    ck.add_argument("dump")
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--out", required=True, help="the result to write (.json)")
    rp = sub.add_parser("report", help="one table from several result files")
    rp.add_argument("results", nargs="+")
    a = p.parse_args(argv)
    if a.cmd != "report":
        from vibehash.cli import pin_gpu
        pin_gpu(a)  # --device cuda:1 as the worker takes it: this process sees that GPU alone, as "cuda"
    {"answer": cmd_answer, "check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
