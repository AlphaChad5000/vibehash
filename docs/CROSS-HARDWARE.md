# Honest work on any hardware: checking Evo across machine kinds

November 2026. The plan, and what's built of it, for one goal: **no honest worker is turned down because its machine
rounds differently from its checker's.** It builds on docs/GPUS.md (what each GPU family does) and
docs/evo-margin-measurement.md (measuring the gaps).

## The problem

An Evo checker teacher-forces another worker's answers through the same twisted model and finds, for every token,
how far it is below the checker's own greedy choice: a gap, in logits. A claim passes when every gap is within the
job's `margin`. Two machines with the same arithmetic get the same bits, so their gaps are 0. Others don't:

- bf16 on CUDA and ROCm, float32 on Apple, Intel and the CPU (`es_client.runtime2`): up to 0.375 apart, measured;
- cuBLAS promises equal results only on the same architecture with the same number of SMs, and the same toolkit;
- a PyTorch default (bf16 split-K rounding) alone moves gaps by 0.25 on one card.

A cheat (answers its model didn't write) gives gaps of 3.5 or more. So there's plenty of room between honest and
cheating, but a single margin for every pair doesn't use it. On `vibehash-evo-1` (margin 0.1) an RTX 4070 SUPER
rejected a RunPod pod's honest work for gaps of 0.125-0.375 (HANDOFF.md, 2026-11-06). In a rule-2 round one
rejection makes a claim UNCHECKED: unpaid and out of the update, with no referee (`Coordinator._settle_es2`), so the
honest worker loses the round and nothing tells that apart from a checker turning work down on purpose.

## The plan

| | What | Rule change? | Status |
|---|---|---|---|
| A | Each worker registers its **number setup** (`hardware.numerics`): backend, Evo dtype, arch, SM/CU count, PyTorch and CUDA/ROCm version | no | **built** |
| G | Checkers send their **worst gap** with each verdict the gaps alone decided; the node counts them by pair of setups at `/network` (`gaps`) | no | **built** |
| F1 | AMD: matmuls on rocBLAS, not hipBLASLt (deterministic mode covers it; GPUS.md risk 3) | no | **built** |
| F2 | macOS 14: `install-gpu` installs PyTorch 2.14, which still uses the GPU there (GPUS.md risk 12) | no | **built** |
| C | **Pair checkers by setup**: each claim's first checker, when there is one, shares the worker's setup | yes | next |
| D | **Margin by pair**: `margin` for the same setup (tight: their gaps are 0), `cross_margin` across setups | yes | next |
| E | **A matching machine referees**: a cross-setup rejection on gaps alone, under the cheat floor, is overruled when a same-setup checker approved | yes | next |
| F3 | Evo without bf16 split-K rounding (GPUS.md risk 2), as a spec field, after a two-card measurement | yes | after |
| F4 | An Evo self-test: a card that can't reproduce its own answers exactly (gap 0) doesn't take Evo work (GPUS.md risk 13) | no | after |
| H | An integer rule: int8 weights and activations, int32 sums, fixed-point nonlinearities. Bit-identical on every GPU, CPU, Mac and NPU, so the margin is 0 | yes | research |

C, D and E come together, behind one spec field (`cross_margin`), so jobs posted before it run exactly as they did,
and a node or miner without the code refuses a spec with it (as with `loop_stop`): nodes and miners upgrade before a
job uses it.

### Why pairing by setup is safe

A worker names its own setup, and could lie to be paired with a friend. That buys little:

- only the first checker is chosen by setup; the second is drawn from everyone, as now;
- a rejection with a gap over the cheat floor, or for anything but gaps (tokens that don't decode to the text, an
  answer that doesn't end right, a replay log-probability off), always stands;
- so a liar's friend can wave through only answers within the floor of the greedy choice: near-greedy work, which
  pays no better than honest work (workers are paid by accepted FLOPs, not by rewards);
- a worker that names a setup it doesn't have is held to that setup's tight margin, and gets its own work rejected.

### The census (G)

`GET /network` has `gaps`: `{"checks", "pairs": [{"worker", "checker", "same", "checks", "max", "p99", "near",
"over"}]}`, per pair of setups ("cuda bf16 sm_89 x24 cuda13.2 torch2.15.0"), worst first. `near` counts gaps of half
their job's margin or more, and `over` those over it. It's the node's own view of the verdicts sent to it, for a day,
and names no worker. A pair whose `p99` creeps toward the margin is the warning that a driver, PyTorch or card rounds
differently, before honest work is turned down. The same numbers decide `cross_margin` for the next job.

## NPUs

NPUs (Intel AI Boost, AMD XDNA, Qualcomm Hexagon) run int8 and fp16 through each vendor's own compiler (OpenVINO,
Ryzen AI, QNN), not PyTorch, with fixed shapes. fp16 can't meet a classic job's 1e-4, and Qwen3.5's recurrent state
needs float32, so an NPU is unlikely to land inside a bf16 checker's margin as it is. The steps:

1. On an NPU laptop, mine on its integrated GPU first (Intel Arc and AMD Radeon already work), and have it check a
   CUDA worker's dump (`scripts/evo_margin_xgpu.py check --device xpu`): a measurement GPUS.md doesn't have yet.
2. Add an NPU answering path to `evo_margin_xgpu.py`: the twist is low-rank (`spec.rank`), so one compiled model takes
   every member's factors as inputs. Measure its gaps against a CUDA checker.
3. If its worst gap is under half the cross margin, an `npu` setup (`hardware.BACKENDS` has it) that answers but
   doesn't check, until the census backs it.
4. If not, the integer rule (H) is where NPUs fit: int8 with int32 sums is what they're built for, and exact.
