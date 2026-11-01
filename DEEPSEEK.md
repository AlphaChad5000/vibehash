# Fine-tuning DeepSeek-V4.1-Flash on Vibe Hash

**Status (2026-11-01): a plan, and its step 1 built.** `vibehash/deepseek.py` is
the model in plain PyTorch, trainable and deterministic, and matches DeepSeek's
own code on a real layer (§7, which also corrects §3's compute estimate and §4's
tolerances). Every other number here is measured from the model itself at
commit `314113f` of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash):
its `config.json`, the headers of all 48 weight files (read with
`vibehash.hub` by byte range, 96,085 tensors, nothing else downloaded), and
DeepSeek's reference code in `inference/`. Where this disagrees with the
estimates in FINETUNING.md and SHARDING.md, this is the measured one.

The goal: a job poster runs one `train --base deepseek-ai/DeepSeek-V4.1-Flash`
command, and teams of ordinary gaming PCs fine-tune it, none of them holding
more than a slice of the model.

## 1. What the model is

| | |
|---|---|
| Parameters | 763.2B (the Hub's count), about 16B used per token |
| Stored | 510.3 GB: FP8 weights and FP4 experts, with power-of-two scales per 32-wide block |
| Licence | MIT, not gated |
| Shape | 40 layers, width 5,120, vocabulary 129,280 |

Where the 510 GB is:

| Part | Size | Notes |
|---|---|---|
| Routed experts | 288.8 GB | 384 per layer (6 used per token), 18.8 MB each in FP4, 7.22 GB per layer |
| Engram tables | 203.1 GB | Two n-gram lookup tables, in layers 1 and 14, 384M rows of 256 FP8 values each |
| MTP and DSpark draft layers | 7.9 GB | Only for faster inference |
| Attention | 5.1 GB | 0.13 GB per layer |
| Embeddings and output head | 2.6 GB | |
| Shared experts | 1.4 GB | One per layer, used by every token |
| Vision tower | 1.0 GB | Only for images |
| Router, hyper-connections, norms | 0.4 GB | |

So a normal layer is 7.4 GB, and layers 1 and 14 are 109 GB because of their
tables. Everything except routed experts and tables, the part every token
touches, is about 9.5 GB.

**It is not the architecture `transformer.py` and `moe.py` implement.** Those
are Llama- and DeepSeek-V3-shaped. V4.1 has:
- **Hyper-connections:** the residual stream is 4 parallel copies of the
  5,120-wide state, mixed before and after each sublayer by coefficients the
  stream itself produces (Sinkhorn-normalised).
- **Attention:** low-rank queries (rank 1,280), a single 512-wide key/value
  head, a grouped low-rank output (8 groups), a 128-token sliding window plus
  compressed keys and values that a sparse indexer picks (top 512 per token),
  attention sinks, and YaRN rotary positions. Later layers reuse what earlier
  "source" layers computed instead of computing their own: compressed keys and
  index keys from layers 2, 8, 14 and 20, and the indexer's picks from those
  and layers 24, 28, 32 and 36.
- **MoE:** 384 routed experts and 1 shared, top 6, scores `sqrt(softplus(x))`,
  a bias that steers selection but not the gates (and a separate bias for
  image tokens), gates scaled by 1.5, SwiGLU clamped at 10.
- **Engram:** each token hashes the 2-, 3- and 4-grams ending at it into 24
  rows of a table, and a gate mixes them into the stream.
- **Extras:** 3 MTP/DSpark draft layers and a vision tower.

DeepSeek's reference code (`inference/model.py`, 62 KB) covers all of it, but
for inference only: `torch.inference_mode`, custom tilelang kernels, and
8-way tensor parallelism on one server. Training needs our own
differentiable version, checked against it. Its `ModelArgs` defaults are a
small 5-layer V4.1 made for running tests.

## 2. How to cut it

Four cuts, so that no worker holds more than a few layers.

**1. Train adapters, not the model (LoRA).** The 510 GB stays frozen. LoRA
adapters at rank 16 on the attention projections (`wq_a`, `wq_b`, `wkv`, the
grouped `wo_a`, `wo_b`) and the shared expert come to about 2M weights a layer,
78M in all: 160 MB in bf16. Routed experts, router, tables and embeddings are
read-only, so they need no gradients and no optimiser state. A worker's
checkpoints are its layers' adapters: about 4 MB a layer.

**2. Layers into pipeline shards.** A team holds the model between them, a few
layers each, as SHARDING.md describes:

| GPU memory | Layers on the GPU | Workers per team |
|---|---|---|
| 12 GB (an RTX 4070 SUPER) | 1 | 40 |
| 24 GB | 2 | 20 |
| 32 GB | 3 | 14 |

A worker with plenty of system RAM can hold more: keep its experts in RAM and
copy each layer's 7.2 GB to the GPU per micro-batch (about 0.3 s over PCIe
4.0, comparable to the layer's compute for a few thousand tokens). A 12 GB card
with 64 GB of RAM could then take about 7 layers.

**3. Engram tables become per-job slices.** Which rows a token reads depends
only on the text, never on what the model computes (`inference/engram.py`
hashes token ids). So a job's pre-flight can hash its whole dataset in
advance, pull out just the rows it touches, and ship them like a dataset.
That's at most 24 rows × 264 bytes = 6.3 KB per token per table layer, and far
less once repeated n-grams are counted once. Whoever builds the slice needs
the two 101 GB tables once (a "table host": the poster, a validator, or a
volunteer with disk). The slice is committed to by a Merkle root over its
rows, and anyone can spot-check a row against Hugging Face with one ranged
read. Workers on layers 1 and 14 then need megabytes, not 101 GB.

**4. Skip what a text fine-tune doesn't use.** The vision tower and the draft
layers (8.9 GB) are never downloaded. The draft layers were trained against
the original model, so speculative decoding may accept fewer drafted tokens
after fine-tuning. The adapter still works.

## 3. The real cost is what crosses between workers

At each layer boundary a token carries the 4-copy residual stream: 20,480
values, 40 KB in bf16 or 20 KB in FP8. Backward sends a gradient of the same
size the other way. Smaller side channels carry the source layers' compressed
keys and values, index keys and picks to the later layers that reuse them.
Measured from the code (§7): compressed keys are 256 values a token at ratio 2,
512 at ratio 1 (FP4 with E4M3 scales, so 144 or 288 bytes), and **their gradient
comes back too**; index keys 64 or 128 FP4 values a token; picks 512 positions a
token per index source (1 KB); routing 6 experts a token a layer.

Compute is small by comparison, because only 6 of 384 experts run. A layer has
about 0.38B active weights, so forward plus backward-to-input is about
1.5 GFLOP a token. (Measured, §7: 0.6–0.75 ms a token a layer on an RTX 4070
SUPER, 8–10× this table's estimate, since deepseek.py dequantises every
expert it uses twice a step and isn't tuned. Two layers, 1.2–1.5 ms a token:
still network-bound at 20 Mbit/s, by 10–25×; at 100 Mbit/s only by 2–5×.)

| Per worker, 2 layers, per token | bf16 edges | FP8 edges |
|---|---|---|
| Compute (at 20 TFLOP/s sustained) | 0.15 ms | 0.15 ms |
| Upload: activations forward plus gradients back | 80 KB | 40 KB |
| … at 20 Mbit/s | 32 ms | 16 ms |
| … at 100 Mbit/s | 6.4 ms | 3.2 ms |

So a V4.1 team is **network-bound by 20 to 200 times**, the opposite of the
dense models in SHARDING.md §4. Sparse compute and a 4× wide residual
stream do that. Consequences:
- **Edges in FP8.** They halve the traffic. The model already runs its
  activations through FP8 with 32-wide block scales. Checkers replay from the
  exact bytes that were committed, so the quantisation is part of the
  protocol, not noise.
- **Speed comes from many teams, not faster teams.** A team does about 30–60
  tokens a second on 20 Mbit/s uploads and 150–300 on 100 Mbit/s. Ten teams at
  150 a second fine-tune on 10M tokens in under 2 hours. Typical instruction
  fine-tunes are 1–50M tokens: hours to a day.
- **Group workers by measured upload** (capacity already tracks delivered
  work), since a team runs at its slowest link.
- **An option: adapters on the top layers only.** Then the lower layers only
  run forward, with no gradients coming back, which halves their traffic.

## 4. Checking it

SHARDING.md's cells carry over: each worker commits to its adapters' checkpoints
and every tensor that crossed its edges, and a checker replays one cell (one
shard, one segment) holding only that shard. What V4.1 adds:
- **Three discrete choices, not one.** MoE's top 6 of 384, the indexer's top 512,
  and the candidate blocks' top 2,048 are all decided by comparing scores, so
  rounding can flip near-ties between GPUs. Each is committed with the edges and
  accepted within a tie margin, as `moe.py` does for routing. Measured (§7), the
  margins are wider than `moe.py`'s 1e-4: given the same layer input, DeepSeek's
  own code on the CPU and on CUDA picks different compressed tokens for 22% of
  tokens (the indexer scores in bf16, so ties are common) and different experts
  for 4%. The worst gap between two implementations was 7e-4 of the best score
  for routing, and half a bf16 step for picks.
- **Frozen weights are exact everywhere.** The scales are powers of two (`ue8m0`),
  so dequantising FP8 and FP4 to bf16 is exact: every GPU computes with the same
  weights. Checked on every quantised tensor of a real layer (§7). Only the
  matmuls and attention need tolerances, measured across GPU vendors as DESIGN
  §8 did for MLPs.
- **The tolerance is about 1e-2, not 1e-4.** The model rounds its activations
  to FP8 (FP4 for compressed keys) before almost every matmul, and that turns a
  last-bit difference into a whole FP8 step. One real layer, same input and same
  choices, run on the CPU and on the GPU differs by 0.8% rms (1.5% for DeepSeek's
  code with its own choices). So a cell replayed on other hardware matches only
  to about 1e-2 per layer, and errors grow layer by layer. Checkers want small
  cells replayed from the committed edge bytes (as planned), and a tolerance
  calibrated per layer and per vendor, not DESIGN §8's 1e-4.
- **Engram rows** are checked against the job's slice root, and the slice against
  Hugging Face by spot-checks.
- **Disputes stay small.** One cell is 2 layers, about 15 GB: a validator can
  replay it on a CPU with enough RAM, in minutes for a segment of a few
  thousand tokens.

## 5. What to build, in order

1. **Done (2026-11-01, §7).** A differentiable V4.1 layer (`deepseek.py`):
   attention with its compressor and indexer, MoE, hyper-connections and engram,
   in plain PyTorch, dequantising FP8 and FP4 on the fly, deterministic. Proven
   against `inference/model.py`:
   - the whole model on the small 5-layer config, from random weights;
   - one real layer from the real weights (7.4 GB, fits the 4070 SUPER).
2. **LoRA on it**, named so the adapter loads in Hugging Face transformers
   (`deepseek_v41`, transformers 5.6) and PEFT.
3. **Loading one shard:** `hub.py` fetches exactly one stage's tensors by range.
   (`describe` now reads the counts nested in `text_config`: done.) Measured
   while fetching layer 2: the Hub's CDN slows one long stream to about
   1.5 MiB/s after a couple of minutes, while fresh requests get 30–37 MiB/s.
   `TensorCache` fetches a run of back-to-back tensors in one request, and a
   layer is one 7.4 GB run, so it should split runs into pieces fetched several
   at once, as `deepseek.fetch_file` does (6 × 64 MiB: about 45–75 MiB/s).
4. **Engram slices:** hash a dataset (with `engram.py`'s compressed token map),
   extract the rows, commit, spot-check.
5. **V4.1 cells in `shards.py`:** the 4-copy stream and side channels as edges,
   committed choices with tie margins, the FP8 edge codec.
   **Milestone:** two processes on one PC train as a 2-stage pipeline, first on
   the small config, then on 2 real layers.
6. **Teams on the network:** worker-to-worker transport with relays, teams formed
   by memory and upload, backup workers per shard (SHARDING.md §6).
7. **The job flow** FINETUNING.md describes, with these numbers in its pre-flight.

Steps 1–4 can be built and tested on one PC. Steps 5–6 need several.

## 6. Open questions

- **Sequence length.** Attention needs a whole sequence inside one layer, so a
  micro-batch is whole sequences. Measured (§7): with a layer's experts on a
  12 GB card, about 1,500 tokens; with them in pinned RAM, 4,096 (activations
  take 1.5 GB per 1,000 tokens). Recomputing each block in the backward would
  cut activations to its inputs, about 40 KB a token.
- **Using the result.** Transformers 5.6 knows `deepseek_v41`; whether vLLM serves
  LoRA adapters for it is unchecked.
- **Image fine-tunes** would add the vision tower and the image routing bias.
- **The network.** A team is 14–40 GPUs. The public network has none yet, so steps
  1–5 come first, and teams come when there are miners to form them.

## 7. Step 1: the model in PyTorch, checked against DeepSeek's

`vibehash/deepseek.py` (2026-11-01) is DeepSeek-V4.1-Flash as a function
autograd can differentiate: plain PyTorch ops, no kernels of its own, prefill
only, text only (no vision tower, no MTP/DSpark draft layers). Its docstring
describes the model. What matters for the plan:

- **Weights stay as stored.** FP8 matrices, FP4 experts and FP8 engram rows are
  frozen buffers in DeepSeek's formats and names, so the safetensors load as
  they are. Each matrix is dequantised inside its matmul and again in the
  backward, never kept, so a 7.4 GB layer runs on a 12 GB card. Experts can
  also stay in pinned CPU memory and be copied over as they're used.
- **The reference's arithmetic, by default.** Activations in bf16, float32
  where it upcasts, and rounded to FP8 or FP4 exactly where its kernels round
  them, with a straight-through gradient. `Numerics(torch.float64,
  quantize=False)` is the same function unrounded, for gradient checks.
  Attention follows the kernel's online softmax, 64 keys a block. A simpler
  softmax differs by under a bf16 step, but the FP8 rounding after attention
  amplified that to 1% rms.
- **Choices are inputs.** Routing, indexer picks and candidate blocks can be
  given (`Choices`); what isn't given is computed and returned, with its scores
  on request. `choice_gap` measures how far a choice is from a top-k.
- **Side channels are explicit** (`Side`): compressed keys (with a gradient),
  index keys and picks, from source layers to the layers after them. A
  `Model` can hold any run of layers, with or without the embedding and head.
- **Downloading**: `fetch_file` and `load_layer` fetch whole weight files of
  the pinned commit into `~/.vibehash/cache/deepseek`, 64 MiB pieces over six
  connections, and check each against the Hub's sha256.

**The reference, and how it ran.** DeepSeek's `inference/model.py` and
`engram.py` at the pinned commit are fetched from the Hub by the tests and
checked against their Git blob ids; they aren't copied into this repository.
They need tilelang kernels. We didn't run those under WSL2, because the
attention kernel at the real head size wants about 200 KB of shared memory per
block and an RTX 4070 SUPER has 99 KB. So the real layer couldn't use them on
this GPU anyway. Instead `tests/deepseek_reference.py` has pure-PyTorch
stand-ins that follow each kernel line by line, written apart from
deepseek.py's versions so the two check each other; activation rounding
matches them bit for bit. Not checked: the stand-ins against tilelang itself,
which needs a Hopper or Blackwell GPU.

**Results** (`tests/test_deepseek.py`; errors relative to the reference):

| | Stream, per layer | Choices the same | End to end |
|---|---|---|---|
| ModelArgs defaults, random weights (CPU, CUDA) | 0.6–2e-3 rms, ≤1.7e-2 max | ≥99.2% routing, 100% picks | logits 5% rms, 7% max |
| Every feature on (engram, candidate blocks, YaRN, clamps) | ≤1.7e-2 max | ≥99.6% routing, 100% picks | logits ≤6% max |
| **Real layer 2, 2,048 README tokens, GPU** | **2.7e-3 rms, 4.6e-3 max** | 99.76% routing, 99.95% picks | |

Per layer means each layer gets the reference's input, side channels and
choices. That's where the proof is: random-weight models are chaotic. A 1e-3
rms nudge after layer 0 moves the final logits by 6% rms with every choice
held, so 5% end to end is at the floor, and a wiring mistake would be O(1). The
yardsticks for the real layer, same input:

| Real layer 2, same input | Stream rms |
|---|---|
| DeepSeek's code, CPU vs GPU (its own choices: 22% of picks and 4% of routing differ) | 1.5e-2 |
| deepseek.py, CPU vs GPU (the same choices) | 8.0e-3 |
| deepseek.py vs DeepSeek's code, same GPU, same choices | 2.7e-3 |

Also proven: dequantising is exact for every quantised tensor of layer 2
(1,161, bf16 equals float32 bit for bit); gradients match finite differences
(a tiny config in float64 with every feature on); forward and backward repeat
bit for bit on the CPU and on CUDA, with PyTorch refusing nondeterministic ops;
the engram bucket primes give the real tables' row counts (384,006,168 and
384,016,682); hashes match the reference's; `Config.from_hf(config.json)` is
`inference/config.json` is `Config.flash()`.

**Speed and memory**, one real layer (layer 2), RTX 4070 SUPER, forward with
autograd plus backward to the input stream and the layer's bf16/float32
weights (`scripts/deepseek_bench.py`):

| Tokens | Experts | Forward s | Backward s | Activations GiB | ms a token |
|---|---|---|---|---|---|
| 512 | on the GPU (6.9 GiB) | 0.44 | 0.32 | 0.87 | 1.5 |
| 1,024 | on the GPU | 0.45 | 0.50 | 1.63 | 0.93 |
| 1,536 | on the GPU | 0.53 | 0.62 | 2.40 | 0.75 |
| 2,048 | in pinned RAM | 0.73 | 0.90 | 3.15 | 0.80 |
| 4,096 | in pinned RAM | 1.00 | 1.38 | 6.21 | 0.58 |

With the experts on the GPU, 2,048 tokens no longer fit. The desktop held
1.4–2.4 GB, and Windows then quietly spills to system RAM and takes 50 s.
About 40% of the time is dequantising experts (0.19 ms per expert matrix,
against 0.04 ms for its matmul). Caching them between the forward and the
backward, or an FP8 matmul for experts once activations are FP8 anyway, would
cut it.

**Not proven yet:** gradients on the real weights (only the tiny float64
config was checked by finite differences); real layers of other kinds (0:
window only; 20: candidate source; 1 and 14: engram, which needs the tables or
a slice of them); real hidden states for layer 2 (its input here is token
embeddings, as layer 0 sees them, since layer 1's output needs its 101 GB
table); other GPU vendors; the tilelang kernels themselves; decoding.
