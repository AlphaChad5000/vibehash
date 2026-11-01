# Fine-tuning DeepSeek-V4.1-Flash on Vibe Hash

**Status (2026-11-01): a plan, and its step 1 built.** `vibehash/deepseek.py` is
the model in plain PyTorch, trainable and deterministic, and matches DeepSeek's
own code on a real layer (§7, which also corrects §3's compute estimate and §4's
tolerances). §8 is the first project's shape: only the top layers trained, from
precomputed inputs, so that a miner needs about 31 GB rather than a team. Every other number here is measured from the model itself at
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
2. **LoRA on it**, named so the adapter loads in PEFT. Prototyped for the
   top-layers route (§8). Transformers has no `deepseek_v41` at all (§6).
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

Steps 1–4 can be built and tested on one PC. Steps 5–6 need several. The first
project takes another route, which needs none of steps 4–6: §8.

## 6. Open questions

- **Sequence length.** Attention needs a whole sequence inside one layer, so a
  micro-batch is whole sequences. Measured (§7): with a layer's experts on a
  12 GB card, about 1,500 tokens; with them in pinned RAM, 4,096 (activations
  take 1.5 GB per 1,000 tokens). Recomputing each block in the backward would
  cut activations to its inputs, about 40 KB a token.
- **Using the result.** Transformers has no `deepseek_v41`: not 5.6.0 (the version
  config.json names, which hasn't even V4), not 5.22.1, not its main branch
  (checked 2026-11-01; `deepseek_v4` is a different architecture). So an adapter
  is served by DeepSeek's own inference code, or an engine that implements V4.1,
  applying the LoRA at run time, or merged into the weights. Merging means
  dequantising, adding and requantising to FP8 32x32 blocks, which loses a
  little. §8 saves adapters in PEFT's format under the checkpoint's own module
  names, and PEFT loads them. Whether vLLM or SGLang serve V4.1 with LoRA is
  unchecked.
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
  on request. Given ones must be well formed or the forward refuses them
  (`check_experts`, `check_picks`): no expert or position twice, none out of
  range, picks ascending, and "none" (-1) only where fewer than `index_topk`
  positions are open. Otherwise a worker could name its top expert twice and
  skip the second one's work. `choice_gap` and `candidate_gap` measure how far a
  committed choice is from a top-k of the scores, and give +inf to one that
  isn't a top-k in form.
- **Side channels are explicit** (`Side`): compressed keys (with a gradient),
  index keys and picks, from source layers to the layers after them. A `Model`
  can hold any run of layers, with or without the embedding and head. A run
  that starts after a source must be given the stream, `pre_mix`, the `Side`
  of the last kv and index sources before it, the candidate source's blocks
  if it picks inside them, and the engram hashes. Its layers refuse to run
  without what they reuse, rather than quietly attending to the window alone.
  Given those, a shard reproduces the whole model to the bit.
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

Candidate blocks, in the small model with every feature on: this code's are the
reference's for every token, on the CPU and on CUDA, and `candidate_blocks`
equals the reference's `select_candidate_blocks` bit for bit on the same scores.
In CI, a reference that can't be fetched fails the tests instead of skipping
them.

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
table); other GPU vendors; the tilelang kernels themselves; decoding. And
`Model.run` takes engram hashes only, not table rows, so a model built with
`engram_table=False` can't run its engram layers through it yet: a block takes
`rows` directly, and wiring them through is part of step 4.

## 8. The first project: train only the top layers

**Status (2026-11-01): designed and prototyped on one PC; the speed and quality
runs wait for the PC to be free.** Code in `vibehash/deepseek_top.py`, tests in
`tests/test_deepseek_top.py`. Numbers marked *estimate* aren't measured yet.

**Why this shape.** Miners won't give up 350 GB of SSD, so each should need
about a game install's worth. And teams passing activations live (§2–§3) are
network-bound by 20–200×. So the first project freezes the bottom `40 − K`
layers (K = 4 by default, chosen per job) and trains LoRA only on the top K.
The frozen part's output for the job's text never changes. So one machine with
the whole model, the preparer, computes what crosses the cut once per dataset,
and publishes it as the job's data with a Merkle root. Miners download a shard
of it plus the top K layers. Then they train like PR #5's fine-tune units:
alone, with no traffic between miners but checkpoints.

**What crosses the cut** (`Cut.crossing`, from the config). At every cut from
24 to 36, the first trained layer is an index source that reuses layer 20's
keys. So what crosses is the stream and pre_mix leaving the last frozen layer,
plus layer 20's compressed keys and index keys. Layer 20's candidate blocks
cross only for sequences over 16,384 tokens. No picks cross: a cut at 37
would add layer 36's, 1 KB a token. No engram either, since its tables are in
layers 1 and 14.

| Per token, cut at 36 (K = 4) or 32 (K = 8) | bf16 stream | FP8 stream |
|---|---|---|
| Stream (4 × 5,120) | 40,960 B | 21,120 B |
| Layer 20's compressed keys (512) and index keys (128) | 1,280 B (or 356 B packed as the FP4 values they are) | 1,280 B |
| pre_mix, token id, learn bit | 20 B | 20 B |
| **Total** | **42.3 KB** | **22.4 KB** |

So a dataset is 42 GB per million tokens (22 GB with an FP8 stream). The FP8
stream is rounded the way the model rounds its own activations: 2.7% rms. On the
small model, training from it lowered the loss as fast as training from bf16
(0.204 vs 0.205 over 8 steps). So it's an option, worth checking on real weights
first; bf16 is lossless and the default. `pack` and `unpack` write and read
records, and `record_root` gives the Merkle root.

**Exactness** (`tests/test_deepseek_top.py`). Training the top layers from
precomputed inputs is training the whole model with its bottom frozen, to the
bit: the same losses, and the same B, m and v after three Adam steps. Shown on:
- the small model `ModelArgs` defaults to, with 1 and with 2 trained layers
  (with 1, layer 3's compressed keys and picks cross the cut);
- the small model with every feature, cut so that compressed keys, index keys
  and candidate blocks cross, and an engram layer stays below;
- packed records, read back.

A cut that trains a kv source above another trained layer would need that
source's compressed-keys gradient carried between trained layers, so
`Cut.check` refuses it. No cut above layer 20 is affected.

**Train in float32, not bf16** (measured, the small model, four Adam steps, the
same inputs, CPU against CUDA):

| How the trained layers compute | B apart, CPU vs CUDA |
|---|---|
| The reference's bf16 with FP8 activation rounding | 8.9e-2 of B |
| bf16, no activation rounding | 7.1e-2 |
| **float32, no activation rounding** (`deepseek_top.TRAIN`) | **7.1e-5** |

bf16's rounding of every activation, amplified by Adam's normalised steps, makes
a replay on other hardware useless as a check. Float32 brings it to where PR
#5's float32 units are. So the preparer runs the frozen layers in the
reference's arithmetic, and miners run the top layers and the head in float32
(`Model(layer_num=..., head_num=...)` mixes the two; still exact, tested). The
frozen FP8 and FP4 weights dequantise into float32 exactly. The cost is a small
mismatch with how the model was trained and is served, with FP8 activations.

**Adapters.** LoRA-FA, as PR #5: A frozen, drawn from the job's seed as PEFT
draws it; B trained; Adam. Rank 16 on `wq_a`, `wq_b`, `wkv` and `wo_b` is
635K weights of B a layer, 2.5M for K = 4: a 30 MB `[B|m|v]` checkpoint. The
shared expert can be added; `wo_a` can't, because PEFT has no block-diagonal
LoRA. Saved in PEFT's format under the checkpoint's module names, with
`layers_to_transform`. PEFT loads them and computes what `deepseek.py` does
(tested). Transformers has no `deepseek_v41` to load them into (§6).

**Checking.**
- *Miners check miners* as in PR #5: a checker replays a segment with the same
  K layers and the records that segment used. It fetches those records by index,
  with Merkle proofs against the job's root, not the whole dataset. A dispute
  goes to PR #5's bisection game, which narrows to one op the node recomputes
  on its CPU.
- *The precompute is refereed the same way.* The preparer commits, per record
  (one sequence) and per frozen layer, a 32-byte hash of that layer's output:
  stream, pre_mix, the side channels it produces, and its choices (routing,
  picks, candidate blocks). The record's leaf is the hash of those hashes and of
  the record's bytes. That's 1.2 KB a sequence for 36 layers.
  - A challenger disputes a record. The game first bisects over layers, to the
    first one whose output the two disagree on given an agreed input, then over
    that layer's ops.
  - The preparer stores no intermediate streams. It keeps the token ids and the
    model, and recomputes any record's layers when asked. Its code is
    deterministic, so its recomputation hashes the same, in seconds per record.
- *The referee's DeepSeek-specific ops*, beyond Qwen's. Every one has small
  inputs, so the node can do them on a CPU:

  | Op | Inputs for one op | Exact or tolerant |
  |---|---|---|
  | FP8 dequant (32×32 ue8m0), FP4 dequant (packed, per 32) | one weight row or tile, fetched from the pinned commit by range | exact |
  | Activation rounding: FP8 per 32, FP4 per 32 (indexer), FP4 with E4M3 per 16 (compressed keys) | 16–32 values | exact (numpy needs the rounding written out) |
  | Matmul row or tile, also grouped `wo_a` | a row of x and of W | tolerant |
  | Hyper-connections: `hc_mixes` (24 × 20,480), Sinkhorn (20 rounds on 4×4), `hc_pre`, `hc_post` | one token | tolerant |
  | RMSNorm, RoPE with YaRN (the reference's float32 table), the compressor's 2-way softmax | one token | tolerant |
  | Sparse attention, one query and head: 128 window keys plus 512 picked ones, the kernel's 64-key online softmax | 640 × 512 values | tolerant |
  | Indexer scores for one query | its 32 × 128 query and the record's n index keys | tolerant; O(n), about 1 s in numpy at 64K tokens |
  | Top-k with tie margins: experts (6 of 384), picks (512 of n), candidate blocks (2,048 of n/8) | one score row | exact given the scores, within the margin |
  | Expert SwiGLU with its clamps | a token-slot's rows of w1, w3, w2 | tolerant |
  | Engram hashing; table rows; the gate | token ids; 24 rows of 264 B by range; one token | exact; exact; tolerant |

  None needs more than a few MB. Two things are less cheap. Long contexts make
  an indexer row O(n). And bisecting inside a layer means both sides hash the
  layer's intermediate tensors, hundreds of MB each for a 4K-token record. That's
  work for them, not the node.
- *The weak spot.* A float op is judged within a tolerance, and the reference's
  arithmetic differs by about 1% a layer between devices (§4). So a preparer
  could bend each frozen layer's output within that. A check that is bitwise on
  the preparer's own hardware class would close the gap.

**Miners' requirements.** A step is layer-major, as `train_step` does it: each
trained layer runs every micro-batch, then the next layer, then the backward in
reverse, each layer recomputed from its stored input. On a small GPU,
`train_step_streamed` goes expert-major inside each layer:
- attention, router and shared expert run for all of the step's micro-batches;
- then each routed expert (18.8 MB) is copied to the GPU once, runs every token
  routed to it, and the results are scattered back;
- the backward runs the same stages in reverse.

Each expert crosses PCIe once per layer per direction, the same traffic as
loading the whole layer, so the GPU holds only the non-expert weights. The
forward is `train_step`'s exactly. The gradients agree to float32 rounding
(2e-7), since they're summed in another order (tested).

| | Whole layer on the GPU | Expert-major (streamed) |
|---|---|---|
| Weights on the GPU | 7.4 GB layer, 1.3 GB head | 0.7 GB (four layers' non-expert weights), 1.3 GB head |
| Kept for the backward, float32 (measured on the CPU at real sizes) | 878 KB a token (the whole layer's graph) | 368 KB a token (the attention stage's; the MoE stages keep a few rows) |
| Longest sequence on 12 GB (with 1.4–2.4 GB taken by the desktop) | about 1,500 tokens | — |
| On 8 GB / 6 GB | doesn't fit | roughly 7,000 / 4,500 tokens (*estimate*: the counts, about 280 KB a token the step keeps on the GPU, the indexer's transient scores at 128 KB per token of context, and a token × token score matrix) |
| Speed | §7: 0.6–0.75 ms a token a layer in bf16; float32 *to be measured* | *to be measured*: the same PCIe traffic, plus 384 small copies and launches a layer |

Without recomputing (keeping every layer's graph through the step), four layers
need 3.5 MB a token, so 8 GB holds about 1,500 tokens for the whole step.
Recomputing is the only way to longer sequences, at the price of one more
forward.

- **GPU floor.** Training computes in float32, and FP8 and FP4 are only how
  weights are stored (plain casts dequantise them). So no bf16 or FP8 hardware is
  needed: any CUDA GPU with 6–8 GB that PyTorch supports, RTX 20-series and
  newer for sure, GTX 10-series where its build still covers Pascal. Speed comes
  from PCIe and overheads more than from FLOPs. Intel and Apple GPUs would need
  float8 tensors in their PyTorch back ends (unchecked), or dequantising on the
  CPU.
- **RAM and disk.** The top 4 layers and the head are 30.9 GB on disk (files
  39–43; 60.4 GB for the top 8), plus the miner's data shard.
  - Held in pinned RAM, they need a 48 GB PC.
  - Streamed from an NVMe SSD instead: 7.2 GB per layer per direction per step,
    about 19 s a step at 3 GB/s for K = 4. That's balanced against a step of
    about 8K tokens at §7's 0.6 ms a token a layer, and needs 16 GB of RAM.
- **Data.** One step of 8 sequences of 512 tokens is 170 MB of records (90 MB
  with FP8). So a miner holds a shard of the dataset (say 10 GB, 240K tokens)
  that its units draw their batches from. Units can't draw from the whole
  dataset as PR #5's do.
- **What a miner downloads.**
  - Once: the model, 30.9 GB for K = 4 (measured: 30,888,459,112 bytes in 448 s
    here, 69 MB/s) or 16.1 GB for K = 2.
  - Then data at **42.3 GB per million tokens** of its shard, 22.4 GB with an
    FP8 stream.
  - What cuts it:
    - an FP8 stream (−47%);
    - keeping a miner on the same shard across rounds and epochs, so it
      downloads its data once (the averaging still covers the whole dataset
      across miners: a 10M-token job over 100 miners is 100K tokens, 4.2 GB, each);
    - shards sized to each miner's link;
    - packing the side channels as the FP4 values they are (−2%);
    - a smaller K.
  - A checker fetches only the records of the segment it replays (16K tokens is
    0.7 GB).

**The prep: one command, on a rented GPU or a home PC** (`vibehash/deepseek_prep.py`;
the output isn't trusted, it's checked like any work):

    pip install torch huggingface_hub tokenizers vibehash
    hf auth login        # the runner's own token: the output goes to their own repo
    python -m vibehash.deepseek_prep --data tokens.npz --repo USER/NAME [--cut 36] [--fp8]

It runs on Linux or Windows, unattended, and can be stopped at any point: the same
command carries on where it was, after a reboot or a dropped connection too. It
logs progress with an ETA, and works in five phases:
1. The dataset's engram hashes, and so the table rows it reads. The tokenizer's
   compressed token map is rebuilt here; it gives config.json's 99,092 ids.
2. Each engram table streamed once from the pinned commit, in parallel 64 MiB
   pieces, keeping only those rows: a slice, memory-mapped from disk. Tested
   against the real Hub on a few rows.
3. The frozen layers' files, fetched whole and sha256-checked.
4. Blocks of sequences (128K tokens) through every frozen layer. Each sequence's
   state waits in RAM. Each routed expert comes to the GPU once per block, with
   all the tokens routed to it, if the GPU can't hold the whole layer. After
   each layer each sequence's state is hashed. Each block's records and hashes
   are uploaded as it finishes.
5. The manifest: the Merkle root and the blocks' leaves, uploaded last. The job
   then pins the repo at that commit.

On the small config, every record the prep writes is bit-identical to
`deepseek_top.precompute`'s. A prep stopped mid-table and again mid-blocks, then
rerun, ends with the same root as an uninterrupted one (`tests/test_deepseek_prep.py`).

| What the runner needs (cut at 36) | |
|---|---|
| Download | 266 GB of frozen layers (kept) + 1.3 GB embedding + 203 GB of engram tables streamed (not kept) ≈ **470 GB** |
| Disk | **~280 GB**, plus the slices (at most 13 GB per million tokens, less for repeated n-grams) |
| RAM | ~24 GB for 128K-token blocks (`--block-tokens 65536`: ~16 GB) |
| GPU | any CUDA GPU with 6 GB or more (forward only, experts one at a time); 14 GB free holds whole layers |
| Upload | **42,300 MB per million tokens** (22,400 with `--fp8`), plus 1.2 KB of hashes a sequence |
| Time on a 4070 SUPER | ~2.8 h per million tokens (*estimate*: 9.4 ms a token for 36 layers, from §7's 0.24 ms a token a layer with experts in RAM, plus reading the layers off an NVMe disk once a block) |

The tables touch nearly every 64 MiB piece for any real dataset, since hashed rows
have no locality. So the table stream is ~203 GB whatever the dataset's size.

*Estimate* for a rented L40S (48 GB, ~$1.57/h). Assumptions:
- 1.7× a 4070 SUPER's speed, from memory bandwidth (864 vs 504 GB/s), since the code
  is bound by dequantising and memory traffic, not FLOPs;
- whole layers on the GPU;
- 250 MB/s from the Hub and 150 MB/s to it.

| | 1M tokens | 10M tokens |
|---|---|---|
| Download (470 GB) | 0.5 h | 0.5 h |
| Prep (36 layers) | 1.6 h | 16 h |
| Upload (42 GB per million) | 0.1 h | 0.8 h |
| **Total, cost** | **~2.3 h, ~$3.60** | **~17 h, ~$27** |
| Disk to provision | 300 GB | 420 GB |

A home runner on a link like the owner's dorm network (70 MB/s down) takes ~1.9 h to
download. Upload is their bottleneck: 42 GB per million tokens at 20 Mbit/s is
4.7 hours, so `--fp8` (2.5 h) suits them.

**Quality: what training only the top costs** (`scripts/topk_quality.py`).
Qwen3-0.6B (28 layers), LoRA-FA on q, k, v and o, the same steps and data,
held-back loss on 237 everyday chats; all layers vs the top 6 (~20%) vs the top
3 (~10%). *Not run yet*: the run was stopped while the PC was needed. It resumes
from partial results and takes about 20 minutes on a free 4070 SUPER. Qwen3-4B
in float32 won't fit under 10 GB of GPU memory.

**Build order for this route** (replacing §5's 3–7 for the first project):
1. Done (§7): `deepseek.py`.
2. Done, prototype: LoRA on it, precompute, layer-major and expert-major
   training, records and their root, exactness tests, PEFT format.
3. *Measure* on a free PC: `scripts/deepseek_top_bench.py` (precompute and
   training speed, with layer 2 standing in for each top layer), the streamed
   schedule's speed and largest sequence on 6 and 8 GB (a cap on GPU memory),
   and `scripts/topk_quality.py`.
4. *Train the top 4 for real.* Their 30.9 GB are downloaded (2026-11-01). Still
   to do: loss on real chats, which needs a prep's records, and DeepSeek's own
   top-4 vs top-8 (60.4 GB, not downloaded).
5. *The prep, for real.* `deepseek_prep` is prototyped: one command, resumable,
   tested on the small config, with its table streaming tried on the real Hub.
   Still to do: a run on real weights (on a rented GPU or a friend's PC: ~470 GB
   down, ~280 GB of disk), and answering challenges (serving a record's layer
   inputs on request, by recomputing them).
6. *A DeepSeek unit kind,* once PR #5 merges: a spec with the cut and targets,
   the record dataset in place of token ids, `train_step_streamed` behind
   `train_segment`, and miners' data shards.
7. *The referee's DeepSeek ops* in numpy (the table above), for the dispute game.
8. *The job flow:* where the datasets live (42 GB per million tokens: an HF
   dataset repo or the preparer's server, pinned by root), and the pre-flight's
   numbers.

Skipped for this route: engram slices (§5 step 4: no trained layer reads a
table, and the preparer holds the tables), pipeline cells and teams (steps
5–6), worker-to-worker transport, and FP8 edges as live traffic (only as a
dataset format).
