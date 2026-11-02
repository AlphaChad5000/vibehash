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

**Status (2026-11-01): designed and prototyped on one PC. There are no offline
speed and quality runs: by the owner's decision, DeepSeek is proven on its
first live training job, which measures itself ("The first job measures
itself" below).** Code in `vibehash/deepseek_top.py`, tests in
`tests/test_deepseek_top.py`. The prep is mining work: small units that anyone
holding their layers can run, check or challenge (`vibehash/deepseek_split.py`,
"The split prep" below). There is no preparer role. The one-PC prep is a dev
and test tool. Numbers marked *estimate* get replaced by what the first live job
measures.

**Why this shape.** Miners won't give up 350 GB of SSD, so each should need
about a game install's worth. And teams passing activations live (§2–§3) are
network-bound by 20–200×. So the first project freezes the bottom `40 − K`
layers (K = 4 by default, chosen per job) and trains LoRA only on the top K.
The frozen part's output for the job's text never changes. So what crosses the
cut is computed once per dataset and published as the job's data with a Merkle
root. Miners download a shard of it plus the top K layers. Then they train like
PR #5's fine-tune units: alone, with no traffic between miners but checkpoints.
The computing once, the prep, is mining work too. It's cut into units of a few
layers that ordinary miners run, handing off through storage ("The split prep"
below). One person or one machine doing it would break "nobody controls it".

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
#5's float32 units are. So the prep's units run the frozen layers in the
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
- *The prep is refereed the same way.* Whoever runs a prep unit commits, per
  sequence and per frozen layer it ran, a 32-byte hash of that layer's output:
  stream, pre_mix, the side channels it produces, and its choices (routing,
  picks, candidate blocks). A record's leaf is the hash of all 36 layers' hashes
  and of the record's bytes. That's 1.2 KB a sequence.
  - Anyone who can hold a unit's layers can replay it and challenge it. The game
    first bisects over the unit's layers, to the first one whose output the two
    disagree on given an agreed input, then over that layer's ops.
  - A unit's inputs stay in storage until every unit reading them is past its
    window, so both sides replay from them. Units are deterministic, so a replay
    hashes the same, in seconds.
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
  arithmetic differs by about 1% a layer between devices (§4). So a unit's
  holder could bend each frozen layer's output within that. A replay that is
  bitwise, by a checker on the same hardware class, would close the gap.

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
| Speed | §7: 0.6–0.75 ms a token a layer in bf16; float32 *measured on the first live job* | *measured on the first live job*: the same PCIe traffic, plus 384 small copies and launches a layer |

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
  - Mapped from the disk instead, they need 16 GB of RAM
    (`read_safetensors(..., mmap=True)` and `deepseek_top.build_trained`).
    - Everything but the routed experts goes on the GPU.
    - The experts stay the file's own bytes, read as each one's turn comes. The
      OS keeps what RAM allows, so 32 GB is faster.
    - Tested: the experts aren't copied, and a streamed step equals one from
      RAM to the bit.
    - On an NVMe SSD the reads are 7.2 GB per layer per direction per step:
      about 19 s a step at 3 GB/s for K = 4. That's balanced against a step of
      about 8K tokens at §7's 0.6 ms a token a layer. A SATA SSD is about 6×
      slower.
  - Disk for the pilot: 30.9 GB of model plus the miner's share of the 10.7 GB
    of train records (and the 3.8 GB held back, for a scorer). That's under
    40 GB for anyone sharing the records with others, and up to 45 GB for one
    miner training on all of them. A prep piece's holder needs 29.6 GB plus
    ~2 GB of hand-offs in flight.
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

**The one-PC prep: a dev and test tool** (`vibehash/deepseek_prep.py`). The
protocol has no path for it. On the network, a machine big enough for every
unit takes them as units, like anyone else, and they're checked unit by unit.
The command runs the whole prep on one machine, to try things and to test the
split prep against:

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

*Notes, not paths:* a rented cloud GPU and Hugging Face Jobs (compute next to
the weights) were weighed for the whole prep and dropped, since one machine
doing it is a central point. For the record, an *estimate* for a rented L40S
(48 GB, ~$1.57/h), at 1.7× a 4070 SUPER from memory bandwidth, and 250 MB/s
from the Hub, 150 MB/s to it: ~2.3 h and ~$3.60 for 1M tokens, ~17 h and ~$27
for 10M.

**The split prep: the design** (settled by the owner, 2026-11-01; prototype in
`vibehash/deepseek_split.py`, tests in `tests/test_deepseek_split.py`, numbers
from `scripts/deepseek_split_plan.py`). One person or one machine doing the prep
would break "nobody controls it", and even 70 GB a person is too much. So:
- The protocol knows only units. There's no preparer role: a big machine simply
  takes many units, even all of them in a row, with no special path.
- Anyone who can hold a unit's layers can run it, check it, or challenge it.
- The node only schedules and referees.

*The shape.* The bottom is frozen, so a unit's output depends only on its inputs.
Nobody needs anyone else online: each output waits in storage for the unit that
reads it, an asynchronous assembly line. The work graph (`plan`) has two kinds
of unit:
- **rows** (an engram table, a row range): 15 ranges a table, 6.8 GB each, each
  streamed once from the pinned commit.
  - For each chunk of the text, the unit stores the rows that chunk's n-grams hash
    to in its range. That's 3.4 KB a token a table in 64K-token chunks (measured on
    the everyday chats: 54% of a chunk's 24 rows a token are distinct).
  - Rows units need no other unit.
  - A row is checkable exactly: the node reads its 264 bytes from the Hub by range.
- **layers** (a piece of G layers, a chunk): the piece's layers over the chunk's
  sequences.
  - It reads the stream entering its first layer and writes the stream leaving
    its last.
  - Side channels (compressed keys, index keys, picks, candidate blocks) are
    stored once, by the piece that makes them. Every later piece that reuses one
    reads it directly; nothing is relayed.
  - The piece holding layer 1 or 14 also reads the chunk's rows.
  - The last piece writes the job's records.

*Exact* (the small config). Split into pieces of 1 and 2 layers, with the engram
table in 3 row ranges, the units write exactly the one-PC prep's records, block
files and Merkle root. Order doesn't matter: the same holds with the last chunk
run first, and with a unit overturned and redone after the next piece had read
its output.

*What crosses each cut* (sequences of 4,096 tokens):

| Cut | Side channels crossing (their source layer) | Bytes a token, stored | Compressed losslessly |
|---|---|---|---|
| 2, 8, 14, 20 (a kv source starts the next piece) | none | 40,976 | 27,864 |
| 4, 16 | compressed keys, picks (2; 14) | 42,512 | 28,128 |
| 21 | compressed keys, index keys, picks (20) | 43,280 | 28,472 |
| 24, 28, 32, 36 | compressed keys, index keys (20) | 42,256 | 28,220 |

The stream is all but 1–3% of it, so where the cuts fall barely matters. Side
channels pack as the FP4 values they are, and picks as a bitmap.

*Hand-offs stay exact, compressed losslessly.* Measured on real weights, on the
CPU (`scripts/deepseek_stream_stats.py`). The chain: the embedding, then layer 2
standing in as layer 20, then the real layers 36–39, over 4,096 tokens of
everyday chats.
- **Lossless.** Byte planes (every value's high byte, then every low byte) through
  zstd level 3 take the bf16 stream to 0.68–0.71 of its size (level 19:
  0.60–0.66); zstd on the plain values only to 0.78. So a cut is 27.9 KB a
  token, not the ~21 KB that FP8 would give.
- **The 4 copies aren't redundant enough to exploit.**
  - Their cosine is 0.71–0.98, and they deviate from their mean by 34–56% of
    the stream.
  - Low rank is lossy: a rank-3 approximation still errs by 0.1–1.4%, rank 2 by
    2.2–6.3%.
  - The best structured code tried, an FP8 mean plus FP4 deviations, errs by
    3.8–6.5% for 23–36% fewer bytes than FP8 (which errs by 2.7%).
  - So nothing tried beats FP8 for its size.
- **FP8 hand-offs** (21.1 KB a token, 17 KB after zstd) round at every cut, and
  their cost to the model isn't known.
  - The stand-in chain can't tell. Flipping the last bit of half the values after
    each layer, a 0.4% nudge, moves its stream 22% from exact after two cuts and
    58% after five: about as far as FP8 does (24% and 64%). At the head the two
    end equally far from exact: KL 2.0 and 2.2 nats, with the top token agreeing
    35% and 38% of the time. One FP8 rounding at the last cut alone costs KL
    0.003. The chain is out of distribution, and its routing and picks flip under
    any change.
  - Measuring it needs consecutive real bottom layers (layers 0–3: 29.6 GB, not
    downloaded).
  - Lossy codes are deterministic, so checking would still work; the question is
    only quality.
  - FP8 at the training cut alone, one rounding, stays an option.

*Experts: a chunk touches nearly all of each layer's.*
- **Layer 0, on its real input.** Layer 0's input is the embedding itself, so
  its routing needs only its 170 MB outside the experts (read from the Hub by
  range; routing doesn't run the experts). Over 406K tokens of everyday chats:
  - every one of its 384 experts has tokens by 262K tokens. A 64K-token chunk
    touches 382, a 2K-token sequence 340;
  - the load is uneven on chat text: the busiest 10% take 43% of the picks, and
    12 experts get under 1% of the mean (the rarest, 2 picks in 406K tokens);
  - at these rates a million tokens leave 0.007 experts untouched. So a million
    tokens do touch them all.
- **Off its input, it isn't.** Layer 2 fed the embedding directly, skipping
  layers 0–1, touches only 247 of 384 in 64K tokens, adding about 14 a doubling,
  with the busiest 10% taking 89%. Layers 36–39 on the stand-in chain are as
  skewed. So routing balance holds on the model's own inputs, which is all a
  prep ever feeds it.
- So scoping by expert can't shrink a prep's download. A miner would have to run
  the layer to learn which experts it needs, and a 64K-token chunk needs 382 of
  384 (99.5%).
- Splitting a layer's experts among people never pays.
  - Each token would go out to its 6 experts' holders and back: about 120 KB a
    token a layer (6 × 2 × 10 KB), against 28 KB a cut.
  - It adds two more hand-offs a layer.
  - A miner whose GPU can't hold a layer streams its experts from RAM or disk
    instead (expert-major, as in training).

*Layers per piece, G: the main lever.* Every cut moves 28 GB per million tokens
up to the bucket and down again. A layer's weights are 7.39 GB, so above about
265K tokens a cut costs more than a layer (7.39 GB / 27.9 KB). Traffic, in GB,
with one holder a piece, hand-offs compressed, and miners' upload including the
records (42.3 GB per million tokens). The weights are 266 GB of layers, the
1.3 GB embedding, and the 203 GB of tables the rows units stream. The bill counts
the trainers reading the records once and a month of storing them:

| G | Pieces | Tokens | Weights from HF (once) | Hand-offs: bucket egress | Miners' upload | Spaces bill |
|---|---|---|---|---|---|---|
| 1 | 36 | 0.1M | 470 | 99 | 103 | $5.00 |
| 1 | 36 | 1M | 470 | 994 | 1,027 | $5.00 |
| 1 | 36 | 10M | 470 | 9,937 | 10,265 | $94.11 |
| 2 | 18 | 0.1M | 470 | 49 | 52 | $5.00 |
| 2 | 18 | 1M | 470 | 485 | 525 | $5.00 |
| 2 | 18 | 10M | 470 | 4,854 | 5,250 | $46.78 |
| 4 | 9 | 0.1M | 470 | 23 | 27 | $5.00 |
| 4 | 9 | 1M | 470 | 232 | 273 | $5.00 |
| 4 | 9 | 10M | 470 | 2,316 | 2,732 | $23.14 |
| 9 | 4 | 0.1M | 470 | 9 | 13 | $5.00 |
| 9 | 4 | 1M | 470 | 92 | 134 | $5.00 |
| 9 | 4 | 10M | 470 | 916 | 1,339 | $10.10 |
| 36 (one PC) | 1 | 0.1M | 470 | 1 | 5 | $5.00 |
| 36 (one PC) | 1 | 1M | 470 | 7 | 49 | $5.00 |
| 36 (one PC) | 1 | 10M | 470 | 69 | 491 | $7.87 |

- **Weights** come from Hugging Face once per holder and stay cached across jobs;
  hand-offs are per job. On a busy network with layers cached, the hand-offs are
  all that's left, so fewer cuts matter more than the weights column suggests.
- **Each unit is network-bound.**
  - A 64K-token chunk through a piece downloads 1.85 GB and uploads 1.83 GB,
    whatever G is.
  - The GPU needs 20 s a layer for it (*estimate*: 0.3 ms a token a layer on a
    mid-range card; §7 has 0.24 on a 4070 SUPER).
  - On a 100/20 Mbit/s cable link, the transfer is 2.5 min down and 12 min up.
    On 300/300 fiber it's under a minute each way.
  - Per million tokens through a piece: 28.2 GB down and 27.9 GB up, plus
    G × 7.39 GB of weights once.
- **End to end**, 1M tokens in 64K chunks, *simulated*. Each miner's downlink, GPU
  and uplink overlap across chunks. An output is read as soon as it's written,
  or after a 1-hour window.

| G | Miners | Cable 100/20: read at once | Cable: wait 1 h a unit | Cable, weights cached | Fiber 300/300: read at once | Fiber: wait 1 h |
|---|---|---|---|---|---|---|
| 1 | 36 | 13.9 | 49.9 | 13.9 | 1.5 | 37.5 |
| 1 | 72 | 11.4 | 47.4 | 11.4 | 1.4 | 37.4 |
| 2 | 18 | 9.7 | 27.7 | 9.5 | 1.1 | 19.1 |
| 2 | 36 | 7.2 | 25.2 | 7.0 | 0.9 | 18.9 |
| 4 | 9 | 7.8 | 16.8 | 7.3 | 1.0 | 10.0 |
| 4 | 18 | 5.3 | 14.3 | 4.8 | 0.8 | 9.8 |
| 4 | 36 | 4.1 | 13.1 | 3.6 | 0.7 | 9.7 |
| 9 | 4 | 7.4 | 11.4 | 6.0 | 1.5 | 5.5 |
| 9 | 8 | 4.9 | 8.9 | 3.6 | 1.1 | 5.1 |
| 9 | 36 | 3.1 | 7.1 | 1.7 | 0.9 | 4.9 |

**Recommendation: G = 4 by default**: nine pieces of 29.6 GB of weights each.
- That's the K = 4 trainer's footprint (30.9 GB), so one class of miner does
  both.
- Against one-layer pieces it cuts the hand-offs 4×: 232 vs 994 GB of egress per
  million tokens. It cuts the time 3.4×: 4.1 h vs 13.9 h for 1M tokens with 36
  miners on cable.
- Below about 265K tokens a cut moves less than a layer's weights. There, G = 2
  or 1 lets more miners take part, and the traffic is small anyway (100K tokens
  at G = 1: 99 GB).
- G = 9 cuts the egress 2.5× again, but needs 67 GB a person, over the owner's
  limit.
- Within a job, the plan fixes G. Miners claim pieces whose layers they hold or
  have room for, preferring layers already in their cache, so the network
  downloads each layer once.

*Checking.* Each layers unit commits two things.
- Per sequence and per layer, deepseek_prep's 32-byte hash: stream, pre_mix, the
  side channels made, the choices.
- A digest of every object it read and wrote, taken of the decoded tensors so the
  codec can change. The chain of digests ties each unit's input to its producer's
  output (tested).

Anyone holding the piece's layers can replay a unit (`replay`, tested). It runs
the unit again from the same checked inputs, writing nothing, and gives each
sequence's first layer whose hash differs. That's where a challenge starts: the
state entering that layer is agreed (its hash is the one the unit committed for
the layer before), and the layer's output isn't. Either side serves that agreed
state (`state_before`, tested to hash as committed), and the dispute bisects
through that layer's ops, down to one op the node referees on its CPU (the
table above).
- A wrong output poisons everything above it for that chunk. Waiting out each
  unit's window before the next reads it puts every window on the critical path.
  With 1-hour windows, 1M tokens at G = 4 take 13.1 h instead of 4.1 h (36
  miners, cable), and 49.9 h at G = 1.
- So consumption is optimistic (`Line`). A unit's output can be read as soon as
  it's committed.
  - If a challenge against it succeeds, it and everything downstream of it for
    that chunk go back to pending and are redone (`Line.overturn`, tested).
  - Honest units that read a bad input did right by what they were given. They're
    paid, out of the cheater's stake.
  - The job pins the dataset's root once the last unit's window closes: one
    window at the end, not one per piece.

*Hosting the hand-offs.* The 2 GB node can't hold them, and no host has to be
trusted (`tests/test_deepseek_split.py`):
- Every reader checks each object against the digest its writer committed to,
  which the node holds. A host that alters one is caught (`Tampered`).
- A reader takes the first host with the committed copy (`Mirrors`: the bucket,
  the writer's own storage, peers).
- A withheld object is made again, to the bit, by anyone holding its piece
  (`Line.redo`). If its inputs are gone too, their writers are redone first,
  back to the token ids and the public weights if need be. Withholding costs
  recomputation, not data.

So hosts that miners run can slot in later, without changing units or
commitments. The options:
- **(a) For v1: a DigitalOcean Spaces bucket the operator pays for.**
  - The node signs a URL per object and method, so miners need no account
    (`presign`: S3's Signature Version 4, checked against AWS's worked example;
    `Signer` on the node, `PresignedStore` on the miner). A signed URL can't be
    bent to another method, object or part.
  - Nothing big travels in one request. A hand-off goes up as S3's multipart
    upload, in 64 MiB parts, each retried on its own. Its progress is kept in a
    journal, so after a crash only the parts not yet acknowledged are sent. It
    comes down in 64 MiB ranges. Tested against a local S3 that checks every
    signature, through a link failing every third request, and cut off part way.
  - It costs $5 a month for 250 GiB stored and 1 TiB out, then $0.01 a GiB out
    and $0.02 a GiB-month stored. Uploads are free.
  - Each object is deleted once every unit that reads it is accepted
    (`Line.accept`), so the bucket holds only the frontier and the records.
  - Bills from the table: $5 for any 1M-token prep; for 10M, $23 at G = 4 and
    $94 at G = 1.
- **(b) Each uploader's own HF dataset.**
  - Every uploader needs an account and a token.
  - Each object is a git commit, and deleting LFS files frees no space without
    rewriting history. That fights a frontier that turns over by the hour.
  - Free accounts get 100 GB private. Public storage is best effort, for content
    useful to others, which day-old hand-offs aren't.
  - HF's Storage Buckets fit better: mutable, $12–18 a TB-month, egress up to 8×
    the storage included. They still need an HF account per uploader, and they
    document no presigned URLs.
  - HF is the right home for a reusable dataset's *records*, not for hand-offs.
- **(c) Peer to peer, later.** The next piece's holder pulls from the producer,
  or from any miner who kept a copy. That saves the egress bill and the central
  host. A producer that must stay online until its output is read loses "nobody
  online together", so peers should mirror each other's outputs: any copy
  matching the digest will do.

**Known centralization points.**
- **The bigger one: the public network has a single validator today.**
- **The hand-off bucket.** It can't alter data undetected, but it can withhold
  or delete it, and its operator can stop paying. What that costs is
  recomputation, as above. The way out is (b) or (c) behind `Mirrors`.
  A validator re-refereeing a DeepSeek game reads the records from it
  (`replay_block`'s `records`), each checked against the pinned root. One
  without read access can't, and would refuse the block: a second validator
  needs the bucket's read keys or a copy of the records.
- **The weights' source.** DeepSeek's weights and engram tables come from one
  Hugging Face repo, pinned by commit and checked by sha256. The licence is MIT,
  so anyone may mirror them: a convenience, not a dependency.
- **Who the referee trusts for weights (the owner's choice, option C,
  2026-11-02).**
  - No party to a dispute ever supplies weights: the node's referee reads them
    from the pinned commit on Hugging Face, and a game's record keeps none.
  - The four trained layers and the head (31 GB) are checked tensor by tensor
    against `vibehash/data/deepseek_flash_top.json`. That's a per-tensor
    manifest made from the pinned files (`scripts/deepseek_tensor_manifest.py`,
    each file's sha256 checked against the Hub's), and its sha256 is pinned in
    `deepseek_job`.
  - The frozen layers and the engram tables are read by range and checked only
    at the file level. So for prep disputes, the referee trusts Hugging Face to
    serve its own pinned files.
  - The way out (option B): anyone downloads the whole model once (~470 GB, on
    any rented machine) and pins per-tensor hashes, plus per-piece hashes for
    the tables. Anyone can recheck them.

*Pay.* Prep units are mining work like any other: a layers unit earns by
layer-tokens (G × its chunk), and a rows unit by the gigabytes it streamed.
Uploads are a miner's real cost (12 minutes a unit on cable), so the credit
should count hand-off bytes too.

*Reuse.*
- A prep's records at cut 36 serve K = 4 directly. They serve K = 2 or 1 too,
  after one more frozen piece is run over them (layers 36–37 for K = 2, 36–38
  for K = 1), like any other unit.
- They can't serve K = 8, which needs the stream at 32. But in the split prep
  every cut's stream exists for a while anyway. Keeping cut 32's as well (28 GB
  per million tokens) serves K = 4 and K = 8 from one prep.
- Popular open datasets could be prepped once, and their records kept public on
  HF, pinned by root, for every job that trains on them.

*One machine, one layer at a time* (a note, not built). Each layer runs over
the whole dataset, then is deleted, and the next is fetched.
- Peak disk is one layer plus the dataset's states: 7.4 GB + 42 GB per million
  tokens (28 GB compressed).
- On top of that come the side channels still to be read (1.3 KB a token for
  layer 20's keys) and the slices.
- So ~50 GB for 1M tokens, and ~430 GB for 10M (~290 GB compressed). The
  download stays 470 GB.
- It's the split prep with one miner taking every unit in turn, with no special
  path.

**The training footprint.** Trained layers change every step, so they stay
inside one miner: no live relay. The per-miner knob is K, counting the 1.3 GB
head: K = 4 is 30.9 GB, K = 2 16.1 GB, K = 1 8.7 GB. What each step down costs in
quality is for live jobs to show. `scripts/topk_quality.py` would also run
Qwen3's top 2 layers and top 1, standing in by fraction for DeepSeek's top 2.9
and 1.4.

**Quality: what training only the top costs.** The first live job's loss
curve answers it. The offline comparison, `scripts/topk_quality.py`, is skipped
but kept:
- Qwen3-0.6B (28 layers), LoRA-FA on q, k, v and o, the same steps and data,
  held-back loss on 237 everyday chats;
- all layers against the top 6 (~20%), the top 3 (~10%), and the top 2 and 1
  (DeepSeek's K = 2 and K = 1, by fraction);
- resumable, about 30 minutes on a free 4070 SUPER. Qwen3-4B in float32 won't
  fit under 10 GB of GPU memory.

**The first job measures itself.** VIBE has no value yet, so a weak first result
costs nothing. What the job has to record:
- **Phases that slow miners can make.** A unit's deadline (`deadline`) assumes
  the slowest link the network takes (25 Mbit/s down, 10 up) and 1 ms a token a
  layer, about 4× §7's estimate. It doubles that, and is never under 10 minutes.
  A claim past its deadline goes to another miner (`Line.expire`, tested). The
  training unit kind sizes its phases the same way.
- **The loss curve.** /network's job summary (`netstats.py`, from
  `store.job_rounds`) and the worker dashboard already show it for fine-tunes.
- **What each worker manages.** `deepseek_top.Meter` gives a unit's seconds,
  tokens a second and peak GPU memory.
  - Every prep unit's commitment carries them, with the bytes and seconds it
    read and wrote ("stats", outside any hash).
  - The training unit kind reports the same per step into the worker's stats
    and logs.
  - So the run gives the real speeds, links and memory, which replace this
    section's *estimates*.

**The first job: "think shorter", the pilot** (the owner's choice, 2026-11-01).
It teaches V4.1-Flash to reach correct answers with much less thinking, from
GSM8K's short human solutions written in its own thinking-mode template. The
data is built by `scripts/make_think_shorter.py`; the card is
`docs/datasets/think-shorter.md` (source, MIT licence, transformation, risks).

| The job's spec | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` at commit `314113f8` |
| Trained | the top 4 layers (cut at 36, K = 4): LoRA-FA, rank 16, alpha 32, on `wq_a`, `wq_b`, `wkv` and `wo_b`, in float32 |
| Data | `think-shorter-pilot.npz`, sha256 `f4d7f822…523a3a`: 2,000 sequences, one example each, 341,753 tokens |
| Train | the first 1,500 sequences: 252,500 tokens, 122,878 of them learned |
| Held back | the last 500: 89,253 tokens, for the scorers' loss |
| Passes | 3 over the 1,500: 4,500 examples, 757,500 tokens |
| Root | the prep's Merkle root over all 2,000 records, pinned once its last unit's window closes |
| Id | contains "think-shorter"; the front page's featured panel shows 1,500 and 500 |

The input format is `deepseek_prep`'s, unchanged. Its prep, at G = 4 in
32K-token chunks (`scripts/deepseek_split_plan.py`; sequences average 171
tokens, so no picks or candidate blocks cross a cut):
- **Units:** 11 chunks, so 30 rows units and 99 layers units (9 pieces × 11).
- **From Hugging Face:**
  - 267 GB of layers, taken by the 9 pieces' holders at 29.6 GB each, and kept
    for later jobs;
  - 203 GB of engram tables, streamed by the rows units at 6.8 GB each. The
    tables cost the same for any dataset.
- **The bucket:**
  - 79 GB of hand-offs out;
  - 14.4 GB of records (10.7 GB train, 3.8 GB held back), stored for the
    trainers and scorers;
  - $5 in all.
- **A unit** moves ~0.93 GB each way, and gets 39 minutes (`deadline`). A
  piece's holder that takes all 11 chunks moves 9.6 GB down and 9.5 GB up.
- **End to end**, *simulated*:
  - on 100/20 Mbit/s cable, ~3.5 h with 9 miners, 2.2 h with 36;
  - on fiber, under an hour.
  - 32K-token chunks fill the line sooner than 64K: 3.5 h against 4.7 h with 9
    miners. They also halve each hand-off for home links.
- **A trainer** downloads 30.9 GB of weights once, plus its share of the train
  records (all 10.7 GB if it trains alone).

**Prep chunks of 4K tokens (the owner's call, 2026-11-02).**
- Prep units are refereed now. A layers unit computes op by op
  (`deepseek_program.PrepProgram`) so that any of its ops can be checked, and
  its states are chunk wide. One float32 stream tensor is 335 MB at 4K tokens
  and 2.7 GB at 32K.
- So the pilot preps in 4K-token chunks (`chunk_tokens` 4096): 84 chunks, so
  756 layers units. The figures above are 32K's.
- Each unit streams its layers' experts from the miner's disk to its GPU once,
  8× as often as at 32K: about 10 seconds a unit on NVMe and PCIe 4.
- Each unit moves about 0.12 GB of hand-offs each way. The bucket's totals
  don't change.
- The full set can take bigger chunks later, with states kept in bf16 and
  streamed.

The full set (7,473 train, 1,319 held back) comes later and reuses the pilot's
records. A record is its sequence's alone, since a sequence's arithmetic
doesn't depend on its chunk-mates (tested exact in the split prep). So the full
prep runs only the 6,792 new sequences, under a root over all 8,792.

The wiring around it, once PR #5 merges:
1. **Post the job** with that spec. The node needs only the 1.7 MB token file,
   checked against its sha256. It plans:
   - the chunks: `deepseek_prep.blocks_of`, 32K tokens;
   - the pieces: `pieces(36, 4)`;
   - the work graph: `plan(...)`, with 15 row ranges a table.
2. **The prep phase.** `Line` hands out the rows units, then layers units as
   their inputs land, each with its deadline. The node signs bucket URLs only
   for each unit's own objects. Challenge windows run, and `Line.summary` is
   the job summary's "prep" field.
3. **The root.** When the last window closes, the job pins the manifest's root.
   The hand-offs are deleted, and the records stay.
4. **The training phase.** The DeepSeek unit kind trains from the records, each
   miner on a shard, for 3 passes. Each step reports `Meter`'s stats. The
   scorers' held-back loss on the 500 goes into `store.job_rounds`.
5. **The result:** the adapter in PEFT's format, `layers_to_transform` 36–39.

**The launch checklist** (the wiring branch, `claude/deepseek-wiring`, stacked on
PR #5 with this PR merged in; started 2026-11-01 without waiting for PR #5 to
merge). [x] done, [ ] to do.
- Data and records:
  - [x] The pilot's token file (`make_think_shorter.py`), and the job's spec
    (above).
  - [x] Trainers and scorers read records from the bucket, each checked against
    the pinned root (`deepseek_split.read_block`).
- Units and the node:
  - [x] The DeepSeek job kind in the spec checks (`deepseek_job.DeepSeekSpec`, a `LoraSpec`, so its training
    rounds are a fine-tune's; `tasks.py` parses and validates it). A node without it refuses the spec.
  - [x] The node needs no PyTorch for it: `deepseek_config.py` and `deepseek_graph.py`.
  - [x] Prep units as claims in the work graph (`deepseek_rounds.py`): a prep
    round is a version 4 round whose units are the graph's ready ones, one
    holder each. A miner commits to the sha256 of its unit's commitment in the
    bucket; the node reads it at the reveal and checks it names the unit's
    objects; two checkers replay the unit. The node signs links for a unit's own
    objects only, during compute (`Coordinator.may_sign`, the "sign" message);
    the compute phase is the unit's deadline; `Line.summary` is the job
    summary's "prep" field. The last unit's round pins the root and writes the
    manifest. Tested end to end over HTTP (`tests/test_deepseek_network.py`).
  - [x] The DeepSeek training unit (`deepseek_unit.py`): records read from the
    bucket and checked, `train_segment` as a fine-tune's (replays to the bit),
    the scorers' held-back loss. `model.train_segment` and `TrainingJob.evaluate`
    dispatch to it.
  - [x] Its rounds after the prep: trainers, checkers and scorers read the
    records from the bucket against the pinned root (`/job/<id>` gives it), and
    the scorers' held-back loss goes into `store.job_rounds`, as a fine-tune's.
  - [ ] Records shards per miner (each now fetches the chunks it trains on),
    and `Meter`'s stats in the round's record.
  - [x] The judge (PR #5's bisection game, op table, record version 4) merged
    in (2026-11-01).
  - [x] Training units refereed through the judge, losers slashed as for
    fine-tunes.
    - `deepseek_program.py` is a training unit's tree of ops. Each step runs its
      sequences together, layer by layer, so each layer's experts stream once a
      pass. Attention runs a sequence at a time. The routed experts go by slot
      (row t·k + j uses token t's j-th expert), so the tree follows from the
      data alone, as a game needs.
    - `deepseek_tracer.py` runs those ops in PyTorch. Units train by it, so a
      worker's dispute states chain to its checkpoints.
    - `referee_deepseek.py` checks each op in NumPy on the node (the fine-tune
      session wrote it). A step's records are read as `record.{seq}.{part}`
      through the weights' interface, each checked against the pinned root. A
      validator re-refereeing a game reads them from the bucket
      (`replay_block`'s `records`).
    - Tested (`tests/test_deepseek_program.py`, tiny config):
      - op by op agrees with autograd to about 1e-7;
      - every honest op passes the referee, and 3% bends fail;
      - a trainer that scaled one op loses its game at that op.
    - The local run (`tests/test_deepseek_network.py`) has a tampering
      trainer, slashed, and a false accuser, which withdraws (unpaid, not
      slashed).
  - [x] Per-tensor hashes of Flash's trained layers and head
    (`vibehash/data/deepseek_flash_top.json`, pinned as
    `deepseek_job.TOP_MANIFESTS`; see "Who the referee trusts for weights"
    above). `deepseek_job.FlashWeights` is the node's source.
  - [ ] Prep layers units refereed, losers slashed: written and tested, but
    off for the pilot (`coordinator.PREP_GAMES`).
    - **The pilot's rule (the fourth review, 2026-11-02).** A rejected prep
      unit, rows or layers, ends unchecked and is redone. There's no game,
      nobody is slashed, and its rejecter isn't paid. It stays until a prep
      game's moves are timed from its states' real size (one 4K-token chunk's
      state is 320 MB and more, and a move is timed for a LoRA adapter's, 240
      s) and `PREP_TOLERANCE` is measured across devices.
    - Their program (`deepseek_program.PrepProgram`) runs in the reference's
      bf16, chunk wide, in 4K-token chunks. Its commitments carry a state root
      before the unit and after each of its layers. A state between layers
      holds only what crosses the cut above it, so roots don't depend on the
      pieces.
    - With prep games on, when a checker rejects one, the node first checks
      that the unit's outputs in the bucket are its last committed state
      (`deepseek_graph.handed_root`, `records_root`, in NumPy). If they aren't,
      that's fraud; if they can't be read, it's unchecked. If they are, a game
      runs over the unit, layer by layer, then op by op.
    - The node also requires each unit's first root to be the last root of the
      unit below it.
  - [ ] Rows units (copies of the engram tables) aren't refereed, even with
    prep games on: a rejected one ends unchecked and goes out again. Its
    consumers read rows only from accepted rows units, and each layer's engram
    op is refereed against the table itself. Two weaknesses of that rule, and
    of the pilot's for every prep unit:
    - One checker that rejects everything costs honest miners their pay for
      those units, and the prep a round's delay each time. It isn't paid for
      a rejection (the fourth review's finding 8), but its stake is never at
      risk, since there's no game to lose.
    - Prep canaries can be spotted: a canary hands over another unit's
      commitment, and the commitment names its unit. The task no longer names
      the commitment's object or the donor (it carries the bytes inline: the
      fourth review's finding 6), but the commitment itself still does. So they only catch a
      checker that approves without reading anything. A checker that reads the
      commitment but doesn't replay goes uncaught; only the any-rejection rule
      and the second checker stand against it. Real canaries need outputs bent
      by the node (it can do that in numpy, decoding a hand-off) under names
      that don't give them away.
  - [ ] After launch, from the fourth review (2026-11-02). Each is latent while
    the network has one validator and prep games are off:
    - Validators without the bucket's key can't re-referee DeepSeek games (one
      would refuse the block), and a replay takes the leader's accepted
      digests and records root on its word.
    - The prep's progress is dropped when the job's data isn't attached.
    - Signed uploads don't bind a size, a multipart upload has no limit, and
      any keypair gets download links.
    - Objects are named by graph unit, not by round and worker. A write's link
      dies with its compute phase, so only the bucket can change an accepted
      object now. If one changes, its readers find it Tampered and the prep
      stalls: nothing calls `Line.redo` on a reader's report yet.
    - The node's hand-off check (`_handoff`, off with prep games) reads whole
      states inside its lock, with no size cap. It needs the commitment's
      treatment (read before the lock, size asked first) before prep games go
      on.
- The worker:
  - [x] `modelstore` (below), memory-mapped experts (16 GB of RAM).
  - [x] Setup's step 4 asks where big models go and how much space they may
    use (once; flags set it). `vibehash cache` shows them; `cache clear [JOB]`
    removes them.
  - [x] Running and checking prep units (`deepseek_worker.py`), the model files
    fetched through `modelstore` (asked first, within the cap).
  - [x] What to tell a PC that can't take DeepSeek work (`can_take`: no NVIDIA
    GPU with 6 GB for the prep or 8 for training, under 16 GB of RAM, or the files
    don't fit), and that it keeps earning on other jobs.
  - [x] The worker's steps for DeepSeek units (`deepseek_client.py`, called from
    `client.py`): prep units through links the node signs, committed and
    revealed; other miners' units replayed; records for training rounds. The
    download is asked once: `worker --deepseek ask|yes|no`, saved.
  - [ ] Disk use on the dashboard.
  - [x] The menu item (main's terminal menu, `menu.py`): "Mine the DeepSeek
    job", and "Model files" under Settings.
- Storage:
  - [x] Multipart, resumable, presigned (`PresignedStore`, `Signer`), tested
    against a local fake S3.
  - [x] The node reads the bucket from the environment (`node.bucket_from_env`:
    `VIBEHASH_BUCKET_HOST`, `_REGION`, `_KEY`, `_SECRET`).
  - [ ] The bucket and its keys, which the owner sets up on the node at launch.
- Docs:
  - [ ] A "Mining the DeepSeek job" section in `docs/guide.html`, in the
    guide's voice, keeping its ids and JS working. It covers:
    - the job (the think-shorter pilot) and its requirements;
    - what the worker downloads and when;
    - prep pieces and training units, from a miner's point of view;
    - where the data comes from: the bucket, through links from the node;
    - how to follow it: the front page's featured panel and `job watch`;
    - where the model files go, how much space they may take, and when
      they're deleted (below).
  - [x] The front page's requirements, from this section: "an NVIDIA card with
    8 GB or more (RTX 20-series or newer), 16 GB of memory (32 GB is faster),
    about 40 GB free on an SSD, about 30 GB downloaded once". They hold with
    the mapped experts above. A miner who trains alone on all the pilot's
    records needs up to 45 GB.
- Proof:
  - [ ] A full local run: node, prep miners, trainers and a dispute, on the
    small config.
  - [ ] Then on real layers, if they fit on one PC.

**Model files on a miner's PC** (`vibehash/modelstore.py`, built and tested in
`tests/test_modelstore.py`). The owner's rule: the ~30 GB must never become
permanent clutter on someone's C: drive.
1. **Asked first.** Before any big download, the worker says what it is, how
   big, and where it goes, and offers another place. The drives are listed with
   their free space; the default is the one with the most room, and the choice
   is remembered (config.json's "models"). An unattended flag says yes, and the
   folder can be set outright.
2. **A cap the user sets** ("use up to 40 GB").
   - The store never goes over it. It evicts other jobs' least recently used
     model files first.
   - It refuses a job whose files won't fit, under the cap or on the disk (it
     leaves 2 GB free), rather than overflowing. Other jobs carry on.
3. **Not permanent by default.**
   - A job's model files are deleted when the job ends, or after 7 days without
     mining it, whichever comes first.
   - A setting keeps them for next time, which saves re-downloading on a slow
     link. Kept files still give way to the cap.
   - A file two jobs use stays while either needs it.
   - Hand-offs, data shards and checkpoints are temporary: deleted once used
     and accepted.
4. **Visible and removable.**
   - `entries()` is what `vibehash cache` lists: what, where, how big and why.
   - `clear()` removes everything, or one job's files.
   - `usage()` is the dashboard's disk use.
   - `choose()` is setup's one-time location question. Changing the place
     deletes what the old one held instead of leaving it behind.
5. **No partial download left behind.** Downloads resume from their `.part`
   file (`deepseek.fetch_file`, which now takes the store's folder). The index
   knows a download while it's under way, so `clear`, the job's end and the
   weekly sweep remove any that won't be resumed.

The front page says: "You choose where it goes and how much space it may use,
and it's deleted when the job ends, unless you'd rather keep it for next time."
What's left for the wiring PR, since these all live in PR #5's files:
- the worker's prompt and its unattended flag;
- `vibehash cache` and `vibehash cache clear [job]` in the CLI;
- the dashboard's disk use;
- the setup question.

The fine-tune path can use the store the same way.

**Build order for this route** (replacing §5's 3–7 for the first project):
1. Done (§7): `deepseek.py`.
2. Done, prototype: LoRA on it, precompute, layer-major and expert-major
   training, records and their root, exactness tests, PEFT format.
3. *Off the critical path:* the offline runs, skipped by the owner. The scripts
   stay: `scripts/deepseek_top_bench.py` (speed, with layer 2 standing in for
   each top layer), the streamed schedule's 6 and 8 GB caps, and
   `scripts/topk_quality.py`. Their numbers come from the first live job.
4. *Train the top 4 for real*, as the first live job: think shorter's pilot
   (above). Its 30.9 GB are downloaded (2026-11-01). DeepSeek's own top 4
   against top 8 (60.4 GB, not downloaded) is for a later job.
5. *The prep, for real.* The split prep (`deepseek_split`) is the design,
   prototyped and tested exact to the one-PC prep. It has the work graph, exact
   lossless hand-offs checked against their writers' digests, untrusted hosts
   behind `Mirrors`, presigned storage, and optimistic consumption with
   rollback and redo. The one-PC `deepseek_prep` is the dev and test tool: one
   resumable command, with its table streaming tried on the real Hub.
   - Done since: multipart, resumable uploads and ranged downloads; answering
     a challenge by replaying a unit (`replay`, `state_before`); claim
     deadlines; each unit's stats.
   - Still to do, once PR #5 merges: wiring the units into the node (claims,
     presigned URLs, windows, pay). The front page's featured-job panel
     (`docs/site/index.html`) reads it from GET /network. So the node puts
     `Line.summary` into that job's summary as its "prep" field (in
     `Node._network`, or through a hook in `netstats.job_summary`). The field
     is the frozen layers, each one's fraction of units accepted, pieces done of
     total, and how many miners ran one in the last 10 minutes; netstats
     documents it.
   - Still to do, later:
     - hosts run by miners (each uploader's own storage, or peer to peer) behind
       `Mirrors`, to retire the bucket as a central point;
     - a run on real weights, piece by piece;
     - FP8 hand-offs measured on consecutive real bottom layers (layers 0–3,
       29.6 GB), which would cut a cut's 28 KB a token to 17 KB if harmless.
6. *A DeepSeek unit kind,* once PR #5 merges: a spec with the cut and targets,
   the record dataset in place of token ids, `train_step_streamed` behind
   `train_segment`, and miners' data shards. For the front page:
   - The spec carries "base", so netstats reports a fine-tune with metric
     "loss".
   - It also carries "batch_size" and "steps_per_unit", so "examples" is units
     × batch × steps.
   - The scorers' median held-back loss goes into `store.job_rounds` as PR #5's
     fine-tunes do it. That gives the loss curve and the "N× as likely, token by
     token" figure.
   - The first job's id contains "think-shorter": the page matches its title,
     blurb and sample on that. It's the pilot, 1,500 problems and 500 held
     back, as the page already shows.
7. *The referee's DeepSeek ops* in numpy (the table above), for the dispute game.
8. *The job flow:* where the datasets live (42 GB per million tokens: the
   bucket while training, or a public HF dataset for a reusable one, pinned by
   root), and the pre-flight's numbers.

Skipped for this route: engram slices (§5 step 4: no trained layer reads a
table, and the rows units stream the tables), pipeline cells and teams (steps
5–6), worker-to-worker transport, and FP8 edges as live traffic (only as a
dataset format).
