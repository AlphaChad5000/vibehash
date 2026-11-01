# Fine-tuning DeepSeek-V4.1-Flash on Vibe Hash

**Status: a plan (2026-11-01), nothing of it built yet.** Every number here is
measured from the model itself at commit `314113f` of
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
keys and values, index keys and picks to the later layers that reuse them;
they're to be measured, but they're a few KB a token.

Compute is small by comparison, because only 6 of 384 experts run. A layer has
about 0.38B active weights, so forward plus backward-to-input is about
1.5 GFLOP a token.

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
  accepted within a tie margin, as `moe.py` does for routing.
- **Frozen weights are exact everywhere.** The scales are powers of two (`ue8m0`),
  so dequantising FP8 and FP4 to bf16 is exact: every GPU computes with the same
  weights. Only the matmuls and attention need tolerances, measured across GPU
  vendors as DESIGN §8 did for MLPs.
- **Engram rows** are checked against the job's slice root, and the slice against
  Hugging Face by spot-checks.
- **Disputes stay small.** One cell is 2 layers, about 15 GB: a validator can
  replay it on a CPU with enough RAM, in minutes for a segment of a few
  thousand tokens.

## 5. What to build, in order

1. **A differentiable V4.1 layer** (`deepseek.py`): attention with its compressor
   and indexer, MoE, hyper-connections and engram, in plain PyTorch, dequantising
   FP8 and FP4 on the fly, deterministic. Proven against `inference/model.py`:
   - the whole model on the small 5-layer config, from random weights;
   - one real layer from the real weights (7.4 GB, fits the 4070 SUPER).
2. **LoRA on it**, named so the adapter loads in Hugging Face transformers
   (`deepseek_v41`, transformers 5.6) and PEFT.
3. **Loading one shard:** `hub.py` fetches exactly one stage's tensors by range.
   Also fix `describe`, which misses the layer and expert counts when they're
   nested in `text_config`, as they are here.
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
  micro-batch is whole sequences. How long can they be on a 12 GB card?
- **Using the result.** Transformers 5.6 knows `deepseek_v41`; whether vLLM serves
  LoRA adapters for it is unchecked.
- **Image fine-tunes** would add the vision tower and the image routing bias.
- **The network.** A team is 14–40 GPUs. The public network has none yet, so steps
  1–5 come first, and teams come when there are miners to form them.
