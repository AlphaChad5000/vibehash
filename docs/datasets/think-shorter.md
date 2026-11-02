# Think shorter: the first DeepSeek-V4.1-Flash fine-tune's dataset

> Gaming PCs taught DeepSeek V4.1 Flash to think shorter.

The first top-layers job (DEEPSEEK.md §8) trains LoRA adapters on the top 4 of
DeepSeek-V4.1-Flash's 40 layers so that, in thinking mode, it reaches correct
answers with far less thinking. The data is GSM8K's grade-school math problems
with their human-written solutions as the thinking. Built by
`scripts/make_think_shorter.py`; tested by `tests/test_think_shorter.py`.

## What "think shorter" means

In thinking mode, DeepSeek-V4.1-Flash writes its reasoning between `<think>`
and `</think>`, and only then writes the answer. Every thinking token costs
generation time before the answer starts. A LoRA can't make the model generate
tokens faster, but it can make it generate fewer. Length and style are what
the top layers shape best.

The targets here think briefly. GSM8K's solutions average **74.7 tokens of
thinking** (median 68), 3.6 lines, then a one-line answer. The claim to prove
is that, with the adapter, the model's thinking on GSM8K's test problems gets
much shorter while its accuracy stays about the same. How long the base model
thinks on these problems isn't measured yet: that needs the whole 510 GB
model generating (see "Measuring the real claim").

What the network itself proves is learning, by the held-back loss shown live
on the website: the loss on GSM8K's 1,319 test solutions, which no trainer
sees. That shows the adapter learned to write like the targets. It doesn't
show the thinking got shorter at the same accuracy. Only generating does.

## Source

- **GSM8K**, `openai/gsm8k` on the Hugging Face Hub, config `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`: 7,473 train and 1,319 test
  problems, written by human problem writers.
  - `main/train-00000-of-00001.parquet`, sha256 `ea82612e…7351419d`
  - `main/test-00000-of-00001.parquet`, sha256 `ee7b8da9…cb44e4f59`
- **Licence: MIT**, from the Hub's dataset card (`license: mit`) and the LICENSE
  of the original repository, github.com/openai/grade-school-math (MIT,
  Copyright (c) 2021 OpenAI). It permits this use, with the notice kept.
- **Citation:**

  ```bibtex
  @article{cobbe2021gsm8k,
    title={Training Verifiers to Solve Math Word Problems},
    author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
    journal={arXiv preprint arXiv:2110.14168},
    year={2021}
  }
  ```

- **The model's chat template and tokenizer**, from `deepseek-ai/DeepSeek-V4.1-Flash`
  at the commit deepseek.py pins, `314113f8a7dadb388215396ba5fbee2c610a04d6`:
  - `chat_template.jinja`, sha256 `d959d804…0040b31c3` (git blob `eb12dd28…`)
  - `tokenizer.json`, sha256 `c90dfa01…7d599476b` (git blob `6a15814d…`)

## The transformation

Each problem becomes one sequence. The sequence is the model's own chat
template in thinking mode at its default reasoning effort ("high", 75),
rendered by the template's Jinja as transformers renders it:

```
<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)

<｜User｜>{question}<｜Assistant｜><think>{solution lines}</think>The answer is **N**.<｜end▁of▁sentence｜>
```

There's no line break before `<｜Assistant｜>`. The model's README says the
release has no Jinja template and points to `encoding/encoding.py`. The repo
at the pinned commit does have `chat_template.jinja`, "ported from
encoding/encoding.py". So the builder also checks every rendering against
encoding.py's format: `BOS + <｜System｜> + effort prefix + <｜User｜> + question +
<｜Assistant｜><think> + reasoning + </think> + content + EOS`.

- **User turn:** the question, unchanged.
- **Thinking:** the solution's lines, with GSM8K's calculator annotations
  (`<<48/2=24>>`) removed, each line trimmed, and blank lines dropped (6
  answers have one). Removing the annotations is safe: 546 of the 27,998
  aren't followed by their exact result, and all 546 are only because the
  visible text writes the number with thousands commas.
- **Answer:** "The answer is **N**.", with N taken from GSM8K's `#### N`
  without thousands commas. Only 79 train and 14 test answers have them; 718
  and 117 answers over 999 don't. Every final answer is an integer.
- **Learned** (the learn mask): every token the model would write after the
  prompt's `<think>`: the thinking, `</think>`, the answer line and
  `<｜end▁of▁sentence｜>`. Learning `</think>` teaches when to stop thinking, and
  the end token teaches when to stop answering. Not learned: the BOS, system
  and effort prefix, the user turn, and `<｜Assistant｜><think>`, which the prompt
  supplies at inference.
- **One example a sequence, no packing.** The longest is 453 tokens.
- **Checks** on every example:
  - the prompt alone tokenizes as the start of the whole sequence, so the
    training prompt is token for token what a server would feed the model;
  - the model's part starts on a token boundary, right after `<think>`
    (128821);
  - no question or solution contains a special token's text.

  All 8,792 sequences pass the mask checks: one BOS, one `<think>`, one
  `</think>`, and EOS last and learned.
- **Reproducible.** Every source is pinned by revision and checked against
  its sha256 before use. `np.savez` stamps each member with the time it was
  written, so the builder writes the zip itself: uncompressed, as np.savez
  does, dated 1980-01-01. A fresh download into an empty folder and a rerun
  gave the same sha256 for every file.

## Files

Written to `~/.vibehash/datasets/think-shorter/` (never into the repo):

| File | sha256 | Bytes | Sequences |
|---|---|---|---|
| `think-shorter.npz` | `ed3458f74ed532e19af362a4d4aca4d70d761d37af171108073cf6f0e7c38b04` | 7,526,162 | 8,792: train 0–7,472, held back 7,473–8,791 |
| `think-shorter-pilot.npz` | `f4d7f822cf54a9f5a7f21441759bee63e2d1294d4a747b89588f7fb762523a3a` | 1,717,459 | 2,000: train 0–1,499, held back 1,500–1,999 |
| `manifest.json` | `777b1327c0e5338d80a93c3de84d8713d26ed2a29661a2e915fd2c4d28cfec9a` | | the sources, the transformation, the split ranges, the pilot's GSM8K rows, the numbers |

Each `.npz` is exactly `python -m vibehash.deepseek_prep --data`'s input:
- `tokens`, int32: every sequence end to end;
- `lengths`, int32: one per sequence;
- `learn`, uint8: 1 where a token is a target.

Train and held-back sequences share one file, so one prep (one stream of the
engram tables) and one Merkle root serve both. The job takes the held-back
range from the manifest. The pilot's problems are the 1,500 train and 500 test
problems whose questions' sha256 come first, so the pick depends on nothing
but the text.

**Against PR #5's "tokens" dataset kind** (`finetune.TokenDataset`): the
labels mean the same thing. A 1 at position t means token t is predicted, and
position 0 is never learned. The container differs:
- PR #5 uses padded 2-D arrays (`train_ids`, `train_len`, `train_mask`, and
  the same for `val_`) with the model's manifest and config.json inside;
- this file uses flat arrays with lengths.

Converting is a few lines: pad to the longest sequence (453) and split at the
held-back range. But `TokenDataset.check` would still refuse the result:
`arch_dims` doesn't know `model_type: deepseek_v41`, whose sizes sit under
`text_config`. More to the point, DeepSeek miners don't train from token ids
but from the prep's records. What the job needs is DEEPSEEK.md §8's build
step 6, a DeepSeek unit kind whose dataset is the records (pinned by their
Merkle root) in place of token ids. This file is that prep's input.

## Numbers

| | Examples | Tokens | Learned | Mean a sequence | Median / p95 / max | Prompt | Thinking, mean (median) | Thinking lines | Answer + EOS |
|---|---|---|---|---|---|---|---|---|---|
| Train | 7,473 | 1,269,386 | 616,768 | 169.9 | 160 / 267 / 453 | 87.3 | 74.4 (67) | 3.6 | 7.1 |
| Held back | 1,319 | 228,674 | 111,348 | 173.4 | 165 / 270 / 402 | 89.0 | 76.3 (70) | 3.7 | 7.1 |
| **All** | **8,792** | **1,498,060** | **728,116** | **170.4** | 161 / 267 / 453 | 87.6 | **74.7 (68)** | 3.6 | 7.1 |
| Pilot train | 1,500 | 252,500 | 122,878 | 168.3 | 159 / 261 / 453 | 86.4 | 73.8 (67) | 3.6 | 7.1 |
| Pilot held back | 500 | 89,253 | 43,802 | 178.5 | 168 / 278 / 402 | 90.9 | 79.5 (72) | 3.7 | 7.1 |
| **Pilot, all** | **2,000** | **341,753** | **166,680** | 170.9 | 162 / 267 / 453 | 87.5 | 75.2 (68) | 3.6 | 7.1 |

"Prompt" includes 27 tokens of BOS, system token and effort prefix. "Learned"
is thinking + `</think>` + answer + EOS.

## The job's size

These figures come from `scripts/deepseek_split_plan.py`'s model (DEEPSEEK.md
§8, "The split prep"), run by the builder on these exact token counts. They
assume a cut at 36 (K = 4) and prep pieces of G = 4 layers, which makes 9
pieces and 8 hand-off cuts. Figures in GB.

| | Full set (1.50M tokens) | Pilot (0.34M tokens) |
|---|---|---|
| Records, the trainers' data (42,260 B a token) | **63.3** (33.6 with an FP8 stream) | **14.4** (7.7) |
| Hand-offs, bucket egress, stored as is (42.3 KB a token a cut) | 514 | 117 |
| Hand-offs compressed losslessly, as designed (~28 KB) | 347 | 79 |
| Hand-offs at FP8's 22.4 KB (quality unmeasured) | 268 | 61 |
| A prep piece's download for this job | **41.7–42.3**; 47.1 for layers 12–15 (engram rows); 5.2 for layers 0–3 (token ids and rows) | **9.5–9.6**; 10.7; 1.2 |
| A prep piece's upload | **41.7–42.3**; 63.3 for layers 32–35 (the records) | **9.5–9.6**; 14.4 |
| A piece's weights, once and cached across jobs | 29.6 (30.9 for layers 0–3, with the embedding) | same |
| Rows units (30, 15 a table) | each streams 6.8 GB of table from HF and uploads 0.34 | each streams 6.8, uploads 0.08 |
| A trainer: top 4 layers and head once, then its shard (6 trainers) | 30.9 + **10.6** (5.6 FP8) | 30.9 + **2.4** (1.3 FP8) |
| Prep end to end, weights fetched too, 9 miners on 100/20 Mbit/s cable | 9.9 h (9.4 with weights cached) | 4.7 h (4.2) |
| Prep end to end, 18 miners | 6.5 h (6.0) | 3.8 h (3.2) |
| Training an epoch of the train set (estimate*) | 0.8–1.1 GPU-hours | 0.2 GPU-hours |

\* §7's measured 0.6–0.75 ms a token a layer in bf16, times 4 layers. That
leaves out the head and float32's slowdown, which isn't measured yet.

Training is cheap. The cost is the prep and the data. Some costs don't shrink
with the dataset:
- each piece's weights, 29.6 GB a holder, once;
- the rows units' 203 GB of engram table streamed from the Hub, which is the
  same for any dataset size.

## Recommended first job: the pilot, 3 epochs

**The pilot: 1,500 train problems, 500 held back, 3 epochs.** Then the full
set as the second job, at 2 epochs.

- **It's the first real run of everything.** The split prep has never run
  piece by piece on real weights, and the DeepSeek unit kind doesn't exist
  yet. Whatever goes wrong first is 4.4 times cheaper to redo on the pilot.
- **It fits home links.** For this job, each prep miner moves about 10 GB each
  way, against 42 GB for the full set. At a 20 Mbit/s upload, 42 GB takes 4.7
  hours, and the last piece's 63 GB of records take 7. Trainers fetch 2.4 GB of records
  against 10.6. The prep takes about 4.7 hours against 9.9 with 9 miners. The
  weights (about 30 GB a miner, once) are the bigger download either way.
- **It's enough to show the effect.** Fine-tunes for length and style commonly
  work from about a thousand examples. The trained part is small: B is 16 × (1,280 + 32,768 + 512 +
  5,120) values a layer for `wq_a`, `wq_b`, `wkv` and `wo_b` at rank 16, so
  2.5M values over 4 layers. Three passes over 1,500 examples won't overfit
  much, and the live held-back loss shows it if they do.
- **Epochs cost GPU time only.** Miners keep their shard across epochs, so the
  third pass downloads nothing. Three epochs of the pilot are 4,500 sequence
  passes, about 0.76M tokens and roughly 0.6 GPU-hours in total. If the
  held-back loss turns up during the third, evaluate the checkpoint from the
  end of the second.
- **Nothing is wasted.** A record depends only on its own sequence: deepseek_prep
  runs "sequence by sequence, to the bit". So the full job can reuse the
  pilot's 2,000 records and prep only the other 6,792 sequences. It does
  repeat the fixed costs: the rows units stream the tables again.
- **The full set at 2 epochs:** 7,473 examples, 14,946 sequence passes, more
  than enough for a style. Its traffic (63 GB of records, about 42 GB each way
  per prep miner) is fine for a network past its first job, not for a handful
  of home miners trying the pipeline for the first time.

## Known risks

- **Shorter thinking can cost accuracy on hard problems.**
  - GSM8K is easy for a frontier-class model, and its human solutions skip
    what long thinking adds: checking, trying another route, catching a
    misread.
  - An adapter that thinks GSM8K-short everywhere may hold its accuracy on
    GSM8K but lose it on harder math (MATH-500, AIME), where long thinking
    pays.
  - The training prompt always says effort 75. So the adapter learns
    "effort 75 means a few lines", which also changes what the default effort
    means.
  - Mitigations:
    - measure on a harder set too, not GSM8K alone;
    - ship it as what it is, an optional adapter loaded only when short
      thinking is wanted.
- **Math-only data may make the model terse everywhere.** This covers both the
  thinking and the reply, since every target's reply is a single line.
  - Nothing in the data tells math from anything else, so after training a
    chat question may get three lines of thinking and a one-line answer.
  - **A small general mix is worth it for the full job, not for the pilot.**
    The pilot should measure the effect cleanly first.
    - If its evaluation shows terseness spreading, the full job should mix in
      about 10% (~750 examples) from **OpenAssistant `oasst2`**:
      **Apache-2.0**, human-written, with ranked replies (English, top-ranked
      assistant replies).
    - It has no thinking traces, so its examples would teach replies only.
      Render them in the template's chat mode (`<｜Assistant｜></think>{reply}`)
      and learn only the reply and EOS. That keeps reply length and style
      without teaching anything about thinking length on chat.
    - Teaching *how long to think on chat* would take the model's own thinking
      on those prompts, which needs the full model generating.
  - Not used:
    - Dolly-15k is CC BY-SA 3.0 (share-alike);
    - No Robots is CC BY-NC 4.0 (non-commercial);
    - UltraChat is MIT but generated by ChatGPT.

  No mix is added here.
- **The model's own effort knob is the baseline to beat.** DeepSeek-V4.1-Flash's
  effort setting (1–100) is built to trade thinking for accuracy, per its
  README. If the base model at a
  low setting matches the adapter at 75 on length and accuracy, the adapter
  adds little. The comparison has to include that.
- **Held-back loss isn't the claim.** The loss measures imitating GSM8K's
  solutions. It falls even if accuracy drops.
- **Off-policy targets.** The solutions are human text, not the model's own
  words. Fine-tuning on them teaches the length and the style. It may also
  make the model stop sooner than its own computation needs. A better second
  recipe is the model's own shortest correct solutions, sampled at low effort
  and filtered by the answer. That needs the full model generating.
- **Speculative decoding.** The MTP/DSpark draft layers (config:
  `dspark_target_layer_ids` 37–39) were trained against the original model.
  As DEEPSEEK.md §2 notes, they may accept fewer drafted tokens with adapted
  top layers. That gives back some of the speed. Measure tokens per second,
  not only tokens.
- **The test set is public, and its labels aren't perfect.** The model may
  have seen GSM8K's test problems, which inflates accuracy but matters less
  for length. A small share of GSM8K's answers are known to be wrong or
  ambiguous.

## Measuring the real claim

For each of the 1,319 test problems, generate in thinking mode with the
template's default effort (75), using:
1. the base model;
2. the base model with the adapter;
3. the base model at lower efforts (50, and a few smaller integers).

Use DeepSeek's evaluation settings (temperature 1.0, top_p 0.95), with a few
samples a problem, or greedy decoding. Record:
- thinking tokens (between `<think>` and `</think>`);
- total tokens;
- accuracy (the number in `**N**`, or the last number, against `#### N`);
- tokens per second.

Repeat on a harder set (MATH-500, or AIME), and on a few dozen everyday chat
prompts to check for terseness.

This needs the whole 510 GB model serving, not a home PC: roughly one node
of 8 data-center GPUs. 8 × 80 GB holds the weights with little room left for
the cache; 8 × 141 GB is comfortable. An engine that can't load the adapter
can serve a merged checkpoint instead. Only layers 36–39's `wq_a`, `wq_b`,
`wkv` and `wo_b` change: 93M weights a layer, about 0.37 GB in all, in the
files holding the top layers. But those weights are FP8 with 32 × 32 block
scales (ue8m0). Merging means
dequantising, adding the scaled BA, and requantising, and the requantising
rounds the adapter's small change. That rounding should be checked against
the adapter run in float32 on the same prompts (deepseek.py can run layers
36–39 with the adapter).

**Can vLLM, SGLang or transformers load a DeepSeek-V4.1 LoRA today? No
released version of any of them can,** as of 2026-11-01. Checked against
their releases and source on GitHub. The adapter is PEFT's format, under the
checkpoint's own module names (`base_model.model.layers.36.attn.wq_a.lora_A.weight`,
`layers_to_transform` 36–39).

| Engine (latest release) | Runs V4.1? | LoRA on V4.1? | Names it expects |
|---|---|---|---|
| **vLLM** v0.32.0 (Oct 18) | Yes, since v0.30.0 ([#56214](https://github.com/vllm-project/vllm/pull/56214)) | **No.** The registered `DeepseekV41ForCausalLM` (the multimodal wrapper, `vllm/models/deepseek_v41/nvidia/vl_model.py`) doesn't declare `SupportsLoRA`. Open: issue [#57166](https://github.com/vllm-project/vllm/issues/57166) and PR [#57245](https://github.com/vllm-project/vllm/pull/57245), "Add DeepSeek V4.1 language-model adapter support", not merged. V4 (not V4.1) got LoRA in [#53361](https://github.com/vllm-project/vllm/pull/53361) (merged Aug 24). | DeepSeek's own (`attn.wq_a`, `wq_b`, `wkv`, `wo_b`), so these adapters should need at most a prefix change |
| **SGLang** v0.5.22 (Oct 19) | Yes, since v0.5.21 | **No.** `srt/models/deepseek_v4.py`, which serves V4.1 too, has no adapter code. Its LoRA name table (`srt/lora/utils.py`) maps a bare `wq_b` to `indexer.wq_b`, so this adapter's `wq_b` would land on the wrong module. | V3-style (`q_b_proj`, `kv_b_proj`, `o_proj`) |
| **transformers** v5.22.1 (Oct 30) | **No:** there's no `deepseek_v41` model (V4 only, since v5.8.0), so AutoConfig won't load the checkpoint | **No**, since it can't load the model. For V4, v5.22 trains adapters on fine-grained FP8 weights ([#50031](https://github.com/huggingface/transformers/pull/50031)). | Renamed for V4: `.wq_a.` → `.q_a_proj.`, `.wkv.` → `.kv_proj.`, `.wo_b.` → `.o_b_proj.` (`conversion_mapping.py`) |
| **DeepSeek's code** | `inference/` (8-GPU torchrun) | No adapter support. `deepseek-recipe` only encodes prompts and parses output. | The checkpoint's, as here |

What would work today, none of it tried:
1. **Merging into the checkpoint.** The adapter touches 16 tensors (4 layers ×
   4 projections). Dequantise each from FP8, add (α/r)·BA, then requantise.
   - ms-swift's DeepSeek-V4 guide warns that "the LoRA delta would be rounded
     to 0" in FP8, and merges against bf16 weights instead.
   - Keeping those 16 tensors in bf16 inside an otherwise FP8 checkpoint would
     avoid that. **Not confirmed:** whether vLLM or SGLang load such a mixed
     checkpoint for V4.1.
2. **vLLM built from PR #57245** (open, unreviewed). It also disables decoder
   replay for LoRA requests.
3. **DeepSeek's reference inference code** with the LoRA term added by hand.
   It uses the adapter's own names, so that's the most direct exact route.
   It's slow, but enough for 1,319 problems.

Also not confirmed: whether any of these keeps the DSpark/MTP drafter working
with an adapter, which the tokens-per-second measurement needs.
