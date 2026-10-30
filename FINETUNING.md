# Fine-tuning real models: the experience we're building

**Status: the target for the next stage of work, not yet built.** These pieces
are under way:
- the transformer and LoRA maths (`transformer.py`);
- pretrained weights, safetensors and token datasets;
- fetching tensors from Hugging Face (`hub.py`);
- sharded rounds (SHARDING.md);
- mixture-of-experts blocks (`moe.py`).

This document is what they should add up to for the people using them. Where
it and the code disagree, the code isn't finished.

## Principles

1. **One command on each side.** A fine-tune is one `train` command, and a
   worker is still just `vibehash worker`. Every setting has a good default,
   and flags only override.
2. **Nothing is spent until it can run.** Before posting, a pre-flight shows
   what will be trained, what it costs, what the network can do today and
   how long it should take. Nothing is paid for until rounds actually train,
   and a job can be cancelled at any time.
3. **No surprise downloads.** Sizes and disk space are shown before anything
   downloads. Downloads resume, and are checked against hashes the job pins.
4. **Results go where people already work.** A finished fine-tune is a
   standard PEFT LoRA adapter that loads in Hugging Face transformers and
   vLLM.
5. **Every error says what to do next.**

## Posting a fine-tune

```bash
vibehash train my-chats.jsonl --base deepseek-ai/DeepSeek-V4.1-Flash
```

That's all a person types. The command then does the following, asking only
where it must.

1. **Finds the model.** A plain name, a `huggingface.co` URL or `hf:name` all
   work. The default branch is resolved to its commit, which is pinned in the
   job, so the run can be reproduced. A gated model asks once for a Hugging
   Face token (`vibehash hub login`); it's kept in `~/.vibehash`, readable by
   the user only, and sent only to Hugging Face. A repository with only
   `.bin` (pickle) weights is refused with an explanation: those files can
   run code, and safetensors can't.
2. **Reads the data the way the model expects.** Plain text, JSON lines with
   a `text` field, and chat logs (`messages`) are recognised without flags.
   - Chats are rendered with the model's own chat template.
   - Text is tokenized with the model's own tokenizer, fetched from its
     repository.
   - A tenth of it is held back to measure progress.
   - Sequence length follows the data, up to the model's limit.
3. **Picks the training settings.**
   - LoRA rank 16 on the attention projections, unless flags say otherwise.
   - A learning rate from a table by model family and size.
   - One epoch, priced at the node's suggested fee per round.
4. **Checks the network can run it, and says so in plain words.** For
   example (illustrative numbers):

   ```
   Model     deepseek-ai/DeepSeek-V4.1-Flash  @ 3f9c2a1 (main, pinned)
             mixture-of-experts · 552B weights, 16B active per token · MIT licence
   Data      my-chats.jsonl: 12,408 chats, 9.1M tokens (model's chat template)
             11,167 to train, 1,241 held back · 312 cut at 4,096 tokens
   Training  LoRA rank 16 on attention: 186M trainable weights (adapter 372 MB)
             one epoch ≈ 2,240 rounds
   Network   needs teams of 31 workers with ≥ 24 GB of GPU memory
             47 such workers online now: 1 team · ≈ 9 days for one epoch
   Cost      2.5 VIBE a round · 5,600 VIBE for one epoch · you have 6,000 VIBE
   Post it? [Y/n]
   ```

   When the network can't run the job yet (too few workers with enough
   memory), it says so before anything is posted. It says what's missing,
   and that posting anyway costs nothing until rounds run.
5. **Makes sure the weights can be checked.** Workers download only their
   own layers, and check each tensor against a manifest of per-tensor hashes
   (`hub.py`). The first time anyone uses a given model and commit, someone
   builds that manifest:
   - the poster's computer streams each file once, hashing it without
     keeping it, and checks it against Hugging Face's published file hash;
   - the manifest is then published to the network like a small dataset,
     so nobody builds it again.

   The pre-flight shows how long that takes at the poster's connection, and
   it runs before anything is posted.
6. **Posts and opens the training dashboard** (`trainui.py`), which follows
   the held-back loss round by round.

The command waits for confirmation at every step that downloads or spends.
`--yes` skips the questions for scripts.

## Getting the result

```bash
vibehash job download
```

This writes `my-chats-lora/`. It holds `adapter_model.safetensors` and
`adapter_config.json` in PEFT's format, and a README naming the base model,
its pinned commit, the data's hash and the job id. Then:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash", revision="3f9c2a1")
model = PeftModel.from_pretrained(base, "my-chats-lora")
```

or `vllm serve deepseek-ai/DeepSeek-V4.1-Flash --enable-lora --lora-modules mine=./my-chats-lora`.

The adapter is small: hundreds of MB even for the largest models. That's
LoRA's point, and why the job only pays to train and move the adapter.

## Working

```bash
vibehash worker
```

Nothing new to type. When a sharded fine-tune is running:
- **The worker gets a shard that fits.** It tells the node how much GPU
  memory it has and how fast it uploads. The coordinator gives it a shard
  that fits, and **keeps giving it the same one** round after round, so its
  download is paid off.
- **It fetches its own layers.** It downloads just its shard's tensors
  straight from Hugging Face, checks each against the manifest, and caches
  them. The download resumes if interrupted, and the mining dashboard shows
  it: "fetching layers 18–19: 12.4 / 20.1 GB".
- **It won't fill the disk.** Before downloading, it checks free disk space.
  The cache is capped (`--cache-gb`, by default most of the free space), and
  models that haven't been used for a while are dropped first.
- **It joins a team by itself.** It reaches its teammates directly where it
  can, through a relay where it can't. The dashboard shows "team of 31 ·
  you hold layers 18–19".
- **Pay works as before:** per verified unit of work, now per verified cell.

`vibehash setup` already offers the right PyTorch build for the card
(`gpusetup.py`). GPUs too small for any shard of the running jobs keep
working on the jobs they fit.

## What people should see when something's wrong

| Situation | Message (gist) |
|---|---|
| Gated model, no token | This model needs you to accept its licence at `<url>`. Then run `vibehash hub login`. |
| Only `.bin` weights | This repository has no safetensors weights. Vibe Hash never loads pickle files, which can run code. |
| Network too small | Needs teams of 31 workers with ≥ 24 GB; 12 are online. You can post now, and nothing is spent until a team forms. |
| Budget short | One epoch costs about 5,600 VIBE and you have 2,000. That trains about 35% of an epoch; post anyway, or lower `--epochs`? |
| Worker disk full | Your shard needs 20.1 GB and 6.3 GB is free in `~/.vibehash/cache`. Free space or set `--cache-gb`. |
| Data in an unknown shape | Line 3 of data.jsonl has no `text` or `messages` field. |
| Model too big for this node's settings | This node relays jobs up to `--max-job-mb`. Sharded jobs measure that per shard. |

## From pieces to this

| Piece | State |
|---|---|
| Transformer + LoRA maths, PEFT-compatible adapter layout | in progress (`transformer.py`) |
| safetensors (incl. BF16/FP8), token datasets, `vibehash tokenize`, jobs starting from given weights | in progress |
| Hugging Face fetcher: name → pinned commit, per-tensor ranged downloads, manifests, cache, `describe` | in progress (`hub.py`) |
| Sharded rounds, teams, and why a job waits | in progress (coordinator, SHARDING.md) |
| Mixture-of-experts with committed routing | in progress (`moe.py`) |
| `train --base`: pre-flight, prompts, manifest publishing | next |
| PEFT export in `job download` | next |
| Workers: sticky shards, fetching their slice, cache limits, dashboard progress | next |
| Worker-to-worker transport (direct, then relays) | next |
| PyTorch backend for transformers, measured GPU tolerances | next |
