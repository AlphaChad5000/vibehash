# Fine-tuning real models

**Status (2026-11-01): models one GPU holds fine-tune for real, on a network
whose node runs fine-tunes; not on the public network yet.** A job poster gives
a file of chats and a base model, workers' GPUs train LoRA adapters in
verified units, and the poster downloads an adapter that loads in Hugging Face
transformers and PEFT. Proven end to end on a local testnet with Qwen3-0.6B
(below). Supported: the Llama, Mistral, Qwen2, Qwen3 and SmolLM3 families
(`llm.py`); `vibehash models` marks the ones a gaming GPU holds as ready.
What 3–4B models on the public network need is the last section. Models too
big for one GPU need teams: DEEPSEEK.md and SHARDING.md.

How it works inside (A frozen and B trained so averaging is exact, Adam in the
checkpoints, the checks, the tolerances measured) is DESIGN.md §8, "Fine-tuning
pretrained models". The rest of this document is the experience, built and not.

## Try it

On one computer with an NVIDIA GPU and PyTorch (`vibehash install-gpu`), plus
`pip install "vibehash[finetune]"` for the tokenizer and chat templates:

```bash
vibehash testnet --job none --workers 3 --device cuda
```

and in another terminal, with a file of chats (JSON lines of
`{"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}`):

```bash
vibehash train chats.jsonl --base qwen3-0.6b --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json
vibehash job download
```

The first run checks the model's weights (it streams them once, 1.5 GB for
Qwen3-0.6B) and each worker fetches them once.

**What happened when it was run** (2026-11-01, RTX 4070 SUPER, three workers on
one GPU, 2,379 short everyday conversations from
[HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k),
Apache 2.0):


```
Model     Qwen/Qwen3-0.6B @ c1899de289a0 (pinned)
          qwen3, 28 layers of width 1024 · 1.2 GB of weights · licence apache-2.0
Data      everyday.jsonl: 2,379 examples (2,379 chats, in the model's chat template), 459,556 tokens, 254,216 of them to learn
          2,251 to train, 128 held back · sequences up to 320 tokens
Training  LoRA rank 16 (alpha 32) on q_proj, k_proj, v_proj, o_proj: 2,293,760 trained weights, A frozen (adapter 18.4 MB)
          Adam at 0.0005, 8 sequences a step, 16 steps a unit, 8 units a round: one epoch is about 3 rounds
Workers   a GPU with at least 4.5 GB, and PyTorch; each fetches the weights from Hugging Face once · 0 workers took part in the last rounds
Cost      10 VIBE a round · a budget of 100 VIBE buys 10 rounds (3.3 epochs), paid only for work that passes its checks
```

| Round | Units accepted | Held-back loss | Reply tokens predicted right |
|---|---|---|---|
| 2 | 0 (the workers were still fetching the model) | 2.461 | 60.2% |
| 3 | 4 | 1.189 | 68.9% |
| 4 | 4 | 1.150 | 70.0% |
| 5 | 5 | 1.131 | 70.5% |
| 6 | 5 | 1.124 | 70.5% |
| 7 | 6 | 1.119 | 70.6% |

- Rounds took about 3¼ minutes: 188 s sized for the job, plus the node
  checking the work itself, since a testnet's workers start without stake
  and nobody can check them until they've earned it. Each worker earned about
  70 VIBE; the job spent 30 of its 100.
- `vibehash chain verify` replayed all six rounds from their records: valid.
- `vibehash job download` wrote `everyday-lora/` (18 MB). Loaded with
  `PeftModel.from_pretrained` on Qwen3-0.6B at the pinned commit,
  transformers measures the held-back loss at **1.1192**, the network's
  number exactly (2.4606 without the adapter). Asked "Hi! I want to start
  running. Any tips?", the base model gives a long numbered list ("Absolutely!
  Here are some tips… 1. **Start Small**…"); with the adapter, it answers in
  the data's brief style: "Great! Start with a 10-minute run, and gradually
  increase the duration and intensity."
- Once, all three workers couldn't connect to the node for about 20 seconds
  while another program on the machine was downloading 8 GB; they retried and
  that round accepted 5 units.

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
             mixture-of-experts · 763B weights, 16B active per token · MIT licence
   Data      my-chats.jsonl: 12,408 chats, 9.1M tokens (model's chat template)
             11,167 to train, 1,241 held back · 312 cut at 4,096 tokens
   Training  LoRA rank 16 on attention: 186M trainable weights (adapter 372 MB)
             one epoch ≈ 2,240 rounds
   Network   needs teams of 20 workers with ≥ 24 GB of GPU memory
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
  it: "fetching layers 18–19: 12.4 / 14.8 GB".
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
| Network too small | Needs teams of 20 workers with ≥ 24 GB; 12 are online. You can post now, and nothing is spent until a team forms. |
| Budget short | One epoch costs about 5,600 VIBE and you have 2,000. That trains about 35% of an epoch; post anyway, or lower `--epochs`? |
| Worker disk full | Your shard needs 14.8 GB and 6.3 GB is free in `~/.vibehash/cache`. Free space or set `--cache-gb`. |
| Data in an unknown shape | Line 3 of data.jsonl has no `text` or `messages` field. |
| Model too big for this node's settings | This node relays jobs up to `--max-job-mb`. Sharded jobs measure that per shard. |

## From pieces to this

| Piece | State |
|---|---|
| PyTorch model for the Llama, Mistral, Qwen2, Qwen3 and SmolLM3 families, checked against transformers | done (`llm.py`) |
| LoRA jobs: A frozen, Adam on B, checkpoints with the moments, part-by-part checks, Adam's bound as the sanity check | done (`finetune.py`, DESIGN §8) |
| Token datasets carrying the model's manifest and config; chats through the model's own chat template, learning only the replies | done |
| `train DATA --base`: pre-flight, checking the weights once, one question, posting | done (manifests are cached by the poster, not yet shared on the network) |
| PEFT export in `job download` | done, loads in PEFT (tested on tiny models of each family, and with Qwen3-0.6B) |
| Workers fetch the base once (parallel ranged requests), keep it on the GPU, skip fine-tunes without PyTorch | done; no cache limit or download progress on the dashboard yet |
| Measured CPU ↔ CUDA tolerances | done on NVIDIA; Intel, AMD and Apple unmeasured |
| Text documents (`{"text": ...}`) and `.txt`/`.md` files | done, lightly tested |
| Scoring and dispute replays off the node, for the public network | next (milestone B, below) |
| `vibehash hub login`, gated models end to end | next: hub.py reads `HF_TOKEN`, but nothing sets it |
| Sharded rounds, teams, and why a job waits | in progress (coordinator, SHARDING.md) |
| Mixture-of-experts with committed routing | in progress (`moe.py`) |
| Workers: sticky shards, fetching their slice | next |
| Worker-to-worker transport (direct, then relays) | next |

## Milestone B: 3–4B models on the public network

The goal: `vibehash train chats.jsonl --base qwen3-4b` against node.vibehash.org,
with SmolLM3 3B, Llama 3.2 3B and Qwen3 4B, on gaming GPUs. Nothing in the
protocol changes for these models: the same job, units, checks and averaging
work at any size `llm.py` handles. What changes is what each machine has to
hold and compute, and the public node is the one that can't.

### The models

Rank 16 on q, k, v and o, 16 Adam steps a unit, sequences of up to 512
tokens; GPU memory is `finetune.memory_mb`'s estimate, which errs high
(Qwen3-0.6B estimated 5.6 GB at batch 8, MEASURED_06):

| Model | Weights | Shape | Trained (B) | A checkpoint | A unit, batch 8 | GPU memory, batch 4 / 8 | Round at the reference GPU |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (milestone A) | 1.5 GB | 28 × 1024 | 2.3M | 27.5 MB | 239 TFLOP | 4.9 / 5.6 GB | 4 min |
| SmolLM3 3B | 6.2 GB | 36 × 2048 | 3.0M | 35.4 MB | 1,207 TFLOP | 11.5 / 13.3 GB | 13 min |
| Llama 3.2 3B Instruct (gated) | 6.4 GB | 28 × 3072 | 3.7M | 44.0 MB | 1,250 TFLOP | 12.5 / 14.3 GB | 14 min |
| Qwen3 4B | 8.0 GB | 36 × 2560 | 5.0M | 60.2 MB | 1,595 TFLOP | 14.8 / 16.9 GB | 18 min |

"Round at the reference GPU" is `TrainingJob.phase_lengths` at batch 8: a
card sustaining 5 TFLOP/s (an RTX 3060's worth) on a 20 Mbit/s uplink. MEASURED_4B

### Workers: fine, with 16 GB, or 12 GB at batch 4

- **Memory.** A 16 GB card takes any of the three at batch 8. A 12 GB card
  takes SmolLM3 at batch 4 and the others with shorter sequences. Two changes
  would bring all three under 12 GB at batch 8, both exact (no change to a
  single bit of the arithmetic): making the head's logits a slice of the
  vocabulary at a time instead of casting the whole embedding to float32 at
  once (1.5 GB for Qwen3-4B), and keeping each layer's saved input in the
  backward only once instead of in autograd's copy as well.
- **Time.** At the 4070 SUPER's 15 TFLOP/s a unit takes 1.3–1.8 minutes; the
  reference card three times that.
- **Download.** 6–8 GB once per model, checked against the manifest
  (1.5 GB came in about 15 s here; at 10 MB/s it's 10–15 minutes, and the
  worker misses rounds meanwhile). The dashboard should show it.
- **Upload.** 4 checkpoints a unit, 140–240 MB, a minute or two at 20 Mbit/s,
  which the phase lengths include. Storing Adam's moments as bf16 would cut a
  third; most of B's checkpoints would also compress, since the moments are
  zero in every round's start.

### The node: it can't, as it stands

A 2 GB droplet with two vCPUs runs node.vibehash.org. It can relay, record,
replay and vote on fine-tune rounds: none of that needs the model, and nodes
without PyTorch already leave fine-tunes waiting rather than fail. But the
round's leader also does five things with the model:

| What the leader does | How often | Qwen3-4B on a GPU | On the droplet's CPU |
|---|---|---|---|
| Scores the round's model on the held-back chats (128 × 512 tokens forward) | every round | ~0.5 PFLOP, under a minute | hours, and it can't hold the 8 GB of weights |
| Checks a task whose checker didn't answer (3 segments) | when a checker misses | ~1.2 PFLOP, about a minute | hours |
| Arbitrates a rejection (1 segment) | when a checker rejects | ~0.4 PFLOP | an hour or more |
| Replays a whole unit when replicas disagree | rare | ~1.6 PFLOP | hours |
| Keeps the challenge window's checkpoints (`--keep-rounds 10`) | every round | ~3 GB a round of 8 units, 30 GB in all | the droplet's disk is 50 GB |

More memory alone doesn't fix it: a CPU is about 100 times too slow for the
replays to fit inside a round. So either the leader has a GPU, or that work
moves to workers.

### What it takes, in order

1. **A validator with a GPU leads fine-tune rounds (works with this branch, not
   yet tried on the public chain).** Validators take turns leading. A leader
   without PyTorch (the droplet) skips fine-tune jobs in its turns and runs
   the CSV jobs, and one with a GPU (`vibehash node --device cuda`, with
   PyTorch) runs them. Everyone else votes by replaying the record, which needs
   no model. So the owner's PC with its 4070 SUPER can join as a second
   validator and take the fine-tunes: the shortest way to 3–4B models on the
   public network. Two things to fix first: when the only funded job is a
   fine-tune, a validator that can't run it should pass its turn at once
   instead of letting the propose timeout run out (`1.5 × round + 5 s`, about
   half an hour for Qwen3-4B); and `--keep-rounds` sized to the disk.
2. **Scores from workers.** Every worker that trains in a round scores the
   round's starting model on the held-back chats (one forward pass, about 2% of
   a unit) and sends it with its reveal; the node shows the median. Scores
   don't decide pay, so a minority of liars can't move them, and the leader
   stops needing the model for anything but disputes.
3. **Disputes to a jury of workers.** Each task goes to two checkers, so one
   that doesn't answer leaves the other's verdict. A rejection, or two verdicts
   that disagree, goes to three more staked checkers in a short extra phase,
   drawn from the round secret like everything else, and the majority decides;
   the losing side is slashed as today. A replica disagreement becomes its
   segments, one each. The leader then only checks hashes and averages vectors
   of a few million weights, which a 2 GB droplet does in milliseconds. The
   extra phase and the jury change the record, so this is record version 4.
4. **Smaller checkpoints and shared manifests.** Moments in bf16; manifests
   published to the network (today each poster streams the weights once to
   build its own; at 8 GB that's minutes).
5. **Release and roll out as block version 2 was.** A 0.5.x node refuses a
   fine-tune's post_job as malformed, so every validator runs the new release
   before anyone posts one. Workers that want fine-tunes need PyTorch
   (`vibehash install-gpu`); those without it skip them and keep mining the
   rest.

Built here toward this: nodes that can't run fine-tunes leave them waiting
with a reason; `vibehash node --device`; workers that skip what they can't
run; 64 MiB parallel downloads; the memory estimate in every job; `models`
marking what's ready. Not built: items 1's fixes, 2, 3 and 4.
