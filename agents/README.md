# The Muse agents

Two small programs that put an LLM on either side of a Vibe Hash fine-tune. By default the LLM is Meta's **Muse
Spark**, through Meta's Model API (`https://api.meta.ai/v1`, OpenAI-compatible chat completions).

- **The Job Architect** (`architect.py`): you describe what you want in plain English. Muse Spark writes 40–150
  example chats and picks the settings (base model, LoRA rank, budget). Code then checks every value against what
  `vibehash train` and the chain would refuse, plus what would quietly spoil the job: duplicate examples, replies too
  long for `--max-len`, test prompts it also trained on. If a check fails, Muse Spark is told what was wrong, up to
  twice. The architect then writes `agent-jobs/<job>.jsonl` (the chats) and `<job>.eval.json` (new prompts for
  testing), and prints or runs the `vibehash train` command.
- **The Semantic Judge** (`judge.py`): once the network has trained the job, `answer` runs the base model and the
  fine-tune (the base plus the adapter `vibehash job download` wrote) on the prompts it never trained on. `grade` has
  Muse Spark compare each pair blind (it sees "A" and "B" in a shuffled order) and counts who won.

They run from a checkout of this repository and aren't part of the `vibehash` package.

## A Meta API key

1. Go to https://dev.meta.ai, sign in, open the **API keys** tab and click **Create API key**. Copy the key: it's
   shown once.
2. Set up billing on the same account. Without it, calls fail with "can't pay" (402). The Model API is a public
   preview, and which countries can use it is up to Meta (US developers first). From elsewhere you may get a 403.
3. Put the key in the environment, never in a file:

```powershell
$env:MODEL_API_KEY = "paste-the-key-here"          # PowerShell: this window only
```

```bash
export MODEL_API_KEY="paste-the-key-here"          # bash
```

That's all Muse Spark needs. `MODEL_API_KEY` is the name Meta's docs use, and `META_API_KEY` works too. The default
model is `muse-spark-1.1`. Meta recommends `muse-spark-1.3` for new work at the same price, so you can set
`LLM_MODEL=muse-spark-1.3`. `LLM_REASONING_EFFORT=low` makes replies faster and cheaper.

Any other OpenAI-compatible provider works too: set `LLM_BASE_URL` (the part before `/chat/completions`),
`LLM_API_KEY` and `LLM_MODEL`. These also override the Meta defaults. A Meta key is only ever sent to Meta.

## Tonight's run

You'll need Python with this repository's requirements, plus `pip install -e ".[finetune]"` so `train` can read
chats. The `answer` step also needs PyTorch, transformers and peft; this PC's GPU venv,
`%USERPROFILE%\.vibehash\venv`, has them. The public node can't run fine-tunes until its upgrade, so use a local
testnet:

```bash
# window 1: a node and three GPU workers
python -m vibehash testnet --job none --workers 3 --device cuda

# window 2, from the repository folder
python agents/architect.py "answer customer-support questions in two friendly sentences" --post --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json
#   it writes agent-jobs/<job>.jsonl, then `vibehash train` shows its pre-flight and asks "Post it?":
#   open the .jsonl first if you like, and answer n to post nothing
#   (without --post it only writes the files and prints the train command, to run later from this folder)
python -m vibehash job watch                                  # follow the rounds
python -m vibehash job download <job> --out <job>-lora        # the adapter, once some rounds have trained
python agents/judge.py answer agent-jobs/<job>.eval.json <job>-lora
python agents/judge.py grade agent-jobs/<job>.answers.json
```

**`--post` spends VIBE** on the node it posts to: the `--node` you give, or else the one `vibehash setup` saved, or
else the public network. `vibehash train` shows its pre-flight and asks "Post it?" before anything is spent (`--yes`
skips the question). The budget goes into escrow when the job is posted and pays only for work that passes its
checks. `vibehash cancel-job <job>` returns what's left. On a local testnet the VIBE is play money.

`answer` loads the base model at the commit the adapter pins: transformers downloads it once (1.5 GB for Qwen3-0.6B).
To use a copy you already have, add `--base <folder>`. It runs on the GPU if there is one; `accelerate` isn't needed.
Both models get the prompt exactly as training rendered chats. For Qwen3 that means the empty think block before the
reply (its non-thinking mode), and decoding is greedy. Then `grade` writes `<job>.graded.json` next to the answers,
with every verdict, the judge model's name and the shuffle's seed.

## What it costs

Muse Spark's list price is $1.25 per million tokens in and $4.25 per million out, and its thinking counts as output.
One design is roughly 5–15 thousand tokens out, so a few cents; each retry costs about as much again. Grading 10
prompts is about 10 short calls, another few cents. Plan on under $0.25 a job. Meta's rate limits (3,000 requests a
minute) don't matter at this size. The VIBE side is the budget the architect picks: at most 200, usually 40–80, at 10
VIBE a round.

## Checking it

```bash
python -m unittest tests.test_agents
```

These tests run offline with no key. A fake OpenAI-compatible server plays the model. `vibehash train` reads the
architect's files up to its pre-flight against a fake Hugging Face. And where transformers and peft are installed, a
tiny model runs the judge's `answer` step.
