# The Muse agents

Plain English in, a judged fine-tune out. Two agents built on Meta's **Muse Spark** (through Meta's Model API,
`https://api.meta.ai/v1`, OpenAI-compatible chat completions) sit on either side of a fine-tune:

- **The Job Architect** (`architect.py`) screens the goal, then writes the training chats, picks the settings (base
  model, LoRA rank, epochs), writes test prompts it keeps out of training, and picks the rules a reply must meet from
  a fixed library (`checks.py`). Code checks every value before anything is spent and sends what's wrong back to the
  model to fix; the judge audits a sample of the chats before training too. It trains on one of two backends:
  - **vibehash** (the default): `vibehash train` posts the job to the Vibe Hash network. The budget is worked out by
    code from the chats and epochs, and the network-only checks are what the chain would refuse (chats past
    `--max-len`, chat-template tokens, a gated base without a Hugging Face token).
  - **local**: `train_local.py` trains a LoRA adapter on this machine's GPU, holding back a fifth of the chats and
    keeping the best epoch.
- **The Quality Judge** (`judge.py`) answers the question loss can't: did the model get better at the goal? The
  fine-tune, the base model, and the base model *told* the goal answer the test prompts, two fixed sets of hard
  prompts (`hard_prompts.txt`, `check_prompts.txt`) and prompts the judge writes itself. Muse Spark compares each pair
  blind, twice, once in each order, and only a verdict both readings agree on counts. Code scores the rules. A second
  model (Muse Wonder by default, `JUDGE2_MODEL`) reads every pair again, and the agreement is reported.

Around them: `teacher.py` has Muse Spark answer the same prompts itself and judges it against the fine-tune;
`publish.py` turns a finished job into a run for the results page (`web/public/results/<job>.json`, the site's
folder); `dispatcher.py` takes requests and chat messages from the site and runs them end to end on this machine,
repairing a run until it reaches the level asked for (`quick`, `standard`, `strict`). [SECURITY.md](SECURITY.md) goes
through the OWASP Top 10 for LLM and for agentic applications.

The site (agents.luhlabs.com, results.luhlabs.com) lives in the hackathon repository,
[Luh-Labs-LLC/muse-finetune-agents](https://github.com/Luh-Labs-LLC/muse-finetune-agents), which is the Meta hackathon submission; the agents here
are its `agents/` folder, adapted to this repository. They run from a checkout and aren't part of the `vibehash`
package.

## A Meta API key

1. Go to https://dev.meta.ai, sign in, open the **API keys** tab and click **Create API key**. Copy the key: it's
   shown once.
2. Set up billing on the same account. Without it, calls fail with "can't pay" (402). The Model API is a public
   preview, and which countries can use it is up to Meta (US developers first). From elsewhere you may get a 403.
3. Put the key in `.env.local` at the top of the repository (git ignores it) or in the environment, never in a file
   that gets committed:

```bash
MODEL_API_KEY=paste-the-key-here      # in .env.local; or $env:MODEL_API_KEY = "..." / export MODEL_API_KEY=...
```

That's all Muse Spark needs. `MODEL_API_KEY` is the name Meta's docs use, and `META_API_KEY` works too. The default
model is `muse-spark-1.3`. `LLM_REASONING_EFFORT=low` makes replies faster and cheaper. Any other OpenAI-compatible
provider works too: set `LLM_BASE_URL` (the part before `/chat/completions`), `LLM_API_KEY` and `LLM_MODEL`. A Meta
key is only ever sent to Meta. `python agents/llm_call.py` checks the settings with one short call.

Also read from `.env.local`: `ALLOWED_BASES` (default `qwen3-0.6b`; add `llama-3.2-1b`, which is gated on Hugging
Face), `MAX_BUDGET` (the most one network job may spend, default 200 VIBE), `JUDGE2_MODEL` (empty to skip the second
judge) and, for the dispatcher, `DISPATCH_TOKEN`.

## On the Vibe Hash network

Install this repository with `pip install -e ".[finetune]"` so `vibehash train` can read chats. The judge's `answer`
step also needs PyTorch, transformers and peft (`pip install -r agents/requirements-train.txt`, with PyTorch's CUDA
build from pytorch.org). On a local testnet, where the VIBE is play money:

```bash
# window 1: a node and three GPU workers
python -m vibehash testnet --job none --workers 3 --device cuda

# window 2, from the repository folder
python agents/architect.py "answer customer-support questions in two friendly sentences" --post --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json
python -m vibehash job watch <job>
python -m vibehash job download <job> --out agent-jobs/adapters/<job>
python agents/judge.py answer agent-jobs/<job>.eval.json agent-jobs/adapters/<job>
python agents/judge.py grade agent-jobs/<job>.eval.answers.json
python agents/publish.py <job>                        # optional: the run, for the results page
```

The architect prints these lines with the job's name filled in once it has posted. Without `--post` it only writes
`agent-jobs/<job>.jsonl` (the chats), `<job>.eval.json` (the test prompts) and `<job>.plan.json`, and prints the
train command. On the public network leave out `--node` and `--wallet`.

**`--post` spends VIBE** on the node it posts to: the `--node` you give, or else the one `vibehash setup` saved, or
else the public network. `vibehash train` shows its pre-flight and asks "Post it?" first (`--yes` skips the
question). The budget is enough rounds for the plan's epochs over its chats at 10 VIBE a round (usually 10 to 40),
capped by `MAX_BUDGET`; `--budget` sets it by hand. It goes into escrow when the job is posted and pays only for work
that passes its checks; `vibehash cancel-job <job>` returns what's left.

`answer` loads the base model at the commit the adapter says it was trained on, without `accelerate`.

## On one machine

Python 3.10+, an NVIDIA GPU, PyTorch's CUDA build, then `pip install -r agents/requirements-train.txt`. One goal end
to end, planned, trained, judged and repaired until it reaches the level, then published for the results page:

```bash
python agents/dispatcher.py --goal "answer support questions in two friendly sentences" --rigor strict
```

Or a step at a time:

```bash
python agents/architect.py "answer support questions in two friendly sentences" --post --backend local
python agents/judge.py answer agent-jobs/<job>.eval.json agent-jobs/adapters/<job>
python agents/judge.py grade agent-jobs/<job>.eval.answers.json
python agents/publish.py <job>
```

The adapter lands in `agent-jobs/adapters/<job>/`, with a `train_report.json` holding the held-back loss and accuracy
after every epoch.

## Requests and chat from the site

The site's agents page has a form. A request (a goal and an access code) waits in a queue on the site (a
Cloudflare Worker and D1 database, in the hackathon repository's `web/`) until a dispatcher takes it, runs it end
to end and sends the run back. Chat messages go through the same queue and the same dispatcher, which answers them
between requests. The site's own README has its setup; then, on the GPU machine, with `DISPATCH_TOKEN` in
`.env.local`:

```powershell
powershell -ExecutionPolicy Bypass -File agents/dispatch.ps1     # or: python agents/dispatcher.py
```

`dispatch.ps1` restarts the dispatcher if it stops and keeps the PC awake while it runs.

## What it costs

Muse Spark's list price is $1.25 per million tokens in and $4.25 per million out, and its thinking counts as output.
A plan is a handful of calls: the screen, the design (and up to three rewrites when the checks or the audit send it
back), the audit and a second batch of chats. Grading is four calls a prompt (two rivals, two readings each), plus two
for the second judge, over the test prompts, the fixed sets and the judge's own. Plan on cents for a plan and well
under a few dollars for a fully judged run; repairs repeat the grading.

## Checking it

```bash
python -m unittest tests.test_agents
```

These tests run offline with no key. A fake OpenAI-compatible server plays the model. `vibehash train` reads the
architect's network files up to its pre-flight against a fake Hugging Face, where PyTorch, tokenizers and jinja2 are
installed.
