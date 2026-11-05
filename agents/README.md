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

## What it measured

During the hackathon (Oct 26 – Nov 4, 2026), three goals were typed into the site's form and run end to end on one
RTX 4060 (8 GB) with the local backend, Qwen3 0.6B as the base; a fourth came from the chat. The runs are on
results.luhlabs.com, from the hackathon repository's `web/public/results/`.
The headline is wins against the same base model told the goal; each run's every answer is on the results page.

The base model is deliberately tiny so a whole run, repairs included, fits on a gaming GPU in under an hour; Meta's
models are the agents, the judges and the auditor, not the student. The base is one line in `agents/models.py`;
Muse Glimmer (30B, Apache 2.0) is the next student, with QLoRA on a 24 GB card.

| Goal | Level | Rounds, wins over the model told the goal | Rules passed, **checked by code** | Off-task messages declined | Held its own against **Muse Spark itself** |
|---|---|---|---|---|---|
| Support tickets → one line of JSON (category, urgency, sentiment, summary) | Standard | 19/20 → **19/20, no losses** (likely 76–99%; about 1 in 500,000 by luck) | one line, the four keys, allowed values: fine-tune **81%**, told the goal 3%, untrained 0% (32 prompts) | 2 of 11 → **5 of 6**; told the goal 2 of 6 | **14 of 20** (14 even, 6 lost); Muse Spark passed every rule and declined 6 of 6 |
| Plain-English request → one Excel formula, nothing else | Standard | 11/20 → 10/18 → **11/19, 8 ties, no losses** (short of the bar) | starts with `=`, one line, balanced brackets: **94%**, told the goal 55%, untrained 0% (31 prompts) | **4 of 4**; told the goal 3 of 4 | 9 of 19 (9 even, 10 lost) |
| Answer like Mark Twain, reason like Plato | Strict | 11/20 → 11/17 → **5/18, and on 8 neither reply did the goal** | — | — | 0 of 18 |
| Only respond with woofs and barks, no matter what (taught from the chat on Nov 4, the dog's model) | Quick | one round: **14/20, 5 ties, 1 loss** (likely 48–86%; about 1 in 2,000 by luck); Muse Wonder read the same pairs: 18/20 | — | — | — |

The last column is the frontier test, beside the bar rather than part of it: Muse Spark, the model that wrote the
training chats, answers the same prompts itself, told the goal, and the same blind judge compares it with the
0.6B fine-tune. The small model matched it on the pairs where the task is the format (identical formulas, the same
JSON, every decline) and lost where judgment is needed (a tricky formula, a sentiment call, anything Twain). Code
scores the big model on the same rules and declines, so the page shows where the gap is, not just that there is one.

- The JSON run shows the repair loop doing its job: round 1 already won 19 of 20, but declined only 2 of 11 off-task
  messages and passed every rule on 72% of replies, both under their bars, so it wasn't published. The Architect
  wrote 20 chats aimed at those misses; round 2 kept 19 of 20, declined 5 of 6 and passed 81%. Winning the
  judge's vote was never the whole bar.
- The Excel run is published short of its bar on purpose: the bar counts wins, and a 0.6B model told the goal
  writes the same `=SUMIFS(…)` for a short request, so eight pairs tied. The fine-tune's edge is on the long briefs
  (every win), the rules and the off-task messages, and it never lost a pair. The Twain run fell short too; the
  repairs didn't help, so it stopped.
- A narrow goal declines what isn't its job: asked for the weather or a poem, both the JSON and the Excel fine-tunes
  answer with the one decline sentence the Architect wrote, and code counts how often each model does. The model
  told the goal is told that sentence too, so the comparison is fair.
- The fine-tune doesn't need the goal sent with every message: the JSON model used **94 prompt tokens a message
  against 201** for the model told the goal (53% fewer), and replied in 1.8 s against 3.4 s on the same GPU; the
  Excel model 99 against 197, in 2.8 s against 7.3 s.
- A second judge, **Muse Wonder**, a different model, read every pair again and agreed with Muse Spark on 20 of 20
  (JSON, Cohen's κ 1.0) and 12 of 19 (Excel, κ 0.45). On Twain it was harsher than Muse Spark: it said neither reply
  did the goal on 15 of 18 pairs. Two models, so the agreement is an independent reading, and the page shows where
  they split.
- Visitors can vote too: the chat shows the fine-tune and the model told the goal as A and B in random order, and
  the tally sits beside the judges' scores.

What didn't work, kept because it's the honest edge of the method: a 0.6B model can learn a format (JSON, one
formula, two sentences) far better than prompting gets it, but not a voice that needs reasoning (the Twain/Plato
run), and not wordplay: asked to open every answer with a pun on a capital city, it learned to say "Lima!" and
then answer, with no pun. The Job Architect now warns about wordplay goals before any GPU time is spent. An early
two-sentence run also scored well by saying "For example, …" in every reply, even to "hi"; the chats are now
checked for stock phrases and the judge marks down forced replies.

## Keeping the scores honest

A fine-tune that beats an untrained model on its own goal proves little: that model was never told what to do. So:

- The headline score is against **the same model told the goal** in its instructions. Beating that means training
  added something prompting didn't.
- The Job Architect writes the test prompts as well as the chats, so a model could be built to pass its own exam.
  Every run is also tested, and scored apart, on prompts it never had a hand in:
  - **hard prompts** (`agents/hard_prompts.txt`): long, demanding requests people wrote, which nothing is trained,
    repaired or chosen by (code scores the rules on them too, but only for the page: the bar's rule count leaves
    them out);
  - **check prompts** (`agents/check_prompts.txt`): the same kind, which Strict stops on, so they're never the
    published hard score;
  - **the judge's own prompts**: two casual, two everyday and two demanding, written fresh after each round's
    training and never shown to the Job Architect.
- Each pair is judged **twice, in both orders**, and only counts when both readings agree.
- The judge can say **neither**: when no reply does what the goal asks at all, the less bad one doesn't win, and the
  pair counts against the fine-tune. A forced choice between two misses used to let a run score without ever doing
  the goal (the Twain run's judge wrote "neither reply uses Twain's styling, but A…" and picked A). The three runs
  above were rejudged under this rule on Nov 4: the Twain run fell from 8 to 5 of 18, with 8 pairs where neither
  reply did the goal; the JSON run didn't move; the Excel run gained a win. The old graded files were kept beside the
  new ones on the GPU machine (`agent-jobs/*.before-neither`).
- The judge won't let style beat a correct answer, and it **flags false or unhelpful replies**; the count is shown.
- Every score comes with its **95% range**: 19 of 20 means about 76–99%, not 95%.
- **Code scores what code can.** The Job Architect picks each goal's rules from a fixed library
  (`agents/checks.py`: `exact_sentences 2`, `no_bullets`, `starts_with "="`, `one_line`, `balanced_brackets`,
  `json_keys [...]`, `line_prefixes [...]`, and so on; it can't invent one). Its training chats must pass them before
  any GPU time is spent; afterwards every reply from all three models is scored on every rule, the results page
  shows each rule's pass rate, a reply that breaks one is named in the repair loop, and Standard and Strict require
  the fine-tune to pass every rule on 80% of its on-task test prompts. The judge is left with what code can't
  measure: whether the reply is true, helpful and natural.
- **A narrow task has to know what isn't its job.** An early Excel-formula model answered "what's the weather" with
  a formula. For a goal that's a task on a kind of input (extract, convert, classify), the Job Architect now states
  the one sentence the model says to anything else, trains it on off-task messages answered with that sentence, and
  writes off-task test prompts; code measures how many the fine-tune declined, the judge is told the scope, and
  Standard and Strict require 75% declined before a run is published.
- No test prompt may be a training chat in disguise: code compares them with case, punctuation and spacing removed.
- Every score also says what **luck** would do: the exact chance a coin flip would win at least that many of the
  decided pairs. 19 of 20 is 1 in 50,000; 10 of 16 could be luck (p = 0.23), and the page says so.
- A **second judge, Muse Wonder**, a different model, reads every pair, and the agreement is shown as a count and as
  Cohen's κ; so are **blind votes from people**.
- Each run has a **receipt**: trained and frozen weight counts, settings, peak GPU memory, and SHA-256 fingerprints
  of the training chats and the adapter.
- With repairs, the **last** round is reported, not the best of them, and every round's score is shown.
- The results page says plainly that Muse Spark wrote the training chats and also judges, which may favour the
  fine-tune, and shows every answer so you can judge for yourself.

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
Face), `MAX_BUDGET` (the most one network job may spend, default and at most 200 VIBE), `JUDGE2_MODEL` (empty to skip the second
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
question). Code sets how long it trains: a unit of network training is 16 steps of 8 chats (128 chats on either
base), so the architect passes `--units` (1 to 8 a round) and a budget of whole rounds that come nearest the plan's
epochs over the chats train keeps (it holds back a tenth). 46 chats at 3 epochs is 1 unit in 1 round, 10 VIBE, about
3 epochs; up to about 400 chats at 3 epochs is still one round of 8 units. The budget is capped by `MAX_BUDGET`,
and the line the architect prints says the rounds, units and epochs it bought. `--budget` sets it by hand (more
rounds, so more epochs), `--fee` the VIBE a round. `--id` names the job yourself and
`--base` picks the model (the LLM picks otherwise, from `ALLOWED_BASES`). It goes into escrow when the job is posted and pays only for work
that passes its checks; `vibehash cancel-job <job>` returns what's left.

`answer` loads the base model at the commit the adapter says it was trained on, without `accelerate`; `--base
<folder>` or `--base org/name@revision` loads another copy, `--device cpu` keeps it off the GPU. It warns when an
adapter's LoRA weights are still all zero (a job that hasn't trained yet).

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
