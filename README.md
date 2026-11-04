# Vibe Hash (VIBE)

A proof-of-useful-work cryptocurrency. Instead of hashing, "miners" train AI
models in small, verifiable chunks sized for mid- and low-end hardware. Every
verified chunk is paid pro rata. There's no lottery, so an old laptop earns the
same rate per FLOP as a gaming rig.

This is a **testnet implementation**: validator nodes that take turns leading
rounds and vote on every block (BFT consensus, so no single node is in charge),
real worker clients that talk to them over HTTP, wallets, and a persistent
chain that anyone can audit.
See [DESIGN.md](DESIGN.md) for the protocol, economics, threat model, and
what's still needed before a mainnet (§9). The [whitepaper](docs/whitepaper.pdf)
sets out the design in eleven pages.

## Install

To start mining, one line installs Vibe Hash and starts the guided setup,
which asks the rest (network, wallet password, GPU support). On Windows, in
PowerShell (it offers to install Python too, if you need it):

```powershell
irm https://vibehash.org/install.ps1 | iex
```

On macOS or Linux:

```bash
curl -fsSL https://vibehash.org/install.sh | sh
```

The scripts are [install.ps1](install.ps1) and [install.sh](install.sh). Run
them again any time to update. A PyTorch that's already installed, such as a
rented GPU's, is kept exactly as it is: the script never reinstalls or
downgrades it, and setup's GPU step sees it works and installs nothing.

**On a rented GPU (RunPod).** Use a template with PyTorch, open its terminal
(SSH or Jupyter), and run the macOS/Linux line above. A pod keeps only its
`/workspace` volume when it restarts, so on a pod Vibe Hash keeps its settings
and wallet in `/workspace/.vibehash` by itself (RunPod sets `$RUNPOD_POD_ID`;
an earlier setup in `~/.vibehash` is kept where it is). After a restart, run the
installer again: Vibe Hash itself was on the part that's wiped, but the wallet
is still there. Anywhere else, `VIBEHASH_HOME=/some/folder` puts everything in
that folder instead.

**With nobody at the keyboard** (`nohup`, a tmux session left running, a
Docker `CMD`), nothing waits for an answer. The wallet password comes from
`$VIBEHASH_PASSWORD`, for setup and for every `worker` start; without it, setup
and the worker stop at once and say so. A new wallet's 24 recovery words are
then not printed into the log, which other people may be able to read: they go
to `recovery-words-ADDRESS.txt` in the settings folder, readable only by you,
and the log says where. Copy them somewhere offline and delete the file.
`setup --deepseek yes` (or `no`) answers the DeepSeek question in advance, and
`--yes` installs GPU support without asking:

```bash
VIBEHASH_PASSWORD='choose-one' nohup vibehash setup --yes --deepseek no > setup.log 2>&1
```
```bash
VIBEHASH_PASSWORD='choose-one' nohup vibehash worker --plain > worker.log 2>&1 &
```

Or install it yourself: Python 3.10+. Vibe Hash is on PyPI:

```bash
python -m pip install --upgrade vibehash
```

To work on the source instead, install the project folder in editable mode so
your changes apply without reinstalling. Run this in the project folder:

```bash
python -m pip install -e .
```

Either adds a `vibehash` command (also available as `python -m vibehash`) and
pulls in NumPy and `cryptography`.

## Check a machine

Before mining on a new computer, or when something seems off, run the self
test. In a minute or so, with no wallet needed and nothing spent, it says
whether this computer can mine and what it can take:

```bash
vibehash selftest
```

It checks, one line each:
- **software**: Python, NumPy, `cryptography`, and whether a newer Vibe Hash
  is out;
- **devices**: every GPU's self-test, its memory and how much is free, and
  whether PyTorch is built for what the graphics driver supports;
- **training, for real**: a unit of the public network's fruit-demo job (its
  exact settings and data) on the device the worker would use, checked the way
  verifiers check it, with every segment replayed on the CPU within the job's
  tolerance. A broken driver or a setup that doesn't repeat its results fails
  here. With PyTorch and a GPU, also a fine-tune step (LoRA on a tiny model),
  replayed on the CPU;
- **network**: the node answers (and how fast), its network, block and round,
  jobs that are waiting and what that means for you, this computer's clock
  against the node's (signed messages more than 2 minutes off are refused),
  download speed, and whether huggingface.co answers (fine-tunes fetch their
  weights there);
- **wallet**: whether setup has run, the saved node and wallet, its balance
  and stake, and roughly how long earning the stake takes at the network's
  recent pace;
- **disk**: room for fine-tunes' weights and for DeepSeek's model files.

It ends with what this computer can take tonight (CSV jobs on its GPU or CPU,
Qwen3-0.6B and Qwen3-4B fine-tunes by GPU memory, the DeepSeek pilot) and the
command to run next. It exits with status 0 only if the computer can mine.
`--node ADDRESS` checks against another network, `--device cuda:1` a
particular GPU, `--offline` skips the network, and `--json` is for scripts.
`--model qwen3-0.6b` also trains one real unit of that model, timed, with its
peak GPU memory; it downloads the weights first (1.5 GB, kept for its
fine-tune jobs), so it asks, unless you add `--yes`. The menu has it as
"Check this computer".

**Several GPUs in one computer or pod.** One worker trains on one GPU. Run a
worker for each, each with its own wallet, so each earns and stakes on its own
(two workers with one wallet would take the same units and undo each other's
work, so the second one refuses to start):

```bash
vibehash wallet create --out gpu1.json
```
```bash
vibehash worker --device cuda:1 --wallet gpu1.json
```

Each wallet gets its own cache (`~/.vibehash/cache/ADDRESS`: checkpoints,
datasets, receipts). Model weights are shared: when two workers need the same
model, one downloads it while the other waits for it.

## Quick start

Try a whole network on one computer: a node, two workers and a training job.
It needs no wallets, files or flags, prints what the workers earn each round,
and carries on where it stopped the next time you run it:

```bash
python -m vibehash testnet
```

To watch validators run it instead of one node, give it more validators. Each
gets its own port, they take turns leading rounds, and every block needs votes
from validators holding more than two thirds of the stake. Add `--reset` if
you've run a testnet before, since the number of validators is fixed when the
testnet is first made:

```bash
python -m vibehash testnet --validators 4
```

To join the public network as a worker, run the guided setup, then start
working. Setup connects to the public network's node, `node.vibehash.org`, by
itself, and you start with no VIBE: your first 5 VIBE of pay becomes your
stake.

```bash
python -m vibehash setup
```
```bash
python -m vibehash worker
```

To join a different network instead, give setup its node's address, such as
`python -m vibehash setup --node 192.168.1.20`.

Or skip the commands: `python -m vibehash` on its own, at a terminal, opens a
menu of what people do most. Pick with the arrow keys or a number. Each item
shows the command it runs, and you're back in the menu when it's done.

```bash
python -m vibehash
```

To train a model, save your examples as a CSV file (a header row, numeric
input columns, and the column to predict last), then:

```bash
python -m vibehash train my-data.csv --budget 100
```
```bash
python -m vibehash job watch
```
```bash
python -m vibehash job download
```
```bash
python -m vibehash predict new-rows.csv
```

`train` sizes a model to the data, scales the inputs, holds back a fifth of
the rows to measure accuracy, and prints what it will train and what it
costs before posting. In a terminal it then follows the job live until the
budget is spent: a forge where the model is a blade that grows with its
accuracy, accuracy and loss charted round by round, the units accepted, the
budget left, and whether it's the job's turn to train. Ctrl+C stops watching,
not training; `job watch` picks it up again, `job status` prints a
summary and `job history` the rounds one by one. `job download` saves the
model as one file (`my-data-model.npz`, with how it was trained beside it),
and `predict` labels new rows (`--out predictions.csv` saves them all).
`python -m vibehash example` writes a sample, `fruit.csv`, to try first. On
your own testnet, pay with its requester wallet by adding
`--node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json` to
`train`. Jobs take turns a round at a time, so a new one starts at the next
round.

**Fine-tuning open-weight models is coming, not live yet.** To see what's
planned, list the models:

```bash
python -m vibehash models
```

It lists popular open-weight models (Qwen, Llama, Gemma, Mistral, Phi,
gpt-oss, GLM and DeepSeek), smallest first, with what Hugging Face says about
each: parameters, experts, weight size, licence and whether it's gated. It
also shows how many graphics cards of 12, 24 or 32 GB a team would need to
hold the model. The answers are cached for a day; `--refresh` asks again, and
`--json` is for scripts. Where Hugging Face is out of reach it gives up after
one short timeout and shows what it said last time. `train --base qwen3-0.6b` takes a name from that
list, an `org/name` or a huggingface.co address, shows what fine-tuning that
model would take, and stops there: it posts nothing and spends nothing. Run
in a terminal with no data file, `train` offers the same choice as a
numbered menu, with today's CSV model first. A data file without `--base` is
read as a table, as before. When a fine-tune trains, its dashboard forges the
model maker's emblem in place of the blade (`scripts/preview_forge.py NAME`
shows one now).

**Fine-tuning** a model the list marks `ready` (one GPU holds it, and it's a
Llama, Mistral, Qwen or SmolLM) works on any network whose node runs this
version: workers train, check and score, and the node needs no GPU. That means
a local testnet today, and the public network once its node is updated:

```bash
pip install "vibehash[finetune]"          # the tokenizer and chat templates, for posting
python -m vibehash testnet --job none --workers 3 --device cuda
python -m vibehash train chats.jsonl --base qwen3-0.6b --node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json
python -m vibehash job download           # a LoRA adapter for transformers, PEFT and vLLM
```

`chats.jsonl` holds one chat a line, `{"messages": [{"role": "user", ...},
{"role": "assistant", ...}]}`, or `{"text": ...}` documents. The chats go
through the model's own chat template and tokenizer, and only the assistant's
replies are learned. Workers need PyTorch and a GPU, and fetch the weights
from Hugging Face once. [FINETUNING.md](FINETUNING.md) has the details, a run's
results, and what the public network still needs.

Disclaimer: All product names, logos, and brands are property of their
respective owners and are used purely for identification. Vibe Hash isn't
affiliated with or endorsed by them.

In a terminal, `worker` and `testnet` show a live mining dashboard: a drill
boring through rock while you train, the round and phase, your speed, units
mined, and VIBE earned as it lands. Add `--plain` to these, `train` or
`job watch` for the classic log lines (piped output and services get those
automatically; `NO_COLOR` turns colours off).

The dashboard's `job:` line, under the round, picks what to mine: auto
(every job your computer can do, the default) or one of the jobs with budget
left, whose rounds alone the worker then takes, sitting the others out. Tab,
`j`/`k` or the arrow keys move along it, `a` or `0` is auto, and `1`-`9` pick a
job by its number. Without the dashboard (`--plain`, a pod),
`vibehash worker --job fruit-demo` does the same; a job the node doesn't list
stops the worker at once, naming the jobs it does.

`setup` saves the node address, creates your wallet (encrypted with a password
you choose, in `~/.vibehash/wallet.json`), finds your GPU and offers to install
the PyTorch build for it, and offers to stake once you hold VIBE. Every
command after that uses the saved node and wallet, so `--node` and `--wallet`
are only needed to override them. Re-running `setup` keeps what's done.

The new wallet comes with **24 recovery words**, shown once. Write them down
and keep them offline: they bring the wallet back if the file or its password
is lost, and anyone who sees them can take its coins:

```bash
python -m vibehash wallet restore
```

`wallet restore` asks for the words (the first four letters of each are
enough) and a new password. Add `--passphrase` if you gave the words one, and
`--out FILE` to restore somewhere other than the default wallet.

**GPUs.** `vibehash install-gpu` does the GPU step on its own. It reads the
card and driver (NVIDIA through `nvidia-smi`, Intel Arc, Apple silicon, AMD
with ROCm on Linux), checks which PyTorch builds are currently published for
your driver and Python version, installs the newest one that fits, and
self-tests it. `vibehash devices` shows what's usable. Workers use the first
GPU they find; `--device cpu|cuda|cuda:1|xpu|mps` overrides it. One worker uses
one GPU: for more, see "Several GPUs" above.

## Evo mode: training by evolution, the chain is the model

Evo jobs train a model by evolution strategies. Each round, miners answer puzzles with
slightly twisted copies of the model, and a block keeps only the twists' seeds and how
they scored. Every node works the model out from those, so the trained model is a
function of the chain: about 2 KB a round, no weight files.

Evo mode runs on a local test network for now; the public network doesn't take Evo jobs
yet. It needs a GPU with PyTorch (`vibehash install-gpu`) and `pip install vibehash[evo]`.

```
vibehash testnet --evo2
```

This is Evo Hash ([white paper](docs/evo-hash.pdf); also in the menu: run `vibehash` and pick it under Mine). It runs a node
and two miners on this computer, training Qwen2.5-0.5B-Instruct on one task after another,
19 of them, then round again: Countdown puzzles and arithmetic, made up as it goes, and
science, commonsense, word-problem math, trivia, sentiment, news and emotion questions from
16 public datasets (downloaded once, about 1 GB, each pinned to one version so every
computer reads the same questions). Each job is
short, because evolution makes most of its gains on a task in the first dozen rounds, and
each starts where the last one ended, so the model keeps what every task taught it. The
dashboard shows each generation incubating in the open block. `--evo2-rounds N` sets a
job's length (default 12). `vibehash testnet --evo` is the first, simpler version: one layer
of the model, Countdown only.

To take the tuned Qwen out, while the testnet runs (or after starting it again: it carries on
where it stopped), in another terminal:

```
vibehash evo export
```

It works the newest Evo job's model out from the chain the way a miner does, every round
checked against its block, and saves it as an ordinary Hugging Face folder (`JOB-model/`:
safetensors in bfloat16, the tokenizer and chat template, and an `EVO.md` saying which chain,
job and block it is). Name a job to take an earlier one (`vibehash evo export evo2-c1-arc_easy`),
`--out DIR` for the folder, `--float32` for unrounded weights. A line a miner on this computer
saved makes it quick; on another computer it replays every round on the CPU (about 5 s a round),
and needs Qwen2.5-0.5B-Instruct in its Hugging Face cache. `vibehash job download JOB` does the
same for an Evo job.

## Logs and records

What a worker or a job poster does is also kept in files, so a night of
mining can be looked at afterwards by someone who wasn't watching. They're in
`~/.vibehash` (or `$VIBEHASH_HOME`), and they're written whether the dashboard
is on or not: with `--plain`, piped, in tmux or over SSH.

- `logs/worker-DATE.log` (DATE is the day the worker started): what the
  mining dashboard's feed shows, one line each with the time. A file stops at
  10 MB and starts again, keeping the one before as `worker-DATE.1.log`, and
  only the last 14 days are kept.
- `logs/rounds.jsonl`: a line for each round a worker took part in. It names
  the computer (`host`), the wallet (`worker`), the node and chain, the round
  and job and what kind of job it was. Then what happened: units given,
  trained, committed, revealed and accepted, and why not when the node said;
  checks done; seconds spent training and uploading; what the round's block
  paid (`earned_vibe`, `paid`) or took (`slashed`); and the balance and stake
  after it. `outcome` says it in words. Each line is one JSON object, so a
  text editor shows it, and files from several computers can be put together.
- `vibehash worker` prints where these go when it starts, and a summary when
  you stop it: rounds, units accepted, VIBE earned. `vibehash testnet` also
  keeps its node's log in `node.log` in the testnet's folder.
- `jobs/JOB/history.csv` (and `history.jsonl`): a job's rounds one by one:
  round (its block's height), time, units accepted, loss and accuracy on the
  held-back data, VIBE paid and budget left. `train` and `job watch` keep it up
  to date while they follow the job. `job history` fetches it from the node for
  any job, including one posted from another computer, prints the latest
  rounds and saves the rest; `--csv FILE` writes it where you want it, and
  `--json` prints all of it for scripts:

```bash
python -m vibehash job history fruit-demo --csv fruit-demo.csv
```

`job download` writes the same history beside the model it saves
(`my-data-model-training-history.csv`, or `training-history.csv` inside a
fine-tune's adapter folder), with a short `TRAINING.md` saying how it was
trained: the base model, the data, the rounds, accuracy or loss from the first
round to the last, the VIBE spent, the node and chain, and the dates. Every
fact in it comes from the node or from the receipt `train` kept.

## Publishing a release

Releases publish themselves from GitHub. `.github/workflows/publish.yml` runs
when a version tag is pushed: it checks the tag matches `pyproject.toml`, runs
the fast tests, builds the wheel and source archive, and uploads them to PyPI
through a **trusted publisher**, so no API token is stored anywhere. PyPI's
trusted publisher for `vibehash` names owner `AlphaChad5000`, repository
`vibehash`, workflow `publish.yml` and environment `pypi`.

1. Bump `version` in `pyproject.toml` (PyPI never accepts the same version
   twice; 0.5.10 is out), commit it, and push.
2. Tag the commit with that version and push the tag. This publishes it:

```bash
git tag v0.5.11
```
```bash
git push origin v0.5.11
```

Follow the run under the repository's Actions tab. To try a build locally
first, `python -m build` writes the packages to `dist/`.

## Run a public node

`deploy/setup-node.sh` turns a fresh Ubuntu 24.04 server into a network's
public node:
- it installs Vibe Hash from PyPI and creates the node's wallet (write down
  the recovery words it shows);
- it writes the genesis: this node as the first validator, with bootstrap
  stake, and workers earning their stake in;
- it gets a Let's Encrypt certificate and serves HTTPS on port 443, with the
  node's own per-client rate limits;
- it runs the node as a service that restarts itself.

This is how `node.vibehash.org` runs, on a small DigitalOcean server (2 GB).
First point the domain's **A** record at the server's IPv4 address. The node
listens on IPv4 only, so give it no AAAA record. Then copy the script over and
run it:

```bash
scp deploy/setup-node.sh root@SERVER-IP:
```
```bash
ssh root@SERVER-IP bash setup-node.sh --domain node.vibehash.org --fund YOUR-WALLET-ADDRESS=10000
```

`--fund` gives one wallet test VIBE to post the first training jobs, since
with earn-in nobody else starts with any. Running the script again upgrades
Vibe Hash and restarts the node, keeping its wallet and chain. Read the log
with `journalctl -u vibehash-node -f`. The node's files are in
`/var/lib/vibehash`, and its wallet password is in `/etc/vibehash/env`:
include both in backups.

## Run a network for other people

`vibehash testnet --host 0.0.0.0` is the quick way: other computers on your
network can join it with `setup --node YOUR-IP`, and it prints how to send
them VIBE to stake. Its wallets are unencrypted test keys, so for anything
longer-lived, set the network up by hand:

The commands below use bash syntax; on PowerShell set environment variables
with `$env:VIBEHASH_PASSWORD = "..."` instead.

1. Make wallets: one for the node operator, one per worker, one for whoever posts jobs. Each prints its 24 recovery words once.

```bash
python -m vibehash wallet create --out node.json
```
```bash
python -m vibehash wallet create --out alice.json
```
```bash
python -m vibehash wallet create --out requester.json
```

2. Write a genesis file (use the addresses printed above). `--validator` names the node as a validator, with the address other machines reach it at, and `--bootstrap` gives it **bootstrap stake**: enough to vote on block one, but it can never be withdrawn or spent, and it's burned if the validator leaves. Workers need nothing. By default they start with no stake and **earn it in**: their first 5 VIBE of pay goes into their stake, and until then they get small batches of work and don't check anyone else's. So every spendable coin starts from work. `--no-earn-in` makes workers be given stake instead (`--stake`). `--economics mainnet` picks the long issuance schedule (3.125 VIBE a round, halving about every 4 years, 21M VIBE at most) instead of the fast testnet one. The genesis hash commits to all of it. Here only the requester is handed coins, to pay for training:

```bash
python -m vibehash genesis create --chain-id my-testnet-1 --out genesis.json --bootstrap NODE_ADDR=10 --validator NODE_ADDR=http://192.168.1.20:8545 --alloc REQUESTER_ADDR=500
```

3. Start the node. Use `--host 0.0.0.0` to accept workers from other machines.

```bash
python -m vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0
```

   For more validators, give each its own staked wallet and its own `--validator ADDR=URL` in the genesis, and start one node per wallet on its own machine, all with the same `genesis.json`. They find each other from the addresses in it. A staked account can also join a running network as a validator. Its node announces the address where other machines can reach it:

```bash
python -m vibehash validator join --url http://203.0.113.7:8545 --wallet bob.json
```
```bash
python -m vibehash node --genesis genesis.json --wallet bob.json --data-dir bob-data --host 0.0.0.0 --announce http://203.0.113.7:8545 --peer http://192.168.1.20:8545
```

   A node whose wallet isn't a validator, started with `--peer`, follows the chain: it checks and serves every block, and points workers at whichever validator is leading. `vibehash validator list` shows the validators and their share of the votes.

4. Each worker runs setup with the node's address and their wallet, then starts contributing. With earn-in there's nothing to stake first.

```bash
python -m vibehash setup --node http://127.0.0.1:8545 --wallet alice.json
```
```bash
python -m vibehash worker
```

5. Post a training job: from a CSV with `train`, or with full control over every setting from a job file and a prepared `.npz` dataset.

```bash
python -m vibehash train my-data.csv --node http://127.0.0.1:8545 --wallet requester.json --budget 150
```
```bash
python -m vibehash post-job --node http://127.0.0.1:8545 --wallet requester.json --spec examples/job.json --budget 150 --dataset examples/demo-dataset.npz
```

6. Check balances and independently audit the whole chain. `chain verify` re-checks every block from genesis, checks each block's commit certificate (the validators' signatures that made it final), and replays every round from the record its block commits to, so it catches a node that favoured a worker, invented a registration or forged a verdict, not just bad arithmetic. It also checks the signed receipts workers on this computer received for their messages (or any receipts file with `--receipts FILE`), which prove a node that accepted a message and then left it out of the round. For the most recent rounds (the node keeps their checkpoints for `--keep-rounds`, 10 by default) it downloads the checkpoints and re-runs every check that needs the model weights, so it catches a node that misreported a sanity check or sided with a false accusation.

```bash
python -m vibehash balance --node http://127.0.0.1:8545 --wallet alice.json
```
```bash
python -m vibehash chain verify --node http://127.0.0.1:8545 --genesis genesis.json
```

A dataset is an `.npz` with `x_train`, `y_train`, `x_val`, `y_val`. It's never
unpickled, and workers check its sha256 against the job before using it.

### Fees, HTTPS and rate limits

**Fees.** Every transaction pays a small fee to the validator that proposes
its block. A node relays transactions paying at least `--min-fee` (0.0001 VIBE
by default; `--min-fee 0` relays free ones). `send`, `stake`, `unstake`,
`post-job` and `cancel-job` pay the node's minimum unless you give `--fee`,
and `balance` shows it. The fee comes out of your spendable balance, so keep
a little unstaked VIBE to unstake with. Worker messages are free. When the
mempool is full, higher fees get in first.

**HTTPS.** A node can serve HTTPS itself. For a testnet, make a self-signed
certificate for the names and addresses workers will use, and start the node
with it:

```bash
python -m vibehash tls cert --host 192.168.1.20
```
```bash
python -m vibehash node --genesis genesis.json --wallet node.json --data-dir node-data --host 0.0.0.0 --tls-cert node-cert.pem --tls-key node-key.pem
```

Both print the certificate's fingerprint (`vibehash tls fingerprint
node-cert.pem` shows it again). Workers pin it in setup, and every later
command then trusts that exact certificate from that node and nothing else:

```bash
python -m vibehash setup --node 192.168.1.20 --tls-fingerprint FINGERPRINT
```

With a certificate from a real CA, plain `setup --node https://...` works.
`--tls-ca FILE` trusts a private CA instead, and `$VIBEHASH_TLS_FINGERPRINT`
or `$VIBEHASH_TLS_CA` set the trust without setup. Without the `cryptography`
package, `tls cert` prints the openssl command to run instead.

**Rate limits.** Each client address gets its own budget of reads, worker
messages, transactions and uploads, at most 32 connections (256 in all), and
socket timeouts that cut off slow-loris clients. Over budget, a node answers
429, and clients wait and retry. Clients on the node's own computer are
never limited, so a local testnet is never throttled. Behind a reverse proxy
every client looks local, so limit at the proxy or add `--rate-limit-local`.
`--max-job-mb` (64 by default) stops a node relaying jobs too big for its memory. `--rate-limit off` turns the budgets off, `--rate-limit tx=1/10,read=100`
changes some (requests a second and burst; the buckets are read, worker, tx,
upload and other), and `--max-connections N` sets the overall cap. DESIGN.md
§10 has the numbers.

## Simulate the economics

Runs the same round logic as a real node, with 14 simulated rigs (7 of them
cheating), in a few seconds:

```bash
python simulate.py --rounds 20
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

`tests/test_network.py` starts a real node on localhost with four worker
clients (one cheating), uploads a dataset, runs three rounds, audits the chain
from the outside, and restarts the node from disk.
`tests/test_validators.py` runs four validators on localhost:
- they take turns leading rounds and outvote a leader that pays itself;
- they carry on without one that goes offline, and it catches up when it's back;
- a new node syncs the whole chain and refuses a block whose certificate is
  short of two thirds of the stake.

`tests/test_finetune.py` fine-tunes tiny made-up models of each supported
family: against Hugging Face transformers and PEFT when they're installed,
through rounds with cheaters, and over HTTP from `train --base` to `job
download` against a fake Hub. `VIBEHASH_REAL_MODELS=1` adds Qwen3-0.6B itself
(a 1.5 GB download): against transformers, and replayed on the CPU after
training on CUDA.
`tests/test_consensus.py` puts the voting itself through crashed, lying and
double-voting validators, lost messages and a split network, on a simulated
clock. `tests/test_hardening.py` covers fees, issuance schedules, HTTPS
nodes, rate limits and recovery words, including the official BIP39 and
SLIP-0010 test vectors. `tests/test_catalog.py` checks the model catalog,
`models` and `train --base` against a fake Hugging Face Hub, and that
`train --base` never reaches the node or the wallet. `tests/test_selftest.py`
runs `selftest` against an in-process node with made-up GPUs (one whose
results drift, one that fails its self-test), a clock five minutes off and a
node out of reach, and checks the advice, one worker per wallet and the pod
rule; `tests/test_setup.py` runs `setup` and `worker` with no terminal.

## Layout

| File | What it does |
|---|---|
| `vibehash/crypto.py` | Ed25519 (OpenSSL, or RFC 8032 pure-Python fallback), hashing, Merkle trees |
| `vibehash/model.py` | Reference workload: multi-layer MLP with deterministic, replayable SGD (float32 or float64); dataset loading |
| `vibehash/compute.py` | Compute backends: NumPy on CPU, PyTorch on NVIDIA / Intel / Apple / AMD GPUs; device detection |
| `vibehash/tasks.py` | Job specs (validated, stored on chain), work units, commitments, reveals, verdicts |
| `vibehash/verification.py` | Commit–reveal, Merkle openings, sanity checks, replay spot-checks |
| `vibehash/economics.py` | Issuance schedules (testnet, mainnet), reward split, reputation, slashing |
| `vibehash/ledger.py` | Transactions, blocks, state, unbonding; derives rewards/slashes from round summaries |
| `vibehash/fees.py` | Transaction fees: the relay minimum, the fee-ordered mempool, eviction |
| `vibehash/coordinator.py` | The round state machine, shared by the node and the simulator |
| `vibehash/worker.py` | Training, committing, revealing, verifying (plus cheat modes for tests) |
| `vibehash/node.py` | HTTP node: API, wall-clock round driver, leading rounds and committing blocks validators agree on |
| `vibehash/consensus.py` | Tendermint-style BFT voting among validators; commit certificates |
| `vibehash/p2p.py` | Peers: gossiping votes and transactions between nodes |
| `vibehash/replay.py` | Re-runs a block's round from its record: how validators check a proposal |
| `vibehash/receipts.py` | Signed receipts that prove a node dropped a message it accepted |
| `vibehash/ratelimit.py` | Per-client rate limits, connection caps and timeouts for the node's server |
| `vibehash/tls.py` | HTTPS: server contexts, self-signed certificates, certificate pinning for clients |
| `vibehash/client.py` | HTTP client and the worker loop, with live `WorkerStats` |
| `vibehash/records.py` | What's kept on disk: workers' event logs and a line per round, a job's history, the page beside a downloaded model |
| `vibehash/minerui.py` | The terminal mining dashboard (plain ANSI, ASCII fallback) |
| `vibehash/trainui.py` | The terminal training dashboard: a job followed live from the node, politely, with a log-line mode |
| `vibehash/emblems.py` | Pixel-art emblems of the model makers, which the training dashboard forges instead of a blade for a fine-tune (`scripts/preview_forge.py NAME` plays one) |
| `vibehash/termui.py` | What the two dashboards share: palette, the mine's rock, banner, phase bar, event feed, the alternate screen |
| `vibehash/tabular.py` | CSV files to training datasets, and downloaded models to predictions |
| `vibehash/catalog.py` | The open-weight models `models` lists and `train --base` describes: live facts from Hugging Face, cached, the GPUs a team would need, and which are ready to fine-tune. `scripts/bake_models.py` writes them into the guide |
| `vibehash/llm.py` | Pretrained Llama, Mistral, Qwen2, Qwen3 and SmolLM3 models in PyTorch, with LoRA: what fine-tunes train |
| `vibehash/finetune.py` | Fine-tune jobs: the spec, token datasets, chats through the model's chat template, Adam on the adapters, PEFT export |
| `vibehash/hub.py` | Hugging Face: pinned commits, manifests of tensor hashes, verified downloads by range into a cache |
| `vibehash/data/` | The `fruit.csv` sample `vibehash example` writes, and the BIP39 word list |
| `vibehash/storage.py` | SQLite chain and weights, content-addressed datasets |
| `vibehash/blobs.py` | Content-addressed, chunked, resumable binary storage for checkpoints and datasets |
| `vibehash/wire.py` | JSON wire format and signed message envelopes |
| `vibehash/wallet.py` | Key files: scrypt + AES-256-GCM |
| `vibehash/mnemonic.py` | Recovery words: BIP39 mnemonics and SLIP-0010 Ed25519 key derivation |
| `vibehash/hardware.py` | RAM detection and a workload benchmark |
| `vibehash/cli.py` | The `vibehash` command, including `setup` and `testnet` |
| `vibehash/selftest.py` | `vibehash selftest`: whether this computer can mine (a real training unit checked as verifiers check it, the node, the clock, disk) and what it can take |
| `vibehash/advice.py` | What went wrong, in a line or two a newcomer can act on: a node out of reach, a clock that's off, a GPU out of memory, a full disk, a job waiting for staked workers |
| `vibehash/unattended.py` | Running with nobody at the keyboard: no questions that can't be answered, recovery words to a private file, one worker per wallet |
| `vibehash/filelock.py` | Locks between processes on one computer, so workers sharing it don't fetch the same model into the same files |
| `vibehash/settings.py` | `~/.vibehash`: the default wallet, and the node address and its TLS trust that `setup` saves |
| `vibehash/gpusetup.py` | GPU detection before PyTorch is installed; picks and installs the PyTorch build for the card |
| `vibehash/protocol.py` | In-process driver for simulations |
| `vibehash/shards.py` | Prototype, not yet used by the protocol: a model split into shards across workers, checked one shard at a time ([SHARDING.md](SHARDING.md)) |

## Status

Unaudited testnet software. Don't attach real value to it. DESIGN.md §9 lists
the remaining work:
- validators rechecking weight-based results before voting, with on-chain
  disputes;
- slashing double-signing validators;
- replicating fraud evidence;
- model types beyond MLPs, and models bigger than one computer
  ([SHARDING.md](SHARDING.md) has the design).

[FINETUNING.md](FINETUNING.md) describes fine-tuning open models from Hugging Face with one `train --base`
command. Models one GPU holds fine-tune today on a node that runs fine-tunes (a local testnet); the public
network needs the work listed there first, and models bigger than one GPU need teams of workers.

§10 covers what's been hardened: fees, the issuance schedule, TLS, rate limits
and recovery words.

## License

MIT. See [LICENSE](LICENSE): use, change and share it freely, keeping the copyright notice.

