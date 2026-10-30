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
what's still needed before a mainnet (§9).

## Install

Python 3.10+. Vibe Hash is on PyPI:

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

To train a model, save your examples as a CSV file (a header row, numeric
input columns, and the column to predict last), then:

```bash
python -m vibehash train my-data.csv --budget 100
```
```bash
python -m vibehash job status
```
```bash
python -m vibehash job download
```
```bash
python -m vibehash predict new-rows.csv
```

`train` sizes a model to the data, scales the inputs, holds back a fifth of
the rows to measure accuracy, and prints what it will train and what it
costs before posting. `job status` shows the accuracy climbing round by round,
`job download` saves the model as one file (`my-data-model.npz`), and
`predict` labels new rows (`--out predictions.csv` saves them all).
`python -m vibehash example` writes a sample, `fruit.csv`, to try first. On
your own testnet, pay with its requester wallet by adding
`--node http://127.0.0.1:8545 --wallet ~/.vibehash/testnet/requester.json` to
`train`. Jobs take turns a round at a time, so a new one starts at the next
round.

In a terminal, `worker` and `testnet` show a live mining dashboard: a drill
boring through rock while you train, the round and phase, your speed, units
mined, and VIBE earned as it lands. Add `--plain` for the classic log lines
(piped output and services get those automatically; `NO_COLOR` turns colours
off).

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
GPU they find; `--device cpu|cuda|xpu|mps` overrides it.

## Publishing a release

Releases publish themselves from GitHub. `.github/workflows/publish.yml` runs
when a version tag is pushed: it checks the tag matches `pyproject.toml`, runs
the fast tests, builds the wheel and source archive, and uploads them to PyPI
through a **trusted publisher**, so no API token is stored anywhere. PyPI's
trusted publisher for `vibehash` names owner `AlphaChad5000`, repository
`vibehash`, workflow `publish.yml` and environment `pypi`.

1. Bump `version` in `pyproject.toml` (PyPI never accepts the same version
   twice; 0.5.1 is out), commit it, and push.
2. Tag the commit with that version and push the tag. This publishes it:

```bash
git tag v0.5.2
```
```bash
git push origin v0.5.2
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
`--rate-limit off` turns the budgets off, `--rate-limit tx=1/10,read=100`
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

`tests/test_consensus.py` puts the voting itself through crashed, lying and
double-voting validators, lost messages and a split network, on a simulated
clock. `tests/test_hardening.py` covers fees, issuance schedules, HTTPS
nodes, rate limits and recovery words, including the official BIP39 and
SLIP-0010 test vectors.

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
| `vibehash/minerui.py` | The terminal mining dashboard (plain ANSI, ASCII fallback) |
| `vibehash/tabular.py` | CSV files to training datasets, and downloaded models to predictions |
| `vibehash/data/` | The `fruit.csv` sample `vibehash example` writes, and the BIP39 word list |
| `vibehash/storage.py` | SQLite chain and weights, content-addressed datasets |
| `vibehash/blobs.py` | Content-addressed, chunked, resumable binary storage for checkpoints and datasets |
| `vibehash/wire.py` | JSON wire format and signed message envelopes |
| `vibehash/wallet.py` | Key files: scrypt + AES-256-GCM |
| `vibehash/mnemonic.py` | Recovery words: BIP39 mnemonics and SLIP-0010 Ed25519 key derivation |
| `vibehash/hardware.py` | RAM detection and a workload benchmark |
| `vibehash/cli.py` | The `vibehash` command, including `setup` and `testnet` |
| `vibehash/settings.py` | `~/.vibehash`: the default wallet, and the node address and its TLS trust that `setup` saves |
| `vibehash/gpusetup.py` | GPU detection before PyTorch is installed; picks and installs the PyTorch build for the card |
| `vibehash/protocol.py` | In-process driver for simulations |

## Status

Unaudited testnet software. Don't attach real value to it. DESIGN.md §9 lists
the remaining work:
- validators rechecking weight-based results before voting, with on-chain
  disputes;
- slashing double-signing validators;
- replicating fraud evidence;
- model types beyond MLPs.

§10 covers what's been hardened: fees, the issuance schedule, TLS, rate limits
and recovery words.
