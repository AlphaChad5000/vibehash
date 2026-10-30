# Vibe Hash: handoff notes

Snapshot of where the project stands, for whoever (human or AI) picks it up
next. Read this first, then `README.md` (how to run it) and `DESIGN.md`
(protocol, economics, threat model).

## ⚠️ Do this first

The project lives in `C:\Users\Schlo\OneDrive\Documents\vibehash`, under git
since 2026-10-29, with a private GitHub repo at
`https://github.com/AlphaChad5000/vibehash` (branch `main`). Commit and push
your work there. `dist/` holds stale local builds and is ignored. Pushing a
`v*` tag publishes to PyPI (README, "Publishing a release").

**Don't run a node with its `--data-dir` inside OneDrive.** OneDrive syncing a
live SQLite database and fast-changing checkpoint blobs can corrupt or lock
them. Use a local folder such as `C:\vibehash-node`. Worker caches already
default to `~\.vibehash\cache`, which is outside OneDrive.

## What Vibe Hash is

A proof-of-useful-work cryptocurrency (coin: **VIBE**, base unit **syn**,
1 VIBE = 10^8 syn, addresses start `vibe1`). Instead of hashing, workers train
neural networks in small verifiable units on CPUs or GPUs. Every unit that
passes verification is paid pro rata per FLOP, so there's no lottery and small
rigs earn the same rate as big ones. It's a **testnet** run by validators
(BFT consensus) that take turns leading rounds. Python 3.10+, NumPy,
`cryptography`, and optionally PyTorch for GPUs.

## Code map (`vibehash/`)

| File | Role |
|---|---|
| `coordinator.py` | **The core.** Round state machine (register → compute/commit → beacon/reveal → verify → finalize). Shared by the real node and the simulator. |
| `ledger.py` | Accounts, signed txs, blocks. Rewards and slashes are **re-derived from each block's `RoundSummary`**, never trusted from the proposer. Unbonding, chain id, job specs on chain. |
| `verification.py` | Commit–reveal, Merkle openings, sanity checks, replay spot-checks, full replay for arbitration. |
| `economics.py` | Subsidy and halving, 80/15/5 split, reputation (per round), slashing. |
| `model.py` | Reference MLP (multi-layer, tanh, softmax), deterministic SGD, float32 or float64. Batch indices come from `(unit_seed, step)`, so any segment can be replayed alone. |
| `compute.py` | Backends: `NumpyBackend` (CPU reference) and `TorchBackend` (cuda / xpu / mps), with identical maths. `self_test()` catches broken GPU drivers; `get_backend("auto")` falls back to the CPU. |
| `blobs.py` | **Content-addressed, chunked, resumable binary storage** for checkpoints and datasets. id = sha256 of the bytes. Arrays load memory-mapped. |
| `node.py` | HTTP node. JSON for small messages; `/blob/<sha>` and `/dataset/<sha>` for chunked binary transfer (GET status, GET `?offset=`, signed PUT). On a validator chain: `_lead` runs the round when it's our turn and proposes it, `_validate` (apply + replay) decides our vote, `_commit` adopts final blocks, `_sync` catches up from peers, `_ensure_params` / `_fetch_missing_data` pull trained weights and datasets from peers. |
| `consensus.py` | Tendermint Algorithm 1 (`Engine`: proposals, prevote/precommit, locking, timeouts, round skipping), stake-weighted `proposer_order`, `check_certificate`. Transport-agnostic: node.py feeds it messages and ticks. |
| `p2p.py` | `Peers`: `--peer` addresses plus validators' on-chain addresses; background POST fan-out with a back-off for unreachable peers. |
| `client.py` | `NodeClient` (API, upload/download with retry on transport errors only) and `WorkerClient` (worker loop). |
| `worker.py` | Training, commit, reveal, verify. Cheat behaviours exist **for tests only**. `blobs=` spills checkpoints to disk. |
| `storage.py` | SQLite chain and weights; datasets and blobs directories. The chain is re-validated on start-up. |
| `wire.py` | Signed envelopes (chain id + timestamp), blob-reference openings. |
| `wallet.py` | scrypt + AES-256-GCM key files. |
| `hardware.py` | RAM detection, benchmark on the chosen backend. |
| `cli.py` | `vibehash` / `python -m vibehash` commands. **`setup`** (guided: node, wallet, GPU, stake), **`install-gpu`**, **`testnet`** (node + workers + job in one process, resumable), `devices [--json]`, `job template [--gpu]`. `--node` / `--wallet` default to what setup saved. |
| `tabular.py` | `vibehash train` / `job download` / `predict`: CSV → standardised dataset (80/20 split), and self-contained model `.npz` files (weights, widths, scaling, column and class names; loaded without pickle). `train` keeps a receipt in `~/.vibehash/jobs/<job>/job.json`; `job download` merges it into the model file. Sample data in `vibehash/data/` (package data). |
| `minerui.py` | Live terminal dashboard for `worker` / `testnet`: animated mine (drill speed follows `WorkerStats.busy`, cart fills with committed units, gold popup on payouts), round/phase bar, earnings, and an event feed that captures the `vibehash.worker` / `vibehash.node` loggers. Alternate screen, ANSI only, Unicode with a pure-ASCII fallback, `NO_COLOR`; off when stdout isn't a TTY or with `--plain`. `tests/test_minerui.py` checks every frame fits 40/66/120-column terminals. The Claude Code tool shell sets `NO_COLOR=1`, so previews from there come out colourless. |
| `settings.py` | `~/.vibehash` (or `$VIBEHASH_HOME`): `config.json` with the node and wallet, the default wallet, worker cache, testnets. Lookup order: flag, `$VIBEHASH_NODE` / `$VIBEHASH_WALLET`, config, default wallet. |
| `gpusetup.py` | GPU detection without PyTorch (nvidia-smi, OS device list, ROCm). Picks PyTorch builds by driver CUDA version and compute capability, **probes PyTorch's index** (`pip index versions`) for what's published, installs, and self-tests in a fresh subprocess (Windows locks loaded DLLs). The build table needs updating when CUDA versions change. |
| `protocol.py` | In-process driver for `simulate.py` and the tests. |

Other files: `simulate.py` (economics simulation with 14 rigs, 7 cheating),
`tests/` (`test_core.py`, `test_network.py`, `test_gpu.py`), `docs/guide.html`
(user-facing guide, published as a claude.ai artifact), `examples/`,
`pyproject.toml` (package `vibehash`, console script `vibehash`).

## Key design decisions (don't undo without reason)

- **The same `Coordinator` runs in the simulator and the real node**, so economics experiments match production behaviour.
- **The ledger recomputes all system transactions** from the round summary. A proposer can't inflate rewards.
- **Withholding (commit, then never reveal) forfeits the round but is NOT slashed.** It's indistinguishable from a crash. Selective withholding is still unprofitable (`P(caught) ≥ fraction faked`; see DESIGN §3).
- **Reputation moves per round, not per unit.** Per-unit was unfair to small rigs.
- **Workers upload checkpoint blobs right after training each unit, *before* committing.** The reveal is then one small message, and uploads count toward the compute deadline, so a connection too slow to deliver in time gets less work next round. Blob ids are content hashes, so nobody can find a blob before the reveal.
- **float32 checkpoints are rounded to float32 before hashing**, so GPU and CPU produce identical Merkle leaves. Cross-device differences are absorbed by the job's `tolerance` (1e-4 recommended for float32).
- **GPU workers disable TF32 and use deterministic kernels.**
- **Rounds are replayable (decentralisation step 1, 2026-10-29).** The coordinator has no private randomness or clock: assignments and duplicates are drawn (`crypto.Draw`, hash-based, version-stable) from the previous block plus the signed registrations; checkers and canaries from the beacon; capacity from delivered work on chain (`Coordinator.capacity`: newcomer cap, then up to `RAMP`=2× last round's delivered FLOPs if it delivered everything, exactly what it delivered if it missed a deadline, never above the claim; 4× with no back-off made a worker with an optimistic benchmark swing between full and failed rounds); the job from the last summary. Every block's summary carries `record_hash` of a round record (signed registrations/commitments/reveals/verdicts, rule settings, and weight-based check results via `Coordinator._claim`). `replay.py` re-runs a round from its record with the same `Coordinator` code (`replay=record` mode) and diffs the summary; `chain verify` does it for every round. Old blocks without `record_hash` still validate (it's omitted from their hash). This is what lets validators check a leader's block before voting (below).
- **Challenge window (2026-10-29).** Nodes keep the checkpoints the last `keep_rounds` records reference (`Store.evidence`, `--keep-rounds`, default 10; reported as `keeps_checkpoints_for` in `/status`) and prune the rest. In replay, `Coordinator(recheck=...)` loads each reveal's checkpoints (start weights = any reveal's opening 0 whose hash matches `start_hash`), re-runs every `_claim` and lists mismatches in `disputes`; missing data counts as `unchecked`, never as a failure (`_weights` guards). `chain verify` rechecks rounds in the window (`--no-recheck` to skip). The new model is now `_mean`: a float64 sum in round order, then rounded. Before, `np.mean` worked in float32 on a node and float64 elsewhere, and the first live recheck caught the difference. Protocol simulations record node-style reveals with an in-memory `Checkpoints` store, so they can be rechecked too.
- **Receipts against censorship (2026-10-29).** The node signs a receipt (`receipts.py`) for every register/commit/reveal/verdict it accepts: chain id, round, the round's random `attempt` label (in the record, so a re-run round doesn't look like censorship), kind, and the sha256 of the message exactly as the record holds it. `WorkerClient._signed` checks each receipt vouches for what it sent and appends it to `<cache>/receipts.jsonl`; `audit_receipts` checks every sealed round and logs `censored: …` (shown red in the dashboard). `chain verify` audits local receipts by default, or `--receipts FILE`. Tested against a node patched to drop a worker's commitments: replay passes (the record is self-consistent), and the receipts catch it.
- **Validators and BFT consensus (decentralisation step 2, 2026-10-29).** A genesis `validators` map, `{addr: url}`, becomes genesis txs with memo `"validator <url>"`. Genesis files without it keep their hash and run single-node (`Node._solo`), exactly as before. `State.validators` holds `addr -> url`; `validator_set()` is the ones with at least `MIN_STAKE`, weighted by stake. The `validator` tx joins or leaves, and only works on chains that already have validators (otherwise anyone could take over a one-node chain). On a validator chain the proposer must be a validator, and the block's **commit certificate** (precommits from more than 2/3 of the stake) is stored in SQLite `certificates`, served in `/blocks`, checked by `Store.open_chain`, `_sync` and `chain verify`.
  - A height's leader runs the training round (`Coordinator.finalize(seal=False)` builds the block without appending it), then proposes `{block, record}`.
  - Validators vote using `_validate`, which is `apply_block` plus `replay_block`. `Coordinator.adopt` appends the final block, prunes the mempool, and saves the record and params (ours, or fetched later and checked against `params_hash`).
  - The propose timeout is `1.5 × round + 5 s`. `has_work` keeps idle chains from churning rounds.
  - Workers follow `/status` `leader` (None means "this node"). `done` is keyed by the round's `attempt`, so they rejoin a re-run round.
  - `testnet --validators N` runs N in-process validators on consecutive ports. A new testnet always has at least one validator, so the default single validator uses this path too.
  - Tests: `test_consensus.py` (engine on a simulated clock, ledger rules) and `test_validators.py` (4 live validators: rotation, a self-paying leader outvoted, an offline validator, sync, forged certificates, an edited data dir).
  - Not done: rechecking weight-based claims before voting, on-chain disputes, double-sign slashing (evidence is only collected, in `Engine.evidence`), replicating checkpoints. DESIGN §9 lists these.
- **A fair start (2026-10-29): earn-in and bootstrap stake.** Two genesis rules, committed to by genesis txs with memos `"earn-in"` and `"bootstrap"`; `genesis create` and new `testnet`s turn both on, while older genesis files lack them and behave as before.
  - **Earn-in:** `State.earn_in` lets unstaked addresses register (`Coordinator.eligible`), at most `max_newcomers` (64) a round. They're capped at `newcomer_units`, assigned only units staked workers don't need (`pick` prefers staked), and never chosen as verifiers (`pool` takes staked only). A `reward` to an address below `MIN_STAKE` tops up its stake first (`_apply_system_tx`). `_check_summary` lets unstaked workers be paid, but verifiers must be staked.
  - **Bootstrap stake:** `State.locked`. It votes and counts for proposing, but `unstake` can't touch it, a validator leaving burns it, and a slash takes withdrawable stake first.
  - The simulator's `_proposer` picks only staked workers.
  - Also fixed: `apply_block` now refuses a second genesis block appended to a chain, which previously re-applied the allocations.
  - Tests: `tests/test_earn_in.py`.
- **Jobs with budget left take turns, one round each** (`Coordinator.active_job_id`). Before 2026-10-29 it was first come, first served, so a new job waited for every earlier job to spend its budget.
- **Job posters can get their model back.** `GET /job/<id>` reports each round's validation loss/accuracy (SQLite `job_rounds`, written when a round is sealed) and the model's size and sha256; `GET /job/<id>/model?offset=` serves the weights. They're public, because workers receive them every round anyway.
- **A replay must match the segment's own update to within 1% (`UPDATE_TOLERANCE`), not just the job's `tolerance`.** For big or nearly trained models one segment's whole update can be smaller than the tolerance, and then a worker who resubmits its start weights untrained (the `freezer` test cheat) would pass. Replica comparisons use the same `reproduces()` rule.
- **Retries only on transport errors / 5xx.** Retrying 4xx stalled workers for whole rounds.

## Current state

Done and tested (as of the last full runs):
- Ledger, economics, verification, every cheat type, live multi-process network, chain audit, restart from disk.
- Chunked blob transfer: a live network trained a **2.2M-weight float32 model (8.8 MB checkpoints)** with all units accepted, and a **20 MB dataset** moved in chunks. The checkpoint cap is now **1 GiB** (`tasks.MAX_CHECKPOINT_BYTES`).
- float32 and deep-model jobs through the full protocol on CPU.
- PyTorch Intel XPU build installed and detected on the dev machine (Intel Arc iGPU).

**In flight when work stopped:**
1. **Tests: done (2026-10-29).** All 58 pass, `test_network` included, on CPU and CUDA. To re-run:
   ```bash
   python -m unittest discover -s tests -t .
   ```
   Measured with PyTorch's CPU device vs the NumPy reference: replay divergence **1.4e-8**, while one honest segment moves the weights **2.4e-2**. That leaves a lot of room under the 1e-4 tolerance.
2. **GPU verification on NVIDIA: done (2026-10-29).** RTX 4070 SUPER, driver 616.92, PyTorch 2.14.1+cu126 (the README's install command). All 58 tests pass. Measured CPU↔GPU replay divergence is 4e-9 to 8e-8, at least 1,250× under 1e-4; full table in DESIGN §8. The same calibration found the `freezer` hole above, which is now fixed and has tests (`test_core`: three freezer tests; `test_gpu`: honest tiny-update work verifies GPU→CPU and CPU→GPU). Still unmeasured: Intel XPU (the old Intel driver failed), Apple MPS, AMD ROCm.
   - A CUDA venv for this machine is at `%USERPROFILE%\.vibehash\venv` (outside OneDrive). Run the tests with `%USERPROFILE%\.vibehash\venv\Scripts\python.exe -m unittest discover -s tests -t .`
3. ~~Docs stale after the blob change~~ Done: DESIGN §8 "Model size" and §9 rewritten around blobs; README layout has `blobs.py`.
4. **`docs/guide.html` redesigned and published (2026-10-29)** to `https://claude.ai/artifact/61GPfio1nUHrmksTpR1Rwp` (version 3). The hero is a canvas replica of the `minerui.py` dashboard, driven by a simulated session in the page's script; if the dashboard's look changes, port the change to `build()` in the guide's script too. The artifact is private until it's shared from its Share menu.
5. **0.5.0 is on PyPI (2026-10-29)**, published by `.github/workflows/publish.yml` from the `v0.5.0` tag through PyPI trusted publishing (0.4.0 before it), and checked by installing it into an empty venv. `pip install vibehash` works now. The next release must be 0.5.1 or later. The guide (artifact version 9) installs from PyPI and has a "Who runs the network" section on validators. `dist/` holds stale local 0.2.0/0.3.0 builds; the 0.2.0 ones are the only copy of that source, so they were kept.
6. **Easier setup: done (2026-10-29).** `setup`, `install-gpu` and `testnet` are tested end to end on this machine: a scratch newcomer ran `setup`, was sent VIBE, staked by re-running setup, and earned with a flagless `vibehash worker` alongside the testnet's own workers. Building the testnet exposed a **blob upload race**: replicated units give byte-identical checkpoints, both holders uploaded the same blob into one shared partial file, the hash failed, and both lost the unit. The fix gives each uploader its own partial file (`GET /blob/<id>/status?from=ADDR` for resuming), with a lock around completion. Tests: `test_setup.py` (detection, build choice, settings, concurrent uploads) and `CommandLineTests` in `test_network.py`.

## Pre-deployment checklist

Blockers before inviting anyone onto a public testnet:

- [x] Put the OneDrive folder under git (see top).
- [ ] Items 1–5 above.
- [ ] **Choose a license** (there is none, so others legally can't use or modify the code). This is the owner's decision.
- [ ] **Faucet.** New users can't get VIBE to stake except through genesis or a transfer. Add a rate-limited testnet faucet (endpoint or CLI) or newcomers can't join. `setup` already tells newcomers to ask the operator for VIBE; a faucet would let it stake for them.
- [x] **Publish to PyPI:** 0.4.0 is out (2026-10-29). Later releases: bump the version and push a `v*` tag (README, "Publishing a release").
- [ ] **Node hardening:** ~~TLS; per-IP rate limits; connection caps; transaction fees~~ done 2026-10-29 (DESIGN §10: `tls.py`, `ratelimit.py`, `fees.py`). Still open: a per-address dataset quota (anyone with ≥1 VIBE can upload 20 GB datasets repeatedly); public blob GETs can be abused for bandwidth; a block size limit and replace-by-fee.
- [ ] **Operations:** run the node as a service; back up `node-data/` and the node wallet; monitoring and alerting; disk-space alarms (opened checkpoints are kept for `--keep-rounds` rounds, about units × (2q+2) × model size per round; datasets are never deleted).
- [ ] **Load test:** many workers, a large model, and real home upload speeds. Tune phase durations (`--compute-s`, etc.). Defaults are sized for small models.
- [ ] **Security review:** fuzz the HTTP API; have someone else review ledger/coordinator/verification. Nothing has been audited.
- [x] **Wallet recovery:** done 2026-10-29. `wallet create` and `setup` show 24 BIP39 words (SLIP-0010 Ed25519, `m/44'/1'/0'/0'/0'`), and `wallet restore` rebuilds the wallet from them. A mainnet needs its own SLIP-0044 coin type.
- [ ] **Issuance:** `genesis create --economics mainnet` gives a Bitcoin-like schedule (3.125 VIBE a round, ~4-year halvings, 21M cap). Choose the real numbers before any launch.
- [ ] **Versioning:** protocol and client version in `/status`; refuse incompatible clients; plan for chain upgrades.
- [ ] **Legal:** keep it explicitly valueless and testnet-only; terms and privacy notice (the node sees worker IPs); legal advice before anything that could carry value.

Known limits (fine for testnet, document rather than fix now):
- One leader per round: it can't forge payments, bend the round rules (the other validators replay its record and vote it down) or drop messages it accepted (receipts) undetected. It can still refuse messages outright in the rounds it leads, and its weight-based check results are taken on trust unless someone rechecks them within the challenge window. The chain keeps going while validators holding more than two thirds of the stake are up. DESIGN §9 lists the fixes in order.
- MLPs only; transformers and CNNs need new backend kernels and tolerance calibration.
- `Dataset.from_npz` loads the whole dataset into RAM; the node evaluates on the CPU.
- Verifiers download ~2q checkpoints per task, which is significant bandwidth for big models.

## Working tips

- The original session folder's path was too long for some `pip` installs (numpy, torch). The OneDrive path is short enough. A virtual environment works there, as long as it's excluded from git, or kept outside OneDrive so OneDrive doesn't sync gigabytes of PyTorch.
- In the Claude Code Bash tool, long heredocs containing quotes sometimes failed to parse. Writing edit scripts to a `.py` file and running them was reliable.
- `simulate.py` and `test_core.py` are fast (seconds). `test_network.py` takes about a minute (live HTTP nodes). `test_gpu.py` needs PyTorch.
- Signatures fall back to slow pure-Python Ed25519 without `cryptography`. Install it for realistic timing.
