# Vibe Hash: handoff notes

Snapshot of where the project stands, for whoever (human or AI) picks it up
next. Read this first, then `README.md` (how to run it) and `DESIGN.md`
(protocol, economics, threat model).

## ⚠️ Do this first

The project lives in `C:\Users\Schlo\OneDrive\Documents\vibehash`, under git
since 2026-10-29, with a public GitHub repo (public since 2026-10-30, MIT licensed) at
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
| `client.py` | `NodeClient` (API, upload/download with retry on transport errors only) and `WorkerClient` (worker loop). The loop notes each round into `record` (a `records.RoundLog` when the command keeps one, else nothing) and writes its line once the round's block is final. |
| `records.py` | What's kept on disk so a night of mining can be read afterwards (README, "Logs and records"). Workers: `Session` (made by `worker` and `testnet`) attaches `LogFile` to the dashboard feed's loggers (`logs/worker-DATE.log`, 10 MB then `.1`, 14 days kept, control characters written out by `clean`) and gives each client a `RoundLog` (`logs/rounds.jsonl`: a line per round, pay read from the round's block by `settle`: its reward and slash txs and its summary; units accepted are the worker's FLOPs over a unit's). Posters: `history_rows` from `/job` plus each round's block (time, its `fee_release`), budget left worked back from the escrow; `JobHistory` keeps `jobs/JOB/history.csv`/`.jsonl` current for `trainui.JobWatch`; `fetch_history` for `job history`; `write_report` puts the history and a `TRAINING.md` beside a downloaded model. `testnet` logs its nodes to `node.log` too. Tested: `tests/test_records.py`, and end to end in `test_network.CommandLineTests`. |
| `worker.py` | Training, commit, reveal, verify. Cheat behaviours exist **for tests only**. `blobs=` spills checkpoints to disk. |
| `storage.py` | SQLite chain and weights; datasets and blobs directories. The chain is re-validated on start-up. |
| `wire.py` | Signed envelopes (chain id + timestamp), blob-reference openings. |
| `wallet.py` | scrypt + AES-256-GCM key files. |
| `hardware.py` | RAM detection, benchmark on the chosen backend. |
| `cli.py` | `vibehash` / `python -m vibehash` commands. **`setup`** (guided: node, wallet, GPU, stake), **`install-gpu`**, **`testnet`** (node + workers + job in one process, resumable), `devices [--json]`, `job template [--gpu]`, `selftest` (selftest.py), `job history [JOB] [--csv FILE] [--json]` (records.py). `--node` / `--wallet` default to what setup saved. |
| `tabular.py` | `vibehash train` / `job download` / `predict`: CSV → standardised dataset (80/20 split), and self-contained model `.npz` files (weights, widths, scaling, column and class names; loaded without pickle). `train` keeps a receipt in `~/.vibehash/jobs/<job>/job.json`; `job download` merges it into the model file. Sample data in `vibehash/data/` (package data). |
| `minerui.py` | Live terminal dashboard for `worker` / `testnet`: animated mine (drill speed follows `WorkerStats.busy`, cart fills with committed units, gold popup on payouts), round/phase bar, the job selector, earnings, and an event feed that captures the `vibehash.worker` / `vibehash.node` loggers. The selector (`selector`, `MinerDashboard.press`), a `job:` line under the round: auto, or one of the jobs `/network` lists with budget left (`NodeClient.open_jobs`, sorted so the number keys stay put, polled every 30 s on its own thread; ids tasks.py wouldn't accept are dropped, as they'd reach the terminal), set for every rig at once through their shared `client.JobChoice`. Tab, j/k or the arrows move it, a or 0 is auto, 1-9 a job; keys are read between frames without waiting (`termui.keypresses`: msvcrt on Windows, termios cbreak and select on POSIX, put back on the way out), and only when stdin is a terminal. Alternate screen, ANSI only, Unicode with a pure-ASCII fallback, `NO_COLOR`; off when stdout isn't a TTY or with `--plain` (`worker --job` picks the job there). `tests/test_minerui.py` checks every frame fits 40/66/120-column terminals, the selector included. The Claude Code tool shell sets `NO_COLOR=1`, so previews from there come out colourless. |
| `trainui.py` | Live dashboard for job posters: `train` shows it after posting, `job watch [JOB]` brings it back. A forge in the mine: the model is a blade that grows with accuracy, hammered when it's the job's turn; per-round accuracy and loss charts, budget left, an event feed. Polls `/status` every 1.5 s and `/job` on new blocks; `--plain` or a pipe gives log lines. A fine-tune (a job whose spec names a `base`, or `JobWatch(base=...)`) forges its model maker's emblem instead (`emblems.py`): the forge grows to 9 rows, and the emblem takes on its colours from left to right as accuracy climbs. |
| `termui.py` | What the two dashboards share: colours, the rock, the banner, the phase bar, the feed, the alternate screen. |
| `settings.py` | `~/.vibehash` (or `$VIBEHASH_HOME`): `config.json` with the node and wallet, the default wallet, worker cache, testnets. Lookup order: flag, `$VIBEHASH_NODE` / `$VIBEHASH_WALLET`, config, default wallet. On a RunPod pod (`$RUNPOD_POD_ID`, a `/workspace` folder) the home is `/workspace/.vibehash`, the only place a restart keeps (`pod_volume`), unless `~/.vibehash` already holds a setup. |
| `selftest.py` | `vibehash selftest` (and the menu's "Check this computer"): Python and packages, PyPI's newest release, every device's self-test, PyTorch's CUDA build against the driver, GPU memory free; **a real fruit-demo unit** (the public job's spec, and its dataset: the packaged fruit.csv makes it byte for byte) on the worker's device, checked as verifiers check it with every segment replayed on the CPU, and whether it repeats bit for bit; a LoRA step on a tiny Qwen3 built in memory (put into `finetune._runtimes` directly, so nothing is fetched) replayed on the CPU; `--model NAME` for one real unit (asked first). The node's /status, round trip, clock skew from its HTTP `Date` header (`clock_skew`, which `worker` also checks before it starts), download speed, Hugging Face; the wallet read without its password; disk for the cache and model store; then what the computer can take (`finetune_need` sizes Qwen3-0.6B/4B from baked shapes with `finetune.memory_mb`; `deepseek_worker.can_take`) and the next command. Exit 0 only if it can mine. `--json`. Tests: `tests/test_selftest.py` (no GPU or internet). |
| `advice.py` | Plain-language problems and what to do, a line or two each: a node out of reach (DNS, refused, timeout with the dorm-Wi-Fi hint, TLS/captive portal), the clock, a different network, out of GPU memory, a full disk, no stake on a network without earn-in, a job waiting for staked workers (by this worker's stake and memory). `explain(e)` maps exceptions to them; every command's error exit, setup, the worker loop (`WorkerClient._say`, at most once a minute for the same problem) and selftest use it. |
| `unattended.py` | No terminal (nohup, Docker, a pod): `interactive()` (Windows calls NUL a terminal and getpass then waited for ever; it asks the console), `need()` (the one-line error naming the variable or flag), `save_words` (a new wallet's words to `recovery-words-ADDRESS.txt`, 0600, when output isn't a terminal, with a loud note in the log), `WorkerLock` (one worker per wallet: a second would prune the first's checkpoints and register as the same worker). |
| `filelock.py` | Locks between processes (msvcrt / fcntl, released when the process ends): `hub.TensorCache.fetch` holds one per commit, and `modelstore.Store.acquire` one per store (in the temp folder), so workers on one computer don't fetch the same files at once. |
| `gpusetup.py` | GPU detection without PyTorch (nvidia-smi, OS device list, ROCm). Picks PyTorch builds by driver CUDA version and compute capability, **probes PyTorch's index** (`pip index versions`) for what's published, installs, and self-tests in a fresh subprocess (Windows locks loaded DLLs). The build table needs updating when CUDA versions change. |
| `protocol.py` | In-process driver for `simulate.py` and the tests. |
| `transformer.py` | Llama-style decoder (RoPE, grouped-query attention, RMSNorm, SwiGLU) with LoRA, NumPy reference: Hugging Face tensor names and shapes for the frozen base, PEFT-compatible adapters (`peft_names`), deterministic SGD `train_segment`, and a stage API bit-identical to whole-model training for sharding. Jobs use `llm.py`; this is its float64 reference in `test_finetune` and the basis for sharded transformers. |
| `llm.py` | Pretrained decoders in plain PyTorch: Llama (with Llama 3's rotary scaling), Mistral, Qwen2 (q/k/v biases), Qwen3 (q/k norms), SmolLM3 (no rotary on every fourth layer). float32 with frozen weights kept as stored (`_Frozen` casts per matmul), attention written out, layers recomputed in the backward, rotary tables in float64 on the CPU, head and loss fused over reply tokens only (`_HeadLoss`). `Arch.from_hf_config` refuses what it doesn't compute (Gemma, YaRN, sliding windows shorter than the sequence). Matches transformers to 2e-6 on Qwen3-0.6B. |
| `finetune.py` | Fine-tune jobs (FINETUNING.md, DESIGN §8 "Fine-tuning pretrained models"). `LoraSpec`: ModelSpec's interface for LoRA with **A frozen** (drawn from the job's seed like PEFT) and **B trained with Adam**, the vector `[B | m | v]`, `parts` for part-by-part checks, `max_step` (Adam's rigorous bound) for the sanity check, `fresh` zeroing moments after averaging. `TokenDataset`: ids, lengths and a learn-mask per sequence, plus the model's manifest and config.json as bytes. `ChatTokenizer`/`prepare`: chats through the model's own Jinja chat template and tokenizer, learning only the replies and their end-of-turn token (found by text in the rendering, so any template works). `runtime`: the base model fetched once per process and device. `train_segment`, `evaluate`, `export_peft` (adapter_model.safetensors + adapter_config.json + README). `plan`: what `train DATA --base` shows before posting. |
| `moe.py` | Mixture-of-experts feed-forward block (DeepSeek style: top-k router, routed and shared SwiGLU experts, optional LoRA), NumPy reference. Workers commit their routing; checkers replay with it and accept it only if no left-out expert beat a chosen one by more than a tie margin (1e-4 in logits), so rounding can't fail an honest worker. Experts are checkable one at a time. SHARDING.md §7. Not wired in yet. |
| `hub.py` | Hugging Face Hub client, standard library only: `org/name[@rev]` to a pinned commit, `describe` for a pre-flight, one ranged request per tensor (never the whole checkpoint), a per-tensor sha256 manifest checked against the Hub's own file hashes, a resumable cache with LRU pruning, and `HF_TOKEN` sent only to the Hub, never to its CDN. Mirrors `huggingface_hub` 2.4; tested against a fake Hub (huggingface.co was blocked where it was built), and since 2026-11-01 used against the real one for DeepSeek's headers and weights. `describe` reads shapes nested in `text_config` (multimodal configs). Fine-tunes use it: workers fetch the base through `TensorCache`, 64 MiB runs over six connections. |
| `deepseek.py` | DeepSeek-V4.1-Flash in plain PyTorch (DEEPSEEK.md §7): trainable, deterministic, weights kept in DeepSeek's FP8/FP4 formats and names and dequantised exactly inside each matmul (`_FrozenMatmul`, which also lets experts stay in pinned RAM). `Numerics()` is the reference's bf16-with-FP8-rounding arithmetic, `Numerics(float64, quantize=False)` the smooth one. Discrete choices (routing, indexer picks, candidate blocks) are inputs (`Choices`), side channels between layers explicit (`Side`). `Model` holds any run of layers. `fetch_file` / `load_layer` download whole weight files into `~/.vibehash/cache/deepseek` (parallel pieces, sha256-checked). Needs PyTorch 2.8+. Prefill and text only. Not wired into jobs. |
| `deepseek_top.py` | DeepSeek-V4.1-Flash's first project (DEEPSEEK.md §8): only the top layers trained, from the frozen layers' output precomputed once. `Cut` (what crosses a cut, bytes a token), `precompute`, records (`pack`/`unpack`/`record_root`), `Adapters` (LoRA-FA, A from the job's seed, PEFT's format), `train_step` (layer-major) and `train_step_streamed` (expert-major, for 6–8 GB GPUs), both exact to the whole model with its bottom frozen (tests/test_deepseek_top.py). Trains in float32 (`TRAIN`): bf16 makes CPU and CUDA replays 4–9% apart. `build_trained` keeps the routed experts as memory-mapped file bytes (`deepseek.read_safetensors(..., mmap=True)`), so 16 GB of RAM trains four layers; `Meter` gives a unit's tokens a second and peak GPU memory. Not wired into jobs (waits for PR #5's fine-tune units). |
| `deepseek_prep.py` | The one-time prep of a top-layers job (DEEPSEEK.md §8): `python -m vibehash.deepseek_prep --data tokens.npz --repo USER/NAME`, on Linux or Windows, resumable after a stop or reboot, with progress and an ETA. It streams the engram tables for the rows the dataset reads (the compressed token map is rebuilt from the tokenizer: 99,092 ids), fetches the frozen layers, runs blocks of sequences through them with each sequence's state waiting in RAM and experts brought over one at a time, and uploads records, per-layer hashes and the Merkle root to the runner's own dataset repo (huggingface_hub). Bit-identical to `precompute` (tests/test_deepseek_prep.py). Not run on real weights yet. A dev and test tool: the protocol knows only the split prep's units. |
| `modelstore.py` | Where a miner's PC keeps big model files, and for how long (DEEPSEEK.md §8, "Model files on a miner's PC"). It asks before a download (what, how big, where, another drive), keeps a user-set cap with least-recently-used eviction (and refuses a job that won't fit), deletes a job's files when it ends or after 7 days unmined unless kept, treats hand-offs and shards as temporary, and leaves no partial download behind. It keeps to a folder of its own (`<chosen>/vibehash-models`) and deletes only what its index names. `entries` and `clear` are for `vibehash cache`, `usage` is for the dashboard, `choose` is setup's question. The worker sweeps it at start and calls `job_ended` when a DeepSeek job's escrow is spent. Tested (tests/test_modelstore.py). |
| `deepseek_split.py` | The split prep, the design (DEEPSEEK.md §8): the frozen layers' pass cut into units ordinary miners run, connected by stored hand-offs, nobody online together. No preparer role: anyone holding a unit's layers runs, checks or challenges it; the node only schedules and referees. `plan` makes the work graph: rows units (an engram table's row range, ~7 GB streamed once) and layers units (a piece of G layers over a chunk). Hand-offs are exact (bf16, byte planes + zstd: 28 KB a token a cut). `Line` is the node's side: claims, optimistic consumption, rollback when a challenge succeeds, redo of a withheld output, deletion of what everyone has read. Hosts aren't trusted: every read is checked against its writer's committed digest (`Tampered`), and `Mirrors` reads from the first host with the committed copy. Big hand-offs go up in resumable multipart uploads and come down in ranges. `replay` checks a unit and finds the layer a challenge starts at; `state_before` serves the agreed state entering it. `deadline` and `Line.expire` keep slow or vanished miners from stalling the line; each commitment carries the unit's stats. `presign`/`PresignedStore`: S3-signed URLs (DigitalOcean Spaces), so miners need no storage account. Exact to `deepseek_prep` (records, root) in any unit order (tests/test_deepseek_split.py). Not wired into the node (waits for PR #5). |
| `deepseek_config.py`, `deepseek_graph.py` | Without PyTorch, for the node: DeepSeek's config, the cut, the work graph (`plan`, `Line`), deadlines, and the bucket (`PresignedStore`, `Signer`). `deepseek_split` re-exports them. |
| `deepseek_job.py` | The DeepSeek job kind, without PyTorch: `DeepSeekSpec` (a `LoraSpec`, so its training rounds are a fine-tune's; a node from before it refuses the spec), the job's token file (`TokenFile`) and its training and held-back ranges. |
| `deepseek_rounds.py` | A DeepSeek job's prep on the node, round by round: which ready units a round hands out, checking a revealed commitment, settling into the work graph, and the records' root once all are in. Its docstring has the interim dispute rule (DEEPSEEK.md §8 launch checklist). |
| `deepseek_program.py`, `deepseek_tracer.py` | A DeepSeek training unit op by op, as program.py and tracer.py are for fine-tunes: the tree a dispute bisects (a step's sequences together, layer by layer; attention a sequence at a time; the routed experts by slot, so the tree follows from the data alone) and its ops in PyTorch, which units train by. referee_deepseek.py checks each op on the node. Tested: tests/test_deepseek_program.py. |
| `deepseek_worker.py`, `deepseek_client.py`, `deepseek_unit.py` | A miner's side: running and checking prep units, the model files through `modelstore`, `can_take`'s messages (`deepseek_worker`); the worker loop's steps for prep and training rounds, links signed by the node (`deepseek_client`, called from `client.py`); training and scoring from the records (`deepseek_unit`). The whole thing over HTTP: `tests/test_deepseek_network.py`. |
| `emblems.py` | Pixel-art emblems (at most 18×10, two pixels a cell in half blocks) of DeepSeek, Meta, Qwen, Gemma, OpenAI, Microsoft, Mistral and Hugging Face, by model family (`FAMILIES`: an organisation and the start of the model's name, so Google's T5 or Meta's Prompt-Guard don't get Gemma's star or Llama's loop); a generic gem for anything else (Z.ai's GLM too). `scripts/preview_forge.py NAME` plays a made-up job so you can see one. |
| `catalog.py` | The model picker: 17 popular open-weight models (`CATALOG`, short name → Hugging Face repo, smallest first), their facts from `Hub.describe` cached a day in `~/.vibehash/cache/models.json`, and "what it takes" (`team`: GPUs of 12/24/32 GB holding the weights as stored, whole layers each, a quarter of each card kept free; DEEPSEEK.md §2's 40/20/14). `ready`: an architecture `llm.py` trains and one 12 GB card holds it (qwen3-0.6b, llama-3.2-1b, smollm3-3b, qwen3-4b today). `vibehash models` and `train --base` print it; `train DATA --base` fine-tunes a ready one (`cli._finetune`) and shows the pre-flight for the rest. |
| `shards.py` | Sharded units: a model split into shards across a team of workers, each cell (shard × segment) checked holding only that shard, and signed edge messages. The coordinator runs sharded jobs in-process (teams, cell tasks, edge evidence, per-shard pay and averaging, record version 3; `simulate.py --shards N`). Nodes don't yet (`Coordinator.runs_teams` is off, `waiting` says why), and there's no worker-to-worker transport. See `SHARDING.md`. |

Other files: `simulate.py` (economics simulation with 14 rigs, 7 cheating),
`tests/` (`test_core.py`, `test_network.py`, `test_gpu.py`), `docs/guide.html`
(user-facing guide, published as a claude.ai artifact), `docs/whitepaper.pdf` (the design in eleven pages, in the
style of Bitcoin's; printed from `docs/whitepaper.html`, whose first lines say how, and copied to
vibehash.org/whitepaper.pdf by `scripts/build_site.py`), `examples/`,
`pyproject.toml` (package `vibehash`, console script `vibehash`), `agents/` (the Muse agents, from calebluh's
branch: an LLM, Meta's Muse Spark by default, designs a fine-tune from a plain-English goal and judges the result
against the base model; `agents/README.md`, `tests/test_agents.py`; run from a checkout, not in the package).

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
- **Round length follows the job (2026-10-29).** `TrainingJob.phase_lengths()` sizes compute, reveal and verify so a reference machine (`tasks.REF_FLOPS` 2 GFLOP/s, `REF_UPLOAD` 20 Mbit/s, `REF_DOWNLOAD` 100 Mbit/s) can download the weights, train a unit and upload its checkpoints. An optional job field `compute_s` (5–3600) asks for a longer compute phase; it's omitted from specs when 0, so old job specs are byte-identical. `Node.times_for(job)` uses the longer of the node's phases and the job's, at most: a phase ends once everything it waits for is in (`Coordinator.all_in`), never sooner than the node's own length for it, so a round nobody works in lasts the node's own round and none is shorter (2026-11-02). `Coordinator.timing` writes the round's compute seconds into `record["rules"]["compute_s"]`, which capacity uses. `Engine(propose_wait=...)` makes validators wait as long as the longest funded job's round. **A replay now copies the recorded rules verbatim:** before, it rebuilt them from today's config, so adding `max_newcomers` would have made every older record hash differently and fail `chain verify` (test: `test_round_length.py`).
- **Jobs with budget left take turns, one round each** (`Coordinator.active_job_id`). Before 2026-10-29 it was first come, first served, so a new job waited for every earlier job to spend its budget.
- **Job posters can get their model back.** `GET /job/<id>` reports each round's validation loss/accuracy (SQLite `job_rounds`, written when a round is sealed) and the model's size and sha256; `GET /job/<id>/model?offset=` serves the weights. They're public, because workers receive them every round anyway.
- **A replay must match the segment's own update to within 1% (`UPDATE_TOLERANCE`), not just the job's `tolerance`.** For big or nearly trained models one segment's whole update can be smaller than the tolerance, and then a worker who resubmits its start weights untrained (the `freezer` test cheat) would pass. Replica comparisons use the same `reproduces()` rule.
- **Retries only on transport errors / 5xx.** Retrying 4xx stalled workers for whole rounds.

## Current state

**Unreleased (branch `job-picker`, 2026-11-02): pick the job to mine.** The owner asked for "a easy selector in the
mining area of the terminal that lets you select jobs, auto, then each of the listed jobs". The mining dashboard has a
`job:` line under the round (code map, `minerui.py`), and `worker --job JOB` does the same for pods and `--plain`,
checked against the node's list at start (a job it doesn't list stops the worker, naming the ones it does). A worker
set to one job doesn't register in other jobs' rounds and says so in its activity line ("sitting out fruit-demo's
round: mining think-shorter-pilot only"), not its log; a round it already registered in is finished whatever the
choice says by then; once its job is no longer listed with budget left (asked every 30 s while it sits out) it goes back
to auto with one log line (`WorkerClient.choose`, `_sits_out`). Round log lines say `mining` when a job was chosen.
Workers only. `testnet` has no flag (its `--job` is the demo job's template), but its dashboard's selector moves all its
workers. Tested: `tests/test_minerui.py`, `test_network.JobPickerTest` (two live jobs taking turns) and the `--job` check;
keys on a real Windows console (key events written into its input buffer, a real Ctrl+C) and on a real pty under WSL.
On Windows, Shift+Tab reaches msvcrt as Tab, so it goes forward there. Not done: the guide's canvas replica of the
dashboard (`docs/guide.html`, `build()`) has no `job:` line yet.

**Release 0.5.8 (2026-11-02, test night): two worker fixes from the live run.** The mining dashboard called any drop in
balance plus stake a slash, so funding a friend's wallet showed "lost 97.42 VIBE: slashed"; a drop with a new nonce is now
the wallet's own transaction (minerui.Rig.nonce, .sent). And a worker far from the node (a RunPod pod) checked 2 of 9
fruit-demo tasks in the 10 s verify phase, fetching checkpoints and sending verdicts one round trip at a time: they now
come down together first and verdicts go while the next task is checked (WorkerClient._prefetch, _answer). Workers only.
Found the same night and not fixed here: honest fine-tune workers on different machines slash each other. A Qwen3-0.6B
segment's B differs by 5.6e-4 between this PC's GPU and CPU (over the 1e-4 tolerance; same on torch 2.14 and 2.15), most
likely Adam's sign-like first steps from fresh moments, and a disputed defender can't make its first split in the 53.8 s a
move gets. Rounds 7277 and 7280 of job pokemon-pun-starter-2d950a; a separate session (branch from task
"Fine-tunes slash honest workers across GPUs") works on it.

**Release 0.5.7 (2026-11-02): for test night (PR #17, TONIGHT.md).** The three parts below, and the Muse agents (`agents/`,
calebluh's `Meta-AI-Agents` with fixes; agents/README.md). The faster rounds are the node's: they take effect when
the public node is upgraded with `deploy/setup-node.sh`.

**In 0.5.7: a phase ends once its work is in, and a round nobody works in lasts the node's own 38 s.** On the
public node the DeepSeek pilot, which nobody mines, took 1,521 s a round (its phases are a prep unit's deadline and
its checks', never under 10 minutes), alternating with fruit-demo's 39 s ones: a newcomer earning its stake on
fruit-demo got a paid round every 26 minutes. Now compute, reveal and verify each end as soon as everything they wait
for is in (Coordinator.all_in: every unit handed out committed, every commitment revealed, every task answered), but
never sooner than the node's own length for the phase (Node._run_round). So a round in which nothing was handed out
(nobody registered who fits, or a fine-tune short of its quorum) lasts exactly the node's own round, and the pilot's
empty rounds take 38 s; a unit or verdict still out keeps the job's full time. No round is shorter than the node's
own, so a chain nobody mines still seals a block every 38 s, not every few seconds, and the mainnet halvings keep
their timing. Records, blocks and replays are unchanged: rules.compute_s is still the job's, and nothing more can come
in once a phase is all in. Only the node needs it: workers already poll. Expected on the public node, with nobody on
the pilot: fruit-demo and the pilot alternate at about 39 s each, so a fruit-demo round every 78 s or so (20 times
as often); a qwen3-0.6b fine-tune's round (264 s at 192 tokens) ends once its 8 units, reveals and checks are in,
however fast that is. This folds in the commit before it, which ended a prep round's compute phase once every unit
was committed (then Coordinator.prep_committed, with no floor).

**In 0.5.7: `vibehash selftest`, and setup and the worker for tonight's machines.** For
the owner's mixed Windows laptops and desktops and RunPod pods:
- `selftest` (code map), in the menu as "Check this computer". On the 4070 SUPER it takes about 10 s.
- Unattended: on Windows, `setup < NUL` (and so any setup or worker with no console) waited for ever in getpass; now
  nothing asks without a terminal (`unattended.interactive`), a missing password stops setup and the worker at once
  naming `VIBEHASH_PASSWORD`, and a new wallet's words go to a private file rather than the log. `setup --deepseek
  yes|no` saves the worker's DeepSeek answer.
- RunPod: `install.sh` keeps a PyTorch the Python already has exactly as it is (pip constraints pin `torch==` it,
  and NumPy below 2 if it was 1), gives an externally managed Python's venv `--system-site-packages` when that
  Python has PyTorch, and on a pod puts the settings in `/workspace/.vibehash` (settings.py does the same). Tested
  against a stand-in python3 in four environments under Git Bash; not on a real pod.
- Several GPUs: one worker per GPU, each with its own wallet (`worker --help`, README "Several GPUs"). A second worker
  with the same wallet is refused (`WorkerLock`), and model downloads are locked between processes.
- Errors newcomers hit (advice.py): node out of reach, clock skew (the worker checks it at start), a different
  network, GPU out of memory and a full disk (no traceback), a job waiting for staked workers (once a job, with this
  worker's stake and memory; selftest adds how long the stake should take at the network's recent pace).
- The pilot's ~1,520 s empty rounds that held newcomers back are the node change above.

**In 0.5.7: records of what happened.** For a night of mining on home PCs and rented
pods with nobody watching a terminal: every `vibehash worker` and testnet worker keeps the dashboard feed's events in
`~/.vibehash/logs/worker-DATE.log` and a JSON line per round in `logs/rounds.jsonl` (what it was given and did, the
seconds it took, and what the round's block paid it or slashed, and why), says where at start and sums the session up
at the end; a testnet logs its nodes to `node.log` in its folder. A job's rounds go to `~/.vibehash/jobs/JOB/history.csv`
as `train` and `job watch` follow it, `job history` fetches them from the node for any job, and `job download` puts
them and a `TRAINING.md` beside the model. Workers only ask the node for their account once a round beyond what they
already asked (the receipts' audit fetches the round's block). The DeepSeek prep units' own stats (seconds, tokens a
second, peak GPU memory) go into the round's line; fine-tune and DeepSeek training units have no such stats yet, so
their lines have the seconds alone. `job history fruit-demo` was tried read-only on the public node (7,178 rounds, 37
that trained, the budget worked back to its 5,000 VIBE exactly); a worker's records have only run on local networks.



**Release 0.5.6 (2026-11-02): prep pieces stay with their miners.** A prep round now takes ready units a piece at a
time (round robin), and each piece goes first to the miners it stands with by rendezvous hashing (coordinator._spread,
_affinity). So a miner fetches the few pieces it's given once, instead of every 29.6 GB piece in turn, which needed
about 350 GB and a 45 MB/s link per prep miner. A miner's first download of a piece still has to fit one compute phase;
giving it its own time is the next step.


**Release 0.5.5 (2026-11-02): fine-tuning, the dispute judge, and DeepSeek-V4.1-Flash's first job.** It brings together:
- PR #5: LoRA fine-tuning of open models where the node never runs the model, plus the bisection dispute judge, after
  two reviews and a re-check;
- PRs #6, #9 and #14: the DeepSeek top-layers route, its split prep, the think-shorter pilot data and the DeepSeek
  referee ops;
- PR #7 and the PRs after it: the node's /network, the new front page, the terminal menu, the installers with the
  finetune extra;
- PR #4: the whitepaper.

The DeepSeek wiring (#14) had one review, and its eight launch-blocking findings were fixed. Three more are noted in
DEEPSEEK.md §8 for after launch. #14 merged on the owner's call before CI finished on its last commit; the publish
workflow's tests are the check on what shipped.

Launch steps that come after this release, in order:
1. the node upgrade, which restarts it and picks up the VIBEHASH_BUCKET_* lines already in /etc/vibehash/env;
2. a short live fine-tune check;
3. post the think-shorter pilot;
4. publish the site with scripts/build_site.py;
5. post on r/deepseek.

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
4b. **The website, https://vibehash.org (2026-10-29)**, is the guide served by GitHub Pages from the public repo `AlphaChad5000/vibehash.org`, checked out at `C:\Users\Schlo\Dev\vibehash.org`. The repo holds only `index.html`, the one-line installers `install.ps1` and `install.sh` (copied from this repo's root), `CNAME` and `.nojekyll`. To update it after changing `docs/guide.html` or an installer, run `python scripts/build_site.py C:\Users\Schlo\Dev\vibehash.org`, then commit and push that folder. The domain's DNS is at Squarespace: A records to GitHub's 185.199.108–111.153, and `www` a CNAME to `alphachad5000.github.io`.
4c. **The public node, https://node.vibehash.org, live since 2026-10-29 on 0.5.1.** It runs on a DigitalOcean Droplet (Ubuntu 24.04, 2 GB, NYC) at 165.227.37.0, reached with `ssh root@165.227.37.0` (key `~\.ssh\id_ed25519`).
   - `deploy/setup-node.sh` installs it: a venv in `/opt/vibehash`, data and wallet in `/var/lib/vibehash`, the wallet password in `/etc/vibehash/env`, the systemd unit `vibehash-node`, native TLS on 443 with a Let's Encrypt certificate (certbot's deploy hook copies it and restarts the node), and ufw open for 22/80/443.
   - The genesis is `vibehash-public-1`: mainnet schedule, earn-in, the node as sole bootstrap validator, and the user's wallet `vibe14164b…7c15fe83` funded for the first jobs.
   - `settings.PUBLIC_NODE` makes `setup` and every command default to it. `settings.chosen_node()` is the user's own pick; the guide's old `NODE-ADDRESS` placeholder counts as no pick.
   - The node listens on IPv4 only, so `node` has an A record and no AAAA.
5. **0.5.1 is on PyPI (2026-10-29)**, published by `.github/workflows/publish.yml` from the `v0.5.1` tag through PyPI trusted publishing (0.4.0 and 0.5.0 before it), and checked by installing it into an empty venv. `pip install vibehash` works now. The next release must be 0.5.2 or later. The guide (artifact version 9) installs from PyPI and has a "Who runs the network" section on validators. `dist/` holds stale local 0.2.0/0.3.0 builds; the 0.2.0 ones are the only copy of that source, so they were kept.
6. **Easier setup: done (2026-10-29).** `setup`, `install-gpu` and `testnet` are tested end to end on this machine: a scratch newcomer ran `setup`, was sent VIBE, staked by re-running setup, and earned with a flagless `vibehash worker` alongside the testnet's own workers. Building the testnet exposed a **blob upload race**: replicated units give byte-identical checkpoints, both holders uploaded the same blob into one shared partial file, the hash failed, and both lost the unit. The fix gives each uploader its own partial file (`GET /blob/<id>/status?from=ADDR` for resuming), with a lock around completion. Tests: `test_setup.py` (detection, build choice, settings, concurrent uploads) and `CommandLineTests` in `test_network.py`.

7. **Audit fixes (2026-10-30), merged from branch `claude/kind-clarke-g5fp9r` 2026-11-01, not yet released.** An outside review found that anyone could exhaust the public node's memory, that canaries and the beacon could be gamed, and several ledger bugs. All fixed, each with a test that fails without it (`test_dos.py`, `test_rules_v2.py`, `test_record_v2.py`):
   - **Node limits** (DESIGN §10, "Message sizes", "Memory", "Peers"): strict `Content-Length`, per-kind message fields and sizes, authenticated future-height consensus messages, `--max-job-mb` (64), no init weights built for untrained jobs, dataset header checks, sync that can't be wedged, capped client reads, the `;`-in-path rate-limit bypass, and proposals whose cached verdict can't be poisoned by a relayed copy.
   - **Block version 2** (`ledger.VERSION`, DESIGN §10 "Block versions"): a duplicate-txid rule, a 1 MiB block limit, bootstrap stake slashed last and only burned, unbonding released after slashes, `cancel_job` paying the round it's in, no proposer pay for an offender, no empty voting set, and exact payout fractions. Version 1 blocks keep their hashes and rules.
   - **Record version 2** (`coordinator.RECORD_VERSION`, DESIGN §3 steps 4 and 9): a leader's round secret, committed in `/status` at open and revealed in the record, goes into the beacon, task ids and canaries. Canaries are rescaled copies of real tasks. Workers check the commitment (`WorkerClient.secret_commits`).
   - **Validators vote only for current versions** (`Node._validate`), while sync and `chain verify` still accept old ones.
   - **Rolling it out:** every validator must run the new release before any version 2 block exists, since 0.5.1 nodes refuse them. On `vibehash-public-1` the node is the only validator, so: release, re-run `deploy/setup-node.sh` (it upgrades from PyPI and restarts), and its first new block is the switch. 0.5.1 workers keep working. One exception: `job download` of a job that hasn't trained yet now gets a 404, since 0.5.1 clients don't build the starting weights themselves.
8. **Sharded units: designed, prototyped (2026-10-30).** `SHARDING.md` is how a model too big for one computer gets trained by a team of workers and checked one shard at a time. `shards.py` and `test_shards.py` prove the core on the MLP: bit-identical to whole-model training, and cheats caught in exactly the cheater's cells. The coordinator runs sharded jobs in simulation (`simulate.py --shards N`), but nodes don't yet; SHARDING.md §5 lists what's left.
9. **One-line install (2026-11-01).** `irm https://vibehash.org/install.ps1 | iex` (Windows) and `curl -fsSL https://vibehash.org/install.sh | sh` (macOS/Linux) install from PyPI and start `setup`; the guide's Earn and Train tabs and the README lead with them. Both open with the dashboard's VIBE HASH banner. `install.ps1` runs inside a script block (so `iex` never closes the user's window), keeps its own text ASCII (irm may not decode UTF-8), and offers `winget install Python.Python.3.14` when there's no Python 3.10+. `install.sh` reads setup's answers from `/dev/tty`, and gives an externally managed Python (Homebrew, Debian, Ubuntu) its own `~/.vibehash/venv`, getting pip from bootstrap.pypa.io when `python3-venv` is missing, and links `~/.local/bin/vibehash`. A node given in the guide reaches setup as `VIBEHASH_NODE`. Tested: both PowerShells (served as octet-stream), and stock Ubuntu 26.04 under WSL (no pip, no venv, no sudo) in a pseudo-terminal. Not tested: the winget path, and a real Mac. Unreleased, in `cli.py`: setup ends by asking "Start mining now?" and hands the unlocked wallet to the worker.
10. **DeepSeek-V4.1-Flash plan (2026-11-01).** `DEEPSEEK.md`: the model measured from its own files (763B weights, 510 GB, 40 layers, 384 FP4 experts a layer, two 101 GB engram tables, a 4-copy residual stream), how to cut it (LoRA, layer shards, per-job engram slices) and what to build, in order. Teams would be network-bound, so FP8 edges and many teams. Step 1 is built (item 13).
11. **The public network works for a newcomer (2026-11-01), with one bug, fixed for 0.5.2.** The first accepted units ever were the owner's, at block 4725. A fresh wallet made by the one-line installer then mined on a CPU for 5 minutes: 19.3 VIBE, the first 5 of it staked automatically, verifying others from its third round. **Bug:** capacity ramps it 4 → 8 → 12 units, and with 12 it misses the 5 s reveal window, which forfeits the round and drops it back to 4, so it loses about every third round. Measured: each reveal took 350–550 ms, because the 0.5.1 worker's `_send_reveal` first asked the node about every checkpoint it had uploaded during compute (one ~60 ms round trip each to New York), and sent the reveals one after another. **Fixed in 0.5.2:** the worker remembers what the node confirmed this round (`WorkerClient.uploaded`) and sends a round's reveals together (`PARALLEL_REQUESTS`). Workers only, so it works with 0.5.1 nodes. Running it against the live node then showed a dropped connection losing one reveal, and with it a round, so a reveal that hits a connection error is now sent again (a duplicate gets "already revealed", which counts as delivered). `DistantWorkerTest` adds 50 ms to each request and drops two reveals; it fails without either fix. Live, the fixed worker then ran 12-unit rounds 4791–4794 with nothing missed.
12. **0.5.2 is on PyPI and the public node runs it (2026-11-01).** Published from the `v0.5.2` tag; the node was upgraded by re-running `setup-node.sh` (the copy already on the server: the script hasn't changed since 10-29). Block 4816 is the first version 2 block, with a version 2 record (the round secret). For a few minutes after the restart the node reset some connections and rounds 4817–4821 accepted nothing; since then both 0.5.1 and 0.5.2 workers mine there (0.5.1 ones still sometimes miss the reveal with 8+ units, which 0.5.2 fixes). A CPU worker at 12 units then committed its last unit just after the compute phase (a missed deadline, which costs reputation though the 11 others are paid): each of a unit's ~9 checkpoint uploads asks the node for its status and then sends it, one after another, about 18 round trips a unit to New York. **Fixed in 0.5.3 (worker only; the node stays on 0.5.2):** a unit's uploads go up together (`PARALLEL_REQUESTS`). `DistantWorkerTest` now uses 9 checkpoints a unit and a 6 s compute phase, and fails with uploads one at a time. Live: 12 units committed in rounds 4844–4847.

13. **DeepSeek-V4.1-Flash step 1: the model in PyTorch (2026-11-01, in 0.5.4).** `vibehash/deepseek.py`, checked against DeepSeek's own `inference/model.py` (fetched by the tests, never copied in; its tilelang kernels replaced by line-by-line PyTorch stand-ins in `tests/deepseek_reference.py`, since they need a Hopper GPU). The small 5-layer model the reference defaults to and one with every feature on match layer by layer at the reference's own CPU-vs-GPU spread; the real layer 2 on real tokens matches to 2.7e-3 rms. Gradients pass finite differences, runs repeat bit for bit. DEEPSEEK.md §7 has the numbers and what they change: tolerances near 1e-2 per layer (FP8 activation rounding), wider tie margins, compressed keys' gradient crossing shards, 0.6–0.75 ms a token a layer, and the CDN slowing long single downloads. `tests.test_deepseek` is in CI (CPU PyTorch, and the Hub for the ~100 KB reference); `RealLayerTests` are opt-in: `python scripts/deepseek_bench.py --fetch-only` downloads layer 2 and the embedding (8.7 GB) first, and the same script times the layer.
14. **Model picker (2026-11-01), released in 0.5.4.** The owner asked for "a dropdown with the popular open weight models". Fine-tuning isn't built, so the picker is honest about that, and nothing in it posts a job or spends VIBE.
   - `catalog.py` holds 17 models, from `qwen3-0.6b` to `deepseek-v4.1-flash`. They're the most downloaded text models of the big open-weight makers on the Hub that day, plus Qwen3.8 27B, GLM-5.3 and GLM-5.3-Flash, which the owner asked for. Every repo was checked against the real Hub. The owner also asked for a Meta model they called "Wonder". The Hub has no such model: Meta's newest open text models there are still Llama 4 Scout and Maverick (April 2025, gated). The owner chose to leave Meta at Llama 3.2 1B and 3.1 8B.
   - `vibehash models [--json] [--refresh]` lists them. `train --base NAME` prints a pre-flight and exits with status 1 before the node or wallet is touched. With no data file and on a terminal, `train` shows a numbered menu with the CSV model first, built from the cache alone so it opens at once offline; only the model picked is looked up. A data file without `--base` is read as a table whatever its name, as on main; when a `.jsonl`/`.txt`/`.json`/`.md` file isn't one, the error adds that fine-tuning on text is coming. Lookups use one 8 s timeout and no retries, and once the Hub proves unreachable the remaining models aren't asked.
   - The guide's Train tab has a Model menu. Its default is today's CSV model, with the steps unchanged. The other models show their baked facts, the pre-flight command, and the future `train my-chats.jsonl --base …` command, drawn dashed and with no Copy button. Re-bake with `python scripts/bake_models.py` after changing `CATALOG`; `test_catalog.GuideTests` fails until you do. Checked at 375 px and 1280 px.
   - **Forge emblems** (asked for by the owner): the training dashboard works the base model's emblem instead of a blade (`emblems.py`, `trainui.TrainDashboard._forge`). No real job has a base yet, so `trainui.parse_job` reads a future spec field `base` (`hf:org/name@commit`, as hub.py's docstring has it) and `JobWatch` also takes `base=`. Either passes only if it's a Hugging Face reference in plain ASCII (`trainui.safe_base`): a spec comes back from the node as its poster wrote it, and anything else could carry escape sequences to the watcher's terminal. **Trademarks:** each emblem is a simple pixel rendering of its maker's logo, used only to say whose model is being trained, with no wordmarks or slogans. The owner decided to keep them as drawn, logo-like ones included ("Keep the emblems", 2026-11-01), with a disclaimer wherever model names and emblems appear (`catalog.DISCLAIMER`, the guide's Train tab and footer, the README, `emblems.py`). Drawn by hand and from shape equations at 18×10; the OpenAI knot is the roughest. To change one, edit its rows in `emblems.py` and run `python scripts/preview_forge.py NAME`.
   - The site went up with the model menu once 0.5.4 shipped `models` and `--base` (0.5.3 hadn't): the Train tab tells people to run them.
   - **`Hub.describe` failed on every gated model** (Llama, gated Gemmas) for an account that hadn't accepted the licence: the real tree API hides those repos' LFS sha256 as `****…`. The DeepSeek-layer session's PR #2 makes `hub.py` read that as hidden. `catalog.describe` keeps its fallback for now: it uses `model_info` alone and estimates the size from the parameter count per dtype (shown with `*`). `catalog.py` also reads layers and experts from `text_config` until `describe` does (DEEPSEEK.md §5 step 3).

15. **Fine-tuning pretrained models (2026-11-01, branch `claude/finetune-0d3007`, PR #5).** A poster runs
   `vibehash train chats.jsonl --base qwen3-0.6b`; workers' GPUs train LoRA adapters in verified units; `job download`
   writes a PEFT adapter. Proven on a local testnet with Qwen3-0.6B (FINETUNING.md "Try it" has the run).
   - **Combining:** A frozen, B trained, so averaging units' B is exactly averaging their change to the model (DESIGN
     §8 has the alternatives). Adam per unit, moments in each checkpoint (`[B|m|v]`, 3× B's size), zeroed after
     averaging. Checks compare B, m and v each on its own; the sanity check is Adam's own bound on how far a weight
     can move (B starts at zero, so "relative to the start" meant nothing).
   - **Spec:** its own shape (`tasks.FINETUNE_KEYS`): `base` (hf:org/name@commit), `manifest` (sha256 of hub's
     manifest), `model` (the dims), `lora`, `seq_len`, `optimizer`, data `{"kind": "tokens"}`. MLP specs are
     unchanged byte for byte; old nodes refuse a fine-tune's post_job as malformed, so they can't accept a fine-tune
     round either. The dataset .npz carries the manifest and config.json, so workers need nothing else.
   - **Measured:** matches transformers (2.1e-6 on Qwen3-0.6B; every family on tiny models), CPU replay of a CUDA
     segment within 2.6e-6 of the segment's change, same GPU bit-identical; tolerance 1e-4. A 16-step unit of 8
     chats is 9.5 s on the 4070 SUPER. Learning rate from a sweep (DESIGN §8).
   - **Nodes don't run the model (record version 4, DESIGN §8 "Fine-tune rounds without the model on the node").**
     A fine-tune's round gives each claim two checkers and has three scorers score its starting model (the record
     keeps the median, for people only). Juries are gone (below): a rejection is a dispute game.
   - **After the review of PR #5 (2026-11-01):**
     - **What counts as a fine-tune:** only a spec with `lora` and a model without `n_in` (`tasks.is_finetune_spec`),
       so an MLP spec with a stray `base` stays an MLP spec.
     - **Block version 3** is the first that may hold a fine-tune's post_job (`ledger.FINETUNE_VERSION`).
     - **Bad data:** config.json is checked strictly when a dataset arrives (`finetune.arch_dims`), and a job that
       won't load is skipped, not fatal.
     - **Canaries** are another unit's real training (the old rescaled ones broke Adam's invariants, which a lazy
       checker could test with numpy, `verification.adam_impossible`). A canary-approver's verdicts don't count.
     - **Scorers** are paid only when three agree.
     - **Draws** are seeded per claim.
     - **Timing:** no phase is over an hour (`node.MAX_PHASE_S`).
     - **Tolerance:** comparing replicas uses `finetune.WHOLE_TOLERANCE` (1e-3; measured 5e-5 CPU vs CUDA).
   - **Disputes in place of juries (DESIGN §8, `dispute.py`):** a checker's rejection, or two replicas that disagree,
     is a bisection game in a `dispute` phase of the same round. The worker posts the states between its training's
     ops (unit → segment → step → sequence → layer → op, up to 64 children a split), the challenger picks where they
     go wrong, and the node recomputes one tile of one op with NumPy (`referee.py`: float64 with a bound on float32
     rounding; no libm, so every machine agrees). Losers are slashed; a false accusation's slash is burned; a checker
     that finds no number the referee would refuse withdraws (unpaid, not slashed). The weights it needs are fetched
     whole from the pinned commit and checked against the manifest (`dispute.HubWeights`); Hub unreachable means
     unchecked.
     - **Training changed with it:** it runs `program.py`'s ops one by one (`tracer.py`), so the states a worker posts
       are its training's own. Gradients match autograd to ~1e-6. It's 2.2× slower a step on Qwen3-0.6B (1.62 s
       against 0.72 s at batch 8); batching a step's sequences in each op is the follow-up.
     - **Start gate:** `FINETUNE_QUORUM` = 2 (one to train, one to check): one honest checker is enough for DESIGN's
       security statement.
     - **New HTTP:** `GET /round/games`, `POST /move`; a dispute's defender may upload its states (quota per round).
     - **Records:** `games` (signed moves and outcomes; no weights, and evidence by blob id), `checks/game`. A
       validator replaying a round with disputes rebuilds each game's starting states from the proposer's checkpoints,
       fetches the evidence and the job's dataset, and reads only weights it checked; no PyTorch. Moves get receipts.
     - **Second review** (12 findings): one commit each on the PR. Left as known limits in DESIGN: colluders can tell a
       canary by its checkpoints' ids; a split can name gigabytes on big adapters with many steps a segment.
     `vibehash node --device` still exists for MLP checks. `testnet --job none` posts no demo job, for `train --base`.
   - **hub.py:** `TensorCache` now fetches 64 MiB runs over six connections (1.5 GB in ~15 s here); manifests pin
     `chat_template.jinja` too.
   - **Tests:** `tests/test_finetune.py` (in CI with CPU PyTorch and `.[finetune]`; transformers/PEFT checks run where
     they're installed; `VIBEHASH_REAL_MODELS=1` for Qwen3-0.6B). `tests/finetune_fixtures.py` makes tiny checkpoints.
   - **Rolling it out:** this release makes block version 3, which a 0.5.x node refuses, so every validator runs it
     before the first new block (as with version 2); on `vibehash-public-1` the node is the only validator, so
     updating it is the switch. Size `--keep-rounds` to the disk (about 3 GB a round of 8 Qwen3-4B units). The public
     network then needs two staked GPU workers before a fine-tune trains, and the node needs to reach
     huggingface.co for disputes (byte ranges of the base model; a few MB each). The live chain was scanned on 2026-11-01
     (height 6187): its one post_job (height 2) has no `base` field, so nothing on it changes meaning.
   - **Not done:** `vibehash hub login` (gated Llamas need `HF_TOKEN` set by hand); manifests shared on the network
     (each poster builds and caches its own); moments in bf16; cache limits on the worker; Intel/AMD/Apple
     tolerances.
16. **DeepSeek's first project: only the top layers trained (2026-11-01, branch `claude/deepseek-top-layers`, a draft PR).** DEEPSEEK.md §8: freeze the bottom 40 − K layers, precompute what crosses the cut once (42 KB a token: the stream and layer 20's keys), and have miners train LoRA on the top K from it, needing 31 GB for K = 4. `deepseek_top.py` and `tests/test_deepseek_top.py`.
   - Done: training from precomputed inputs is bit-exact to the whole model with its bottom frozen.
   - Done: train in float32, since bf16 replays differ 4–9% between CPU and CUDA.
   - Done: an expert-major schedule for 6–8 GB GPUs, gradients within float32 rounding.
   - Done: adapters PEFT loads; the dispute game's commitments and referee ops, written up.
   - Skipped by the owner: the offline speed and quality runs (`scripts/deepseek_top_bench.py`, `scripts/topk_quality.py`, the 6/8 GB caps). The scripts stay. DeepSeek is proven on its first live job instead, which measures itself: deadlines slow miners can make (`deepseek_split.deadline`), the loss curve (netstats' job summary, the worker dashboard), and each unit's tokens a second and peak GPU memory (`deepseek_top.Meter`, in a prep unit's commitment "stats").
   - Done: the real top 4 layers and the head are downloaded (30.9 GB, in `~/.vibehash/cache/deepseek`).
   - Done: the prep as one resumable command (`deepseek_prep.py`), for a rented GPU or a home PC. About 470 GB down, ~280 GB of disk; estimated ~$3.60 per million tokens on an L40S. Not run on real weights. Now a dev and test tool only.
   - Done, prototype: the split prep (`deepseek_split.py`), the design: the owner settled that one person or machine doing the prep would break "nobody controls it", so there is no preparer role. Units of G layers by chunk, lossless hand-offs (28 KB a token a cut), a DigitalOcean Spaces bucket through presigned URLs, optimistic consumption with rollback. Recommended G = 4 (29.6 GB a miner): 232 GB of hand-off egress per million tokens, ~4 h end to end for 1M tokens with 36 miners on cable (simulated). Numbers: `scripts/deepseek_split_plan.py`. Wiring into the node waits for PR #5.
   - Done: multipart, resumable hand-off uploads and ranged downloads (`PresignedStore`, `Signer`; tested against a local fake S3 that checks signatures, with a flaky link and a crash part way); answering a challenge by replaying a unit (`replay`, `state_before`); claim deadlines (`Line.expire`).
   - The first job (the owner's choice): think shorter's pilot (DEEPSEEK.md §8). Cut at 36, K = 4; `~/.vibehash/datasets/think-shorter/think-shorter-pilot.npz` (sha256 f4d7f822…523a3a): 1,500 GSM8K problems to train on and 500 held back, 341,753 tokens; 3 passes. Its builder and card (PR #9's `scripts/make_think_shorter.py`, `docs/datasets/think-shorter.md`) are on this branch. Prep at G = 4 in 32K-token chunks: 129 units, 79 GB of hand-offs, ~3.5 h with 9 cable miners (simulated).
   - Wired (branch `claude/deepseek-wiring`, draft PR #14, stacked on PR #5 with its judge merged in): the job kind, prep rounds on the node (links signed for each unit's own objects, two checkers replaying each unit, the root pinned when the last is in), the worker's steps, training and scoring from the records, the menu item and `worker --deepseek`, setup's model-files step and `vibehash cache`. Run end to end over HTTP on the tiny config (`tests/test_deepseek_network.py`). Training units and prep layers units are refereed through the judge (deepseek_program.py: DeepSeekProgram and PrepProgram; deepseek_tracer.py; the fine-tune session's referee_deepseek.py), losers slashed. Prep runs in 4K-token chunks (the owner's call: chunk-wide states stay small). For a rejected prep unit the node first checks its outputs are its last committed state, then plays a game over the unit. Rows units still end unchecked when rejected. The referee's Flash weights: the trained layers and head checked per tensor (vibehash/data/deepseek_flash_top.json, pinned), the frozen layers and tables read from the pinned Hub commit, checked at the file level (the owner's option C; a trust point in DESIGN). Still to do there: merge PR #5's review fixes (Recorded gone, dispute_size, ipow), the guide section, disk use on the dashboard. Also for launch day (DEEPSEEK.md §8, "For launch day"): the worker asks before the ~30 GB download (with an unattended yes), says clearly when a PC can't take DeepSeek work, and docs/guide.html gets a "Mining the DeepSeek job" section. When wiring, fill the front page's featured-job panel (DEEPSEEK.md §8, build order 5–6): `Line.summary` as the job summary's "prep" field; "base", "batch_size" and "steps_per_unit" in the spec; held-back loss into `store.job_rounds`; "think-shorter" in the first job's id.
   - The fourth review (2026-11-02, PR #14 at b557b54) found 11 problems. Fixed: the model store deleting files outside its own folder (and never cleaning up), zstd hand-offs the node's Python 3.12 couldn't read (zlib is the only codec now), write links that outlived their compute phase, the node reading miners' objects inside its lock with no size cap, a bucket error slashing an honest miner, prep canaries their task gave away, prep games timed for a LoRA adapter (off for the pilot: a rejected prep unit is redone, nobody slashed; `coordinator.PREP_GAMES`), and paid rejections. Left for after launch, in DEEPSEEK.md §8: validators without the bucket's key, prep progress without the job's data, uploads with no size bound, objects named by graph unit, and the hand-off check's reads.
   - Known centralization points, written up in DEEPSEEK.md §8: the public network's single validator (the bigger one), and v1's operator-paid hand-off bucket, which can't alter data undetected but can withhold it (that costs recomputation; the way out is miner-run hosts behind `Mirrors`).
   - Measured on real weights on the CPU (`scripts/deepseek_stream_stats.py`): the 4 stream copies don't compress beyond FP8, and lossless byte planes + zstd take the stream to 0.68–0.71. FP8 hand-offs at every cut are unmeasured, since the stand-in chain is thrown as far by a 0.4% nudge, so hand-offs stay exact. Layer 0's routing on its real input (only its 170 MB outside the experts, read by range) uses all 384 experts within 262K tokens of chats, and 382 in a 64K-token chunk.

## Pre-deployment checklist

Blockers before inviting anyone onto a public testnet:

- [x] Put the OneDrive folder under git (see top).
- [ ] Items 1–5 above.
- [x] **License:** MIT, chosen by the owner 2026-10-30 (`LICENSE`, and `license = "MIT"` in pyproject, which needs setuptools 77+).
- [ ] **Faucet.** Earn-in (Key design decisions) means workers no longer need one: they start with nothing and earn their stake. People who want to *post* jobs still need VIBE from a transfer, so a rate-limited faucet would still help requesters.
- [x] **Publish to PyPI:** 0.4.0 is out (2026-10-29). Later releases: bump the version and push a `v*` tag (README, "Publishing a release").
- [ ] **Node hardening:** ~~TLS; per-IP rate limits; connection caps; transaction fees~~ done 2026-10-29 (DESIGN §10: `tls.py`, `ratelimit.py`, `fees.py`). Message, memory and peer limits, and a block size limit, done 2026-10-30 (item 7). Still open: a per-address dataset quota (anyone with ≥1 VIBE can upload 20 GB datasets repeatedly); public blob GETs can be abused for bandwidth; replace-by-fee; a funded job up to `--max-job-mb` still costs the leader real memory each round.
- [ ] **Operations:** run the node as a service; back up `node-data/` and the node wallet; monitoring and alerting; disk-space alarms (opened checkpoints are kept for `--keep-rounds` rounds, about units × (2q+2) × model size per round; datasets are never deleted).
- [ ] **Load test:** many workers, a large model, and real home upload speeds. Tune phase durations (`--compute-s`, etc.). Defaults are sized for small models.
- [ ] **Security review:** fuzz the HTTP API; have someone else review ledger/coordinator/verification. Nothing has been audited.
- [x] **Wallet recovery:** done 2026-10-29. `wallet create` and `setup` show 24 BIP39 words (SLIP-0010 Ed25519, `m/44'/1'/0'/0'/0'`), and `wallet restore` rebuilds the wallet from them. A mainnet needs its own SLIP-0044 coin type.
- [ ] **Issuance:** `genesis create --economics mainnet` gives a Bitcoin-like schedule (3.125 VIBE a round, ~4-year halvings, 21M cap). Choose the real numbers before any launch.
- [ ] **Versioning:** (block and record versions exist since 0.5.2, but `/status` still doesn't report the software version) protocol and client version in `/status`; refuse incompatible clients; plan for chain upgrades.
- [ ] **Legal:** keep it explicitly valueless and testnet-only; terms and privacy notice (the node sees worker IPs); legal advice before anything that could carry value.

Known limits (fine for testnet, document rather than fix now):
- One leader per round: it can't forge payments, bend the round rules (the other validators replay its record and vote it down) or drop messages it accepted (receipts) undetected. It can still refuse messages outright in the rounds it leads, and its weight-based check results are taken on trust unless someone rechecks them within the challenge window. The chain keeps going while validators holding more than two thirds of the stake are up. DESIGN §9 lists the fixes in order.
- MLPs, and LoRA fine-tunes of models one GPU holds (Llama, Mistral, Qwen, SmolLM). The public node can't run fine-tunes yet (FINETUNING.md, milestone B).
- `Dataset.from_npz` loads the whole dataset into RAM; the node evaluates on the CPU.
- Verifiers download ~2q checkpoints per task, which is significant bandwidth for big models.

## Working tips

- The original session folder's path was too long for some `pip` installs (numpy, torch). The OneDrive path is short enough. A virtual environment works there, as long as it's excluded from git, or kept outside OneDrive so OneDrive doesn't sync gigabytes of PyTorch.
- In the Claude Code Bash tool, long heredocs containing quotes sometimes failed to parse. Writing edit scripts to a `.py` file and running them was reliable. The repository's files are LF, and Python's `open(p, "w")` writes CRLF on Windows: pass `newline="\n"` (Git Bash's `grep` doesn't show the `\r`, `git diff --stat` does).
- `simulate.py` and `test_core.py` are fast (seconds). `test_network.py` takes about a minute (live HTTP nodes). `test_gpu.py` needs PyTorch.
- Signatures fall back to slow pure-Python Ed25519 without `cryptography`. Install it for realistic timing.
