# What comes next: an app, every kind of computer, less bandwidth, faster checks

**Status: a design, not code.** This is the order of work after the Muse agents merge (PR #28), with
each idea checked against what the code does today (release 0.5.15). Where it says *today*, there is a
file to read; where it says *to build*, there isn't yet. Nothing here changes the chain's rules until it
is built, tested and in a record version of its own.

The order:

1. **Documentation**: done in the same PR as this file (`docs/CLI.md`, generated from the parser).
2. **An app people install**, which joins the network on its own and shows what is happening.
3. **Every kind of computer useful**: laptops, phones and tablets alongside gaming rigs.
4. **Bandwidth**: units that don't ship a gigabyte of floats around.
5. **Faster checks** that stay as thorough.

Items 3 to 5 depend on each other: a laptop only helps once a unit is small, and a small unit only pays
once checking it is cheap. So they share one design, and 2 is built first because it needs none of them.

---

## 1. The app

**Today.** Everything is the `vibehash` command (`cli.py`), with a menu when run with no arguments
(`menu.py`), a live mining scene in the terminal (`minerui.py`, `esui.py` for Evo) and a training
dashboard (`trainui.py`). The node already computes what a public page needs at `GET /network`
(`netstats.py`: who mined in the last ten minutes, what is training, how well), and reputation is a
number on every account (`economics.update_reputation`, shown by `vibehash balance`). `setup --yes` and
`unattended.py` already run without a terminal.

**What the app adds.**

- **Start and forget.** One entry point that does setup, joins the network and starts the worker with
  no prompts, and starts again on boot. A tray icon with *pause while I'm gaming* (the worker already
  sits rounds out cleanly; the pause is a flag it reads between units).
- **Three views of the same thing.** The wallet (balance, stake, what each round paid); the network
  (jobs training, who is online, as `/network` shows them); and the worker, either the mining scene as
  it is or a step-by-step log of every unit: registered, trained, committed, checked, paid.
- **Who is online, anonymously.** A count per hardware class (so many of one card, so many of another)
  and the most and least reputable addresses. Reputation is public on the chain already; what is new is
  the hardware class, which workers don't report today (§3).

**How to ship it.** Not a PyInstaller bundle with PyTorch inside: that is a 3 to 5 GB download, trips
Windows SmartScreen on an unsigned binary, and carries one CUDA build to people with AMD cards, Apple
silicon or no GPU. Instead:

- a **small launcher** (tens of MB) that, on first run, detects the hardware the way `install-gpu` does
  (`gpusetup.py`) and makes an isolated Python environment with the one matching PyTorch wheel, with
  `uv` doing the environment work;
- a **window** around a local web dashboard (reading `/network` and the worker's own status; built: window.py, the
  system's web view through pywebview, with the browser as the fallback) and a
  terminal tab that streams `minerui.py` as it is, so the cinematic isn't rewritten. Tauri is the
  lightest shell for that; the first version can be the terminal alone behind the launcher.

What to build first: the hardware census (§3, it is tiny), then the launcher, then the window.

## 2. Payment that stays honest when the machines differ

**Today.** Pay is split by observed training speed: a worker's share of the round's 80% is its FLOP/s
(`hardware.benchmark`, which times the job's own training loop, capped by what the coordinator sees)
weighted by reputation; checkers share 15%, the proposer 5% (`economics.py:21-23`, `compute_payout`).
A phone or an office laptop earns a rounding error, and would earn nothing useful even if it tried a
unit, because a fine-tune unit needs the whole adapter, its two Adam moments and the base model in
memory.

**What not to do.** No multiplier for weak devices. Anything that pays a phone more per FLOP than a GPU
pays whoever runs two hundred phone emulators on one server.

**What to do: give each class work it is actually good at, paid from its own pool.**

| Tier | Machines | Work | How it is checked and paid |
|---|---|---|---|
| Heavy trainers | discrete GPUs, pods, Macs with 16 GB+ | training units as today; checking units of the same hardware class (§3) | per verified FLOP, from the worker and checker pools, as today |
| Standard laptops | integrated graphics, 8 GB laptops, office PCs | small LoRA shards (rank 4, a few layers: `shards.py` is the start) and forward-only work: scoring held-out data, blind judging of a fine-tune's answers (no backward pass, no optimiser state, about a third of the memory) | forward passes are deterministic enough to agree on greedy tokens, so two of three agreeing hashes pay; a new eval pool |
| Phones, tablets, browsers | iPhone, iPad, a web page | tokenising and de-duplicating datasets, canary audits (a prepared piece with a known answer), and human blind preference votes | per verified micro-batch, or per vote that lands with the majority; Sybil-gated by stake or device attestation |

Three facts set the boundaries. **Memory:** replaying one segment of a 0.6B LoRA unit needs the base
weights and activations; iOS stops a background page past about 2 GB, so phones can't check training
units, only forward or data work. **Floating point:** a phone's CoreML or WebGPU and an Intel CPU don't
accumulate in the order an RTX card's tensor cores do, so cross-class replay is what forces loose
tolerances (§3), and the laptop tier's checks are on token outputs, not weights. **Sybils:** every pool
pays only for work that was checked, and the phone tier's votes count only against a stake or an
attested device.

## 3. The hardware census, and checks between like machines

**Today.** A worker registers with `{round, flops_per_sec, memory_mb}` (`protocol.py:107`,
`coordinator.py:170-174`). No device name or vendor. Checkers are drawn at random from the staked
workers with enough memory who don't hold the unit (`coordinator.py:1332-1395`). Replays are compared
by distance, never hash: a fine-tune's B within 1e-4, its Adam moments within 1e-3, sign flips set
aside on the first segment (`verification.py:99-152`, `finetune.py:76`). Those numbers were measured
with an RTX 4070 SUPER checking a CPU; there is no MPS-specific handling. Fine-tunes train in float32
because bf16 puts CPU and CUDA replays 4 to 9% apart (HANDOFF.md).

**To build.** Registration gains a hardware class: a canonical tuple from `hardware.py`,
`(vendor, arch_class, vram_gb, label)`, for example `("nvidia", "ada_sm89", 8, "RTX 4060")` or
`("apple", "mps_m3", 16, "M3")`. Never free text: the label is looked up from the detected device,
so nobody registers a card that doesn't exist to top a leaderboard. The node:

- **counts classes** for `/network` (how many of each, anonymously);
- **pairs checkers by class** when it can: the pool filter at `coordinator.py:1332-1336` prefers
  checkers of the worker's class, and a per-class tolerance table lets same-class checks tighten by
  orders of magnitude (two Ada cards running the same float32 kernels should agree far closer than
  1e-4). Cross-class checks stay allowed with today's numbers, so a class with few members is still
  checked. Later, same-class checks of a canonically rounded state (§4) can be hash-exact.

This is small (a field, a lookup table, a preference in one filter) and unlocks both the online view in
§1 and the tolerances §5 needs.

## 4. Bandwidth

**Today.** A fine-tune unit is 16 steps in 4 segments, 3 of them challenged (`finetune.py:831`,
`coordinator.py:151`). What a worker uploads is every segment boundary plus the final state, all
before it commits (`client.py:726-735`); the reveal only names which openings the beacon asked for.
Each upload is the full `[B | m | v]` vector, raw little-endian float32, no compression
(`finetune.py:192-200`, `blobs.py`). Blobs go to the node over HTTP; `p2p.py` gossips consensus, not
blobs. LoRA-FA means A is never sent (`finetune.py:8-14`), which already halves what a plain LoRA would
move. Checkpoints up to 1 GiB, and every checker holding the whole model, is the cap on job size that
SHARDING.md addresses.

**To build, cheapest first.**

1. **Upload after the beacon, not before.** Commit the Merkle root (32 bytes a segment), then upload
   only the boundaries the beacon challenged plus the final state. Today that is 4 of 5 blobs, so the
   saving is small; it grows with the number of segments, and it is what makes finer segments (§5)
   affordable at all. Touches `client.py:726-735` and `worker.py:183-185`; the reveal already carries
   the openings.
2. **Send the change, rounded to a grid.** Every checker knows the round's starting state, so send
   `[B | m | v]` as its change since the round start, rounded to a canonical grid (bf16, or block-wise
   int8 with a per-block scale) *before* hashing, so worker and checker hash the same bytes. Then zstd
   on the wire. Raw float32 compresses by about 1.2x because mantissas are noise; a rounded delta
   compresses 4 to 8x. The moments are two thirds of today's bytes, so this matters most for them.
   It changes the record format (a new version, as v2 to v4 did) and the tolerance rules in
   `verification.py:99-152`, which must be re-measured against the rounded state.
3. **Shards**, as SHARDING.md: the only way past one machine's memory, and what makes the laptop tier's
   small shards possible.

## 5. Faster checks that stay thorough

**Today.** A checker replays the challenged segments in full (`verification.py` layer 4), which is 3 of
4 segments for a fine-tune: most of the network's compute is checking. Disputes go to a bisection
game, unit to segment to step to sequence to layer to op to tile, up to 40 moves, and the node recomputes
one tile in float64 (`dispute.py`, `program.py:355-408`, `referee.py`). 20% of units are secretly given
to a second worker and compared by tolerance (`coordinator.py:150`, `verification.py:132-139`).

**To build: two stages, optimistic.**

- **Stage 1, on every unit, cheap.** Two tests the checker runs without replaying 16 steps:
  - *A sketch of the update.* The beacon, which comes after the commitment, seeds a random probe. For
    LoRA-FA the weight change is `ΔW = ΔB · A`, so a checker verifies the committed `ΔB` against the
    claimed `ΔW` with one matrix-vector product, `ΔB (A r)`, in time linear in the adapter instead of
    quadratic. Freivalds' check and the Johnson-Lindenstrauss lemma say a random projection catches a
    wrong product with probability close to 1 per probe.
  - *One honest step.* Pick one of the 16 steps from the beacon, replay just that step from the
    committed boundary, and compare against the logged gradient norm and the next committed state.
    That is a sixteenth of today's compute.
- **Stage 2, on anomaly or a random 5% audit.** Full replay of the challenged segments as today, then
  the existing dispute game.

**Say what it proves.** The sketch verifies a matrix product: that the committed `B` and the claimed
weight change agree. It does not verify that `B` came from 16 honest Adam steps on the right data; the
one-step replay is what covers that, and it is a sample. Before this replaces full replay, measure the
false-accept rate of the one-step check against the cheats the tests already play (a skipped step,
a wrong batch, a copied neighbour), and keep the random full audit high enough that cheating a sample
check costs more in slashed stake than it pays. Same-class pairing (§3) is what lets Stage 1 use tight
tolerances; without it the sketch has to be as loose as today's cross-device numbers.

---

## What to build, in order

1. Hardware class on registration, counted at `/network`, preferred in checker choice (§3). Small.
2. The launcher and the window (§1).
3. Upload after the beacon (§4.1). Small, and a prerequisite for finer segments.
4. Rounded deltas and compression on the wire, as a new record version (§4.2).
5. The laptop tier's forward-only eval pool (§2), which needs none of the verification changes.
6. Stage 1 checks, measured on the testnet before they replace anything (§5).
7. Shards on the network (SHARDING.md §5), and with them small shards for laptops.
