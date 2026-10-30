# Vibe Hash (VIBE) — Proof-of-Useful-Work for AI Training

Vibe Hash is a cryptocurrency whose new coins are issued for **verified AI training
compute** instead of for hashing. It is aimed at mid- and low-end machines: old
laptops, gaming PCs with a single consumer GPU, small home servers.

This repository contains a runnable **testnet implementation**: a node that
coordinates rounds over HTTP, a worker client that benchmarks the machine it
runs on and trains real models, wallets, a persistent ledger anyone can
re-validate, and an in-process simulator that drives the exact same round
logic for fast experiments. Section 9 lists what still stands between this and
a mainnet.

---

## 1. How it differs from Bitcoin mining

| | Bitcoin | Vibe Hash |
|---|---|---|
| Work | SHA-256 hashes (useless outside the chain) | SGD training steps on a real model (useful to whoever posted the job) |
| Who gets paid | One lottery winner per block | **Everyone** whose work verified, pro rata to FLOPs |
| Small rigs | Effectively priced out by ASICs | Paid the same rate per verified FLOP as a big rig |
| Verifying work | Trivial (one hash) | Hard — solved with replay spot-checks, redundancy, commit–reveal, staking and slashing |
| Where money comes from | Block subsidy + tx fees | Block subsidy (halving) + fees paid by people who want models trained |

Because payment is proportional instead of a lottery, a 5 GFLOP/s laptop earns
a small but **steady** income, with no need for a mining pool.

---

## 2. Actors

- **Requester** — posts a training job (model spec, dataset commitment, hyper-
  parameters) and escrows a VIBE budget to pay for it.
- **Worker** — stakes a small amount of VIBE, benchmarks its hardware, and
  computes *work units*.
- **Verifier** — any staked worker, assigned at random to re-check another
  worker's unit. Verification is itself paid work.
- **Proposer** — a staked node chosen (stake-weighted, from the round beacon)
  to assemble and sign the round's block.

---

## 3. A round, step by step

1. **Schedule.** The current global model `W_r` is split into *work units*:
   "starting from `W_r`, run `S` SGD steps with batches derived from seed `σ`,
   checkpointing every `c` steps". Units are deliberately small so the slowest
   eligible rig can finish one before the deadline. Faster rigs get *more*
   units, not bigger ones. Units whose memory requirement exceeds a rig's RAM
   are never assigned to it. A random fraction of units is secretly
   **replicated** to a second worker. Capacity comes from the worker's own
   benchmark over the round's compute phase, but can grow by at most 2× a
   round when it delivered everything, falls back to what it delivered when
   it missed a deadline, and starts at a few units for new workers. So
   overstating your speed to grab work and then failing doesn't pay.

   **Round length follows the job.** A job's checkpoints have to move:
   the starting weights down to every worker, each unit's checkpoints up,
   and the challenged ones down to its checkers. That grows with the model.
   So each job's phases are at least what `TrainingJob.phase_lengths` says
   a modest computer on a home connection needs (an old laptop's CPU at
   2 GFLOP/s, 20 Mbit/s up, 100 Mbit/s down; the constants are in
   `tasks.py`):
   - **compute:** download the weights, train one unit, and upload its
     checkpoints, with 25% to spare;
   - **reveal:** resend two checkpoints;
   - **verify:** fetch and replay the challenged segments of two units.

   A job file can also ask for a longer compute phase (`compute_s`, 5–3600 s),
   for bigger units. Nodes run each round for the longer of their own phase
   lengths and the job's. The compute phase a round actually had is written
   into its record's rules, since capacity depends on it. Validators wait for
   a proposal as long as the longest funded job's round needs. A small job
   keeps the 38-second default; the GPU template (270k weights) still fits
   in it. A model of three 4096-wide layers (34M weights, 134 MB per
   checkpoint) gets about 17-minute rounds: 12 minutes of compute, most of
   it spent uploading 8 checkpoints per unit at 20 Mbit/s.
2. **Compute.** The worker trains, saving checkpoints `C_0 … C_k`. It builds a
   **Merkle tree** over the checkpoint hashes.
3. **Commit.** Before the deadline the worker publishes a signed commitment
   `H(unit_id ‖ pubkey ‖ merkle_root ‖ salt)`. Nothing about the result is
   revealed yet, so other workers cannot copy it.
4. **Beacon.** After the deadline, the round's randomness is
   `beacon = H(prev_block_hash ‖ round secret ‖ all commitments)`. Nobody can
   know it before committing, so nobody can predict which segments will be
   checked. The round secret is drawn by the round's leader when it opens the
   round. It publishes the secret's hash in `/status` (`beacon_commit`) right
   away and the secret itself only in the sealed round's record. Without the
   secret, a worker that knew every commitment could re-salt its own until the
   checks missed what it faked: a worker alone in a round, say. A worker
   managed 4 half-faked units in 4 seconds. Workers check the record's
   commitment against the one they saw, so the leader can't swap the secret
   after seeing the commitments either. A leader that also mines in a round
   it leads still knows the secret; the rounds other validators lead are
   safe from it.
5. **Reveal.** Workers reveal `merkle_root`, `salt`, the final weights, and
   the checkpoints the beacon selected (first, last, and both ends of each
   challenged segment) with Merkle proofs. A reveal that doesn't match its
   commitment, or is missing a proven checkpoint, is fraud. **Committing and
   then not revealing** forfeits the worker's entire round (every unit, and
   any verification pay), costs reputation, and resets it to the newcomer
   cap. It is *not* slashed, because to the protocol it is indistinguishable
   from a laptop that crashed or lost Wi-Fi mid-round. Withholding
   selectively still doesn't pay: faking a fraction `f` of a unit gets caught
   with probability `p = 1 − (1 − f)^q ≥ f`, so backing out loses `p × R` to
   save `f × C`, which only comes out ahead if compute costs more than it
   earns (`C > R`), and then nobody would mine at all.
6. **Sanity check.** Final weights must be finite, must equal the last
   checkpoint, and the update norm must be within bounds. Catches garbage and
   poisoned "boosted" updates cheaply.
7. **Replay spot-check.** A randomly assigned verifier derives `q` segment
   indices from the beacon, asks the worker for checkpoints `C_i` and `C_{i+1}`
   with Merkle proofs, re-runs the `c` steps from `C_i`, and compares to
   `C_{i+1}` within a tolerance (to allow for hardware floating-point
   differences). The first checkpoint must hash to `W_r`, so a worker can't
   start from made-up weights. A rejection is only a *claim*: the coordinator
   re-runs the disputed segment, and a verifier who rejected honest work is
   slashed for a false accusation. A verifier who doesn't respond loses
   reputation and the coordinator checks the work itself.
8. **Redundancy check.** For replicated units, both workers' final weights must
   agree. On disagreement, a full replay decides who lied.
9. **Verifier audits.** The protocol slips **canary** submissions (known-fake)
   into verifiers' queues. A verifier that approves a canary is lazy and is
   slashed. Verifiers can't tell canaries from real work. A canary is a copy
   of a real task with one segment's update rescaled by 15–25%:
   - it fails a replay;
   - it keeps the structure and size of honest training, unlike random
     noise, which shows without one (honest updates to a softmax layer's
     biases sum to zero);
   - it goes to a checker that holds neither the unit nor a real task on it;
   - its task id is drawn, like every task id, from the round secret.

   Before that secret existed, canary ids followed from the public beacon,
   and a checker that replayed nothing picked out every canary (176 of 176
   tasks in a simulation) and was paid in full. A round needs three staked
   workers able to check before it can plant a canary.
10. **Aggregate.** Accepted updates are averaged into `W_{r+1}` (local-SGD /
    federated averaging). Validation loss is tracked so the requester sees
    their model improve.
11. **Pay and seal.** The block carries a *round summary* (verified FLOPs per
    worker and verifier, accepted units, slashes). Every node derives the
    exact reward and slash transactions from the summary and the chain state;
    a block whose transactions differ is invalid.

---

## 4. Why cheating doesn't pay

If a worker fakes a fraction `f` of its segments and the verifier checks `q`
random segments, the chance of getting away with it is `(1 − f)^q`. With
`q = 3` and `f = 0.5` that's 12.5 %, and replication adds a second,
independent chance of being caught.

Getting caught costs **50 % of stake** plus the reward for that round, and
resets reputation. The protocol sets the minimum stake so that

```
expected_gain_from_cheating  <  P(caught) × (slash + lost_reward)
```

for any `f` large enough to save meaningful compute. Faking a tiny `f` saves a
tiny amount of work, so it's not worth the risk either.

| Attack | Defence |
|---|---|
| Return random weights (lazy) | Replay spot-check, sanity check |
| Do half the steps and interpolate | Replay spot-check, redundancy |
| Copy another worker's answer | Commit–reveal bound to pubkey; units have unique seeds |
| Boost / poison the update | Final must equal last checkpoint; norm bound; replay |
| Rubber-stamp as a verifier | Canary submissions that look like real work and whose ids come from a secret; slashing |
| Sybil (split one big rig into many identities to farm bonuses) | No per-identity bonus exists; pay is per verified FLOP; each identity needs stake; new identities start with lower reputation |
| Predict which segments get checked | Beacon depends on all commitments and the leader's committed round secret, published after the deadline |
| Commit, see the challenge, then withdraw | Forfeits the whole round; unprofitable whenever mining is profitable (see §3 step 5) |
| Hog units and never deliver | Withholding resets capacity to the newcomer cap; reputation drops |
| Unstake right before getting caught | Unstaked coins sit in a 20-round unbonding queue and stay slashable |
| Replay a signed message on another network or later | Chain id and timestamp are inside every signature; stale messages rejected |
| Proposer pays itself extra or skips a slash | System transactions are re-derived by every node from the summary; claimed work is bounded by what the round contained |
| Claim huge speed to hog units | Capacity starts at the newcomer cap and grows at most 2× per round in which the chain shows you delivered everything; missing a deadline drops it back to what you actually delivered (like network congestion control) |
| Coordinator favours a worker, invents a registration or verdict, or rewinds the model | Every choice follows from the round record the block commits to; `chain verify` replays each round and reports any mismatch (§9) |
| Malicious dataset file (pickle exploit) | Datasets load with `allow_pickle=False`, are shape/label-checked, and are verified against the job's sha256 |

---

## 5. Payment: how much and to whom

Each round has a pool:

```
pool = block_subsidy(round) + fee_release(job)
block_subsidy(round) = 50 VIBE / 2^(round // HALVING_INTERVAL)
fee_release          = min(job_escrow_remaining,
                           job_fee_per_round × accepted_units / units_per_round)
```

The requester pays only for the share of the round that verified.

The pool is split:

| Share | Who | Split by |
|---|---|---|
| 80 % | Workers | verified training FLOPs × reputation |
| 15 % | Verifiers | replay FLOPs × reputation |
| 5 %  | Block proposer | — |

Anything unclaimed (for example, no unit verified that round) is burned, not
handed to the proposer, so proposers can't profit from rejecting good work.

**Reputation** is a multiplier in `[0.5, 1.0]`: new workers start at 0.8, gain
+0.02 per round with verified work, lose 0.05 per round with a missed
deadline, and drop to 0.5 on fraud. It moves per *round*, not per unit, so a
laptop builds reputation exactly as fast as a big rig. Reputation lives in
chain state and is derived from round summaries, so every node agrees on it.

**Slashing** takes 50 % of the offender's stake (including coins still
unbonding). Half goes to whoever caught it as a bounty, half is burned. An
offender is slashed at most once per round and forfeits that round's earnings,
including the proposer's share if it proposed that round's block. A slash
takes withdrawable stake first, then unbonding coins, then bootstrap stake
(§6), and the part that comes from bootstrap stake is always burned, never
paid as a bounty, so a validator can't turn it into spendable coins by having
its own second identity catch it.

All amounts are integers in **syn** (1 VIBE = 10^8 syn), and leftovers from
rounding are burned, so supply is exactly
`genesis + Σ subsidies − Σ burns` (checked by the tests).

**Fairness to small rigs.** Pay per verified FLOP is identical for every
machine. A low-end rig earns less in total because it does less work, but its
*rate* (VIBE per GFLOP) is the same as a high-end rig's, and it gets paid every
round it completes work instead of almost never, as in lottery mining.

---

## 6. Ledger

- Account-based state: balances, stakes, unbonding queues, nonces, job escrow,
  job specs, reputation.
- Signed transactions use **Ed25519** (OpenSSL via `cryptography` when
  installed, otherwise a slow pure-Python fallback). User transactions are
  `transfer`, `stake`, `unstake`, `post_job` (escrow a budget; the job spec is
  stored on chain and validated), `cancel_job` (refund what's left of the
  escrow, after the fee the block's own round owes, so cancelling in the
  middle of a round doesn't get that round's work for free) and `validator`
  (announce the address a staked account's validator node is reachable at, or
  leave the set). Only a chain whose genesis names validators can add more, so
  nobody can take over a one-node chain by announcing themselves. Neither
  leaving nor unstaking may leave the chain with no validator that has a vote,
  since then nobody could propose a block. (A slash can't be refused, so it
  still can.)
  Each needs a valid signature, the sender's next nonce, and the right
  **chain id**.
- **Unstaking** moves coins into a 20-round unbonding queue where they can
  still be slashed, so a cheater can't pull their stake out before the
  verdict lands. Coins whose 20 rounds end at a block are released after that
  block's slashes, so they can't slip out in the very block that slashes them.
- **A fair start.** Bitcoin needed no stake: a hash proves itself, so anyone
  with a CPU could mine from block one. Training can only be spot-checked,
  so a stake is what a cheater stands to lose. But if every stake has to be
  handed out in the genesis, the first coins are a premine. Two genesis
  rules (`genesis create` and `testnet` turn both on) let a network start
  with nothing given away:
  - **Earn-in** (`"earn_in": true`). Anyone may register without stake. Pay
    to a worker below the 5 VIBE minimum goes into its stake until it
    reaches the minimum, and only the rest into its balance. So a
    newcomer's bond grows with what it has earned, and a fraud slash or a
    forfeited round costs it that. While earning in, a worker is held to the
    newcomer limit (4 units). It only gets units staked workers didn't need,
    so newcomers can't crowd them out. It never checks others' work, so a
    swarm of free identities can't wave each other's fakes through: their
    work is checked by staked workers, or by the coordinator's own replay.
    At most `max_newcomers` (64) register per round. On the mainnet
    schedule, with about 100 workers sharing 3.125 VIBE a round, earning in
    takes a couple of hours. A faked unit is still unprofitable without a
    stake: the beacon checks 3 of a unit's 8 segments, so faking a share f
    of a unit is caught with probability ≥ f and costs the whole round's
    pay.
  - **Bootstrap stake** (`"bootstrap": {addr: VIBE}`). It gives the first
    validators votes when nobody has earned anything yet. It counts as stake
    for voting and for proposing, but it can never be unstaked, and it's
    burned when its validator leaves the set, so it never becomes spendable
    money. As others earn stake and join as validators, it gets outweighed.

  The genesis hash commits to both. Genesis files without them keep their
  hash, and every worker must be given stake, as before. `vibehash testnet`
  now starts this way: its validators have only bootstrap stake, and its
  workers start with nothing. The demo requester is the one account handed
  coins, to pay for the demo jobs.
- System transactions (`slash`, `fee_release`, `reward`) are never trusted
  from the proposer. Every node re-derives them from the block's round
  summary and the current state, so a proposer can't pay itself extra, skip a
  slash or overcharge a requester. Claimed work is bounded by what the round
  could contain, and only staked addresses can be paid.
- Blocks are hash-linked, timestamped, carry a Merkle root of transactions and
  the hash of the round summary, and are signed by a staked proposer. On a
  chain with validators, the proposer must be one, and the block needs a
  commit certificate to be final (§7).
- The **genesis hash** identifies the network. Nodes refuse a data directory
  from a different genesis, and workers stop if their node switches networks.
- Anyone can run `vibehash chain verify` to download every block and replay it
  from genesis, trusting nothing the node says about balances. It also
  replays every **round** from its record (below), so the summary's contents
  are checked too, not just its arithmetic.
- **Round records.** Each block's summary carries the hash of a record of the
  round: the workers' signed registrations, their self-signed commitments, the
  reveals, the verifiers' signed verdicts, the rule settings, and the results
  of the checks that need the model's weights. The coordinator draws every
  choice from public seeds:
  - which units each worker gets and which are duplicated: the previous block
    and the round's signed registrations;
  - who checks whom and the planted fakes: the beacon and the round secret
    (step 4), which the record reveals once every verdict is in;
  - how much each worker may take: its delivered work on chain;
  - which job runs: the job after the last round's.

  It uses a hash-based generator (`crypto.Draw`) that gives the same draws on
  every machine and library version. Three draws still come from NumPy's
  generators: the challenged segments, each step's batch and a job's starting
  weights. NumPy doesn't promise those stay the same across versions, so
  `tests/test_record_v2.py` pins their outputs, and a NumPy that changed them
  fails the tests rather than splitting the network. Records carry a version:
  version 2 brought the round secret, and a replay follows each record's own
  version, so older rounds replay as they were run. Validators vote only for
  proposals whose record is the current version. `replay.py` re-runs the round from the
  record with the same `Coordinator` code and compares every field of the
  summary. Nodes serve records at `GET /round/<height>/record`.
- **Receipts.** The node answers each accepted registration, commitment,
  reveal and verdict with a signed receipt, its promise that the message will
  be in the round's record. Workers save them to `receipts.jsonl` in their
  cache, check each round once it's sealed, and log `censored: …` if the
  node broke the promise (§9).

---

## 7. Network architecture (testnet)

```
                                     proposals, votes,
 requester --post_job/dataset-->  +-------------+  transactions  +-------------+
                                  | validator A | <------------> | validator B |
 worker --signed HTTP, this round-> (leads this  |                +-------------+
 worker --signed HTTP, this round->   round)    | <------------> | validator C | <-- sync -- follower node
                                  +-------------+                +-------------+
 auditor <--GET /blocks (with commit certificates), /round/<h>/record-- any node
```

- **Node** (`vibehash node`): runs the round state machine on a wall clock
  (register → compute → reveal → verify → finalize), serves a JSON API, stores
  blocks, model weights and datasets on disk, and re-validates the whole chain
  on start-up.
- **Validators** run the chain. The genesis names them (`genesis create
  --validator ADDR=URL`), and a staked account can join later with a
  `validator` transaction (`vibehash validator join --url …`). Voting power
  is stake. For each height, validators take turns leading in a stake-weighted
  order drawn from the height alone (`consensus.proposer_order`), so it's known
  in advance and can't be steered by whoever made the previous block. The
  leader runs the round with the workers, then proposes the block together
  with its round record. Every validator checks the proposal before voting:
  the block must apply to the chain (signatures, balances, derived rewards),
  and its round must replay from the record (`replay.py`). Voting is
  Tendermint's (Buchman, Kwon & Milosevic 2018, Algorithm 1: prevote,
  precommit, locking), in `consensus.py`. A block is **final** once
  validators holding more than two thirds of the stake precommit it. Those
  signed precommits are its **commit certificate**, stored and served with the
  block.
  - **Safety:** two different blocks can't both be final at one height unless
    more than a third of the stake signs both ways. A validator that votes
    both ways leaves two signed, conflicting votes behind, which are kept as
    evidence. Before each vote it sends, a validator saves its round and
    lock, so after a crash it resumes past the rounds it voted in instead of
    voting in them again.
  - **Liveness:** if the leader is offline, slow, or proposes a block that
    fails the checks, the others time out (one and a half round lengths plus
    5 seconds) and the next validator in the order leads. A block more than
    two thirds prevoted for is proposed again rather than trained again.
  - **Idle chains** don't churn rounds. Nobody waits for a proposal until
    there's something to put in a block: a funded job or a waiting
    transaction.
  - **Workers** ask their own node who's leading (`/status` → `leader`) and do
    the round there. A round run again by the next leader is a new attempt,
    and receipts from the old one audit as "abandoned".
- **Peers and sync** (`p2p.py`). Validators find each other from the
  addresses the chain lists; any node can add `--peer` addresses. Votes and
  proposals are pushed to every validator and re-sent until the height is
  decided. Transactions are passed on by every node that accepts them. A node
  that's behind (it was offline, or a vote names a later height) fetches the
  missing blocks, with their certificates and records, from a peer. It trusts
  none of them: each must apply, replay, and carry a valid certificate. On
  start-up, a data directory where a block's certificate is missing or forged
  is refused.
- **Weights and data move between validators.** After a block another
  validator led, a validator fetches the weights its round trained, checked
  against the hash on chain, so it can lead the job's next round. It also
  fetches the datasets of funded jobs from whichever peer the requester
  uploaded them to. Checkpoints stay with the round's leader, and
  `chain verify` fetches them from it to re-run checks.
- **Followers** are nodes with no vote. They sync the chain, serve it, pass
  transactions on, and point workers at the current leader.
- A chain whose genesis names no validators runs the old way: one node seals
  every block. `vibehash testnet --validators 4` runs four validators on one
  computer. Its default is a single validator, which finalises each block the
  moment it votes.
- **Jobs** with budget left take turns, one round each, so a newly posted job
  starts at the next round. After every round the node scores the job's
  model on its held-back validation rows and keeps that history; the poster
  follows it with `vibehash job status` and downloads the trained weights
  with `vibehash job download`. The weights are public: every worker receives
  them each round anyway, so restricting the download would protect nothing.
  `vibehash train` builds the job from a CSV (inputs standardised, a fifth of
  the rows held back for validation) and `vibehash predict` applies the model.
- **Worker** (`vibehash worker`): picks a GPU if it has one (see §8),
  detects its memory, benchmarks the job's real training loop on that device,
  downloads and hash-checks datasets, checks that the starting weights match
  the unit's commitment, and then trains, commits, reveals and verifies.
- Every worker message is a **signed envelope** (chain id + timestamp + body),
  so the node knows exactly who sent it and messages can't be replayed on
  another network or later. Assignments are only served to their owner, so
  nobody can tell which units are replicated.
- Arrays travel as raw little-endian floats at the job's precision (float32
  jobs move half the bytes). Nothing is ever unpickled.
- The simulator (`simulate.py`) drives the **same** `Coordinator` class, so
  economic experiments and the real network can't drift apart.

---

## 8. GPU compute

AI is trained on GPUs, so workers train on them. `compute.py` has two
backends that run **identical maths in identical order**:

- **NumPy** on the CPU: the reference implementation.
- **PyTorch** on `cuda` (NVIDIA, and AMD through ROCm builds), `xpu` (Intel
  Arc and Core Ultra graphics) or `mps` (Apple silicon).

A worker uses the first GPU it finds (`--device` overrides it). It reports
graphics memory instead of RAM, and benchmarks on the GPU, so the scheduler
gives it units in proportion to its real speed. Pay per verified FLOP is
unchanged, so a GPU earns more by doing more, not at a better rate.

**Precision.** Jobs declare `"precision": "float32"` or `"float64"`. GPUs run
float32 jobs; float64 jobs always run on the CPU, because many consumer GPUs
(Intel Arc, Apple) have no float64 at all. Every checkpoint is rounded to the
job's precision before hashing, so a float32 job's Merkle leaves are identical
whether its weights came from a GPU or a CPU.

**Cross-device verification.** Different chips accumulate float32 sums in
different orders, so a replay on another device never matches bit for bit.
A claimed checkpoint passes if it is within the job's `tolerance` of the
replay (around 1e-4 for float32 jobs) **and** within 1% of what the segment
actually changed (`UPDATE_TOLERANCE` in `verification.py`, floored at one
rounding step of the job's precision). Both checks are needed. For big or
nearly trained models, one segment can move the weights less than the
tolerance, and with the tolerance alone a worker could resubmit its start
weights untrained and still pass. Replicas of a unit are compared by the same
rule. GPU workers also disable TF32 and other reduced-precision matmul modes,
which would otherwise blow past the tolerance, and request deterministic
kernels.

**Measured** on an RTX 4070 SUPER (PyTorch 2.14.1, CUDA 12.6) against the
NumPy reference on the CPU, October 2026. `tests/test_gpu.py` re-measures on
whatever devices a machine has.

| Measurement | Result |
|---|---|
| Replay divergence, 69k to 2.2M weights, segments of 5 to 500 steps | 4e-9 to 8e-8: 1,250 to 22,000 times under 1e-4 |
| Divergence as a fraction of the segment's update, learning rates 0.05 to 1e-7 | at most 7.5e-4 (at lr 1e-6), usually 1e-6 to 1e-4 |
| Same GPU, same inputs, run twice | bit-identical |
| One honest 5-step segment, 2.2M weights after 200 steps | moves the weights 4.3e-5, *less than the 1e-4 tolerance* |
| Speed, 2.2M weights, batch 256 | GPU 1.3 to 2.7 ms per step; 28-thread CPU about 20 ms |
| Speed, the 270k-weight `job template --gpu` | GPU about 1 ms per step; CPU 3 to 5 ms |

The fourth row is why the update check exists. Small models gain little from
a GPU because Python's per-step overhead dominates. Intel XPU, Apple MPS and
AMD ROCm haven't been measured yet.

**Arbitration** (re-running a disputed segment or unit) runs on the node's
backend, normally the CPU, with the same checks.

**Model size.** Checkpoints travel as content-addressed binary blobs
(`blobs.py`), never inside JSON. They upload in 8 MB chunks that resume after
a dropped connection, whoever receives one re-hashes it, and they load
memory-mapped. Each uploader's partial upload is kept separately, because
two workers holding a replicated unit produce byte-identical checkpoints and
upload the same blob at the same time; whichever finishes first becomes the
blob, and one uploader can't spoil another's upload. A checkpoint may hold up
to 1 GiB of weights (about 268M float32 parameters). Workers upload a unit's checkpoints right after training
it, before committing, so the reveal is one small message and the node's
speed measurement includes upload time. A verifier downloads about 2q
checkpoints per task (both ends of each challenged segment), which is
significant bandwidth for big models.

---

## 9. What still stands between this and a mainnet

Being explicit about what this is and isn't:

- **One leader per round.** Validators run the chain (§7). No node is in
  charge, and a block is final only once validators holding more than two
  thirds of the stake have replayed its round and signed it. The leader of a
  round can't forge payments (the ledger re-derives them). It can't bend the
  rules either: favouring a worker, inventing a registration, forging a
  verdict, picking checkers, or restarting a job's model all fail the replay,
  so the other validators vote the block down and the next leader takes over.
  What's still trusted:
  1. **Censorship.** A replay can't see messages that were never recorded, so
     the node signs a **receipt** for every registration, commitment, reveal
     and verdict it accepts. The receipt names the round, the round's attempt
     label and the hash of the message as the record will hold it
     (`receipts.py`). Workers keep their receipts and check each round once
     it's sealed. A receipt for a message missing from its round's record is
     signed proof the node dropped it, and `chain verify --receipts FILE`
     checks anyone's. That makes dropping an accepted message (to penalise a
     worker, or to re-draw the beacon) provable. Two gaps remain: a node can
     refuse a message outright, which its sender sees at once but can't prove
     to others, and it can abandon a round and run it again without someone.
     Both are visible to the excluded worker, and neither earns the node
     anything on chain. With leaders taking turns, a censoring validator
     controls only the rounds it leads. Closing the gap fully needs workers'
     messages sent to several validators at once.
  2. **Weight-based checks, within a challenge window.** The record states the
     outcome of the Merkle and sanity checks, the coordinator's own replays,
     arbitration, replica disputes, and the averaged model's hash. Nodes keep
     every checkpoint the last `--keep-rounds` records point to (10 by
     default): the opened checkpoints and finals, not the unopened ones.
     `chain verify` downloads them, plus the dataset, and re-runs every one of
     those checks for rounds in the window, reporting any result the record
     got wrong. That catches, for example, a node slashing an honest worker
     over a sanity check it actually passed, or upholding a false accusation.
     The round's starting model needn't be stored separately: every honest
     reveal's first checkpoint is it, and the record has its hash. The
     averaged model is defined exactly (a float64 sum in round order, then
     rounded) so everyone gets the same bits. Outside the window, results are
     taken from the record. A rechecker whose arithmetic differs by rounding
     gets the same pass/fail results, because every comparison has tolerance;
     only the averaged model's hash needs bit-exact agreement. These are
     **fraud proofs**. Validators replay every record before voting, but they
     don't re-run these weight-based checks: that would mean downloading every
     opened checkpoint, which is the verification work itself. So a leader
     could still misstate one of them if more than two thirds of the stake
     votes without looking, and `chain verify` would only catch it after the
     fact. Next: validators recheck a random sample before voting, and a
     dispute transaction slashes a leader whose record a recheck contradicts.
  3. **Validator accountability.** Conflicting votes are kept as signed
     evidence, but nothing slashes them on chain yet. A validator earns only
     the 5% proposer share of the rounds it leads. Any account with the worker
     minimum stake can join the set, and changes take effect at the next
     height. A mainnet needs double-sign slashing, a higher validator stake,
     and a delay before set changes apply. Bootstrap stake (§6) should be
     spread across several independent first validators, not one operator.
- **Consensus at scale.** Votes go straight to every validator, which suits
  tens of validators, not thousands, and there's no light-client protocol
  yet. Timeouts are sized for testnets. Pure proof-of-useful-work consensus
  (letting training decide whose block wins) is still an open research
  problem. Here, useful work decides who gets paid and stake decides who votes.
- **Evidence availability.** Nodes keep a challenge window of checkpoints
  (above), but only the round's leader holds them, and `chain verify` fetches
  them from it. They should be replicated across validators, so one node
  deleting evidence can't hide a bad verdict.
- **Transport between validators.** Nodes serve HTTPS and rate-limit every
  client (§10), but two gaps are specific to validators:
  - A validator whose certificate is self-signed is only trusted by peers
    that have pinned its fingerprint in their own `config.json`. A mainnet
    should put each validator's certificate fingerprint on chain, next to its
    address.
  - Transactions passed between validators count against the receiving
    node's per-address `tx` budget. Validators off this computer should get
    a bucket of their own.
- **Workloads.** Workers train multi-layer perceptrons on CPU or GPU (§8).
  Transformers and convolutional networks need their own backend kernels,
  plus per-architecture tolerance calibration across GPU vendors.
- **Model size.** Checkpoints are chunked binary blobs of up to 1 GiB (§8),
  and rounds lengthen with a job's size (§3), so big models fit the
  schedule. They're still costly to move: a worker uploads every checkpoint
  of each unit, each verify task downloads about 2q of them, and the node
  keeps every opened one for its challenge window. Next: upload only the
  checkpoints the beacon challenges, and send them at 16-bit precision.
  Beyond what one computer holds, [SHARDING.md](SHARDING.md) splits a model
  into shards trained by a team of workers as a pipeline. Each shard's work
  is checked one cell (one shard over one segment) at a time, so no worker,
  checker or arbitrating node ever runs the whole model. That document has
  the design, and `shards.py` a tested prototype of its core.
- **Security review.** None of this has been audited. Don't put real value
  on it, and get legal advice before issuing a token anyone can buy.

---

## 10. Hardening

Five changes for running a node on a network you don't control. They cover
transport security and transaction spam; §9 keeps only what's specific to
validators. Everything stays compatible with
existing chains: blocks already on disk keep their hashes, genesis files
written before this section keep their genesis hash, and wallet files from
before recovery words load as they are.

### Transaction fees (`fees.py`)

- Every user transaction (`transfer`, `stake`, `unstake`, `post_job`,
  `cancel_job`, `validator`) can carry a `fee` in syn, on top of its amount. It's part of
  what the sender signs. A zero fee is left out of the signed payload and the
  JSON, the way `RoundSummary.record_hash` is, so fee-free transactions, and
  every block made before fees, keep their hashes.
- The sender's spendable balance pays the fee (an `unstake` too, so a wallet
  needs a little unstaked VIBE to unstake). The block's proposer,
  `address_of(block.proposer_pubkey)`, receives it. There's no fee system
  transaction: every node credits a block's fees to its proposer once the
  block's user transactions are applied. `build_block` runs before the block
  is signed, when the proposer isn't known yet, and since nothing it derives
  afterwards (rewards, slashes) depends on balances it needs no change. The
  rule is the same in `build_block`, `apply_block` and `check_tx`, and a
  proposer can't spend a fee in the block that pays it. Fees move coins, so
  supply is unchanged.
- **Relay minimum.** A node only accepts transactions paying at least its
  `--min-fee`, which `/status` advertises as `min_fee`. `vibehash node` and
  `vibehash testnet` default to 0.0001 VIBE: nothing for a person sending
  coins, and it costs 1 VIBE to fill a 10,000-transaction mempool once. The
  library default (`CoordinatorConfig.min_fee`) is 0, so simulations and
  in-process tests run unchanged. `NodeClient.send_tx` pays the node's
  advertised minimum unless told otherwise, so `send`, `stake`, `unstake`,
  `post-job`, `cancel-job`, `train`, `setup` and `testnet` work out of the box;
  `--fee` overrides it.
- **Mempool.** Kept in fee order, highest first, so a block takes the
  best-paying transactions first, but a transaction never moves ahead of one
  it may depend on: its sender's earlier nonce, or a transfer to its sender
  that arrived since that sender's previous transaction. Any order that keeps
  those dependencies applies as cleanly as the order transactions arrived in.
  When the mempool is full, the cheapest transaction that nothing else in it
  depends on is dropped, but only for a newcomer paying more; anything else is
  refused with the fee it would take. There's no replace-by-fee yet: a
  pending transaction's nonce can't be reused.
- Worker protocol messages (register, commit, reveal, verdict) aren't
  transactions and stay free. Stake and the rate limits bound them.
- **Block size.** A block's user transactions may take at most 1 MiB as JSON
  (`ledger.MAX_BLOCK_TX_BYTES`), which leaves room for the round record in the
  4 MiB proposal a leader sends. A block takes the mempool in fee order until
  the next transaction wouldn't fit, so the lowest fees wait for a later block.
- **Relay policy.** On top of what makes a transaction valid, a node relays
  none over 16 KiB, no `transfer`, `stake` or `unstake` memo over 256 bytes
  (it's only a note), and no `post_job` whose budget can't pay for one round.
  Blocks don't enforce these; they keep what a node holds, and so what it
  proposes, to sensible sizes.

### Issuance schedule (`economics.Economics`)

The block subsidy, halving interval and supply cap are fixed in the genesis
file's `"economics"` field (`genesis create --economics testnet|mainnet`):

| | testnet | mainnet |
|---|---|---|
| Subsidy per round | 50 VIBE | 3.125 VIBE |
| Halving interval | 210 blocks | 3,320,000 blocks |
| One halving, at 38 s rounds | ~2.2 hours | ~4.0 years |
| Issued in total | ~21,000 VIBE | ~20,750,000 VIBE |
| Cap on all coins ever (genesis + issued) | none | 21,000,000 VIBE |

The testnet numbers are the old ones, so existing chains and tests are
unchanged. Mainnet is Bitcoin's shape stretched to Vibe Hash's block time.
The node's default round is 38 s (register 3 + compute 20 + reveal 5 +
verify 10), and 4 × 365.25 days / 38 s ≈ 3.32M blocks. 3.125 VIBE per block
then issues 20.75M VIBE, which leaves room for up to 250,000 VIBE of genesis
allocations under the 21M cap. The cap is enforced as well as designed in: a
genesis that allocates more is invalid, and a subsidy never takes coins
created past the cap. Halvings count block heights, and blocks that only
carry transactions (when no job is running) mint nothing, so the 4 years
assume the network is busy.

**Commitment.** A schedule other than the testnet one adds one genesis
transaction, `memo = "economics:" + canonical JSON of the numbers`, so it's
in the genesis block's transaction root and hash. Nodes with different
schedules never share a genesis hash, refuse each other's data directories,
and workers stop if their node changes networks. A genesis file without the
field, or naming the testnet schedule, gets no such transaction and keeps its
old hash. The schedule lives in `State.economics` and is read by settlement,
`Coordinator.finalize` and `/status` (`economics`, `subsidy`).

### TLS (`tls.py`)

- `vibehash node --tls-cert FILE --tls-key FILE` serves HTTPS itself (TLS 1.2
  or later). The handshake runs in the connection's own thread, under the
  header timeout, so a client that never finishes one can't stall the
  server's accept loop.
- `vibehash tls cert` makes a self-signed ECDSA P-256 certificate for a
  testnet node, for the names and addresses workers use, with the extensions
  Python 3.13+'s strict verification needs. It is its own CA, so it also
  works as a CA file. Without `cryptography` it prints the equivalent openssl
  command. `vibehash tls fingerprint FILE` prints a certificate's SHA-256.
- Clients trust a node in one of three ways. They can **pin** the SHA-256 of
  its DER certificate, checked right after the handshake and before any
  request bytes are sent, with no CA needed. They can verify it against a
  **CA file**, with the usual hostname check. Or they use the **system's
  CAs**, as for any website. `setup --tls-fingerprint` / `--tls-ca` save the
  choice in `config.json` against that node's address, and every
  `NodeClient` for that address applies it (`$VIBEHASH_TLS_FINGERPRINT` and
  `$VIBEHASH_TLS_CA` override it). `NodeClient(url, timeout)` is unchanged.
  `fingerprint=` and `ca_file=` are optional keywords, and plain http works
  as before.

### Rate limits and DoS protection (`ratelimit.py`)

- **Token buckets per client IP**, one per kind of request, so a flood of
  one kind can't use up the others' budget:

  | Bucket | Requests | Per second | Burst |
  |---|---|---|---|
  | read | every GET | 50 | 500 |
  | worker | POST register, assignment, commit, reveal, tasks, verdict | 20 | 200 |
  | tx | POST /tx (each is checked against the whole mempool) | 2 | 20 |
  | upload | PUT checkpoint and dataset chunks | 20 | 200 |
  | other | every other POST, such as node-to-node routes | 50 | 500 |

  The check is central: `GuardedHandler.parse_request` runs it once a
  request's headers are in, before any route, keyed by `bucket_of(method,
  first path segment)`. A route added later lands in `read` or `other`
  without being listed, and `ROUTES` can give it a bucket of its own.
- **Over budget** gets `429` with `Retry-After` and a JSON error, before any
  work is done. `NodeClient` waits and resends on 429 and 503 (Retry-After,
  or 0.25 s doubling up to 30 s, with jitter; six tries), and counts 429 as
  retryable.
- **Connections.** At most 256 at once (each is a thread), and 32 from one
  address; past that, a `503` straight from the accept loop. That reply
  needs no thread, and the socket is closed a second later so the client
  reads the reply instead of a reset.
- **Timeouts.** 20 s per read while the request line and headers arrive,
  against slow-loris, then 120 s per operation for bodies and replies. An
  8 MB checkpoint chunk takes a while on a slow home link.
- **Message sizes.** A request body needs a plain decimal `Content-Length`
  (anything else, a negative one included, gets `400`).
  - A worker message may carry only the fields its kind has, at most 4 KiB
    (64 KiB for a reveal). The round record keeps those fields, rebuilt from
    what was parsed, so nobody can pad records.
  - A `/consensus` proposal may be up to 64 MiB on a validator of a chain
    with several validators, since a large round's record is big. At most
    two such bodies are read at once.
  - Votes and proposals for later heights are queued only once their
    signature checks out and their signer is a validator, at most 4 heights
    ahead and 64 messages per sender.
  - `/blocks` pages stop at about 4 MiB.
  - Out-of-range numbers in paths and queries get `400`, not a traceback.
- **Memory.** `vibehash node --max-job-mb N` (64 by default) refuses to
  relay jobs whose checkpoints are larger. Building a job's starting weights
  takes about seven times a checkpoint's size, and a round holds a few
  copies, so 64 MB keeps a 2 GB server well clear.
  - `/job/<id>` doesn't build the weights of a job that hasn't trained yet:
    it reports their size, and `/job/<id>/model` answers 404 until there's a
    trained model.
  - A dataset's arrays are checked from their headers before loading, and
    one that would unpack to more than 64 times its file size (or 64 MB, if
    larger) is refused. Only the upload that completes a dataset checks it,
    and a check never deletes one already accepted.
- **Peers.** Sync gives up on a peer whose page adds nothing or whose block
  it refuses, and moves on past a malformed reply. Clients cap what they read:
  128 MiB of JSON, one chunk of binary, downloads at the size the job allows.
  A follower only takes a leader hint that's a validator's address on chain.
- **This computer is never limited by default.** Tests and local testnets run
  a node, up to 8 workers and several validators all from 127.0.0.1. Behind
  a reverse proxy every client looks local, so limit at the proxy, or pass
  `--rate-limit-local`. `--rate-limit off|tx=1/10,read=100,...` changes or
  disables the buckets, and `--max-connections` the overall cap.

### Recovery words (`mnemonic.py`)

- `wallet create` and `setup` derive the wallet key from **24 BIP39 words**
  (256 bits of entropy and an 8-bit checksum) and show them once, numbered,
  with a warning that they restore the wallet and that anyone who sees them
  can take its coins. `wallet restore` rebuilds the file from the words, read
  from the terminal without echo or from `$VIBEHASH_MNEMONIC`. It takes an
  optional BIP39 passphrase (`--passphrase`); a different passphrase is a
  different wallet. The first four letters of each word are enough.
- Derivation follows the standards hardware wallets use: BIP39 seed
  (PBKDF2-HMAC-SHA512, 2048 rounds, salt `"mnemonic" + passphrase`), then
  SLIP-0010 Ed25519 along **`m/44'/1'/0'/0'/0'`**. The path has BIP44's
  shape (purpose / coin type / account / change / index), as Aptos and Sui
  use for Ed25519, with every step hardened because SLIP-0010 defines only
  hardened Ed25519 derivation. Coin type 1 is SLIP-0044's "testnet, all
  coins". A mainnet should register its own coin type and use it, so the
  same words give different testnet and mainnet keys.
- The word list is the official `bip-0039/english.txt`, shipped as package
  data and checked against its published SHA-256 when loaded. The tests run
  the Trezor reference vectors and SLIP-0010's Ed25519 vectors.
- **Wallet format.** Unchanged, plus an optional `"hd": {"scheme":
  "bip39+slip10-ed25519", "path": ...}`. The derived key is stored as before,
  encrypted or not, and the words are never written anywhere. Older files
  have no `hd` and load unchanged; `wallet show` says whether a wallet has
  recovery words. `vibehash testnet`'s throwaway wallets are still plain
  random keys.

### Block versions (`ledger.VERSION`)

Fixing a ledger rule changes which blocks are valid, but a node re-applies
every block on disk from genesis when it starts, and `chain verify` does the
same. So blocks carry a version, and a rule applies only to blocks of the
version that brought it and later. Version 1 blocks (everything 0.5.1 and
earlier made) hash, serialise and settle exactly as before; from version 2
the proposer signs the version, and it's in the block's JSON. A block's
version can't be lower than its parent's or higher than the node knows, and
validators only vote for blocks of the current version, so once a chain has
a version 2 block it stays on version 2. Every validator must run a release
that knows a version before any block of it appears: older nodes refuse it.

Version 2 brought the block size limit above, and these fixes:
- a transaction may appear only once in a block (`crypto.merkle_root` pads an
  odd level by repeating its last node, so a block with its last transaction
  repeated had the same hash and signature as the real one);
- bootstrap stake is slashed last and only ever burned (§5);
- coins finishing unbonding are released after the block's slashes (§6);
- a `cancel_job` leaves the fee its job owes the block's own round (§6);
- a proposer slashed in its own round isn't paid for proposing it (§5);
- the last validator with a vote can't leave or unstake below the minimum
  (§6);
- the pool is split in exact fractions. Floats could round a payout up by a
  syn, which a round with no verifiers then didn't catch.

### Still open

- Replace-by-fee and fee estimation.
- Transaction signatures name the chain id but not the genesis hash, so a
  transaction could be replayed on a new network that reuses a chain id.
- `check_tx` re-verifies every pending transaction for each new one, so
  admission is O(mempool). The `tx` bucket bounds that per address, but not
  across many addresses.
- Per-IP limits don't stop an attacker with many addresses, and the node
  doesn't trust `X-Forwarded-For`, so behind a proxy, limit at the proxy.
- Headers are bounded per read, not in total: a client trickling one byte
  every 19 s holds a connection, up to the per-address cap.
