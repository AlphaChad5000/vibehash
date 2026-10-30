# Sharded units: models bigger than one computer

**Status: design, with a working prototype of the core.** `vibehash/shards.py`
splits a model into shards, trains a unit as a pipeline across them, commits
to every shard's work, and checks one cell at a time. `tests/test_shards.py`
shows the properties below hold. The protocol doesn't use any of it yet: the
coordinator, the workers' transport and the job format still assume one
worker holds the whole model (§5 lists what has to change).

---

## 1. The problem

Today a work unit is the whole model trained for a few dozen steps by one
worker, and a checker replays segments of it. So the worker, every checker,
and the node arbitrating a dispute all hold the whole model, which caps a job
at what one consumer machine can hold (checkpoints of up to 1 GiB, about 268M
float32 weights) and makes the node's arbitration as heavy as the job.

For a model of tens or hundreds of billions of weights, nobody on the network
can hold all of it. A worker can hold a slice, and a checker must be able to
check a slice.

## 2. The idea: check cells, not units

**Shards.** Split the model into shards: contiguous blocks of layers, sized
so one shard fits a worker's memory. A unit is trained by a **team**, one
worker per shard, as a pipeline. Each step:
- activations flow forward from shard to shard;
- gradients flow back;
- each shard updates its own weights.

This is the same arithmetic one computer would do, cut at layer boundaries.
The prototype does the same operations in the same order, so a sharded unit
ends at **exactly the same bits** as the whole-model reference (tested for
float32 and float64, 1 to 5 shards). Sharded units therefore average, replay
and pay exactly like whole ones.

**Cells.** Every worker commits, before the beacon, to a Merkle root over:
- its shard's weights every `checkpoint_every` steps;
- every tensor that crossed its edges, at every step:
  - `h_in`, the activations the shard before it sent;
  - `g_out`, the gradient the shard after it sent back;
  - `h_out` and `g_in`, what it sent each way.

A unit's work is then a grid:

```
            segment 0    segment 1    segment 2  ...
 shard 0   [ cell ]  -> [ cell ]  -> [ cell ]       reads the dataset
              |  ^         |  ^         |  ^
         h_out|  |g_in     |  |         |  |        (each step's tensors at the edge)
              v  |         v  |         v  |
 shard 1   [ cell ]  -> [ cell ]  -> [ cell ]
              |  ^         |  ^         |  ^
              v  |         v  |         v  |
 shard 2   [ cell ]  -> [ cell ]  -> [ cell ]       reads the labels, computes the loss
```

A cell is one shard over one segment. Its outputs are a pure function of four
inputs:
- the shard's weights at the segment's start;
- the tensors that came in at its edges during those steps;
- the dataset batch, for the first shard;
- the labels, for the last shard.

Its outputs are the tensors it sent at each step and its weights at the
segment's end.

**Checking a cell needs that one shard and nothing else.** The checker
(`shards.check_cell`) does the following:
1. Takes the cell's two checkpoints and its edge tensors, and verifies each
   is what the worker committed to. Each Merkle proof must also match its
   leaf's *position*, so a valid leaf can't be offered in another slot.
2. Replays the segment with that shard's layers alone.
3. Compares each step's outgoing activations and gradients, within the job's
   tolerance.
4. Compares the end weights with the same rule a whole-model replay uses (the
   job's tolerance, and within 1% of the segment's actual update).

**Neighbours must agree on what crossed between them.** Shard k's `h_out` at
step t must hash to the same value as shard k+1's `h_in`. Shard k+1's `g_in`
must hash to the same value as shard k's `g_out`. `shards.edges_agree`
compares the committed hashes: no computation, just Merkle openings. So a
cell's inputs are what its neighbours really produced.

## 3. Why checking cells one at a time is enough

- **Lies stay local.** A worker can only misreport its own cell. Each cell is
  checked against the inputs it committed to, and those must match its
  neighbours' commitments. A shard that received garbage computed honestly on
  it, so its cells pass and the liar's fail. The prototype's tests cover
  three cheats, and each fails only the cheater's own cells:
  - noisy activations sent forward;
  - zeroed gradients sent back;
  - weights left untrained.

  A worker that sends one tensor but commits to another is caught at the
  edge instead.
- **Same odds as today.** The beacon picks `q` segments per shard per unit.
  A worker faking a fraction `f` of its cells is caught with probability
  `1 − (1 − f)^q ≥ f`, the same arithmetic as DESIGN §3–4, so the stake and
  slashing economics carry over unchanged.
- **Floating-point noise doesn't add up across shards.** Every cell is
  replayed from the exact bits its neighbours sent, not from a recomputation
  of them. So the tolerance applies per cell, however many shards the model
  has, and GPUs of different vendors can sit in one team.
- **Blame at an edge is provable.** Each edge message is signed by its sender
  over (round, unit, step, micro-batch, tensor hash), and the receiver keeps
  it. If two neighbours' commitments disagree, the signed message shows which
  of them committed to something other than what was sent. It works the way
  receipts do for messages to the node (DESIGN §9).
- **What a cheater gains is what a checker can check.** Either it lies in
  what it computed (its cell fails) or in what it says it sent (the edge
  fails). The only other option is not to send at all: that stalls the team
  and shows up as a missed deadline, pinned on whoever stopped signing edge
  messages.

## 4. What it costs

**A checker holds one shard,** that is 1/S of the model. Its compute for a
cell is the cell's own work. Its download is:
- the cell's two checkpoints (only the adapters, with LoRA);
- the tensors that crossed the cell's edges, which is exactly the traffic its
  worker sent and received during the segment.

With `q` of `n` segments checked, checking costs `q/n` of training in compute
and in edge traffic, as it does today. **A dispute is one cell too,** so a
validator with a few GB of memory can arbitrate a job far bigger than itself.

**Pipeline speed on home connections.** In a transformer, the tensor at an
edge is one vector of the model's width per token. With LoRA on a frozen
base, a shard of P weights does about 4P FLOPs a token: forward, and backward
to its input, with negligible adapter gradients. Each shard uploads two edge
vectors a token: activations forward and a gradient back. At 20 TFLOP/s
sustained (a mid-range gaming GPU), with 4-bit base weights:

| Model | Shard | Shard weights (4-bit) | Compute per token | Upload per token, 8-bit edges | at 20 Mbit/s | at 100 Mbit/s |
|---|---|---|---|---|---|---|
| 70B-class, width 8192, 8 shards | 8.8B | 4.1 GiB | 1.8 ms | 16 KiB | 6.6 ms | 1.3 ms |
| 405B-class, width 16384, 16 shards | 25B | 11.8 GiB | 5.1 ms | 32 KiB | 13.1 ms | 2.6 ms |
| 1T dense (hypothetical, width 25600), 25 shards | 40B | 18.6 GiB | 8.0 ms | 50 KiB | 20.5 ms | 4.1 ms |

On 100 Mbit/s upload every row is compute-bound. On 20 Mbit/s the network is
the limit, at 2.5 to 4 times the compute time. 16-bit edges double the upload
column. Compute per token grows with width², while edge traffic grows with
width, so bigger models get *more* compute per byte sent: the square-cube
argument from SWARM parallelism. A team runs at the pace of its slowest link,
so the coordinator should team workers by measured upload, as capacity
already follows delivered work.

A 1T team (25 GPUs of 24 GB) at 8 ms a token trains about 125 tokens a
second. Many teams run side by side, data-parallel, and their units are
averaged each round exactly as single workers' are today. With 10 teams,
100M tokens takes about a day.

## 5. Fitting it into the protocol

| Piece | Change |
|---|---|
| Job spec | The model's architecture (transformer blocks, width), the shard cuts, edge precision. For LoRA: the frozen base's sha256 (fetched once, as datasets are) and the adapter rank and targets. |
| Model code | Transformer stage functions (`forward(shard, h_in)`, `backward(shard, h_in, g_out)`) on PyTorch, with deterministic kernels, and tolerance calibration per stage across GPU vendors. The prototype's MLP stages are the pattern. |
| Scheduling | The coordinator forms teams: each shard goes to a worker with the memory for it, and the slowest link sets a team's pace. Units are assigned to teams, and capacity tracks teams' delivered work. |
| Commit | Each team member commits to its trace root (`Trace.root`). |
| Beacon and challenges | The record v2 beacon picks `q` segments per shard per unit, including the leader's secret. |
| Reveal | Open the challenged cells' checkpoints and edge tensors. Neighbours open their matching edge hashes. Edge tensors upload as blobs only when challenged: they're large, and this is DESIGN §9's "upload only the checkpoints the beacon challenges". |
| Verify | Verify tasks are cells. Canaries are the record v2 construction applied to a cell: a real cell's openings with one step's update rescaled. |
| Settle | Each worker is paid per verified cell's FLOPs. A team's update is averaged in only if every challenged cell of every shard passed. A cheating member is slashed, and honest members are paid for their verified cells, so a cheat can't cost them their pay. |
| Aggregate | Averaging is per weight, so averaging each shard across teams is averaging the model. The node streams shard by shard and never holds the whole model in memory. Shards could be stored and served by their workers rather than the node, BitTorrent-style, with the node keeping only hashes. |
| Arbitrate and recheck | One cell, so one shard's memory (§4). |
| Evaluate | A team runs a forward pass on the validation rows, as forward-only cells, checked the same way. |
| Transport | Team members send edge tensors to each other directly, signed (§3), on a 1F1B pipeline schedule with micro-batches in flight so link latency is hidden. |

## 6. Open problems

- **Reaching each other.** Home workers sit behind NAT. Teams need hole
  punching (libp2p, as Petals and hivemind use) or paid relays. The node
  can't relay: edge traffic is far more than a small server's bandwidth.
- **Stragglers and dropouts.** One slow or vanished member stalls its whole
  team. SWARM parallelism's answer is several workers per shard, with
  micro-batches routed to whichever is free. Cells still work, because each
  micro-batch's edge messages name the worker that handled it.
- **Big steps.** A step of hundreds of thousands of tokens makes a cell's
  edge tensors gigabytes. Checks can sample micro-batches within a step if
  workers also commit to each micro-batch's weight gradient. That's cheap
  with LoRA, where those gradients are adapter-sized.
- **Optimiser state.** Adam's moments are part of the state a cell carries,
  so checkpoints would hold them too, about three times the weights. With
  LoRA that's still small. Otherwise use SGD or 8-bit state.
- **Mixture-of-experts.** Experts can be cells of their own (an expert's
  feed-forward block is a pure function of the tokens routed to it), but the
  routing makes the traffic all-to-all at every layer. That needs its own
  design.
- **Determinism.** The prototype's MLP is bit-exact across shard counts on
  one device. Transformer kernels in bf16, attention above all, need
  deterministic modes and measured tolerances, as DESIGN §8 did for MLPs.

## References

- Huang et al., *GPipe* (NeurIPS 2019), and Narayanan et al., *PipeDream*
  (SOSP 2019): pipeline-parallel training and the 1F1B schedule.
- Ryabinin, Dettmers, Diskin and Borzunov, *SWARM Parallelism: Training Large
  Models Can Be Surprisingly Communication-Efficient* (ICML 2023): pipelines
  over unreliable, slow networks, and the square-cube law.
- Borzunov et al., *Petals: Collaborative Inference and Fine-tuning of Large
  Models* (2023): 100B+ models split across consumer GPUs over the internet.
- Jia et al., *Proof-of-Learning* (IEEE S&P 2021): verifying training by
  replaying committed segments, the idea Vibe Hash's segments and these
  cells extend from time into layers.
- Douillard et al., *DiLoCo* (2023): local training with infrequent
  averaging, which teams-as-units inherit.
