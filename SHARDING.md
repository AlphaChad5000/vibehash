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
- **Mixture-of-experts.** Routing flips under rounding noise, and between
  experts held by different workers the traffic is all-to-all at every layer.
  §7 has the design and a tested prototype: committed routing with a tie
  margin, and each expert a cell of its own. Placing experts in shards and
  the all-to-all are still open (§7, "What wave 2 must do").
- **Determinism.** The prototype's MLP is bit-exact across shard counts on
  one device. Transformer kernels in bf16, attention above all, need
  deterministic modes and measured tolerances, as DESIGN §8 did for MLPs.

## 7. Mixture-of-experts (prototype: moe.py)

**Status: a tested prototype of the block and its checks.**
`vibehash/moe.py` is a mixture-of-experts (MoE) feed-forward block in NumPy,
meant to replace a transformer's MLP, together with the checks below.
`tests/test_moe.py` shows they hold. Nothing uses it yet.

**The block.** For each token (a row of [tokens, d_model] activations):
- a linear router scores the E experts with a softmax;
- the token goes to its top k experts, with gates: their scores normalised
  over the k;
- the output is the gate-weighted sum of those experts' SwiGLU outputs,
  plus the outputs of any shared experts, which every token goes through
  (DeepSeek's design).

Each expert's arrays are its own (`expert.3.gate`, `.up`, `.down`, plus
LoRA adapters if the job has them) and pack into one flat vector (`pack`).
So a worker or a checker can hold any subset of experts.

Backward gives gradients for the experts, the shared experts, the router and
the input. The router's gradient comes through the gates. The block has no
auxiliary balancing loss, which would tie each token's gradient to every
other token's routing. Bias-based balancing can come later and keeps a step
a pure function: DeepSeek-V3 adds a per-expert bias to the scores for
choosing (never for the gates) and adjusts it between steps by each expert's
load. The bias would be committed like the weights.

**The routing problem.** Top-k is a jump. When two experts' scores are within
rounding of each other, a worker's GPU and a checker's can pick different
ones. That token's output is then a different expert's, so an honest worker
fails a naive replay and a checker can't tell it from a cheat. The tests make
32 near-tie tokens among 48:
- a float32 worker and a float64 checker route 9 of them differently, and
  the checker's own forward pass is 15% off the worker's output, 1,500 times
  a float32 job's tolerance;
- nudging the input by 1e-9, as recomputing the layers below on other
  hardware would, flips 17 of them, and the output is 21% off.

**Committed routing.** Each worker commits to its routing (which experts each
token went to, and their gates) along with its edge tensors. A checker:
1. checks the routing (`check_routing`): for every token, no expert left out
   may beat a chosen one by more than a margin, measured in the router's
   logits; and the gates must be the chosen scores normalised, within the
   same margin;
2. replays with the committed choices (`check_block`). It computes the gates
   from its own scores, because they're continuous and the router's gradient
   flows through them.

With the routing fixed, the rest of the block is continuous, and the job's
tolerance absorbs rounding as before. The replays above land 1.6e-7 and
1.6e-9 from the worker's output, and the honest routing passes its check;
with no margin, it would fail.

**The tie margin.** Two choices matter:
- *Compare against every expert left out, not just the k-th best.* "Each
  chosen expert within the margin of the k-th best" is weaker: it would let a
  worker drop the top expert for one tied with the k-th (tested).
- *Relative, in logits.* Logits are log-scores up to a per-token constant,
  so a margin there is a relative margin on the scores, and honest rounding
  is relative. An absolute margin on probabilities would let a confident
  router's lower slots go to any expert at all, since every runner-up's score
  is near zero.

The default, `ROUTING_MARGIN`, is 1e-4: the size of a float32 job's
tolerance.
- **Headroom.** Honest float32 and float64 routers measured at most 4e-6
  apart, which gives 25 times headroom. That covers widths from 16 to 7168
  and 8 to 256 experts, with logits ~N(0, 1) as an RMS-normalised input
  gives.
- **What a cheat gains.** On tokens with such a near-tie (0.005% to 0.16% of
  them, for 8 to 256 experts), it can send the token to an expert the router
  rates as good to within 0.01%. That saves no work: every token still runs k
  experts.
- **Precision.** The router has to compute in float32 or better even where
  the experts run in bfloat16, whose logits are only good to about 1e-2.
  Like tolerances (DESIGN §8), the margin should be measured across GPU
  vendors, and a job can set its own.

**Expert-level cells.** With the routing fixed, an expert's outputs are a
pure function of its weights and the tokens routed to it, so each expert is a
cell, as each shard is (§2). `MoETrace` extends `Trace`'s commitment. Its
leaves are, in order:
1. the block's four edge tensors, hashed as a shard's are, so neighbours'
   commitments must agree (`edges_agree`);
2. the router's and shared experts' weights and gradient;
3. the routing;
4. per expert: its weights, the tokens it was sent (`x`), the gradient that
   came back for them (`dy`), what it returned each way (`y`, `dx`) and its
   weights' gradient.

Between them, two kinds of cell check every leaf:

| Cell | The checker holds | It checks |
|---|---|---|
| Expert (`check_expert`) | One expert's weights, and the rows of the tokens routed to it | Its outputs, the gradient it returned and its weights' gradient, replayed from its committed inputs |
| Router (`check_router`) | The router and shared experts, and what went to and from each expert (the all-to-all's traffic) | The routing; that each expert was sent exactly its tokens (a gather, so bit for bit); and the combination: the block's output, the gradients handed to the experts and sent back, and the router's gradient |

As with shards, a lie fails only the cell it's in (tested):
- a faked expert, zeroed as if it never ran or noisy, and combined as if real,
  fails only its own cell;
- a bad routing or a wrong combination fails only the router's cell;
- an expert run honestly on a token it wasn't routed fails only the router's
  cell.

**What it costs.** An expert's checker never holds another expert. In a
DeepSeek-V3-shaped layer (7168 wide, 256 experts of width 2048, top 8), that's
44M weights (22 MB at 4 bits) instead of the layer's 11.3B. Its tensors are
the tokens routed to it, k/E of them on average.

Cells differ in cost, since experts get different numbers of tokens. So the
beacon should pick cells in proportion to their FLOPs, and pay should follow
them. Otherwise a worker could fake the busiest experts: few cells, much of
the work.

**What the tests show** (`tests/test_moe.py`, under a second):
- Backward matches finite differences in float64 for the experts, the
  router, the shared expert, the input and LoRA adapters.
- Routing flips are real, as above, and committed routing fixes them.
- `check_routing` rejects these routings: every token sent to the same two
  experts, an expert skipped, malformed routings, and wrong gates.
- A replay rejects a correct routing with wrong expert outputs, or with
  wrong gradients.
- Each lie fails only its own cell.
- The same inputs give the same bits in float32 and float64, and exact ties
  go to the lower index.
- An expert's cell holds that expert's weights alone, and the router's cell
  holds no routed expert's.

**What wave 2 must do.**
- **Put it in the transformer.**
  - Swap `transformer.py`'s SwiGLU MLP for this block, reshaping
    [batch, seq, d_model] to [tokens, d_model] and back, with the same
    RMSNorm before it, so the router's logits stay near unit scale as the
    margin assumes.
  - Share one SwiGLU between the two files.
  - Add each MoE layer's routing to a step's commitment.
  - With LoRA, adapt each expert's matrices and train the router in full;
    it's d_model × E, 1.8M weights for DeepSeek-V3's shape.
- **Place experts in shards.** Where a layer's experts fit one worker, a
  shard holds whole layers as now, and there's no all-to-all. At 4 bits, the
  reportedly ~552B-weight target is about 280 GB: about 14 workers holding
  20 GB each, if a layer fits in one. Experts are then cells only for
  checking:
  - per step, a shard's `Trace` adds each MoE layer's routing and expert
    leaves;
  - per checkpoint, each expert's weights get a leaf of their own, so a
    challenged expert opens alone;
  - a cell becomes one expert over one segment, and the worker keeps its
    tensors to open if challenged.
- **Handle all-to-all traffic where a layer doesn't fit one worker.** Expert
  parallelism spreads a layer's experts over workers. Each token's
  activations then go to up to k of them and back, forward and backward:
  4·k·d_model values per token per MoE layer, against 2·d_model at a
  pipeline edge. At k = 8 that's 16 times a pipeline edge, per layer, and
  over home links it's the bottleneck. Ways to cut it:
  - device-limited routing, as DeepSeek does: each token's experts come from
    at most M workers;
  - placing experts often chosen together on one worker;
  - 8-bit dispatch.

  Both ends of each exchange commit to the same hashes of `x`, `dy`, `y` and
  `dx`. These are compared like `edges_agree`, with signed messages for
  blame (§3). The `experts` argument of `moe.forward` is where the exchange
  plugs in. A slow expert holder stalls every worker that routes tokens to
  it, so the stragglers problem (§6) gets worse.
- **Calibrate.** Measure the margin, with the tolerances, on real routers
  across GPU vendors; keep routers in float32; and add bias balancing, as
  committed state, if load balance needs it.

Background: Shazeer et al., *Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer* (ICLR 2017); Lepikhin et al.,
*GShard* (ICLR 2021) and Fedus, Zoph and Shazeer, *Switch Transformers*
(JMLR 2022): top-k routing and expert parallelism; DeepSeek-AI,
*DeepSeek-V3 Technical Report* (2024): shared experts, auxiliary-loss-free
balancing by bias, and node-limited routing.

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
