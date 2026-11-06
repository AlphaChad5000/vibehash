# Evo's margin across GPU kinds: the measurement

**Question.** PR #43 set Evo's logit margin to 0.5 from one pair of GPU kinds: an RTX 4070 SUPER checking answers
made on RTX 4090s. Does 0.5 also accept every honest member, and still reject every skipped-work cheat, when the
worker and the checker are other kinds of card? Turing (RTX 2060, `sm_75`) matters most: it has no native bf16,
so PyTorch emulates the arithmetic the Evo engine runs in.

**Tool.** `scripts/evo_margin_xgpu.py` runs the network's own code, unchanged:
- **answer** uses the worker's path (`es_client.runtime2`, `round_factors`, `es_fast.answer_members`);
- **check** uses the checker's path (`es_fast.gaps_items`, one member at a time, as `es_client.Es2Work.check`).

So what it reports is what a checker on that card would decide.

## The machines

| Machine | Card | Architecture | Who |
| --- | --- | --- | --- |
| A | RTX 4060, 8 GB | Ada, `sm_89` | Caleb |
| B | RTX 2060, 6 GB | Turing, `sm_75` | Caleb |
| C | RTX 3090 or RTX A5000, 24 GB | Ampere, `sm_86` | RunPod |
| D | RTX 4070 SUPER (and a 4090 if one is free) | Ada, `sm_89` | Cole |

Every machine answers once and checks every machine's answers, its own included: 4 dumps × 4 checkers = 16 results.
A machine checking its own answers is the baseline, and should show a worst gap of exactly 0.

## The fixed round

The same on every machine, so the answers are comparable:

- model `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` (pinned; `check` refuses a dump from another commit);
- the live job's 16-task mix (`--tasks vibe`, the default), 16 problems, answers up to 512 tokens;
- 16 antithetic pairs (32 members) for each of two round seeds: `5eed0e7a1c3b2d4f 0ddba11c0ffee123`;
- the base weights with no training applied. The live job has drifted a few percent from them: that moves where
  the logits sit, not how two GPUs round them.

## Steps on each machine

1. **Set up** (once). On Windows, from the repository:
   ```
   git pull
   pip install -e .[evo]
   vibehash install-gpu
   ```
   On a RunPod pod, start from a PyTorch template with CUDA 12 or later, then:
   ```
   git clone https://github.com/VibeHash/vibehash && cd vibehash && pip install -e .[evo]
   ```
   Don't install `causal_conv1d` or `flash-linear-attention` for this, unless every machine does: transformers
   uses them for Qwen3.5's linear-attention layers when present, and the script records which a machine had.

2. **Answer** (a few minutes; the model downloads once, about 1.7 GB):
   ```
   python scripts/evo_margin_xgpu.py answer --model Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17 --seed 5eed0e7a1c3b2d4f 0ddba11c0ffee123 --out answers-<card>.json.gz
   ```
   Machine B with two cards in one PC: add `--device cuda:1` for the second card (`nvidia-smi` lists the order).

3. **Share the dump.** It is well under 1 MB. Put it wherever the others can fetch it (the team chat is fine).

4. **Check every dump**, its own included (on an RTX 4060 about 8 minutes a seed for short answers; allow 20 to 30
   minutes a dump for the vibe mix's two seeds):
   ```
   python scripts/evo_margin_xgpu.py check answers-<other card>.json.gz --out check-<other card>-by-<card>.json
   ```

5. **Send back the `check-*.json` files.** Anyone then runs:
   ```
   python scripts/evo_margin_xgpu.py report check-*.json
   ```

## What the report says

One row per pair (answered on → checked on):
- **honest gaps.** The worst gap below the greedy logit over all honest members, and its 99th percentile, per
  member and per answer.
- **false rejections.** At margins 0.1, 0.375, 0.5, 0.75 and 1.0, how many honest members fail.
- **false acceptances.** At the same margins, how many twin-swap cheats pass. A twin-swap cheat is a member that
  commits its twin's answers instead of doing its own work: the cheat that pays.
- **identical twins.** Pairs whose answers match on every problem: no margin can tell them apart, and nobody
  gains from them, so they are counted separately.

It ends with the range of margins that separates every pair measured, or says that none does.

Measured so far: on the RTX 4060 checking its own answers (countdown, 32 members, 512 tokens), the worst honest
gap was 0. The smallest cheat gap was 2.25, so all 32 cheats were caught at every margin.

## Deciding

- **Keep 0.5** if no pair's worst honest gap exceeds about 0.4 (so honest rounding stays below the margin with room
  to spare) and no cheat sits below 1.0.
- **If Turing (or any card) breaks 0.5**, the options, in order of preference:
  1. pair checkers by hardware class, so a card mostly checks its own kind. This is
     `docs/ARCHITECTURE-NEXT.md` §3; the census is already on registration;
  2. raise the margin, if the smallest cheat gap leaves room (stay at most half of it);
  3. keep that card kind from checking Evo jobs, while it can still answer.
- **The cost of a wider margin** isn't in this measurement: PR #43 measured that 26.9% of answer-key letter swaps
  fall within 0.5. They don't pay, since pay follows accepted FLOPs and not rewards, but a wider margin lets more of
  them through.

## Time and cost

On an RTX 4060, one seed takes 30 s to answer and about 8 minutes to check, for countdown's short answers. The
vibe mix answers at greater length, so allow two to three times that. The RunPod machine answers once and checks
four dumps: about an hour, well under $15 on a 3090 or A5000. Stop the pod when it's done.
