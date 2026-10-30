"""Run a Vibe Hash network: honest low/mid-end rigs plus a handful of cheaters.

    python simulate.py --rounds 20
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from vibehash import economics as econ
from vibehash import crypto
from vibehash.crypto import KeyPair
from vibehash.ledger import BURN, Chain
from vibehash.coordinator import CoordinatorConfig
from vibehash.protocol import Protocol
from vibehash.tasks import SYN_PER_VIBE, TrainingJob, Verdict
from vibehash.verification import detection_probability
from vibehash.worker import (
    GTX_1060,
    LOW_END_LAPTOP,
    MINI_PC,
    NETBOOK,
    OFFICE_DESKTOP,
    RTX_3060,
    Worker,
)

ROSTER = [
    # name,              hardware,        behaviour,  verifier behaviour
    ("ana-laptop",       LOW_END_LAPTOP,  "honest",   "honest"),
    ("ben-desktop",      OFFICE_DESKTOP,  "honest",   "honest"),
    ("cho-minipc",       MINI_PC,         "honest",   "honest"),
    ("dev-gtx1060",      GTX_1060,        "honest",   "honest"),
    ("eli-rtx3060",      RTX_3060,        "honest",   "honest"),
    ("fay-laptop",       LOW_END_LAPTOP,  "honest",   "honest"),
    ("gus-netbook",      NETBOOK,         "honest",   "honest"),   # too little RAM for this job
    ("lazy-gtx1060",     GTX_1060,        "lazy",     "honest"),
    ("shortcut-desktop", OFFICE_DESKTOP,  "shortcut", "honest"),
    ("copycat-minipc",   MINI_PC,         "copycat",  "honest"),
    ("poison-laptop",    LOW_END_LAPTOP,  "poisoner", "honest"),
    ("rubberstamp-pc",   OFFICE_DESKTOP,  "honest",   "lazy"),
    ("accuser-gtx1060",  GTX_1060,        "honest",   "false_accuser"),
    ("withhold-minipc",  MINI_PC,         "withholder", "honest"),
]


def vibe(syn: int) -> str:
    return f"{syn / SYN_PER_VIBE:,.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    workers = [Worker(n, KeyPair.generate(n.encode()), hw, b, vb) for n, hw, b, vb in ROSTER]
    requester = KeyPair.generate(b"acme-labs")
    stake = 10 * SYN_PER_VIBE
    chain = Chain(
        allocations={requester.address: 1_000 * SYN_PER_VIBE, **{w.address: 1 * SYN_PER_VIBE for w in workers}},
        stakes={w.address: stake for w in workers},
    )
    proto = Protocol(chain, workers, CoordinatorConfig(), seed=args.seed)
    names = {w.address: w.name for w in workers} | {requester.address: "acme-labs (requester)"}

    job = TrainingJob("acme-classifier-v1", requester.address, budget=300 * SYN_PER_VIBE, fee_per_round=15 * SYN_PER_VIBE, units_per_round=40)
    proto.post_job(requester, job)

    print(f"Vibe Hash simulation - {len(workers)} workers, job '{job.job_id}' "
          f"({job.spec.n_params} params, {job.units_per_round} units/round, {job.steps_per_unit} SGD steps/unit)\n")
    print(f"{'rnd':>3} {'units':>5} {'fraud':>5} {'miss':>4} {'val loss':>8} {'val acc':>7} {'pool VIBE':>9} {'burned':>7}  proposer")
    for _ in range(args.rounds):
        r = proto.run_round(job.job_id)
        n_ok = len({x.unit.unit_id for x in r.results if x.verdict == Verdict.ACCEPTED and x.worker in r.worker_flops})
        n_fraud = sum(x.verdict.is_fraud for x in r.results)
        n_miss = sum(x.verdict == Verdict.MISSED_DEADLINE for x in r.results)
        print(f"{r.round:>3} {n_ok:>2}/{job.units_per_round:<2} {n_fraud:>5} {n_miss:>4} {r.val_loss:>8.4f} {r.val_acc:>7.1%} "
              f"{vibe(r.pool):>9} {vibe(r.burned):>7}  {names[r.proposer]}")

    # --- earnings, read back from the chain itself ---------------------------
    earned = defaultdict(lambda: defaultdict(int))
    for block in chain.blocks[1:]:
        for tx in block.txs:
            if tx.kind == "reward":
                earned[tx.recipient][tx.memo] += tx.amount
            elif tx.kind == "slash":
                if tx.recipient != BURN:
                    earned[tx.recipient]["bounty"] += tx.amount
                earned[tx.sender]["slashed"] += tx.amount
    flops = defaultdict(int)
    units = defaultdict(int)
    for rep in proto.history:
        for a, f in rep.worker_flops.items():
            flops[a] += f
        for res in rep.results:
            if res.verdict == Verdict.ACCEPTED and res.worker in rep.worker_flops:
                units[res.worker] += 1

    print("\nWho got paid (VIBE)")
    print(f"{'worker':<17} {'hardware':<21} {'role':<22} {'units':>5} {'work':>8} {'verify':>7} {'propose':>7} "
          f"{'bounty':>7} {'slashed':>8} {'stake':>6} {'rep':>4} {'VIBE/MFLOP':>9}")
    for w in workers:
        e = earned[w.address]
        role = w.behavior if w.verifier_behavior == "honest" else w.verifier_behavior.replace("_", " ") + " verifier"
        rate = e["work"] / SYN_PER_VIBE / (flops[w.address] / 1e6) if flops[w.address] else 0
        print(f"{w.name:<17} {w.hardware.name:<21} {role:<22} {units[w.address]:>5} {vibe(e['work']):>8} {vibe(e['verify']):>7} "
              f"{vibe(e['propose']):>7} {vibe(e['bounty']):>7} {vibe(e['slashed']):>8} {vibe(chain.state.stake(w.address)):>6} "
              f"{proto.reputation[w.address]:>4.2f} {rate:>9.4f}")

    print("\nFraud caught")
    reasons = defaultdict(lambda: defaultdict(int))
    for rep in proto.history:
        for offender, _catcher, reason in rep.slashes:
            reasons[names[offender]][reason] += 1
    for who, rs in sorted(reasons.items()):
        print(f"  {who:<17} " + ", ".join(f"{k} x{v}" for k, v in rs.items()))

    st = chain.state
    print("\nLedger")
    print(f"  blocks: {len(chain.blocks)}   chain valid: {chain.validate()}")
    print(f"  supply: {vibe(st.total_supply())} VIBE = genesis {vibe(st.genesis)} + minted {vibe(st.minted)} - burned {vibe(st.burned)}"
          f"  ({'balanced' if st.total_supply() == st.expected_supply() else 'MISMATCH'})")
    print(f"  requester escrow left: {vibe(st.escrow.get(job.job_id, 0))} of {vibe(job.budget)} VIBE")
    print(f"  signatures: {crypto.BACKEND}")

    tampered = chain.blocks[3].txs[-1]
    chain.blocks[3].txs[-1] = type(tampered)(tampered.kind, tampered.sender, tampered.recipient, tampered.amount * 100,
                                             memo=tampered.memo, chain_id=tampered.chain_id)
    print(f"  after inflating one past reward x100, chain valid: {chain.validate()}")
    chain.blocks[3].txs[-1] = tampered

    print("\nSpot-check strength (chance a cheater faking fraction f is caught per unit)")
    q = proto.config.challenge_segments
    for f in (0.1, 0.25, 0.5, 1.0):
        print(f"  f={f:<4}  q={q}: {detection_probability(f, q):.1%}")
    print(f"  (plus {proto.config.replication_rate:.0%} of units replicated, and slashing = {econ.SLASH_FRACTION:.0%} of stake)")


if __name__ == "__main__":
    main()
