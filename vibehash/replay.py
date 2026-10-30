"""Re-run a block's round from its record, with the same Coordinator code that ran it.

A block's summary (who did how much work, who was caught, so who gets paid what)
has to follow from the round's signed messages by the protocol's rules. This
re-runs those rules on the record the block commits to and compares the result,
field by field. It checks:

  * every registration and verdict carries its sender's valid signature, and
    every commitment its worker's;
  * the round trained a job that still had budget, from that job's latest model;
  * the round secret is the one the leader committed to when it opened the round
    (records from version 2; older ones replay by their own version's rules);
  * each worker's share of the units, the secretly duplicated units, the beacon,
    the challenged segments, who checked whom, and the planted fakes;
  * every verdict, slash, missed deadline and payment share.

Given the round's checkpoints (`checkpoints`: something with .array(blob_id,
job) and .attach(job)), it also re-runs every check that needs the model's
weights: Merkle openings and sanity, the coordinator's own replays,
arbitration, replica disputes, and the averaged model. Any result the record
got wrong is reported. Nodes keep each round's opened checkpoints for a
challenge window (`--keep-rounds`); for older rounds, those results are taken
from the record. A replay can't see messages the coordinator never recorded;
receipts (receipts.py) cover those.
"""
from __future__ import annotations

from .coordinator import Coordinator, CoordinatorConfig, ProtocolError
from .crypto import h256
from .ledger import Block, Chain, canonical
from .tasks import InvalidJob, Reveal
from .wire import WireError, commitment_in, open_envelope


def _sender(env: dict, kind: str, chain: Chain, round_no: int) -> tuple[str, dict]:
    if "sig" not in env:
        raise ProtocolError(f"a {kind} in the record isn't signed")
    addr, body = open_envelope(env, kind, chain.chain_id, now=env.get("ts"))  # its age doesn't matter now
    if body.get("round") != round_no:
        raise ProtocolError(f"a {kind} in the record is for round {body.get('round')}")
    return addr, body


def replay_block(chain: Chain, block: Block, record: dict, checkpoints=None, stats: dict | None = None) -> list[str]:
    """Problems found re-running `block`'s round on `chain`, which must be the chain just before `block`.
    An empty list means every decision in the block follows from its record (and, with `checkpoints`, every
    weight-based result it records checks out). `stats` collects how many results were re-checked or couldn't be."""
    s = block.summary
    if s is None:
        return []
    if h256(canonical(record)).hex() != s.record_hash:
        return ["the record doesn't match the hash the block commits to"]
    try:
        rules = record["rules"]
        c = Coordinator(chain, config=CoordinatorConfig(**rules), replay=record, recheck=checkpoints)
        c.open_round()
        if c.r.job_id != s.job_id:
            return [f"the record is for job {c.r.job_id}, the block for {s.job_id}"]
        for env in record["registrations"]:
            addr, body = _sender(env, "register", chain, c.r.round_no)
            c.register(addr, float(body["flops_per_sec"]), int(body["memory_mb"]), envelope=env)
        c.start_compute()
        for d in record["commitments"]:
            c.commit(commitment_in(d))
        c.start_reveal()
        for d in record["reveals"]:
            c.reveal(Reveal(str(d["unit_id"]), str(d["worker"]), bytes.fromhex(d["merkle_root"]), bytes.fromhex(d["salt"]), None, {}), wire=d)
        c.start_verify()
        for env in record["verdicts"]:
            addr, body = _sender(env, "verdict", chain, c.r.round_no)
            bad = body.get("bad_segment")
            c.submit_verdict(addr, str(body["task_id"]), bool(body["ok"]), None if bad is None else int(bad), envelope=env)
        replayed = c.replayed_summary(s.proposer)
    except (ProtocolError, WireError, InvalidJob, KeyError, TypeError, ValueError) as e:
        return [f"the round can't be replayed from its record: {e}"]
    if stats is not None:
        stats["rechecked"] = stats.get("rechecked", 0) + c.rechecked
        stats["unchecked"] = stats.get("unchecked", 0) + c.unchecked
    want, got = s.to_dict(), replayed.to_dict()
    return c.disputes + [f"{k}: the block says {want.get(k)!r}, a replay gives {got.get(k)!r}"
                         for k in sorted(set(want) | set(got)) if want.get(k) != got.get(k)]
