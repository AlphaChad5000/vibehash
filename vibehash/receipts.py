"""Signed receipts: a node's promise that a message it accepted will be in its round's record.

When the node accepts a worker's registration, commitment, reveal or verdict, it
signs a receipt naming the round, the round's attempt label and the hash of the
message exactly as the record holds it. If the round's block then commits to a
record without that message, the receipt proves the node dropped it. A replay
can't see a message that was never recorded, but its sender can show it was
accepted. Workers keep their receipts and check each round once it's sealed;
`vibehash chain verify` checks any receipts it's given.

A round the node abandons (it crashed, say) and runs again under the same
number gets a new attempt label, so receipts from the abandoned attempt don't
count as censorship.
"""
from __future__ import annotations

from .crypto import KeyPair, h256, verify
from .ledger import Block, canonical

KINDS = {"register": "registrations", "commit": "commitments", "reveal": "reveals", "verdict": "verdicts"}


def entry_digest(entry: dict) -> str:
    """A message as the round record holds it, hashed: what a receipt vouches for."""
    return h256(canonical(entry)).hex()


def _signed_part(chain_id: str, round_no: int, attempt: str, kind: str, digest: str) -> bytes:
    return canonical([chain_id, "receipt", round_no, attempt, kind, digest])


def make(keys: KeyPair, chain_id: str, round_no: int, attempt: str, kind: str, entry: dict) -> dict:
    digest = entry_digest(entry)
    return {"chain_id": chain_id, "round": round_no, "attempt": attempt, "kind": kind, "digest": digest,
            "node": keys.public.hex(), "sig": keys.sign(_signed_part(chain_id, round_no, attempt, kind, digest)).hex()}


def valid(receipt: dict, chain_id: str | None = None) -> bool:
    try:
        ok = verify(bytes.fromhex(receipt["node"]),
                    _signed_part(receipt["chain_id"], int(receipt["round"]), str(receipt["attempt"]), receipt["kind"], receipt["digest"]),
                    bytes.fromhex(receipt["sig"]))
    except (KeyError, ValueError, TypeError):
        return False
    return ok and receipt["kind"] in KINDS and (chain_id is None or receipt["chain_id"] == chain_id)


def audit(receipt: dict, block: Block | None, record: dict | None) -> tuple[str, str]:
    """What a receipt shows about its round, as (status, explanation). Statuses:
    recorded   the message is in the record, as promised
    censored   the node signed for it, then sealed the round without it: proof of censorship
    abandoned  that attempt at the round was never sealed (another attempt, or no round, took its block)
    pending    the round hasn't been sealed yet
    unrelated  signed by a different node from the one that sealed the round, so it proves nothing about it
    invalid    the signature doesn't check out"""
    n, kind = receipt.get("round"), receipt.get("kind")
    if not valid(receipt):
        return "invalid", f"round {n}: a receipt whose signature doesn't check out"
    if block is None:
        return "pending", f"round {n} hasn't been sealed yet"
    if block.summary is None or record is None or record.get("attempt") != receipt["attempt"]:
        return "abandoned", f"round {n}: the attempt you took part in was abandoned and the block went to another"
    if block.proposer_pubkey.hex() != receipt["node"]:
        return "unrelated", f"round {n}: the receipt is from a different node from the one that sealed the block"
    if receipt["digest"] in {entry_digest(e) for e in record.get(KINDS[kind], [])}:
        return "recorded", f"round {n}: your {kind} is in the record"
    return "censored", (f"round {n}: the node signed a receipt for your {kind} but sealed the round without it. "
                        "Keep the receipt: it's proof the node dropped your message")
