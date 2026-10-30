"""JSON wire format: arrays, proofs, protocol objects, and signed envelopes.

Every request from a worker is a signed envelope. The signature covers the
chain id, a timestamp and the body, so a message can't be replayed on another
network or long after it was made, and the node knows exactly who sent it.
Arrays are raw little-endian float64 in base64 — never pickle.
"""
from __future__ import annotations

import base64
import time

import numpy as np

from .crypto import KeyPair, address_of, verify
from .ledger import canonical
from .blobs import check_id
from .tasks import Commitment, WorkUnit

MAX_CLOCK_SKEW_S = 120
MAX_PROOF_DEPTH = 64


class WireError(ValueError):
    pass


def arr_out(a: np.ndarray, dtype: str = "<f8") -> str:
    """Encode at the job's precision (<f4 or <f8). Values must already be representable in it."""
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def arr_in(s: str, n: int, dtype: str = "<f8") -> np.ndarray:
    try:
        raw = base64.b64decode(s, validate=True)
    except (ValueError, TypeError) as e:
        raise WireError("bad array encoding") from e
    if len(raw) != np.dtype(dtype).itemsize * n:
        raise WireError(f"expected {n} values of {dtype}")
    return np.frombuffer(raw, dtype=dtype).astype(np.float64)


def proof_out(proof: list[tuple[bytes, bool]]) -> list:
    return [[h.hex(), bool(left)] for h, left in proof]


def proof_in(p) -> list[tuple[bytes, bool]]:
    if not isinstance(p, list) or len(p) > MAX_PROOF_DEPTH:
        raise WireError("bad merkle proof")
    try:
        return [(bytes.fromhex(h), bool(left)) for h, left in p]
    except (ValueError, TypeError) as e:
        raise WireError("bad merkle proof") from e


def blob_openings_out(openings: dict[int, tuple[str, list]]) -> dict:
    """Openings by reference: {index: [blob id, merkle proof]}. The weights travel as blobs."""
    return {str(i): [bid, proof_out(pr)] for i, (bid, pr) in openings.items()}


def blob_openings_in(d: dict, max_index: int) -> dict[int, tuple[str, list]]:
    if not isinstance(d, dict) or len(d) > max_index + 1:
        raise WireError("bad openings")
    out = {}
    for k, v in d.items():
        try:
            i, (bid, pr) = int(k), v
        except (TypeError, ValueError) as e:
            raise WireError("bad opening") from e
        if not 0 <= i <= max_index:
            raise WireError("opening index out of range")
        out[i] = (check_id(bid), proof_in(pr))
    return out


def commitment_out(c: Commitment) -> dict:
    return {"unit_id": c.unit_id, "worker": c.worker, "pubkey": c.pubkey.hex(), "digest": c.digest.hex(), "signature": c.signature.hex()}


def commitment_in(d: dict) -> Commitment:
    try:
        return Commitment(str(d["unit_id"]), str(d["worker"]), bytes.fromhex(d["pubkey"]), bytes.fromhex(d["digest"]), bytes.fromhex(d["signature"]))
    except (KeyError, ValueError, TypeError) as e:
        raise WireError("bad commitment") from e


def unit_in(d: dict) -> WorkUnit:
    try:
        return WorkUnit.from_dict(d)
    except (KeyError, ValueError, TypeError) as e:
        raise WireError("bad work unit") from e


# --- signed envelopes --------------------------------------------------------------

def seal(keys: KeyPair, kind: str, body: dict, chain_id: str, now: float | None = None) -> dict:
    ts = int(now if now is not None else time.time())
    msg = canonical([chain_id, kind, ts, body])
    return {"kind": kind, "body": body, "ts": ts, "chain_id": chain_id, "pubkey": keys.public.hex(), "sig": keys.sign(msg).hex()}


def open_envelope(env: dict, kind: str, chain_id: str, now: float | None = None) -> tuple[str, dict]:
    """Returns (sender address, body) or raises WireError."""
    try:
        pub, sig = bytes.fromhex(env["pubkey"]), bytes.fromhex(env["sig"])
        ts, body = int(env["ts"]), env["body"]
        if env["kind"] != kind or env["chain_id"] != chain_id:
            raise WireError("wrong message kind or chain")
    except (KeyError, ValueError, TypeError) as e:
        raise WireError("malformed envelope") from e
    if abs((now if now is not None else time.time()) - ts) > MAX_CLOCK_SKEW_S:
        raise WireError("message timestamp too far from node clock")
    if not isinstance(body, dict) or not verify(pub, canonical([chain_id, kind, ts, body]), sig):
        raise WireError("bad signature")
    return address_of(pub), body
