"""Hashing, Ed25519 signatures, keypairs and Merkle trees.

Signatures use the `cryptography` package (OpenSSL, constant-time) when it is
installed. Otherwise they fall back to the RFC 8032 reference algorithm in pure
Python, which is correct (see the RFC vectors in the tests) but slow and NOT
constant-time: fine for tests and simulations, not for a node holding real keys.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

    BACKEND = "openssl"
except ImportError:  # pragma: no cover - depends on the environment
    BACKEND = "pure-python"


def h256(*parts: bytes) -> bytes:
    """SHA-256 over length-prefixed parts, so ("ab", "c") != ("a", "bc")."""
    m = hashlib.sha256()
    for part in parts:
        m.update(len(part).to_bytes(8, "big"))
        m.update(part)
    return m.digest()


class Draw:
    """Random choices drawn from a seed of public data, identical on every machine and library version
    (numpy's generators promise neither), so any node can repeat a coordinator's choices exactly."""

    def __init__(self, *parts: bytes):
        self._seed, self._n = h256(*parts), 0

    def _next(self) -> int:
        self._n += 1
        return int.from_bytes(h256(self._seed, self._n.to_bytes(8, "big"))[:8], "big")

    def below(self, n: int) -> int:
        return self._next() % n

    def chance(self, p: float) -> bool:
        return self._next() < p * 2**64

    def shuffle(self, items) -> list:
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


# --- Ed25519 (RFC 8032, section 5.1) -------------------------------------

_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _inv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _add(a, b):
    A = (a[1] - a[0]) * (b[1] - b[0]) % _P
    B = (a[1] + a[0]) * (b[1] + b[0]) % _P
    C = 2 * a[3] * b[3] * _D % _P
    D = 2 * a[2] * b[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _mul(s: int, pt):
    acc = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            acc = _add(acc, pt)
        pt = _add(pt, pt)
        s >>= 1
    return acc


def _equal(a, b) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _inv(_D * y * y + 1)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * _inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(pt) -> bytes:
    zinv = _inv(pt[2])
    x, y = pt[0] * zinv % _P, pt[1] * zinv % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _expand(secret: bytes):
    digest = hashlib.sha512(secret).digest()
    a = int.from_bytes(digest[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, digest[32:]


def _hmodq(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % _Q


def _py_public_key(secret: bytes) -> bytes:
    a, _ = _expand(secret)
    return _compress(_mul(a, _G))


def _py_sign(secret: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(secret)
    pub = _compress(_mul(a, _G))
    r = _hmodq(prefix + msg)
    R = _compress(_mul(r, _G))
    s = (r + _hmodq(R + pub + msg) * a) % _Q
    return R + s.to_bytes(32, "little")


def _py_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(sig) != 64:
        return False
    A = _decompress(pub)
    R = _decompress(sig[:32])
    if A is None or R is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _Q:
        return False
    h = _hmodq(sig[:32] + pub + msg)
    return _equal(_mul(s, _G), _add(R, _mul(h, A)))


def public_key(secret: bytes) -> bytes:
    if BACKEND == "openssl":
        return Ed25519PrivateKey.from_private_bytes(secret).public_key().public_bytes_raw()
    return _py_public_key(secret)


def sign(secret: bytes, msg: bytes) -> bytes:
    if BACKEND == "openssl":
        return Ed25519PrivateKey.from_private_bytes(secret).sign(msg)
    return _py_sign(secret, msg)


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if BACKEND == "openssl":
        if len(pub) != 32 or len(sig) != 64:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
            return True
        except (InvalidSignature, ValueError):
            return False
    return _py_verify(pub, msg, sig)


@dataclass(frozen=True)
class KeyPair:
    secret: bytes
    public: bytes

    @classmethod
    def generate(cls, seed: bytes | None = None) -> "KeyPair":
        secret = hashlib.sha256(seed).digest() if seed is not None else os.urandom(32)
        return cls(secret, public_key(secret))

    @property
    def address(self) -> str:
        return address_of(self.public)

    def sign(self, msg: bytes) -> bytes:
        return sign(self.secret, msg)


def address_of(pub: bytes) -> str:
    return "vibe1" + hashlib.sha256(pub).hexdigest()[:36]


# --- Merkle tree ----------------------------------------------------------

def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return h256(b"empty")
    level = [h256(b"leaf", leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [h256(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[bytes], index: int) -> list[tuple[bytes, bool]]:
    """Sibling hashes from leaf to root. The bool is True if the sibling is on the left."""
    level = [h256(b"leaf", leaf) for leaf in leaves]
    proof = []
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling = index ^ 1
        proof.append((level[sibling], sibling < index))
        level = [h256(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)]
        index //= 2
    return proof


def merkle_verify(leaf: bytes, proof: list[tuple[bytes, bool]], root: bytes) -> bool:
    node = h256(b"leaf", leaf)
    for sibling, is_left in proof:
        node = h256(b"node", sibling, node) if is_left else h256(b"node", node, sibling)
    return node == root
