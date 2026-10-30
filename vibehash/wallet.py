"""Wallet key files.

With a password, the secret key is encrypted with AES-256-GCM under a key
derived by scrypt (N=2^15, r=8, p=1). That needs the `cryptography` package;
we don't hand-roll a cipher. Without a password the secret is stored in the
clear, which is only acceptable for throwaway test keys.

Keys made from recovery words (mnemonic.py) are stored the same way, plus the
derivation path they came from, so the words can restore them anywhere. The
words themselves are never written to disk: they're shown once, to be written
down. Files made before recovery words, without the path, load as before.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .crypto import KeyPair, public_key
from .mnemonic import SCHEME

SCRYPT = {"n": 2**15, "r": 8, "p": 1}


class WalletError(Exception):
    pass


def _kdf(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=128 * r * n * 2, dklen=32)


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:
        raise WalletError("encrypted wallets need the `cryptography` package: pip install cryptography") from e
    return AESGCM


def save(keys: KeyPair, path: str | Path, password: str | None, hd_path: str | None = None) -> None:
    """`hd_path`: the derivation path, when the key came from recovery words."""
    path = Path(path)
    if path.exists():
        raise WalletError(f"{path} already exists; refusing to overwrite a wallet")
    doc = {"version": 1, "address": keys.address, "public": keys.public.hex()}
    if hd_path:
        doc["hd"] = {"scheme": SCHEME, "path": hd_path}
    if password:
        salt, nonce = os.urandom(16), os.urandom(12)
        key = _kdf(password, salt, **SCRYPT)
        doc["crypto"] = {"kdf": "scrypt", **SCRYPT, "salt": salt.hex(), "cipher": "aes-256-gcm", "nonce": nonce.hex(),
                         "ciphertext": _aesgcm()(key).encrypt(nonce, keys.secret, keys.public).hex()}
    else:
        doc["secret"] = keys.secret.hex()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, indent=2)


def load(path: str | Path, password: str | None) -> KeyPair:
    doc = json.loads(Path(path).read_text())
    if "secret" in doc:
        secret = bytes.fromhex(doc["secret"])
    else:
        c = doc["crypto"]
        if not password:
            raise WalletError("this wallet is encrypted; a password is required")
        key = _kdf(password, bytes.fromhex(c["salt"]), c["n"], c["r"], c["p"])
        try:
            secret = _aesgcm()(key).decrypt(bytes.fromhex(c["nonce"]), bytes.fromhex(c["ciphertext"]), bytes.fromhex(doc["public"]))
        except Exception as e:
            raise WalletError("wrong password or corrupted wallet") from e
    keys = KeyPair(secret, public_key(secret))
    if keys.public.hex() != doc["public"]:
        raise WalletError("wallet file is inconsistent")
    return keys


def is_encrypted(path: str | Path) -> bool:
    return "crypto" in json.loads(Path(path).read_text())


def hd_path(path: str | Path) -> str | None:
    """The derivation path of a wallet made from recovery words, or None for one made without them."""
    return (json.loads(Path(path).read_text()).get("hd") or {}).get("path")
