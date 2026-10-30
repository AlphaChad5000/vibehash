"""Recovery words: BIP39 mnemonics and SLIP-0010 Ed25519 key derivation.

A wallet's key is derived from 24 English words (256 bits of entropy plus an
8-bit checksum), so writing the words down backs the wallet up, and they
restore it on any machine. The steps are the standard ones hardware wallets
use, so one can hold the same key later:

    words --BIP39 (PBKDF2-HMAC-SHA512, 2048 rounds, optional passphrase)--> 64-byte seed
    seed  --SLIP-0010 ed25519, path m/44'/1'/0'/0'/0'-->                    Ed25519 private key

The path follows BIP44's shape (purpose / coin type / account / change /
index), every level hardened because SLIP-0010 only defines hardened
derivation for Ed25519. Coin type 1 is SLIP-0044's "testnet, all coins"; a
mainnet would register its own number and use it here, so testnet and
mainnet keys from the same words differ. The word list is the official one,
from github.com/bitcoin/bips (bip-0039/english.txt), and is checked against
its published SHA-256 when loaded.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import unicodedata
from functools import lru_cache
from importlib import resources

from .crypto import KeyPair, public_key

PATH = "m/44'/1'/0'/0'/0'"
SCHEME = "bip39+slip10-ed25519"
WORDLIST_SHA256 = "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"
HARDENED = 2**31


class MnemonicError(ValueError):
    pass


@lru_cache(maxsize=1)
def wordlist() -> tuple[str, ...]:
    """The 2048 BIP39 English words, in order. Line endings don't matter (git may have changed them)."""
    words = tuple(resources.files("vibehash").joinpath("data", "bip39-english.txt").read_text("ascii").split())
    if hashlib.sha256(("\n".join(words) + "\n").encode()).hexdigest() != WORDLIST_SHA256:
        raise MnemonicError("the BIP39 word list in vibehash/data is damaged")
    return words


@lru_cache(maxsize=1)
def _index() -> dict[str, int]:
    return {w: i for i, w in enumerate(wordlist())}


@lru_cache(maxsize=1)
def _by_prefix() -> dict[str, str]:
    return {w[:4]: w for w in wordlist()}  # unique by design, so metal backups can keep just 4 letters


def from_entropy(entropy: bytes) -> str:
    if len(entropy) not in (16, 20, 24, 28, 32):
        raise MnemonicError("entropy must be 128 to 256 bits, in steps of 32")
    bits = len(entropy) * 8
    n = int.from_bytes(entropy, "big") << (bits // 32) | hashlib.sha256(entropy).digest()[0] >> (8 - bits // 32)
    total = bits + bits // 32
    return " ".join(wordlist()[(n >> (total - 11 * (i + 1))) & 2047] for i in range(total // 11))


def generate(words: int = 24) -> str:
    return from_entropy(os.urandom(words * 4 // 3))


def normalize(text: str) -> str:
    """The canonical form of typed words: lower case, single spaces, 4-letter abbreviations expanded.
    Raises MnemonicError naming the first word that isn't on the list."""
    out = []
    for i, word in enumerate(unicodedata.normalize("NFKD", text).lower().split(), 1):
        if word not in _index():
            word = _by_prefix().get(word, "") if len(word) == 4 else ""
            if not word:
                raise MnemonicError(f"word {i} isn't one of the 2048 BIP39 English words")
        out.append(word)
    return " ".join(out)


def to_entropy(mnemonic: str) -> bytes:
    """The entropy the words encode. Raises MnemonicError on a wrong length or a failed checksum (a typo)."""
    words = normalize(mnemonic).split()
    if len(words) not in (12, 15, 18, 21, 24):
        raise MnemonicError(f"expected 12, 15, 18, 21 or 24 words, got {len(words)}")
    n = 0
    for w in words:
        n = n << 11 | _index()[w]
    checksum_bits = len(words) * 11 // 33
    entropy = (n >> checksum_bits).to_bytes(checksum_bits * 4, "big")
    if from_entropy(entropy) != " ".join(words):
        raise MnemonicError("the words' checksum doesn't match: check for a mistyped or swapped word")
    return entropy


def check(mnemonic: str) -> str:
    """The normalized words, after checking them."""
    to_entropy(mnemonic)
    return normalize(mnemonic)


def to_seed(mnemonic: str, passphrase: str = "") -> bytes:
    """BIP39 seed. The passphrase is an optional extra word: a different one gives a different, valid wallet."""
    words = unicodedata.normalize("NFKD", mnemonic)
    salt = unicodedata.normalize("NFKD", "mnemonic" + passphrase)
    return hashlib.pbkdf2_hmac("sha512", words.encode(), salt.encode(), 2048)


def parse_path(path: str) -> list[int]:
    parts = path.strip().split("/")
    if parts[0] != "m":
        raise MnemonicError(f"a derivation path starts with m/, not {path!r}")
    out = []
    for p in parts[1:]:
        if not (p.endswith(("'", "h", "H")) and p[:-1].isdigit() and int(p[:-1]) < HARDENED):
            raise MnemonicError(f"Ed25519 keys only derive along hardened steps like 44'; got {p!r}")
        out.append(int(p[:-1]) + HARDENED)
    return out


def slip10_ed25519(seed: bytes, path: str = PATH) -> tuple[bytes, bytes]:
    """(private key, chain code) at `path`, by SLIP-0010's Ed25519 rules."""
    i = hmac.new(b"ed25519 seed", seed, hashlib.sha512).digest()
    key, chain = i[:32], i[32:]
    for index in parse_path(path):
        i = hmac.new(chain, b"\x00" + key + index.to_bytes(4, "big"), hashlib.sha512).digest()
        key, chain = i[:32], i[32:]
    return key, chain


def keypair(mnemonic: str, passphrase: str = "", path: str = PATH) -> KeyPair:
    secret, _ = slip10_ed25519(to_seed(check(mnemonic), passphrase), path)
    return KeyPair(secret, public_key(secret))
