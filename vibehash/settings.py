"""Where Vibe Hash keeps its files, and the settings `vibehash setup` saves.

Everything lives in one folder, ~/.vibehash (or $VIBEHASH_HOME): the default
wallet, config.json, the worker cache and local testnets. Commands look up
--node and --wallet in this order: the flag, $VIBEHASH_NODE / $VIBEHASH_WALLET,
config.json, then the default wallet file. For https nodes, config.json also
keeps what to trust: a pinned certificate fingerprint and/or a CA file, per
node address ($VIBEHASH_TLS_FINGERPRINT / $VIBEHASH_TLS_CA override it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("VIBEHASH_HOME") or Path.home() / ".vibehash")


def config_path() -> Path:
    return home() / "config.json"


def default_wallet() -> Path:
    return home() / "wallet.json"


def load() -> dict:
    try:
        return json.loads(config_path().read_text())
    except (OSError, ValueError):
        return {}


def save(**updates) -> None:
    cfg = load() | updates
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(cfg, indent=2) + "\n")


def node_url(flag: str | None = None) -> str | None:
    return flag or os.environ.get("VIBEHASH_NODE") or load().get("node")


def tls_trust(url: str) -> tuple[str | None, str | None]:
    """(CA file, certificate fingerprint) to trust for the https node at `url`."""
    saved = load().get("tls", {}).get(url.rstrip("/"), {})
    return (os.environ.get("VIBEHASH_TLS_CA") or saved.get("ca"),
            os.environ.get("VIBEHASH_TLS_FINGERPRINT") or saved.get("fingerprint"))


def save_tls_trust(url: str, ca: str | None = None, fingerprint: str | None = None) -> None:
    trust = load().get("tls", {})
    trust[url.rstrip("/")] = {k: v for k, v in (("ca", ca), ("fingerprint", fingerprint)) if v}
    save(tls=trust)


def wallet_path(flag: str | None = None) -> Path | None:
    chosen = flag or os.environ.get("VIBEHASH_WALLET") or load().get("wallet")
    if chosen:
        return Path(chosen)
    return default_wallet() if default_wallet().exists() else None
