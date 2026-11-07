"""Where Vibe Hash keeps its files, and the settings `vibehash setup` saves.

Everything lives in one folder, ~/.vibehash (or $VIBEHASH_HOME; on a RunPod
pod, /workspace/.vibehash, which survives a restart): the default
wallet, config.json, the worker cache and local testnets. Commands look up
--node and --wallet in this order: the flag, $VIBEHASH_NODE / $VIBEHASH_WALLET,
config.json, then the public network's node / the default wallet file. For https nodes, config.json also
keeps what to trust: a pinned certificate fingerprint and/or a CA file, per
node address ($VIBEHASH_TLS_FINGERPRINT / $VIBEHASH_TLS_CA override it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

WATCH_PORT = 7545  # the worker's feed (watch.py): a thousand below a node's 8545, so a testnet's validators never land on it

PUBLIC_NODE = "https://node.vibehash.org"  # the public network: where commands go until you choose another node
PUBLIC_EVO_NODE = "https://evo.vibehash.org:8443"  # the public Evo Hash network (vibehash-evo-1): the menu's Evo Hash


def home() -> Path:
    """$VIBEHASH_HOME, else ~/.vibehash; on a RunPod pod, /workspace/.vibehash (pod_volume), unless ~/.vibehash already
    holds a setup."""
    if os.environ.get("VIBEHASH_HOME"):
        return Path(os.environ["VIBEHASH_HOME"])
    default = Path.home() / ".vibehash"
    volume = pod_volume()
    if volume is not None and not any((default / f).exists() for f in ("config.json", "wallet.json")):
        return volume / ".vibehash"
    return default


def pod_volume() -> Path | None:
    """A RunPod pod's volume, /workspace: the one place in a pod that survives a restart (its home folder doesn't, so a
    wallet kept there would be lost). RunPod sets $RUNPOD_POD_ID in every pod."""
    if os.environ.get("RUNPOD_POD_ID") and os.path.isdir("/workspace"):
        return Path("/workspace")
    return None


def config_path() -> Path:
    return home() / "config.json"


def default_wallet() -> Path:
    return home() / "wallet.json"


def load() -> dict:
    import time

    for _ in range(50 if os.name == "nt" else 1):
        try:
            return json.loads(config_path().read_text())
        except PermissionError:  # Windows, for the moment a save() puts the new file in place: read it again
            time.sleep(0.02)
        except (OSError, ValueError):
            return {}
    return {}


def save(**updates) -> None:
    """config.json with `updates`, read, changed and written under a lock between processes, written whole beside it
    and then put in its place (keeping the old file's permissions: it can hold admin tokens, made 0600). Rewritten in
    place, a load() landing mid-write read {}, and a save() from it wrote just its own keys back: the app's setup and
    its window saving at once lost the node, the wallet or a pinned certificate."""
    from . import filelock
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with filelock.hold(path.with_name("config.lock"), wait_s=10):
            return _write_config(path, updates)
    except TimeoutError:  # (a lock that never comes free: written whole all the same)
        pass
    _write_config(path, updates)


def _write_config(path: Path, updates: dict) -> None:
    import stat
    import threading
    import time

    cfg = load() | updates
    tmp = path.with_name(f"config.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
    try:
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
    except OSError:  # (none yet: it stays only yours)
        pass
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows, while a load() elsewhere has it open for a moment
            if attempt == 99:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


def chosen_node(flag: str | None = None) -> str | None:
    """The node you picked: --node, $VIBEHASH_NODE or the one setup saved. None if you haven't picked one."""
    chosen = flag or os.environ.get("VIBEHASH_NODE") or load().get("node")
    return None if not chosen or "NODE-ADDRESS" in chosen else chosen  # (an older guide's placeholder, saved by mistake)


def node_url(flag: str | None = None) -> str:
    """The node commands talk to: the one you picked, else the public network's."""
    return chosen_node(flag) or PUBLIC_NODE


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
