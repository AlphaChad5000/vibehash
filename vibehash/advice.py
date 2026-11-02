"""What went wrong, in a line or two a newcomer can act on: the problems people hit setting up and mining for the first
time (a node out of reach, a clock that's off, a GPU out of memory, a full disk, a job waiting for staked workers).
Each says what happened and what to do. `explain(e)` maps an exception to its advice, or None for one it doesn't
know; `selftest`, `setup`, the worker loop and every command's error exit use it.
"""
from __future__ import annotations

import errno
import re
from urllib.parse import urlsplit

from .wire import MAX_CLOCK_SKEW_S

GUIDE = "https://vibehash.org/guide/"


def _gb(mb: float) -> str:
    return f"{mb / 1024:.1f} GB"


def _vibe(syn: int) -> str:
    return f"{syn / 1e8:,.8g} VIBE"


def _where(url: str) -> tuple[str, str]:
    u = urlsplit(url if "://" in url else f"http://{url}")
    port = u.port or (443 if u.scheme == "https" else 80)
    return u.hostname or url, str(port)


def reach_kind(reason: str) -> str:
    """dns, refused, timeout, tls or other: why a connection failed, from the operating system's words for it."""
    r = reason.lower()
    if "getaddrinfo" in r or "name or service" in r or "name resolution" in r or "nodename nor servname" in r \
            or "no address associated" in r or "11001" in r or "11004" in r:
        return "dns"
    if "refused" in r or "10061" in r:
        return "refused"
    if "certificate" in r or "ssl" in r or "tls" in r:
        return "tls"
    if "timed out" in r or "timeout" in r or "10060" in r or "unreachable" in r or "10065" in r or "10051" in r:
        return "timeout"
    return "other"


def unreachable(url: str, reason: str) -> str:
    """The node didn't answer at all."""
    host, port = _where(url)
    head = f"can't reach the node at {url} ({reason})."
    kind = reach_kind(reason)
    if kind == "dns":
        return (f"{head}\n{host} doesn't resolve to an address: check the spelling. On a campus or dorm network that "
                "filters names, try the public node (node.vibehash.org) or a phone's hotspot.")
    if kind == "refused":
        return (f"{head}\nNothing is listening at {host} port {port}: the node isn't running there, or it uses another "
                "port (testnets use 8545). Check the address with whoever runs it.")
    if kind == "tls":
        return (f"{head}\n{host}'s certificate didn't check out. If this Wi-Fi has a sign-in page, sign in first; check "
                "this computer's clock is right; a self-signed testnet node needs `setup --tls-fingerprint`.")
    if kind == "timeout":
        return (f"{head}\nNo answer from {host} port {port}: a firewall or the network is in the way. Dorm and campus "
                "Wi-Fi often stop computers reaching each other and unusual ports: use the public node "
                "(node.vibehash.org), or put both computers on a phone's hotspot. A node for other computers needs "
                "--host 0.0.0.0.")
    return f"{head}\nCheck this computer is online and the address is right."


def clock(skew_s: float | None = None) -> str:
    """This computer's clock is too far from the node's for its signed messages to count."""
    off = (f"is {abs(skew_s):.0f} s {'ahead of' if skew_s > 0 else 'behind'} the node's"
           if skew_s is not None else "is too far from the node's")
    return (f"This computer's clock {off}, and nodes refuse signed messages more than {MAX_CLOCK_SKEW_S} s off. Set "
            "it automatically, then start again: on Windows, Settings > Time & language > Date & time > Sync now; on "
            "Linux, `sudo timedatectl set-ntp true` (a container or pod keeps its host's clock).")


def wrong_chain(url: str, old: str | None = None, new: str | None = None) -> str:
    was = f" ({new}, not {old})" if old and new else ""
    return (f"The node at {url} is now a different network{was}: it was reset, or the address points at another "
            "node. Your wallet is fine. Start the worker again to mine there (stake on a new network starts again), "
            "or `vibehash setup --node ADDRESS` to go back.")


def out_of_memory(device: str, need_mb: int | None = None, total_mb: int | None = None,
                  free_mb: int | None = None) -> str:
    needs = f" This job needs about {_gb(need_mb)}" if need_mb else ""
    has = (f" and the card has {_gb(total_mb)}" + (f", {_gb(free_mb)} of it free now" if free_mb is not None else "")
           if total_mb else "")
    return (f"The GPU ({device}) ran out of memory.{needs}{has}{'.' if needs or has else ''} Close what else uses it "
            "(games, browsers, video, another worker on the same card); the next round tries again. Or mine on "
            "another card with --device, or the CPU with --device cpu (CSV jobs only).")


def disk(path, free: int | None = None, need: int | None = None) -> str:
    amounts = (f" ({free / 1e9:.1f} GB free" + (f", {need / 1e9:.1f} GB needed" if need else "") + ")"
               if free is not None else "")
    return (f"The disk holding {path} is full{amounts}. Free some space, or keep Vibe Hash on a bigger drive: set "
            "VIBEHASH_HOME to a folder there (and `vibehash setup --models-dir FOLDER` for big model files).")


def no_stake(min_stake: int) -> str:
    return (f"This network gives work only to workers with {_vibe(min_stake)} staked, and doesn't let newcomers earn "
            f"it: ask its operator for VIBE, then `vibehash stake --amount {min_stake // 10**8}`.")


def waiting(job_id: str, why: str, stake: int | None, min_stake: int, memory_mb: int | None, need_mb: int | None,
            others: list[str] = (), earning: str = "", earn_in: bool = True) -> str:
    """A job the node says waits (/status "waiting"), and what it means for this worker. `earning`: how long its stake
    should take at the network's recent pace (selftest works it out), if known."""
    gist = why.split(":")[0] if why.startswith(f"job {job_id} waits") else f"job {job_id} waits"
    if "staked workers" not in why:
        return f"{why}."
    on = " and ".join(others) if others else "the other jobs"
    if stake is not None and stake < min_stake and not earn_in:
        return f"{gist} for staked workers to check each other's work. {no_stake(min_stake)}"
    if stake is not None and stake < min_stake:
        return (f"{gist} for staked workers to check each other's work. This worker has {_vibe(stake)} of the "
                f"{_vibe(min_stake)} stake: it earns the rest on {on}{earning}, then it can take this job's units too. "
                "Nothing to do but keep it running (`vibehash balance` shows the stake filling up).")
    if memory_mb is not None and need_mb is not None and memory_mb < need_mb:
        return (f"{gist}: it needs {_gb(need_mb)} of GPU memory a worker, and this one has {_gb(memory_mb)}, so it "
                f"keeps earning on {on}.")
    return (f"{gist} for another staked worker with {_gb(need_mb) if need_mb else 'enough GPU memory'}: this one "
            "counts, and the job starts when one more joins. Nothing to do but keep mining.")


def explain(e: BaseException, url: str | None = None) -> str | None:
    """Advice for an exception a command or the worker loop met, or None if it's not one of these."""
    from .client import NodeError

    text = str(e)
    if isinstance(e, NodeError):
        if e.status is None and text.startswith("can't reach node at "):
            where, _, reason = text[len("can't reach node at "):].partition(": ")
            return unreachable(where, reason or "no answer")
        if "timestamp too far from node clock" in text:
            return f"{text}.\n{clock()}"
        if "switched networks" in text or "wrong message kind or chain" in text:
            return f"{text}.\n{wrong_chain(url or 'the node')}"
        m = re.search(r"stake at least (\d+) syn", text)
        if m:
            return no_stake(int(m.group(1)))
        return None
    name = type(e).__name__
    if name == "OutOfMemoryError" or ("out of memory" in text.lower() and ("cuda" in text.lower() or "hip" in text.lower())):
        return out_of_memory("GPU")
    if name == "DiskFullError":
        return text
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return disk(e.filename or "the cache")
    return None
