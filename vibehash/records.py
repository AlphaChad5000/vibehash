"""What Vibe Hash keeps on disk, so a night of mining and training can be looked at afterwards by people who didn't
have a terminal open at the right moment.

A worker's (`vibehash worker`, and the workers inside `vibehash testnet`), in `~/.vibehash/logs` (or $VIBEHASH_HOME's):
- `worker-<date>.log`: the events the mining dashboard's feed shows (the vibehash.worker and vibehash.node loggers), a
  timestamped line each, whether the dashboard is on or not. Control characters are written out (`clean`), so nothing a
  node said can put an escape sequence on the screen of whoever reads the file. A file stops at LOG_CAP bytes and
  starts again, keeping the one before as `worker-<date>.1.log`; only the newest KEEP_LOGS days' files are kept.
- `rounds.jsonl`: a JSON line for every round a worker took part in (RoundLog): what it was given and did and how long
  that took, and, once the round's block is final, what the block paid it and why its units counted or didn't. Kept
  to ROUNDS_CAP bytes, with the one before as `rounds.1.jsonl`.
A testnet's nodes write their log to `node.log` in its folder too (LogFile with NODE_LOGGERS).

A job poster's, in `~/.vibehash/jobs/<job>`:
- `history.csv` and `history.jsonl`: the job's rounds as `GET /job/<id>` reports them (each round's validation loss
  and accuracy, and the units accepted), with each round's time and fee from its block (`GET /blocks`) and the budget
  left after it. `train` and `job watch` keep them up to date as rounds land (JobHistory); `job history` fetches them
  for any job, from any computer (fetch_history).
- beside a model `job download` saves: the same history, and a page of plain facts about how it was trained (report).

Nothing here asks a node more than a worker or watcher already did, beyond a round's block and the worker's account
once a round, and each of a job's blocks once. Writing never stops the work: a full disk, or a file another program
has open, is noted and skipped.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import socket
import threading
import time
import traceback
import unicodedata
from decimal import Decimal
from pathlib import Path

from . import settings
from .tasks import SYN_PER_VIBE

LOG_CAP = 10 * 2**20  # a log file's size before it starts again, the one before kept as NAME.1
ROUNDS_CAP = 20 * 2**20  # rounds.jsonl's, likewise: about 30,000 rounds
KEEP_LOGS = 14  # days of worker logs kept
WORKER_LOGGERS = ("vibehash.worker", "vibehash.node")  # what the mining dashboard's feed shows (minerui.py)
NODE_LOGGERS = ("vibehash.node", "vibehash.consensus", "vibehash.p2p")
SEAL_TRIES = 12  # a sealed round's block asked for this many times, 5 s apart, before its line is written without it
FULL_SPAN = 2000  # a job's rounds spanning more blocks than this get only their working rounds' blocks (and the ends)
BLOCKS_PER_REQUEST = 100  # the most a node's /blocks serves at once (node.py)
HISTORY_FIELDS = ("round", "time", "units_accepted", "loss", "accuracy", "paid_vibe", "budget_left_vibe")
_JOB_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")  # tasks.py's rule for job ids: safe as a folder name


# --- text that's safe to show ---------------------------------------------------------------------------------

_ESCAPED = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}


def clean(text) -> str:
    """`text` with its control and formatting characters (escape sequences, bidi overrides, line breaks) written out
    as \\xNN or \\uNNNN: one line of plain text, whatever a node put in it."""
    s = str(text)
    if s.isprintable():
        return s
    return "".join((f"\\x{ord(ch):02x}" if ord(ch) < 0x100 else f"\\u{ord(ch):04x}")
                   if unicodedata.category(ch) in _ESCAPED else ch for ch in s)


def _vibe(syn: int) -> float:
    return round(syn / SYN_PER_VIBE, 8)


def _vibe_text(syn: int) -> str:
    """12.5, not 12.50000000: exact to the last syn (cli.fmt's way)."""
    return format((Decimal(syn) / SYN_PER_VIBE).normalize(), "f")


def _iso(ts: float, utc: bool = True) -> str:
    if utc:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def _uint(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _real(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _replace(path: Path, text: str) -> None:
    """Write a whole file at once, so a reader never sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(tmp, path)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise PermissionError(f"{path} is open in another program (a spreadsheet?), so it can't be updated") from None


def logs_dir() -> Path:
    return settings.home() / "logs"


# --- the event log -------------------------------------------------------------------------------------------

class LogFile(logging.Handler):
    """Log records as lines in a file of at most `cap` bytes: past that the file is renamed NAME.1 (the one before goes)
    and begun again. Appended to, so several processes can share one; one that can't turn because another process has
    it open (Windows) lets it grow to twice the cap, then drops lines until it can. It never complains out loud: a log
    mustn't stop a worker, or write over its dashboard. `failed` says why the last write failed, if it did."""

    def __init__(self, path, cap: int = LOG_CAP):
        super().__init__()
        self.path, self.cap = Path(path), cap
        self.old = self.path.with_name(f"{self.path.stem}.1{self.path.suffix}")
        self.stream = None
        self.failed: str | None = None
        self.loggers: list[logging.Logger] = []

    def attach(self, names) -> "LogFile":
        for name in names:
            lg = logging.getLogger(name)
            lg.addHandler(self)
            self.loggers.append(lg)
        return self

    def line(self, record: logging.LogRecord) -> str:
        """"2026-11-02 21:14:03  INFO     worker: round 7213: registered", and a traceback's lines indented below it.
        In a testnet, a worker's lines name it (its thread is named after it)."""
        msg = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            msg += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        thread = record.threadName or ""
        who = thread if re.fullmatch(r"worker-\d+", thread) else record.name.removeprefix("vibehash.")
        first, *rest = msg.split("\n")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        return "".join([f"{stamp}  {record.levelname:<8} {clean(who)}: {clean(first)}\n",
                        *(f"    {clean(part)}\n" for part in rest)])

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = self.line(record).encode("utf-8")
            if self.stream is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.stream = open(self.path, "ab")
            if self.stream.tell() + len(data) > self.cap:
                self._turn()
            if self.stream.tell() + len(data) <= 2 * self.cap:
                self.stream.write(data)
                self.stream.flush()
            self.failed = None
        except Exception as e:  # noqa: BLE001  (see the class's docstring)
            self.failed = str(e)

    def _turn(self) -> None:
        self.stream.close()
        self.stream = None
        try:
            os.replace(self.path, self.old)
        except OSError:
            pass  # another process has it open: it turns it, or it grows to twice the cap and waits
        self.stream = open(self.path, "ab")

    def close(self) -> None:
        for lg in self.loggers:
            lg.removeHandler(self)
        self.loggers = []
        self.acquire()
        try:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
        finally:
            self.release()
        super().close()


def prune(folder: Path, keep: int = KEEP_LOGS) -> None:
    """Only the newest `keep` days of worker logs stay (each day's file and its .1)."""
    try:
        days = sorted({m[1] for p in Path(folder).glob("worker-*.log")
                       if (m := re.fullmatch(r"worker-(\d{4}-\d\d-\d\d)(?:\.1)?\.log", p.name))})
        for day in days[:-keep] if keep else days:
            for name in (f"worker-{day}.log", f"worker-{day}.1.log"):
                (Path(folder) / name).unlink(missing_ok=True)
    except OSError:
        pass


# --- a worker's rounds ---------------------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(str(Path(path).resolve()), threading.Lock())


def append_line(path: Path, obj: dict, cap: int = ROUNDS_CAP) -> None:
    """One JSON line onto `path` in a single write (so processes sharing it never interleave), turning it into
    NAME.1 first once it's past `cap`. ASCII only, escapes and all: a line can't do anything to a terminal."""
    path = Path(path)
    data = (json.dumps(obj) + "\n").encode("ascii")
    with _lock_for(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path.stat().st_size + len(data) > cap:
                os.replace(path, path.with_name(f"{path.stem}.1{path.suffix}"))
        except OSError:
            pass  # not there yet, or another process has it open: it turns another time
        with open(path, "ab") as f:
            f.write(data)


class RoundLog:
    """One worker's rounds, a JSON line each in `path` (rounds.jsonl), which a testnet's workers share.

    The worker loop (client.WorkerClient) notes what it does as it goes: `begin` when it registers for a round, `note`,
    `add` and `append` for what it's given and does, and `sealed` once the round's block is final, with the block and
    the worker's account. Pay comes from the block alone: the reward and slash transactions it holds for this worker,
    and its summary's lists of who was accepted, missed a deadline or was slashed, and the training FLOPs it credits
    (units accepted are those FLOPs over a unit's). Nothing is measured that the worker didn't do anyway, except the
    seconds its training, uploads and checks took.

    A line's fields, those that apply: time (when written), host, pid (for two workers on one computer), worker, node
    (and leader, if the round ran at another validator), registered (when), chain_id, round, attempt, job, kind (mlp,
    finetune, deepseek-prep, deepseek-training), device, speed_flops_per_s (what it registered with), assigned,
    trained, committed, revealed, accepted (units),
    accepted_flops, rejected ([{unit, why}] the node refused at the reveal), missed, slashed ([reasons]), caught
    ([reasons] others were slashed for, on this worker's word), checks (verdicts sent), checks_given, check_kinds,
    found_bad (results this worker rejected), dispute_moves, earned_vibe, paid ({work, verify, propose} in VIBE),
    lost_vibe, bounty_vibe, balance_vibe, stake_vibe, reputation (after the round), train_s, upload_s, check_s, prep
    ([a DeepSeek prep unit's own stats: seconds, tokens a second, peak GPU memory, transfers]), note, outcome (in
    words), sealed, block_time, mining (the one job it was set to mine, when it wasn't mining every job)."""

    def __init__(self, path, address: str, node: str, device: str, host: str | None = None):
        self.path, self.address, self.node, self.device = Path(path), address, node.rstrip("/"), device
        self.host = host if host is not None else clean(socket.gethostname())
        self.open: dict[int, dict] = {}
        self.tries: dict[int, tuple[int, float]] = {}
        self.rounds = self.units = self.earned = self.unsealed = 0  # this session's: rounds written, units, syn
        self.failed: str | None = None
        self.closed = False  # flushed: a worker thread still finishing its step notes nothing more
        self._lock = threading.RLock()  # the worker's thread notes; the command flushes when it stops

    # --- as the round goes ---

    def begin(self, rnd: int, *, attempt=None, chain_id: str = "", job_id: str = "", kind: str = "mlp",
              speed: float = 0.0, leader: str | None = None) -> None:
        with self._lock:
            if self.closed:
                return
            e = self.open.get(rnd)
            if e is not None:  # registered again: the round is being run again, by its next leader say
                if attempt != e.get("attempt"):
                    e["attempt"], e["attempts"] = attempt, e.get("attempts", 1) + 1
                return
            leader = (leader or "").rstrip("/")
            self.open[rnd] = {"registered": _iso(time.time(), utc=False), "chain_id": clean(chain_id)[:64], "round": rnd,
                              "attempt": attempt, "job": clean(job_id), "kind": kind, "device": self.device,
                              "speed_flops_per_s": round(speed),
                              **({"leader": leader} if leader and leader != self.node else {}),
                              "assigned": 0, "trained": 0, "committed": 0, "revealed": 0, "checks": 0,
                              "train_s": 0.0, "upload_s": 0.0, "check_s": 0.0}

    def note(self, rnd: int, **fields) -> None:
        with self._lock:
            if rnd in self.open:
                self.open[rnd].update(fields)

    def add(self, rnd: int, **counts) -> None:
        with self._lock:
            e = self.open.get(rnd)
            if e is not None:
                for k, v in counts.items():
                    e[k] = round(e.get(k, 0) + v, 3) if isinstance(v, float) else e.get(k, 0) + v

    def append(self, rnd: int, key: str, item) -> None:
        with self._lock:
            if rnd in self.open:
                self.open[rnd].setdefault(key, []).append(item)

    def tally(self, rnd: int, key: str, name: str) -> None:
        with self._lock:
            if rnd in self.open:
                d = self.open[rnd].setdefault(key, {})
                d[name] = d.get(name, 0) + 1

    # --- once it's sealed ---

    def due(self, height: int, now: float | None = None) -> list[int]:
        """Rounds the chain has sealed (their block is at or below `height`) whose block is due to be asked for."""
        now = time.time() if now is None else now
        with self._lock:
            return [r for r in sorted(self.open) if r <= height and self.tries.get(r, (0, 0.0))[1] <= now]

    def missed_block(self, rnd: int, why: str = "") -> None:
        """Its block couldn't be fetched: asked again in 5 s, and after SEAL_TRIES, written without it."""
        with self._lock:
            n = self.tries.get(rnd, (0, 0.0))[0] + 1
            if n >= SEAL_TRIES:
                self.note(rnd, note=clean(f"its block couldn't be fetched: {why}" if why else "its block couldn't be fetched"))
                self.sealed(rnd, None)
            else:
                self.tries[rnd] = (n, time.time() + 5)

    def sealed(self, rnd: int, block, extra: dict | None = None, account: dict | None = None) -> dict | None:
        """Write the round's line, from what the worker noted and its block (None: written as not known). `extra`:
        {"attempt", "prep"} of the round's record, when the worker fetched it anyway (client.audit_receipts)."""
        with self._lock:
            e = self.open.pop(rnd, None)
            self.tries.pop(rnd, None)
            if e is None:
                return None
            e.update(settle(e, block, self.address, extra))
            if isinstance(account, dict):
                for key, name in (("balance", "balance_vibe"), ("stake", "stake_vibe")):
                    if _uint(account.get(key)) is not None:
                        e[name] = _vibe(account[key])
                if _real(account.get("reputation")) is not None:
                    e["reputation"] = round(account["reputation"], 4)
            if isinstance(block, dict):
                self.rounds += 1
                self.units += e.get("accepted") or 0
                self.earned += e.pop("_earned", 0)
            else:
                self.unsealed += 1
            self._write(e)
            return e

    def flush(self, why: str = "not sealed when the worker stopped") -> None:
        """Rounds still open (the worker is stopping): written with what's known, marked not sealed. Nothing is
        noted after it."""
        with self._lock:
            for rnd in sorted(self.open):
                self.note(rnd, note=why)
                self.sealed(rnd, None)
            self.closed = True

    def _write(self, e: dict) -> None:
        e.pop("_earned", None)
        e.pop("_unit_flops", None)  # (only settle reads it)
        line = {"time": _iso(time.time(), utc=False), "host": self.host, "pid": os.getpid(), "worker": self.address,
                "node": self.node, **e}
        try:
            append_line(self.path, line)
            self.failed = None
        except OSError as e2:
            self.failed = str(e2)


def settle(e: dict, block, me: str, extra: dict | None = None) -> dict:
    """What a round's block says of this worker's part in it, for its line (RoundLog's fields). The block came from a
    node, so every field is checked before it's read."""
    out: dict = {"sealed": isinstance(block, dict)}
    kind = e.get("kind")
    if kind == "deepseek" and isinstance(extra, dict) and "prep" in extra:
        out["kind"] = "deepseek-prep" if extra["prep"] else "deepseek-training"
    if isinstance(extra, dict) and extra.get("attempt") and e.get("attempt") and extra["attempt"] != e["attempt"]:
        out["note"] = "the block came from another attempt at this round than the one this worker joined (it was run again)"
    if not isinstance(block, dict):
        done = [f"{e.get('committed', 0)} of {e['assigned']} units committed" if e.get("assigned") else "no units given",
                *([f"{e['checks']} checks sent"] if e.get("checks") else [])]
        out["outcome"] = ", ".join(done) + "; what counted isn't known (the round's block wasn't seen)"
        return out
    summary = block.get("summary") if isinstance(block.get("summary"), dict) else {}
    if _uint(block.get("timestamp")):
        out["block_time"] = _iso(block["timestamp"])
    if isinstance(summary.get("job_id"), str) and e.get("job") and summary["job_id"] != e["job"]:
        out["note"] = clean(f"the block's round trained {summary['job_id'][:64]}, not {e['job']}")
    pay, lost, bounty = {"work": 0, "verify": 0, "propose": 0}, 0, 0
    for tx in block.get("txs") if isinstance(block.get("txs"), list) else []:
        amount = _uint(tx.get("amount")) if isinstance(tx, dict) else None
        if not amount:
            continue
        if tx.get("kind") == "reward" and tx.get("recipient") == me and tx.get("memo") in pay:
            pay[tx["memo"]] += amount
        elif tx.get("kind") == "slash" and tx.get("sender") == me:
            lost += amount
        elif tx.get("kind") == "slash" and tx.get("recipient") == me:
            bounty += amount
    wf = summary.get("worker_flops") if isinstance(summary.get("worker_flops"), dict) else {}
    flops = _uint(wf.get(me)) or 0
    sizes = {f for f in e.get("_unit_flops") or [] if f}
    if not flops:
        accepted = 0
    elif len(sizes) == 1:  # a round's units are all one size: the FLOPs say how many counted
        accepted = flops // sizes.pop()
    else:
        accepted = None  # units of mixed sizes, or none committed here (the FLOPs alone can't say)
    names = lambda key: [s for s in summary.get(key) or [] if isinstance(s, str)] if isinstance(summary.get(key), list) else []  # noqa: E731
    slashes = [s for s in summary.get("slashes") or [] if isinstance(s, list) and len(s) == 3] \
        if isinstance(summary.get("slashes"), list) else []
    slashed = [clean(s[2])[:200] for s in slashes if s[0] == me]
    caught = [clean(s[2])[:200] for s in slashes if s[1] == me and s[0] != me]
    missed = me in names("missed")
    earned = sum(pay.values()) + bounty
    out.update({"accepted": accepted, "accepted_flops": flops, "missed": missed,
                **({"slashed": slashed} if slashed else {}), **({"caught": caught} if caught else {}),
                "earned_vibe": _vibe(earned), "paid": {k: _vibe(v) for k, v in pay.items() if v},
                **({"lost_vibe": _vibe(lost)} if lost else {}), **({"bounty_vibe": _vibe(bounty)} if bounty else {}),
                "_earned": earned - lost})
    out["outcome"] = _words(e, accepted, earned, missed, slashed, lost)
    return out


def _words(e: dict, accepted, earned: int, missed: bool, slashed: list, lost: int) -> str:
    """A round's outcome in a few words."""
    parts = []
    committed = e.get("committed", 0)
    if committed or accepted:
        parts.append(f"{'?' if accepted is None else accepted} of {committed} unit{'s' * (committed != 1)} accepted")
    elif e.get("assigned"):
        parts.append(f"{e['assigned']} unit{'s' * (e['assigned'] != 1)} given, none committed")
    else:
        parts.append("no units for this worker")
    if e.get("rejected"):
        parts.append(f"{len(e['rejected'])} refused at the reveal")
    if missed:
        parts.append("missed a deadline (late, or a committed unit wasn't revealed)")
    if slashed:
        parts.append(f"slashed {lost / SYN_PER_VIBE:,.2f} VIBE: " + "; ".join(slashed))
    if e.get("checks"):
        parts.append(f"{e['checks']} check{'s' * (e['checks'] != 1)} sent")
    if earned:
        parts.append(f"earned {earned / SYN_PER_VIBE:,.4f} VIBE")
    return ", ".join(parts)


class Session:
    """What one command records for its workers (`vibehash worker`, or a testnet's): the event log, every worker's
    round log (set as each client's `record`), and a summary to print at the end. `folder`: default logs_dir()."""

    def __init__(self, clients, folder=None):
        self.dir = Path(folder) if folder is not None else logs_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        prune(self.dir)
        self.log = LogFile(self.dir / f"worker-{time.strftime('%Y-%m-%d')}.log").attach(WORKER_LOGGERS)
        self.rounds = self.dir / "rounds.jsonl"
        self.logs = []
        for c in clients:
            c.record = RoundLog(self.rounds, c.keys.address, c.home.url, c.backend.describe())
            self.logs.append(c.record)

    def where(self) -> str:
        return f"recording to {self.dir}: {self.log.path.name} (what happens) and {self.rounds.name} (a line per round)"

    def close(self) -> str:
        """Write the rounds still open, stop logging, and say what this session did."""
        for r in self.logs:
            r.flush()
        self.log.close()
        n, units, earned = sum(r.rounds for r in self.logs), sum(r.units for r in self.logs), sum(r.earned for r in self.logs)
        unsealed = sum(r.unsealed for r in self.logs)
        failed = next((r.failed for r in self.logs if r.failed), None) or self.log.failed
        who = f" of {len(self.logs)} workers" if len(self.logs) > 1 else ""
        text = (f"Recorded {n} sealed round{'s' * (n != 1)}{who}: {units:,} unit{'s' * (units != 1)} accepted, "
                f"{earned / SYN_PER_VIBE:+,.4f} VIBE earned, as the blocks paid it"
                + (f" ({unsealed} more not sealed yet)" if unsealed else "") + ".\n"
                f"Logs: {self.log.path} and {self.rounds}")
        return text + (f"\n(couldn't write all of them: {failed})" if failed else "")


# --- a job's history -----------------------------------------------------------------------------------------

def job_folder(job_id: str) -> Path | None:
    """~/.vibehash/jobs/<job>, where `train` keeps the job's receipt; None for an id no job could have."""
    return settings.home() / "jobs" / job_id if isinstance(job_id, str) and _JOB_ID.fullmatch(job_id) else None


def block_facts(block, job_id: str) -> tuple[int, dict] | None:
    """(height, {"time": its timestamp, "paid": syn}) of a block from /blocks: when it was made, and the fee its round
    took from `job_id`'s budget (its fee_release transaction). None if it isn't a block."""
    if not isinstance(block, dict) or _uint(block.get("height")) is None:
        return None
    paid = sum(tx["amount"] for tx in block.get("txs") or [] if isinstance(tx, dict) and tx.get("kind") == "fee_release"
               and tx.get("memo") == job_id and _uint(tx.get("amount"))) if isinstance(block.get("txs"), list) else 0
    return block["height"], {"time": _uint(block.get("timestamp")), "paid": paid}


def fetch_blocks(api, job_id: str, heights, max_requests: int | None = None) -> dict[int, dict | None]:
    """The facts (block_facts) of the blocks at `heights`, in as few /blocks requests as cover them, at most
    BLOCKS_PER_REQUEST blocks each. A height the node didn't return maps to None, so it isn't asked again."""
    todo, got, asked = sorted(set(heights)), {}, 0
    while todo and (max_requests is None or asked < max_requests):
        lo = todo[0]
        hi = max(h for h in todo if h < lo + BLOCKS_PER_REQUEST)
        reply = api.get(f"/blocks?from={lo}&to={hi}")
        asked += 1
        seen = [f for f in (block_facts(b, job_id) for b in (reply if isinstance(reply, list) else [])) if f and lo <= f[0] <= hi]
        got.update(seen)
        last = max((h for h, _ in seen), default=lo)  # a long reply stops early (node.MAX_BLOCKS_REPLY): ask again
        for h in [h for h in todo if h <= last]:
            got.setdefault(h, None)
        todo = [h for h in todo if h > last]
    return got


def wanted(rounds, have) -> list[int]:
    """The heights whose blocks a job's history needs and hasn't asked for: every round's, or, for rounds spanning
    more than FULL_SPAN blocks (a long-lived job), those that trained, and the first and last."""
    heights = [r.height for r in rounds]
    if heights and heights[-1] - heights[0] > FULL_SPAN:
        heights = [heights[0], heights[-1], *(r.height for r in rounds if r.accepted)]
    return sorted({h for h in heights if h not in have})


def history_rows(reply, blocks: dict) -> list[dict]:
    """A job's rounds, oldest first, from its /job reply and the blocks fetched for them: round (its block's height,
    as the dashboards number rounds), time (the block's, UTC; "" when its block wasn't fetched), units_accepted, loss
    and accuracy (on the held-back data, as the node reports them), paid (syn: the block's fee_release for the job,
    else the ledger's fee for the units accepted), and budget_left (syn: what's in escrow now, plus what later rounds
    paid). The budget is worked back from the escrow, so a cancelled job's earlier rounds show what was left minus its
    refund, and a fine-tune round no scorer answered (it has no score, so no row) isn't added back."""
    from .trainui import parse_job

    job = parse_job(reply)
    rows = []
    for r in job.rounds:
        b = blocks.get(r.height)
        paid = b["paid"] if b else (job.fee * r.accepted // job.units if job.units else 0)
        rows.append({"round": r.height, "time": _iso(b["time"]) if b and b.get("time") else "",
                     "units_accepted": r.accepted, "loss": r.loss, "accuracy": r.acc, "paid": paid, "budget_left": None})
    left = job.escrow
    for row in reversed(rows):
        row["budget_left"] = left
        left = None if left is None else left + row["paid"]
    return rows


def _out(row: dict, text: bool) -> list:
    vibe = _vibe_text if text else _vibe
    return [row["round"], row["time"], row["units_accepted"], row["loss"], row["accuracy"], vibe(row["paid"]),
            None if row["budget_left"] is None else vibe(row["budget_left"])]


def csv_text(rows: list[dict]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(HISTORY_FIELDS)
    for row in rows:
        w.writerow(["" if v is None else v for v in _out(row, text=True)])
    return out.getvalue()


def json_rows(rows: list[dict]) -> list[dict]:
    return [dict(zip(HISTORY_FIELDS, _out(row, text=False))) for row in rows]


def save_history(folder: Path, job_id: str, rows: list[dict], name: str = "history") -> list[Path]:
    """NAME.csv and NAME.jsonl in `folder`, each written whole."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.csv", folder / f"{name}.jsonl"]
    _replace(paths[0], csv_text(rows))
    _replace(paths[1], "".join(json.dumps({"job": job_id, **r}) + "\n" for r in json_rows(rows)))
    return paths


def fetch_history(api, job_id: str, info: dict | None = None) -> tuple[dict, list[dict]]:
    """A job's /job reply (`info`, if the caller has it) and its rows (history_rows), with the blocks they need."""
    from .trainui import parse_job

    info = info if info is not None else api.get(f"/job/{job_id}")
    return info, history_rows(info, fetch_blocks(api, job_id, wanted(parse_job(info).rounds, {})))


class JobHistory:
    """A job's history kept up to date while `train` or `job watch` follows it: trainui.JobWatch calls `update` with
    each /job reply. Its rounds' blocks are fetched once each, a few requests at a time and not more than every
    FETCH_EVERY_S after one fails, and the files are rewritten only when something changed."""

    FETCH_EVERY_S = 30.0

    def __init__(self, job_id: str, folder: Path, api, requests_per_update: int = 10):
        self.job_id, self.folder, self.api, self.requests = job_id, Path(folder), api, requests_per_update
        self.blocks: dict[int, dict | None] = {}
        self.rows: list[dict] = []
        self._saved: str | None = None
        self._fetch_at = 0.0

    @property
    def path(self) -> Path:
        return self.folder / "history.csv"

    def update(self, reply) -> str | None:
        """-> why the history couldn't be saved, or None. Nothing a node says makes it raise."""
        from .trainui import parse_job

        try:
            need = wanted(parse_job(reply).rounds, self.blocks)
            if need and time.time() >= self._fetch_at:
                try:
                    self.blocks.update(fetch_blocks(self.api, self.job_id, need, self.requests))
                except Exception:  # noqa: BLE001  (unreachable, busy or garbled: the rows go without times for now)
                    self._fetch_at = time.time() + self.FETCH_EVERY_S
            rows = history_rows(reply, self.blocks)
            key = json.dumps(rows)
            if key != self._saved:
                save_history(self.folder, self.job_id, rows)
                self._saved = key
            self.rows = rows
            return None
        except OSError as e:
            return str(e)


# --- the page beside a downloaded model ----------------------------------------------------------------------

def _model_words(spec: dict) -> str:
    from .trainui import safe_base

    if "base" in spec:
        base = safe_base(spec.get("base")) or "a base model whose name can't be shown"
        lora = spec.get("lora") if isinstance(spec.get("lora"), dict) else {}
        rank, alpha = _uint(lora.get("rank")), _real(lora.get("alpha"))
        targets = [t for t in lora.get("targets") or [] if isinstance(t, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", t)] \
            if isinstance(lora.get("targets"), list) else []
        kind = "DeepSeek top-layers fine-tune" if "deepseek" in spec else "fine-tune"
        shape = ", ".join(x for x in (f"rank {rank}" if rank else "", f"alpha {alpha:g}" if alpha else "",
                                       f"on {', '.join(targets)}" if targets else "") if x)
        return f"a LoRA adapter ({kind}{'; ' + shape if shape else ''}) for `{base}`"
    m = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    hidden = m.get("n_hidden") if isinstance(m.get("n_hidden"), list) else [m.get("n_hidden")]
    widths = [w for w in (_uint(v) for v in [m.get("n_in"), *hidden, m.get("n_out")]) if w]
    precision = spec.get("precision") if spec.get("precision") in ("float32", "float64") else None
    return ("a network trained from scratch" + (f": {' → '.join(str(w) for w in widths)}" if widths else "")
            + (f", {precision}" if precision else ""))


def report(job_id: str, info: dict, rows: list[dict], receipt: dict | None, node: str, chain_id: str,
           history_name: str, now: float | None = None) -> str:
    """TRAINING.md: how a downloaded model was trained, in plain words, from what the node's /job and /blocks said
    and the job receipt `train` kept (`receipt`, None for a job posted from another computer). Nothing else."""
    info = info if isinstance(info, dict) else {}
    spec = info.get("spec") if isinstance(info.get("spec"), dict) else {}
    receipt = receipt if isinstance(receipt, dict) else {}
    fine = "base" in spec
    now = time.time() if now is None else now
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    lines = [f"# How {job_id} was trained", "",
             f"- **Job** `{job_id}` on {clean(node)}, chain `{clean(chain_id)[:64]}`.",
             f"- **Model**: {_model_words(spec)}."]
    data = spec.get("data") if isinstance(spec.get("data"), dict) else {}
    sha = data.get("sha256") if isinstance(data.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", data["sha256"]) else None
    source, train, val = receipt.get("source"), _uint(receipt.get("train_rows")), _uint(receipt.get("val_rows"))
    what = "examples" if fine else "rows"
    if isinstance(source, str) and train is not None and val is not None:
        name = re.split(r"[\\/]", source)[-1]  # (the receipt may have been written on Windows or not)
        lines.append(f"- **Data**: {clean(name)}: {train:,} {what} to train on, {val:,} held back to score "
                     f"it" + (f" (dataset sha256 `{sha}`)" if sha else "") + ".")
    else:
        lines.append(f"- **Data**: dataset sha256 `{sha or 'unknown'}`. The job was posted from another computer, so "
                     "its file's name and size aren't known here.")
    worked = [r for r in rows if r["units_accepted"]]
    units = sum(r["units_accepted"] for r in rows)
    if rows:
        first, last = rows[0], rows[-1]
        when = lambda r: f" ({r['time'][:10]})" if r["time"] else ""  # noqa: E731
        lines.append(f"- **Rounds**: {len(rows):,} round{'s' * (len(rows) != 1)} of this job were scored, from block "
                     f"{first['round']}{when(first)} to block {last['round']}{when(last)}; {len(worked):,} of them "
                     f"trained it, with {units:,} unit{'s' * (units != 1)} of work accepted in all.")
        scored = [r for r in rows if r["accuracy"] is not None or r["loss"] is not None]
        if scored:
            a, b = scored[0], scored[-1]
            loss = f"{a['loss']:.4f} → {b['loss']:.4f}" if a["loss"] is not None and b["loss"] is not None else None
            acc = f"{a['accuracy']:.1%} → {b['accuracy']:.1%}" if a["accuracy"] is not None and b["accuracy"] is not None else None
            if fine:
                lines.append("- **Held-back loss**: " + (loss or "not reported")
                             + (f"; reply tokens predicted right: {acc}" if acc else "") + ". A fine-tune's workers score "
                             "the model each round starts from, so the last score is the model before its newest round.")
            else:
                best = max((r["accuracy"] for r in scored if r["accuracy"] is not None), default=None)
                lines.append("- **Accuracy on the held-back rows**: " + (acc or "not reported")
                             + (f" (best {best:.1%})" if best is not None else "") + (f"; loss {loss}" if loss else "") + ".")
    else:
        lines.append("- **Rounds**: none yet: this is the model as the job started.")
    spent = sum(r["paid"] for r in rows)
    escrow = _uint(info.get("escrow"))
    budget = _uint(receipt.get("budget"))
    money = [f"posted with a budget of {_vibe_text(budget)} VIBE" if budget else "",
             f"{_vibe_text(spent)} VIBE paid for the work in these rounds",
             f"{_vibe_text(escrow)} VIBE left" if escrow is not None else ""]
    lines.append("- **Budget**: " + ", ".join(m for m in money if m) + ".")
    height = _uint(model.get("height"))
    lines.append(f"- **This copy**: downloaded {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}"
                 + (f", the model as of block {height}" if height else ", before any round had trained it") + ".")
    lines += ["", f"Round by round: `{history_name}` (block height, time in UTC, units accepted, loss, accuracy, VIBE "
              "paid, budget left after the round).", "",
              f"These facts come from the node's public `GET /job/{job_id}` and `GET /blocks`"
              + (", and the receipt `train` kept on this computer" if receipt else "")
              + ". `vibehash chain verify` checks the chain they come from.", ""]
    return "\n".join(lines)


def write_report(model_path: Path, job_id: str, info: dict, rows: list[dict], receipt: dict | None, node: str,
                 chain_id: str) -> list[Path]:
    """The history and TRAINING.md beside a model `job download` saved: inside a fine-tune's adapter folder, or next
    to a model file with its name in front (fruit-model-training-history.csv), so two models in one folder, or a
    TRAINING.md of the user's own, are never written over."""
    model_path = Path(model_path)
    if model_path.is_dir():
        folder, csv_name, md_name = model_path, "training-history.csv", "TRAINING.md"
    else:
        folder, csv_name, md_name = model_path.parent, f"{model_path.stem}-training-history.csv", f"{model_path.stem}-TRAINING.md"
    _replace(folder / csv_name, csv_text(rows))
    _replace(folder / md_name, report(job_id, info, rows, receipt, node, chain_id, csv_name))
    return [folder / csv_name, folder / md_name]
