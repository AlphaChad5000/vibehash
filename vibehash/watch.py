"""A worker you can watch from outside: `vibehash worker --watch` serves what the mining dashboard shows, as JSON,
on this computer only, so a window, a tray icon or a script can follow it step by step.

    GET /                what each worker is doing now, its counts this round and this session, its account as the
                         node last reported it, and the latest events
    GET /events?after=N  every event after the one numbered N (the page polls with the last number it saw)

Served on 127.0.0.1 only (DEFAULT_PORT, or --watch PORT), read-only, with nothing in it that isn't on the dashboard
already: no keys, no wallet file, no password. A page from this computer (http://127.0.0.1:..., http://localhost:...,
or an app's own origin) may read it from a browser; other websites get no CORS header, so a page you happen to have
open can't read your balance.

Events are the lines the dashboard's feed shows (the vibehash.worker and vibehash.node loggers), kept in memory as
the last EVENTS of them, each with a number that only goes up. Account figures (balance, stake, reputation) are asked
of the node at most once every ACCOUNT_S seconds, and only while something is reading.
"""
from __future__ import annotations

import collections
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .client import NodeClient, NodeError
from .records import clean
from .settings import WATCH_PORT as DEFAULT_PORT
from .tasks import SYN_PER_VIBE
DASHBOARD = Path(__file__).resolve().parent / "data" / "dashboard.html"  # the app's window's page (window.py), at /dashboard
EVENTS = 200  # kept in memory
ACCOUNT_S = 20.0
LOGGERS = ("vibehash.worker", "vibehash.node", "vibehash.client")
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")  # the names this computer answers to: hostnames compared whole
APP_HOSTS = ("tauri.localhost",)  # an app shell's own origin (http(s)://tauri.localhost, tauri://localhost)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")  # the dashboard's colours, which a JSON feed has no use for


def local_origin(origin: str) -> bool:
    """Whether a page's Origin is this computer's (or the app's own window), compared by hostname, whole. A prefix
    test let http://localhost.example.com and http://127.0.0.1.example.com through: any website can be named so."""
    try:
        u = urlsplit(origin)
        host = u.hostname
    except ValueError:
        return False
    if u.scheme == "tauri":
        return host == "localhost"
    return u.scheme in ("http", "https") and (host in LOCAL_HOSTS or host in APP_HOSTS)


def local_host(host_header: str) -> bool:
    """Whether a request was addressed to this computer by name (its Host header). A website can't send the feed a
    request whose Host is 127.0.0.1 from a browser, except from this computer's own pages: so this stops DNS
    rebinding, where attacker.example re-resolves to 127.0.0.1 and its page then reads the feed as its own origin."""
    try:
        host = urlsplit("//" + (host_header or "")).hostname
    except ValueError:
        return False
    return host in LOCAL_HOSTS


class Events(logging.Handler):
    """The last EVENTS log records, numbered, as the dashboard's feed shows them."""

    def __init__(self, keep: int = EVENTS):
        super().__init__()
        self.items: collections.deque = collections.deque(maxlen=keep)
        self.last = 0
        self._lock = threading.Lock()
        self.loggers: list[logging.Logger] = []

    def attach(self, names=LOGGERS) -> "Events":
        for name in names:
            lg = logging.getLogger(name)
            lg.addHandler(self)
            self.loggers.append(lg)
        return self

    def detach(self) -> None:
        for lg in self.loggers:
            lg.removeHandler(self)
        self.loggers.clear()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            who = record.threadName if (record.threadName or "").startswith("worker-") else record.name.removeprefix("vibehash.")
            text = clean(_ANSI.sub("", record.getMessage().split("\n")[0]))
            with self._lock:
                self.last += 1
                self.items.append({"id": self.last, "time": record.created, "level": record.levelname.lower(),
                                   "who": clean(who or ""), "text": text})
        except Exception:  # noqa: BLE001  (a feed mustn't stop a worker)
            pass

    def since(self, after: int) -> list[dict]:
        with self._lock:
            return [e for e in self.items if e["id"] > after]


class Watch:
    """What `vibehash worker --watch` serves: one or more workers (`rigs`: (name, WorkerClient)), their round logs
    (records.RoundLog, or None) for this session's totals, and the events feed."""

    def __init__(self, rigs: list[tuple], node_url: str, events: Events | None = None, sessions: dict | None = None):
        self.rigs = list(rigs)
        self.node_url = node_url.rstrip("/")
        self.events = events or Events()
        self.sessions = sessions or {}  # worker address -> RoundLog
        self.started = time.time()
        self.chain_id: str | None = None
        self._accounts: dict[str, tuple[float, dict]] = {}  # address -> (when, the node's /account answer)
        self._api = NodeClient(self.node_url)
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- what it says ---

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        workers = [self._worker(name, client, now) for name, client in self.rigs]
        return {"now": now, "started": self.started, "node": self.node_url, "chain_id": self._chain_id(),
                "syn_per_vibe": SYN_PER_VIBE, "workers": workers, "events": self.events.since(0)[-50:],
                "last_event": self.events.last}

    def _worker(self, name: str, client, now: float) -> dict:
        s = client.stats
        job = s.job
        out = {
            "name": name, "address": client.keys.address, "device": client.backend.describe(),
            "hardware": getattr(client, "hardware_class", None),
            "round": s.round, "phase": s.phase, "phase_ends": s.phase_ends, "progress": list(s.progress) if s.progress else None,
            "height": s.height, "job": None if job is None else {"id": job.job_id, "kind": _kind(job)},
            "activity": clean(s.activity), "busy": s.busy, "flops_per_sec": s.flops_per_sec,
            "this_round": {"assigned": s.assigned, "trained": s.trained, "committed": s.committed, "checked": s.checked},
            "session": {"rounds": s.rounds, "units": s.units, "checks": s.checks, "training_flops": s.training_flops},
            "account": self._account(client.keys.address, now),
        }
        log = self.sessions.get(client.keys.address)
        if log is not None:
            out["session"] |= {"sealed_rounds": log.rounds, "accepted_units": log.units, "earned_vibe": log.earned / SYN_PER_VIBE,
                               "log": str(log.path)}
        return out

    def _account(self, address: str, now: float) -> dict | None:
        with self._lock:
            when, acct = self._accounts.get(address, (0.0, None))
            if now - when < ACCOUNT_S:
                return acct
            try:
                full = self._api.account(address)
                acct = {k: full.get(k) for k in ("balance", "stake", "reputation", "unbonding", "bootstrap") if k in full}
            except (NodeError, OSError, ValueError):
                acct = acct  # what it last said, if anything
            self._accounts[address] = (now, acct)
            return acct

    def _chain_id(self) -> str | None:
        if self.chain_id is None:
            try:
                self.chain_id = self._api.status().get("chain_id")
            except (NodeError, OSError, ValueError):
                pass
        return self.chain_id

    # --- serving it ---

    def serve(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> int:
        """Start answering on host:port (0: any free port) in a background thread. Returns the port."""
        watch = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if not local_host(self.headers.get("Host", "")):  # DNS rebinding (local_host)
                    self.send_error(403, "this feed answers to 127.0.0.1 and localhost only")
                    return
                url = urlsplit(self.path)
                if url.path in ("/dashboard", "/dashboard/"):  # the app's window: a page that reads the two below
                    page = DASHBOARD.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self' " + watch.node_url)
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if url.path == "/":
                    body = watch.snapshot()
                elif url.path == "/events":
                    try:
                        after = int(parse_qs(url.query).get("after", ["0"])[0])
                    except ValueError:
                        after = 0
                    body = {"events": watch.events.since(after), "last": watch.events.last}
                else:
                    self.send_error(404, "only / and /events")
                    return
                data = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                origin = self.headers.get("Origin", "")
                if local_origin(origin):  # a page from this computer, or the app's own window
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # quiet: the dashboard owns the terminal
                pass

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="watch")
        self._thread.start()
        return self._server.server_address[1]

    @property
    def url(self) -> str | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self.events.detach()


def _kind(job) -> str:
    for name in ("es", "deepseek", "finetune", "sharded"):
        if getattr(job, name, False):
            return name
    return "mlp"
