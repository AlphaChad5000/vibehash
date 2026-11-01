"""DoS protection for the node's HTTP server: per-client rate limits, connection caps and socket timeouts.

Every client IP gets a token bucket per kind of request (`BUCKETS`), so a flood
of one kind, say transactions, can't use up the budget for the others.
`bucket_of` sorts a request by method and first path segment, and anything it
doesn't know falls into a default bucket, so new routes are covered without
being listed. The check runs once a request's headers are in, before any
route: a request over budget gets a 429 with Retry-After, which NodeClient
waits out and retries.

The server takes at most `max_connections` connections at once (each is a
thread), and `max_connections_per_ip` from one address; past that it answers
503. Sockets time out after `header_timeout` while the request line and
headers arrive, against slow-loris, and after `io_timeout` while bodies and
replies move, which is longer because an 8 MB checkpoint chunk takes a while
on a slow link.

Loopback clients are exempt from the per-IP limits by default: a local testnet
runs a node, its workers and validator nodes all from 127.0.0.1, and must
never be throttled. A reverse proxy on the same host makes every client look
local, so rate-limit at the proxy then, or pass --rate-limit-local.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import math
import socket
import ssl
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

log = logging.getLogger("vibehash.node")

BUCKETS = {  # per client IP: (requests per second, burst)
    "read": (50.0, 500.0),  # GET: status polls, blocks, checkpoint and dataset chunks
    "worker": (20.0, 200.0),  # signed worker messages
    "tx": (2.0, 20.0),  # each new transaction is checked against the whole mempool
    "upload": (20.0, 200.0),  # PUT chunks of checkpoints and datasets
    "other": (50.0, 500.0),  # every other POST, e.g. node-to-node routes
}
ROUTES = {"tx": "tx", "register": "worker", "assignment": "worker", "commit": "worker", "reveal": "worker",
          "tasks": "worker", "verdict": "worker"}  # POST /<first segment> -> bucket; the rest use "other"
MAX_TRACKED = 50_000  # (ip, bucket) pairs remembered before idle ones are forgotten
MAX_DRAIN = 2**20  # unread request bytes a 429 reads first, so closing doesn't reset the connection under the reply
MAX_TURNED_AWAY = 1024  # connections answered 503 and waiting to be closed (beyond that, they're closed at once)


def path_parts(path: str) -> list[str]:
    """A request path's segments, read the way the node routes it: urlparse drops a scheme and host, the query, and
    ';params' after the last segment. The rate limit must read it the same way, or `/tx;x` would be counted as
    "other" and still reach /tx."""
    return [p for p in urlparse(path).path.split("/") if p]


def bucket_of(method: str, path: str) -> str:
    if method in ("GET", "HEAD"):
        return "read"
    if method == "PUT":
        return "upload"
    parts = path_parts(path)
    return ROUTES.get(parts[0] if parts else "", "other")


def is_loopback(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return addr.is_loopback or bool(getattr(addr, "ipv4_mapped", None) and addr.ipv4_mapped.is_loopback)


@dataclass
class Limits:
    buckets: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(BUCKETS))
    enabled: bool = True  # the token buckets and the per-IP connection cap
    local: bool = False  # apply those to loopback clients too
    max_connections: int = 256  # from everyone at once: always on, since each one is a thread
    max_connections_per_ip: int = 32
    header_timeout: float = 20.0
    io_timeout: float = 120.0

    @classmethod
    def parse(cls, spec: str = "default", **kw) -> "Limits":
        """'default', 'off', or overrides like 'tx=1/10,read=100': requests per second, then the burst
        (default: ten seconds' worth). Buckets not named keep their defaults."""
        limits = cls(**kw)
        spec = spec.strip().lower()
        if spec == "off":
            limits.enabled = False
            return limits
        if spec in ("", "default"):
            return limits
        for item in spec.split(","):
            name, _, value = item.strip().partition("=")
            rate, _, burst = value.partition("/")
            if name not in limits.buckets:
                raise ValueError(f"no {name!r} rate limit; there are {', '.join(limits.buckets)}")
            try:
                r = float(rate)
                b = float(burst) if burst else 10 * r
            except ValueError:
                raise ValueError(f"expected {name}=REQUESTS_PER_SECOND[/BURST], got {item.strip()!r}") from None
            if not (r > 0 and b >= 1):
                raise ValueError(f"{name}: the rate must be above 0 and the burst at least 1")
            limits.buckets[name] = (r, b)
        return limits


class RateLimiter:
    """Token buckets per (client IP, bucket). Thread-safe."""

    def __init__(self, limits: Limits, clock=time.monotonic):
        self.limits, self.clock = limits, clock
        self._buckets: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()
        self._pruned = 0.0

    def exempt(self, ip: str) -> bool:
        return not self.limits.enabled or (not self.limits.local and is_loopback(ip))

    def check(self, ip: str, bucket: str) -> float:
        """0 if the request may go ahead (spending a token), else the seconds until it could."""
        if self.exempt(ip):
            return 0.0
        rate, burst = self.limits.buckets.get(bucket) or self.limits.buckets["other"]
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get((ip, bucket), (burst, now))
            tokens = min(burst, tokens + (now - last) * rate)
            wait = 0.0 if tokens >= 1 else (1 - tokens) / rate
            self._buckets[(ip, bucket)] = [tokens - 1 if not wait else tokens, now]
            if len(self._buckets) > MAX_TRACKED and now - self._pruned > 1:
                self._forget_idle(now)
        return wait

    def _forget_idle(self, now: float) -> None:
        """Drop buckets that have refilled: forgetting them changes nothing, and keeps the table bounded."""
        self._pruned = now
        for key, (tokens, last) in list(self._buckets.items()):
            rate, burst = self.limits.buckets.get(key[1]) or self.limits.buckets["other"]
            if tokens + (now - last) * rate >= burst:
                del self._buckets[key]


class GuardedHandler(BaseHTTPRequestHandler):
    """Base class for the node's handler: socket timeouts, and the rate limit, checked centrally once the
    headers are in, so every route (including ones added later) is covered before it runs."""

    def setup(self):
        self.timeout = self.server.limits.header_timeout  # StreamRequestHandler applies it to the socket
        super().setup()

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        self.connection.settimeout(self.server.limits.io_timeout)
        bucket = bucket_of(self.command, self.path)
        wait = self.server.limiter.check(self.client_address[0], bucket)
        if not wait:
            return True
        self._drain()
        body = json.dumps({"error": f"rate limited: too many {bucket} requests from your address; retry in {wait:.1f} s"}).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(max(1, math.ceil(wait))))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True
        return False

    def _drain(self) -> None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < n <= MAX_DRAIN:
            try:
                self.rfile.read(n)
            except OSError:
                pass


class GuardedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap on simultaneous connections, overall and per client IP, and optional TLS.
    The TLS handshake happens in the connection's own thread, under the header timeout, so a client that never
    finishes one can't stall accept()."""
    daemon_threads = True

    def __init__(self, address, handler, limits: Limits | None = None, tls: ssl.SSLContext | None = None):
        self.limits = limits or Limits()
        self.limiter = RateLimiter(self.limits)
        self.tls = tls
        self._lock = threading.Lock()
        self._open: dict[str, int] = {}
        self._open_total = 0
        self._closing: deque = deque()  # (when, socket): turned away, closed once the client has had time to read why
        super().__init__(address, handler)

    @property
    def scheme(self) -> str:
        return "https" if self.tls is not None else "http"

    def _admit(self, ip: str) -> str | None:
        """None if a new connection from `ip` may go ahead, else why not."""
        with self._lock:
            if self._open_total >= self.limits.max_connections:
                return "the node is at its connection limit"
            mine = self._open.get(ip, 0)
            if mine >= self.limits.max_connections_per_ip and not self.limiter.exempt(ip):
                return "too many connections from your address"
            self._open[ip], self._open_total = mine + 1, self._open_total + 1
        return None

    def _release(self, ip: str) -> None:
        with self._lock:
            self._open_total -= 1
            self._open[ip] -= 1
            if not self._open[ip]:
                del self._open[ip]

    def process_request(self, request, client_address):
        why = self._admit(client_address[0])
        if why:
            self._turn_away(request, why)
            return
        try:
            super().process_request(request, client_address)  # starts the connection's thread
        except BaseException:
            self._release(client_address[0])
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(client_address[0])

    def finish_request(self, request, client_address):
        if self.tls is None:
            return super().finish_request(request, client_address)
        request.settimeout(self.limits.header_timeout)
        try:
            conn = self.tls.wrap_socket(request, server_side=True)
        except OSError as e:  # a scanner, a client that doesn't trust us, or one that never finished the handshake
            log.debug("TLS handshake with %s failed: %s", client_address[0], e)
            return
        try:
            self.RequestHandlerClass(conn, client_address, self)
        finally:
            conn.close()

    def _turn_away(self, request, why: str) -> None:
        """A 503 straight from the accept loop, without a thread. The send never blocks (a new socket's buffer takes
        it). The socket is closed a second later, from the serve loop: closing at once, with the client's request
        still unread, would reset the connection and could destroy the reply before the client reads it. Over TLS
        there's no handshake yet to answer in, so those clients are just disconnected."""
        if self.tls is not None:
            self.shutdown_request(request)
            return
        body = json.dumps({"error": f"busy: {why}; retry shortly"}).encode()
        head = (f"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                "Retry-After: 1\r\nConnection: close\r\n\r\n").encode()
        try:
            request.setblocking(False)
            request.send(head + body)
            request.shutdown(socket.SHUT_WR)
        except OSError:
            request.close()
            return
        if len(self._closing) >= MAX_TURNED_AWAY:
            request.close()
        else:
            self._closing.append((time.monotonic() + 1.0, request))

    def service_actions(self):
        """Called by serve_forever between connections: closes turned-away sockets whose time is up."""
        super().service_actions()
        now = time.monotonic()
        while self._closing and self._closing[0][0] <= now:
            self._discard(self._closing.popleft()[1])

    def server_close(self):
        super().server_close()
        while self._closing:
            self._discard(self._closing.popleft()[1])

    @staticmethod
    def _discard(sock) -> None:
        try:
            for _ in range(16):  # read what the client sent (up to 1 MB), so the close is a clean one
                if not sock.recv(65536):
                    break
        except OSError:
            pass
        sock.close()
