"""The admin app: `vibehash admin`, a window for the people who run the network (Cole and Caleb): a globe of where
devices connect from, with arcs to the nodes they report to; the jobs, to confirm, deny (hold), cancel or post; the
ledger as blocks land; the devices online; the validators and peers.

    vibehash admin                                  # the nodes and tokens saved in config.json ("admin")
    vibehash admin --node https://node.vibehash.org --token SECRET --node http://10.0.0.5:8545
    vibehash admin --browser                        # in the browser instead of a window of its own
    vibehash admin --install-shortcut               # "Vibe Hash Admin" in the Start menu (or the apps menu), for good
    vibehash admin --remove-shortcut

It is a small local server (127.0.0.1:7546, `--port`) serving data/admin.html, with the nodes behind it: the page asks
this server (/api/*), and this server asks each node's /admin/overview with its token (nodeadmin.py), so no token
ever reaches the page, and a node without a token, or an older one, still shows what its public routes say. Every
/api request carries the session key the page was given when it loaded (X-Admin-Session), so no other page on this
computer can use the server. The window is pywebview's (the `app` extra, as window.py), the browser the fallback.

Confirm and deny are holds on every node the app knows (a hold is a node's own policy: nodeadmin.py), cancel and post
are transactions signed with the admin's wallet (settings.wallet_path, `--wallet`), unlocked in the page with its
password and kept only in this process's memory.

Places on the globe come from the devices' connection addresses, which only an admin's node gives out. They are looked
up in batches at GEO_URL (ip-api.com's free service unless $VIBEHASH_GEO_URL names another; nothing but the addresses
is sent), cached for a month in ~/.vibehash/admin/geo.json, and never for private or loopback addresses; "geo": false
in the admin settings sends nothing out, and the globe shows only the nodes' own places from their URLs.

The shortcut (`--install-shortcut`) is how an operator keeps it as an app: "Vibe Hash Admin" with its own violet icon
(data/admin.ico) in the Start menu on Windows, the applications menu on Linux, ~/Applications on a Mac. It runs this
same Python (pythonw on Windows: no console) with `-m vibehash admin`, so make it from the environment that has
Vibe Hash and pywebview, and it keeps working as that environment is updated. Any --node, --token and --wallet given
with it are saved first, so the shortcut opens with them. It is never part of the public download: the admin app is for
the people who run the nodes. Started without a console, the app logs to ~/.vibehash/admin/admin.log."""
from __future__ import annotations

import concurrent.futures
import hmac
import http.client
import ipaddress
import json
import logging
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import settings
from . import wallet as wallets
from .client import NodeClient, NodeError
from .tasks import SYN_PER_VIBE, InvalidJob, TrainingJob

log = logging.getLogger("vibehash.admin")
BROKEN = (ValueError, OSError, http.client.HTTPException)  # a node's reply that isn't one: never the whole view's error
PAGE = Path(__file__).resolve().parent / "data" / "admin.html"
GLOBE = Path(__file__).resolve().parent / "data" / "globe.json"
ICON = Path(__file__).resolve().parent / "data" / "admin.ico"
ICON_PNG = Path(__file__).resolve().parent / "data" / "admin.png"
APP_ID = "VibeHash.Admin"  # Windows' AppUserModelID: the taskbar shows the window as an app of its own, with its icon
SHORTCUT = "Vibe Hash Admin"
DEFAULT_PORT = 7546  # beside the worker's feed (7545)
TITLE = "Vibe Hash admin"
SIZE, MIN_SIZE = (1280, 820), (900, 600)
GEO_URL = "http://ip-api.com/batch?fields=status,country,city,lat,lon,query"
GEO_TTL_S = 30 * 86400
GEO_BATCH = 100
NODE_TIMEOUT = 15.0
MAX_BODY = 256 * 1024
BLOCKS_PAGE = 100


# --- where an address is ---------------------------------------------------------------------------

def is_public_ip(text: str) -> bool:
    try:
        addr = ipaddress.ip_address(text.split("%")[0])
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved
                or addr.is_unspecified)


def lookup_ip_api(ips: list[str], url: str = GEO_URL) -> dict[str, dict | None]:
    """ip-api.com's batch lookup: {ip: {"lat", "lon", "city", "country"} or None}."""
    body = json.dumps([{"query": ip} for ip in ips]).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json",
                                                                          "User-Agent": "vibehash-admin"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (the admin's own chosen service)
        answers = json.loads(resp.read(2**20))
    out: dict[str, dict | None] = dict.fromkeys(ips)
    for a in answers if isinstance(answers, list) else []:
        if isinstance(a, dict) and a.get("status") == "success" and a.get("query") in out:
            try:
                out[a["query"]] = {"lat": float(a["lat"]), "lon": float(a["lon"]), "city": str(a.get("city") or ""),
                                   "country": str(a.get("country") or "")}
            except (KeyError, TypeError, ValueError):
                pass
    return out


class Geo:
    """Places for addresses and host names, cached on disk, looked up in the background so no view waits on the
    network: `place` answers what is known now and queues the rest."""

    def __init__(self, path: Path, lookup=None, now=time.time, enabled: bool = True):
        self.path, self.lookup, self.now, self.enabled = path, lookup or lookup_ip_api, now, enabled
        self.lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.names: dict[str, str | None] = {}  # host name -> ip, this run
        self.pending: set[str] = set()
        self.thread: threading.Thread | None = None
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.cache = {k: v for k, v in saved.items() if isinstance(v, dict) and now() - v.get("when", 0) < GEO_TTL_S}
        except (OSError, ValueError):
            pass

    def ip_of(self, host: str) -> str | None:
        """A host name's address (an address is its own); None when it doesn't resolve."""
        if is_public_ip(host) or _is_ip(host):
            return host
        if host not in self.names:
            try:
                self.names[host] = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)[0][4][0]
            except (OSError, IndexError):
                self.names[host] = None
        return self.names[host]

    def place(self, host: str | None) -> dict | None:
        """{"lat", "lon", "city", "country"} for an address or host name, or None: unknown yet (queued), private, or
        not found. Nothing leaves this computer when `enabled` is off."""
        if not host:
            return None
        ip = self.ip_of(host)
        if not ip or not is_public_ip(ip):
            return None
        with self.lock:
            hit = self.cache.get(ip)
            if hit is not None:
                return hit if "lat" in hit else None
            if self.enabled:
                self.pending.add(ip)
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(target=self._run, daemon=True, name="geo")
                    self.thread.start()
        return None

    def waiting(self) -> int:
        with self.lock:
            return len(self.pending)

    def _run(self) -> None:
        while True:
            with self.lock:
                batch = sorted(self.pending)[:GEO_BATCH]
                if not batch:
                    return
            try:
                found = self.lookup(batch)
            except Exception as e:  # noqa: BLE001  (the service is down: nothing cached, asked again on the next view)
                log.debug("geolocation failed: %s", e)
                with self.lock:
                    self.pending.difference_update(batch)
                continue
            when = self.now()
            with self.lock:
                for ip in batch:
                    hit = found.get(ip) if isinstance(found, dict) else None
                    self.cache[ip] = dict(hit, when=when) if isinstance(hit, dict) else {"when": when}
                    self.pending.discard(ip)
                self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        except OSError:
            pass


def _is_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text.split("%")[0])
        return True
    except ValueError:
        return False


# --- one node ----------------------------------------------------------------------------------------

class NodeLink:
    """A node the app watches, with the token that opens its admin routes (none: its public routes only)."""

    def __init__(self, url: str, token: str | None = None, client: NodeClient | None = None):
        self.url, self.token = url.rstrip("/"), token or None
        self.client = client or NodeClient(self.url, timeout=NODE_TIMEOUT)

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"} if self.token else {}

    def admin_get(self, path: str):
        return json.loads(self.client._req("GET", path, headers=self._headers()))  # noqa: SLF001

    def admin_post(self, path: str, body: dict):
        return json.loads(self.client._req("POST", path, json.dumps(body).encode(), headers=self._headers()))  # noqa: SLF001

    def overview(self) -> dict:
        """/admin/overview with the token; a node that doesn't serve it (no token, none here, or an older release)
        gives what its public routes say instead, marked "admin": False."""
        if self.token:
            try:
                out = self.admin_get("/admin/overview")
                out["admin"], out["url"] = True, self.url
                return out
            except NodeError as e:
                if e.status not in (401, 404):
                    return {"url": self.url, "admin": False, "error": str(e)}
                error = str(e)
            except BROKEN as e:  # a reply that isn't the node's (a proxy's error page, a cut connection)
                return {"url": self.url, "admin": False, "error": f"a reply that wasn't JSON: {e}"}
        else:
            error = None
        try:
            status, net = self.client.get("/status"), self.client.get("/network")
            vals = self.client.get("/validators")
            if not all(isinstance(x, dict) for x in (status, net)):
                raise ValueError("not the node's answer")
        except NodeError as e:
            return {"url": self.url, "admin": False, "error": str(e)}
        except BROKEN as e:
            return {"url": self.url, "admin": False, "error": f"a reply that wasn't JSON: {e}"}
        return {"url": self.url, "admin": False, "error": error, "now": time.time(), "review": None,
                "node": {"url": self.url, "address": None, "version": status.get("version"), "chain_id": status.get("chain_id"),
                         "height": status.get("height"), "tip": status.get("tip"), "tip_time": net.get("time"),
                         "round": {"no": status.get("round"), "phase": status.get("phase"), "job_id": status.get("job_id")},
                         "halted": status.get("halted"), "validator": None, "supply": None, "subsidy": status.get("subsidy")},
                "validators": vals if isinstance(vals, dict) else {}, "peers": [], "mempool": [],
                "jobs": [{"id": j["id"], "owner": None, "starter": None, "escrow": j.get("escrow"), "model": j.get("model"),
                          "status": j.get("status"), "fee_per_round": j.get("fee_per_round"), "rounds_trained": j.get("rounds_trained"),
                          "units": j.get("units"), "metric": j.get("metric"), "start": j.get("start"), "now": j.get("now"),
                          "last_trained": j.get("last_trained"), "held": None, "policy": None,
                          "waiting": (status.get("waiting") or {}).get(j["id"])} for j in net.get("jobs", []) if isinstance(j, dict)],
                "devices": [], "miners": net.get("miners"), "hardware": net.get("hardware")}


def links_from(cfg: dict, urls: list[str] = (), token: str | None = None) -> list[NodeLink]:
    """The nodes to watch: `--node` URLs first (with `--token`, else the token saved for that URL, else
    $VIBEHASH_ADMIN_TOKEN, since the user named the node and set the token together), then the saved ones with the
    tokens saved for them (a saved node without one gets none: the environment's token is this operator's own
    node's, not something to send to another operator's); with neither, the node setup chose (settings.node_url)
    with $VIBEHASH_ADMIN_TOKEN."""
    saved = {n["url"].rstrip("/"): n.get("token") for n in cfg.get("nodes", []) if isinstance(n, dict) and n.get("url")}
    env = os.environ.get("VIBEHASH_ADMIN_TOKEN") or None
    out: dict[str, NodeLink] = {}
    for u in urls:
        u = u.rstrip("/")
        out[u] = NodeLink(u, token or saved.get(u) or env)
    for u, t in saved.items():
        out.setdefault(u, NodeLink(u, t or None))
    if not out:
        u = settings.node_url().rstrip("/")
        out[u] = NodeLink(u, token or env)
    return list(out.values())


# --- the app ----------------------------------------------------------------------------------------

class AdminApp:
    def __init__(self, links: list[NodeLink], wallet: Path | None, geo: Geo, save=settings.save, load=settings.load):
        self.links, self.wallet, self.geo = links, wallet, geo
        self.save, self.load = save, load
        self.keys = None  # the wallet, once unlocked
        self.session = secrets.token_urlsafe(24)
        self.hosts: set[str] = set()  # the names this server answers to, once it serves
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="admin")
        self._server: ThreadingHTTPServer | None = None
        self.lock = threading.Lock()

    # -- what the page reads --

    def overview(self) -> dict:
        views = list(self.pool.map(lambda link: link.overview(), self.links))
        primary = views[0] if views else {"url": None, "admin": False, "error": "no node", "jobs": [], "devices": []}
        devices: dict[str, dict] = {}
        for v in views:
            for d in v.get("devices") or []:
                d = dict(d, node=v["url"], place=self.geo.place(d.get("ip")) if d.get("ip") else None)
                if d["address"] not in devices or d["when"] > devices[d["address"]]["when"]:
                    devices[d["address"]] = d
        held_on: dict[str, list[str]] = {}
        for v in views:
            for j in v.get("jobs") or []:
                if j.get("policy"):
                    held_on.setdefault(j["id"], []).append(v["url"])
        nodes = []
        for v in views:
            n = dict(v.get("node") or {"url": v["url"]})
            # Its place: the address it announces to other nodes (the public one, behind a proxy), else the one we reach it at
            place = self.geo.place(urlsplit(n.get("url") or "").hostname) or self.geo.place(urlsplit(v["url"]).hostname)
            n.update(url=v["url"], announce=n.get("url"), admin=v.get("admin", False), error=v.get("error"), review=v.get("review"),
                     place=place, peers=v.get("peers") or [], mempool=len(v.get("mempool") or []), devices=len(v.get("devices") or []))
            nodes.append(n)
        validators = {}
        for v in views:
            for a, info in (v.get("validators") or {}).items():
                validators.setdefault(a, dict(info, place=self.geo.place(urlsplit(info.get("url") or "").hostname)))
        jobs = [dict(j, held_on=held_on.get(j["id"], []), mine=bool(self.keys and j.get("owner") == self.keys.address))
                for j in primary.get("jobs") or []]
        return {"now": time.time(), "primary": primary.get("url"), "nodes": nodes, "jobs": jobs,
                "devices": sorted(devices.values(), key=lambda d: -d["when"]), "validators": validators,
                "mempool": primary.get("mempool") or [], "miners": primary.get("miners"), "hardware": primary.get("hardware"),
                "review": primary.get("review"), "geo": {"enabled": self.geo.enabled, "waiting": self.geo.waiting()},
                "me": self._me(), "settings": self.settings_view()}

    def _me(self) -> dict:
        out = {"wallet": str(self.wallet) if self.wallet else None, "unlocked": self.keys is not None, "address": None,
               "encrypted": None, "balance": None, "stake": None}
        if self.wallet and self.wallet.exists():
            try:
                doc = json.loads(self.wallet.read_text())
                out["address"], out["encrypted"] = doc.get("address"), "crypto" in doc
            except (OSError, ValueError):
                pass
        if self.keys is not None:
            out["address"] = self.keys.address
        if out["address"] and self.links:
            try:
                acct = self.links[0].client.account(out["address"])
                out["balance"], out["stake"] = acct.get("balance"), acct.get("stake")
            except NodeError:
                pass
        return out

    def settings_view(self) -> dict:
        return {"nodes": [{"url": l.url, "token": bool(l.token)} for l in self.links],
                "wallet": str(self.wallet) if self.wallet else "", "geo": self.geo.enabled}

    def blocks(self, lo: int, hi: int) -> list:
        if not self.links:
            return []
        lo, hi = max(0, lo), max(0, hi)
        return self.links[0].client.get(f"/blocks?from={lo}&to={min(hi, lo + BLOCKS_PAGE - 1)}")

    # -- orders --

    def _each_admin(self, fn) -> list[dict]:
        admins = [l for l in self.links if l.token]
        if not admins:
            raise AdminError("no node has an admin token: add one under Settings")
        out = []
        for link, result in zip(admins, self.pool.map(lambda l: _try(lambda: fn(l)), admins)):
            out.append({"url": link.url, **result})
        return out

    def hold(self, job_id: str, why: str = "") -> dict:
        return {"ok": True, "nodes": self._each_admin(lambda l: l.admin_post(f"/admin/jobs/{job_id}/hold", {"why": why}))}

    def release(self, job_id: str) -> dict:
        return {"ok": True, "nodes": self._each_admin(lambda l: l.admin_post(f"/admin/jobs/{job_id}/release", {}))}

    def review(self, on: bool) -> dict:
        return {"ok": True, "nodes": self._each_admin(lambda l: l.admin_post("/admin/review", {"on": bool(on)}))}

    def unlock(self, password: str | None) -> dict:
        if not self.wallet or not self.wallet.exists():
            raise AdminError("no wallet file: set one under Settings")
        try:
            self.keys = wallets.load(self.wallet, password or None)
        except (wallets.WalletError, ValueError, OSError) as e:
            raise AdminError(f"the wallet didn't open: {e}") from None
        return {"ok": True, "address": self.keys.address}

    def _need_keys(self):
        if self.keys is None:
            raise AdminError("unlock the wallet first")
        if not self.links:
            raise AdminError("no node")
        return self.keys, self.links[0].client

    def cancel(self, job_id: str) -> dict:
        keys, client = self._need_keys()
        return {"ok": True, "txid": client.send_tx(keys, "cancel_job", "system", 0, job_id)}

    def post_job(self, spec: dict, budget_vibe) -> dict:
        keys, client = self._need_keys()
        try:
            budget = int(round(float(budget_vibe) * SYN_PER_VIBE))
        except (TypeError, ValueError):
            raise AdminError("the budget is a number of VIBE") from None
        if budget <= 0:
            raise AdminError("the budget is a number of VIBE above 0")
        try:
            job = TrainingJob.from_spec_json(json.dumps(spec), keys.address, budget)
        except (InvalidJob, ValueError, TypeError) as e:
            raise AdminError(f"invalid job spec: {e}") from None
        return {"ok": True, "job_id": job.job_id, "txid": client.send_tx(keys, "post_job", "system", budget, job.spec_json())}

    def template(self) -> dict:
        from .cli import template_job  # (the command line's starter job; cli is heavy, so only when asked)
        return template_job("job-1", False).to_spec()

    def set_settings(self, nodes: list, wallet: str | None, geo: bool | None) -> dict:
        """The page's Settings: nodes with their tokens (an empty token keeps the saved one), the wallet, geolocation."""
        saved = {l.url: l.token for l in self.links}
        links = []
        for n in nodes if isinstance(nodes, list) else []:
            if not isinstance(n, dict) or not str(n.get("url", "")).strip():
                continue
            url = str(n["url"]).strip().rstrip("/")
            if "://" not in url:
                url = "http://" + url
            token = str(n.get("token") or "").strip() or saved.get(url)
            links.append(NodeLink(url, token))
        if links:
            self.links = links
        if wallet is not None:
            chosen = Path(wallet).expanduser() if str(wallet).strip() else None
            if chosen != self.wallet:
                self.wallet, self.keys = chosen, None
        if geo is not None:
            self.geo.enabled = bool(geo)
        self.save(admin={"nodes": [{"url": l.url, **({"token": l.token} if l.token else {})} for l in self.links],
                         "wallet": str(self.wallet) if self.wallet else "", "geo": self.geo.enabled})
        try:  # the tokens are secrets: the file is this user's alone (a no-op on Windows)
            os.chmod(settings.config_path(), 0o600)
        except OSError:
            pass
        return {"ok": True, "settings": self.settings_view()}

    # -- serving the page --

    def page(self) -> bytes:
        text = PAGE.read_text(encoding="utf-8")
        boot = {"session": self.session, "nodes": [l.url for l in self.links], "version": _version()}
        safe = json.dumps(boot).replace("</", "<\\/")  # a saved URL can't end the script early
        return text.replace("<head>", f"<head>\n<script>window.ADMIN = {safe};</script>", 1).encode("utf-8")

    def serve(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> int:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "vibehash-admin"

            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload) -> None:
                self._send(code, json.dumps(payload).encode("utf-8"))

            def _host_ok(self) -> bool:
                """Only requests addressed to this server by its own name: a web page whose domain was pointed at
                127.0.0.1 (DNS rebinding) sends its own domain as Host, and gets nothing, not even the page."""
                return (self.headers.get("Host") or "").lower() in app.hosts

            def _session_ok(self) -> bool:
                given = self.headers.get("X-Admin-Session") or ""
                origin = (self.headers.get("Origin") or "").lower()
                return (self._host_ok() and hmac.compare_digest(given, app.session)
                        and (not origin or origin in {"http://" + h for h in app.hosts}))

            def do_GET(self):  # noqa: N802
                url = urlsplit(self.path)
                if not self._host_ok():
                    self._json(403, {"error": "not this server's name"})
                    return
                if url.path == "/":
                    self.send_response(200)
                    page = app.page()
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'; "
                                     "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                    self.send_header("X-Frame-Options", "DENY")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.end_headers()
                    self.wfile.write(page)
                    return
                if url.path == "/globe.json":
                    self._send(200, GLOBE.read_bytes())
                    return
                if not url.path.startswith("/api/"):
                    self._json(404, {"error": "not found"})
                    return
                if not self._session_ok():
                    self._json(403, {"error": "not this page's session"})
                    return
                q = parse_qs(url.query)
                try:
                    if url.path == "/api/overview":
                        self._json(200, app.overview())
                    elif url.path == "/api/blocks":
                        lo = int(q.get("from", ["0"])[0])
                        hi = int(q.get("to", [str(lo + BLOCKS_PAGE - 1)])[0])
                        self._json(200, app.blocks(lo, hi))
                    elif url.path == "/api/template":
                        self._json(200, app.template())
                    else:
                        self._json(404, {"error": "not found"})
                except (NodeError, AdminError, ValueError) as e:
                    self._json(400, {"error": str(e)})

            def do_POST(self):  # noqa: N802
                url = urlsplit(self.path)
                if not url.path.startswith("/api/") or not self._session_ok():
                    self._json(403, {"error": "not this page's session"})
                    return
                try:
                    n = int(self.headers.get("Content-Length") or "0")
                    if n > MAX_BODY:
                        raise ValueError("request too large")
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                    parts = [p for p in url.path.split("/") if p][1:]
                    if parts == ["review"]:
                        out = app.review(bool(body.get("on")))
                    elif len(parts) == 3 and parts[0] == "jobs" and parts[2] in ("hold", "release", "cancel"):
                        job_id = _job_id(parts[1])
                        out = (app.hold(job_id, str(body.get("why") or "")) if parts[2] == "hold"
                               else app.release(job_id) if parts[2] == "release" else app.cancel(job_id))
                    elif parts == ["jobs", "post"]:
                        spec = body.get("spec")
                        if isinstance(spec, str):
                            spec = json.loads(spec)
                        if not isinstance(spec, dict):
                            raise AdminError("the job spec is a JSON object")
                        out = app.post_job(spec, body.get("budget"))
                    elif parts == ["unlock"]:
                        out = app.unlock(str(body.get("password") or ""))
                    elif parts == ["settings"]:
                        out = app.set_settings(body.get("nodes"), body.get("wallet"), body.get("geo"))
                    else:
                        self._json(404, {"error": "not found"})
                        return
                    self._json(200, out)
                except (NodeError, AdminError, ValueError, TypeError) as e:
                    self._json(400, {"error": str(e)})

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        port = self._server.server_address[1]
        self.hosts = {f"{host}:{port}", f"localhost:{port}", f"[::1]:{port}"}  # what Host and Origin may name
        threading.Thread(target=self._server.serve_forever, daemon=True, name="admin-http").start()
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
        self.pool.shutdown(wait=False, cancel_futures=True)


class AdminError(Exception):
    """Something the page asked that can't be done, in words for it."""


def _try(fn) -> dict:
    try:
        return {"ok": True, "result": fn()}
    except (NodeError, *BROKEN) as e:
        return {"ok": False, "error": str(e)}


def _job_id(text: str) -> str:
    from .tasks import _JOB_ID
    if not _JOB_ID.match(text):
        raise AdminError("not a job id")
    return text


def _version() -> str:
    try:
        from . import _version as v
        return v()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- the window ----------------------------------------------------------------------------------------

def show(url: str, browser: bool = False, fallback=webbrowser.open, say=lambda text: print(text, file=sys.stderr)) -> str:
    """The app's page in a window of its own (pywebview), until it is closed; or in the browser (`browser`, or when
    there is no web view), and then until Ctrl+C. -> "window" or "browser"."""
    if not browser:
        try:
            import webview
        except ImportError:
            say("the window needs pywebview (pip install vibehash[app]); opening the admin page in the browser instead")
            browser = True
    if not browser:
        try:
            own_taskbar_button()
            window = webview.create_window(TITLE, url=url, width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE, text_select=True)
            try:
                window.events.shown += lambda: set_window_icon(window)
            except (AttributeError, TypeError):  # (a stand-in without events)
                pass
            webview.start(icon=str(ICON_PNG))  # GTK and Qt take the icon here; Windows from set_window_icon
            return "window"
        except KeyboardInterrupt:
            return "window"
        except Exception as e:  # noqa: BLE001  (no WebKitGTK or Qt on this Linux)
            say(f"the window can't open here ({e}); opening the admin page in the browser instead")
    fallback(url)
    return "browser"


def own_taskbar_button(system: str | None = None) -> bool:
    """On Windows, the process's own AppUserModelID, so its window is a taskbar button of its own (with its icon), not
    one more python.exe window grouped with any other."""
    if (system or sys.platform) != "win32":
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        return True
    except (AttributeError, OSError):
        return False


def set_window_icon(window, icon: Path = ICON) -> bool:
    """The violet icon on the window and its taskbar button on Windows (pywebview's WinForms form, `window.native`),
    set on the form's own thread. Anything missing leaves Python's icon."""
    try:
        from System import Action  # (pythonnet, which pywebview brings on Windows)
        from System.Drawing import Icon

        form = window.native
        form.Invoke(Action(lambda: setattr(form, "Icon", Icon(str(icon)))))
        return True
    except Exception:  # noqa: BLE001
        return False


# --- the shortcut ----------------------------------------------------------------------------------------

def shortcut_command(python: str | None = None) -> list[str]:
    """What the shortcut runs: this Python (pythonw beside it on Windows, so no console opens) with -m vibehash admin."""
    python = python or sys.executable
    exe = Path(python)
    if exe.name.lower() == "python.exe" and (exe.parent / "pythonw.exe").exists():
        python = str(exe.parent / "pythonw.exe")
    return [python, "-m", "vibehash", "admin"]


def _ps(text) -> str:
    """A PowerShell string literal."""
    return "'" + str(text).replace("'", "''") + "'"


def _powershell(script: str, run) -> list[str]:
    out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
              timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0:
        raise AdminError(f"PowerShell said: {(out.stderr or out.stdout or '').strip()[:300]}")
    return (out.stdout or "").strip().splitlines()


def _linux_entry(home: Path) -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share") / "applications" / "vibehash-admin.desktop"


def install_shortcut(system: str | None = None, run=None, home: Path | None = None, python: str | None = None) -> Path:
    """'Vibe Hash Admin' where this system keeps its apps; -> where. Replaces one already there."""
    system = system or sys.platform
    run = run or subprocess.run
    home = Path(home) if home else Path.home()
    cmd = shortcut_command(python)
    if system == "win32":
        script = ("$p = [Environment]::GetFolderPath('Programs'); "
                  "if (-not $p) { $p = Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu\\Programs' }; "
                  "New-Item -ItemType Directory -Force $p | Out-Null; "
                  f"$l = Join-Path $p {_ps(SHORTCUT + '.lnk')}; "
                  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($l); "
                  f"$s.TargetPath = {_ps(cmd[0])}; $s.Arguments = {_ps(' '.join(cmd[1:]))}; "
                  f"$s.WorkingDirectory = {_ps(home)}; $s.IconLocation = {_ps(str(ICON) + ',0')}; "
                  "$s.Description = 'The Vibe Hash network: devices, jobs, the ledger'; $s.Save(); Write-Output $l")
        where = _powershell(script, run)
        if not where:
            raise AdminError("the shortcut couldn't be made: PowerShell named no file")
        return Path(where[-1])
    if system == "darwin":  # an app bundle whose program is a two-line script
        app = home / "Applications" / f"{SHORTCUT}.app"
        macos = app / "Contents" / "MacOS"
        macos.mkdir(parents=True, exist_ok=True)
        program = macos / "admin"
        program.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(c) for c in cmd) + "\n", encoding="utf-8")
        program.chmod(0o755)
        (app / "Contents" / "Info.plist").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
            f"<key>CFBundleName</key><string>{SHORTCUT}</string><key>CFBundleExecutable</key><string>admin</string>"
            "<key>CFBundleIdentifier</key><string>org.vibehash.admin</string><key>CFBundlePackageType</key>"
            "<string>APPL</string></dict></plist>\n", encoding="utf-8")
        return app
    entry = _linux_entry(home)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("[Desktop Entry]\nType=Application\nName=" + SHORTCUT + "\nComment=The Vibe Hash network: devices, "
                     "jobs, the ledger\nExec=" + " ".join(shlex.quote(c) for c in cmd) + f"\nIcon={ICON_PNG}\n"
                     "Terminal=false\nCategories=Network;\n", encoding="utf-8")
    entry.chmod(0o755)
    return entry


def remove_shortcut(system: str | None = None, run=None, home: Path | None = None) -> Path | None:
    """Takes the shortcut away again; -> where it was, or None when there was none."""
    system = system or sys.platform
    run = run or subprocess.run
    home = Path(home) if home else Path.home()
    if system == "win32":
        where = _powershell("$p = [Environment]::GetFolderPath('Programs'); "
                            f"$l = Join-Path $p {_ps(SHORTCUT + '.lnk')}; "
                            "if (Test-Path $l) { Remove-Item -Force $l; Write-Output $l }", run)
        return Path(where[-1]) if where else None
    target = home / "Applications" / f"{SHORTCUT}.app" if system == "darwin" else _linux_entry(home)
    if not target.exists():
        return None
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return target


def save_flags(a, save=settings.save, load=settings.load) -> bool:
    """--node, --token and --wallet given with --install-shortcut, saved as the page's Settings would save them, so
    the shortcut opens with them; the nodes named come first, in their order (the first is the primary). -> whether
    anything was saved."""
    urls = [u.rstrip("/") for u in (getattr(a, "node", None) or [])]
    token, wallet = getattr(a, "token", None), getattr(a, "wallet", None)
    if not (urls or wallet):
        return False
    cfg = dict(load().get("admin") or {})
    saved = {n["url"].rstrip("/"): dict(n, url=n["url"].rstrip("/"))
             for n in cfg.get("nodes", []) if isinstance(n, dict) and n.get("url")}
    named = []
    for u in urls:
        n = saved.pop(u, {"url": u})
        if token:
            n["token"] = token
        named.append(n)
    cfg["nodes"] = named + list(saved.values())
    if wallet:
        cfg["wallet"] = str(Path(wallet).expanduser().resolve())
    save(admin=cfg)
    try:
        os.chmod(settings.config_path(), 0o600)
    except OSError:
        pass
    return True


def build(a) -> AdminApp:
    cfg = settings.load().get("admin") or {}
    links = links_from(cfg, list(getattr(a, "node", None) or []), getattr(a, "token", None))
    wallet = settings.wallet_path(getattr(a, "wallet", None)) or (Path(cfg["wallet"]) if cfg.get("wallet") else None)
    geo = Geo(settings.home() / "admin" / "geo.json", enabled=bool(cfg.get("geo", True)))
    return AdminApp(links, wallet, geo)


def _log_without_console() -> None:
    """Started from the shortcut (pythonw: no console), what it would print goes to ~/.vibehash/admin/admin.log."""
    if sys.stderr is not None:
        return
    try:
        path = settings.home() / "admin" / "admin.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 2**20:
            path.unlink()
        sys.stderr = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  (for the life of the process)
        sys.stdout = sys.stdout or sys.stderr
        sys.stderr.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    except OSError:
        pass


def main(a) -> None:
    _log_without_console()
    if getattr(a, "remove_shortcut", False):
        where = remove_shortcut()
        print(f"removed {where}" if where else "there was no shortcut to remove")
        return
    if getattr(a, "install_shortcut", False):
        if save_flags(a):
            print("saved the nodes and wallet given, for the shortcut to open with")
        try:
            import webview  # noqa: F401
        except ImportError:
            print("note: pywebview isn't in this Python, so the shortcut will open the browser; for a window of its own: "
                  f"{Path(sys.executable).name} -m pip install pywebview")
        where = install_shortcut()
        print(f"{SHORTCUT}: {where}\n  it runs {' '.join(shortcut_command())}")
        return
    app = build(a)
    try:
        port = app.serve(getattr(a, "port", DEFAULT_PORT) or DEFAULT_PORT)
    except OSError:  # the port is taken (another admin app, say): any free one does, the page carries its own session
        port = app.serve(0)
    url = f"http://127.0.0.1:{port}/"
    print(f"admin app at {url}  nodes: {', '.join(l.url + ('' if l.token else ' (no token)') for l in app.links)}", file=sys.stderr)
    try:
        if show(url, browser=bool(getattr(a, "browser", False))) == "browser":
            print("Ctrl+C stops it", file=sys.stderr)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
