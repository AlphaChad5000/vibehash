"""What a node does for the people who run the network (the admin app, admin.py): a few routes behind a token,
served only when the node was started with one (`vibehash node --admin-token`, or $VIBEHASH_ADMIN_TOKEN).

    GET  /admin/overview              this node, its jobs (held or not, and why they wait), the devices that registered
                                      lately and where they connected from, the validators and peers, the mempool
    POST /admin/jobs/<id>/hold        deny: this node leads no round for the job until it is released ({"why": ...})
    POST /admin/jobs/<id>/release     confirm: the job takes its turns again
    POST /admin/review                {"on": true}: every job posted from now on starts held until an admin confirms it

Every request carries `Authorization: Bearer <token>`; without the right one the answer is 401, and a node with no
token answers 404 to all of them, as if they weren't there. The token is compared in constant time and never logged.

A hold is this node's own policy, not a rule of the chain: the coordinator skips a held job when it picks what the
next round trains (Coordinator.policy -> `reason`), and the job's budget stays in escrow untouched. Another validator
that doesn't hold the job still leads rounds for it, so the admin app tells every node it knows. The holds and the
review setting are kept in the node's database (Store meta "admin"), so they survive a restart.

Devices: a worker's registration comes through POST /register, and the node notes the address it came from (the
proxy's X-Forwarded-For when the connection is the proxy's own, else the socket's peer) with what the worker said of
its machine. Kept in memory for the admin app's map; never served on a public route, since an IP is a person's."""
from __future__ import annotations

import hmac
import ipaddress
import threading
import time

from . import hardware
from .ledger import SYSTEM

META = "admin"
MAX_DEVICES = 5000
ONLINE_S = 600  # a device that registered in the last 10 minutes is online, as netstats counts miners
MAX_WHY = 200


class Forbidden(Exception):
    """A request for an admin route without the right token (401)."""


def client_ip(peer: str, forwarded_for: str | None) -> str:
    """Where a request came from: X-Forwarded-For's first hop when the connection itself is from this computer or a
    private network (a reverse proxy in front of the node), else the socket's peer. A header from the open internet
    is ignored, since anyone can send one."""
    try:
        addr = ipaddress.ip_address(peer.split("%")[0])
        local = addr.is_loopback or addr.is_private
    except ValueError:
        local = False
    if local and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        try:
            ipaddress.ip_address(first)
            return first
        except ValueError:
            pass
    return peer


class Admin:
    def __init__(self, store, token: str | None, now=time.time):
        self.store, self.token, self.now = store, token or None, now
        self.lock = threading.Lock()
        saved = (store.load_meta(META) if store is not None else None) or {}
        self.held: dict[str, dict] = dict(saved.get("held") or {})  # job id -> {"why", "when"}
        self.review: bool = bool(saved.get("review", False))
        self.confirmed: set[str] = set(saved.get("confirmed") or [])  # jobs an admin let through under review
        self.devices: dict[str, dict] = {}  # worker address -> what it registered with, and from where
        self.changed = lambda job_id: None  # the node hooks this to drop a job's "waits" line when it is released

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def check(self, authorization: str | None) -> None:
        """The request's Authorization header against the token; KeyError (404) with no token, Forbidden (401)
        with the wrong one."""
        if not self.enabled:
            raise KeyError("admin")
        given = (authorization or "").strip()
        if given[:7].lower() == "bearer ":
            given = given[7:].strip()
        if not given or not hmac.compare_digest(given.encode(), self.token.encode()):
            raise Forbidden("the admin token is missing or wrong")

    # --- the job policy ------------------------------------------------------------------

    def reason(self, job_id: str, owner: str | None) -> str | None:
        """Why this node leads no round for the job, or None: it is held, or review is on and no admin confirmed it
        (a starter job, owned by nobody, is never held that way)."""
        held = self.held.get(job_id)
        if held is not None:
            return "an admin holds it" + (f": {held['why']}" if held.get("why") else "")
        if self.review and owner != SYSTEM and job_id not in self.confirmed:
            return "it awaits an admin's review"
        return None

    def hold(self, job_id: str, why: str = "") -> dict:
        with self.lock:
            self.held[job_id] = {"why": str(why or "")[:MAX_WHY], "when": self.now()}
            self.confirmed.discard(job_id)
            self._save()
        self.changed(job_id)
        return {"ok": True, "job_id": job_id, "held": self.held[job_id]}

    def release(self, job_id: str) -> dict:
        with self.lock:
            self.held.pop(job_id, None)
            self.confirmed.add(job_id)
            self._save()
        self.changed(job_id)
        return {"ok": True, "job_id": job_id, "held": None}

    def set_review(self, on: bool, current_jobs=()) -> dict:
        """Turning review on confirms every job already on chain, so only new ones wait."""
        with self.lock:
            self.review = bool(on)
            if self.review:
                self.confirmed.update(current_jobs)
            self._save()
        for job_id in current_jobs:
            self.changed(job_id)
        return {"ok": True, "review": self.review}

    def _save(self) -> None:
        if self.store is not None:
            self.store.save_meta(META, {"held": self.held, "review": self.review, "confirmed": sorted(self.confirmed)})

    # --- devices -----------------------------------------------------------------------------

    def saw(self, address: str, ip: str | None, body: dict, round_no: int | None = None) -> None:
        """A worker registered: what it said of its machine, and where from."""
        klass = body.get("hardware")
        entry = {"address": address, "ip": ip, "when": self.now(), "round": round_no,
                 "flops_per_sec": body.get("flops_per_sec"), "memory_mb": body.get("memory_mb"),
                 "hardware": klass if hardware.valid_class(klass) else None}
        with self.lock:
            self.devices[address] = entry
            if len(self.devices) > MAX_DEVICES:  # the ones not seen for longest go first
                for old in sorted(self.devices, key=lambda a: self.devices[a]["when"])[:len(self.devices) - MAX_DEVICES]:
                    del self.devices[old]

    def device_list(self) -> list[dict]:
        now = self.now()
        with self.lock:
            out = [dict(d, online=now - d["when"] <= ONLINE_S) for d in self.devices.values()]
        out.sort(key=lambda d: -d["when"])
        return out

    # --- the overview --------------------------------------------------------------------------

    def overview(self, node) -> dict:
        """Everything the admin app shows of this node, in one answer."""
        now = self.now()
        s = node.chain.state
        net = node.network()  # cached: the jobs' summaries (netstats), the census
        summaries = {j["id"]: j for j in net.get("jobs", [])}
        with node.coord.lock:
            r = node.coord.r
            mempool = [{"txid": tx.txid.hex(), "kind": tx.kind, "sender": tx.sender, "recipient": tx.recipient,
                        "amount": tx.amount, "fee": tx.fee, "memo": tx.memo[:120]} for tx in node.coord.mempool[:50]]
            waiting = dict(node.coord.waiting)
            rnd = {"no": r.round_no if r else None, "phase": node.coord.phase.value, "job_id": r.job_id if r else None,
                   "registered": len(r.regs) if r else 0}
        jobs = []
        for job_id, spec in list(s.jobs.items()):
            owner = s.job_owner.get(job_id)
            summary = summaries.get(job_id, {})
            jobs.append({"id": job_id, "owner": owner, "starter": owner == SYSTEM, "escrow": s.escrow.get(job_id, 0),
                         "model": summary.get("model"), "status": summary.get("status"),
                         "fee_per_round": summary.get("fee_per_round"), "rounds_trained": summary.get("rounds_trained"),
                         "units": summary.get("units"), "metric": summary.get("metric"), "start": summary.get("start"),
                         "now": summary.get("now"), "last_trained": summary.get("last_trained"),
                         "held": self.held.get(job_id), "policy": self.reason(job_id, owner), "waiting": waiting.get(job_id),
                         "spec_bytes": len(spec)})
        voting = s.validator_set()
        dead = dict(node.peers.dead_until)
        return {
            "now": now, "review": self.review,
            "node": {"url": node.announce, "address": node.keys.address, "version": net.get("version") or node.status().get("version"),
                     "chain_id": node.chain.chain_id, "height": node.chain.tip.height, "tip": node.chain.tip.hash.hex(),
                     "tip_time": node.chain.tip.timestamp, "round": rnd, "halted": node.halted_public,
                     "validator": node.keys.address in voting, "supply": _safe(node.supply),
                     "subsidy": s.subsidy(node.chain.tip.height + 1)},
            "validators": {a: {"url": u, "stake": s.stake(a), "votes": a in voting} for a, u in s.validators.items()},
            "peers": [{"url": u, "alive": dead.get(u, 0) <= now} for u in node.peers.urls()],
            "mempool": mempool, "jobs": jobs, "devices": self.device_list(),
            "miners": net.get("miners"), "hardware": net.get("hardware"),
        }


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001  (a view: nothing in it may fail the whole answer)
        return None
