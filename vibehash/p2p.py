"""How nodes reach each other: the peers they gossip to, and catching up on blocks.

A node's peers are the addresses it was started with (--peer) plus every
validator's announced address on chain, so validators find each other from
the genesis file alone and a new validator becomes reachable once its
announcement is in a block. Messages are pushed to every peer: consensus votes
and proposals (re-sent until the height is decided, so a peer that missed one
still gets it), and transactions (forwarded once by each node that accepts
them, so they reach whoever proposes next).

A node that falls behind (it was offline, or a vote names a later height)
fetches the missing blocks from a peer (node.py). It trusts none of them: each
must apply to its chain, replay from its round record, and carry a commit
certificate signed by more than two thirds of the validators' stake.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .client import NodeClient, NodeError

log = logging.getLogger("vibehash.p2p")
BACKOFF_S = 5.0  # a peer that didn't answer isn't tried again for this long


class Peers:
    def __init__(self, own_url: str, own_address: str, static: list[str], validators):
        """`validators()` returns the chain's {address: announced url}. Neither our own url nor the one the chain
        lists for our address is a peer."""
        self.own, self.address = own_url.rstrip("/"), own_address
        self.static = [u.rstrip("/") for u in static]
        self._validators = validators
        self.pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gossip")
        self.lock = threading.Lock()
        self.dead_until: dict[str, float] = {}

    def urls(self, include_dead: bool = True) -> list[str]:
        now = time.time()
        vals = self._validators()
        found = dict.fromkeys([*self.static, *(u.rstrip("/") for a, u in vals.items() if a != self.address)])
        with self.lock:
            return [u for u in found if u != self.own and (include_dead or self.dead_until.get(u, 0) <= now)]

    def client(self, url: str, timeout: float = 10) -> NodeClient:
        return NodeClient(url, timeout=timeout)

    def broadcast(self, path: str, body: dict) -> None:
        """Send to every live peer in the background; failures only mark the peer as down for a while."""
        for url in self.urls(include_dead=False):
            self.pool.submit(self._post, url, path, body)

    def _post(self, url: str, path: str, body: dict) -> None:
        try:
            self.client(url, timeout=5).post(path, body)
        except NodeError as e:
            if e.status is None:  # unreachable; a 4xx (say, a transaction it already has) is an answer
                with self.lock:
                    self.dead_until[url] = time.time() + BACKOFF_S
                log.debug("peer %s: %s", url, e)

    def alive(self, url: str) -> None:
        with self.lock:
            self.dead_until.pop(url, None)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
