"""Transaction fees, and the mempool policy built on them.

Every user transaction (transfer, stake, unstake, post_job, cancel_job) can pay
a fee on top of its amount. The sender's balance pays it and the block's
proposer receives it, credited once the block's user transactions are
applied (ledger.apply_block), so it is a transfer and supply is unchanged. A
fee is part of what the sender signs; a fee of 0 is left out of the signed
payload, so transactions and blocks from before fees keep their hashes.

Worker protocol messages (register, commit, reveal, verdict) aren't
transactions and stay free.

Nodes only relay transactions paying at least their minimum fee (`--min-fee`),
keep the mempool in fee order (`by_fee`) so blocks take the best-paying
transactions first (a block holds up to ledger.MAX_BLOCK_TX_BYTES of them; the
rest wait), and when it's full drop the cheapest one nothing else in it
depends on (`evictable`).
"""
from __future__ import annotations

import heapq
from collections import defaultdict

from .ledger import Tx
from .tasks import SYN_PER_VIBE

DEFAULT_MIN_FEE = SYN_PER_VIBE // 10_000  # 0.0001 VIBE: nothing to a testnet user, 1 VIBE to fill a 10,000-tx mempool


def _parents(txs: list[Tx]) -> list[set[int]]:
    """For each tx, the earlier ones it may depend on: its sender's previous tx (nonces go in order), and transfers
    to its sender since then (it may be spending them). Earlier transfers are ancestors through the previous tx."""
    parents, last, incoming = [], {}, defaultdict(list)
    for j, tx in enumerate(txs):
        p = set(incoming.pop(tx.sender, []))
        if tx.sender in last:
            p.add(last[tx.sender])
        parents.append(p)
        last[tx.sender] = j
        if tx.kind == "transfer":
            incoming[tx.recipient].append(j)
    return parents


def by_fee(txs: list[Tx]) -> list[Tx]:
    """Highest fee first, but never ahead of a transaction it may depend on, so a sequence that applies in order
    still does. Ties keep their order."""
    parents = _parents(txs)
    waiting = [len(p) for p in parents]
    children = defaultdict(list)
    for j, p in enumerate(parents):
        for i in p:
            children[i].append(j)
    ready = [(-txs[j].fee, j) for j in range(len(txs)) if not waiting[j]]
    heapq.heapify(ready)
    out = []
    while ready:
        _, i = heapq.heappop(ready)
        out.append(txs[i])
        for j in children[i]:
            waiting[j] -= 1
            if not waiting[j]:
                heapq.heappush(ready, (-txs[j].fee, j))
    return out


def evictable(txs: list[Tx]) -> Tx | None:
    """The cheapest transaction nothing after it depends on (the newest of equals), so dropping it can't make
    another one in the pool invalid."""
    depended = set().union(*_parents(txs)) if txs else set()
    free = [(tx.fee, -j) for j, tx in enumerate(txs) if j not in depended]
    return txs[-min(free)[1]] if free else None
