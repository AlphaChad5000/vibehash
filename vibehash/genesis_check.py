"""`vibehash genesis check FILE`: a genesis file held against what a network meant to carry value needs.

The list is HANDOFF.md's mainnet checklist ("Before VIBE is a real, listed currency") and DESIGN §9, in the parts a
genesis file decides. A chain can't change these later: there's no upgrade process for the genesis, so a network
started with one validator, a testnet's chain id or no Evo gate keeps them. Anyone can run the check on a proposed
genesis before agreeing to validate it; nothing here needs a node or a wallet.

Each finding is "ok", "warn" (a choice to make on purpose: a premine, Evo off), "fail" (a network launched with it
can be halted, taken over or confused with a testnet by one party) or "info". The command exits 1 if anything fails.

Voting: a block is final once validators holding more than two thirds of the voting stake sign it (consensus.py), so
validators holding a third or more can stop the chain by going offline, and more than two thirds can finalise blocks
on their own. `tolerates` is how many validators, the biggest first, can be offline with the chain still going.
"""

from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from urllib.parse import urlsplit

from . import economics as econ
from . import mnemonic
from .crypto import checksummed
from .ledger import Chain
from .tasks import SYN_PER_VIBE

TESTNET_IDS = {"vibehash-public-1"}  # chain ids already used by a network whose coins are test coins
MIN_VALIDATORS = 4  # the fewest equal validators that keep going with one offline (3f + 1, f = 1)


@dataclass
class Finding:
    status: str  # "ok", "warn", "fail" or "info"
    title: str
    detail: str


def _vibe(syn: int) -> str:
    return f"{(Decimal(syn) / SYN_PER_VIBE).normalize():,f} VIBE"


def tolerates(stakes: list[int]) -> int:
    """How many validators can be offline, the biggest first, while the rest hold more than two thirds."""
    total, left = sum(stakes), sum(stakes)
    for n, s in enumerate(sorted(stakes, reverse=True)):
        left -= s
        if 3 * left <= 2 * total:
            return n
    return len(stakes)  # pragma: no cover (an empty set has nothing to take away)


def _host_problem(url: str) -> str | None:
    host = urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return "is this computer"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_loopback:
        return "is this computer"
    if ip.is_private or ip.is_link_local or ip.is_unspecified:
        return "is a private address other networks can't reach"
    return None


def check(genesis: dict) -> list[Finding]:
    out: list[Finding] = []
    add = lambda status, title, detail: out.append(Finding(status, title, detail))  # noqa: E731
    try:
        chain = Chain.from_genesis(genesis)
    except Exception as e:  # anything the ledger refuses: the network couldn't start from it at all
        return [Finding("fail", "genesis", f"the ledger refuses it: {e}")]
    s = chain.state

    # The network's name. Transactions name the chain id they're signed for (not the genesis hash), so a network
    # reusing a test network's id would accept that network's signed transactions.
    cid = genesis["chain_id"]
    if cid in TESTNET_IDS or cid.startswith("vibehash-local-"):
        add("fail", "chain id", f"{cid!r} is a test network's id: transactions signed there would be valid here too")
    elif "test" in cid.lower() or "local" in cid.lower():
        add("warn", "chain id", f"{cid!r} reads like a test network's name")
    else:
        add("ok", "chain id", f"{cid!r}, genesis hash {chain.genesis_hash}")

    # Issuance.
    e = s.economics
    if e.count != econ.PAID_ROUNDS:
        add("fail", "issuance", "every round mints and the halvings count blocks, so a quiet network burns the coins "
                                "of its empty rounds; write the schedule with \"count\": \"paid_rounds\" "
                                "(`genesis create` does)")
    else:
        add("ok", "issuance", f"{_vibe(e.initial_subsidy)} per round that pays for work, halving every "
                              f"{e.halving_interval:,} such rounds, {_vibe(e.total_issuance())} in all")
    if not e.max_supply:
        add("fail", "supply cap", "none: the schedule's numbers are the only limit on coins ever created")
    else:
        add("ok", "supply cap", f"{_vibe(e.max_supply)}, genesis included")

    # Coins handed out at the start that someone can spend or withdraw: a premine. Bootstrap stake isn't one (it
    # votes, but can never be withdrawn or spent).
    alloc = sum(int(v) for v in genesis.get("allocations", {}).values())
    staked = sum(int(v) for v in genesis.get("stakes", {}).values())
    premine = alloc + staked
    if premine:
        share = f" ({Fraction(premine, e.max_supply) * 100:.3g}% of the cap)" if e.max_supply else ""
        holders = len(set(genesis.get("allocations", {})) | set(genesis.get("stakes", {})))
        add("warn", "premine", f"{_vibe(premine)}{share} to {holders} address{'es' if holders != 1 else ''} at the "
                               "start: it weakens \"nobody controls it\" and the legal position (HANDOFF's mainnet "
                               "checklist), and should be a decision made on purpose")
    else:
        add("ok", "premine", "none: every spendable coin is earned by work")
    if not s.earn_in:
        add("warn", "earn-in", "off: workers must be handed stake, so whoever holds coins at the start picks who mines")
    else:
        add("ok", "earn-in", "on: newcomers start with nothing and earn their stake")

    # Who runs it.
    voting = s.validator_set()
    if not s.validators:
        add("fail", "validators", "none: one node seals every block alone and can halt or censor the chain")
    else:
        stakes = list(voting.values())
        total = sum(stakes)
        top = max(stakes) if stakes else 0
        n_off = tolerates(stakes) if stakes else 0
        line = (f"{len(voting)} with a vote ({len(s.validators)} listed), {_vibe(total)} of voting stake; the biggest "
                f"holds {Fraction(top, total or 1) * 100:.3g}%; the chain keeps going with "
                f"{n_off} of them offline")
        if len(voting) < MIN_VALIDATORS or n_off == 0 or 3 * top >= total:
            add("fail", "validators", line + f". It needs at least {MIN_VALIDATORS}, none holding a third or more, "
                                             "so no single one can halt it")
        else:
            add("ok", "validators", line)
        hosts: dict[str, list[str]] = {}
        for addr, url in sorted(s.validators.items()):
            hosts.setdefault((urlsplit(url).hostname or "").lower(), []).append(addr)
            problem = _host_problem(url)
            if problem:
                add("fail", "validator address", f"{checksummed(addr)}'s {url} {problem}")
            elif urlsplit(url).scheme != "https":
                add("warn", "validator address", f"{checksummed(addr)}'s {url} isn't HTTPS")
        for host, addrs in sorted(hosts.items()):
            if len(addrs) > 1:
                add("fail", "validator hosts", f"{len(addrs)} validators share {host}: one machine, or one operator")
        boot = sum(s.locked.get(a, 0) for a in voting)
        add("info", "validator stake", f"{_vibe(boot)} of the voting stake is bootstrap stake (never withdrawable); "
                                       f"{_vibe(total - boot)} is ordinary stake")

    # What can't be added later.
    if s.es:
        add("ok", "Evo Hash", "on (\"es\": true): Evo jobs can run")
    else:
        add("warn", "Evo Hash", "off: Evo jobs are refused, and there's no upgrade process to turn them on later")

    # Keys made from recovery words. The path is the wallet's, not the genesis's, but a mainnet launch has to settle
    # it: with the testnet coin type, the same 24 words give the same key on every network.
    coin = mnemonic.PATH.split("/")[2]
    if coin == "1'":
        add("warn", "recovery words", f"wallets derive keys at {mnemonic.PATH}, SLIP-0044's coin type 1 (\"testnet, "
                                      "all coins\"): the same words give the same key here and on every testnet")
    else:
        add("ok", "recovery words", f"wallets derive keys at {mnemonic.PATH}")
    return out


def to_json(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]


def failed(findings: list[Finding]) -> bool:
    return any(f.status == "fail" for f in findings)
