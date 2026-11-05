"""How much VIBE exists and where it is: what a node serves at GET /supply, /supply/total and /supply/circulating, for
listing sites and anyone checking issuance (EXCHANGES.md).

Every number is read from the chain's State, which every node derives from the blocks alone (ledger.apply_block), so
any honest node gives the same numbers at the same height, and `vibehash chain verify` can check them. Amounts are
whole syn (1 VIBE = 10^8 syn), and each one is also given as an exact decimal VIBE string.

Where every coin is, at the chain's tip (these add up to `total`, State.total_supply):
    circulating   balances at real addresses: what their keys can spend
    unspendable   balances at recipients no key has (a transfer to a mistyped `vibe1...` the ledger took, before
                  relay policy refused them; DESIGN §10): counted, but nobody can ever move them
    staked        stake, `bootstrap` of it the genesis gave validators (it votes, can never be unstaked or spent, and
                  is burned when its validator leaves)
    unbonding     unstaked coins waiting out their 20 rounds (still slashable)
    escrow        job budgets not yet paid out
How it came to be (`total` = genesis + minted - burned, State.expected_supply):
    genesis       allocations, stakes and bootstrap stake in the genesis block
    minted        every subsidy so far
    burned        destroyed: the burned half of slashes, each round's pool nobody earned (on chains that count blocks,
                  that's every empty round's whole subsidy), and bootstrap stake whose validator left
"""
from __future__ import annotations

from decimal import Decimal

from . import economics as econ
from .crypto import is_canonical_address
from .tasks import SYN_PER_VIBE

AMOUNTS = ("total", "circulating", "unspendable", "staked", "bootstrap", "unbonding", "escrow", "genesis", "minted",
           "burned", "max_supply", "schedule_issuance", "subsidy")


def vibe(syn: int) -> str:
    """Exact decimal VIBE, with no exponent, thousands separator or trailing zeros: 150000000 -> '1.5', 0 -> '0'. What
    /supply/total and /supply/circulating answer with, as CoinGecko and CoinMarketCap ask."""
    text = format(Decimal(int(syn)).scaleb(-8), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def halving(s, height: int) -> dict:
    """Where the next halving is, on the clock the schedule counts (economics.Economics.count): rounds that paid for
    work (State.paid_rounds), or block heights. `next_at` is the count it comes at, `to_go` how many more rounds that
    pay for work (or blocks) mint today's subsidy first; both None once the subsidy has halved to nothing."""
    e = s.economics
    counted = s.paid_rounds if e.count == econ.PAID_ROUNDS else height + 1  # what the next block's subsidy reads
    done = counted // e.halving_interval
    over = done >= 64 or not e.initial_subsidy >> done  # (Economics.subsidy pays nothing from 64 halvings on)
    next_at = None if over else (done + 1) * e.halving_interval
    return {"counts": e.count, "interval": e.halving_interval, "halvings": done, "next_at": next_at,
            "to_go": None if over else next_at - counted,
            "subsidy_after": None if over else (0 if done + 1 >= 64 else e.initial_subsidy >> (done + 1))}


def supply(s, height: int) -> dict:
    """/supply for State `s` at block `height` (the tip it's the state after)."""
    e = s.economics
    spendable = sum(v for a, v in s.balances.items() if is_canonical_address(a))
    amounts = {
        "total": s.total_supply(),
        "circulating": spendable,
        "unspendable": sum(s.balances.values()) - spendable,
        "staked": sum(s.stakes.values()),
        "bootstrap": sum(s.locked.values()),
        "unbonding": sum(s.unbonding_total(a) for a in s.unbonding),
        "escrow": sum(s.escrow.values()),
        "genesis": s.genesis, "minted": s.minted, "burned": s.burned,
        "max_supply": e.max_supply or None,  # the cap on everything ever created; None: the schedule has none
        "schedule_issuance": e.total_issuance(),  # every subsidy the schedule will ever pay, before the cap
        "subsidy": s.subsidy(height + 1),  # what the next round that mints mints
    }
    return {"height": height, "unit": "syn", "syn_per_vibe": SYN_PER_VIBE, **amounts,
            "vibe": {k: vibe(v) for k, v in amounts.items() if v is not None},
            "economics": e.to_dict(),
            # rounds that paid for work so far: the halving clock, on chains whose schedule counts them
            "paid_rounds": s.paid_rounds if e.count == econ.PAID_ROUNDS else None,
            "halving": halving(s, height)}
