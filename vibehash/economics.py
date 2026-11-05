"""Emission schedule, reward split, reputation and slashing rules.

All amounts are integers in syn (1 VIBE = 10^8 syn). Rounding leftovers are
burned so supply accounting is exact.

The issuance schedule (block subsidy, halving interval, supply cap) is fixed
by each network's genesis (`Economics`, below). The testnet schedule is
deliberately fast, so a test network sees its halvings within days; the
mainnet one is Bitcoin's shape stretched to Vibe Hash's block time.

Which rounds a schedule counts is fixed there too (`Economics.count`). On a
network made since 2026-11-03 only a round that paid for work mints its
subsidy, and the halvings count those rounds (PAID_ROUNDS): a round nobody's
work is accepted in mints nothing and burns nothing, so a quiet network
delays its halvings instead of destroying the coins of every empty round, and
the coins minted add up to the schedule's total however busy it was. Genesis
files from before keep the original rule (BLOCKS): every block that seals a
round mints its subsidy, what nobody earned is burned, and the halvings count
block heights. What "paid for work" means is `ledger.RoundSummary.pays_for_work`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction

from .tasks import SYN_PER_VIBE

INITIAL_SUBSIDY = 50 * SYN_PER_VIBE  # the testnet schedule: ~21,000 VIBE in about a week of rounds
HALVING_INTERVAL = 210  # rounds
WORKER_SHARE = 0.80
VERIFIER_SHARE = 0.15
PROPOSER_SHARE = 0.05
SLASH_FRACTION = 0.5  # of stake, on fraud
BOUNTY_FRACTION = 0.5  # of the slashed amount, paid to whoever caught it
MIN_STAKE = 5 * SYN_PER_VIBE
# Joining the validator set (block version 4, ledger.ACCOUNTABILITY_VERSION) takes this much stake. A validator votes
# with all of its stake and loses all of it for double-signing, so the price of joining is what a validator puts at
# risk. The owner's call: 100 VIBE (20 times a worker's minimum) is a placeholder. Voting still needs only
# MIN_STAKE (State.validator_set), so validators that joined before version 4, like the public node with 10 VIBE of
# bootstrap stake, keep their vote.
VALIDATOR_STAKE = 100 * SYN_PER_VIBE
# A validator caught double-signing loses all of its stake. Whoever reports it gets this fraction (1/N) of the
# withdrawable part, which pays for reporting it; bootstrap stake is only ever burned (double_sign_amounts).
DOUBLE_SIGN_BOUNTY = 10

REP_START = 0.8
REP_MIN = 0.5
REP_MAX = 1.0
REP_GAIN_ACCEPTED = 0.02
REP_LOSS_MISSED = 0.05

# What a schedule counts (Economics.count), as genesis files name it.
BLOCKS = "blocks"  # the original rule: every block sealing a round mints, and the halvings count heights
PAID_ROUNDS = "paid_rounds"  # only a round that paid for work mints, and the halvings count those rounds


@dataclass(frozen=True)
class Economics:
    """A network's issuance schedule. The genesis block commits to it, so two nodes that disagree about it
    never share a genesis hash and can't mistake each other's chains for their own."""
    initial_subsidy: int  # syn per round that mints
    halving_interval: int  # rounds on the halving clock `count` names: blocks, or rounds that paid for work
    max_supply: int = 0  # cap on all coins ever created (genesis allocations + subsidies), in syn; 0: none
    # Which rounds mint and move the halving clock: BLOCKS (absent from the genesis, as on every network before
    # 2026-11-03) or PAID_ROUNDS. State.subsidy and State.round_subsidy apply it.
    count: str = BLOCKS

    def __post_init__(self):
        if not (isinstance(self.initial_subsidy, int) and isinstance(self.halving_interval, int)
                and isinstance(self.max_supply, int)):
            raise ValueError("issuance parameters are whole numbers of syn and blocks")
        if self.initial_subsidy < 0 or self.halving_interval < 1 or self.max_supply < 0:
            raise ValueError("issuance needs a subsidy >= 0, a halving interval >= 1 block and a cap >= 0")
        if self.count not in (BLOCKS, PAID_ROUNDS):
            raise ValueError(f"economics count is {BLOCKS!r} (every block, the default) or {PAID_ROUNDS!r}")

    def subsidy(self, counted: int, created: int = 0) -> int:
        """A minting round's subsidy once `counted` rounds are on the halving clock (the block's height, or with
        PAID_ROUNDS the rounds before it that paid for work: State.subsidy passes the right one), when `created`
        coins exist so far. It never takes supply past the cap."""
        halvings = counted // self.halving_interval
        amount = 0 if halvings >= 64 else self.initial_subsidy >> halvings
        return max(0, min(amount, self.max_supply - created)) if self.max_supply else amount

    def total_issuance(self) -> int:
        """Every subsidy the schedule will ever pay, before the cap. With PAID_ROUNDS that's what rounds that pay for
        work mint between them; with BLOCKS empty rounds mint their share too, and it's burned."""
        return sum((self.initial_subsidy >> k) * self.halving_interval for k in range(64))

    @property
    def name(self) -> str:
        """The preset whose numbers these are, whichever rounds they count; "custom" for any others."""
        numbers = replace(self, count=BLOCKS)
        return next((n for n, e in SCHEDULES.items() if e == numbers), "custom")

    def consensus_dict(self) -> dict:
        d = {"initial_subsidy": self.initial_subsidy, "halving_interval": self.halving_interval, "max_supply": self.max_supply}
        if self.count != BLOCKS:  # only then, so a genesis from before the field commits to what it always did
            d["count"] = self.count
        return d

    def to_dict(self) -> dict:
        return {"name": self.name, **self.consensus_dict()}

    @classmethod
    def from_dict(cls, d) -> "Economics":
        """A schedule as genesis files write it: a preset's name, which is the original rule (BLOCKS) as it always
        was, or the numbers with an optional "count" (any "name" is just a label)."""
        if isinstance(d, str):
            if d not in SCHEDULES:
                raise ValueError(f"unknown economics {d!r}; choose one of {', '.join(SCHEDULES)}")
            return SCHEDULES[d]
        if not isinstance(d, dict):
            raise ValueError("economics must be a preset name or an object of numbers")
        try:
            return cls(d["initial_subsidy"], d["halving_interval"], d.get("max_supply", 0), d.get("count", BLOCKS))
        except KeyError as e:
            raise ValueError(f"economics is missing {e}") from None

    @classmethod
    def from_genesis(cls, genesis: dict) -> "Economics | None":
        """None for genesis files from before schedules were configurable: they keep the testnet schedule."""
        return None if genesis.get("economics") is None else cls.from_dict(genesis["economics"])


TESTNET = Economics(INITIAL_SUBSIDY, HALVING_INTERVAL)
# Mainnet: halving about every 4 years at the node's default 38 s round (register 3 + compute 20 + reveal 5 +
# verify 10): 4 * 365.25 days / 38 s = 3.32M rounds. 3.125 VIBE a round then issues 20.75M VIBE in all, which
# leaves room for up to 250,000 VIBE of genesis allocations under the 21M cap.
MAINNET_BLOCK_S = 38
MAINNET = Economics(3_125 * SYN_PER_VIBE // 1000, 3_320_000, 21_000_000 * SYN_PER_VIBE)
SCHEDULES = {"testnet": TESTNET, "mainnet": MAINNET}  # what a genesis naming a preset gets: the original rule
# What a new network gets (`genesis create`, a new `vibehash testnet` folder): a preset's numbers, written out with
# "count": "paid_rounds". With it the mainnet halves every 3.32M rounds that paid for work: about 4 years if every
# 38 s round is worked, longer if not, and 20.75M VIBE minted in all whatever the pace.
NEW_SCHEDULES = {n: replace(e, count=PAID_ROUNDS) for n, e in SCHEDULES.items()}


def block_subsidy(height: int, economics: Economics = TESTNET, created: int = 0) -> int:
    return economics.subsidy(height, created)


def split_pro_rata(total: int, weights: dict[str, float], exact: bool = False) -> dict[str, int]:
    """Integer split proportional to weights (largest-remainder). Zero weights get nothing.

    Sums to exactly `total` when any weight is positive, else returns {}. That's guaranteed with `exact` (v2 blocks),
    which works the shares out in fractions. In floats a share can round up past a whole number (for totals past
    about 2**53 syn); the shares then add up to more than `total`, and the negative leftover hands all but the last
    few one more syn ([:-1]). Older blocks keep the float split, so their rewards never change.
    """
    weights = {k: w for k, w in weights.items() if w > 0}
    wsum = sum(weights.values())
    if total <= 0 or wsum <= 0:
        return {}
    if exact:
        wsum = sum(Fraction(w) for w in weights.values())  # a float converts to a Fraction exactly
        share = {k: total * Fraction(w) / wsum for k, w in weights.items()}
        out = {k: math.floor(v) for k, v in share.items()}
    else:
        share = {k: total * w / wsum for k, w in weights.items()}
        out = {k: int(v) for k, v in share.items()}
    leftover = total - sum(out.values())
    for k in sorted(share, key=lambda k: (share[k] - out[k], k), reverse=True)[:leftover]:
        out[k] += 1
    return {k: v for k, v in out.items() if v > 0}


@dataclass
class RoundPayout:
    pool: int
    worker_pay: dict[str, int]
    verifier_pay: dict[str, int]
    proposer_pay: int
    burned: int


def compute_payout(
    pool: int,
    worker_flops: dict[str, int],
    verifier_flops: dict[str, int],
    reputation: dict[str, float],
    exact: bool = False,
) -> RoundPayout:
    """Split a round's pool. Unclaimed shares are burned, never handed to the proposer. `exact` (v2 blocks): no
    rounding can pay out more than the pool (split_pro_rata); the three budgets are exact too, where in floats they
    could add up to more than a pool past 2**53 syn."""
    if exact:
        worker_budget, verifier_budget, proposer_budget = (
            math.floor(pool * Fraction(str(share))) for share in (WORKER_SHARE, VERIFIER_SHARE, PROPOSER_SHARE))
    else:
        worker_budget = int(pool * WORKER_SHARE)
        verifier_budget = int(pool * VERIFIER_SHARE)
        proposer_budget = int(pool * PROPOSER_SHARE)

    def weights(flops: dict[str, int]) -> dict[str, float]:
        return {a: f * reputation.get(a, REP_START) for a, f in flops.items()}

    worker_pay = split_pro_rata(worker_budget, weights(worker_flops), exact)
    verifier_pay = split_pro_rata(verifier_budget, weights(verifier_flops), exact)
    # A proposer only earns if the round produced verified work.
    proposer_pay = proposer_budget if worker_pay else 0
    paid = sum(worker_pay.values()) + sum(verifier_pay.values()) + proposer_pay
    return RoundPayout(pool, worker_pay, verifier_pay, proposer_pay, pool - paid)


def slash_amounts(stake: int) -> tuple[int, int]:
    """Returns (bounty to the catcher, amount burned)."""
    slashed = int(stake * SLASH_FRACTION)
    bounty = int(slashed * BOUNTY_FRACTION)
    return bounty, slashed - bounty


def double_sign_amounts(stake: int, withdrawable: int) -> tuple[int, int]:
    """A double-signer's punishment (block version 4): all of `stake` (its stake and unbonding coins). Returns (bounty
    to whoever reported it, amount burned). The bounty comes only out of `withdrawable` (stake that isn't bootstrap
    stake, plus unbonding coins), so bootstrap stake never becomes anyone's spendable coins."""
    bounty = withdrawable // DOUBLE_SIGN_BOUNTY
    return bounty, stake - bounty


def update_reputation(rep: float, accepted: bool, missed: bool, fraud: bool) -> float:
    """Per round, not per unit: otherwise big rigs (more units) would max out reputation sooner
    and earn a better rate per FLOP than small ones."""
    if fraud:
        return REP_MIN
    rep += REP_GAIN_ACCEPTED * accepted - REP_LOSS_MISSED * missed
    return min(REP_MAX, max(REP_MIN, rep))
