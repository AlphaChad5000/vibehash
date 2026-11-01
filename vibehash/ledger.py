"""Account-based ledger: signed transactions, hash-linked signed blocks.

User transactions (transfer, stake, unstake, post_job, cancel_job, validator)
need an Ed25519 signature over a payload that includes the chain id, plus the
sender's next nonce. They may pay a fee to the block's proposer (fees.py).

Chains whose genesis (or a later `validator` transaction) names validators are
run by them: only a validator may propose a block, and a block is final once
validators holding more than two thirds of the stake have signed it
(consensus.py). A chain without validators is sealed by one node, as before.

System transactions (slash, fee_release, reward) are never trusted from the
proposer. Each block carries a RoundSummary (who did how much verified work,
who cheated) and every node re-derives the exact system transactions from it
and the current state. A block whose system transactions differ is invalid, so
a proposer cannot pay itself extra, skip a slash, or overcharge a requester.

Blocks carry a version (VERSION). A rule change applies only to blocks of the
version that brought it and later, so every block already on a chain still
validates from genesis exactly as it did, with the same derived transactions
and the same hash.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

from . import economics as econ
from .crypto import KeyPair, address_of, h256, merkle_root, verify
from .tasks import InvalidJob, TrainingJob

DEFAULT_CHAIN_ID = "vibehash-testnet-1"
USER_KINDS = {"transfer", "stake", "unstake", "post_job", "cancel_job", "validator"}
SYSTEM = "system"
BURN = "burn"
UNBONDING_ROUNDS = 20  # unstaked coins stay slashable this long
VALIDATOR = "validator "  # genesis memo naming a validator: "validator <url>"
ECONOMICS_MEMO = "economics:"  # a genesis tx with this memo fixes the network's issuance schedule
EARN_IN = "earn-in"  # genesis memo: workers may start with no stake and earn it (see State.earn_in)
BOOTSTRAP = "bootstrap"  # genesis memo: stake that votes but can never be withdrawn (see State.locked)
# The block version build_block makes. A chain's versions never go down, and a node refuses versions it doesn't know.
#   1: the original rules (0.5.1 and before).
#   2: a txid appears at most once (crypto.merkle_root pads odd levels by repeating the last node, so a list with its
#      last tx repeated has the same root); user txs fit in MAX_BLOCK_TX_BYTES; a slash takes bootstrap stake last and
#      burns it, never pays it as bounty; coins finishing unbonding are released after the round's slashes; a
#      cancel_job leaves the fee its job owes the block's own round; a proposer slashed in its own round isn't paid for
#      proposing; the last voting validator can't unstake or leave; payouts are split in exact fractions.
VERSION = 2
# User txs per block, as canonical JSON of to_dict(). A leader proposes its block and the round's record in one JSON
# message, which nodes take up to 4 MiB of (node.MAX_JSON); 1 MiB leaves the rest for the record. That's about 2,400
# plain transfers (430 bytes), or 64 of the largest tx a node relays (MAX_TX_BYTES).
MAX_BLOCK_TX_BYTES = 2**20
# Relay policy (Chain.check_tx), on top of what makes a tx valid. Blocks don't enforce it; it keeps what a node holds in
# its mempool, and so the blocks it builds, to txs of a sensible size.
MAX_NOTE_BYTES = 256  # a transfer/stake/unstake memo does nothing on chain; it's a note (a payment reference, an
# invoice id), and 256 bytes keeps such a tx under 1 KB.
# Whole tx. The largest real one is a post_job: a spec of up to 4 KiB (tasks.MAX_SPEC_BYTES), which JSON escaping at
# most triples (a 2-byte é is the 6 characters \u00e9), plus about 400 bytes of other fields. Nothing else bounds a
# tx's recipient, or the amount on a cancel_job, which the ledger ignores.
MAX_TX_BYTES = 16 * 1024


class InvalidBlock(Exception):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# --- reading JSON from the network ------------------------------------------------
# Every from_dict reads messages anyone can send. A malformed one raises ValueError and nothing else: before, an
# `Infinity` amount or a height of 2**64 got as far as int() or to_bytes() and raised OverflowError, which callers
# didn't expect. Types are checked, not converted, and a negative or too-big number is refused only where no valid
# block or tx could carry one, so everything to_dict writes reads back exactly as before.

_MISSING = object()


def _get(d, key: str, kind: type, default=_MISSING):
    if not isinstance(d, dict):
        raise ValueError(f"expected an object holding {key!r}")
    v = d.get(key, default)
    if v is _MISSING:
        raise ValueError(f"missing {key!r}")
    if not isinstance(v, kind) or (kind is int and isinstance(v, bool)):
        raise ValueError(f"{key!r} has the wrong type")
    return v


def _uint(d, key: str, default=_MISSING, below: int | None = None, least: int = 0) -> int:
    """Amounts, nonces, flops: never negative in a valid block. `below`: what to_bytes(8) in a header can hold."""
    v = _get(d, key, int, default)
    if v < least or (below is not None and v >= below):
        raise ValueError(f"{key!r} out of range")
    return v


def _hex(d, key: str) -> bytes:
    return bytes.fromhex(_get(d, key, str))


def _strs(d, key: str) -> list[str]:
    v = _get(d, key, list)
    if not all(isinstance(a, str) for a in v):
        raise ValueError(f"{key!r} must be a list of strings")
    return list(v)


@dataclass(frozen=True)
class Tx:
    kind: str
    sender: str
    recipient: str
    amount: int
    nonce: int = 0
    memo: str = ""
    pubkey: bytes = b""
    signature: bytes = b""
    chain_id: str = DEFAULT_CHAIN_ID
    fee: int = 0  # paid by the sender to the block's proposer, on top of `amount`

    def payload(self) -> bytes:
        fields = [self.chain_id, self.kind, self.sender, self.recipient, self.amount, self.nonce, self.memo]
        return canonical(fields + [self.fee] if self.fee else fields)  # fee-free txs keep the payload they had before fees

    @property
    def txid(self) -> bytes:
        return h256(self.payload(), self.pubkey, self.signature)

    @classmethod
    def signed(cls, keys: KeyPair, kind: str, recipient: str, amount: int, nonce: int, memo: str = "",
               chain_id: str = DEFAULT_CHAIN_ID, fee: int = 0) -> "Tx":
        unsigned = cls(kind, keys.address, recipient, amount, nonce, memo, keys.public, b"", chain_id, fee)
        return cls(kind, keys.address, recipient, amount, nonce, memo, keys.public, keys.sign(unsigned.payload()), chain_id, fee)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pubkey"], d["signature"] = self.pubkey.hex(), self.signature.hex()
        if not self.fee:
            del d["fee"]  # as fee-free txs looked before fees, so older nodes and clients read them unchanged
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Tx":
        return cls(_get(d, "kind", str), _get(d, "sender", str), _get(d, "recipient", str), _uint(d, "amount"),
                   _uint(d, "nonce"), _get(d, "memo", str), _hex(d, "pubkey"), _hex(d, "signature"),
                   _get(d, "chain_id", str), _uint(d, "fee", 0))

    def size(self) -> int:
        """Bytes it takes in a block (MAX_BLOCK_TX_BYTES)."""
        return len(canonical(self.to_dict()))


@dataclass
class RoundSummary:
    """The verifiable outcome of one round. Everything payment-related derives from this."""
    job_id: str
    accepted_units: int
    worker_flops: dict[str, int]
    verifier_flops: dict[str, int]
    slashes: list[list]  # [offender, catcher or None, reason]
    accepted: list[str]  # workers with at least one accepted unit (reputation +)
    missed: list[str]  # workers that missed a deadline (reputation -)
    params_hash: str
    proposer: str
    record_hash: str = ""  # the round's record (coordinator.py), so anyone can replay it; "" on blocks from before records

    @property
    def offenders(self) -> set[str]:
        return {s[0] for s in self.slashes}

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.record_hash:
            del d["record_hash"]  # older blocks were hashed without it, and must still validate
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoundSummary":
        def flops(key: str) -> dict[str, int]:
            v = _get(d, key, dict)
            return {k: _uint(v, k) for k in v}

        slashes = _get(d, "slashes", list)
        if not all(isinstance(s, list) and len(s) == 3 and isinstance(s[0], str) and isinstance(s[2], str)
                   and (s[1] is None or isinstance(s[1], str)) for s in slashes):
            raise ValueError("a slash is [offender, catcher or null, reason]")
        return cls(_get(d, "job_id", str), _uint(d, "accepted_units"), flops("worker_flops"), flops("verifier_flops"),
                   [list(s) for s in slashes], _strs(d, "accepted"), _strs(d, "missed"),
                   _get(d, "params_hash", str), _get(d, "proposer", str), _get(d, "record_hash", str, ""))

    def hash(self) -> bytes:
        return h256(canonical(self.to_dict()))


@dataclass
class Block:
    height: int
    prev_hash: bytes
    timestamp: int
    summary: RoundSummary | None
    txs: list[Tx]
    chain_id: str = DEFAULT_CHAIN_ID
    proposer_pubkey: bytes = b""
    signature: bytes = b""
    version: int = 1  # the rules it's made under (VERSION)

    def header(self) -> bytes:
        tx_root = merkle_root([tx.txid for tx in self.txs])
        summary_hash = self.summary.hash() if self.summary else h256(b"no-summary")
        # Version 1 blocks were hashed without it, and must keep their hashes. From 2 on the proposer signs it.
        version = [self.version.to_bytes(8, "big")] if self.version >= 2 else []
        return h256(self.chain_id.encode(), self.height.to_bytes(8, "big"), self.prev_hash,
                    self.timestamp.to_bytes(8, "big"), tx_root, summary_hash, self.proposer_pubkey, *version)

    @property
    def hash(self) -> bytes:
        return h256(self.header(), self.signature)

    def sign(self, keys: KeyPair) -> None:
        self.proposer_pubkey = keys.public
        self.signature = keys.sign(self.header())

    def to_dict(self) -> dict:
        d = {"height": self.height, "prev_hash": self.prev_hash.hex(), "timestamp": self.timestamp,
             "summary": self.summary.to_dict() if self.summary else None,
             "txs": [t.to_dict() for t in self.txs], "chain_id": self.chain_id,
             "proposer_pubkey": self.proposer_pubkey.hex(), "signature": self.signature.hex()}
        if self.version >= 2:  # version 1 blocks look as they did before versions
            d["version"] = self.version
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        summary = _get(d, "summary", (dict, type(None)))
        return cls(_uint(d, "height", below=2**64), _hex(d, "prev_hash"), _uint(d, "timestamp", below=2**64),
                   RoundSummary.from_dict(summary) if summary else None,
                   [Tx.from_dict(t) for t in _get(d, "txs", list)], _get(d, "chain_id", str),
                   _hex(d, "proposer_pubkey"), _hex(d, "signature"), _uint(d, "version", 1, below=2**64, least=1))


@dataclass
class State:
    balances: dict[str, int] = field(default_factory=dict)
    stakes: dict[str, int] = field(default_factory=dict)
    unbonding: dict[str, list[list[int]]] = field(default_factory=dict)  # addr -> [[amount, release_height]]
    nonces: dict[str, int] = field(default_factory=dict)
    escrow: dict[str, int] = field(default_factory=dict)  # job_id -> syn
    jobs: dict[str, str] = field(default_factory=dict)  # job_id -> spec json
    job_owner: dict[str, str] = field(default_factory=dict)
    reputation: dict[str, float] = field(default_factory=dict)
    validators: dict[str, str] = field(default_factory=dict)  # addr -> the address other nodes reach it at
    # Bootstrap stake: what the genesis gives its first validators so someone can vote on block one. It counts as
    # stake but can never be unstaked, and it's burned when its validator leaves, so it never becomes spendable.
    locked: dict[str, int] = field(default_factory=dict)
    # Earn-in: anyone may take work without stake, and their pay goes into their stake until it reaches the
    # minimum, so every spendable coin starts from work. Set by the genesis; older chains don't have it.
    earn_in: bool = False
    genesis: int = 0
    minted: int = 0
    burned: int = 0
    economics: econ.Economics = econ.TESTNET  # set by the genesis block

    def subsidy(self, height: int) -> int:
        """The block subsidy at `height` on this chain, before that block is applied."""
        return self.economics.subsidy(height, self.genesis + self.minted)

    def balance(self, addr: str) -> int:
        return self.balances.get(addr, 0)

    def validator_set(self) -> dict[str, int]:
        """Who votes on the next block, and with how much weight (their stake). A validator whose stake has fallen
        below the minimum stays listed but has no vote."""
        return {a: self.stake(a) for a in sorted(self.validators) if self.stake(a) >= econ.MIN_STAKE}

    def stake(self, addr: str) -> int:
        return self.stakes.get(addr, 0)

    def staked(self, addr: str) -> bool:
        """Has the minimum stake: can check others' work, propose and validate, not just earn in."""
        return self.stake(addr) >= econ.MIN_STAKE

    def unbonding_total(self, addr: str) -> int:
        return sum(a for a, _ in self.unbonding.get(addr, []))

    def rep(self, addr: str) -> float:
        return self.reputation.get(addr, econ.REP_START)

    def job(self, job_id: str) -> TrainingJob:
        return TrainingJob.from_spec_json(self.jobs[job_id], self.job_owner[job_id], 0)

    def total_supply(self) -> int:
        return (sum(self.balances.values()) + sum(self.stakes.values()) + sum(self.escrow.values())
                + sum(self.unbonding_total(a) for a in self.unbonding))

    def expected_supply(self) -> int:
        return self.genesis + self.minted - self.burned

    def _credit(self, addr: str, amount: int) -> None:
        self.balances[addr] = self.balance(addr) + amount

    def _debit(self, addr: str, amount: int) -> None:
        if amount < 0 or self.balance(addr) < amount:
            raise InvalidBlock(f"{addr} cannot spend {amount}")
        self.balances[addr] -= amount


# --- deterministic settlement ---------------------------------------------------

def _slash_txs(s: State, summary: RoundSummary, chain_id: str, version: int) -> list[Tx]:
    txs, seen = [], set()
    for offender, catcher, reason in summary.slashes:
        if offender in seen:
            continue  # at most one slash per offender per round
        seen.add(offender)
        bounty, burn = econ.slash_amounts(s.stake(offender) + s.unbonding_total(offender))
        if catcher is None or catcher == offender:
            bounty, burn = 0, burn + bounty
        elif version >= 2:
            # Bootstrap stake is taken last (_apply_system_tx) and only ever burned. Before, the bounty could come out
            # of it, so a bootstrap validator falsely accusing its own sybil turned it into spendable coins.
            free = _withdrawable(s, offender) + s.unbonding_total(offender)
            bounty, burn = min(bounty, free), burn + bounty - min(bounty, free)
        if bounty:  # first, so it comes out of withdrawable coins; the burn takes the rest
            txs.append(Tx("slash", offender, catcher, bounty, memo=reason, chain_id=chain_id))
        if burn:
            txs.append(Tx("slash", offender, BURN, burn, memo=reason, chain_id=chain_id))
    return txs


def _withdrawable(s: State, addr: str) -> int:
    """Stake its owner could unstake: all of it but bootstrap stake."""
    return max(0, s.stake(addr) - s.locked.get(addr, 0))


def _round_fee(s: State, summary: RoundSummary) -> int:
    """What the round's job pays into the pool: its fee per round for the share of units that verified, while its
    escrow lasts."""
    if summary.job_id not in s.jobs:
        return 0
    job = s.job(summary.job_id)
    return min(s.escrow.get(summary.job_id, 0), job.fee_per_round * summary.accepted_units // job.units_per_round)


def _reward_txs(s: State, summary: RoundSummary, height: int, chain_id: str, version: int,
                owed: dict[str, int]) -> list[Tx]:
    offenders = summary.offenders
    # v1 read the fee from the escrow after the block's user txs, so cancelling the job in the same block dodged it.
    fee = owed[summary.job_id] if version >= 2 else _round_fee(s, summary)
    pool = s.subsidy(height) + fee
    reps = {a: s.rep(a) for a in set(summary.worker_flops) | set(summary.verifier_flops)}
    try:
        payout = econ.compute_payout(
            pool,
            {a: f for a, f in summary.worker_flops.items() if a not in offenders},
            {a: f for a, f in summary.verifier_flops.items() if a not in offenders},
            reps,
            exact=version >= 2,
        )
    except OverflowError as e:  # flops past what a float holds: such a block never applied (it raised this instead)
        raise InvalidBlock("flops out of range") from e
    # v2: an offender proposing its own round isn't paid for it either; its share is burned, like any unclaimed one.
    proposer_pay = 0 if version >= 2 and summary.proposer in offenders else payout.proposer_pay
    txs = [Tx("fee_release", SYSTEM, SYSTEM, fee, memo=summary.job_id, chain_id=chain_id)] if fee else []
    txs += [Tx("reward", SYSTEM, a, v, memo="work", chain_id=chain_id) for a, v in sorted(payout.worker_pay.items())]
    txs += [Tx("reward", SYSTEM, a, v, memo="verify", chain_id=chain_id) for a, v in sorted(payout.verifier_pay.items())]
    if proposer_pay:
        txs.append(Tx("reward", SYSTEM, summary.proposer, proposer_pay, memo="propose", chain_id=chain_id))
    return txs


def _check_summary(s: State, summary: RoundSummary, proposer: str) -> None:
    if summary.proposer != proposer:
        raise InvalidBlock("summary names a different proposer")
    # Checkers must be staked. Workers too, unless the chain lets them earn their stake in.
    paid = set(summary.verifier_flops) | (set() if s.earn_in else set(summary.worker_flops) | set(summary.accepted))
    if any(not s.staked(a) for a in paid):
        raise InvalidBlock("rewards to an address without minimum stake")
    if any(v < 0 for v in [*summary.worker_flops.values(), *summary.verifier_flops.values()]):
        raise InvalidBlock("negative flops")
    if summary.worker_flops or summary.accepted_units:
        if summary.job_id not in s.jobs:
            raise InvalidBlock("summary references an unknown job")
        job = s.job(summary.job_id)
        if not 0 <= summary.accepted_units <= job.units_per_round:
            raise InvalidBlock("accepted units out of range")
        # Each unit can be done by at most two replicas.
        if sum(summary.worker_flops.values()) > 2 * job.units_per_round * job.unit_flops:
            raise InvalidBlock("claimed work exceeds what the round contained")


def _apply_user_tx(s: State, tx: Tx, height: int, chain_id: str, version: int,
                   owed: dict[str, int] | None = None) -> None:
    """`owed`: the fee each job owes the block's own round (v2), which a cancel_job leaves in escrow to be paid."""
    if tx.kind not in USER_KINDS:
        raise InvalidBlock(f"{tx.kind} is not a user transaction")
    if tx.chain_id != chain_id:
        raise InvalidBlock("transaction is for a different chain")
    if address_of(tx.pubkey) != tx.sender or not verify(tx.pubkey, tx.payload(), tx.signature):
        raise InvalidBlock(f"bad signature on {tx.kind}")
    if tx.nonce != s.nonces.get(tx.sender, 0):
        raise InvalidBlock("bad nonce")
    if tx.amount < 0:
        raise InvalidBlock("negative amount")
    if tx.fee < 0:
        raise InvalidBlock("negative fee")
    s.nonces[tx.sender] = tx.nonce + 1
    if tx.fee:
        s._debit(tx.sender, tx.fee)  # the proposer is paid once the block's user txs are all applied (_pay_fees)

    if tx.kind == "transfer":
        if tx.recipient in (SYSTEM, BURN) or not tx.recipient.startswith("vibe1"):
            raise InvalidBlock("bad recipient")
        s._debit(tx.sender, tx.amount)
        s._credit(tx.recipient, tx.amount)
    elif tx.kind == "stake":
        s._debit(tx.sender, tx.amount)
        s.stakes[tx.sender] = s.stake(tx.sender) + tx.amount
    elif tx.kind == "unstake":
        if s.stake(tx.sender) < tx.amount:
            raise InvalidBlock("unstake exceeds stake")
        if s.stake(tx.sender) - tx.amount < s.locked.get(tx.sender, 0):
            raise InvalidBlock("bootstrap stake can't be withdrawn")
        s.stakes[tx.sender] -= tx.amount
        s.unbonding.setdefault(tx.sender, []).append([tx.amount, height + UNBONDING_ROUNDS])
        if version >= 2 and s.validators and not s.validator_set():
            raise InvalidBlock("that would leave no validator with a vote: nobody could propose blocks")
    elif tx.kind == "post_job":
        try:
            job = TrainingJob.from_spec_json(tx.memo, tx.sender, tx.amount)
        except InvalidJob as e:
            raise InvalidBlock(str(e)) from e
        if job.job_id in s.jobs:
            raise InvalidBlock("job id already used")
        s._debit(tx.sender, tx.amount)
        s.escrow[job.job_id] = tx.amount
        s.jobs[job.job_id] = tx.memo
        s.job_owner[job.job_id] = tx.sender
    elif tx.kind == "cancel_job":
        if s.job_owner.get(tx.memo) != tx.sender:
            raise InvalidBlock("only the requester can cancel a job")
        # v2: the fee the block's own round owes stays, and is paid. Before, user txs ran before the fee was worked
        # out, so cancelling in the round's block got the round's work for free.
        keep = (owed or {}).get(tx.memo, 0) if version >= 2 else 0
        s._credit(tx.sender, s.escrow.get(tx.memo, 0) - keep)
        s.escrow[tx.memo] = keep
    elif tx.kind == "validator":
        # memo: the address other nodes reach this validator at; empty to stop validating.
        if tx.amount:
            raise InvalidBlock("a validator announcement carries no amount")
        if not s.validators:  # else the first account to announce itself would take over a one-node chain
            raise InvalidBlock("this chain is run by one node; only a chain whose genesis names validators can add more")
        if tx.memo:
            if not valid_endpoint(tx.memo):
                raise InvalidBlock("a validator's address must look like http(s)://host:port")
            if s.stake(tx.sender) < econ.MIN_STAKE:
                raise InvalidBlock("validators need at least the minimum stake")
            s.validators[tx.sender] = tx.memo
        elif tx.sender in s.validators:
            # v2 counts who votes, not who's listed: a listed validator below the minimum stake can't propose either.
            if (not s.validator_set().keys() - {tx.sender}) if version >= 2 else len(s.validators) == 1:
                raise InvalidBlock("the last validator can't leave: nobody would be left to agree on blocks")
            del s.validators[tx.sender]
            bootstrap = s.locked.pop(tx.sender, 0)  # it was only ever there to vote with
            s.stakes[tx.sender] = s.stake(tx.sender) - bootstrap
            s.burned += bootstrap


def valid_endpoint(url: str) -> bool:
    scheme, sep, rest = url.partition("://")
    return scheme in ("http", "https") and bool(sep) and 0 < len(rest) <= 200 and not any(c.isspace() for c in rest)


def _relayable(tx: Tx) -> None:
    """The relay policy (MAX_TX_BYTES): checked before check_tx replays the mempool, so an oversized tx costs a node
    almost nothing. The ledger already bounds the other memos: a post_job's is a job spec of at most
    tasks.MAX_SPEC_BYTES (TrainingJob.from_spec_json), a cancel_job's must name one of the sender's jobs, so it's a job
    id of at most 64 characters, and a validator's is empty or what valid_endpoint takes."""
    if tx.size() > MAX_TX_BYTES:
        raise InvalidBlock(f"transaction too big: nodes relay up to {MAX_TX_BYTES} bytes")
    if tx.kind in ("transfer", "stake", "unstake") and len(tx.memo.encode()) > MAX_NOTE_BYTES:
        raise InvalidBlock(f"a {tx.kind} memo is a note of at most {MAX_NOTE_BYTES} bytes")
    if tx.kind == "post_job":
        # A job that can't pay for a whole round would have its first round's workers paid less than it promised, and
        # one with no budget never runs (Coordinator.active_job_id) but keeps its spec in every node's state.
        try:
            fee_per_round = TrainingJob.from_spec_json(tx.memo, tx.sender, tx.amount).fee_per_round
        except InvalidJob as e:
            raise InvalidBlock(str(e)) from e
        if tx.amount < max(1, fee_per_round):
            raise InvalidBlock("a job's budget must pay for at least one round (its fee_per_round, and above 0)")


def _pay_fees(s: State, user_txs: list[Tx], proposer: str) -> None:
    """Fees go to the block's proposer after all of its user txs, so none of them can spend fees from the same
    block. build_block needn't do this: nothing it derives afterwards depends on balances."""
    total = sum(tx.fee for tx in user_txs)
    if total:
        s._credit(proposer, total)


def _apply_system_tx(s: State, tx: Tx, version: int) -> int:
    """Returns the amount added to (fee_release) or drawn from (reward, as negative) the pool."""
    if tx.kind == "slash":
        # v1 takes stake (bootstrap stake included), then unbonding coins. v2 takes withdrawable stake, then unbonding
        # coins, then bootstrap stake, which only a burn may take (_slash_txs keeps bounties within the rest).
        # A slash can leave no validator with the minimum stake, and so nobody able to propose. Unlike an unstake it
        # can't be refused: that would let a cheater keep its stake by being the last validator.
        locked = s.locked.get(tx.sender, 0) if version >= 2 else 0
        take = min(tx.amount, max(0, s.stake(tx.sender) - locked))
        s.stakes[tx.sender] = s.stake(tx.sender) - take
        rest = tx.amount - take
        for entry in s.unbonding.get(tx.sender, []):
            t = min(rest, entry[0])
            entry[0] -= t
            rest -= t
        if rest and locked:
            if tx.recipient != BURN:
                raise InvalidBlock("a bounty can't come from bootstrap stake")
            t = min(rest, locked)
            s.stakes[tx.sender] -= t
            s.locked[tx.sender] -= t
            rest -= t
        if rest:
            raise InvalidBlock("slash exceeds stake")
        if tx.sender in s.locked:  # v1: a slash takes withdrawable stake first, then bootstrap stake
            s.locked[tx.sender] = min(s.locked[tx.sender], s.stake(tx.sender))
        if tx.recipient == BURN:
            s.burned += tx.amount
        else:
            s._credit(tx.recipient, tx.amount)
        return 0
    if tx.kind == "fee_release":
        if s.escrow.get(tx.memo, 0) < tx.amount:
            raise InvalidBlock("fee release exceeds escrow")
        s.escrow[tx.memo] -= tx.amount
        return tx.amount
    if tx.kind == "reward":
        # Earn-in: pay tops up a worker's stake to the minimum first. (Without earn-in, everyone paid is staked.)
        into_stake = min(tx.amount, max(0, econ.MIN_STAKE - s.stake(tx.recipient))) if s.earn_in else 0
        if into_stake:
            s.stakes[tx.recipient] = s.stake(tx.recipient) + into_stake
        s._credit(tx.recipient, tx.amount - into_stake)
        return -tx.amount
    raise InvalidBlock(f"unknown system tx {tx.kind}")


def _release_unbonded(s: State, height: int) -> None:
    for addr, entries in s.unbonding.items():
        for entry in entries:
            if entry[1] <= height and entry[0]:
                s._credit(addr, entry[0])
                entry[0] = 0
        entries[:] = [e for e in entries if e[0]]


def _settle_first(s: State, summary: RoundSummary | None, height: int, chain_id: str,
                  version: int) -> tuple[list[Tx], dict[str, int]]:
    """What a block settles before its user txs: the round's slashes, and the coins whose unbonding ends at its height.
    Returns the slash txs, and the fee the round's job owes its pool (v2), fixed before any user tx can cancel the
    job. v1 released the coins first, so those finishing unbonding at that height escaped the round's slashes."""
    if version < 2:
        _release_unbonded(s, height)
    pre = _slash_txs(s, summary, chain_id, version) if summary is not None else []
    for tx in pre:
        _apply_system_tx(s, tx, version)
    if version < 2:
        return pre, {}
    _release_unbonded(s, height)
    return pre, ({summary.job_id: _round_fee(s, summary)} if summary is not None else {})


def _economics_tx(economics: econ.Economics | None, chain_id: str) -> list[Tx]:
    """The genesis tx that commits a network to its issuance schedule. Networks on the original testnet schedule
    have none, so genesis files from before schedules existed keep their genesis hash."""
    if economics is None or economics == econ.TESTNET:
        return []
    return [Tx("genesis", SYSTEM, SYSTEM, 0, memo=ECONOMICS_MEMO + canonical(economics.consensus_dict()).decode(), chain_id=chain_id)]


def _economics_of(tx: Tx) -> econ.Economics:
    try:
        if tx.amount or tx.recipient != SYSTEM:
            raise ValueError("it can't move coins")
        return econ.Economics.from_dict(json.loads(tx.memo[len(ECONOMICS_MEMO):]))
    except ValueError as e:
        raise InvalidBlock(f"bad economics in genesis: {e}") from e


def apply_block(state: State, block: Block, chain_id: str, prev: Block | None) -> State:
    """Validates `block` against `state` and returns the new state. Never mutates `state`."""
    s = copy.deepcopy(state)
    if block.chain_id != chain_id:
        raise InvalidBlock("block is for a different chain")

    if block.height == 0:
        if prev is not None:
            raise InvalidBlock("a chain has only one genesis block")
        for tx in block.txs:
            if tx.kind != "genesis" or tx.amount < 0:
                raise InvalidBlock("genesis block may only contain genesis allocations")
            if tx.memo.startswith(VALIDATOR):
                s.validators[tx.recipient] = tx.memo[len(VALIDATOR):]
            elif tx.memo == "stake":
                s.stakes[tx.recipient] = s.stake(tx.recipient) + tx.amount
            elif tx.memo == BOOTSTRAP:
                s.stakes[tx.recipient] = s.stake(tx.recipient) + tx.amount
                s.locked[tx.recipient] = s.locked.get(tx.recipient, 0) + tx.amount
            elif tx.memo == EARN_IN:
                if tx.amount or tx.recipient != SYSTEM:
                    raise InvalidBlock("the earn-in rule can't move coins")
                s.earn_in = True
            elif tx.memo.startswith(ECONOMICS_MEMO):
                s.economics = _economics_of(tx)
            else:
                s._credit(tx.recipient, tx.amount)
            s.genesis += tx.amount
        if s.economics.max_supply and s.genesis > s.economics.max_supply:
            raise InvalidBlock("genesis allocates more than the schedule's maximum supply")
        return s

    if prev is None or block.prev_hash != prev.hash or block.height != prev.height + 1:
        raise InvalidBlock("block does not extend its parent")
    parent_version = prev.version if prev.height else 1  # the genesis counts as 1
    if not parent_version <= block.version <= VERSION:
        raise InvalidBlock(f"block version {block.version} after {parent_version}; this node knows up to {VERSION}")
    if block.version >= 2:
        if len({t.txid for t in block.txs}) != len(block.txs):
            # A copy of a block with its last tx repeated has the same tx root, so the same hash and signature
            # (crypto.merkle_root). The copy never applies, but now it's refused on sight, before anything else.
            raise InvalidBlock("a transaction appears twice")
        if sum(tx.size() for tx in block.txs if tx.kind in USER_KINDS) > MAX_BLOCK_TX_BYTES:
            raise InvalidBlock(f"user transactions over {MAX_BLOCK_TX_BYTES} bytes")
    if block.timestamp < prev.timestamp:
        raise InvalidBlock("timestamp went backwards")
    if not verify(block.proposer_pubkey, block.header(), block.signature):
        raise InvalidBlock("bad proposer signature")
    proposer = address_of(block.proposer_pubkey)
    if not s.staked(proposer):
        raise InvalidBlock("proposer is not staked")
    if s.validators and proposer not in s.validator_set():
        raise InvalidBlock("proposer is not a validator")

    summary = block.summary
    if summary is not None:
        _check_summary(s, summary, proposer)
    pre, owed = _settle_first(s, summary, block.height, chain_id, block.version)

    user = [tx for tx in block.txs if tx.kind in USER_KINDS]
    for tx in user:
        _apply_user_tx(s, tx, block.height, chain_id, block.version, owed)
    _pay_fees(s, user, proposer)

    post = _reward_txs(s, summary, block.height, chain_id, block.version, owed) if summary is not None else []
    pool = subsidy = s.subsidy(block.height) if summary is not None else 0
    for tx in post:
        pool += _apply_system_tx(s, tx, block.version)

    if [t.txid for t in block.txs] != [t.txid for t in pre + user + post]:
        raise InvalidBlock("system transactions don't match the round summary")
    if pool < 0:
        raise InvalidBlock("rewards exceed subsidy + released fees")

    if summary is not None:
        s.minted += subsidy
        s.burned += pool  # unclaimed pool is burned
        accepted, missed, offenders = set(summary.accepted), set(summary.missed), summary.offenders
        for addr in accepted | missed | offenders:
            s.reputation[addr] = econ.update_reputation(s.rep(addr), addr in accepted, addr in missed, addr in offenders)
    return s


class Chain:
    def __init__(self, allocations: dict[str, int], stakes: dict[str, int] | None = None,
                 chain_id: str = DEFAULT_CHAIN_ID, genesis_time: int = 0, economics: econ.Economics | None = None,
                 validators: dict[str, str] | None = None, bootstrap: dict[str, int] | None = None, earn_in: bool = False):
        """`bootstrap`: stake for the first validators that votes but can never be withdrawn or spent. `earn_in`:
        workers may start with no stake and earn it. Genesis files without them keep their genesis hash."""
        self.chain_id = chain_id
        txs = [Tx("genesis", SYSTEM, a, v, chain_id=chain_id) for a, v in sorted(allocations.items())]
        txs += [Tx("genesis", SYSTEM, a, v, memo="stake", chain_id=chain_id) for a, v in sorted((stakes or {}).items())]
        txs += _economics_tx(economics, chain_id)
        txs += [Tx("genesis", SYSTEM, a, 0, memo=VALIDATOR + url, chain_id=chain_id) for a, url in sorted((validators or {}).items())]
        txs += [Tx("genesis", SYSTEM, a, v, memo=BOOTSTRAP, chain_id=chain_id) for a, v in sorted((bootstrap or {}).items())]
        txs += [Tx("genesis", SYSTEM, SYSTEM, 0, memo=EARN_IN, chain_id=chain_id)] if earn_in else []
        genesis = Block(0, b"\x00" * 32, genesis_time, None, txs, chain_id)
        self.state = apply_block(State(), genesis, chain_id, None)
        self.blocks: list[Block] = [genesis]

    @classmethod
    def from_genesis(cls, g: dict) -> "Chain":
        return cls({k: int(v) for k, v in g["allocations"].items()}, {k: int(v) for k, v in g.get("stakes", {}).items()},
                   g["chain_id"], int(g.get("timestamp", 0)), econ.Economics.from_genesis(g),
                   {k: str(v) for k, v in g.get("validators", {}).items()},
                   {k: int(v) for k, v in g.get("bootstrap", {}).items()}, bool(g.get("earn_in", False)))

    @property
    def tip(self) -> Block:
        return self.blocks[-1]

    @property
    def genesis_hash(self) -> str:
        """The network's identity. Clients pin it so a node can't swap in different allocations."""
        return self.blocks[0].hash.hex()

    def next_nonce(self, addr: str) -> int:
        return self.state.nonces.get(addr, 0)

    def append(self, block: Block) -> None:
        self.state = apply_block(self.state, block, self.chain_id, self.tip)
        self.blocks.append(block)

    def check_tx(self, tx: Tx, pending: list[Tx]) -> None:
        """Raises InvalidBlock if `tx` can't be applied after `pending` in the next block we'd build, or breaks the
        relay policy (_relayable). Used to admit txs to a mempool, and to keep them there after each block."""
        _relayable(tx)
        s = copy.deepcopy(self.state)
        for p in [*pending, tx]:
            _apply_user_tx(s, p, self.tip.height + 1, self.chain_id, VERSION)

    def build_block(self, summary: RoundSummary | None, user_txs: list[Tx], timestamp: int,
                    version: int = VERSION) -> tuple[Block, list[Tx]]:
        """Assemble the next block: derived slashes, the user txs that are still valid and fit, derived rewards.

        Returns (unsigned block, dropped txs): the ones that don't apply now and, once the block is full, all the rest,
        which wait for the next block. The caller signs it and calls append(). Only tests ask for an older `version`.
        """
        height = self.tip.height + 1
        s = copy.deepcopy(self.state)
        pre, owed = _settle_first(s, summary, height, self.chain_id, version)
        kept, dropped, size = [], [], 0
        for i, tx in enumerate(user_txs):
            n = tx.size()
            if version >= 2 and size + n > MAX_BLOCK_TX_BYTES:
                dropped += user_txs[i:]  # in fee order (fees.by_fee), so the cut falls at the lowest fees
                break
            trial = copy.deepcopy(s)
            try:
                _apply_user_tx(trial, tx, height, self.chain_id, version, owed)
            except (InvalidBlock, InvalidJob):
                dropped.append(tx)
                continue
            s = trial
            kept.append(tx)
            size += n
        post = _reward_txs(s, summary, height, self.chain_id, version, owed) if summary else []
        block = Block(height, self.tip.hash, max(timestamp, self.tip.timestamp), summary, pre + kept + post,
                      self.chain_id, version=version)
        return block, dropped

    def validate(self) -> bool:
        """Replays the whole chain from genesis. Detects any tampering with past blocks."""
        try:
            state = apply_block(State(), self.blocks[0], self.chain_id, None)
            for prev, block in zip(self.blocks, self.blocks[1:]):
                state = apply_block(state, block, self.chain_id, prev)
            return state.total_supply() == state.expected_supply()
        except (InvalidBlock, InvalidJob, KeyError, ValueError):
            return False
