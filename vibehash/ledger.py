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


class InvalidBlock(Exception):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


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
        return cls(str(d["kind"]), str(d["sender"]), str(d["recipient"]), int(d["amount"]), int(d["nonce"]),
                   str(d["memo"]), bytes.fromhex(d["pubkey"]), bytes.fromhex(d["signature"]), str(d["chain_id"]),
                   int(d.get("fee", 0)))


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
        return cls(str(d["job_id"]), int(d["accepted_units"]),
                   {str(k): int(v) for k, v in d["worker_flops"].items()},
                   {str(k): int(v) for k, v in d["verifier_flops"].items()},
                   [[str(o), None if c is None else str(c), str(r)] for o, c, r in d["slashes"]],
                   [str(a) for a in d["accepted"]], [str(a) for a in d["missed"]],
                   str(d["params_hash"]), str(d["proposer"]), str(d.get("record_hash", "")))

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

    def header(self) -> bytes:
        tx_root = merkle_root([tx.txid for tx in self.txs])
        summary_hash = self.summary.hash() if self.summary else h256(b"no-summary")
        return h256(self.chain_id.encode(), self.height.to_bytes(8, "big"), self.prev_hash,
                    self.timestamp.to_bytes(8, "big"), tx_root, summary_hash, self.proposer_pubkey)

    @property
    def hash(self) -> bytes:
        return h256(self.header(), self.signature)

    def sign(self, keys: KeyPair) -> None:
        self.proposer_pubkey = keys.public
        self.signature = keys.sign(self.header())

    def to_dict(self) -> dict:
        return {"height": self.height, "prev_hash": self.prev_hash.hex(), "timestamp": self.timestamp,
                "summary": self.summary.to_dict() if self.summary else None,
                "txs": [t.to_dict() for t in self.txs], "chain_id": self.chain_id,
                "proposer_pubkey": self.proposer_pubkey.hex(), "signature": self.signature.hex()}

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        return cls(int(d["height"]), bytes.fromhex(d["prev_hash"]), int(d["timestamp"]),
                   RoundSummary.from_dict(d["summary"]) if d["summary"] else None,
                   [Tx.from_dict(t) for t in d["txs"]], str(d["chain_id"]),
                   bytes.fromhex(d["proposer_pubkey"]), bytes.fromhex(d["signature"]))


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

def _slash_txs(s: State, summary: RoundSummary, chain_id: str) -> list[Tx]:
    txs, seen = [], set()
    for offender, catcher, reason in summary.slashes:
        if offender in seen:
            continue  # at most one slash per offender per round
        seen.add(offender)
        bounty, burn = econ.slash_amounts(s.stake(offender) + s.unbonding_total(offender))
        if catcher is None or catcher == offender:
            bounty, burn = 0, burn + bounty
        if bounty:
            txs.append(Tx("slash", offender, catcher, bounty, memo=reason, chain_id=chain_id))
        if burn:
            txs.append(Tx("slash", offender, BURN, burn, memo=reason, chain_id=chain_id))
    return txs


def _reward_txs(s: State, summary: RoundSummary, height: int, chain_id: str) -> list[Tx]:
    offenders = summary.offenders
    fee = 0
    if summary.job_id in s.jobs:
        job = s.job(summary.job_id)
        fee = min(s.escrow.get(summary.job_id, 0), job.fee_per_round * summary.accepted_units // job.units_per_round)
    pool = s.subsidy(height) + fee
    reps = {a: s.rep(a) for a in set(summary.worker_flops) | set(summary.verifier_flops)}
    payout = econ.compute_payout(
        pool,
        {a: f for a, f in summary.worker_flops.items() if a not in offenders},
        {a: f for a, f in summary.verifier_flops.items() if a not in offenders},
        reps,
    )
    txs = [Tx("fee_release", SYSTEM, SYSTEM, fee, memo=summary.job_id, chain_id=chain_id)] if fee else []
    txs += [Tx("reward", SYSTEM, a, v, memo="work", chain_id=chain_id) for a, v in sorted(payout.worker_pay.items())]
    txs += [Tx("reward", SYSTEM, a, v, memo="verify", chain_id=chain_id) for a, v in sorted(payout.verifier_pay.items())]
    if payout.proposer_pay:
        txs.append(Tx("reward", SYSTEM, summary.proposer, payout.proposer_pay, memo="propose", chain_id=chain_id))
    return txs


def _check_summary(s: State, summary: RoundSummary, proposer: str) -> None:
    if summary.proposer != proposer:
        raise InvalidBlock("summary names a different proposer")
    paid = set(summary.worker_flops) | set(summary.verifier_flops) | set(summary.accepted)
    if any(s.stake(a) < econ.MIN_STAKE for a in paid):
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


def _apply_user_tx(s: State, tx: Tx, height: int, chain_id: str) -> None:
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
        s.stakes[tx.sender] -= tx.amount
        s.unbonding.setdefault(tx.sender, []).append([tx.amount, height + UNBONDING_ROUNDS])
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
        s._credit(tx.sender, s.escrow.get(tx.memo, 0))
        s.escrow[tx.memo] = 0
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
            if len(s.validators) == 1:
                raise InvalidBlock("the last validator can't leave: nobody would be left to agree on blocks")
            del s.validators[tx.sender]


def valid_endpoint(url: str) -> bool:
    scheme, sep, rest = url.partition("://")
    return scheme in ("http", "https") and bool(sep) and 0 < len(rest) <= 200 and not any(c.isspace() for c in rest)


def _pay_fees(s: State, user_txs: list[Tx], proposer: str) -> None:
    """Fees go to the block's proposer after all of its user txs, so none of them can spend fees from the same
    block. build_block needn't do this: nothing it derives afterwards depends on balances."""
    total = sum(tx.fee for tx in user_txs)
    if total:
        s._credit(proposer, total)


def _apply_system_tx(s: State, tx: Tx) -> int:
    """Returns the amount added to (fee_release) or drawn from (reward, as negative) the pool."""
    if tx.kind == "slash":
        take = min(tx.amount, s.stake(tx.sender))
        s.stakes[tx.sender] = s.stake(tx.sender) - take
        rest = tx.amount - take
        for entry in s.unbonding.get(tx.sender, []):
            t = min(rest, entry[0])
            entry[0] -= t
            rest -= t
        if rest:
            raise InvalidBlock("slash exceeds stake")
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
        s._credit(tx.recipient, tx.amount)
        return -tx.amount
    raise InvalidBlock(f"unknown system tx {tx.kind}")


def _release_unbonded(s: State, height: int) -> None:
    for addr, entries in s.unbonding.items():
        for entry in entries:
            if entry[1] <= height and entry[0]:
                s._credit(addr, entry[0])
                entry[0] = 0
        entries[:] = [e for e in entries if e[0]]


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
        for tx in block.txs:
            if tx.kind != "genesis" or tx.amount < 0:
                raise InvalidBlock("genesis block may only contain genesis allocations")
            if tx.memo.startswith(VALIDATOR):
                s.validators[tx.recipient] = tx.memo[len(VALIDATOR):]
            elif tx.memo == "stake":
                s.stakes[tx.recipient] = s.stake(tx.recipient) + tx.amount
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
    if block.timestamp < prev.timestamp:
        raise InvalidBlock("timestamp went backwards")
    if not verify(block.proposer_pubkey, block.header(), block.signature):
        raise InvalidBlock("bad proposer signature")
    proposer = address_of(block.proposer_pubkey)
    if s.stake(proposer) < econ.MIN_STAKE:
        raise InvalidBlock("proposer is not staked")
    if s.validators and proposer not in s.validator_set():
        raise InvalidBlock("proposer is not a validator")

    _release_unbonded(s, block.height)
    summary = block.summary
    if summary is not None:
        _check_summary(s, summary, proposer)
        pre = _slash_txs(s, summary, chain_id)
    else:
        pre = []
    for tx in pre:
        _apply_system_tx(s, tx)

    user = [tx for tx in block.txs if tx.kind in USER_KINDS]
    for tx in user:
        _apply_user_tx(s, tx, block.height, chain_id)
    _pay_fees(s, user, proposer)

    post = _reward_txs(s, summary, block.height, chain_id) if summary is not None else []
    pool = subsidy = s.subsidy(block.height) if summary is not None else 0
    for tx in post:
        pool += _apply_system_tx(s, tx)

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
                 validators: dict[str, str] | None = None):
        self.chain_id = chain_id
        txs = [Tx("genesis", SYSTEM, a, v, chain_id=chain_id) for a, v in sorted(allocations.items())]
        txs += [Tx("genesis", SYSTEM, a, v, memo="stake", chain_id=chain_id) for a, v in sorted((stakes or {}).items())]
        txs += _economics_tx(economics, chain_id)
        txs += [Tx("genesis", SYSTEM, a, 0, memo=VALIDATOR + url, chain_id=chain_id) for a, url in sorted((validators or {}).items())]
        genesis = Block(0, b"\x00" * 32, genesis_time, None, txs, chain_id)
        self.state = apply_block(State(), genesis, chain_id, None)
        self.blocks: list[Block] = [genesis]

    @classmethod
    def from_genesis(cls, g: dict) -> "Chain":
        return cls({k: int(v) for k, v in g["allocations"].items()}, {k: int(v) for k, v in g.get("stakes", {}).items()},
                   g["chain_id"], int(g.get("timestamp", 0)), econ.Economics.from_genesis(g),
                   {k: str(v) for k, v in g.get("validators", {}).items()})

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
        """Raises InvalidBlock if `tx` can't be applied after `pending`. Used to admit txs to a mempool."""
        s = copy.deepcopy(self.state)
        for p in [*pending, tx]:
            _apply_user_tx(s, p, self.tip.height + 1, self.chain_id)

    def build_block(self, summary: RoundSummary | None, user_txs: list[Tx], timestamp: int) -> tuple[Block, list[Tx]]:
        """Assemble the next block: derived slashes, the user txs that are still valid, derived rewards.

        Returns (unsigned block, dropped txs). The caller signs it and calls append().
        """
        height = self.tip.height + 1
        s = copy.deepcopy(self.state)
        _release_unbonded(s, height)
        pre = _slash_txs(s, summary, self.chain_id) if summary else []
        for tx in pre:
            _apply_system_tx(s, tx)
        kept, dropped = [], []
        for tx in user_txs:
            trial = copy.deepcopy(s)
            try:
                _apply_user_tx(trial, tx, height, self.chain_id)
            except (InvalidBlock, InvalidJob):
                dropped.append(tx)
                continue
            s = trial
            kept.append(tx)
        post = _reward_txs(s, summary, height, self.chain_id) if summary else []
        block = Block(height, self.tip.hash, max(timestamp, self.tip.timestamp), summary, pre + kept + post, self.chain_id)
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
