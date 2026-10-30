"""Validators agreeing on each block: Tendermint's BFT consensus, with voting power = stake.

This follows Algorithm 1 of Buchman, Kwon and Milosevic, "The latest gossip on BFT consensus" (2018). For each height,
validators take turns proposing, in a stake-weighted order fixed by the height. A proposal is a block plus its round
record. Every validator checks it (the block applies to the chain, and its round replays from the record, see
replay.py) and then votes in two steps, prevote and precommit. A validator that sees more than two thirds of the
stake prevote for a block locks on it, and won't vote for any other block at that height unless a later round shows
the network moved on. A block is final once validators holding more than two thirds of the stake precommit it in
the same round. Those signed precommits are its commit certificate, which anyone can check against the validator set
without trusting the node that serves the block.

It's safe as long as validators holding more than two thirds of the stake follow the protocol: two different
blocks can never both be final at one height, even if the network splits or messages arrive late or out of order.
It's live once the network delivers messages in time. A proposer that's slow, offline or proposes something
invalid only costs a round, and the next one takes over.

This module doesn't know about the network, like coordinator.py. node.py feeds it messages and clock ticks, and
carries out what it asks for: broadcasting messages, building a proposal, committing a block.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

from .crypto import Draw, KeyPair, address_of, h256, verify
from .ledger import Block, canonical

log = logging.getLogger("vibehash.consensus")
MAX_ROUNDS_AHEAD = 1_000  # later rounds than this are ignored (an attacker's memory bomb, not a slow network)
# Validators' messages for later heights are kept until we catch up, so we can vote as soon as we get there: only the
# next few heights' (sync fetches the blocks further ahead, and validators re-send what's current every few seconds),
# and at most this many messages, and bytes of them, from any one validator.
FUTURE_HEIGHTS = 4
FUTURE_PER_SENDER = 64
FUTURE_BYTES = 2**26


class Step(str, Enum):
    PROPOSE = "propose"
    PREVOTE = "prevote"
    PRECOMMIT = "precommit"


@dataclass
class Timeouts:
    propose: float = 60.0  # a proposer runs a whole training round before it has something to propose
    vote: float = 3.0  # how long to wait for the rest of the votes once two thirds are in
    grow: float = 1.0  # each later round at the same height waits this much longer, so a slow network catches up
    regossip: float = 2.0  # how often our messages at this height are sent again, for peers that missed them

    def wait(self, step: Step, round_: int, propose: float | None = None) -> float:
        """`propose` replaces the propose wait when it depends on the round's job (Engine's propose_wait)."""
        base = (self.propose if propose is None else propose) if step == Step.PROPOSE else self.vote
        return base + round_ * self.grow


def quorum(power: int, total: int) -> bool:
    """More than two thirds of the stake: two such sets always share an honest validator."""
    return 3 * power > 2 * total


def proposer_order(validators: dict[str, int], height: int) -> list[str]:
    """Who proposes in each round at `height`: a stake-weighted shuffle of the validators. It depends only on the
    height, so it's known in advance and whoever made the previous block can't steer it."""
    draw, left, order = Draw(b"proposer", height.to_bytes(8, "big")), dict(validators), []
    while left:
        slot = draw.below(sum(left.values()))
        for addr in sorted(left):
            slot -= left[addr]
            if slot < 0:
                order.append(addr)
                del left[addr]
                break
    return order


# --- signed messages --------------------------------------------------------------------

def _signed_part(chain_id: str, kind: str, height: int, round_: int, block_id: str | None, valid_round: int | None = None) -> bytes:
    body = [chain_id, kind, height, round_, block_id]
    return canonical(body + [valid_round] if kind == "proposal" else body)


def vote(keys: KeyPair, chain_id: str, kind: str, height: int, round_: int, block_id: str | None) -> dict:
    """A prevote or precommit for a block (by hash), or for nothing (None)."""
    return {"type": kind, "chain_id": chain_id, "height": height, "round": round_, "block": block_id,
            "pubkey": keys.public.hex(), "sig": keys.sign(_signed_part(chain_id, kind, height, round_, block_id)).hex()}


def proposal(keys: KeyPair, chain_id: str, height: int, round_: int, valid_round: int, value: dict, block_id: str) -> dict:
    """`value` is {"block": block as a dict, "record": its round record, or None for a block without a round}.
    `valid_round` is -1 for a new block, or the round in which more than two thirds prevoted for this one."""
    return {"type": "proposal", "chain_id": chain_id, "height": height, "round": round_, "valid_round": valid_round,
            "block": value["block"], "record": value.get("record"), "pubkey": keys.public.hex(),
            "sig": keys.sign(_signed_part(chain_id, "proposal", height, round_, block_id, valid_round)).hex()}


def certificate(height: int, round_: int, block_id: str, precommits: list[dict]) -> dict:
    return {"height": height, "round": round_, "block": block_id,
            "votes": [{"pubkey": m["pubkey"], "sig": m["sig"]} for m in precommits]}


def check_certificate(cert: dict | None, height: int, block_id: str, validators: dict[str, int], chain_id: str) -> str | None:
    """None if `cert` shows validators holding more than two thirds of the stake committed block `block_id` at
    `height`, else what's wrong. `validators` is the set that voted on it: the one in the state just before it."""
    if not cert:
        return "no commit certificate"
    try:
        votes, round_ = list(cert["votes"]), int(cert["round"])
        if int(cert["height"]) != height or str(cert["block"]) != block_id:
            return "the commit certificate is for a different block"
    except (KeyError, TypeError, ValueError):
        return "malformed commit certificate"
    msg, power, seen = _signed_part(chain_id, "precommit", height, round_, block_id), 0, set()
    for v in votes:
        try:
            pub, sig = bytes.fromhex(v["pubkey"]), bytes.fromhex(v["sig"])
        except (KeyError, TypeError, ValueError):
            return "malformed vote in the commit certificate"
        addr = address_of(pub)
        if addr in seen or addr not in validators:
            continue
        if not verify(pub, msg, sig):
            return "a forged vote in the commit certificate"
        seen.add(addr)
        power += validators[addr]
    total = sum(validators.values())
    if not quorum(power, total):
        return f"signed by validators with {power} of {total} stake; more than two thirds must sign"
    return None


def _block_id(value: dict) -> str:
    return Block.from_dict(value["block"]).hash.hex()


def _value_key(value: dict) -> str:
    """What a proposal's checks are cached under: its whole value, not its block id. Whoever relays a proposal can
    change it and keep both the id and the proposer's signature: the proposer signs only the block id, the record isn't
    part of the block, and the transaction root's Merkle tree repeats an odd level's last node, so a copy with its last
    transaction repeated has the same root."""
    return h256(canonical(value)).hex()


# --- the state machine ----------------------------------------------------------------------

class Engine:
    """One validator's part in the voting (or an observer's, with no keys or no stake). Thread-safe. Callbacks run
    holding the engine's lock and must return quickly:

        validators()              the validator set for the next height, {address: stake}
        validate(value) -> bool   whether a proposed block may be committed next
        commit(height, value, certificate)   add the block to the chain (it's final)
        broadcast(message)        send a message to the other validators
        has_work() -> bool        whether there's anything to put in a block (a funded job, waiting transactions);
                                  until there is, nobody's waiting for a proposal, so idle chains don't churn rounds
        behind(height)            another validator is already at `height`: fetch the blocks we're missing
        persist(state)            save {height, round, locked_id, locked_round} before each of our votes goes out, so a
                                  validator that restarts mid-height resumes past the rounds it voted in (pass it to
                                  start()) instead of voting in them twice
        propose_wait() -> float   optional: how long to wait for a proposal, when it depends on the chain (a job
                                  sized for a big model runs longer rounds); otherwise timeouts.propose

    When it's our turn to propose a new block, `wanted` is set to (height, round); node.py builds the block (which can
    take a whole training round) and calls propose().
    """

    def __init__(self, chain_id: str, keys: KeyPair | None, timeouts: Timeouts, *, validators, validate, commit, broadcast,
                 has_work=lambda: True, behind=lambda height: None, persist=lambda state: None, block_id=_block_id,
                 propose_wait=None):
        self.chain_id, self.keys, self.timeouts = chain_id, keys, timeouts
        self.me = keys.address if keys else None
        self._validators, self._validate, self._commit, self._broadcast = validators, validate, commit, broadcast
        self._has_work, self._behind, self._persist, self._id = has_work, behind, persist, block_id
        self._propose_wait = propose_wait
        self.lock = threading.RLock()
        self.now = 0.0
        self.height = -1
        self.round = 0
        self.step = Step.PROPOSE
        self.wanted: tuple[int, int] | None = None
        self.validators: dict[str, int] = {}
        self.total, self.order, self.locked_round, self.waiting = 0, [], -1, False
        self.future: dict[tuple, tuple[dict, int]] = {}  # (sender, kind, height, round) -> (message, its size)
        self.evidence: list[tuple[dict, dict]] = []  # pairs of conflicting messages from one validator: grounds for slashing

    # --- inputs ------------------------------------------------------------------------------

    def start(self, height: int, now: float | None = None, resume: dict | None = None) -> None:
        """Begin voting on block `height` (the chain's tip is height - 1). Also used after catching up by sync.
        `resume` is what persist() last saved: if it's for this height, we carry on after the round we last voted in,
        still locked on whatever we were locked on."""
        with self.lock:
            self.now = time.time() if now is None else now
            if height > self.height:
                self._new_height(height)
                if resume and resume.get("height") == height:
                    self.locked_id, self.locked_round = resume.get("locked_id"), int(resume.get("locked_round", -1))
                    self._start_round(int(resume["round"]) + 1)
                self._progress()

    def receive(self, msg: dict, now: float | None = None) -> None:
        with self.lock:
            self.now = time.time() if now is None else now
            self._on_message(msg)
            self._progress()

    def propose(self, height: int, round_: int, value: dict, now: float | None = None) -> bool:
        """The block we were asked for. False if it's too late: the round moved on while it was being built."""
        with self.lock:
            self.now = time.time() if now is None else now
            if self.wanted != (height, round_) or self.round != round_ or self.step != Step.PROPOSE:
                return False
            self.wanted = None
            self._send(proposal(self.keys, self.chain_id, height, round_, -1, value, self._id(value)))
            self._progress()
            return True

    def tick(self, now: float | None = None) -> None:
        with self.lock:
            self.now = time.time() if now is None else now
            if self.height < 0:
                return
            if self.waiting:
                self._arm_propose()
            due = sorted(t for t in self.timers if t[0] <= self.now)
            self.timers = [t for t in self.timers if t[0] > self.now]
            for _, step, round_ in due:
                self._timeout(Step(step), round_)
            self._progress()
            if self.mine and self.now - self.gossiped >= self.timeouts.regossip:
                self.gossiped = self.now
                for m in self.mine:
                    self._broadcast(m)

    def wants(self, height: int, round_: int) -> bool:
        with self.lock:
            return self.wanted == (height, round_)

    def proposer(self, round_: int | None = None) -> str | None:
        with self.lock:
            return self.order[(self.round if round_ is None else round_) % len(self.order)] if self.order else None

    def status(self) -> dict:
        with self.lock:
            return {"height": self.height, "round": self.round, "step": self.step.value, "proposer": self.proposer(),
                    "validators": len(self.validators), "locked_round": self.locked_round}

    # --- bookkeeping -------------------------------------------------------------------------------

    def _new_height(self, height: int) -> None:
        self.height = height
        self.validators = self._validators()
        self.total = sum(self.validators.values())
        self.order = proposer_order(self.validators, height) if self.total else []
        self.locked_value = self.locked_id = self.valid_value = self.valid_id = None
        self.locked_round = self.valid_round = -1
        self.proposals: dict[int, tuple[str, dict, int, str]] = {}  # round -> (block id, value, valid round, _value_key)
        self.votes: dict[str, dict[int, dict[str, tuple[str | None, dict]]]] = {"prevote": {}, "precommit": {}}
        self.timers: list[tuple[float, str, int]] = []
        self.fired: set[tuple[str, int]] = set()  # rules that act only the first time they hold in a round
        self.checked: dict[str, bool] = {}  # _value_key -> whether that value passed validate()
        self.mine: list[dict] = []
        self.gossiped = self.now
        self._start_round(0)
        later, self.future = self.future, {}
        for m, _ in later.values():
            self._on_message(m)

    def _start_round(self, round_: int) -> None:
        self.round, self.step, self.wanted = round_, Step.PROPOSE, None
        if self.me is not None and self.me == self.proposer(round_):
            if self.valid_value is not None:  # a block the network nearly agreed on: propose it again rather than a new one
                self._send(proposal(self.keys, self.chain_id, self.height, round_, self.valid_round, self.valid_value, self.valid_id))
            else:
                self.wanted = (self.height, round_)
        self.waiting = True
        self._arm_propose()

    def _arm_propose(self) -> None:
        """Start the wait for this round's proposal, once anyone has reason to expect one."""
        r = self.round
        if self.total and (r > 0 or r in self.proposals or self._senders(r) or self._has_work()):
            self.waiting = False
            self._schedule(Step.PROPOSE, r)

    def _schedule(self, step: Step, round_: int) -> None:
        propose = self._propose_wait() if step == Step.PROPOSE and self._propose_wait else None
        self.timers.append((self.now + self.timeouts.wait(step, round_, propose), step.value, round_))

    def _send(self, msg: dict) -> None:
        self.mine.append(msg)
        self._on_message(msg)
        self._broadcast(msg)

    def _vote(self, kind: str, block_id: str | None) -> None:
        if self.me in self.validators:
            self._persist({"height": self.height, "round": self.round, "locked_id": self.locked_id, "locked_round": self.locked_round})
            self._send(vote(self.keys, self.chain_id, kind, self.height, self.round, block_id))

    def _on_message(self, msg: dict) -> None:
        try:
            kind, height, round_ = msg["type"], int(msg["height"]), int(msg["round"])
            if msg["chain_id"] != self.chain_id or kind not in ("proposal", "prevote", "precommit"):
                return
        except (KeyError, TypeError, ValueError):
            return
        # A later height's messages are checked like this height's, signer included (it must be one of the validators
        # we know), before they're kept or make us look for blocks: anyone can send them.
        if height < self.height or not 0 <= round_ <= (self.round if height == self.height else 0) + MAX_ROUNDS_AHEAD:
            return
        try:
            pub, sig = bytes.fromhex(msg["pubkey"]), bytes.fromhex(msg["sig"])
            addr = address_of(pub)
            if addr not in self.validators:  # before working out a proposal's block id, which takes a while
                return
            if kind == "proposal":
                valid_round = int(msg["valid_round"])
                if height == self.height and (addr != self.proposer(round_) or not -1 <= valid_round < round_):
                    return
                value = {"block": msg["block"], "record": msg.get("record")}
                block_id = self._id(value)
                signed = _signed_part(self.chain_id, kind, height, round_, block_id, valid_round)
            else:
                block_id = msg["block"]
                if block_id is not None and not isinstance(block_id, str):
                    return
                signed = _signed_part(self.chain_id, kind, height, round_, block_id)
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        if not verify(pub, signed, sig):
            return
        if height > self.height:
            self._keep((addr, kind, height, round_), msg)
            self._behind(height)
            return
        if kind == "proposal":
            have = self.proposals.get(round_)
            if have is not None and have[0] != block_id:
                self.evidence.append(({"type": "proposal", "height": height, "round": round_, "block": have[0]}, msg))
                log.warning("proposer %s proposed two blocks in round %d at height %d", addr, round_, height)
                return
            if have is not None and self.checked.get(have[3]) is not False:
                return  # the same block again (proposers re-send theirs), and our copy passed or isn't checked yet
            key = _value_key(value)
            if have is not None and (key == have[3] or self.checked.get(key) is False):
                return
            # The round's first proposal, or a different copy of one whose copy we have failed its checks: a relay may
            # have altered that one (see _value_key), and this may be the proposer's own.
            self.proposals[round_] = (block_id, value, valid_round, key)
            return
        votes = self.votes[kind].setdefault(round_, {})
        if addr in votes:
            if votes[addr][0] != block_id:
                self.evidence.append((votes[addr][1], msg))
                log.warning("validator %s sent conflicting %ss in round %d at height %d", addr, kind, round_, height)
            return
        votes[addr] = (block_id, msg)

    def _keep(self, key: tuple, msg: dict) -> None:
        """Hold a validator's message for a later height until we get there, within the FUTURE_* bounds. `key` is
        (sender, kind, height, round): one message each, since validators re-send theirs every few seconds."""
        sender, _, height, _ = key
        if height > self.height + FUTURE_HEIGHTS or key in self.future:
            return
        theirs = [size for k, (_, size) in self.future.items() if k[0] == sender]
        size = len(canonical(msg))
        if len(theirs) < FUTURE_PER_SENDER and sum(theirs) + size <= FUTURE_BYTES:
            self.future[key] = (msg, size)

    def _power(self, kind: str, round_: int, block_id=...) -> int:
        """Stake behind `kind` votes in a round: for `block_id` (None means a vote for nothing), or for anything."""
        votes = self.votes[kind].get(round_, {})
        return sum(self.validators[a] for a, (b, _) in votes.items() if block_id is ... or b == block_id)

    def _senders(self, round_: int) -> set[str]:
        out = {a for kind in self.votes for a in self.votes[kind].get(round_, {})}
        if round_ in self.proposals:
            out.add(self.proposer(round_))
        return out

    def _valid(self, prop: tuple) -> bool:
        """Whether a proposal (an entry of self.proposals) passes validate(): checked once per distinct value."""
        _, value, _, key = prop
        if key not in self.checked:
            try:
                self.checked[key] = bool(self._validate(value))
            except Exception:  # noqa: BLE001 - a proposal we can't even evaluate is an invalid one
                log.exception("checking a proposed block failed")
                self.checked[key] = False
        return self.checked[key]

    def _first(self, rule: str, round_: int) -> bool:
        if (rule, round_) in self.fired:
            return False
        self.fired.add((rule, round_))
        return True

    # --- the rules (line numbers are Algorithm 1's) ----------------------------------------------

    def _progress(self) -> None:
        while self.height >= 0 and self.total and self._rule():
            pass

    def _rule(self) -> bool:
        """Applies the first rule that holds. True if it changed anything."""
        r, step, q = self.round, self.step, lambda p: quorum(p, self.total)
        for pr, p in sorted(self.proposals.items()):  # 49: a block is decided, in whatever round
            if q(self._power("precommit", pr, p[0])) and self._valid(p):
                self._decide(pr, p[0], p[1])
                return True
        later = [x for x in {*self.proposals, *self.votes["prevote"], *self.votes["precommit"]} if x > r]
        for x in sorted(later, reverse=True):  # 55: over a third of the stake is in a later round, so we're late
            if 3 * sum(self.validators[a] for a in self._senders(x)) > self.total:
                self._start_round(x)
                return True
        prop = self.proposals.get(r)
        if step == Step.PROPOSE and prop:
            bid, _, vr, _ = prop
            if vr == -1:  # 22: a new block
                ok = self._valid(prop) and (self.locked_round == -1 or self.locked_id == bid)
            elif q(self._power("prevote", vr, bid)):  # 28: a block more than two thirds prevoted for in round vr
                ok = self._valid(prop) and (self.locked_round <= vr or self.locked_id == bid)
            else:
                ok = None
            if ok is False and not self._valid(prop):
                # It failed its checks, but our copy may be one a relay altered (see _value_key), and proposers re-send
                # theirs every few seconds: give a good copy that long to replace it before voting nil (the timer
                # votes, unless the propose timeout comes first).
                if self._first("bad-proposal", r):
                    self.timers.append((self.now + 2 * self.timeouts.regossip, Step.PROPOSE.value, r))
                    return True
                ok = None
            if ok is not None:
                self.step = Step.PREVOTE
                self._vote("prevote", bid if ok else None)
                return True
        if step == Step.PREVOTE and q(self._power("prevote", r)) and self._first("prevote-wait", r):  # 34
            self._schedule(Step.PREVOTE, r)
            return True
        if (step != Step.PROPOSE and prop and q(self._power("prevote", r, prop[0])) and self._valid(prop)
                and self._first("polka", r)):  # 36: more than two thirds prevoted for this round's block
            bid, value, _, _ = prop
            if step == Step.PREVOTE:
                self.locked_value, self.locked_id, self.locked_round = value, bid, r
                self.step = Step.PRECOMMIT
                self._vote("precommit", bid)
            self.valid_value, self.valid_id, self.valid_round = value, bid, r
            return True
        if step == Step.PREVOTE and q(self._power("prevote", r, None)):  # 44: more than two thirds prevoted for nothing
            self.step = Step.PRECOMMIT
            self._vote("precommit", None)
            return True
        if q(self._power("precommit", r)) and self._first("precommit-wait", r):  # 47
            self._schedule(Step.PRECOMMIT, r)
            return True
        return False

    def _timeout(self, step: Step, round_: int) -> None:
        if round_ != self.round:
            return
        if step == Step.PROPOSE and self.step == Step.PROPOSE:  # 57: no (valid) proposal in time
            self.step = Step.PREVOTE
            self._vote("prevote", None)
        elif step == Step.PREVOTE and self.step == Step.PREVOTE:  # 61
            self.step = Step.PRECOMMIT
            self._vote("precommit", None)
        elif step == Step.PRECOMMIT:  # 65: this round won't decide; on to the next proposer
            self._start_round(round_ + 1)

    def _decide(self, round_: int, block_id: str, value: dict) -> None:
        height = self.height
        precommits = [m for b, m in self.votes["precommit"][round_].values() if b == block_id]
        log.debug("height %d decided in round %d: block %s…", height, round_, block_id[:12])
        try:
            self._commit(height, value, certificate(height, round_, block_id, precommits))
        except Exception:  # noqa: BLE001 - e.g. sync got the block first; node.py restarts us at its tip either way
            log.exception("committing block %d failed", height)
        self._new_height(height + 1)
