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

from .crypto import Draw, KeyPair, address_of, verify
from .ledger import Block, canonical

log = logging.getLogger("vibehash.consensus")
MAX_FUTURE = 10_000  # messages for later heights kept until we catch up
MAX_ROUNDS_AHEAD = 1_000  # later rounds than this are ignored (an attacker's memory bomb, not a slow network)


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

    def wait(self, step: Step, round_: int) -> float:
        return (self.propose if step == Step.PROPOSE else self.vote) + round_ * self.grow


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

    When it's our turn to propose a new block, `wanted` is set to (height, round); node.py builds the block (which can
    take a whole training round) and calls propose().
    """

    def __init__(self, chain_id: str, keys: KeyPair | None, timeouts: Timeouts, *, validators, validate, commit, broadcast,
                 has_work=lambda: True, behind=lambda height: None, persist=lambda state: None, block_id=_block_id):
        self.chain_id, self.keys, self.timeouts = chain_id, keys, timeouts
        self.me = keys.address if keys else None
        self._validators, self._validate, self._commit, self._broadcast = validators, validate, commit, broadcast
        self._has_work, self._behind, self._persist, self._id = has_work, behind, persist, block_id
        self.lock = threading.RLock()
        self.now = 0.0
        self.height = -1
        self.round = 0
        self.step = Step.PROPOSE
        self.wanted: tuple[int, int] | None = None
        self.validators: dict[str, int] = {}
        self.total, self.order, self.locked_round, self.waiting = 0, [], -1, False
        self.future: list[dict] = []
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
        self.proposals: dict[int, tuple[str, dict, int]] = {}  # round -> (block id, value, valid round)
        self.votes: dict[str, dict[int, dict[str, tuple[str | None, dict]]]] = {"prevote": {}, "precommit": {}}
        self.timers: list[tuple[float, str, int]] = []
        self.fired: set[tuple[str, int]] = set()  # rules that act only the first time they hold in a round
        self.checked: dict[str, bool] = {}
        self.mine: list[dict] = []
        self.gossiped = self.now
        self._start_round(0)
        later, self.future = self.future, []
        for m in later:
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
        self.timers.append((self.now + self.timeouts.wait(step, round_), step.value, round_))

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
        if height < self.height:
            return
        if height > self.height:
            if len(self.future) < MAX_FUTURE:
                self.future.append(msg)
            self._behind(height)
            return
        if not 0 <= round_ <= self.round + MAX_ROUNDS_AHEAD:
            return
        try:
            pub, sig = bytes.fromhex(msg["pubkey"]), bytes.fromhex(msg["sig"])
            if kind == "proposal":
                value = {"block": msg["block"], "record": msg.get("record")}
                block_id, valid_round = self._id(value), int(msg["valid_round"])
                signed = _signed_part(self.chain_id, kind, height, round_, block_id, valid_round)
            else:
                block_id = msg["block"]
                if block_id is not None and not isinstance(block_id, str):
                    return
                signed = _signed_part(self.chain_id, kind, height, round_, block_id)
        except (KeyError, TypeError, ValueError):
            return
        addr = address_of(pub)
        if addr not in self.validators or not verify(pub, signed, sig):
            return
        if kind == "proposal":
            if addr != self.proposer(round_) or not -1 <= valid_round < round_:
                return
            if round_ in self.proposals:
                if self.proposals[round_][0] != block_id:
                    self.evidence.append(({"type": "proposal", "height": height, "round": round_, "block": self.proposals[round_][0]}, msg))
                    log.warning("proposer %s proposed two blocks in round %d at height %d", addr, round_, height)
                return
            self.proposals[round_] = (block_id, value, valid_round)
            return
        votes = self.votes[kind].setdefault(round_, {})
        if addr in votes:
            if votes[addr][0] != block_id:
                self.evidence.append((votes[addr][1], msg))
                log.warning("validator %s sent conflicting %ss in round %d at height %d", addr, kind, round_, height)
            return
        votes[addr] = (block_id, msg)

    def _power(self, kind: str, round_: int, block_id=...) -> int:
        """Stake behind `kind` votes in a round: for `block_id` (None means a vote for nothing), or for anything."""
        votes = self.votes[kind].get(round_, {})
        return sum(self.validators[a] for a, (b, _) in votes.items() if block_id is ... or b == block_id)

    def _senders(self, round_: int) -> set[str]:
        out = {a for kind in self.votes for a in self.votes[kind].get(round_, {})}
        if round_ in self.proposals:
            out.add(self.proposer(round_))
        return out

    def _valid(self, block_id: str, value: dict) -> bool:
        if block_id not in self.checked:
            try:
                self.checked[block_id] = bool(self._validate(value))
            except Exception:  # noqa: BLE001 - a proposal we can't even evaluate is an invalid one
                log.exception("checking a proposed block failed")
                self.checked[block_id] = False
        return self.checked[block_id]

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
        for pr, (bid, value, _) in sorted(self.proposals.items()):  # 49: a block is decided, in whatever round
            if q(self._power("precommit", pr, bid)) and self._valid(bid, value):
                self._decide(pr, bid, value)
                return True
        later = [x for x in {*self.proposals, *self.votes["prevote"], *self.votes["precommit"]} if x > r]
        for x in sorted(later, reverse=True):  # 55: over a third of the stake is in a later round, so we're late
            if 3 * sum(self.validators[a] for a in self._senders(x)) > self.total:
                self._start_round(x)
                return True
        prop = self.proposals.get(r)
        if step == Step.PROPOSE and prop:
            bid, value, vr = prop
            if vr == -1:  # 22: a new block
                ok = self._valid(bid, value) and (self.locked_round == -1 or self.locked_id == bid)
            elif q(self._power("prevote", vr, bid)):  # 28: a block more than two thirds prevoted for in round vr
                ok = self._valid(bid, value) and (self.locked_round <= vr or self.locked_id == bid)
            else:
                ok = None
            if ok is not None:
                self.step = Step.PREVOTE
                self._vote("prevote", bid if ok else None)
                return True
        if step == Step.PREVOTE and q(self._power("prevote", r)) and self._first("prevote-wait", r):  # 34
            self._schedule(Step.PREVOTE, r)
            return True
        if (step != Step.PROPOSE and prop and q(self._power("prevote", r, prop[0])) and self._valid(prop[0], prop[1])
                and self._first("polka", r)):  # 36: more than two thirds prevoted for this round's block
            bid, value, _ = prop
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
