"""Ledger rules that came with block version 4 (ledger.ACCOUNTABILITY_VERSION): a double-sign costs a validator its
stake and its place, joining the validator set takes economics.VALIDATOR_STAKE, and joins and leaves take effect
ledger.VALIDATOR_DELAY heights after their block. Blocks of versions 1-3 keep their rules (test_rules_v2,
test_consensus.ValidatorSetTests)."""
import copy
import unittest

from vibehash import economics as econ
from vibehash import ledger
from vibehash.consensus import _compact, evidence_memo, proposal, vote
from vibehash.crypto import KeyPair
from vibehash.ledger import (EVIDENCE_WINDOW, SYSTEM, VALIDATOR_DELAY, VERSION, Chain, InvalidBlock, Tx)
from vibehash.tasks import SYN_PER_VIBE

VIBE = SYN_PER_VIBE
CID = "rules-v4"
A, B, C, D, REP = (KeyPair.generate(f"rules v4 {n}".encode()) for n in ("a", "b", "c", "d", "reporter"))
URL = {k.address: f"http://10.0.0.{i}:8545" for i, k in enumerate((A, B, C, D), 1)}
X, Y = "aa" * 32, "bb" * 32  # two block ids


def chain(validators=(A, B, C), **kw) -> Chain:
    args = {"allocations": {REP.address: 10 * VIBE, C.address: 300 * VIBE, D.address: 300 * VIBE},
            "stakes": {A.address: 30 * VIBE, B.address: 30 * VIBE, C.address: 30 * VIBE}, "chain_id": CID,
            "validators": {k.address: URL[k.address] for k in validators}}
    return Chain(**(args | kw))


def tx(c: Chain, keys: KeyPair, kind: str, recipient: str = SYSTEM, amount: int = 0, memo: str = "") -> Tx:
    return Tx.signed(keys, kind, recipient, amount, c.next_nonce(keys.address), memo, c.chain_id)


def seal(c: Chain, signer: KeyPair = A, txs=(), version: int = VERSION):
    block, dropped = c.build_block(None, list(txs), c.tip.timestamp + 1, version=version)
    block.sign(signer)
    c.append(block)
    return block, dropped


def until(c: Chain, height: int, signer: KeyPair = A) -> None:
    while c.tip.height < height:
        seal(c, signer)


def votes(keys: KeyPair, height: int, kind: str = "precommit", round_: int = 0, blocks=(X, Y), chain_id: str = CID) -> str:
    """The memo of an evidence tx: two signed votes for different blocks in one round."""
    return evidence_memo(*(vote(keys, chain_id, kind, height, round_, b) for b in blocks))


def evidence(c: Chain, reporter: KeyPair, offender: KeyPair, memo: str) -> Tx:
    return tx(c, reporter, "evidence", offender.address, 0, memo)


class DoubleSignTests(unittest.TestCase):
    def test_a_double_sign_costs_all_its_stake_and_its_place(self):
        # C has 20 VIBE it could withdraw, 10 of bootstrap stake, and 5 VIBE unbonding from before version 4.
        c = chain(stakes={A.address: 30 * VIBE, B.address: 30 * VIBE, C.address: 20 * VIBE},
                  bootstrap={C.address: 10 * VIBE})
        seal(c, txs=[tx(c, C, "unstake", amount=5 * VIBE)], version=3)
        seal(c)  # height 2, which C signs both ways
        before = c.state
        self.assertEqual((before.stake(C.address), before.unbonding_total(C.address)), (25 * VIBE, 5 * VIBE))
        report = evidence(c, REP, C, votes(C, 2))
        _, dropped = seal(c, txs=[report])
        self.assertEqual(dropped, [])
        s = c.state
        self.assertEqual((s.stake(C.address), s.unbonding_total(C.address), s.locked.get(C.address)), (0, 0, None))
        bounty = 15 * VIBE // econ.DOUBLE_SIGN_BOUNTY + 5 * VIBE // econ.DOUBLE_SIGN_BOUNTY  # a tenth of what's free
        self.assertEqual(s.balance(REP.address), 10 * VIBE + bounty)
        self.assertEqual(s.burned - before.burned, 30 * VIBE - bounty, "the rest burned, bootstrap stake included")
        self.assertEqual((s.banned, sorted(s.validators), sorted(s.validator_set())),
                         ({C.address: 3}, sorted([A.address, B.address]), sorted([A.address, B.address])))
        self.assertEqual(s.total_supply(), s.expected_supply())
        self.assertTrue(c.validate())
        # It can stake again, as a worker, but never be a validator again.
        seal(c, txs=[tx(c, C, "stake", amount=200 * VIBE)])
        with self.assertRaisesRegex(InvalidBlock, "can't be a validator again"):
            c.check_tx(tx(c, C, "validator", memo=URL[C.address]), [])
        with self.assertRaisesRegex(InvalidBlock, "already punished at height 3|already punished, at height 3"):
            c.check_tx(evidence(c, REP, C, votes(C, 2, round_=1)), [])

    def test_conflicting_proposals_and_a_vote_for_nothing_against_a_block_count_too(self):
        c = chain()
        seal(c)
        pair = [_compact(proposal(B, CID, 1, 2, vr, {"block": {}}, b), b) for vr, b in ((-1, X), (1, Y))]
        nil = evidence_memo(vote(C, CID, "prevote", 1, 0, None), vote(C, CID, "prevote", 1, 0, X))
        self.assertEqual(seal(c, txs=[evidence(c, REP, B, evidence_memo(*pair)), evidence(c, D, C, nil)])[1], [])
        self.assertEqual(sorted(c.state.banned), sorted([B.address, C.address]))
        self.assertEqual(list(c.state.validator_set()), [A.address])

    def test_a_validator_reporting_itself_gets_nothing_back(self):
        c = chain()
        seal(c)
        before = c.state
        seal(c, txs=[evidence(c, C, C, votes(C, 1))])
        self.assertEqual(c.state.balance(C.address), before.balance(C.address))
        self.assertEqual(c.state.burned - before.burned, 30 * VIBE)

    def test_evidence_needs_a_version_4_block(self):
        c = chain()
        seal(c, version=3)
        report = evidence(c, REP, C, votes(C, 1))
        with self.assertRaisesRegex(InvalidBlock, "version 4"):
            ledger._apply_user_tx(copy.deepcopy(c.state), report, 2, CID, 3)
        self.assertEqual(seal(c, txs=[report], version=3)[1], [report])
        self.assertEqual(seal(c, txs=[report])[1], [], "the next block, of version 4, takes it")

    def test_what_isnt_evidence_of_a_double_sign(self):
        c = chain()
        seal(c)
        forged = vote(C, CID, "precommit", 1, 0, Y)
        forged["sig"] = vote(C, CID, "precommit", 1, 0, X)["sig"]
        late = vote(C, CID, "precommit", 1, 0, Y)
        cases = {
            "forged": evidence_memo(vote(C, CID, "precommit", 1, 0, X), forged),
            "another chain": votes(C, 1, chain_id="other-chain"),
            "same block": votes(C, 1, blocks=(X, X)),
            "already on the chain": votes(C, 3),
            "no vote at height 1": votes(D, 1),
            "malformed": votes(C, 1).replace('"round":0', '"round":true'),
            "malformed evidence: kind": evidence_memo(*(dict(vote(C, CID, "precommit", 1, 0, b), type="commit")
                                                        for b in (X, Y))),
            "a block id": votes(C, 1, blocks=(X, "c" * 65)),
            "at most": votes(C, 1)[:-1] + " " * ledger.MAX_EVIDENCE_BYTES + "}",
        }
        for why, memo in cases.items():
            with self.subTest(why), self.assertRaisesRegex(InvalidBlock, why):
                c.check_tx(evidence(c, REP, D if why.startswith("no vote") else C, memo), [])
        with self.assertRaisesRegex(InvalidBlock, "its recipient"):
            c.check_tx(evidence(c, REP, B, votes(C, 1)), [])
        with self.assertRaisesRegex(InvalidBlock, "its recipient"):
            c.check_tx(tx(c, REP, "evidence", C.address, 1, votes(C, 1)), [])
        # Two votes of different rounds aren't one offence: the second's signature doesn't cover what the memo says.
        mixed = evidence_memo(vote(C, CID, "precommit", 1, 0, X), dict(late, round=1, sig=vote(C, CID, "precommit", 1, 1, Y)["sig"]))
        with self.assertRaisesRegex(InvalidBlock, "forged"):
            c.check_tx(evidence(c, REP, C, mixed), [])
        self.assertEqual(c.state.banned, {})

    def test_evidence_is_taken_for_a_window_of_heights(self):
        c = chain()
        until(c, 2)
        old, edge = evidence(c, REP, C, votes(C, 1)), evidence(c, REP, B, votes(B, 2))
        until(c, 1 + EVIDENCE_WINDOW)  # the next block is 2 + EVIDENCE_WINDOW
        with self.assertRaisesRegex(InvalidBlock, "too long ago"):
            c.check_tx(old, [])
        self.assertEqual(seal(c, txs=[old, edge])[1], [old], "exactly EVIDENCE_WINDOW heights back is still taken")
        self.assertEqual(list(c.state.banned), [B.address])

    def test_every_coin_it_had_bonded_is_still_there_at_the_end_of_the_window(self):
        """The fastest way out: C's leave takes effect just after it double-signs, and it unstakes at once. Its coins
        unbond for UNBONDING_ROUNDS, one height longer than evidence is taken."""
        c = chain()
        seal(c, txs=[tx(c, C, "validator")])  # height 1: takes effect at 1 + VALIDATOR_DELAY
        h = VALIDATOR_DELAY  # the last height C votes on, and the one it signs both ways
        until(c, h)
        self.assertNotIn(C.address, c.state.validators)
        seal(c, txs=[tx(c, C, "unstake", amount=30 * VIBE)])
        until(c, h + EVIDENCE_WINDOW - 1)
        before = c.state
        self.assertEqual(seal(c, txs=[evidence(c, REP, C, votes(C, h))])[1], [])
        self.assertEqual(c.state.unbonding_total(C.address), 0)
        self.assertEqual(c.state.burned - before.burned, 27 * VIBE)
        self.assertEqual(c.state.balance(REP.address) - before.balance(REP.address), 3 * VIBE)

    def test_a_chain_whose_only_validator_double_signs_stops(self):
        """It keeps its place in the list, with no vote and no stake: if the list were empty, the chain would become
        one any staked node could seal without certificates (Node.bft)."""
        c = chain(validators=(A,))
        seal(c)
        seal(c, txs=[evidence(c, REP, A, votes(A, 1))])
        self.assertEqual((list(c.state.validators), c.state.validator_set()), ([A.address], {}))
        for signer, why in ((A, "not staked"), (B, "not a validator")):
            with self.assertRaisesRegex(InvalidBlock, why):
                seal(c, signer)
        with self.assertRaisesRegex(InvalidBlock, "last validator"):
            ledger._apply_user_tx(copy.deepcopy(c.state), tx(c, A, "validator"), 3, CID, VERSION)


class JoinTests(unittest.TestCase):
    def test_joining_takes_the_validator_stake_in_version_4_and_the_minimum_before(self):
        below = econ.VALIDATOR_STAKE - 1
        for version, joins in ((3, True), (4, False)):
            with self.subTest(version=version):
                c = chain()
                seal(c, txs=[tx(c, D, "stake", amount=below)], version=version)
                _, dropped = seal(c, txs=[tx(c, D, "validator", memo=URL[D.address])], version=version)
                self.assertEqual(dropped == [], joins)
                self.assertEqual(D.address in c.state.validators, joins, "version 3: at once")
        with self.assertRaisesRegex(InvalidBlock, f"takes at least {econ.VALIDATOR_STAKE // VIBE} VIBE"):
            c.check_tx(tx(c, D, "validator", memo=URL[D.address]), [])
        seal(c, txs=[tx(c, D, "stake", amount=1)])
        self.assertEqual(seal(c, txs=[tx(c, D, "validator", memo=URL[D.address])])[1], [])

    def test_a_validator_in_the_set_can_move_with_only_the_minimum_stake(self):
        c = chain()
        seal(c, txs=[tx(c, A, "validator", memo="https://a.example:8545")])
        self.assertEqual(c.state.validators[A.address], "https://a.example:8545", "at once: who votes doesn't change")


class DelayTests(unittest.TestCase):
    def test_a_join_and_a_leave_take_effect_exactly_validator_delay_heights_later(self):
        c = chain(bootstrap={B.address: 10 * VIBE})
        seal(c, txs=[tx(c, D, "stake", amount=econ.VALIDATOR_STAKE)])
        joined, _ = seal(c, txs=[tx(c, D, "validator", memo=URL[D.address]), tx(c, B, "validator")])
        at = joined.height + VALIDATOR_DELAY
        self.assertEqual(c.state.pending, [[at, D.address, URL[D.address]], [at, B.address, ""]])
        burned = c.state.burned
        until(c, at - 2)
        self.assertNotIn(D.address, c.state.validators)
        self.assertEqual(sorted(c.state.validator_set()), sorted([A.address, B.address, C.address]))
        with self.assertRaisesRegex(InvalidBlock, "not a validator"):
            seal(c, D)  # block at - 1 is still the old set's
        seal(c, B)
        # After block at - 1, the set voting on block `at` is the new one.
        self.assertEqual(sorted(c.state.validator_set()), sorted([A.address, C.address, D.address]))
        self.assertEqual((c.state.pending, c.state.stake(B.address), c.state.burned - burned),
                         ([], 30 * VIBE, 10 * VIBE), "B's bootstrap stake burned once its leave took effect")
        seal(c, D)
        with self.assertRaisesRegex(InvalidBlock, "not a validator"):
            seal(c, B)
        self.assertTrue(c.validate())

    def test_a_validators_stake_is_bonded_until_its_leave_takes_effect(self):
        c = chain()
        for kind in ("stake", "unstake"):
            with self.assertRaisesRegex(InvalidBlock, "bonded"):
                c.check_tx(tx(c, A, kind, amount=VIBE), [])
        self.assertEqual(seal(c, txs=[tx(c, A, "unstake", amount=VIBE)], version=3)[1], [], "version 3 let it")
        left, _ = seal(c, txs=[tx(c, A, "validator")], signer=B)
        until(c, left.height + VALIDATOR_DELAY - 2, signer=B)
        with self.assertRaisesRegex(InvalidBlock, "bonded"):
            c.check_tx(tx(c, A, "unstake", amount=VIBE), [])
        seal(c, B)
        self.assertEqual(seal(c, B, txs=[tx(c, A, "unstake", amount=29 * VIBE)])[1], [], "from the height it's out")

    def test_a_joiner_is_bonded_too_and_can_call_its_join_off(self):
        c = chain()
        seal(c, txs=[tx(c, D, "stake", amount=econ.VALIDATOR_STAKE)])
        seal(c, txs=[tx(c, D, "validator", memo=URL[D.address])])
        for t in (tx(c, D, "unstake", amount=VIBE), tx(c, D, "validator", memo=URL[D.address])):
            with self.assertRaisesRegex(InvalidBlock, "bonded|waiting"):
                c.check_tx(t, [])
        seal(c, txs=[tx(c, D, "validator")])
        self.assertEqual(c.state.pending, [])
        self.assertEqual(seal(c, txs=[tx(c, D, "unstake", amount=econ.VALIDATOR_STAKE)])[1], [])

    def test_the_last_validator_with_a_vote_cant_leave_counting_leaves_still_waiting(self):
        c = chain(validators=(A, B))
        seal(c, txs=[tx(c, D, "stake", amount=econ.VALIDATOR_STAKE)])
        seal(c, txs=[tx(c, A, "validator"), tx(c, D, "validator", memo=URL[D.address])])
        leave = tx(c, B, "validator")
        with self.assertRaisesRegex(InvalidBlock, "last validator"):
            c.check_tx(leave, [])  # A is leaving, and D's join could still be called off
        with self.assertRaisesRegex(InvalidBlock, "already leaves"):
            c.check_tx(tx(c, A, "validator"), [])
        until(c, c.tip.height + VALIDATOR_DELAY, signer=B)
        self.assertEqual(sorted(c.state.validator_set()), sorted([B.address, D.address]))
        self.assertEqual(seal(c, B, txs=[tx(c, B, "validator")])[1], [], "now D has a vote")


class OlderBlocksTests(unittest.TestCase):
    def test_version_3_blocks_keep_their_rules_and_leave_no_pending_changes(self):
        c = chain()
        seal(c, txs=[tx(c, D, "stake", amount=econ.MIN_STAKE)], version=3)
        seal(c, txs=[tx(c, D, "validator", memo=URL[D.address]), tx(c, B, "validator")], version=3)
        self.assertEqual((c.state.pending, D.address in c.state.validator_set(), B.address in c.state.validators),
                         ([], True, False))
        self.assertEqual(c.state.voters[-1], [3, sorted([A.address, C.address, D.address])])


if __name__ == "__main__":
    unittest.main()
