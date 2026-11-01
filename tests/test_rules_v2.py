"""Ledger rules that changed with block version 2 (ledger.VERSION), and the version 1 behaviour that every block
already on a chain keeps: a node re-validates its chain from genesis on start-up, so a version 1 block must derive the
same transactions and hash the same as it always did."""
import json
import unittest
from unittest import mock

from vibehash import economics as econ
from vibehash import ledger
from vibehash.crypto import KeyPair
from vibehash.ledger import (BURN, SYSTEM, UNBONDING_ROUNDS, VERSION, Block, Chain, InvalidBlock, RoundSummary, Tx,
                             apply_block)
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

VIBE = SYN_PER_VIBE
MIN = econ.MIN_STAKE
V, V2, W, X, C, REQ = (KeyPair.generate(f"rules v2 {n}".encode()) for n in ("validator", "validator 2", "worker",
                                                                             "worker 2", "catcher", "requester"))
JOB = TrainingJob("j", REQ.address, budget=100 * VIBE, fee_per_round=50 * VIBE, units_per_round=4)


def chain(**kw) -> Chain:
    return Chain(**({"allocations": {REQ.address: 1000 * VIBE}, "stakes": {}, "chain_id": "rules-v2"} | kw))


def tx(c: Chain, keys: KeyPair, kind: str, recipient: str = SYSTEM, amount: int = 0, memo: str = "",
       nonce: int | None = None) -> Tx:
    nonce = c.next_nonce(keys.address) if nonce is None else nonce
    return Tx.signed(keys, kind, recipient, amount, nonce, memo, c.chain_id)


def seal(c: Chain, signer: KeyPair, summary: RoundSummary | None = None, txs=(), version: int = VERSION):
    block, dropped = c.build_block(summary, list(txs), c.tip.timestamp + 1, version=version)
    block.sign(signer)
    c.append(block)
    return block, dropped


def summary(proposer: KeyPair, worker_flops=None, slashes=(), accepted_units=0) -> RoundSummary:
    flops = dict(worker_flops or {})
    return RoundSummary(JOB.job_id, accepted_units, flops, {}, [list(s) for s in slashes], sorted(flops), [], "00",
                        proposer.address)


def slash_burned(before: ledger.State, after: ledger.State) -> int:
    """Burned by slashes: a round with no verified work burns its subsidy too."""
    return after.burned - before.burned - (after.minted - before.minted)


def kinds(block: Block) -> list[tuple]:
    return [(t.kind, t.recipient, t.amount, t.memo) for t in block.txs if t.kind not in ledger.USER_KINDS]


class VersionTests(unittest.TestCase):
    def test_versions_only_go_up_and_are_signed(self):
        c = chain(stakes={V.address: 10 * VIBE})
        old, _ = seal(c, V, version=1)
        self.assertNotIn("version", old.to_dict(), "version 1 blocks look as they always did")
        self.assertEqual(Block.from_dict(json.loads(json.dumps(old.to_dict()))).hash, old.hash)
        relabelled, _ = c.build_block(None, [], c.tip.timestamp + 1)
        relabelled.sign(V)
        relabelled.version = 1  # the proposer signed version 2
        with self.assertRaisesRegex(InvalidBlock, "signature"):
            c.append(relabelled)
        new, _ = seal(c, V)
        self.assertEqual((new.version, new.to_dict()["version"]), (VERSION, VERSION))
        self.assertEqual(Block.from_dict(json.loads(json.dumps(new.to_dict()))).hash, new.hash)
        for version in (1, VERSION + 1):  # never back down, never a version this node doesn't know
            block, _ = c.build_block(None, [], c.tip.timestamp + 1, version=version)
            block.sign(V)
            with self.assertRaisesRegex(InvalidBlock, "version"):
                c.append(block)
        self.assertTrue(c.validate())


class DuplicateTxTests(unittest.TestCase):
    def test_a_block_carrying_a_tx_twice_is_refused_on_sight(self):
        c = chain(allocations={REQ.address: 10 * VIBE, W.address: 10 * VIBE, X.address: 10 * VIBE},
                  stakes={V.address: 10 * VIBE})
        block, _ = c.build_block(None, [tx(c, REQ, "transfer", W.address, 1), tx(c, W, "transfer", X.address, 2),
                                        tx(c, X, "transfer", REQ.address, 3)], 1)
        block.sign(V)
        d = block.to_dict()
        d["txs"].append(d["txs"][-1])
        copy = Block.from_dict(d)
        self.assertEqual(copy.hash, block.hash, "crypto.merkle_root pads [a, b, c] to [a, b, c, c]")
        with self.assertRaisesRegex(InvalidBlock, "appears twice"):
            apply_block(c.state, copy, c.chain_id, c.tip)
        c.append(block)


class BootstrapSlashTests(unittest.TestCase):
    """A bootstrap validator's own sybil "catches" it: before, a quarter of its bootstrap stake became the sybil's
    spendable coins."""

    def setUp(self):
        self.c = chain(validators={V.address: "http://10.0.0.1:8545"}, bootstrap={V.address: 1000 * VIBE})

    def test_bootstrap_stake_is_burned_never_paid_as_bounty(self):
        before = self.c.state
        block, _ = seal(self.c, V, summary(V, slashes=[[V.address, C.address, "failed_replay"]]))
        s = self.c.state
        self.assertEqual(kinds(block), [("slash", BURN, 500 * VIBE, "failed_replay")])
        self.assertEqual((s.balance(C.address), s.stake(V.address), s.locked[V.address]), (0, 500 * VIBE, 500 * VIBE))
        self.assertEqual(slash_burned(before, s), 500 * VIBE)
        self.assertEqual(s.total_supply(), s.expected_supply())

    def test_bounties_come_from_withdrawable_coins_and_the_tx_list_says_so(self):
        c = chain(allocations={V.address: 3 * VIBE}, stakes={V.address: 2 * VIBE},
                  validators={V.address: "http://10.0.0.1:8545"}, bootstrap={V.address: 10 * VIBE})
        seal(c, V, txs=[tx(c, V, "unstake", amount=VIBE)])
        before = c.state
        self.assertEqual((before.stake(V.address), before.locked[V.address], before.unbonding_total(V.address)),
                         (11 * VIBE, 10 * VIBE, VIBE))
        # Half of the 12 VIBE is slashed. Only 2 VIBE of it can be withdrawn (1 staked, 1 unbonding): that's the most
        # a bounty can be, and the burn takes 4 VIBE of bootstrap stake.
        block, _ = seal(c, V, summary(V, slashes=[[V.address, C.address, "failed_replay"]]))
        s = c.state
        self.assertEqual(kinds(block), [("slash", C.address, 2 * VIBE, "failed_replay"),
                                        ("slash", BURN, 4 * VIBE, "failed_replay")])
        self.assertEqual(s.balance(C.address), 2 * VIBE, "exactly the bounty the block states")
        self.assertEqual(slash_burned(before, s), 4 * VIBE)
        self.assertEqual((s.stake(V.address), s.locked[V.address], s.unbonding_total(V.address)), (6 * VIBE, 6 * VIBE, 0))
        self.assertEqual(s.total_supply(), s.expected_supply())

    def test_version_1_still_paid_it_as_bounty(self):
        block, _ = seal(self.c, V, summary(V, slashes=[[V.address, C.address, "failed_replay"]]), version=1)
        self.assertEqual(kinds(block), [("slash", C.address, 250 * VIBE, "failed_replay"),
                                        ("slash", BURN, 250 * VIBE, "failed_replay")])
        self.assertEqual(self.c.state.balance(C.address), 250 * VIBE)
        self.assertTrue(self.c.validate())

    def test_a_bounty_can_never_draw_on_bootstrap_stake(self):
        """Even in a block that says so: the slash txs are derived, but this is checked where they're applied."""
        s = self.c.state
        with self.assertRaisesRegex(InvalidBlock, "bootstrap"):
            ledger._apply_system_tx(s, Tx("slash", V.address, C.address, VIBE, memo="x", chain_id=self.c.chain_id), 2)


class VotingSetTests(unittest.TestCase):
    """Who's listed isn't who votes: a validator below the minimum stake stays listed but can't propose."""

    def listed_but_not_voting(self, version):
        c = chain(allocations={V2.address: 20 * VIBE}, validators={V.address: "http://10.0.0.1:8545"},
                  bootstrap={V.address: 10 * VIBE}, earn_in=True)
        seal(c, V, txs=[tx(c, V2, "stake", amount=MIN)], version=version)
        seal(c, V, txs=[tx(c, V2, "validator", memo="http://10.0.0.2:8545")], version=version)
        seal(c, V, txs=[tx(c, V2, "unstake", amount=MIN)], version=version)
        self.assertEqual((len(c.state.validators), list(c.state.validator_set())), (2, [V.address]))
        return c

    def test_the_last_voting_validator_cant_leave(self):
        c = self.listed_but_not_voting(VERSION)
        leave = tx(c, V, "validator")
        with self.assertRaisesRegex(InvalidBlock, "last validator"):
            c.check_tx(leave, [])
        _, dropped = seal(c, V, txs=[leave])
        self.assertEqual(dropped, [leave])
        self.assertEqual(list(c.state.validator_set()), [V.address])

    def test_or_unstake_below_the_minimum(self):
        c = chain(stakes={W.address: MIN + VIBE}, validators={W.address: "http://10.0.0.1:8545"})
        self.assertEqual(seal(c, W, txs=[tx(c, W, "unstake", amount=VIBE)])[1], [], "it still votes after this one")
        unstake = tx(c, W, "unstake", amount=1)
        with self.assertRaisesRegex(InvalidBlock, "no validator with a vote"):
            c.check_tx(unstake, [])
        self.assertEqual(seal(c, W, txs=[unstake])[1], [unstake])
        seal(c, W)  # and the chain goes on

    def test_version_1_let_both_happen(self):
        c = self.listed_but_not_voting(1)
        self.assertEqual(seal(c, V, txs=[tx(c, V, "validator")], version=1)[1], [])
        self.assertEqual(c.state.validator_set(), {}, "nobody can propose the next block")
        solo = chain(stakes={W.address: MIN}, validators={W.address: "http://10.0.0.1:8545"})
        self.assertEqual(seal(solo, W, txs=[tx(solo, W, "unstake", amount=1)], version=1)[1], [])
        self.assertEqual(solo.state.validator_set(), {})


class CancelMidRoundTests(unittest.TestCase):
    def round_with_cancel(self, version, cancels=1):
        c = chain(stakes={V.address: 10 * VIBE, W.address: 10 * VIBE})
        seal(c, V, txs=[tx(c, REQ, "post_job", amount=JOB.budget, memo=JOB.spec_json())], version=version)
        cancel = [tx(c, REQ, "cancel_job", memo=JOB.job_id, nonce=c.next_nonce(REQ.address) + i) for i in range(cancels)]
        block, dropped = seal(c, V, summary(V, {W.address: JOB.unit_flops * 4}, accepted_units=4), cancel, version)
        self.assertEqual(dropped, [])
        return c, block

    def test_the_round_in_the_same_block_is_still_paid_for(self):
        for cancels in (1, 2):
            c, block = self.round_with_cancel(VERSION, cancels)
            s = c.state
            self.assertIn(("fee_release", SYSTEM, 50 * VIBE, JOB.job_id), kinds(block))
            self.assertEqual((s.escrow[JOB.job_id], s.balance(REQ.address)), (0, 950 * VIBE), "refunds what's left")
            self.assertEqual(s.balance(W.address), int((s.subsidy(block.height) + 50 * VIBE) * econ.WORKER_SHARE))
            self.assertEqual(s.total_supply(), s.expected_supply())

    def test_version_1_still_validates_with_no_fee_release(self):
        c, block = self.round_with_cancel(1)
        self.assertNotIn("fee_release", [k[0] for k in kinds(block)])
        self.assertEqual(c.state.balance(REQ.address), 1000 * VIBE)
        self.assertTrue(c.validate())


class UnbondingTests(unittest.TestCase):
    def slash_as_unbonding_ends(self, version):
        c = chain(stakes={V.address: 10 * VIBE, W.address: 100 * VIBE})
        unstaked, _ = seal(c, V, txs=[tx(c, W, "unstake", amount=95 * VIBE)], version=version)
        while c.tip.height < unstaked.height + UNBONDING_ROUNDS - 1:
            seal(c, V, version=version)
        seal(c, V, summary(V, slashes=[[W.address, C.address, "failed_replay"]]), version=version)
        self.assertEqual(c.tip.height, unstaked.height + UNBONDING_ROUNDS, "the height its coins come free")
        return c.state

    def test_coins_finishing_unbonding_are_slashed_first(self):
        s = self.slash_as_unbonding_ends(VERSION)
        self.assertEqual(s.balance(C.address), 25 * VIBE, "a quarter of the 100 VIBE it had bonded")
        self.assertEqual((s.balance(W.address), s.stake(W.address), s.unbonding_total(W.address)), (50 * VIBE, 0, 0))

    def test_version_1_released_them_first(self):
        s = self.slash_as_unbonding_ends(1)
        self.assertEqual(s.balance(C.address), VIBE + VIBE // 4, "a quarter of the 5 VIBE left staked")
        self.assertEqual(s.balance(W.address), 95 * VIBE)


class ProposerOffenderTests(unittest.TestCase):
    def round_led_by_an_offender(self, version):
        c = chain(stakes={V.address: 10 * VIBE, W.address: 10 * VIBE})
        seal(c, V, txs=[tx(c, REQ, "post_job", amount=JOB.budget, memo=JOB.spec_json())], version=version)
        block, _ = seal(c, V, summary(V, {W.address: JOB.unit_flops, V.address: JOB.unit_flops}, accepted_units=1,
                                      slashes=[[V.address, C.address, "failed_replay"]]), version=version)
        return c, block

    def test_isnt_paid_for_proposing(self):
        c, block = self.round_led_by_an_offender(VERSION)
        self.assertNotIn("propose", [k[3] for k in kinds(block)])
        self.assertEqual(c.state.balance(V.address), 0)
        self.assertEqual(c.state.total_supply(), c.state.expected_supply())

    def test_version_1_paid_it(self):
        c, block = self.round_led_by_an_offender(1)
        self.assertIn("propose", [k[3] for k in kinds(block)])
        self.assertGreater(c.state.balance(V.address), 0)


class PayoutRoundingTests(unittest.TestCase):
    # Past 2**53 syn floats can't hold every whole number, and both the split and the budgets could pay out too much.
    FEE = 50_572_348_557_154_241  # a job's fee per round that makes the float split overpay workers by 2 syn

    def test_split_and_budgets_never_exceed_what_they_divide(self):
        total, weights = 17_596_633_759_510_723, {"a": 80.0, "b": 27.0}
        self.assertEqual(sum(econ.split_pro_rata(total, weights).values()), total + 2, "floats: the old split")
        self.assertEqual(sum(econ.split_pro_rata(total, weights, exact=True).values()), total)
        pool = 17_699_313_173_195_095
        budgets = int(pool * econ.WORKER_SHARE) + int(pool * econ.VERIFIER_SHARE) + int(pool * econ.PROPOSER_SHARE)
        self.assertGreater(budgets, pool, "floats: the old budgets")
        p = econ.compute_payout(pool, {"a": 3, "b": 1}, {"c": 1}, {}, exact=True)
        self.assertGreaterEqual(p.burned, 0)
        for total, weights in ((1_000_003, {"a": 1, "b": 1, "c": 1, "d": 0}), (6_500_000, {"a": 300.0, "b": 50.0})):
            self.assertEqual(econ.split_pro_rata(total, weights, exact=True), econ.split_pro_rata(total, weights),
                             "the same as before on everyday amounts")

    def worker_pay(self, version) -> tuple[int, int]:
        """What a round's workers are paid, and the round's pool."""
        job = TrainingJob("big", REQ.address, budget=self.FEE, fee_per_round=self.FEE, units_per_round=4)
        c = chain(allocations={REQ.address: self.FEE}, stakes={V.address: 10 * VIBE, W.address: MIN, X.address: MIN})
        seal(c, V, txs=[tx(c, REQ, "post_job", amount=job.budget, memo=job.spec_json())], version=version)
        sm = RoundSummary(job.job_id, 4, {W.address: 80, X.address: 27}, {}, [], [W.address, X.address], [], "00",
                          V.address)
        block, _ = seal(c, V, sm, version=version)
        return sum(t.amount for t in block.txs if t.memo == "work"), c.state.subsidy(block.height) + self.FEE

    def test_a_block_pays_exactly_its_budget(self):
        paid, pool = self.worker_pay(VERSION)
        self.assertEqual(paid, pool * 4 // 5)

    def test_version_1_keeps_its_rewards(self):
        paid, pool = self.worker_pay(1)
        self.assertEqual(paid, int(pool * econ.WORKER_SHARE) + 2, "valid all the same: the verifiers' share covered it")


class BlockSizeTests(unittest.TestCase):
    def setUp(self):
        self.payers = [KeyPair.generate(f"rules v2 payer {i}".encode()) for i in range(4)]
        self.c = chain(allocations={k.address: 10 * VIBE for k in self.payers}, stakes={V.address: 10 * VIBE})

    def test_a_block_holds_at_most_a_mebibyte_of_user_txs(self):
        big = tx(self.c, self.payers[0], "transfer", W.address, 1, memo="x" * ledger.MAX_BLOCK_TX_BYTES)
        _, dropped = seal(self.c, V, txs=[big])
        self.assertEqual(dropped, [big])
        forced = Block(self.c.tip.height + 1, self.c.tip.hash, self.c.tip.timestamp + 1, None, [big], self.c.chain_id,
                       version=VERSION)
        forced.sign(V)
        with self.assertRaisesRegex(InvalidBlock, "bytes"):
            self.c.append(forced)

    def test_build_block_stops_at_the_limit_and_the_rest_wait(self):
        txs = [tx(self.c, k, "transfer", W.address, 1) for k in self.payers]
        with mock.patch.object(ledger, "MAX_BLOCK_TX_BYTES", txs[0].size() * 5 // 2):
            block, dropped = seal(self.c, V, txs=txs)
            self.assertEqual((block.txs, dropped), (txs[:2], txs[2:]))
            block, dropped = seal(self.c, V, txs=dropped)
            self.assertEqual((block.txs, dropped), (txs[2:], []))

    def test_version_1_had_no_limit(self):
        big = tx(self.c, self.payers[0], "transfer", W.address, 1, memo="x" * ledger.MAX_BLOCK_TX_BYTES)
        self.assertEqual(seal(self.c, V, txs=[big], version=1)[1], [])


class FromDictTests(unittest.TestCase):
    """Malformed JSON from the network raises ValueError: before, some of it raised OverflowError or AttributeError."""

    def setUp(self):
        c = chain(stakes={V.address: 10 * VIBE, W.address: 10 * VIBE})
        seal(c, V, txs=[tx(c, REQ, "post_job", amount=JOB.budget, memo=JOB.spec_json())])
        sm = summary(V, {W.address: JOB.unit_flops}, accepted_units=1, slashes=[[V.address, None, "bad_reveal"]])
        sm.record_hash = "ab" * 32
        fee = Tx.signed(REQ, "transfer", W.address, 1, c.next_nonce(REQ.address), "rent", c.chain_id, fee=7)
        self.v2, _ = seal(c, V, sm, [fee])
        self.v1, _ = c.build_block(None, [], c.tip.timestamp + 1, version=1)

    def test_everything_to_dict_writes_reads_back_the_same(self):
        for block in (self.v1, self.v2):
            again = Block.from_dict(json.loads(json.dumps(block.to_dict())))
            self.assertEqual((again, again.hash), (block, block.hash))

    def test_malformed_fields_are_value_errors(self):
        good = self.v2.to_dict()
        t, sm = good["txs"][-1], good["summary"]
        bad_txs = [{"amount": "Infinity"}, {"amount": 1.0}, {"amount": "1"}, {"amount": True}, {"amount": -1},
                   {"fee": "Infinity"}, {"nonce": None}, {"memo": ["x"]}, {"pubkey": "zz"}, {"signature": 5}]
        bad_summaries = [{"worker_flops": []}, {"worker_flops": {"a": -1}}, {"verifier_flops": {"a": 1.5}},
                         {"slashes": [[1, 2]]}, {"slashes": [["a", None]]}, {"slashes": ["abc"]}, {"accepted": [1]},
                         {"accepted_units": -1}, {"record_hash": None}]
        bad_blocks = [{"height": 2**64}, {"height": -1}, {"timestamp": -1}, {"timestamp": 2**64}, {"timestamp": "5"},
                      {"version": 0}, {"version": "2"}, {"version": 2**64}, {"summary": []}, {"txs": {}},
                      {"prev_hash": None}, {"chain_id": 1}]
        cases = ([("tx", {**good, "txs": [*good["txs"][:-1], t | b]}) for b in bad_txs]
                 + [("summary", good | {"summary": sm | b}) for b in bad_summaries]
                 + [("block", good | b) for b in bad_blocks]
                 + [("missing", {k: v for k, v in good.items() if k != "chain_id"}), ("not an object", [])])
        for label, d in cases:
            with self.subTest(label, d=d), self.assertRaises(ValueError):
                Block.from_dict(json.loads(json.dumps(d).replace('"Infinity"', "Infinity")))
        with self.assertRaises(ValueError):
            Tx.from_dict(json.loads(json.dumps(t).replace(f'"amount": {t["amount"]}', '"amount": Infinity')))

    def test_flops_too_big_for_a_float_are_an_invalid_block(self):
        c = chain(stakes={V.address: 10 * VIBE, W.address: 10 * VIBE})
        block, _ = c.build_block(None, [], 1)
        block.summary = RoundSummary("j", 0, {}, {W.address: 10**400}, [], [], [], "00", V.address)
        block.sign(V)
        with self.assertRaisesRegex(InvalidBlock, "flops"):  # not OverflowError, which a node's vote didn't expect
            c.append(Block.from_dict(json.loads(json.dumps(block.to_dict()))))


class RelayPolicyTests(unittest.TestCase):
    def setUp(self):
        self.c = chain(stakes={V.address: 10 * VIBE})

    def test_notes_are_short(self):
        self.c.check_tx(tx(self.c, REQ, "transfer", W.address, 1, memo="x" * ledger.MAX_NOTE_BYTES), [])
        for kind, recipient, amount in (("transfer", W.address, 1), ("stake", SYSTEM, 1)):
            long = tx(self.c, REQ, kind, recipient, amount, memo="é" * (ledger.MAX_NOTE_BYTES // 2 + 1))
            with self.assertRaisesRegex(InvalidBlock, "note"):
                self.c.check_tx(long, [])
        self.assertEqual(seal(self.c, V, txs=[long])[1], [], "it's policy, not a rule: a block may carry one")

    def test_a_job_must_fund_a_round(self):
        for job, budgets in ((JOB, (0, JOB.fee_per_round - 1)), (TrainingJob("free", REQ.address, 0, 0), (0,))):
            for budget in budgets:
                with self.assertRaisesRegex(InvalidBlock, "at least one round"):
                    self.c.check_tx(tx(self.c, REQ, "post_job", amount=budget, memo=job.spec_json()), [])
            self.c.check_tx(tx(self.c, REQ, "post_job", amount=max(1, job.fee_per_round), memo=job.spec_json()), [])

    def test_nothing_oversized(self):
        big = tx(self.c, REQ, "transfer", "vibe1" + "0" * ledger.MAX_TX_BYTES, 1)  # the ledger doesn't bound recipients
        with self.assertRaisesRegex(InvalidBlock, "too big"):
            self.c.check_tx(big, [])


if __name__ == "__main__":
    unittest.main()


class ProposalVersionTests(unittest.TestCase):
    """Validators vote only for blocks and round records of the current versions (Node._validate), so a leader can't
    propose under the old rules to get round a fix. Older ones still apply when syncing or verifying a chain."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from vibehash.node import Node

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        genesis = {"chain_id": "rules-v2", "allocations": {REQ.address: 1000 * VIBE},
                   "validators": {V.address: "http://127.0.0.1:1"}, "bootstrap": {V.address: 10 * VIBE}}
        self.node = Node(Path(tmp.name) / "node", genesis, V, port=0)
        self.addCleanup(self.node.store.close)
        self.addCleanup(self.node.peers.close)
        self.addCleanup(self.node.server.server_close)

    def proposal(self, version: int, summary_=None, record=None) -> dict:
        c = self.node.chain
        block, _ = c.build_block(summary_, [tx(c, REQ, "transfer", W.address, VIBE)], c.tip.timestamp + 1, version=version)
        block.sign(V)
        return {"block": block.to_dict(), "record": record}

    def test_only_current_version_blocks_get_a_vote(self):
        self.assertTrue(self.node._validate(self.proposal(VERSION)))
        with self.assertLogs("vibehash.node", "WARNING") as logs:
            self.assertFalse(self.node._validate(self.proposal(1)))
        self.assertIn(f"must be version {VERSION}", logs.output[0])

    def test_only_current_version_records_get_a_vote(self):
        from vibehash.coordinator import RECORD_VERSION

        with self.assertLogs("vibehash.node", "WARNING") as logs:
            self.assertFalse(self.node._validate(self.proposal(VERSION, summary(V), {"version": 1})))
        self.assertIn(f"record must be version {RECORD_VERSION}", logs.output[0])
