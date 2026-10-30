import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import coordinator as coord_module
from vibehash import economics as econ
from vibehash import receipts
from vibehash.coordinator import Coordinator, CoordinatorConfig
from vibehash.crypto import KeyPair, h256, merkle_proof, merkle_root, merkle_verify, public_key, sign, verify
from vibehash.ledger import SYSTEM, UNBONDING_ROUNDS, Chain, InvalidBlock, RoundSummary, Tx, canonical
from vibehash.replay import replay_block
from vibehash.model import ModelSpec, params_hash, relative_distance
from vibehash.protocol import Protocol
from vibehash.tasks import SYN_PER_VIBE, TrainingJob, Verdict, WorkUnit
from vibehash.verification import check_openings, check_sanity, detection_probability, replay_openings, required_openings
from vibehash.worker import GTX_1060, LOW_END_LAPTOP, NETBOOK, OFFICE_DESKTOP, RTX_3060, HardwareProfile, Worker

RELIABLE = lambda hw: HardwareProfile(hw.name, hw.flops_per_sec, hw.memory_mb, reliability=1.0)
VIBE = SYN_PER_VIBE
# Every honest update is smaller than the 1e-4 tolerance (a segment moves ~1e-5, a unit ~7e-5), as for
# big or nearly trained models, so skipping all the work lands within tolerance of the honest result.
SLOW = dict(spec=ModelSpec(8, 32, 4, "float32"), lr=5e-6, tolerance=1e-4)


class CryptoTests(unittest.TestCase):
    def test_rfc8032_vector_1(self):
        sk = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        self.assertEqual(public_key(sk).hex(), "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        self.assertEqual(
            sign(sk, b"").hex(),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        )

    def test_rfc8032_vector_2(self):
        sk = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
        self.assertEqual(public_key(sk).hex(), "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
        self.assertTrue(verify(public_key(sk), b"\x72", sign(sk, b"\x72")))

    def test_pure_python_matches_backend(self):
        from vibehash import crypto
        k = KeyPair.generate(b"x")
        self.assertEqual(crypto._py_sign(k.secret, b"m"), k.sign(b"m"))
        self.assertTrue(crypto._py_verify(k.public, b"m", k.sign(b"m")))

    def test_signature_rejects_tampering(self):
        k = KeyPair.generate(b"t")
        sig = k.sign(b"pay 5")
        self.assertTrue(verify(k.public, b"pay 5", sig))
        self.assertFalse(verify(k.public, b"pay 6", sig))
        self.assertFalse(verify(KeyPair.generate(b"u").public, b"pay 5", sig))
        self.assertFalse(verify(k.public, b"pay 5", sig[:-1]))

    def test_merkle_proofs(self):
        leaves = [bytes([i]) * 3 for i in range(9)]
        root = merkle_root(leaves)
        for i, leaf in enumerate(leaves):
            self.assertTrue(merkle_verify(leaf, merkle_proof(leaves, i), root))
        self.assertFalse(merkle_verify(b"forged", merkle_proof(leaves, 4), root))


class EconomicsTests(unittest.TestCase):
    def test_subsidy_halves(self):
        self.assertEqual(econ.block_subsidy(1), econ.INITIAL_SUBSIDY)
        self.assertEqual(econ.block_subsidy(econ.HALVING_INTERVAL), econ.INITIAL_SUBSIDY // 2)
        self.assertEqual(econ.block_subsidy(econ.HALVING_INTERVAL * 64), 0)

    def test_split_is_exact(self):
        out = econ.split_pro_rata(1_000_003, {"a": 1, "b": 1, "c": 1, "d": 0})
        self.assertEqual(sum(out.values()), 1_000_003)
        self.assertNotIn("d", out)

    def test_payout_conserves_pool(self):
        p = econ.compute_payout(6_500_000_007, {"a": 300, "b": 100}, {"c": 50}, {"a": 1.0, "b": 0.5})
        self.assertEqual(sum(p.worker_pay.values()) + sum(p.verifier_pay.values()) + p.proposer_pay + p.burned, p.pool)
        self.assertGreater(p.worker_pay["a"], 5 * p.worker_pay["b"])

    def test_no_verified_work_means_pool_is_burned(self):
        p = econ.compute_payout(1000, {}, {}, {})
        self.assertEqual((p.proposer_pay, p.burned), (0, 1000))

    def test_reputation_bounds(self):
        self.assertAlmostEqual(econ.update_reputation(0.8, True, False, False), 0.82)
        self.assertEqual(econ.update_reputation(0.99, True, False, False), econ.REP_MAX)
        self.assertEqual(econ.update_reputation(0.52, False, True, False), econ.REP_MIN)
        self.assertEqual(econ.update_reputation(1.0, True, False, True), econ.REP_MIN)


def small_job(job_id="j", budget=0, **kw) -> TrainingJob:
    return TrainingJob(job_id, "req", budget=budget, fee_per_round=kw.pop("fee_per_round", 10 * VIBE), units_per_round=kw.pop("units", 12), **kw)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.alice, self.bob = KeyPair.generate(b"alice"), KeyPair.generate(b"bob")
        self.chain = Chain({self.alice.address: 100 * VIBE}, stakes={self.alice.address: 10 * VIBE})

    def seal(self, txs, summary=None, signer=None):
        block, dropped = self.chain.build_block(summary, txs, self.chain.tip.height + 1)
        block.sign(signer or self.alice)
        self.chain.append(block)
        return block, dropped

    def tx(self, kind, recipient, amount, memo="", keys=None, chain_id=None):
        keys = keys or self.alice
        return Tx.signed(keys, kind, recipient, amount, self.chain.next_nonce(keys.address), memo, chain_id or self.chain.chain_id)

    def test_signed_transfer(self):
        self.seal([self.tx("transfer", self.bob.address, 40)])
        self.assertEqual(self.chain.state.balance(self.bob.address), 40)
        self.assertTrue(self.chain.validate())

    def test_build_block_drops_invalid_txs(self):
        good = self.tx("transfer", self.bob.address, 40)
        forged = Tx("transfer", self.alice.address, self.bob.address, 40, 1, pubkey=self.bob.public, signature=self.bob.sign(b"x"))
        _, dropped = self.seal([good, forged])
        self.assertEqual(dropped, [forged])

    def test_rejects_cross_chain_replay(self):
        other = self.tx("transfer", self.bob.address, 40, chain_id="some-other-chain")
        with self.assertRaises(InvalidBlock):
            self.chain.check_tx(other, [])

    def test_rejects_replay_and_overspend(self):
        tx = self.tx("transfer", self.bob.address, 40)
        self.seal([tx])
        with self.assertRaises(InvalidBlock):
            self.chain.check_tx(tx, [])
        with self.assertRaises(InvalidBlock):
            self.chain.check_tx(self.tx("transfer", self.bob.address, 100 * VIBE), [])

    def test_proposer_cannot_inject_rewards(self):
        block, _ = self.chain.build_block(None, [], 1)
        block.txs.append(Tx("reward", SYSTEM, self.alice.address, 1, chain_id=self.chain.chain_id))
        block.sign(self.alice)
        with self.assertRaises(InvalidBlock):
            self.chain.append(block)

    def test_proposer_cannot_inflate_summary_rewards(self):
        self.seal([self.tx("post_job", SYSTEM, 50 * VIBE, small_job().spec_json())])
        summary = RoundSummary("j", 12, {self.alice.address: 10**12}, {}, [], [self.alice.address], [], "00", self.alice.address)
        with self.assertRaises(InvalidBlock):  # more work than the round could contain
            self.seal([], summary)

    def test_unstake_unbonds_and_stays_slashable(self):
        self.seal([self.tx("unstake", SYSTEM, 4 * VIBE)])
        st = self.chain.state
        self.assertEqual((st.stake(self.alice.address), st.unbonding_total(self.alice.address)), (6 * VIBE, 4 * VIBE))
        before = st.balance(self.alice.address)
        for _ in range(UNBONDING_ROUNDS - 1):
            self.seal([])
        self.assertEqual(self.chain.state.balance(self.alice.address), before)
        self.seal([])
        self.assertEqual(self.chain.state.balance(self.alice.address), before + 4 * VIBE)
        self.assertTrue(self.chain.validate())

    def test_post_and_cancel_job_refunds_escrow(self):
        self.seal([self.tx("post_job", SYSTEM, 30 * VIBE, small_job().spec_json())])
        self.assertEqual(self.chain.state.escrow["j"], 30 * VIBE)
        with self.assertRaises(InvalidBlock):  # only the owner may cancel
            self.chain.check_tx(self.tx("cancel_job", SYSTEM, 0, "j", keys=self.bob), [])
        self.seal([self.tx("cancel_job", SYSTEM, 0, "j")])
        self.assertEqual(self.chain.state.escrow["j"], 0)
        self.assertEqual(self.chain.state.balance(self.alice.address), 100 * VIBE)

    def test_rejects_malformed_job(self):
        bad = small_job().to_spec() | {"checkpoint_every": 7}
        import json
        with self.assertRaises(InvalidBlock):
            self.chain.check_tx(self.tx("post_job", SYSTEM, VIBE, json.dumps(bad)), [])

    def test_unstaked_proposer_rejected(self):
        block, _ = self.chain.build_block(None, [], 1)
        block.sign(self.bob)
        with self.assertRaises(InvalidBlock):
            self.chain.append(block)

    def test_tampering_breaks_validation(self):
        self.seal([self.tx("transfer", self.bob.address, 10)])
        self.seal([])
        self.assertTrue(self.chain.validate())
        t = self.chain.blocks[1].txs[0]
        self.chain.blocks[1].txs[0] = Tx(t.kind, t.sender, t.recipient, 10_000, t.nonce, t.memo, t.pubkey, t.signature, t.chain_id)
        self.assertFalse(self.chain.validate())


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.job = small_job()
        self.start = self.job.spec.init_params(self.job.init_seed)
        self.unit = WorkUnit.make(self.job, 1, 0, params_hash(self.start))
        self.beacon = b"beacon"

    def submit(self, behavior, q=3):
        w = Worker(behavior, KeyPair.generate(behavior.encode()), GTX_1060, behavior)
        w.compute(self.unit, self.job, self.start)
        return w.reveal(self.unit, self.job, self.beacon, q)

    def test_honest_passes_every_segment(self):
        q = self.job.n_segments
        r = self.submit("honest", q)
        self.assertTrue(check_openings(self.job, r, required_openings(self.job, self.beacon, self.unit.unit_id, r.worker, q)))
        self.assertTrue(check_sanity(self.job, self.unit, self.start, r))
        self.assertTrue(replay_openings(self.job, self.unit, r.openings, list(range(q)))[0])

    def test_lazy_fails_replay(self):
        r = self.submit("lazy", self.job.n_segments)
        self.assertTrue(check_sanity(self.job, self.unit, self.start, r))  # looks plausible...
        self.assertFalse(replay_openings(self.job, self.unit, r.openings, [0])[0])  # ...but isn't

    def test_freezer_fails_replay_even_when_updates_are_below_tolerance(self):
        self.job = small_job(**SLOW)
        self.start = self.job.spec.init_params(self.job.init_seed)
        self.unit = WorkUnit.make(self.job, 1, 0, params_hash(self.start))
        n = self.job.n_segments
        honest = self.submit("honest", n)
        self.assertLess(relative_distance(honest.final_params, self.start), self.job.tolerance)
        self.assertTrue(replay_openings(self.job, self.unit, honest.openings, list(range(n)))[0])
        frozen = self.submit("freezer", n)
        self.assertTrue(check_sanity(self.job, self.unit, self.start, frozen))
        self.assertFalse(replay_openings(self.job, self.unit, frozen.openings, [0])[0])

    def test_missing_opening_rejected(self):
        r = self.submit("honest")
        need = required_openings(self.job, self.beacon, self.unit.unit_id, r.worker, 3)
        del r.openings[need[1]]
        self.assertFalse(check_openings(self.job, r, need))

    def test_poisoner_fails_sanity(self):
        self.assertFalse(check_sanity(self.job, self.unit, self.start, self.submit("poisoner")))

    def test_shortcut_caught_only_on_faked_segments(self):
        n = self.job.n_segments
        r = self.submit("shortcut", n)
        ok, bad = replay_openings(self.job, self.unit, r.openings, list(range(n)))
        self.assertFalse(ok)
        self.assertGreaterEqual(bad, n // 2)

    def test_detection_probability(self):
        self.assertAlmostEqual(detection_probability(0.5, 3), 0.875)
        self.assertEqual(detection_probability(1.0, 1), 1.0)


def network(behaviors, hardware=None, seed=3, job_kw=None, **cfg):
    hardware = hardware or [RELIABLE(OFFICE_DESKTOP)] * len(behaviors)
    workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, *b) for i, (b, hw) in enumerate(zip(behaviors, hardware))]
    req = KeyPair.generate(b"req")
    chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers})
    proto = Protocol(chain, workers, CoordinatorConfig(**cfg), seed=seed)
    job = TrainingJob("job", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=12, **(job_kw or {}))
    proto.post_job(req, job)
    return proto, workers, job


class ReplayTests(unittest.TestCase):
    """Every round can be re-run from its record with the same code, and a coordinator's lies are caught."""

    def run_rounds(self, behaviors, n=1, hardware=None, **cfg):
        proto, workers, job = network(behaviors, hardware, **cfg)
        before = copy.deepcopy(proto.chain)
        return proto, workers, before, [proto.run_round() for _ in range(n)]

    @staticmethod
    def replay_all(before, reports, edit=None):
        """Problems found in each round, replayed on the chain as it stood just before that round's block."""
        chain, found = copy.deepcopy(before), []
        for i, rep in enumerate(reports):
            block, record = rep.block, copy.deepcopy(rep.record)
            if edit:
                block, record = edit(i, block, record)
            found.append(replay_block(chain, block, record))
            chain.append(rep.block)
        return found

    @staticmethod
    def resealed(block, record):
        """What a careful liar publishes with an edited record: a block whose record hash matches it."""
        return dataclasses.replace(block, summary=dataclasses.replace(block.summary, record_hash=h256(canonical(record)).hex()))

    def test_every_round_replays(self):
        behaviors = [("honest",), ("honest",), ("honest", "lazy"), ("lazy",), ("shortcut",), ("copycat",), ("poisoner",),
                     ("withholder",), ("honest", "false_accuser"), ("freezer",)]
        proto, workers, before, reports = self.run_rounds(behaviors, n=4, replication_rate=0.5)
        self.assertEqual(self.replay_all(before, reports), [[]] * 4)
        seen = {res.verdict for rep in reports for res in rep.results}  # the rounds went down every path that matters
        self.assertLessEqual({Verdict.ACCEPTED, Verdict.FAILED_REPLAY, Verdict.BAD_REVEAL, Verdict.FAILED_SANITY, Verdict.WITHHELD}, seen)

    def recheck_all(self, proto, before, reports):
        chain, found, stats = copy.deepcopy(before), [], {}
        for rep in reports:
            found.append(replay_block(chain, rep.block, rep.record, proto.checkpoints, stats))
            chain.append(rep.block)
        return found, stats

    def test_every_weight_based_result_rechecks(self):
        behaviors = [("honest",), ("honest",), ("honest", "lazy"), ("lazy",), ("shortcut",), ("copycat",), ("poisoner",),
                     ("withholder",), ("honest", "false_accuser"), ("freezer",)]
        proto, workers, before, reports = self.run_rounds(behaviors, n=3, replication_rate=0.5)
        found, stats = self.recheck_all(proto, before, reports)
        self.assertEqual(found, [[]] * 3)
        self.assertGreater(stats["rechecked"], 30)
        self.assertEqual(stats["unchecked"], 0)

    def test_false_sanity_failure_is_caught(self):
        """A coordinator that slashes an honest worker by claiming its result failed the sanity check."""
        proto, workers, job = network([("honest",)] * 3, replication_rate=0, canaries_per_round=0)
        victim, before = workers[0].address, copy.deepcopy(proto.chain)
        honest = coord_module.check_sanity
        with mock.patch.object(coord_module, "check_sanity", lambda j, u, s, rv: rv.worker != victim and honest(j, u, s, rv)):
            report = proto.run_round()
        self.assertIn(victim, {s[0] for s in report.slashes})
        self.assertEqual(self.replay_all(before, [report]), [[]])  # self-consistent: a replay alone can't tell
        found, _ = self.recheck_all(proto, before, [report])
        self.assertTrue(found[0] and all("the record says 'failed_sanity', re-running it gives 'ok'" in p for p in found[0]), found)

    def test_siding_with_a_false_accuser_is_caught(self):
        """A coordinator that upholds a false accusation, slashing the honest worker instead of the accuser."""
        proto, workers, job = network([("honest",), ("honest", "false_accuser")], replication_rate=0, canaries_per_round=0)
        before = copy.deepcopy(proto.chain)
        with mock.patch.object(Coordinator, "_arbitrate", lambda self, job, t, bad: False):
            report = proto.run_round()
        self.assertIn(workers[0].address, {s[0] for s in report.slashes})
        found, _ = self.recheck_all(proto, before, [report])
        self.assertTrue(found[0] and all(p.startswith("arbitration check") and "re-running it gives True" in p for p in found[0]), found)

    def test_pruned_checkpoints_mean_unchecked_not_failed(self):
        proto, workers, before, reports = self.run_rounds([("honest",)] * 3, replication_rate=0)
        proto.checkpoints.arrays.clear()  # outside the challenge window: the node has deleted them
        found, stats = self.recheck_all(proto, before, reports)
        self.assertEqual(found, [[]])
        self.assertEqual(stats["rechecked"], 0)
        self.assertGreater(stats["unchecked"], 0)

    def test_round_with_a_caught_lazy_checker_replays(self):
        # The only worker gets every planted fake to check, approves it, and is slashed for it.
        proto, workers, before, reports = self.run_rounds([("honest", "lazy")], canaries_per_round=2)
        self.assertIn("approved a canary", {s[2] for s in reports[0].slashes})
        self.assertEqual(self.replay_all(before, reports), [[]])

    def test_edited_record_doesnt_match_its_block(self):
        proto, workers, before, reports = self.run_rounds([("honest",)] * 3)

        def edit(i, block, record):
            record["commitments"].pop()
            return block, record
        self.assertIn("doesn't match the hash", self.replay_all(before, reports, edit)[0][0])

    def test_forged_verdict_is_caught(self):
        proto, workers, before, reports = self.run_rounds([("honest",)] * 4, canaries_per_round=0)

        def forge(i, block, record):  # turn an approval into a rejection, to slash an honest worker
            env = record["verdicts"][0]
            env["body"] = dict(env["body"], ok=False, bad_segment=0)
            return self.resealed(block, record), record
        self.assertIn("bad signature", self.replay_all(before, reports, forge)[0][0])

    def test_invented_registration_is_caught(self):
        proto, workers, before, reports = self.run_rounds([("honest",)] * 3)

        def invent(i, block, record):
            record["registrations"].append({"address": workers[0].address, "flops_per_sec": 1e9, "memory_mb": 8192})
            return self.resealed(block, record), record
        self.assertIn("isn't signed", self.replay_all(before, reports, invent)[0][0])

    def test_favouring_a_worker_is_caught(self):
        """A coordinator that quietly gives one worker more of the work than the rules allow."""
        behaviors, crony = [("honest",)] * 4, None
        proto, workers, job = network(behaviors, replication_rate=0, canaries_per_round=0)
        crony, before = workers[0].address, copy.deepcopy(proto.chain)
        honest_rule = Coordinator.capacity
        with mock.patch.object(Coordinator, "capacity", lambda self, a, claim, job: claim if a == crony else honest_rule(self, a, claim, job)):
            report = proto.run_round()
        self.assertGreater(report.worker_flops[crony], max(f for a, f in report.worker_flops.items() if a != crony))
        problems = self.replay_all(before, [report])[0]
        self.assertTrue(problems and "not assigned" in problems[0], problems)

    def test_rewinding_the_model_is_caught(self):
        proto, workers, before, reports = self.run_rounds([("honest",)] * 3, n=2)

        def rewind(i, block, record):  # train round 2 from the starting weights again, discarding round 1's work
            if i == 1:
                record["start_hash"] = reports[0].record["start_hash"]
                block = self.resealed(block, record)
            return block, record
        problems = self.replay_all(before, reports, rewind)
        self.assertEqual(problems[0], [])
        self.assertIn("latest model", problems[1][0])


class ReceiptTests(unittest.TestCase):
    """A node's signed receipt turns a dropped message into proof."""

    def setUp(self):
        self.node = KeyPair.generate(b"node")
        self.entry = {"unit_id": "u1", "worker": "vibe1abc", "digest": "00"}
        self.receipt = receipts.make(self.node, "net", 7, "attempt-1", "commit", self.entry)
        self.block = SimpleNamespace(summary=object(), proposer_pubkey=self.node.public)

    def status(self, block, record, receipt=None):
        return receipts.audit(receipt or self.receipt, block, record)[0]

    def test_statuses(self):
        recorded = {"attempt": "attempt-1", "commitments": [{"other": 1}, self.entry]}
        self.assertEqual(self.status(self.block, recorded), "recorded")
        self.assertEqual(self.status(self.block, {"attempt": "attempt-1", "commitments": [{"other": 1}]}), "censored")
        self.assertEqual(self.status(self.block, {"attempt": "attempt-2", "commitments": []}), "abandoned")  # round re-run
        self.assertEqual(self.status(SimpleNamespace(summary=None, proposer_pubkey=self.node.public), None), "abandoned")
        self.assertEqual(self.status(None, None), "pending")
        other = SimpleNamespace(summary=object(), proposer_pubkey=KeyPair.generate(b"other").public)
        self.assertEqual(self.status(other, {"attempt": "attempt-1", "commitments": []}), "unrelated")
        forged = dict(self.receipt, digest=receipts.entry_digest({"something": "else"}))
        self.assertEqual(self.status(self.block, recorded, forged), "invalid")

    def test_digest_ignores_key_order(self):
        self.assertEqual(receipts.entry_digest({"a": 1, "b": 2}), receipts.entry_digest({"b": 2, "a": 1}))


class ProtocolTests(unittest.TestCase):
    def verdicts_for(self, report, worker):
        return {r.verdict for r in report.results if r.worker == worker.address}

    def test_each_cheater_gets_the_right_verdict(self):
        cases = {"lazy": Verdict.FAILED_REPLAY, "shortcut": Verdict.FAILED_REPLAY, "freezer": Verdict.FAILED_REPLAY,
                 "copycat": Verdict.BAD_REVEAL, "poisoner": Verdict.FAILED_SANITY}
        for behavior, expected in cases.items():
            with self.subTest(behavior=behavior):
                proto, workers, job = network([("honest",)] * 3 + [(behavior,)], replication_rate=0, canaries_per_round=0)
                report = proto.run_round()
                cheater = workers[-1]
                self.assertIn(expected, self.verdicts_for(report, cheater))
                self.assertIn(cheater.address, {s[0] for s in report.slashes})
                self.assertNotIn(cheater.address, report.worker_flops)
                self.assertEqual(proto.chain.state.stake(cheater.address), 5 * VIBE)
                for w in workers[:3]:
                    self.assertEqual(self.verdicts_for(report, w), {Verdict.ACCEPTED})

    def test_withholding_forfeits_round_without_slash(self):
        """Committing then going silent looks exactly like a crash, so it costs the round, not the stake."""
        proto, workers, job = network([("honest",)] * 3 + [("withholder",)], replication_rate=0, canaries_per_round=0)
        report = proto.run_round()
        w = workers[-1]
        self.assertEqual(self.verdicts_for(report, w), {Verdict.WITHHELD})
        self.assertNotIn(w.address, {s[0] for s in report.slashes})
        self.assertNotIn(w.address, report.worker_flops)
        self.assertNotIn(w.address, report.verifier_flops)
        self.assertEqual(proto.chain.state.stake(w.address), 10 * VIBE)
        self.assertLess(proto.chain.state.rep(w.address), econ.REP_START)
        self.assertEqual(proto.coord.track_record(w.address), (0, True))  # the chain shows nothing delivered: back to the newcomer cap

    def test_freezer_caught_when_updates_are_below_tolerance(self):
        proto, workers, job = network([("honest",)] * 3 + [("freezer",)], replication_rate=0, canaries_per_round=0, job_kw=SLOW)
        report = proto.run_round()
        self.assertEqual(self.verdicts_for(report, workers[-1]), {Verdict.FAILED_REPLAY})
        for w in workers[:3]:
            self.assertEqual(self.verdicts_for(report, w), {Verdict.ACCEPTED})

    def test_frozen_replica_caught_without_replays(self):
        """With every replay skipped (lazy verifiers), replicas of the same unit must still disagree."""
        proto, workers, job = network([("honest", "lazy"), ("freezer", "lazy"), ("honest", "lazy")],
                                      replication_rate=1.0, canaries_per_round=0, job_kw=SLOW)
        report = proto.run_round()
        self.assertIn(Verdict.FAILED_REDUNDANCY, self.verdicts_for(report, workers[1]))
        self.assertNotIn(workers[1].address, report.worker_flops)  # fraud forfeits the whole round
        for w in (workers[0], workers[2]):
            self.assertEqual(self.verdicts_for(report, w), {Verdict.ACCEPTED})

    def test_jobs_take_turns(self):
        """A job posted later starts at the next round, not when the earlier job's budget runs out."""
        proto, workers, job = network([("honest",)] * 3, replication_rate=0, canaries_per_round=0)
        req = KeyPair.generate(b"req")
        proto.post_job(req, TrainingJob("second", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=12))
        self.assertEqual([proto.run_round().job_id for _ in range(4)], ["job", "second", "job", "second"])

    def test_copycat_on_replicated_unit_still_fails(self):
        proto, workers, job = network([("honest",), ("copycat",)], replication_rate=1.0, canaries_per_round=0)
        self.assertEqual(self.verdicts_for(proto.run_round(), workers[1]), {Verdict.BAD_REVEAL})

    def test_lazy_verifier_caught_by_canary(self):
        # Which checker gets each fake depends on the beacon; as the only worker, the lazy one gets them all.
        proto, workers, job = network([("honest", "lazy")], canaries_per_round=2)
        report = proto.run_round()
        self.assertIn(("approved a canary", workers[0].address), {(s[2], s[0]) for s in report.slashes})

    def test_false_accusation_overturned(self):
        proto, workers, job = network([("honest",), ("honest", "false_accuser")], canaries_per_round=0)
        report = proto.run_round()
        self.assertEqual(self.verdicts_for(report, workers[0]), {Verdict.ACCEPTED})
        self.assertEqual({s[0] for s in report.slashes}, {workers[1].address})

    def test_memory_too_small_gets_no_units(self):
        proto, workers, job = network([("honest",), ("honest",)], hardware=[RELIABLE(OFFICE_DESKTOP), NETBOOK])
        self.assertEqual(self.verdicts_for(proto.run_round(), workers[1]), set())

    def test_newcomers_capped_then_ramp_up(self):
        proto, workers, job = network([("honest",)] * 2, hardware=[RELIABLE(RTX_3060)] * 2, replication_rate=0,
                                      canaries_per_round=0, newcomer_units=2)
        # 2 units each, then twice what each delivered (4), then 8 each, which covers all 12.
        self.assertEqual([proto.run_round().unassigned for _ in range(3)], [job.units_per_round - 4, job.units_per_round - 8, 0])

    def test_missing_a_deadline_backs_off_to_what_was_delivered(self):
        proto, workers, job = network([("honest",)] * 2, hardware=[RELIABLE(RTX_3060)] * 2, replication_rate=0,
                                      canaries_per_round=0, newcomer_units=2)
        for _ in range(2):
            proto.run_round()
        w = workers[0]
        delivered, missed = proto.coord.track_record(w.address)
        self.assertEqual((delivered, missed), (4 * job.unit_flops, False))
        self.assertEqual(proto.coord.capacity(w.address, 10**18, job), 8 * job.unit_flops)  # delivered in full: may double
        w.hardware = HardwareProfile(w.hardware.name, w.hardware.flops_per_sec, w.hardware.memory_mb, reliability=0.0)
        proto.run_round()  # it drops out mid-round: every unit misses the deadline
        self.assertEqual(proto.coord.track_record(w.address), (0, True))
        self.assertEqual(proto.coord.capacity(w.address, 10**18, job), 2 * job.unit_flops)  # back to the newcomer cap

    def test_low_and_high_end_paid_the_same_rate(self):
        hw = [RELIABLE(LOW_END_LAPTOP), RELIABLE(OFFICE_DESKTOP), RELIABLE(RTX_3060)]
        proto, workers, job = network([("honest",)] * 3, hardware=hw, canaries_per_round=0)
        flops = {w.address: 0 for w in workers}
        for _ in range(4):
            r = proto.run_round()
            rates = [r.worker_pay[a] / f for a, f in r.worker_flops.items()]
            self.assertLess(max(rates) / min(rates), 1.0001)  # same pay per FLOP within every round
            for a, f in r.worker_flops.items():
                flops[a] += f
        self.assertGreater(flops[workers[2].address], 3 * flops[workers[0].address])  # big rig just does more

    def test_model_improves_and_supply_balances(self):
        proto, workers, job = network([("honest",)] * 3 + [("lazy",), ("honest", "lazy")])
        first = proto.run_round()
        for _ in range(3):
            last = proto.run_round()
        self.assertLess(last.val_loss, first.val_loss)
        st = proto.chain.state
        self.assertEqual(st.total_supply(), st.expected_supply())
        self.assertTrue(proto.chain.validate())
        self.assertLess(st.escrow["job"], job.budget)

    def test_repeat_offender_loses_eligibility(self):
        proto, workers, job = network([("honest",)] * 2 + [("lazy",)], replication_rate=0, canaries_per_round=0)
        proto.run_round()
        proto.run_round()
        self.assertFalse(proto.eligible(workers[2].address))
        self.assertEqual(self.verdicts_for(proto.run_round(), workers[2]), set())


if __name__ == "__main__":
    unittest.main()
