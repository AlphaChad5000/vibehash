"""A fair start: workers who begin with no stake earn it by working, and the first validators vote with bootstrap
stake that can never be spent. Every round still replays from its record."""
import unittest

from vibehash import economics as econ
from vibehash.coordinator import CoordinatorConfig, ProtocolError
from vibehash.crypto import KeyPair
from vibehash.ledger import SYSTEM, Chain, InvalidBlock, Tx
from vibehash.protocol import Protocol
from vibehash.replay import replay_block
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker

VIBE = SYN_PER_VIBE
MIN = econ.MIN_STAKE
RELIABLE = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)
REQ = KeyPair.generate(b"earn-in requester")


def genesis(workers, bootstrapped, earn_in=True):
    return dict(allocations={REQ.address: 1000 * VIBE}, stakes={}, chain_id="earn-in-test", genesis_time=0,
                bootstrap={w.address: 10 * VIBE for w in bootstrapped}, earn_in=earn_in)


def network(newcomers=3, behaviors=None, units=12, **cfg):
    """One worker with bootstrap stake (someone has to propose blocks and check work) and `newcomers` with nothing."""
    staked = [Worker("s0", KeyPair.generate(b"earn-in staked"), RELIABLE, "honest")]
    behaviors = behaviors or ["honest"] * newcomers
    new = [Worker(f"n{i}", KeyPair.generate(f"earn-in new {i}".encode()), RELIABLE, b) for i, b in enumerate(behaviors)]
    g = genesis([*staked, *new], staked)
    proto = Protocol(Chain(**g), [*staked, *new], CoordinatorConfig(canaries_per_round=0, **cfg), seed=3)
    proto.post_job(REQ, TrainingJob("earn-job", REQ.address, budget=500 * VIBE, fee_per_round=10 * VIBE, units_per_round=units))
    return proto, staked, new, g


def replays(test, proto, g):
    """Every block re-checked from genesis, and every round re-run from its record."""
    chain, records = Chain(**g), {r.block.height: r.record for r in proto.history}
    for block in proto.chain.blocks[1:]:
        if block.summary is not None:
            test.assertEqual(replay_block(chain, block, records[block.height]), [], f"block {block.height}")
        chain.append(block)
    test.assertTrue(chain.validate())


class EarnInTests(unittest.TestCase):
    def test_newcomers_start_with_nothing_and_their_first_pay_becomes_stake(self):
        proto, staked, new, g = network()
        s = proto.chain.state
        self.assertTrue(all(s.stake(w.address) == 0 and s.balance(w.address) == 0 for w in new))
        report = proto.run_round()
        s = proto.chain.state
        for w in new:
            pay = report.worker_pay[w.address]
            self.assertGreater(pay, 0, "a newcomer was given work and paid for it")
            self.assertEqual(s.stake(w.address), min(pay, MIN))
            self.assertEqual(s.balance(w.address), pay - min(pay, MIN))
        self.assertEqual(set(report.verifier_pay), {staked[0].address}, "only staked workers check others' work")
        for _ in range(3):
            proto.run_round()
        s = proto.chain.state
        self.assertTrue(all(s.staked(w.address) for w in new), "a few rounds in, every newcomer is fully staked")
        self.assertTrue(set(proto.history[-1].verifier_pay) - {staked[0].address}, "and then they check work too")
        self.assertEqual(s.total_supply(), s.expected_supply())
        replays(self, proto, g)

    def test_a_newcomer_that_fakes_its_work_is_caught_and_keeps_nothing(self):
        proto, staked, new, g = network(behaviors=["honest", "lazy", "honest"])
        cheat = new[1].address
        report = proto.run_round()
        self.assertIn(cheat, {o for o, _, _ in report.slashes}, "a staked worker checked its units and caught it")
        self.assertNotIn(cheat, report.worker_pay)
        report = proto.run_round()  # the honest newcomers are staked now, and staked workers get the work first
        self.assertNotIn(cheat, report.worker_pay)
        s = proto.chain.state
        self.assertEqual((s.stake(cheat), s.balance(cheat)), (0, 0))
        self.assertEqual(s.rep(cheat), econ.REP_MIN)
        self.assertTrue(all(s.staked(w.address) for w in (new[0], new[2])))
        replays(self, proto, g)

    def test_staked_workers_get_work_first_and_newcomers_take_whats_left(self):
        proto, staked, new, g = network(newcomers=5, units=6)
        c = proto.coord
        c.open_round()
        for w in proto.workers.values():
            c.register(w.address, w.hardware.flops_per_sec, w.hardware.memory_mb)
        c.start_compute()
        cap = c.config.newcomer_units
        self.assertEqual(len(c.r.assignments[staked[0].address]), cap, "the staked worker is filled to its limit first")
        self.assertLessEqual(sum(len(c.r.assignments.get(w.address, [])) for w in new), 6 - cap + 2)  # + replicas
        self.assertTrue(all(len(c.r.assignments.get(w.address, [])) <= cap for w in new))

    def test_only_so_many_newcomers_a_round(self):
        proto, staked, new, g = network(newcomers=3, max_newcomers=2)
        c = proto.coord
        c.open_round()
        c.register(new[0].address, 1e9, 8192)
        c.register(new[1].address, 1e9, 8192)
        with self.assertRaisesRegex(ProtocolError, "workers without stake"):
            c.register(new[2].address, 1e9, 8192)
        c.register(staked[0].address, 1e9, 8192)  # staked workers aren't counted

    def test_without_earn_in_an_unstaked_worker_gets_no_work(self):
        w = Worker("n", KeyPair.generate(b"no earn-in"), RELIABLE, "honest")
        s0 = Worker("s", KeyPair.generate(b"no earn-in staked"), RELIABLE, "honest")
        proto = Protocol(Chain(**genesis([w, s0], [s0], earn_in=False)), [w, s0], CoordinatorConfig(canaries_per_round=0))
        proto.post_job(REQ, TrainingJob("j", REQ.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=4))
        self.assertFalse(proto.eligible(w.address))
        proto.coord.open_round()
        with self.assertRaisesRegex(ProtocolError, "stake at least"):
            proto.coord.register(w.address, 1e9, 8192)


class BootstrapStakeTests(unittest.TestCase):
    def setUp(self):
        self.v1, self.v2, self.other = (KeyPair.generate(s) for s in (b"boot-v1", b"boot-v2", b"boot-other"))
        self.chain = Chain({self.v1.address: 3 * VIBE}, {}, "boot-test", 0,
                           validators={self.v1.address: "http://10.0.0.1:8545", self.v2.address: "http://10.0.0.2:8545"},
                           bootstrap={self.v1.address: 10 * VIBE, self.v2.address: 10 * VIBE}, earn_in=True)

    def seal(self, txs, signer=None):
        block, dropped = self.chain.build_block(None, txs, self.chain.tip.timestamp + 1)
        block.sign(signer or self.v1)
        self.chain.append(block)
        return dropped

    def tx(self, keys, kind, amount=0, memo=""):
        return Tx.signed(keys, kind, SYSTEM, amount, self.chain.next_nonce(keys.address), memo, self.chain.chain_id)

    def test_it_votes_but_can_never_be_withdrawn(self):
        s = self.chain.state
        self.assertEqual(s.validator_set(), {self.v1.address: 10 * VIBE, self.v2.address: 10 * VIBE})
        self.assertEqual(len(self.seal([self.tx(self.v1, "unstake", 1)])), 1, "not even a little of it")
        self.assertEqual(self.seal([self.tx(self.v1, "stake", 2 * VIBE)]), [])
        self.assertEqual(self.seal([self.tx(self.v1, "unstake", 2 * VIBE)]), [], "stake added on top is free to leave")
        self.assertEqual(self.chain.state.stake(self.v1.address), 10 * VIBE)

    def test_a_validator_that_leaves_burns_it(self):
        before = self.chain.state.burned
        self.assertEqual(self.seal([self.tx(self.v2, "validator")]), [])
        s = self.chain.state
        self.assertEqual((s.stake(self.v2.address), s.locked.get(self.v2.address)), (0, None))
        self.assertEqual(s.burned - before, 10 * VIBE)
        self.assertEqual(s.total_supply(), s.expected_supply())
        self.assertTrue(self.chain.validate())

    def test_the_genesis_hash_commits_to_the_fair_start_rules(self):
        base = dict(allocations={}, stakes={self.v1.address: 10 * VIBE}, chain_id="h", genesis_time=0)
        hashes = {Chain(**base).genesis_hash, Chain(**base, earn_in=True).genesis_hash,
                  Chain(**base, bootstrap={self.v2.address: 10 * VIBE}).genesis_hash}
        self.assertEqual(len(hashes), 3)
        self.assertEqual(Chain(**base).genesis_hash, Chain.from_genesis(
            {"chain_id": "h", "timestamp": 0, "allocations": {}, "stakes": {self.v1.address: 10 * VIBE}}).genesis_hash)
        with self.assertRaisesRegex(InvalidBlock, "only one genesis"):
            self.chain.append(self.chain.blocks[0])  # its allocations can't be paid out twice


if __name__ == "__main__":
    unittest.main()
