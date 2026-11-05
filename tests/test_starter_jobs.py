"""Starter jobs (ledger.GENESIS_JOB): a job in the genesis, owned by nobody and paying only new coins, so a network with
no starting coins can still mine. Rounds through the same Coordinator a node runs (protocol.py), replayed from their
records as validators and `chain verify` do."""

import unittest

from vibehash import economics as econ
from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import GENESIS_JOB, STARTER_ESCROW, SYSTEM, Block, Chain, InvalidBlock, State, Tx, apply_block, canonical
from vibehash.protocol import Protocol
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker
from tests.test_hardening import replayed

V = SYN_PER_VIBE
MAINNET = econ.NEW_SCHEDULES["mainnet"]
HW = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)
VALIDATOR = KeyPair.generate(b"starter validator")  # votes and proposes, doesn't mine


def spec(job_id: str = "starter", **over) -> str:
    return canonical({**TrainingJob(job_id, SYSTEM, budget=0, fee_per_round=0, units_per_round=12).to_spec(), **over}).decode()


def network(workers, allocations=None, starters=None):
    """No starting coins (unless `allocations`), earn-in, one validator with bootstrap stake that proposes every block."""
    chain = Chain(allocations or {}, economics=MAINNET, earn_in=True, starter_jobs=[spec()] if starters is None else starters,
                  validators={VALIDATOR.address: "https://v.example.org:8545"}, bootstrap={VALIDATOR.address: 10 * V})
    proto = Protocol(chain, workers, CoordinatorConfig(), seed=3)
    proto._proposer = lambda seed: VALIDATOR
    return proto


def genesis_block(txs) -> Block:
    return Block(0, b"\x00" * 32, 0, None, txs, "starter-net")


class LedgerTests(unittest.TestCase):
    def test_a_starter_job_is_on_chain_from_block_one_owned_by_nobody(self):
        s = Chain({}, economics=MAINNET, starter_jobs=[spec()]).state
        self.assertEqual((s.jobs["starter"], s.job_owner["starter"], s.escrow["starter"]), (spec(), SYSTEM, STARTER_ESCROW))
        self.assertEqual((s.genesis, s.total_supply()), (1, 1))  # its syn of budget is counted like any allocation
        self.assertEqual(s.job("starter").fee_per_round, 0)

    def test_a_genesis_without_one_keeps_its_hash(self):
        self.assertEqual(Chain({}, economics=MAINNET, starter_jobs=[]).genesis_hash, Chain({}, economics=MAINNET).genesis_hash)
        self.assertNotEqual(Chain({}, economics=MAINNET, starter_jobs=[spec()]).genesis_hash,
                            Chain({}, economics=MAINNET).genesis_hash)

    def test_what_isnt_a_starter_job(self):
        with self.assertRaisesRegex(InvalidBlock, "fee_per_round must be 0"):
            Chain({}, starter_jobs=[spec(fee_per_round=5 * V)])
        with self.assertRaisesRegex(InvalidBlock, "job id already used"):
            Chain({}, starter_jobs=[spec(), spec()])
        with self.assertRaisesRegex(InvalidBlock, "starter job:"):
            Chain({}, starter_jobs=['{"id": "broken"}'])
        for recipient, amount in ((SYSTEM, 0), (SYSTEM, 5), (VALIDATOR.address, STARTER_ESCROW)):
            tx = Tx("genesis", SYSTEM, recipient, amount, memo=GENESIS_JOB + spec(), chain_id="starter-net")
            with self.subTest(recipient=recipient, amount=amount), self.assertRaisesRegex(InvalidBlock, "belongs to nobody"):
                apply_block(State(), genesis_block([tx]), "starter-net", None)

    def test_nobody_can_cancel_it(self):
        chain = Chain({VALIDATOR.address: 10 * V}, starter_jobs=[spec()])
        cancel = Tx.signed(VALIDATOR, "cancel_job", SYSTEM, 0, 0, "starter", chain.chain_id, 0)
        with self.assertRaisesRegex(InvalidBlock, "only the requester"):
            chain.check_tx(cancel, [])


class MiningTests(unittest.TestCase):
    def test_a_network_with_no_starting_coins_mines_from_its_first_round(self):
        honest = [Worker(f"h{i}", KeyPair.generate(f"h{i}".encode()), HW, "honest") for i in range(2)]
        lazy = Worker("lazy", KeyPair.generate(b"lazy"), HW, "lazy")
        proto = network(honest + [lazy])
        for _ in range(6):
            report = proto.run_round()
            self.assertEqual(report.block.summary.job_id, "starter")
            self.assertTrue(report.block.summary.pays_for_work)
        s = proto.chain.state
        self.assertEqual(s.minted, 6 * MAINNET.initial_subsidy)  # every round worked, every round minted
        for w in honest:  # newcomers: their first 5 VIBE of pay became their stake
            self.assertEqual(s.stake(w.address), econ.MIN_STAKE)
        self.assertEqual((s.stake(lazy.address), s.balance(lazy.address)), (0, 0), "the cheater never earned anything")
        self.assertEqual(s.escrow["starter"], STARTER_ESCROW, "a starter job never runs out")
        self.assertEqual(s.total_supply(), s.expected_supply())
        again = replayed(self, proto)
        self.assertEqual((again.minted, again.burned), (s.minted, s.burned))

    def test_paid_jobs_come_first_and_the_starter_job_fills_in(self):
        req = KeyPair.generate(b"starter requester")
        workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), HW, "honest") for i in range(3)]
        proto = network(workers, allocations={req.address: 100 * V})
        paid = TrainingJob("paid", req.address, budget=30 * V, fee_per_round=10 * V, units_per_round=12)
        proto.post_job(req, paid)
        jobs = []
        for _ in range(8):
            had_budget = proto.chain.state.escrow["paid"] > 0
            jobs.append(proto.run_round().block.summary.job_id)
            # Every round goes to the job people pay for while it has budget (it pays only for units that passed,
            # so its budget can last more rounds than budget / fee), and to the starter job after.
            self.assertEqual(jobs[-1], "paid" if had_budget else "starter")
        self.assertGreaterEqual(jobs.count("paid"), 3)
        self.assertEqual(jobs[-1], "starter")
        self.assertEqual(proto.chain.state.escrow["paid"], 0)
        s = proto.chain.state
        self.assertEqual(s.total_supply(), s.expected_supply())
        replayed(self, proto)


class LiveTests(unittest.TestCase):
    def test_a_node_and_workers_over_http_mine_the_starter_job_from_nothing(self):
        """A real node (a single validator with bootstrap stake) and three workers that start with nothing, on a
        genesis with no allocations or stakes: they mine the starter job, earn in, and the node serves it as a job."""
        import tempfile
        import threading
        from pathlib import Path

        from vibehash.client import NodeClient, WorkerClient
        from vibehash.node import Node, PhaseTimes
        from tests.test_network import run_until

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        node_keys = KeyPair.generate(b"starter live node")
        workers = [KeyPair.generate(f"starter live worker {i}".encode()) for i in range(3)]
        genesis = {"chain_id": "starter-live", "timestamp": 0, "allocations": {}, "earn_in": True,
                   "bootstrap": {node_keys.address: 10 * V}, "economics": econ.NEW_SCHEDULES["testnet"].to_dict(),
                   "starter_jobs": [{**TrainingJob("starter", SYSTEM, budget=0, fee_per_round=0, units_per_round=6).to_spec()}]}
        node = Node(Path(tmp.name) / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=4.0, reveal_s=2.0, verify_s=3.0, idle_s=0.3),
                    config=CoordinatorConfig(canaries_per_round=1, newcomer_units=6)).start()
        self.addCleanup(node.stop)
        stop = threading.Event()
        self.addCleanup(stop.set)
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e9, poll_s=0.1, behavior="honest",
                                cache_dir=Path(tmp.name) / f"cache{i}") for i, k in enumerate(workers)]
        run_until(node, clients, stop, 3)
        s = node.chain.state
        self.assertGreaterEqual(node.chain.tip.height, 3, "no blocks: the starter job never ran")
        self.assertGreater(s.minted, 0)
        self.assertTrue(all(s.stake(k.address) + s.balance(k.address) > 0 for k in workers), "every worker was paid")
        api = NodeClient(node.url)
        info = api.get("/job/starter")
        self.assertGreaterEqual(len(info["rounds"]), 1)
        self.assertEqual(api.get("/supply")["escrow"], STARTER_ESCROW)


if __name__ == "__main__":
    unittest.main()
