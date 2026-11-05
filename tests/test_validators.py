"""A chain run by validators (consensus.py, p2p.py): real nodes on localhost taking turns leading rounds and voting on
every block, outvoting a dishonest leader, carrying on without a validator that went offline, and new nodes syncing
the chain without trusting whoever serves it."""
import contextlib
import io
import logging
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from vibehash import cli
from vibehash.client import NodeClient, WorkerClient
from vibehash.consensus import Timeouts, check_certificate, proposer_order, vote
from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair, address_of
from vibehash.ledger import Chain, Tx
from vibehash.model import params_hash
from vibehash.node import Node, PhaseTimes
from vibehash.storage import Store
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

VIBE = SYN_PER_VIBE
TIMES = PhaseTimes(register_s=1.0, compute_s=3.0, reveal_s=1.5, verify_s=2.0, idle_s=0.3)
TIMEOUTS = Timeouts(propose=12.0, vote=1.0, grow=1.0, regossip=1.0)
JOB = "bft-job"


class Network:
    """Validators (and workers whose own node is validator 0) on this computer."""

    def __init__(self, test, n_validators=4, n_workers=3):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows keeps memory-mapped files locked
        test.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.vkeys = [KeyPair.generate(f"validator{i}".encode()) for i in range(n_validators)]
        self.req = KeyPair.generate(b"requester")
        self.wkeys = [KeyPair.generate(f"worker{i}".encode()) for i in range(n_workers)]
        self.ports = cli._free_ports(n_validators)
        self.genesis = {"chain_id": "vibehash-bft-test", "timestamp": 0, "allocations": {self.req.address: 500 * VIBE},
                        "stakes": {k.address: 10 * VIBE for k in [*self.vkeys, *self.wkeys]},
                        "validators": {k.address: f"http://127.0.0.1:{p}" for k, p in zip(self.vkeys, self.ports)}}
        self.nodes = [self.start_validator(i) for i in range(n_validators)]
        self.extra: list[Node] = []
        test.addCleanup(self.stop_all)
        self.stop = threading.Event()
        test.addCleanup(self.stop.set)
        self.clients = [WorkerClient(self.nodes[0].url, k, memory_mb=8192, flops_override=1e9, poll_s=0.1,
                                     cache_dir=self.dir / f"cache{i}") for i, k in enumerate(self.wkeys)]

    def start_validator(self, i: int) -> Node:
        return Node(self.dir / f"v{i}", self.genesis, self.vkeys[i], port=self.ports[i], times=TIMES,
                    config=CoordinatorConfig(canaries_per_round=1, newcomer_units=6), timeouts=TIMEOUTS).start()

    def start_node(self, name: str, peers: list[str]) -> Node:
        node = Node(self.dir / name, self.genesis, KeyPair.generate(name.encode()), port=0, times=TIMES,
                    timeouts=TIMEOUTS, peers=peers).start()
        self.extra.append(node)
        return node

    def stop_all(self):
        self.stop.set()
        for n in [*self.nodes, *self.extra]:
            if not n.stopping.is_set():
                n.stop()

    def post_job(self, budget=300):
        job = TrainingJob(JOB, self.req.address, budget=budget * VIBE, fee_per_round=10 * VIBE, units_per_round=6)
        NodeClient(self.nodes[0].url).send_tx(self.req, "post_job", "system", job.budget, job.spec_json())

    def start_workers(self):
        for c in self.clients:
            threading.Thread(target=c.run, args=(self.stop,), daemon=True).start()

    def live(self):
        return [n for n in self.nodes if not n.stopping.is_set()]

    def lowest(self) -> int:
        return min(n.chain.tip.height for n in self.live())

    @staticmethod
    def wait(condition, seconds):
        deadline = time.time() + seconds
        while not condition():
            if time.time() > deadline:
                raise AssertionError("timed out waiting")
            time.sleep(0.2)

    def assert_same_chain(self, test, nodes, upto):
        for h in range(upto + 1):
            test.assertEqual(len({n.chain.blocks[h].hash for n in nodes}), 1, f"nodes disagree at block {h}")


def run_cli(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home, \
            mock.patch.dict(os.environ, {"VIBEHASH_HOME": home}), contextlib.redirect_stdout(out):
        logging.disable(logging.INFO)
        try:
            cli.main(list(argv))
            code = 0
        except SystemExit as e:
            code = e.code or 0
        finally:
            logging.disable(logging.NOTSET)
    return code, out.getvalue()


def verify(*argv) -> tuple[int, str]:
    """`vibehash chain verify`, as anyone auditing the network would run it."""
    return run_cli("chain", "verify", *argv)


class ValidatorNetworkTests(unittest.TestCase):
    def test_validators_take_turns_agree_on_every_block_and_anyone_can_check(self):
        net = Network(self)
        net.post_job()
        net.start_workers()
        net.wait(lambda: net.lowest() >= 5, 180)
        net.stop.set()  # workers stop; validators carry on (with empty rounds)
        upto = net.lowest()
        net.assert_same_chain(self, net.nodes, upto)

        node = net.nodes[0]
        chain, certs = Chain.from_genesis(net.genesis), node.store.certificates(1, upto)
        for b in node.chain.blocks[1:upto + 1]:  # every block final, by signatures anyone can check
            self.assertIsNone(check_certificate(certs.get(b.height), b.height, b.hash.hex(), chain.state.validator_set(), chain.chain_id))
            chain.append(b)
        trained = [b for b in node.chain.blocks[1:upto + 1] if b.summary and b.summary.accepted_units]
        self.assertGreaterEqual(len(trained), 3)
        self.assertGreater(len({address_of(b.proposer_pubkey) for b in trained}), 1, "rounds are led by different validators")
        self.assertTrue(all(node.chain.state.balance(k.address) > 0 for k in net.wkeys), "every worker got paid")

        # Every validator ends up with the job's latest weights, so whoever leads next can carry on training.
        want = node.coord.expected_start_hash(node.chain.state.job(JOB))
        net.wait(lambda: all((p := n.store.load_params(JOB)) is not None and params_hash(p) == want for n in net.nodes), 30)

        code, out = verify("--node", net.nodes[2].url, "--receipts", str(net.dir / "cache0" / "receipts.jsonl"))
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"\d+ blocks final: each signed by validators holding more than two thirds of the stake")
        self.assertIn("every decision follows from the signed messages", out)
        self.assertRegex(out, r"re-ran [1-9]\d* weight-based checks .*every one matches its record", "checkpoints fetched from each round's leader")
        self.assertNotIn("censored", out)
        code, out = run_cli("validator", "list", "--node", node.url)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count(" 25.0%"), 4, out)  # four validators with equal stake
        self.assertIn("leading it:", out)

        # A new node, given one peer, syncs the whole chain, checking every block as it goes.
        follower = net.start_node("follower", [net.nodes[1].url])
        net.wait(lambda: follower.chain.tip.height >= upto, 60)
        net.assert_same_chain(self, [follower, node], upto)
        net.wait(lambda: NodeClient(follower.url).status().get("leader"), 10)  # it can point workers at the leader

        # ...and refuses a block whose certificate doesn't carry two thirds of the stake.
        loner = net.start_node("loner", [])
        peer = NodeClient(node.url)
        d = peer.get("/blocks?from=1&to=1")[0]
        forged = dict(d, certificate=dict(d["certificate"], votes=d["certificate"]["votes"][:1]))
        with self.assertLogs("vibehash.node", "ERROR") as logs:
            self.assertFalse(loner._take(peer, forged))
        self.assertIn("more than two thirds must sign", logs.output[0])
        self.assertEqual(loner.chain.tip.height, 0)
        self.assertTrue(loner._take(peer, d))
        self.assertEqual(loner.chain.tip.height, 1)

        # A data directory edited to drop a block's certificate isn't trusted on restart either.
        follower.stop()
        db = sqlite3.connect(net.dir / "follower" / "chain.sqlite3")
        with db:
            db.execute("DELETE FROM certificates WHERE height = 2")
        db.close()
        store = Store(net.dir / "follower")
        try:
            with self.assertRaisesRegex(ValueError, "block 2 in this data directory isn't final"):
                store.open_chain(net.genesis)
        finally:
            store.close()

    def test_a_dishonest_leader_is_outvoted_and_the_next_one_takes_over(self):
        net = Network(self)
        # Height 1 carries the job's post_job; height 2 is its first training round. Its first leader pays itself
        # for checking work it never did, and votes for its own block.
        cheat = proposer_order(net.nodes[0].chain.state.validator_set(), 2)[0]
        bad = next(n for n in net.nodes if n.keys.address == cheat)
        honest_settle = bad.coord._settle

        def greedy(proposer):
            summary, params, flops = honest_settle(proposer)
            summary.verifier_flops[proposer] = summary.verifier_flops.get(proposer, 0) + 10**12
            return summary, params, flops
        bad.coord._settle = greedy
        bad._validate = lambda value: True
        net.post_job()
        net.start_workers()
        with self.assertLogs("vibehash.node", "WARNING") as logs:
            net.wait(lambda: net.lowest() >= 3, 180)
        self.assertTrue(any("voting against block 2" in line and "verifier_flops" in line for line in logs.output), logs.output)
        node = net.nodes[0]
        cert = node.store.certificates(2, 2)[2]
        self.assertGreaterEqual(cert["round"], 1, "round 0's block never got two thirds of the votes")
        self.assertNotIn(cheat, node.chain.blocks[2].summary.verifier_flops)
        net.assert_same_chain(self, net.nodes, net.lowest())
        code, out = verify("--node", node.url, "--no-recheck")
        self.assertEqual(code, 0, out)

    def test_the_chain_carries_on_without_an_offline_validator_which_catches_up_when_back(self):
        net = Network(self)
        net.post_job()
        net.start_workers()
        net.wait(lambda: net.lowest() >= 2, 120)
        gone = 3
        net.nodes[gone].stop()  # a quarter of the stake: the rest still have more than two thirds
        at = net.lowest()
        net.wait(lambda: net.lowest() >= at + 3, 180)
        node = net.nodes[0]
        self.assertTrue(any(b.summary and b.summary.accepted_units for b in node.chain.blocks[at + 1:]), "training went on")
        net.nodes[gone] = back = net.start_validator(gone)
        net.wait(lambda: back.chain.tip.height >= node.chain.tip.height - 1 and back.chain.tip.height > at + 3, 60)
        net.assert_same_chain(self, [back, node], back.chain.tip.height)
        net.wait(lambda: back.engine.height == back.chain.tip.height + 1, 10)  # and it's voting again

    def test_a_validator_signing_two_precommits_loses_its_stake_and_its_place(self):
        """One of four validators sends, beside its precommit for block 1, a second one for another block. Each honest
        validator that sees both sends the evidence from its own key (block version 4); one lands, and its block takes
        the double-signer's stake and its place. The other three go on alone."""
        net = Network(self, n_workers=0)
        bad, honest = net.nodes[3], net.nodes[:3]
        cid, real, sent = net.genesis["chain_id"], net.nodes[3].engine._broadcast, []

        def two_faced(msg):
            real(msg)
            if msg["type"] == "precommit" and msg["block"] is not None and not sent:
                sent.append(msg)
                real(vote(bad.keys, cid, "precommit", msg["height"], msg["round"], "f" * 64))
        bad.engine._broadcast = two_faced
        api, someone = NodeClient(net.nodes[0].url), KeyPair.generate(b"someone").address
        with self.assertLogs("vibehash.node", "WARNING") as logs:
            api.send_tx(net.req, "transfer", someone, VIBE)
            net.wait(lambda: all(bad.keys.address in n.chain.state.banned for n in honest), 60)
        self.assertTrue(any("sent the evidence" in line for line in logs.output), logs.output)
        for n in honest:
            s = n.chain.state
            self.assertEqual((s.stake(bad.keys.address), bad.keys.address in s.validators), (0, False))
            self.assertEqual(len(s.validator_set()), 3)
        reports = [t for b in net.nodes[0].chain.blocks for t in b.txs if t.kind == "evidence"]
        self.assertEqual(len(reports), 1, "one punishment, however many validators reported it")
        self.assertEqual(net.nodes[0].chain.state.balance(reports[0].sender), VIBE, "a tenth of its 10 VIBE")
        at = min(n.chain.tip.height for n in honest)
        api.send_tx(net.req, "transfer", someone, VIBE)
        net.wait(lambda: min(n.chain.tip.height for n in honest) > at, 60)
        net.assert_same_chain(self, honest, at + 1)
        code, out = verify("--node", net.nodes[1].url, "--no-recheck")
        self.assertEqual(code, 0, out)
        self.assertIn("(3 validators now)", out)

    def test_a_validator_with_a_rival_block_on_disk_stops_and_the_rest_carry_on(self):
        """Another writer has put a different block 1 in one validator's database (2026-11-03: a second node on its
        folder). When the validators decide their block 1, it can't save it, so it stops rather than let its memory
        fork from its disk: it signs nothing more and serves only what it saved. The other three hold more than two
        thirds of the stake and carry on. Its folder isn't trusted on restart either: that block 1 has no certificate."""
        net = Network(self)
        bad = net.nodes[3]
        self.addCleanup(bad.stop)  # (stop_all passes over a node that's stopping, as a halted one is)
        cid = net.genesis["chain_id"]
        rival, _ = bad.chain.build_block(None, [Tx.signed(net.req, "transfer", net.wkeys[0].address, VIBE, 0, "", cid)], 1)
        rival.sign(bad.keys)
        bad.store.save_block(rival)
        api = NodeClient(net.nodes[0].url)
        with self.assertLogs("vibehash.node", "CRITICAL") as logs:
            api.send_tx(net.req, "transfer", net.wkeys[1].address, 2 * VIBE)
            net.wait(lambda: bad.halted, 60)
        self.assertIn("this node's database already has a different block 1 than the one the validators decided",
                      logs.output[0])
        self.assertTrue(bad.engine.halted)
        self.assertEqual(bad.chain.tip.height, 0)
        self.assertEqual(NodeClient(bad.url).get("/blocks?from=1"), [])
        self.assertIn("Another node may be using this folder", bad.halted)  # the operator's reason, in its log
        self.assertEqual(NodeClient(bad.url).status()["halted"], bad.halted_public)  # what anyone asking is told
        self.assertNotIn(str(bad.store.dir), bad.halted_public)
        api.send_tx(net.req, "transfer", net.wkeys[2].address, 3 * VIBE)
        net.wait(lambda: net.lowest() >= 2, 60)  # (live(): the three still running)
        net.assert_same_chain(self, net.live(), 2)
        self.assertNotEqual(net.nodes[0].chain.blocks[1].hash, rival.hash)
        self.assertLessEqual({m["height"] for m in bad.engine.mine}, {1}, "it sent nothing past the block it couldn't save")
        store = Store(net.dir / "v3")
        try:
            with self.assertRaisesRegex(ValueError, "block 1 in this data directory isn't final"):
                store.open_chain(net.genesis)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
