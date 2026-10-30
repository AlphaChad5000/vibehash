"""BFT consensus (consensus.py) on a simulated network: honest runs, crashed and lying proposers, double-voting
validators, lost messages and partitions, with a fake clock so every run is exact and fast."""
import unittest

from vibehash.consensus import Engine, Step, Timeouts, certificate, check_certificate, proposer_order, quorum, vote
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain, InvalidBlock, Tx
from vibehash.tasks import SYN_PER_VIBE

CHAIN = "test-chain"


class Net:
    """N validators wired together through a message queue with a fixed delay. Tests drop, delay or forge messages
    by replacing `link` (from, to, message) -> message or None."""

    def __init__(self, n=4, stakes=None, timeouts=None, has_work=lambda: True, build_s=1.0, propose_wait=None):
        self.keys = [KeyPair.generate(f"validator-{i}".encode()) for i in range(n)]
        self.vals = {k.address: (stakes[i] if stakes else 10) for i, k in enumerate(self.keys)}
        self.index = {k.address: i for i, k in enumerate(self.keys)}
        self.now, self.queue, self.down, self.build_s, self.has_work = 0.0, [], set(), build_s, has_work
        self.link = lambda frm, to, msg: msg
        self.committed = [[] for _ in range(n)]  # per validator: (height, block id, certificate)
        self.building: dict[int, tuple[float, int, int]] = {}  # validator -> (ready at, height, round)
        self.invalid: set[str] = set()
        self.made = 0
        t = timeouts or Timeouts(propose=5.0, vote=1.0, grow=1.0, regossip=1.0)
        self.engines = [Engine(CHAIN, k, t, validators=lambda: dict(self.vals), validate=self.validate,
                               commit=lambda h, v, c, i=i: self.committed[i].append((h, v["block"]["id"], c)),
                               broadcast=lambda m, i=i: self.send(i, m), has_work=has_work,
                               block_id=lambda v: v["block"]["id"], propose_wait=propose_wait) for i, k in enumerate(self.keys)]
        for e in self.engines:
            e.start(0, self.now)

    def validate(self, value):
        return value["block"]["id"] not in self.invalid

    def send(self, frm, msg):
        for to in range(len(self.engines)):
            if to != frm and frm not in self.down and to not in self.down:
                out = self.link(frm, to, msg)
                for m in out if isinstance(out, list) else [out] if out is not None else []:
                    self.queue.append((self.now + 0.05, to, m))

    def value(self, i, height, round_):
        self.made += 1
        return {"block": {"id": f"h{height}-r{round_}-by{i}-#{self.made}"}, "record": None}

    def run(self, seconds, until=None):
        end = self.now + seconds
        while self.now < end and not (until and until()):
            self.now = round(self.now + 0.05, 2)
            due, self.queue = [q for q in self.queue if q[0] <= self.now], [q for q in self.queue if q[0] > self.now]
            for _, to, msg in due:
                if to not in self.down:
                    self.engines[to].receive(msg, self.now)
            for i, e in enumerate(self.engines):
                if i in self.down:
                    continue
                if e.wanted and i not in self.building and self.has_work():  # building one takes a while, like a training round
                    self.building[i] = (self.now + self.build_s, *e.wanted)
                ready = self.building.get(i)
                if ready and ready[0] <= self.now:
                    del self.building[i]
                    e.propose(ready[1], ready[2], self.value(i, ready[1], ready[2]), self.now)
                e.tick(self.now)

    def heights(self, alive_only=True):
        return [len(c) for i, c in enumerate(self.committed) if not (alive_only and i in self.down)]

    def assert_agree(self, test):
        """No two validators finalised different blocks at the same height, and every certificate checks out."""
        by_height = {}
        for chain in self.committed:
            for h, bid, cert in chain:
                test.assertEqual(by_height.setdefault(h, bid), bid, f"fork at height {h}")
                test.assertIsNone(check_certificate(cert, h, bid, self.vals, CHAIN))
        return by_height


class EngineTests(unittest.TestCase):
    def test_honest_validators_agree_on_every_block(self):
        net = Net(4)
        net.run(60, until=lambda: min(net.heights()) >= 5)
        self.assertGreaterEqual(min(net.heights()), 5)
        blocks = net.assert_agree(self)
        proposers = {bid.split("-by")[1].split("-")[0] for bid in blocks.values()}
        self.assertGreater(len(proposers), 1, "proposers take turns")
        for chain in net.committed:  # all in the first round: nobody waited for a timeout
            self.assertTrue(all(cert["round"] == 0 for _, _, cert in chain))

    def test_offline_proposer_costs_one_round(self):
        net = Net(4)
        first = net.engines[0].proposer(0)
        net.down.add(net.index[first])
        net.run(60, until=lambda: min(net.heights()) >= 1)
        self.assertEqual(min(net.heights()), 1)
        _, bid, cert = next(c[0] for i, c in enumerate(net.committed) if i not in net.down)
        self.assertEqual(cert["round"], 1)
        self.assertNotIn(f"by{net.index[first]}-", bid)
        net.assert_agree(self)

    def test_a_proposal_that_fails_its_checks_is_voted_down(self):
        net = Net(4)
        bad = net.index[net.engines[0].proposer(0)]
        real = net.value

        def lying(i, height, round_):
            v = real(i, height, round_)
            if i == bad and round_ == 0:
                net.invalid.add(v["block"]["id"])  # say, a summary that doesn't follow from its record
            return v
        net.value = lying
        net.run(60, until=lambda: min(net.heights()) >= 1)
        blocks = net.assert_agree(self)
        self.assertNotIn(blocks[0], net.invalid)
        self.assertEqual(net.committed[0][0][2]["round"], 1)

    def test_a_proposer_sending_different_blocks_to_different_validators_cannot_fork_the_chain(self):
        net = Net(4)
        liar = net.index[net.engines[0].proposer(0)]
        from vibehash.consensus import proposal

        def split(frm, to, msg):
            if frm == liar and msg["type"] == "proposal" and msg["round"] == 0 and to % 2:
                other = {"block": {"id": msg["block"]["id"] + "-evil"}, "record": None}
                return proposal(net.keys[liar], CHAIN, msg["height"], 0, -1, other, other["block"]["id"])
            return msg
        net.link = split
        net.run(90, until=lambda: min(net.heights()) >= 2)
        self.assertGreaterEqual(min(net.heights()), 2)
        net.assert_agree(self)
        caught = [e for i, e in enumerate(net.engines) if i != liar and any(a[0]["type"] == "proposal" for a in e.evidence)]
        self.assertEqual(caught, [], "each honest validator saw only one of the two proposals")

    def test_a_validator_voting_both_ways_cannot_fork_the_chain_and_is_recorded(self):
        net = Net(4)
        cheat = 3
        real_link = net.link

        def two_faced(frm, to, msg):
            if frm == cheat and msg["type"] in ("prevote", "precommit") and to % 2:
                return [vote(net.keys[cheat], CHAIN, msg["type"], msg["height"], msg["round"], "something-else"), msg]
            return real_link(frm, to, msg)
        net.link = two_faced
        with self.assertLogs("vibehash.consensus", "WARNING"):
            net.run(60, until=lambda: min(net.heights()) >= 3)
        self.assertGreaterEqual(min(net.heights()), 3)
        net.assert_agree(self)
        # The first vote counts; holding both signed votes is proof of double-voting, grounds for slashing.
        caught = [e for i, e in enumerate(net.engines) if i != cheat and e.evidence]
        self.assertTrue(caught)
        first, second = caught[0].evidence[0]
        self.assertEqual((first["round"], first["type"]), (second["round"], second["type"]))
        self.assertNotEqual(first["block"], second["block"])

    def test_a_block_validators_locked_on_is_proposed_again_by_the_next_proposer(self):
        net = Net(4)
        order = [net.index[a] for a in net.engines[0].order]
        p0, p1, x, y = order
        # Round 0: y never gets the proposal, so it prevotes nil. p0 and p1 see three prevotes for the block, lock on
        # it and precommit it; x and y miss a prevote each for a while, time out and precommit nil. With only two
        # precommits for it, round 0 can't decide, but p1, the next proposer, knows the block nearly got through.
        def lossy(frm, to, msg):
            if msg["round"] == 0 and net.now < 10:
                if msg["type"] == "proposal" and to == y:
                    return None
                if msg["type"] == "prevote" and (frm, to) in ((p1, x), (x, y)):
                    return None
            return msg
        net.link = lossy
        net.run(60, until=lambda: min(net.heights()) >= 1)
        self.assertEqual(min(net.heights()), 1)
        h, bid, cert = net.committed[0][0]
        self.assertEqual(cert["round"], 1)
        self.assertIn("-r0-", bid, "round 1 committed round 0's block: everyone was locked on it")
        self.assertEqual(net.made, 1, "the second proposer didn't build a block of its own")
        net.assert_agree(self)

    def test_a_split_network_stops_and_then_carries_on_when_it_heals(self):
        net = Net(4)
        half = lambda i: i < 2  # noqa: E731
        net.link = lambda frm, to, msg: msg if half(frm) == half(to) else None
        net.run(40)
        self.assertEqual(net.heights(), [0, 0, 0, 0], "neither half has more than two thirds of the stake")
        net.link = lambda frm, to, msg: msg
        net.run(120, until=lambda: min(net.heights()) >= 2)
        self.assertGreaterEqual(min(net.heights()), 2)
        net.assert_agree(self)

    def test_a_validator_that_was_offline_catches_up_from_later_messages(self):
        net = Net(4)
        late = 2
        net.down.add(late)
        net.run(60, until=lambda: min(net.heights()) >= 2)
        behind = []
        net.engines[late]._behind = behind.append
        net.down.discard(late)
        net.run(10, until=lambda: bool(behind))
        self.assertTrue(behind and behind[0] >= 2, "it learns the network is ahead, so node.py fetches the blocks")

    def test_a_validator_with_more_stake_proposes_more_often(self):
        a, b = KeyPair.generate(b"a").address, KeyPair.generate(b"b").address
        firsts = [proposer_order({a: 30, b: 10}, h)[0] for h in range(2000)]
        self.assertAlmostEqual(firsts.count(a) / len(firsts), 0.75, delta=0.04)
        self.assertEqual(proposer_order({a: 30, b: 10}, 7), proposer_order({b: 10, a: 30}, 7), "same order everywhere")
        self.assertEqual(sorted(proposer_order({a: 30, b: 10}, 7)), sorted([a, b]), "every validator gets a turn")

    def test_an_idle_chain_waits_without_churning_rounds(self):
        work = []
        net = Net(4, has_work=lambda: bool(work))
        net.run(100)
        self.assertEqual([e.round for e in net.engines], [0, 0, 0, 0])
        self.assertEqual(net.queue, [], "and sends nothing")
        work.append("a transaction")
        net.run(60, until=lambda: min(net.heights()) >= 1)
        self.assertEqual(min(net.heights()), 1)

    def test_a_validator_restarting_mid_height_doesnt_vote_twice_in_a_round(self):
        net = Net(4)
        net.down.add(net.index[net.engines[0].proposer(0)])  # round 0 can't produce a block
        i = next(j for j in range(4) if j not in net.down)
        saved = {}
        net.engines[i]._persist = saved.update
        net.run(30, until=lambda: saved.get("round") == 0 and net.engines[i].step == Step.PRECOMMIT)
        self.assertEqual(saved["round"], 0)

        sent = []  # it crashes and comes back with only what it saved
        fresh = Engine(CHAIN, net.keys[i], Timeouts(propose=5.0, vote=1.0, grow=1.0, regossip=1.0), validators=lambda: dict(net.vals),
                       validate=net.validate, commit=lambda h, v, c: net.committed[i].append((h, v["block"]["id"], c)),
                       broadcast=lambda m: (sent.append(m), net.send(i, m)), block_id=lambda v: v["block"]["id"])
        fresh.start(0, net.now, resume=dict(saved))
        self.assertEqual(fresh.round, 1)
        net.engines[i] = fresh
        net.run(60, until=lambda: min(net.heights()) >= 1)
        self.assertEqual(min(net.heights()), 1)
        self.assertFalse([m for m in sent if m["height"] == 0 and m["round"] == 0], "no second vote in round 0")
        net.assert_agree(self)

    def test_validators_wait_longer_for_a_job_with_long_rounds(self):
        net = Net(4, propose_wait=lambda: 30.0)  # its leader may be running a 20-second training round
        net.down.add(net.index[net.engines[0].proposer(0)])
        net.run(25)
        self.assertEqual(net.heights(), [0, 0, 0], "still waiting for the first leader")
        net.run(30, until=lambda: min(net.heights()) >= 1)
        self.assertEqual(min(net.heights()), 1, "then the next one took over")
        self.assertEqual(net.committed[next(i for i in range(4) if i not in net.down)][0][2]["round"], 1)

    def test_one_validator_alone_finalises_immediately(self):
        net = Net(1)
        net.run(20, until=lambda: net.heights()[0] >= 3)
        self.assertEqual(net.heights()[0], 3)
        net.assert_agree(self)


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.keys = [KeyPair.generate(f"c{i}".encode()) for i in range(4)]
        self.vals = {k.address: 10 for k in self.keys}

    def cert(self, signers, block="b1", height=5, round_=0, for_block=None):
        return certificate(height, round_, block,
                           [vote(k, CHAIN, "precommit", height, round_, for_block or block) for k in signers])

    def test_more_than_two_thirds_of_the_stake_is_needed(self):
        self.assertIsNone(check_certificate(self.cert(self.keys[:3]), 5, "b1", self.vals, CHAIN))
        self.assertIn("more than two thirds", check_certificate(self.cert(self.keys[:2]), 5, "b1", self.vals, CHAIN))
        self.assertTrue(quorum(3, 4) and not quorum(2, 3) and quorum(7, 10) and not quorum(20, 30))

    def test_forged_duplicate_outsider_and_mismatched_votes_dont_count(self):
        c = self.cert(self.keys[:2])
        c["votes"].append(dict(c["votes"][0]))  # the same vote twice
        c["votes"].append(self.cert([KeyPair.generate(b"outsider")])["votes"][0])
        self.assertIsNotNone(check_certificate(c, 5, "b1", self.vals, CHAIN))
        forged = self.cert(self.keys[:3])
        forged["votes"][2]["sig"] = forged["votes"][0]["sig"]
        self.assertIn("forged", check_certificate(forged, 5, "b1", self.vals, CHAIN))
        other = self.cert(self.keys[:3], for_block="b2")  # votes for another block, relabelled
        self.assertIn("forged", check_certificate(other, 5, "b1", self.vals, CHAIN))
        self.assertIn("different block", check_certificate(self.cert(self.keys[:3]), 5, "b9", self.vals, CHAIN))
        self.assertIn("different block", check_certificate(self.cert(self.keys[:3]), 6, "b1", self.vals, CHAIN))
        self.assertEqual(check_certificate(None, 5, "b1", self.vals, CHAIN), "no commit certificate")


class ValidatorSetTests(unittest.TestCase):
    """Who may vote and propose: the ledger's side of it."""

    def setUp(self):
        self.a, self.b, self.poor = (KeyPair.generate(s) for s in (b"val-a", b"val-b", b"poor"))
        self.stakes = {self.a.address: 10 * SYN_PER_VIBE, self.b.address: 30 * SYN_PER_VIBE}
        self.chain = Chain({self.poor.address: 100 * SYN_PER_VIBE}, self.stakes, "vs-test", 0,
                           validators={self.a.address: "http://10.0.0.1:8545"})

    def announce(self, keys, url, chain=None):
        chain = chain or self.chain
        return Tx.signed(keys, "validator", "system", 0, chain.next_nonce(keys.address), url, chain.chain_id)

    def seal(self, keys, txs, chain=None):
        chain = chain or self.chain
        block, dropped = chain.build_block(None, txs, chain.tip.timestamp + 1)
        block.sign(keys)
        chain.append(block)
        return dropped

    def test_the_genesis_names_the_validators_and_its_hash_commits_to_them(self):
        self.assertEqual(self.chain.state.validator_set(), {self.a.address: 10 * SYN_PER_VIBE})
        plain = Chain({self.poor.address: 100 * SYN_PER_VIBE}, self.stakes, "vs-test", 0)
        self.assertEqual(plain.state.validator_set(), {})
        self.assertNotEqual(plain.genesis_hash, self.chain.genesis_hash)
        self.assertEqual(plain.genesis_hash, Chain.from_genesis(
            {"chain_id": "vs-test", "timestamp": 0, "allocations": {self.poor.address: 100 * SYN_PER_VIBE},
             "stakes": self.stakes}).genesis_hash, "genesis files without validators keep their hash")

    def test_only_a_validator_may_propose(self):
        block, _ = self.chain.build_block(None, [], 1)
        block.sign(self.b)  # staked, but not a validator
        with self.assertRaisesRegex(InvalidBlock, "not a validator"):
            self.chain.append(block)

    def test_joining_needs_stake_and_a_reachable_address_and_one_validator_must_stay(self):
        dropped = self.seal(self.a, [self.announce(self.b, "http://10.0.0.2:8545"), self.announce(self.poor, "http://10.0.0.3:8545")])
        self.assertEqual([t.sender for t in dropped], [self.poor.address], "no stake, no vote")
        self.assertEqual(self.chain.state.validator_set(), {self.a.address: 10 * SYN_PER_VIBE, self.b.address: 30 * SYN_PER_VIBE})
        self.assertEqual(self.seal(self.b, [self.announce(self.a, "not a url")])[0].memo, "not a url")
        self.assertEqual(self.seal(self.b, [self.announce(self.a, "")]), [], "a validator can leave")
        self.assertEqual(len(self.seal(self.b, [self.announce(self.b, "")])), 1, "but not the last one")
        self.assertEqual(list(self.chain.state.validator_set()), [self.b.address])
        self.assertTrue(self.chain.validate())

    def test_nobody_can_take_over_a_chain_one_node_runs_by_announcing_themselves(self):
        solo = Chain({}, self.stakes, "solo-test", 0)
        dropped = self.seal(self.a, [self.announce(self.b, "http://10.0.0.2:8545", solo)], solo)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(solo.state.validators, {})


if __name__ == "__main__":
    unittest.main()
