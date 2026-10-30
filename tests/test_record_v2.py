"""Record version 2: a secret the leader commits to when it opens a round keeps workers from grinding the beacon and
checkers from picking out canaries, while every version 1 record still replays exactly as it was run. Plus golden
values for the numpy random streams that workers, checkers and replays must agree on."""
import copy
import dataclasses
import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from vibehash.coordinator import RECORD_VERSION, CoordinatorConfig, Phase, secret_commitment
from vibehash.crypto import Draw, KeyPair, h256, merkle_proof
from vibehash.ledger import SYSTEM, Block, Chain, Tx, canonical
from vibehash.model import Dataset, ModelSpec, batch_indices, params_hash
from vibehash.node import Node
from vibehash.protocol import Protocol
from vibehash.replay import replay_block
from vibehash.tasks import SYN_PER_VIBE, TrainingJob, WorkUnit
from vibehash.verification import _swap_leaf, challenge_segments, plant_canary, replay_openings, replay_segment
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker

VIBE = SYN_PER_VIBE
RELIABLE = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)
V1_FIXTURE = Path(__file__).parent / "data" / "record_v1.json"


def network(n=3, seed=3, job_kw=None, **cfg):
    workers = [Worker(f"v2-{i}", KeyPair.generate(f"v2-{i}".encode()), RELIABLE, "honest") for i in range(n)]
    req = KeyPair.generate(b"v2-req")
    g = dict(allocations={req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers}, chain_id="v2-test")
    proto = Protocol(Chain(**g), workers, CoordinatorConfig(**cfg), seed=seed)
    proto.post_job(req, TrainingJob("v2-job", req.address, budget=200 * VIBE, fee_per_round=10 * VIBE, units_per_round=8, **(job_kw or {})))
    return proto, g


def through_commit(proto):
    """Drive the next round as far as every commitment, as Protocol.run_round does."""
    c = proto.coord
    c.open_round()
    r, job = c.r, c.job(c.r.job_id)
    for addr, w in proto.workers.items():
        body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb}
        c.register(addr, w.hardware.flops_per_sec, w.hardware.memory_mb, envelope=proto._signed(w, "register", body))
    c.start_compute()
    for addr, uids in r.assignments.items():
        for uid in uids:
            c.commit(proto.workers[addr].compute(r.units[uid], job, r.start))
    return c, r


def resealed(block, record):
    return dataclasses.replace(block, summary=dataclasses.replace(block.summary, record_hash=h256(canonical(record)).hex()))


class OldRecordTests(unittest.TestCase):
    def test_a_version_1_record_replays_identically(self):
        """A round run by 0.5.1's coordinator (tests/data/record_v1.json: canaries, one approved by a lazy checker, a
        cheater, a false accuser, a withholder, a replicated unit, a newcomer earning in) replays with today's code."""
        fixture = json.loads(V1_FIXTURE.read_text())
        chain, replayed = Chain(**fixture["genesis"]), 0
        for d in fixture["blocks"]:
            block = Block.from_dict(d)
            if block.summary is not None:
                record = fixture["records"][str(block.height)]
                self.assertEqual(record["version"], 1)
                self.assertEqual(replay_block(chain, block, record), [], f"block {block.height}")
                replayed += 1
            chain.append(block)
        self.assertEqual(replayed, 1)
        self.assertIn("approved a canary", {s[2] for s in chain.blocks[-1].summary.slashes})
        self.assertTrue(chain.validate())

    def test_new_rounds_write_version_2(self):
        proto, g = network()
        record = proto.run_round().record
        self.assertEqual((RECORD_VERSION, record["version"]), (2, 2))
        self.assertEqual(record["beacon_commit"], secret_commitment(bytes.fromhex(record["secret"])).hex())

    def test_a_record_from_a_later_version_isnt_replayed_as_this_one(self):
        proto, g = network()
        before = copy.deepcopy(proto.chain)
        rep = proto.run_round()
        record = dict(copy.deepcopy(rep.record), version=3)
        self.assertIn("version 3", replay_block(before, resealed(rep.block, record), record)[0])


class BeaconTests(unittest.TestCase):
    """The beacon hashes in a secret the leader committed to up front, so a worker that knows every commitment (alone
    in a round, say) can no longer re-salt its own until the challenge misses what it faked."""

    def test_the_beacon_depends_on_the_round_secret(self):
        proto, g = network()
        c, r = through_commit(proto)
        digests = sorted(cm.digest for cm in r.commitments.values())
        beacon = c.start_reveal()
        # What a grinder who knows every commitment computes (and what version 1 used): no longer the beacon.
        self.assertNotEqual(beacon, h256(proto.chain.tip.hash, r.round_no.to_bytes(8, "big"), *digests))
        self.assertEqual(beacon, h256(proto.chain.tip.hash, r.round_no.to_bytes(8, "big"), r.secret, *digests))
        c.phase, r.secret = Phase.COMPUTE, os.urandom(32)  # the same commitments under another secret
        self.assertNotEqual(c.start_reveal(), beacon)

    def test_the_secret_stays_hidden_until_the_round_is_settled(self):
        proto, g = network()
        c, r = through_commit(proto)
        commit = r.record["beacon_commit"]
        self.assertEqual(commit, secret_commitment(r.secret).hex())
        beacon, job = c.start_reveal(), c.job(r.job_id)
        for uid, addr in list(r.commitments):
            rv = proto.workers[addr].reveal(r.units[uid], job, beacon, c.config.challenge_segments)
            c.reveal(rv, wire=proto._wire(rv, job))
        c.start_verify()
        self.assertTrue(any(t.canary for t in r.tasks.values()))
        self.assertNotIn(r.secret.hex(), canonical(r.record).decode(), "nothing serving the record mid-round can leak it")
        report = c.finalize(proposer_keys=proto._proposer(r.beacon), timestamp=proto.chain.tip.height + 1)
        self.assertEqual((report.record["secret"], report.record["beacon_commit"]), (r.secret.hex(), commit))

    def test_a_record_whose_secret_doesnt_match_its_commitment_fails_replay(self):
        proto, g = network()
        before = copy.deepcopy(proto.chain)
        rep = proto.run_round()
        self.assertEqual(replay_block(copy.deepcopy(before), rep.block, rep.record), [])
        for field in ("secret", "beacon_commit"):
            record = dict(copy.deepcopy(rep.record), **{field: os.urandom(32).hex()})
            problems = replay_block(copy.deepcopy(before), resealed(rep.block, record), record)
            self.assertTrue(problems and "secret doesn't match the commitment" in problems[0], (field, problems))

    def test_status_shows_the_commitment_and_nothing_else_of_the_secret(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        keys, req = KeyPair.generate(b"v2 node"), KeyPair.generate(b"v2 requester")
        genesis = {"chain_id": "v2-status", "timestamp": 0, "allocations": {req.address: 100 * VIBE}, "stakes": {keys.address: 10 * VIBE}}
        node = Node(Path(tmp.name) / "node", genesis, keys, port=0)  # not started: only its views are used
        for close in (node.store.close, node.server.server_close, node.peers.close):
            self.addCleanup(close)
        c = node.coord
        job = TrainingJob("v2-job", req.address, budget=10 * VIBE, fee_per_round=VIBE, units_per_round=2)
        c.add_job(job)
        c.submit_tx(Tx.signed(req, "post_job", SYSTEM, job.budget, 0, job.spec_json(), "v2-status"))
        c.seal_pending(timestamp=1)
        self.assertIsNone(node.status()["beacon_commit"])
        c.open_round()
        status = node.status()
        self.assertEqual(status["beacon_commit"], secret_commitment(c.r.secret).hex())
        self.assertNotIn(c.r.secret.hex(), json.dumps(status))


class CanaryTests(unittest.TestCase):
    """A checker sees only what the node serves it: its tasks (id, unit, segments, openings), /status, its own units,
    and here, generously, every registered worker. From that it must not be able to tell canaries from real work."""

    @classmethod
    def setUpClass(cls):
        proto, g = network(n=6, replication_rate=0.3, canaries_per_round=2)
        c, cls.views = proto.coord, []
        served = c.tasks_for

        def tasks_for(addr):
            tasks = served(addr)
            for t in tasks:
                cls.views.append({"beacon": c.r.beacon, "checker": addr, "round": c.r.round_no, "task_id": t.task_id,
                                  "unit": t.unit, "segments": list(t.segments), "openings": t.openings,
                                  "mine": set(c.r.assignments.get(addr, [])), "roster": sorted(c.r.regs),
                                  "canary": t.canary, "holders": set(c.r.holders[t.unit.unit_id])})  # the last two: the truth
            return tasks
        c.tasks_for = tasks_for
        cls.reports = [proto.run_round() for _ in range(6)]
        cls.job = c.job("v2-job")
        cls.canaries = [v for v in cls.views if v["canary"]]

    def test_there_are_canaries_and_they_fail_an_honest_replay(self):
        self.assertGreaterEqual(len(self.canaries), 8)
        for v in self.views:
            ok, bad = replay_openings(self.job, v["unit"], {i: (np.asarray(a), p) for i, (a, p) in v["openings"].items()}, v["segments"])
            self.assertEqual(ok, not v["canary"])
            if v["canary"]:
                self.assertIn(bad, v["segments"])
        self.assertTrue(all(s[2] != "approved a canary" for rep in self.reports for s in rep.slashes))

    def test_the_old_derivation_finds_nothing(self):
        """Version 1's canary keys and task ids came from the public beacon: a checker could compute every canary's id."""
        for v in self.views:
            old = {h256(v["beacon"], v["unit"].unit_id.encode(), KeyPair.generate(v["beacon"] + bytes([n])).address.encode(),
                        v["checker"].encode()).hex()[:20] for n in range(2)}
            self.assertNotIn(v["task_id"], old)

    def test_canary_updates_look_like_training(self):
        """Honest updates to the softmax layer's biases sum to ~0 (every softmax gradient row does); version 1's noise
        didn't. Nor do a canary's openings disagree about their Merkle root."""
        spec = self.job.spec
        for v in self.views:
            for s in v["segments"]:
                (_, before), (_, after) = (spec.unpack(np.asarray(v["openings"][i][0]))[-1] for i in (s, s + 1))
                step = after - before
                self.assertLess(abs(step.sum()), 1e-6 * np.linalg.norm(step), (v["canary"], s))
            roots = set()
            for i, (arr, proof) in v["openings"].items():
                node = h256(b"leaf", params_hash(arr))
                for sibling, left in proof:
                    node = h256(b"node", sibling, node) if left else h256(b"node", node, sibling)
                roots.add(node)
            self.assertEqual(len(roots), 1, v["canary"])

    def test_who_checks_a_canary_gives_nothing_away(self):
        """Real tasks never go to their unit's holders, so neither may a canary; nor may it go to a checker that also
        has a real task on its unit (it would see the same checkpoints twice, one set changed). Its segments are the
        ones the beacon challenged for a worker holding the unit, as a real task's are."""
        units_checked = defaultdict(list)
        for v in self.views:
            units_checked[(v["round"], v["checker"], v["unit"].unit_id)].append(v)
        for v in self.views:
            self.assertNotIn(v["unit"].unit_id, v["mine"], v["canary"])
            if v["canary"]:
                self.assertNotIn(v["checker"], v["holders"])
                self.assertEqual(len(units_checked[(v["round"], v["checker"], v["unit"].unit_id)]), 1)
            challenged = [challenge_segments(v["beacon"], v["unit"].unit_id, w, self.job.n_segments, 3) for w in v["roster"]]
            self.assertIn(v["segments"], challenged, v["canary"])

    def test_a_swapped_leafs_proofs_are_a_fresh_trees(self):
        """Every tree size and leaf, including the padded last node of an odd level, which is its own sibling."""
        for n in range(1, 18):
            leaves = [h256(b"leaf", bytes([n, i])) for i in range(n)]
            for index in range(n):
                swapped = leaves[:index] + [b"new"] + leaves[index + 1:]
                self.assertEqual(_swap_leaf({i: merkle_proof(leaves, i) for i in range(n)}, index, b"new", n),
                                 {i: merkle_proof(swapped, i) for i in range(n)}, (n, index))

    def test_a_planted_segment_fails_however_little_training_moved_it(self):
        """A rescaled update smaller than a rounding step would replay as honest, and the honest checker who approved
        it would be slashed: such segments are pushed far enough to fail. Exactly one challenged segment fails."""
        for lr in (5e-6, 1e-9, 1e-12):  # float32 segments moving the weights ~5e-5, then far under a rounding step (~3e-6)
            job = TrainingJob("slow", "req", 0, 0, spec=ModelSpec(8, 32, 4, "float32"), lr=lr, tolerance=1e-4)
            start = job.spec.init_params(job.init_seed)
            unit = WorkUnit.make(job, 1, 0, params_hash(start))
            w = Worker("honest", KeyPair.generate(b"honest"), RELIABLE, fp_noise=0)
            w.compute(unit, job, start)
            rv = w.reveal(unit, job, b"beacon", 3)
            segs = challenge_segments(b"beacon", unit.unit_id, w.address, job.n_segments, 3)
            self.assertTrue(all(replay_segment(job, unit, s, rv.openings[s][0], rv.openings[s + 1][0]) for s in segs))
            for n in range(6):
                planted = plant_canary(job, rv.openings, segs, Draw(b"test", bytes([n])))
                bad = [s for s in segs if not replay_segment(job, unit, s, planted[s][0], planted[s + 1][0])]
                self.assertEqual(len(bad), 1, (lr, n))
                self.assertNotIn(bad[0] + 1, segs, "the forged checkpoint starts no other challenged segment")
        frozen = {i: (start, proof) for i, (_, proof) in rv.openings.items()}  # a segment that didn't move at all
        planted = plant_canary(job, frozen, segs, Draw(b"test"))
        self.assertEqual(sum(not np.array_equal(a, start) for a, _ in planted.values()), 1)
        self.assertTrue(all(np.all(np.isfinite(a)) for a, _ in planted.values()))


class NumpyStreamTests(unittest.TestCase):
    """numpy doesn't promise its random streams stay the same across versions, and consensus depends on these: the
    challenged segments, each step's batch, a job's starting weights and synthetic data. Changing them would fork the
    network and change training, so they're pinned (the same under numpy 1.26.4 and 2.5.3) and a numpy that changes
    them fails here. Then the fix is a version-stable generator in a new record version, not a new golden value."""

    def test_challenged_segments(self):
        beacon = h256(b"golden beacon")
        self.assertEqual([challenge_segments(beacon, f"unit{i}", "vibe1golden", 8, 3) for i in range(4)],
                         [[1, 2, 7], [5, 6, 7], [1, 4, 5], [1, 2, 5]])
        self.assertEqual(challenge_segments(beacon, "unit0", "vibe1golden", 40, 3), [9, 13, 39])
        self.assertEqual(challenge_segments(beacon, "unit0", "vibe1golden", 2, 3), [0, 1])

    def test_batches(self):
        seed = int.from_bytes(h256(b"golden unit")[:8], "big")
        self.assertEqual(batch_indices(seed, 0, 4096, 8).tolist(), [3630, 3813, 3711, 480, 2559, 857, 2137, 293])
        self.assertEqual(batch_indices(seed, 39, 4096, 8).tolist(), [1834, 3808, 3845, 893, 3178, 2397, 2965, 2510])
        self.assertEqual(batch_indices(7, 123456, 100, 5).tolist(), [41, 54, 30, 67, 48])

    def test_starting_weights(self):
        for spec, seed, digest, first in (
                (ModelSpec(), 11, "be6826cb2c7ee929ef7309f1dd4ef0f82f7a83247fac7e0c1f633293e7a031d3",
                 [0.012088968796129921, 0.4807433532274511, 0.43300428986510764]),
                (ModelSpec(8, (16, 16), 3, "float32"), 7, "8c9e2e4c1d853f6fadc377c5c1c67d6dfa1e3eb1aeb4ef42c7c3fb8be8e21517",
                 [0.0004349248774815351, 0.10562250018119812, -0.096922367811203])):
            theta = spec.init_params(seed)
            self.assertEqual(theta[:3].tolist(), first)
            self.assertEqual(params_hash(theta).hex(), digest)

    def test_synthetic_data(self):
        self.assertEqual(Dataset.synthetic(ModelSpec(), 7).commitment().hex(),
                         "cfa66abcc3b31b3be1e0399de7443813fd105759b4e7b9ff856f96fe11310b5a")


if __name__ == "__main__":
    unittest.main()
