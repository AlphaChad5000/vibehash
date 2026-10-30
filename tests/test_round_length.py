"""Each job's rounds are as long as its size needs, so a modest computer on a home connection can still download,
train and upload a unit of a big model in time. And records made before a rule existed still replay."""
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain
from vibehash.model import ModelSpec
from vibehash.node import Node, PhaseTimes
from vibehash.protocol import Protocol
from vibehash.replay import replay_block
from vibehash.tasks import REF_UPLOAD, SYN_PER_VIBE, InvalidJob, TrainingJob
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker

VIBE = SYN_PER_VIBE
RELIABLE = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)


def job(widths=(32,), **kw):
    return TrainingJob("sized", "req", budget=0, fee_per_round=VIBE, spec=ModelSpec(8, widths, 4, "float32"), **kw)


def network(**cfg):
    workers = [Worker(f"w{i}", KeyPair.generate(f"len-w{i}".encode()), RELIABLE, "honest") for i in range(3)]
    req = KeyPair.generate(b"len-req")
    g = dict(allocations={req.address: 500 * VIBE}, stakes={w.address: 10 * VIBE for w in workers}, chain_id="len-test")
    proto = Protocol(Chain(**g), workers, CoordinatorConfig(canaries_per_round=1, **cfg), seed=3)
    proto.post_job(req, TrainingJob("len-job", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=6))
    return proto, g


def replays(test, proto, g):
    chain, records = Chain(**g), {r.block.height: r.record for r in proto.history}
    for block in proto.chain.blocks[1:]:
        if block.summary is not None:
            test.assertEqual(replay_block(chain, block, records[block.height]), [], f"block {block.height}")
        chain.append(block)


class PhaseLengthTests(unittest.TestCase):
    def test_a_bigger_model_gets_longer_phases(self):
        small, big = job().phase_lengths(), job((4096, 4096, 4096)).phase_lengths()
        for phase in ("compute", "reveal", "verify"):
            self.assertGreater(big[phase], 10 * small[phase], phase)
        b = job((4096, 4096, 4096))
        # Enough to upload every checkpoint of one unit over a 20 Mbit/s link, and then some.
        self.assertGreater(big["compute"], b.n_segments * b.checkpoint_bytes / REF_UPLOAD)

    def test_a_job_can_ask_for_a_longer_compute_phase_but_not_a_shorter_one(self):
        self.assertEqual(job(compute_s=600).phase_lengths()["compute"], 600)
        big = job((4096, 4096, 4096))
        self.assertEqual(job((4096, 4096, 4096), compute_s=5).phase_lengths()["compute"], big.phase_lengths()["compute"])
        for bad in (1, 4.9, 3601, -5):
            with self.assertRaises(InvalidJob):
                job(compute_s=bad)

    def test_job_specs_without_it_are_unchanged(self):
        plain = job()
        self.assertNotIn("compute_s", plain.to_spec(), "specs posted before it existed must keep their exact text")
        asked = job(compute_s=120)
        self.assertEqual(TrainingJob.from_spec_json(asked.spec_json(), "req", 0).compute_s, 120)
        self.assertEqual(TrainingJob.from_spec_json(plain.spec_json(), "req", 0).compute_s, 0)

    def test_a_node_uses_its_own_phase_lengths_or_the_jobs_whichever_are_longer(self):
        node = SimpleNamespace(times=PhaseTimes(register_s=3, compute_s=20, reveal_s=5, verify_s=10))
        same = Node.times_for(node, job())
        self.assertEqual((same.compute_s, same.reveal_s, same.verify_s), (20, 5, 10), "a small job keeps the node's")
        longer = Node.times_for(node, job((4096, 4096, 4096)))
        self.assertGreater(longer.compute_s, 20)
        self.assertGreater(longer.round_s, same.round_s)
        self.assertEqual(longer.register_s, 3)


class CapacityTests(unittest.TestCase):
    def test_capacity_follows_the_rounds_compute_phase_and_replays(self):
        proto, g = network(newcomer_units=100)
        proto.coord.timing = lambda j: 2.0  # tiny: nobody can claim more than 2 s of work
        short = proto.run_round()
        self.assertEqual(short.record["rules"]["compute_s"], 2.0)
        proto.coord.timing = lambda j: 1e6
        long = proto.run_round()
        self.assertEqual(long.record["rules"]["compute_s"], 1e6)
        self.assertLess(sum(short.worker_flops.values()), sum(long.worker_flops.values()), "more time, more units")
        replays(self, proto, g)

    def test_records_made_before_a_rule_existed_still_replay(self):
        old_rules = CoordinatorConfig.rules

        def without_max_newcomers(self):
            rules = old_rules(self)
            del rules["max_newcomers"]
            return rules
        proto, g = network()
        with mock.patch.object(CoordinatorConfig, "rules", without_max_newcomers):
            proto.run_round()
        self.assertNotIn("max_newcomers", proto.history[-1].record["rules"])
        replays(self, proto, g)  # with today's code, which has the rule


if __name__ == "__main__":
    unittest.main()
