"""Each job's rounds are as long as its size needs, so a modest computer on a home connection can still download,
train and upload a unit of a big model in time. And records made before a rule existed still replay."""
import threading
import time
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


class PhaseEndTests(unittest.TestCase):
    """A phase of a job's round lasts at least the node's own length for it and at most the job's (longer for a big
    model, ten minutes at least for a DeepSeek prep unit), ending in between once everything it waits for is in
    (Coordinator.all_in). So a round nobody was handed work in lasts the node's own round, not the job's."""

    def run_round(self, all_in, own: PhaseTimes, job: PhaseTimes) -> tuple[float, list[str]]:
        """Node._run_round with a coordinator whose phases are all in when `all_in()` says so."""
        steps = []
        c = SimpleNamespace(
            lock=threading.Lock(), job=lambda j: SimpleNamespace(wire_dtype=None, sharded=False),
            r=SimpleNamespace(round_no=1, job_id="j", start=None, regs={}, assignments={}, unassigned=0),
            start_compute=lambda: steps.append("compute"), start_reveal=lambda: steps.append("reveal"),
            start_verify=lambda: steps.append("verify"), start_disputes=lambda now: False, all_in=all_in)
        node = SimpleNamespace(coord=c, blobs=SimpleNamespace(put_array=lambda a, d: None), times=own,
                               times_for=lambda j: job, stopping=threading.Event(), phase_ends=0.0)
        node._wait = lambda *a: Node._wait(node, *a)
        began = time.monotonic()
        self.assertTrue(Node._run_round(node))
        return time.monotonic() - began, steps

    def test_a_round_with_nothing_to_wait_for_takes_the_nodes_own_round(self):
        own = PhaseTimes(register_s=0.2, compute_s=0.4, reveal_s=0.2, verify_s=0.3)
        took, steps = self.run_round(lambda: True, own, PhaseTimes(register_s=0.2, compute_s=600, reveal_s=600,
                                                                    verify_s=600))
        self.assertEqual(steps, ["compute", "reveal", "verify"])
        self.assertGreaterEqual(took, own.round_s - 0.05, "never shorter than the node's own round")
        self.assertLess(took, own.round_s + 3, "not the job's 1,800 s")

    def test_a_phase_waiting_for_work_runs_the_jobs_time(self):
        own, job = PhaseTimes(register_s=0, compute_s=0.1, reveal_s=0.1, verify_s=0.1), \
            PhaseTimes(register_s=0, compute_s=0.6, reveal_s=0.3, verify_s=0.3)
        took, _ = self.run_round(lambda: False, own, job)
        self.assertGreaterEqual(took, job.round_s - 0.05, "a unit still out: each phase runs the job's time")
        ends = time.monotonic() + 0.5  # in once a unit trains (half a second in), with ten minutes to do it
        took, _ = self.run_round(lambda: time.monotonic() > ends, own, PhaseTimes(register_s=0, compute_s=600,
                                                                                  reveal_s=0.1, verify_s=0.1))
        self.assertGreaterEqual(took, 0.5)
        self.assertLess(took, 3, "the compute phase ends once it's all in")

    def test_a_phase_with_its_work_in_still_stops_for_the_node(self):
        node = SimpleNamespace(stopping=threading.Event(), phase_ends=0.0)
        began = time.monotonic()
        self.assertTrue(Node._wait(node, 600, done=lambda: True))
        wall = time.time()
        self.assertTrue(Node._wait(node, 600, done=lambda: True, least=0.4))
        self.assertGreaterEqual(time.monotonic() - began, 0.4, "no sooner than `least`")
        self.assertLess(time.monotonic() - began, 3)
        self.assertLess(abs(node.phase_ends - (wall + 0.4)), 0.3, "/status says when it really ends, not ten minutes")
        self.assertFalse(Node._wait(node, 600, still=lambda: False, done=lambda: False), "the round's no longer wanted")
        node.stopping.set()
        self.assertFalse(Node._wait(node, 600, done=lambda: False))
        self.assertFalse(Node._wait(node, 600, done=lambda: True, least=0.1), "stopping, even with the work in")

    def test_status_says_the_latest_a_phase_can_end_then_when_it_will(self):
        """On test night the dashboard counted down a DeepSeek round's ten minutes, then the round ended twenty seconds
        in: /status's phase_ends was always the latest. While work is out it still is (nobody waiting is cut short);
        once the work is in, it's when the phase really ends."""
        node, seen, arrived = SimpleNamespace(stopping=threading.Event(), phase_ends=0.0), [], threading.Event()

        def done():
            seen.append(node.phase_ends)
            return arrived.is_set()
        threading.Timer(0.5, arrived.set).start()
        wall = time.time()
        self.assertTrue(Node._wait(node, 600, done=done, least=0.1))
        self.assertGreater(seen[0], wall + 590, "work still out: the latest it can end")
        self.assertLess(node.phase_ends, wall + 2, "then: when it really ended")
        self.assertLess(time.time() - wall, 3)


class AllInTests(unittest.TestCase):
    """Coordinator.all_in on real rounds: a phase is all in once its last commitment, reveal or verdict is, and a phase
    with nothing to wait for is all in at once. Such rounds seal and replay as they always did."""

    def register(self, proto):
        c, r = proto.coord, proto.coord.r
        for addr, w in proto.workers.items():
            body = {"round": r.round_no, "flops_per_sec": w.hardware.flops_per_sec, "memory_mb": w.hardware.memory_mb}
            c.register(addr, w.hardware.flops_per_sec, w.hardware.memory_mb,
                       envelope=proto._signed(w, "register", body))

    def seal(self, proto, beacon):
        from vibehash.protocol import report_ts
        return proto.coord.finalize(proposer_keys=proto._proposer(beacon), timestamp=report_ts(proto.chain))

    def test_each_phase_is_all_in_once_its_last_message_is(self):
        proto, g = network(newcomer_units=100)
        c = proto.coord
        c.open_round()
        r, job = c.r, c.job(c.r.job_id)
        self.assertFalse(c.all_in(), "nobody can tell who else will register")
        self.assertIsNone(c.progress(), "registration counts nothing (/status's progress)")
        self.register(proto)
        c.start_compute()
        held = sorted((uid, a) for a, uids in r.assignments.items() for uid in uids)
        self.assertGreater(len(held), 1)
        for i, (uid, a) in enumerate(held):
            self.assertFalse(c.all_in(), f"{uid} isn't committed yet")
            self.assertEqual(c.progress(), (i, len(held)))
            c.commit(proto.workers[a].compute(r.units[uid], job, r.start))
        self.assertTrue(c.all_in())
        self.assertEqual(c.progress(), (len(held), len(held)))
        beacon = c.start_reveal()
        for i, (uid, a) in enumerate(held):
            self.assertFalse(c.all_in(), f"{uid} isn't revealed yet")
            self.assertEqual(c.progress(), (i, len(held)))
            rv = proto.workers[a].reveal(r.units[uid], job, beacon, c.config.challenge_segments)
            c.reveal(rv, wire=proto._wire(rv, job))
        self.assertTrue(c.all_in())
        c.start_verify()
        asked = [t for t in r.tasks.values() if t.verifier]
        self.assertGreater(len(asked), 1)
        self.assertTrue(any(t.canary for t in asked))
        for i, t in enumerate(asked):
            self.assertFalse(c.all_in(), "a task isn't answered yet")
            self.assertEqual(c.progress(), (i, len(asked)))
            w = proto.workers[t.verifier]
            ok, bad = w.verify(job, t.unit, t.openings, t.segments)
            body = {"round": r.round_no, "task_id": t.task_id, "ok": bool(ok), "bad_segment": bad}
            c.submit_verdict(t.verifier, t.task_id, ok, bad, envelope=proto._signed(w, "verdict", body))
        self.assertTrue(c.all_in())
        self.assertFalse(c.start_disputes())
        rep = self.seal(proto, beacon)
        self.assertGreater(rep.block.summary.accepted_units, 0)
        replays(self, proto, g)

    def test_a_round_nobody_was_handed_work_in_is_all_in_at_once(self):
        proto, g = network()
        c = proto.coord
        c.open_round()
        c.start_compute()  # nobody registered
        self.assertTrue(c.all_in())
        beacon = c.start_reveal()
        self.assertTrue(c.all_in())
        c.start_verify()
        self.assertFalse(c.r.tasks)
        self.assertTrue(c.all_in())
        rep = self.seal(proto, beacon)
        self.assertEqual(rep.block.summary.accepted_units, 0)
        replays(self, proto, g)

    def test_a_fine_tune_short_of_its_quorum_is_all_in_at_once(self):
        from tests.test_finetune import job_of

        w = Worker("w0", KeyPair.generate(b"quorum-w0"), HardwareProfile("gpu", 1e12, 16384, 1.0), "honest")
        req = KeyPair.generate(b"quorum-req")
        proto = Protocol(Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE}), [w], CoordinatorConfig())
        proto.post_job(req, job_of(units_per_round=4, memory_mb=2048))
        c = proto.coord
        c.open_round()
        self.register(proto)  # one staked worker: a fine-tune needs two, one to train a unit and one to check it
        c.start_compute()
        self.assertIn("waits", c.r.waiting)
        self.assertEqual(c.r.unassigned, 4)
        self.assertTrue(c.all_in())
        c.start_reveal()
        self.assertTrue(c.all_in())
        c.start_verify()
        self.assertFalse(c.r.tasks, "no scorers when nothing was trained")
        self.assertTrue(c.all_in())

    def test_a_sharded_round_keeps_its_phases(self):
        from tests.test_sharded_rounds import network as sharded
        proto, _, _ = sharded([("honest",)] * 3)
        c = proto.coord
        c.open_round()
        c.start_compute()
        self.assertFalse(c.all_in(), "teams trade edges during compute, and send edge evidence during the reveal")
        c.start_reveal()
        self.assertFalse(c.all_in())


class EmptyRoundTests(unittest.TestCase):
    """On a real node: a job sized for ten-minute compute phases that nobody works on has rounds as long as the node's
    own, no shorter; a round with work ends once its work is in. Their records and blocks are as they always were: the
    compute phase in the rules is the job's, and every round replays."""

    def audit(self, api, top: int) -> list[dict]:
        """Every round's record up to `top`, each replayed against the chain as an outside auditor rebuilds it."""
        from vibehash.ledger import Block
        audit, records = Chain.from_genesis(api.get("/genesis")), []
        for d in api.get(f"/blocks?from=1&to={top}"):
            block = Block.from_dict(d)
            if block.summary is not None:
                records.append(record := api.get(f"/round/{block.height}/record"))
                self.assertEqual(replay_block(audit, block, record), [], f"block {block.height}")
                self.assertEqual(record["round"], block.height)
            audit.append(block)
        self.assertTrue(audit.validate())
        return records

    def test_a_round_nobody_works_in_lasts_the_nodes_own_round(self):
        import tempfile
        from pathlib import Path

        from vibehash.client import NodeClient

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        keys, req = KeyPair.generate(b"empty-node"), KeyPair.generate(b"empty-req")
        genesis = {"chain_id": "empty-rounds", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {keys.address: 10 * VIBE}}
        own = PhaseTimes(register_s=0.3, compute_s=0.5, reveal_s=0.3, verify_s=0.4, idle_s=0.1)
        node = Node(Path(tmp.name) / "node", genesis, keys, port=0, times=own).start()
        self.addCleanup(node.stop)
        api = NodeClient(node.url)
        big = TrainingJob("asks-long", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=4,
                          compute_s=600)
        self.assertGreater(node.times_for(big).round_s, 600)
        api.send_tx(req, "post_job", "system", big.budget, big.spec_json())
        sealed, deadline = {}, time.monotonic() + 60
        while len(sealed) < 4 and time.monotonic() < deadline:
            tip = node.chain.tip
            if tip.summary is not None:
                sealed.setdefault(tip.height, time.monotonic())
            time.sleep(0.02)
        self.assertEqual(len(sealed), 4, "four empty rounds in a minute, where one of the job's would take ten")
        heights = sorted(sealed)
        self.assertEqual(heights, list(range(heights[0], heights[0] + 4)))
        for a, b in zip(heights, heights[1:]):
            self.assertGreaterEqual(sealed[b] - sealed[a], own.round_s - 0.1, "no shorter than the node's own round")
            self.assertLess(sealed[b] - sealed[a], own.round_s + 8)
        records = self.audit(api, heights[-1])
        self.assertGreaterEqual(len(records), 4)
        for record in records:
            self.assertEqual(record["rules"]["compute_s"], 0.8 * 600, "the job's compute phase, which capacity reads")
            self.assertFalse(record["commitments"])

    def test_a_round_with_work_ends_once_its_work_is_in(self):
        from tests.test_network import run_until, start_network

        own = PhaseTimes(register_s=1.0, compute_s=1.0, reveal_s=0.5, verify_s=0.5, idle_s=0.2)
        node, api, req, _, clients, stop, _, _, _ = start_network(self, 3, times=own)
        job = TrainingJob("asks-long", req.address, budget=60 * VIBE, fee_per_round=10 * VIBE, units_per_round=4,
                          compute_s=120)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        height = node.chain.tip.height
        run_until(node, clients, stop, height + 4, seconds=90)
        self.assertGreaterEqual(node.chain.tip.height, height + 4, "three rounds in 90 s, each given 120 s of compute")
        records = self.audit(api, node.chain.tip.height)
        self.assertTrue(any(r["commitments"] for r in records), "the workers trained")
        self.assertTrue(all(r["rules"]["compute_s"] == 0.8 * 120 for r in records))
        accepted = sum(node.chain.blocks[h].summary.accepted_units for h in range(height + 1, node.chain.tip.height + 1)
                       if node.chain.blocks[h].summary is not None)
        self.assertGreater(accepted, 0)


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
