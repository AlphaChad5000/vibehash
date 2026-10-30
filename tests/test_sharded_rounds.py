"""Sharded jobs in the round protocol (coordinator.py, shards.py, SHARDING.md): teams of workers, one per shard, train
units as pipelines; checkers each check one cell, holding one shard; neighbours' edge hashes are compared, and the
signed edge message a receiver kept says who lied; members are paid per verified cell; and every round replays from
its record, while whole units' rounds and records stay exactly as they were."""
import copy
import dataclasses
import json
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import coordinator as coord_module
from vibehash import shards as sh
from vibehash.coordinator import (RECORD_VERSION, SHARDED_RECORD_VERSION, CellTask, Coordinator, CoordinatorConfig,
                                  _mean)
from vibehash.crypto import KeyPair, h256
from vibehash.ledger import Block, Chain, canonical
from vibehash.model import ModelSpec, params_hash, train_segment_numpy
from vibehash.protocol import Protocol
from vibehash.replay import replay_block
from vibehash.tasks import SYN_PER_VIBE, InvalidJob, TrainingJob, Verdict, WorkUnit
from vibehash.verification import challenge_segments
from vibehash.worker import NETBOOK, OFFICE_DESKTOP, HardwareProfile, Worker

VIBE = SYN_PER_VIBE
RELIABLE = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)
SMALL = HardwareProfile(NETBOOK.name, NETBOOK.flops_per_sec, NETBOOK.memory_mb, reliability=1.0)  # 512 MB
# Four layers of uneven size (216, 500, 336 and 68 weights), float32 as GPUs train, small enough for many rounds.
SHARDED = dict(spec=ModelSpec(8, (24, 20, 16), 4, "float32"), steps_per_unit=12, checkpoint_every=3, batch_size=16,
               tolerance=1e-4, units_per_round=4)
V1_FIXTURE = Path(__file__).parent / "data" / "record_v1.json"


def network(behaviors, shards=3, hardware=None, seed=3, job_kw=None, **cfg):
    hardware = hardware or [RELIABLE] * len(behaviors)
    workers = [Worker(f"t{i}", KeyPair.generate(f"t{i}".encode()), hw, *b) for i, (b, hw) in enumerate(zip(behaviors, hardware))]
    req = KeyPair.generate(b"sharded req")
    chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers}, chain_id="sharded-test")
    proto = Protocol(chain, workers, CoordinatorConfig(**cfg), seed=seed)
    job = TrainingJob("sharded", req.address, budget=500 * VIBE, fee_per_round=10 * VIBE, shards=shards, **(SHARDED | (job_kw or {})))
    proto.post_job(req, job)
    return proto, workers, job


def shard_work(job, k) -> int:
    """A member's pay for one unit: the FLOPs of all its cells, which add up to its shard of the unit."""
    shard = sh.split(job.spec, job.shards)[k]
    return job.n_segments * sh.cell_flops(job.spec, shard, job.batch_size, job.checkpoint_every)


def resealed(block, record):
    return dataclasses.replace(block, summary=dataclasses.replace(block.summary, record_hash=h256(canonical(record)).hex()))


def replay_all(before, reports, checkpoints=None, stats=None, edit=None):
    chain, found = copy.deepcopy(before), []
    for i, rep in enumerate(reports):
        block, record = rep.block, copy.deepcopy(rep.record)
        if edit:
            block, record = edit(i, block, record)
        found.append(replay_block(chain, block, record, checkpoints, stats))
        chain.append(rep.block)
    return found


class JobSpecTests(unittest.TestCase):
    def test_specs_without_shards_are_unchanged(self):
        """Specs already on chain parse and serialise to the same bytes, and a job without shards writes none."""
        fixture = json.loads(V1_FIXTURE.read_text())
        memo = next(tx["memo"] for b in fixture["blocks"] for tx in b["txs"] if tx["kind"] == "post_job")
        job = TrainingJob.from_spec_json(memo, "req", 1)
        self.assertEqual((job.shards, job.sharded, job.spec_json()), (1, False, memo))
        self.assertNotIn("shards", TrainingJob("j", "r", 0, 0, shards=1).to_spec())
        three = TrainingJob("j", "r", 0, 0, shards=3, **SHARDED)
        self.assertEqual(three.to_spec()["shards"], 3)
        self.assertEqual(TrainingJob.from_spec_json(three.spec_json(), "r", 0).shards, 3)

    def test_shards_are_blocks_of_whole_layers(self):
        for bad in (0, -1, 5):
            with self.assertRaises(InvalidJob):
                TrainingJob("j", "r", 0, 0, shards=bad, **SHARDED)
        TrainingJob("j", "r", 0, 0, shards=4, **SHARDED)  # one layer each
        with self.assertRaises(InvalidJob):
            TrainingJob("j", "r", 0, 0, shards=3)  # the default model has two layers


class HonestTeamTests(unittest.TestCase):
    def test_an_honest_sharded_job_trains_pays_per_cell_and_matches_unsharded_training(self):
        proto, workers, job = network([("honest",)] * 6)
        start = proto.coord.start_params(job.job_id).copy()
        report = proto.run_round()
        self.assertEqual(len(report.teams), 2)
        self.assertEqual(report.block.summary.accepted_units, job.units_per_round)
        self.assertEqual(report.slashes, [])
        # Each member is paid for its cells' FLOPs: its shard of each of its team's units.
        units = {tuple(t["members"]): len(t["units"]) for t in report.record["teams"]}
        want = {a: n * shard_work(job, k) for members, n in units.items() for k, a in enumerate(members)}
        self.assertEqual(report.worker_flops, want)
        self.assertEqual(sum(want.values()), job.units_per_round * job.unit_flops)  # shards add up to whole units
        rates = [report.worker_pay[a] / f for a, f in report.worker_flops.items()]
        self.assertLess(max(rates) / min(rates), 1.0001)  # the same pay per FLOP whatever the shard
        # The model is exactly the average of the same units trained whole, on one computer.
        units = [WorkUnit.make(job, report.round, i, params_hash(start)) for i in range(job.units_per_round)]
        whole = [train_segment_numpy(job.spec, job.data, start, u.seed, 0, job.steps_per_unit, job.lr, job.batch_size) for u in units]
        np.testing.assert_array_equal(report.params, _mean(whole, job))
        # And the same as a whole-unit job's round of the same units, by honest workers that don't add noise.
        proto1, _, _ = network([("honest",)] * 6, shards=1, replication_rate=0, canaries_per_round=0)
        for w in proto1.workers.values():
            w.fp_noise = 0
        np.testing.assert_allclose(proto1.run_round().params, report.params, rtol=1e-6, atol=0)
        later = [proto.run_round() for _ in range(3)][-1]
        self.assertLess(later.val_loss, report.val_loss)

    def test_teams_dont_overlap_and_no_one_checks_a_unit_it_holds_a_shard_of(self):
        proto, workers, job = network([("honest",)] * 9, shards=2)
        served = proto.coord.tasks_for
        seen = []

        def tasks_for(addr):
            tasks = served(addr)
            seen.extend((t, set(proto.coord.r.holders[t.unit.unit_id])) for t in tasks)
            return tasks
        proto.coord.tasks_for = tasks_for
        for _ in range(3):
            report = proto.run_round()
            members = [a for team in report.teams for a in team]
            self.assertEqual(len(members), len(set(members)))
        self.assertTrue(seen and all(isinstance(t, CellTask) and t.verifier not in holders for t, holders in seen))

    def test_a_shard_fits_where_the_whole_model_doesnt(self):
        """Netbooks have half the memory the job needs, so none can train it whole, but a team of them can, a layer
        each (the biggest layer needs 458 MB of the model's 1024)."""
        proto, workers, job = network([("honest",)] * 6, shards=4, hardware=[SMALL] * 6)
        report = proto.run_round()
        self.assertEqual(report.block.summary.accepted_units, job.units_per_round)
        self.assertEqual(len(report.teams), 1)
        whole, _, _ = network([("honest",)] * 6, shards=1, hardware=[SMALL] * 6)
        self.assertEqual(whole.run_round().unassigned, job.units_per_round)


class TooFewWorkersTests(unittest.TestCase):
    def test_too_few_for_a_team_trains_nothing_and_says_why(self):
        """A unit needs every shard, so two workers can't train a three-shard job: the round settles with nothing
        trained or paid, nothing released from escrow, and a reason a person can act on."""
        proto, workers, job = network([("honest",)] * 2)
        before, start = copy.deepcopy(proto.chain), proto.coord.start_params(job.job_id).copy()
        report = proto.run_round()
        self.assertEqual((report.teams, report.unassigned, report.worker_flops), ([], job.units_per_round, {}))
        self.assertEqual(report.block.summary.params_hash, params_hash(start).hex())
        self.assertEqual(proto.chain.state.escrow[job.job_id], job.budget)
        self.assertEqual(report.waiting, "job sharded waits: it needs 3 workers with at least 655 MB of memory for its "
                                         "3 shards; 2 of the 2 registered this round have that much")
        self.assertEqual(proto.coord.waiting[job.job_id], report.waiting)
        self.assertNotIn("waits", canonical(report.record).decode())  # for people, not for the record
        self.assertEqual(replay_all(before, [report]), [[]])

    def test_too_little_memory_says_how_much_is_needed(self):
        proto, workers, job = network([("honest",)] * 4, job_kw={"memory_mb": 40_000})
        report = proto.run_round()
        self.assertIn("needs 3 workers with at least 25.0 GB of memory for its 3 shards; 0 of the 4", report.waiting)
        workers[0].hardware = workers[1].hardware = workers[2].hardware = HardwareProfile("big", 2e6, 32_000, 1.0)
        report = proto.run_round()
        self.assertEqual((len(report.teams), report.waiting), (1, ""))
        self.assertNotIn(job.job_id, proto.coord.waiting)

    def test_a_node_leaves_sharded_jobs_waiting(self):
        """A node's workers can't send each other tensors yet, so a node runs whole-unit jobs and says why the sharded
        one waits, rather than assigning units nobody could train."""
        proto, workers, job = network([("honest",)] * 3)
        node = Coordinator(proto.chain)
        node.add_job(job)
        self.assertIsNone(node.active_job_id())
        self.assertIn("can't run teams yet", node.waiting[job.job_id])
        self.assertEqual(proto.coord.active_job_id(), job.job_id)


class CheatTests(unittest.TestCase):
    def run_with(self, behavior, n=6):
        proto, workers, job = network([(behavior,)] + [("honest",)] * (n - 1), replication_rate=0, canaries_per_round=0)
        report = proto.run_round()
        cheater = workers[0].address
        team = next(t for t in report.record["teams"] if cheater in t["members"])
        return proto, job, report, cheater, team

    def check_caught(self, behavior, reason, verdict):
        proto, job, report, cheater, team = self.run_with(behavior)
        self.assertEqual({s[2] for s in report.slashes if s[0] == cheater}, {reason}, behavior)
        self.assertEqual({r.verdict for r in report.results if r.worker == cheater}, {verdict}, behavior)
        self.assertNotIn(cheater, report.worker_flops)
        self.assertEqual(proto.chain.state.stake(cheater), 5 * VIBE)
        self.assertEqual({s[0] for s in report.slashes}, {cheater}, behavior)  # nobody else is blamed
        # Its honest teammates are paid in full for their cells, though their team's update is left out.
        for k, a in enumerate(team["members"]):
            if a != cheater:
                self.assertEqual(report.worker_flops[a], len(team["units"]) * shard_work(job, k), behavior)
        self.assertEqual(report.block.summary.accepted_units, job.units_per_round - len(team["units"]), behavior)
        return report

    def test_each_cheat_fails_its_cells_and_its_teammates_are_paid(self):
        for behavior in sh.CHEATS:
            with self.subTest(behavior=behavior):
                self.check_caught(behavior, "failed_replay", Verdict.FAILED_REPLAY)

    def test_a_lying_edge_commitment_is_caught_at_the_edge(self):
        """It computes honestly, so its cells would pass: what it lies about is what crossed an edge, one way or the
        other, and its neighbour's signed edge message shows it."""
        for behavior in sh.EDGE_CHEATS:
            with self.subTest(behavior=behavior):
                report = self.check_caught(behavior, "lied at an edge", Verdict.BAD_REVEAL)
                catchers = {s[1] for s in report.slashes}
                self.assertTrue(catchers and None not in catchers)  # the honest neighbour, whose evidence proved it

    def test_both_sides_of_an_edge_lying_are_both_caught(self):
        proto, workers, job = network([("lies_sent",), ("lies_received",)], shards=2, replication_rate=0, canaries_per_round=0)
        report = proto.run_round()
        self.assertEqual({(s[0], s[1], s[2]) for s in report.slashes},
                         {(w.address, None, "lied at an edge") for w in workers})

    def test_a_receiver_that_cant_show_what_it_was_sent_is_blamed(self):
        """An edge dispute is settled by the receiver's copy of what the sender signed; without one, the receiver is
        the one who can't back its commitment up, even if (as here) the sender lied."""
        proto, workers, job = network([("lies_sent",)] + [("honest",)] * 5, replication_rate=0, canaries_per_round=0)
        with mock.patch.object(Worker, "edge_evidence", lambda self, uid, step, kind: None):
            report = proto.run_round()
        liar = workers[0].address
        team = next(t for t in report.record["teams"] if liar in t["members"])
        k = team["members"].index(liar)
        receivers = team["members"][max(0, k - 1):k] + team["members"][k + 1:k + 2]  # it lies to each neighbour it sends to
        self.assertEqual({s[0] for s in report.slashes}, set(receivers))
        self.assertEqual(report.record["evidence"], [])

    def test_a_dropped_teammate_costs_the_unit_not_blame(self):
        hw = [HardwareProfile("flaky", 2e6, 8192, reliability=0.0)] + [RELIABLE] * 5
        proto, workers, job = network([("honest",)] * 6, hardware=hw, replication_rate=0, canaries_per_round=0)
        report = proto.run_round()
        gone = workers[0].address
        team = next(t for t in report.record["teams"] if gone in t["members"])
        k = team["members"].index(gone)
        neighbours = set(team["members"][max(0, k - 1):k] + team["members"][k + 1:k + 2])
        self.assertEqual(report.slashes, [])
        self.assertEqual(set(report.block.summary.missed), {gone})
        self.assertFalse(neighbours & set(report.worker_flops))  # their edge with it can't be checked
        self.assertEqual(report.block.summary.accepted_units, job.units_per_round - len(team["units"]))


class CellCanaryTests(unittest.TestCase):
    """A checker sees what the node would serve it: each task's id, unit, shard, segment, root and cell. From that it
    must not be able to tell a canary from a real cell."""

    @classmethod
    def setUpClass(cls):
        proto, workers, cls.job = network([("honest",)] * 8, shards=2, canaries_per_round=2)
        c, cls.views, cls.leaked = proto.coord, [], {}
        served = c.tasks_for

        def tasks_for(addr):
            tasks = served(addr)
            if c.r.round_no not in cls.leaked:  # anything serving the record while checkers work would show this
                cls.leaked[c.r.round_no] = c.r.secret.hex() in canonical(c.r.record).decode()
            for t in tasks:
                team = c.r.team_of[t.unit.unit_id].members
                cls.views.append({"beacon": c.r.beacon, "checker": addr, "round": c.r.round_no, "task": t,
                                  "canary": t.canary, "holder": team[t.shard.index], "team": set(team)})  # the truth
            return tasks
        c.tasks_for = tasks_for
        cls.reports = [proto.run_round() for _ in range(6)]
        cls.canaries = [v for v in cls.views if v["canary"]]

    def check(self, t):
        j = self.job
        return sh.check_cell(j.spec, t.shard, j.data, t.root, t.segments[0], t.cell, t.unit.seed, j.steps_per_unit,
                             j.checkpoint_every, j.lr, j.batch_size, j.tolerance)

    def test_the_round_secret_stays_hidden_until_the_round_is_settled(self):
        """Record version 2's guarantee, kept: the secret behind the beacon, the task ids and the canaries is in the
        record only once every verdict is in, and it's the one the round opened by committing to."""
        self.assertEqual(self.leaked, {rep.round: False for rep in self.reports})
        for rep in self.reports:
            self.assertEqual(rep.record["beacon_commit"], coord_module.secret_commitment(bytes.fromhex(rep.record["secret"])).hex())

    def test_there_are_canaries_and_only_their_end_weights_fail(self):
        self.assertGreaterEqual(len(self.canaries), 8)
        for v in self.views:
            self.assertEqual(self.check(v["task"]), "its weights at the segment's end aren't what training gives" if v["canary"] else None)
        self.assertFalse(any(s[2] == "approved a canary" for rep in self.reports for s in rep.slashes))

    def test_ids_dont_follow_from_public_data(self):
        for v in self.views:
            t = v["task"]
            old = {h256(v["beacon"], t.unit.unit_id.encode(), KeyPair.generate(v["beacon"] + bytes([n])).address.encode(),
                        v["checker"].encode(), t.segments[0].to_bytes(8, "big")).hex()[:20] for n in range(2)}
            self.assertNotIn(t.task_id, old)

    def test_a_canary_cell_is_consistent_and_looks_like_training(self):
        """Every opening proves against the task's root, and the update keeps honest training's structure: the
        softmax layer's bias updates sum to ~0 (a last shard's cell holds them), which noise wouldn't."""
        spec = self.job.spec
        for v in self.views:
            t = v["task"]
            n_ckpt = self.job.n_segments + 1
            leaves = {t.segments[0]: t.cell["before"], t.segments[0] + 1: t.cell["after"]}
            for step, rec in zip(range(t.segments[0] * 3, t.segments[0] * 3 + 3), t.cell["tensors"]):
                leaves |= {sh.leaf_index(n_ckpt, step, k): rec[k] for k in sh.KINDS}
            self.assertEqual({sh.root_from(sh.tensor_hash(a), t.cell["proofs"][i]) for i, a in leaves.items()}, {t.root})
            if t.shard.last:
                (_, before), (_, after) = (t.shard.layers(spec, t.cell[e])[-1] for e in ("before", "after"))
                step = after - before
                self.assertLess(abs(step.sum()), 1e-4 * np.linalg.norm(step), v["canary"])
        self.assertTrue(any(v["task"].shard.last for v in self.canaries))

    def test_who_checks_a_canary_gives_nothing_away(self):
        """Like a real cell, it goes to a checker holding no shard of the unit, with a segment the beacon challenged
        for that shard's holder; and to one with no real cell of the unit, whose root or checkpoints it could compare."""
        per_unit = defaultdict(list)
        for v in self.views:
            per_unit[(v["round"], v["checker"], v["task"].unit.unit_id)].append(v)
        for v in self.views:
            t = v["task"]
            self.assertNotIn(v["checker"], v["team"])
            self.assertIn(t.segments[0], challenge_segments(v["beacon"], t.unit.unit_id, v["holder"], self.job.n_segments, 3))
            if v["canary"]:
                self.assertEqual(len(per_unit[(v["round"], v["checker"], t.unit.unit_id)]), 1)

    def test_a_lazy_checker_is_caught_by_them(self):
        """It's too slow to train a shard, so it only checks. A canary on a unit goes to a checker with no real cell of
        it, and each unit's six real cells go to six possible checkers, so it's eligible for a canary about a third of
        the time, and caught in about one round in three (measured): thirty rounds all missing it is one in a million."""
        checker = HardwareProfile("checker only", 1e4, 8192, reliability=1.0)
        proto, workers, job = network([("honest", "lazy")] + [("honest",)] * 7, shards=2, hardware=[checker] + [RELIABLE] * 7,
                                      canaries_per_round=2)
        caught = set()
        for _ in range(30):
            caught |= {s[0] for s in proto.run_round().slashes if s[2] == "approved a canary"}
            if caught:
                break
        self.assertEqual(caught, {workers[0].address})


class ShardedReplayTests(unittest.TestCase):
    # Cheaters fall below the minimum stake after their second slash. Enough honest workers stay for two teams, so
    # some honest member reveals each shard's starting weights every round, which a recheck needs.
    EVERYONE = [("honest",)] * 6 + [("noisy_output",), ("zero_gradient",), ("frozen",), ("lies_sent",), ("lies_received",),
                                    ("honest", "lazy"), ("honest", "false_accuser"), ("withholder",)]

    @classmethod
    def setUpClass(cls):
        cls.proto, cls.workers, cls.job = network(cls.EVERYONE, canaries_per_round=2)
        cls.before = copy.deepcopy(cls.proto.chain)
        cls.reports = [cls.proto.run_round() for _ in range(4)]

    def test_every_sharded_round_replays_and_rechecks(self):
        self.assertEqual(replay_all(self.before, self.reports), [[]] * 4)
        stats = {}
        self.assertEqual(replay_all(self.before, self.reports, self.proto.checkpoints, stats), [[]] * 4)
        self.assertGreater(stats["rechecked"], 40)
        self.assertEqual(stats["unchecked"], 0)
        reasons = {s[2] for rep in self.reports for s in rep.slashes}
        self.assertLessEqual({"failed_replay", "lied at an edge", "false accusation"}, reasons)
        self.assertTrue(all(rep.record["version"] == SHARDED_RECORD_VERSION for rep in self.reports))
        self.assertTrue(any(rep.record["evidence"] for rep in self.reports))

    def test_records_with_other_teams_or_forged_edge_messages_fail(self):
        def swap_teams(i, block, record):
            t = record["teams"][0]
            t["members"] = t["members"][::-1]
            return resealed(block, record), record
        self.assertIn("teams", replay_all(self.before, self.reports[:1], edit=swap_teams)[0][0])
        first = next(i for i, rep in enumerate(self.reports) if rep.record["evidence"])

        def forge(i, block, record):
            if i == first:
                env = record["evidence"][0]
                env["body"] = dict(env["body"], hash=h256(b"forged").hex())
                block = resealed(block, record)
            return block, record
        self.assertIn("bad signature", replay_all(self.before, self.reports[:first + 1], edit=forge)[first][0])

    def test_pruned_checkpoints_mean_unchecked_not_failed(self):
        store = copy.deepcopy(self.proto.checkpoints)
        store.arrays.clear()
        stats = {}
        self.assertEqual(replay_all(self.before, self.reports, store, stats), [[]] * 4)
        self.assertEqual(stats["rechecked"], 0)
        self.assertGreater(stats["unchecked"], 0)

    def test_siding_with_a_false_accuser_is_caught_by_a_recheck(self):
        """Arbitration's replay of a rejected cell is a weight-based result (Coordinator._claim): a coordinator that
        upholds a false accusation, slashing an honest member, is caught by anyone holding the checkpoints."""
        proto, workers, job = network([("honest", "false_accuser")] + [("honest",)] * 5, replication_rate=0, canaries_per_round=0)
        before = copy.deepcopy(proto.chain)
        with mock.patch.object(Coordinator, "_check_task", lambda self, j, t: [False, t.segments[0]]):
            report = proto.run_round()
        self.assertFalse(any(s[2] == "false accusation" for s in report.slashes))
        self.assertTrue(any(s[1] == workers[0].address for s in report.slashes))
        self.assertEqual(replay_all(before, [report]), [[]])  # self-consistent: a replay alone can't tell
        found = replay_all(before, [report], proto.checkpoints)[0]
        self.assertTrue(found and all(p.startswith("arbitration check") and "re-running it gives True" in p for p in found), found)

    def test_versions_belong_to_their_jobs(self):
        """Version 3 is a sharded job's record and only theirs: neither kind of round replays as the other."""
        rep = self.reports[0]
        record = dict(copy.deepcopy(rep.record), version=2)
        problems = replay_block(copy.deepcopy(self.before), resealed(rep.block, record), record)
        self.assertIn("recorded as version 3", problems[0])
        proto, workers, job = network([("honest",)] * 3, shards=1)
        before = copy.deepcopy(proto.chain)
        whole = proto.run_round()
        self.assertEqual((whole.record["version"], RECORD_VERSION), (2, 2))
        self.assertNotIn("teams", whole.record)
        record = dict(copy.deepcopy(whole.record), version=3)
        self.assertIn("version 3", replay_block(copy.deepcopy(before), resealed(whole.block, record), record)[0])
        self.assertEqual(replay_block(copy.deepcopy(before), whole.block, whole.record), [])

    def test_a_version_1_record_still_replays(self):
        fixture = json.loads(V1_FIXTURE.read_text())
        chain = Chain(**fixture["genesis"])
        for d in fixture["blocks"]:
            block = Block.from_dict(d)
            if block.summary is not None:
                self.assertEqual(replay_block(chain, block, fixture["records"][str(block.height)]), [])
            chain.append(block)


if __name__ == "__main__":
    unittest.main()
