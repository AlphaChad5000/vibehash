"""Evo v2, ES rule 2 (vibehash/es2.py, es2_rounds.py, the coordinator's and es_client's rule-2 paths): the spec routed
beside rule 1's, the deterministic cosine, and a local network on the CPU with the tiny random Qwen2 of test_es: rounds
whose state is a hash chain the node keeps without weights, members scored on replay targets once there are some, the
workers' weight change against the canonical one, cheats caught, a child job starting from its parents' average, and a
node restart rebuilding the job's state from the chain.
"""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import es, es2, es2_rounds
from vibehash.crypto import KeyPair
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

from tests.test_es import HAVE_TORCH, fake_reward, spec_for, tiny_model

VIBE = SYN_PER_VIBE


def tiny_spec2(folder: Path, task: str = "countdown", start=(), **kw) -> es2.Spec2:
    from vibehash import es_local
    found = es_local.find(str(folder))
    es_local.register(found)
    s = dict(pairs=4, pairs_per_unit=1, problems=3, max_tokens=6, sigma=0.05, lr=0.5, rounds=6, replay=4, check=2,
             margin=1e-3, lp_tol=0.02)
    return es_local.spec2_for(found, task, start, **(s | kw))


def job_of(spec, job_id: str, requester: str, budget: int) -> TrainingJob:
    from vibehash import es_local
    return es_local.job2(spec, job_id, requester, budget)


class SpecTests(unittest.TestCase):
    def test_rule_two_is_routed_beside_rule_one_and_its_cosine_is_pinned(self):
        rule1_only = ("rule", "pairs", "pairs_per_unit", "problems", "max_tokens", "alpha", "reward", "level",
                      "retarget", "optimizer", "betas", "eps")
        spec = es2.Spec2(**{k: v for k, v in spec_for().__dict__.items() if k not in rule1_only},
                         pairs=4, pairs_per_unit=2, problems=4, max_tokens=8, start=(("evo2-a", 0.5, "ab" * 32),))
        self.assertEqual(spec.problems_list(), [])
        d = {"id": "j2", "fee_per_round": VIBE, **spec.to_json(), "init_seed": 0, "units_per_round": spec.units,
             "memory_mb": 1024}
        job = TrainingJob.from_spec_json(json.dumps(d), "req", 10 * VIBE)
        self.assertIsInstance(job.spec, es2.Spec2)
        self.assertTrue(job.es)
        self.assertEqual(job.spec, spec)
        self.assertEqual(TrainingJob.from_spec_json(job.spec_json(), "req", 10 * VIBE).spec, spec)
        one = {"id": "j1", "fee_per_round": VIBE, **spec_for().to_json(), "init_seed": 0, "units_per_round": 2,
               "memory_mb": 1024}
        self.assertNotIsInstance(TrainingJob.from_spec_json(json.dumps(one), "req", VIBE).spec, es2.Spec2)
        self.assertEqual(es2.lr_at(spec, 0), spec.lr)
        self.assertEqual(es2.lr_at(spec, spec.rounds), spec.lr * spec.lr_floor)
        self.assertAlmostEqual(es2.lr_at(spec, spec.rounds // 2), spec.lr * (1 + spec.lr_floor) / 2, places=15)
        # The state: the start hash in 16 exact words, and a round moving it on over its entry.
        state = spec.init_params()
        self.assertEqual(es2.hash_of(state), es2.start_hash(spec))
        ent = es2.entry(7, [[1, 0.5], [0, -0.25]], 1e-3, [[2, [5, 6]]])
        after = es2.step_state(state, ent, 0.25)
        self.assertEqual(es2.hash_of(after), es2.next_hash(es2.start_hash(spec), ent))
        self.assertEqual((es2.steps_of(after), int(es2.ctrl_of(after)[3])), (1, 1))


class TaskCycleTests(unittest.TestCase):
    """A job that never ends (a genesis starter job) takes its `tasks` in turn, `rounds` rounds each, the cosine starting
    again with each; a job without the field is what it always was, its spec byte for byte."""
    rule1_only = ("rule", "pairs", "pairs_per_unit", "problems", "max_tokens", "alpha", "reward", "level", "retarget",
                  "optimizer", "betas", "eps")

    def spec(self, **kw) -> es2.Spec2:
        base = {k: v for k, v in spec_for().__dict__.items() if k not in self.rule1_only}
        return es2.Spec2(**(base | dict(pairs=4, pairs_per_unit=2, problems=4, max_tokens=8, rounds=3) | kw))

    def test_without_tasks_nothing_changes(self):
        spec = self.spec()
        self.assertNotIn("tasks", spec.to_json()["es"])
        self.assertEqual([es2.task_at(spec, i) for i in (0, 3, 100)], [spec.task] * 3)
        self.assertEqual(es2.lr_at(spec, 7), spec.lr * spec.lr_floor, "flat after the cosine, as before")

    def test_tasks_take_turns_and_the_cosine_starts_again(self):
        spec = self.spec(task="countdown", tasks=("countdown", "arithmetic"))
        self.assertEqual(spec.problems_list(), [])
        self.assertEqual([es2.task_at(spec, i) for i in range(8)],
                         ["countdown"] * 3 + ["arithmetic"] * 3 + ["countdown"] * 2)
        self.assertEqual([es2.lr_at(spec, i) for i in (0, 3, 6)], [spec.lr] * 3)
        self.assertEqual(es2.lr_at(spec, 2), es2.lr_at(self.spec(), 2))
        d = {"id": "cycle", "fee_per_round": 0, **spec.to_json(), "init_seed": 0, "units_per_round": spec.units,
             "memory_mb": 1024}
        self.assertEqual(TrainingJob.from_spec_json(json.dumps(d), "system", 1).spec, spec)
        # A round's problems are its step's task's: the same seed, different problems a task later.
        a, b = es2_rounds.items(spec, 12345, 0), es2_rounds.items(spec, 12345, 3)
        self.assertEqual([p.messages for p in a], [p.messages for p in es2_rounds.items(self.spec(task="countdown"), 12345)])
        self.assertNotEqual([p.messages for p in a], [p.messages for p in b])

    def test_bad_task_lists(self):
        self.assertTrue(self.spec(task="countdown", tasks=("arithmetic", "countdown")).problems_list())  # task first
        self.assertTrue(self.spec(task="countdown", tasks=("countdown", "no-such-task")).problems_list())

    def test_replay_targets_carry_their_rounds_step(self):
        history = [{"seed": "%016x" % s, "targets": [[0, [5]], [1, [6]]]} for s in (7, 8, 9)]
        self.assertEqual(es2_rounds.targets_of(history, 3), [(8, 1, [6], 1), (9, 0, [5], 2), (9, 1, [6], 2)])


class AnchorTests(unittest.TestCase):
    """The anchor (Spec2.decay): before each round's step the change gives back `decay` of itself, so random steps level
    off near the base instead of adding up (es2's docstring). A job without it is what it always was."""
    rule1_only = TaskCycleTests.rule1_only

    def spec(self, **kw) -> es2.Spec2:
        base = {k: v for k, v in spec_for().__dict__.items() if k not in self.rule1_only}
        return es2.Spec2(**(base | dict(pairs=4, pairs_per_unit=2, problems=4, max_tokens=8, rounds=3) | kw))

    def test_the_field(self):
        self.assertNotIn("decay", self.spec().to_json()["es"], "no anchor: the spec byte for byte as before")
        spec = self.spec(decay=1 / 64)
        self.assertEqual(spec.problems_list(), [])
        self.assertEqual(spec.to_json()["es"]["decay"], 0.015625)
        d = {"id": "a", "fee_per_round": VIBE, **spec.to_json(), "init_seed": 0, "units_per_round": spec.units,
             "memory_mb": 1024}
        self.assertEqual(TrainingJob.from_spec_json(json.dumps(d), "req", VIBE).spec, spec)
        self.assertNotEqual(es2.Spec2.from_json(spec.to_json()), self.spec())
        self.assertEqual(self.spec(decay=0).decay, 0.0)
        for bad in (-0.1, 0.75, "0.1"):
            self.assertTrue(self.spec(decay=bad).problems_list(), bad)

    @unittest.skipUnless(HAVE_TORCH, "needs PyTorch")
    def test_the_arithmetic(self):
        """Two linears, three rounds (the middle one with no update, so no step and no decay): a worker's float32
        change and the canonical float64 weights agree, and both are (1 - d) x the first round's step + the last's,
        worked out from each round alone; a parent's anchor applies inside the parent, the child's on top."""
        import torch
        from vibehash.es_fast import stream_layout
        torch.manual_seed(0)
        linears = [("a", torch.nn.Linear(4, 3, bias=False)), ("b", torch.nn.Linear(2, 5, bias=False))]
        layout = stream_layout(linears)
        base = {n: m.weight.detach().float().numpy() for n, m in linears}
        e1 = es2.entry(11, [[0, 0.7], [1, -0.4]], 0.05, [])
        e0 = es2.entry(12, [], 0.05, [])
        e2 = es2.entry(13, [[1, 0.3]], 0.05, [])

        def step(e):
            w = es2.canonical_weights(layout, base, [e])
            return {n: w[n].astype(np.float64) - base[n] for n in base}
        s1, s2 = step(e1), step(e2)
        d = 0.25
        exact = es2.canonical_weights(layout, base, [e1, e0, e2], decay=d)
        worker = es2.rebuild(linears, [e1, e0, e2], device="cpu", decay=d)
        for n in base:
            np.testing.assert_allclose(exact[n] - base[n], (1 - d) * s1[n] + s2[n], atol=1e-6)
            np.testing.assert_allclose(worker.d[n].numpy(), exact[n] - base[n], atol=1e-6)
        # No anchor: the old order (base, then each round), the same as before this field.
        plain = es2.canonical_weights(layout, base, [e1, e0, e2])
        for n in base:
            np.testing.assert_allclose(plain[n] - base[n], s1[n] + s2[n], atol=1e-6)
        # A parent anchored at 1/2 ran e1 (a step from zero: nothing to give back); its child, weight 0.5 and anchor
        # 0.25, gives back a quarter of that before e2.
        child = es2.canonical_weights(layout, base, [e2], parents=[(0.5, [e1], [], 0.5)], decay=d)
        pc = es2.rebuild(linears, [e1], device="cpu", decay=0.5)
        cw = es2.rebuild(linears, [e2], parents=[(0.5, pc)], device="cpu", decay=d)
        for n in base:
            np.testing.assert_allclose(child[n] - base[n], 0.5 * (1 - d) * s1[n] + s2[n], atol=1e-6)
            np.testing.assert_allclose(cw.d[n].numpy(), child[n] - base[n], atol=1e-6)


class FormatTests(unittest.TestCase):
    """es_tasks.lenient: a right answer counts with or without its <answer> tags, so a job's reward (Spec2.lenient) and
    the progress checker's scores are for being right, not for writing the tags; and the curated mix."""

    def check(self, text, gold, kind):
        from vibehash import es_tasks as t
        make = {"letter": t._letter_check, "number": t._number_check, "words": lambda g: t._words_check([g])}[kind]
        item = t.Item([], make(gold), gold)
        return item.check(text), t.lenient(item).check(text)

    def test_right_answers_without_tags_count(self):
        for text, gold, kind in (("B", "B", "letter"), ("(C)", "C", "letter"), ("B. Sports", "B", "letter"),
                                 ("The answer is D.", "D", "letter"), ("Answer: (A)", "A", "letter"),
                                 ("So 3 + 4 = 7. That makes 7", "7", "number"), ("1,250 dollars", "1250", "number"),
                                 ("Yes, because it rains", "yes", "words"), ("no.", "no", "words"),
                                 ("Paris.", "Paris", "words")):
            self.assertEqual(self.check(text, gold, kind), (False, True), text)

    def test_wrong_or_unclear_answers_dont(self):
        for text, gold, kind in (("A dog would bark", "A", "letter"), ("I think it's B", "B", "letter"),
                                 ("C", "B", "letter"), ("The answer is C", "B", "letter"), ("7 or 8", "7", "number"),
                                 ("No, yes", "yes", "words"), ("", "B", "letter")):
            self.assertFalse(self.check(text, gold, kind)[1], text)

    def test_tags_win_when_there_are_any(self):
        self.assertEqual(self.check("<answer>C</answer>", "C", "letter"), (True, True))
        self.assertEqual(self.check("B <answer>C</answer>", "B", "letter"), (False, False))

    def test_the_spec_field_and_the_rounds(self):
        spec = AnchorTests().spec(lenient=True)
        self.assertEqual(spec.problems_list(), [])
        self.assertIs(spec.to_json()["es"]["lenient"], True)
        self.assertNotIn("lenient", AnchorTests().spec().to_json()["es"], "strict: the spec byte for byte as before")
        self.assertEqual(es2.Spec2.from_json(spec.to_json()), spec)
        self.assertTrue(AnchorTests().spec(lenient=1).problems_list())
        strict, loose = es2_rounds.items(AnchorTests().spec(), 5, 0), es2_rounds.items(spec, 5, 0)
        self.assertEqual([p.messages for p in strict], [p.messages for p in loose])
        gold = loose[0].gold
        self.assertEqual((strict[0].check(gold), loose[0].check(gold)), (False, True))

    def test_the_curated_mix(self):
        from vibehash import es_local, es_tasks
        self.assertTrue(set(es_tasks.CURATED) <= set(es_tasks.NAMES))
        self.assertFalse({"trivia_qa", "nq_open", "web_questions", "countdown", "arithmetic"} & set(es_tasks.CURATED))
        self.assertEqual(es_tasks.NAMES[:19][-1], "boolq", "the first nineteen keep their places")
        self.assertIs(es_local.V2_SETTINGS["lenient"], True)
        t = es_tasks
        piqa = t.HUB["piqa"][-1]({"goal": "Open a jar", "sol1": "Twist the lid", "sol2": "Eat the lid", "label": 0})
        self.assertEqual((piqa.gold, piqa.messages[0]["content"].splitlines()[2]), ("A", "A. Twist the lid"))
        siqa = t.HUB["social_iqa"][-1]({"context": "Ash cooked.", "question": "What next?", "answerA": "eat",
                                        "answerB": "sleep", "answerC": "run", "label": "2"})
        self.assertEqual(siqa.gold, "B")
        arc = t.HUB["arc_challenge"][-1]({"question": "Q?", "choices": {"label": ["A", "B", "C", "D"],
                                                                         "text": ["w", "x", "y", "z"]},
                                          "answerKey": "C"})
        self.assertEqual(arc.gold, "C")


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class StarterTemplateTests(unittest.TestCase):
    def test_an_evo_starter_job_from_the_template_goes_into_a_genesis(self):
        from vibehash import cli, economics as econ, es_tasks
        from vibehash.ledger import SYSTEM, Chain, InvalidBlock
        with tempfile.TemporaryDirectory() as d:
            folder, out = tiny_model(Path(d) / "tiny", seed=3), Path(d) / "starter.json"
            with mock.patch("sys.stdout"):
                cli.main(["job", "template", "--evo", str(folder), "--id", "evo-starter", "--rounds", "5",
                          "--out", str(out)])
            spec = json.loads(out.read_text())
        self.assertEqual((spec["id"], spec["fee_per_round"]), ("evo-starter", 0))
        self.assertEqual((spec["es"]["tasks"], spec["es"]["rounds"]), (list(es_tasks.NAMES), 5))
        g = {"chain_id": "evo-start", "timestamp": 0, "allocations": {}, "es": True, "starter_jobs": [spec],
             "economics": econ.NEW_SCHEDULES["mainnet"].to_dict()}
        s = Chain.from_genesis(g).state
        job = s.job("evo-starter")
        self.assertEqual((job.es, s.job_owner["evo-starter"], s.escrow["evo-starter"]), (True, SYSTEM, 1))
        self.assertEqual([es2.task_at(job.spec, i) for i in (0, 4, 5, 10)],
                         [es_tasks.NAMES[0], es_tasks.NAMES[0], es_tasks.NAMES[1], es_tasks.NAMES[2]])
        with self.assertRaisesRegex(InvalidBlock, "es"):  # an Evo starter job needs a genesis that takes Evo jobs
            Chain.from_genesis({**g, "es": False})
        # Rebuilds that look the spec up on chain (es2_rounds.rebuild_chain without one) find it in the genesis block.
        chain = Chain.from_genesis(g)
        self.assertEqual(es2_rounds.job_spec_in(chain.blocks[0], "evo-starter"), job.spec)
        state, history, rounds = es2_rounds.rebuild_chain(chain.blocks, lambda h: None, "evo-starter")
        self.assertEqual((rounds, history), (0, []))
        self.assertEqual(es.state_hash(state), es.state_hash(job.spec.init_params()))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class NetworkTests2(unittest.TestCase):
    def test_rounds_replay_cheats_a_child_and_a_restart(self):
        from vibehash import economics as econ
        from vibehash import es_client
        from vibehash.client import NodeClient, WorkerClient
        from vibehash.coordinator import Coordinator, CoordinatorConfig
        from vibehash.ledger import Block, Chain
        from vibehash.node import Node, PhaseTimes
        from vibehash.storage import Store
        from tests.test_network import run_until

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        folder = tiny_model(root / "tiny", seed=1)
        spec = tiny_spec2(folder)
        patch = mock.patch.object(es, "reward", fake_reward)  # Countdown's check: a score the tiny model's babble earns
        patch.start()
        self.addCleanup(patch.stop)
        # A job's compute phase lasts until every unit is in, up to its compute_s: a worker stopping as a phase of this
        # test ends can take units for the next round first, which a real job's 300 s would then wait out.
        for p in (mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(root / "home")}),  # where lines are saved
                  mock.patch.object(es_client, "SAVE_EVERY", 1), mock.patch("vibehash.es_local.V2_COMPUTE_S", 30)):
            p.start()
            self.addCleanup(p.stop)
        node_keys, req = KeyPair.generate(b"es2 node"), KeyPair.generate(b"es2 requester")
        workers = [KeyPair.generate(f"es2 worker {i}".encode()) for i in range(4)]
        # As a new testnet's: only rounds that pay for work mint (economics.PAID_ROUNDS), checked below.
        genesis = {"chain_id": "vibehash-es2-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True,
                   "economics": econ.NEW_SCHEDULES["testnet"].to_dict()}
        node = Node(root / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=20.0, reveal_s=3.0, verify_s=15.0, idle_s=0.3),
                    config=CoordinatorConfig(newcomer_units=4)).start()
        stopped = []
        self.addCleanup(lambda: stopped or node.stop())
        api = NodeClient(node.url)
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=root / f"cache{i}") for i, k in enumerate(workers)]
        clients[2].es.cheat = "liar"  # right/wrong scores its answers didn't earn: the node's re-score catches it
        clients[3].es.cheat = "forger"  # answers its twisted models didn't write: the checkers' teacher forcing does
        job = job_of(spec, "evo2-a", req.address, 6 * VIBE)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        stop = threading.Event()
        self.addCleanup(stop.set)
        run_until(node, clients, stop, 7, seconds=400)  # the job's block and six rounds
        st = node.chain.state
        self.assertGreaterEqual(node.chain.tip.height, 7, "the network didn't make its rounds in time")

        blocks = [Block.from_dict(d) for d in api.get(f"/blocks?from=0&to={node.chain.tip.height}")]
        records = {b.height: api.get(f"/round/{b.height}/record") for b in blocks if b.summary is not None}
        mine = [b for b in blocks if b.summary is not None and b.summary.job_id == "evo2-a"]
        state, history, rounds = es2_rounds.rebuild_chain(blocks, records.__getitem__, "evo2-a")
        self.assertEqual(rounds, len(mine))
        self.assertEqual(es.state_hash(state), mine[-1].summary.params_hash)
        np.testing.assert_array_equal(state, node.coord.start_params("evo2-a"))
        self.assertGreaterEqual(len(history), 3, "too few rounds learned anything")
        used = [records[b.height]["es"].get("used") for b in mine]
        self.assertIn("replay", used, f"no round scored members on replay targets: {used}")
        self.assertTrue(any(records[b.height]["es"].get("targets") for b in mine))
        from vibehash.ledger import canonical
        sizes = [len(canonical(es2_rounds.entry_of(records[b.height]["es"]))) for b in mine]
        self.assertLess(max(sizes), 4000)  # what the weights are a function of: a few hundred bytes a round here

        liar, forger = workers[2].address, workers[3].address
        self.assertLess(st.stake(liar), 10 * VIBE, "the liar wasn't slashed")
        self.assertTrue(all(forger not in b.summary.accepted for b in mine), "forged answers were accepted")
        self.assertTrue(any(w.address in b.summary.accepted for b in mine for w in workers[:2]))
        # An Evo round pays for work when a claim was approved and nobody rejected it: those rounds minted 50 VIBE each,
        # and no other did, as the blocks applied again from the genesis say.
        again = Chain.from_genesis(genesis)
        for b in blocks[1:]:
            again.append(b)
        worked = [b for b in mine if b.summary.pays_for_work]
        self.assertTrue(worked)
        self.assertEqual((again.state.paid_rounds, again.state.minted), (len(worked), 50 * VIBE * len(worked)))
        for b in mine:
            self.assertEqual(any(t.memo == "work" for t in b.txs), b.summary.pays_for_work, b.height)
            if not b.summary.pays_for_work:
                self.assertFalse([t for t in b.txs if t.kind == "reward"], b.height)

        # The workers' change (float32, rounds applied as their blocks came) against the canonical float64 one.
        rt = es_client.runtime2(spec, "cpu")
        line = clients[0].es.v2._line(rt, "evo2-a", spec)  # (followed to the tip: its last round too)
        self.assertEqual(line.params_hash, es.state_hash(state))
        from vibehash.es_fast import stream_layout
        layout = stream_layout(rt.linears)
        self.assertEqual(spec.decay, 1 / 64, "the anchor is on by default")
        exact = es2.canonical_weights(layout, {n: rt.base[n].numpy() for n, _, _ in layout}, history,
                                      decay=spec.decay)
        worst = max(float(np.abs((rt.base[n] + line.change.d[n]).numpy() - exact[n]).max()) for n, _, _ in layout)
        moved = max(float(np.abs(line.change.d[n].numpy()).max()) for n, _, _ in layout)
        self.assertGreater(moved, 1e-4, "the weights never moved")
        self.assertLess(worst, 1e-5, f"the workers' change is {worst} from the canonical one")

        # A child starting from its parent's state, weight 1/2: its start is half the parent's change.
        child = tiny_spec2(folder, "arithmetic", start=[["evo2-a", 0.5, es2.hash_of(state)]])
        job2 = job_of(child, "evo2-b", req.address, 2 * VIBE)
        api.send_tx(req, "post_job", "system", job2.budget, job2.spec_json())
        stop2 = threading.Event()
        self.addCleanup(stop2.set)
        top = node.chain.tip.height
        run_until(node, clients[:2], stop2, top + 3, seconds=300)
        self.assertTrue(any(b.summary is not None and b.summary.job_id == "evo2-b" for b in node.chain.blocks[top:]),
                        "the child didn't run")
        cline = clients[0].es.v2._line(rt, "evo2-b", child)
        self.assertEqual(cline.params_hash, es.state_hash(node.coord.start_params("evo2-b")))
        exact = es2.canonical_weights(layout, {n: rt.base[n].numpy() for n, _, _ in layout}, cline.history,
                                      parents=[(0.5, history, [], spec.decay)], decay=child.decay)
        worst = max(float(np.abs((rt.base[n] + cline.change.d[n]).numpy() - exact[n]).max()) for n, _, _ in layout)
        self.assertLess(worst, 1e-5, f"the child's change is {worst} from the canonical one")

        # A worker restarted carries on from the line it saved to disk, not from the start.
        saved = es_client._load_line(next(iter(es_client._chains.values())), "evo2-b", child)
        self.assertIsNotNone(saved, "the child's line wasn't saved")
        self.assertEqual(saved.params_hash, cline.params_hash)
        for n, _, _ in layout:
            np.testing.assert_array_equal(saved.change.d[n].numpy(), cline.change.d[n].numpy())

        # A process with neither line in memory nor on disk replays the child's rounds and its parent's up to the child's
        # start (the parent ran on after it: jobs take turns), and says how far along it is: the dashboards'
        # rebuilding() counts the rounds replayed up to the total it counted first, and the log says so.
        cache = next(iter(es_client._chains.values()))
        cache.lines.clear()
        cache.starts.clear()
        for f in es_client._saved_dir().glob("*.pt"):
            f.unlink()
        seen, step = [], es_client.rebuild.step

        def spy(job_id):
            step(job_id)
            seen.append(es_client.rebuilding())
        with mock.patch.object(es_client.rebuild, "step", spy), self.assertLogs("vibehash.worker", "INFO") as logs:
            again = clients[1].es.v2._line(rt, "evo2-b", child)
        self.assertEqual(again.params_hash, cline.params_hash)
        n, total = len(seen), seen[0][1]
        self.assertGreaterEqual(n, len(history) + 1)  # the parent's rounds that learned something, and the child's
        self.assertEqual([s[:2] for s in seen], [(min(k, total), total) for k in range(1, n + 1)])
        self.assertEqual({s[2] for s in seen}, {"evo2-a", "evo2-b"})
        # The total counts the parent's rounds up to the child's posting; it pins the parent's state as the poster saw
        # it, a round or so before when the parent runs on meanwhile (here, only if a round sealed in those seconds).
        pinned = max(b.height for b in node.chain.blocks if b.summary is not None and b.summary.job_id == "evo2-a"
                     and b.summary.params_hash == es.state_hash(state))
        posted = es_client._posted_at(cache, "evo2-b")
        late = sum(1 for b in node.chain.blocks if pinned < b.height < posted and b.summary is not None
                   and b.summary.job_id == "evo2-a" and b.summary.accepted_units)
        self.assertEqual(total, n + late)
        self.assertIsNone(es_client.rebuilding())
        self.assertTrue(any(f"{total} rounds to replay" in m for m in logs.output), logs.output)
        self.assertTrue(any(f"rebuilt from the chain: {n} rounds" in m for m in logs.output), logs.output)

        # A restart: the node works the jobs' states out from the chain, even with its saved copies gone.
        want = {j: node.coord.start_params(j).copy() for j in ("evo2-a", "evo2-b")}
        node.stop()
        stopped.append(True)
        store = Store(root / "node")
        self.addCleanup(store.close)
        chain = store.open_chain(genesis)
        store.save_params("evo2-a", 1, np.zeros(es2.WORDS + es.CTRL, np.float32))  # a wrong saved copy
        c = Coordinator(chain, store=store)
        for j, s in want.items():
            np.testing.assert_array_equal(c.start_params(j), s)
            self.assertEqual(es.state_hash(c.start_params(j)), c.expected_start_hash(c.job(j)).hex())


def _cuda() -> bool:
    if not HAVE_TORCH:
        return False
    import torch
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda(), "needs a CUDA GPU")
class EngineCheckTests(unittest.TestCase):
    def test_a_member_checked_alone_rounds_as_it_did_among_thousands(self):
        """es_fast's engine in bf16 on CUDA: a worker's members answer in one batch of 2,048 rows (eight passes of
        es_fast.ROWS), and a checker teacher-forcing one member's answers alone (one pass, mostly padding) finds every
        token exactly its greedy choice; a member's answers claimed for its twin fail wherever the two differ. The
        model has Qwen2.5-0.5B's widths, whose products cuBLAS rounds differently by their rows (the old check, on
        transformers' path, found gaps here)."""
        from vibehash import es_fast
        from vibehash.es_client import prompt_rows, round_factors, runtime2

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        folder = tiny_model(Path(tmp.name) / "tiny", seed=2, hidden_size=896, intermediate_size=4864,
                            num_attention_heads=14, num_key_value_heads=2)
        spec = tiny_spec2(folder, pairs=64, pairs_per_unit=8, problems=16, max_tokens=12)
        rt = runtime2(spec, "cuda")
        seed = 12345
        rows = prompt_rows(rt, es2_rounds.items(spec, seed))
        fac = round_factors(rt, spec, seed)
        members = [(j, sign) for u in range(spec.units) for j, _, sign in es.unit_members(spec, seed, u)]
        self.assertEqual(len(members) * len(rows), 8 * es_fast.ROWS)
        got = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank)

        def gaps(k, answers):
            return es_fast.gaps_items(rt, rt.linears, fac, [members[k]], list(zip(rows, answers)), spec.sigma,
                                      spec.rank, max_tokens=spec.max_tokens)[0].tolist()
        for k in (0, 1, 70, len(members) - 1):
            self.assertEqual(gaps(k, got[k]), [0.0] * len(rows), f"member {k}")
        differ = [q for q in range(len(rows)) if got[0][q] != got[1][q]]
        self.assertTrue(differ, "the twins answered alike everywhere")
        claimed = gaps(1, got[0])
        self.assertTrue(all(claimed[q] > 0 for q in differ), [claimed[q] for q in differ])


if __name__ == "__main__":
    unittest.main()
