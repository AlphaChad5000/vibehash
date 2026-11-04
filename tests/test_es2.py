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


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class NetworkTests2(unittest.TestCase):
    def test_rounds_replay_cheats_a_child_and_a_restart(self):
        from vibehash import es_client
        from vibehash.client import NodeClient, WorkerClient
        from vibehash.coordinator import Coordinator, CoordinatorConfig
        from vibehash.ledger import Block
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
        for p in (mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(root / "home")}),  # where lines are saved
                  mock.patch.object(es_client, "SAVE_EVERY", 1)):
            p.start()
            self.addCleanup(p.stop)
        node_keys, req = KeyPair.generate(b"es2 node"), KeyPair.generate(b"es2 requester")
        workers = [KeyPair.generate(f"es2 worker {i}".encode()) for i in range(4)]
        genesis = {"chain_id": "vibehash-es2-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}, "es": True}
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

        # The workers' change (float32, rounds applied as their blocks came) against the canonical float64 one.
        rt = es_client.runtime2(spec, "cpu")
        line = clients[0].es.v2._line(rt, "evo2-a", spec)  # (followed to the tip: its last round too)
        self.assertEqual(line.params_hash, es.state_hash(state))
        from vibehash.es_fast import stream_layout
        layout = stream_layout(rt.linears)
        exact = es2.canonical_weights(layout, {n: rt.base[n].numpy() for n, _, _ in layout}, history)
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
                                      parents=[(0.5, history, [])])
        worst = max(float(np.abs((rt.base[n] + cline.change.d[n]).numpy() - exact[n]).max()) for n, _, _ in layout)
        self.assertLess(worst, 1e-5, f"the child's change is {worst} from the canonical one")

        # A worker restarted carries on from the line it saved to disk, not from the start.
        saved = es_client._load_line(next(iter(es_client._chains.values())), "evo2-b", child)
        self.assertIsNotNone(saved, "the child's line wasn't saved")
        self.assertEqual(saved.params_hash, cline.params_hash)
        for n, _, _ in layout:
            np.testing.assert_array_equal(saved.change.d[n].numpy(), cline.change.d[n].numpy())

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
