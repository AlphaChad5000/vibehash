"""A rule-2 Evo job's loop_stop (es2.Spec2.loop_stop, es2.loop_end): an answer ends where greedy decoding loops.
The rule on token lists, the spec field (left out when off, so earlier jobs' specs and hashes are the same), the
check an answer's end must pass (es2.ends_right), and Qwen3.5's engine stopping a looping row on the device at the
token the rule names, on test_es_hybrid's tiny random model."""
import random
import tempfile
import unittest
from pathlib import Path

from vibehash import es2

from tests.test_es import HAVE_TORCH


def brute_end(ids, period, run):
    """loop_end by its definition: the first t with `run` tokens ending at t each equal to the token p back."""
    for t in range(len(ids)):
        for p in range(1, period + 1):
            if t - run + 1 - p >= 0 and all(ids[t - i] == ids[t - i - p] for i in range(run)):
                return t
    return None


class Spec:
    def __init__(self, max_tokens, loop_stop=()):
        self.max_tokens, self.loop_stop = max_tokens, loop_stop


class LoopRuleTests(unittest.TestCase):
    def test_loop_end_is_its_definition(self):
        rng = random.Random(5)
        for _ in range(3000):
            n = rng.randint(0, 40)
            alphabet = rng.choice([2, 3, 5, 50])
            ids = [rng.randrange(alphabet) for _ in range(n)]
            if rng.random() < 0.5 and n > 4:  # a loop of a random period from a random point
                p, at = rng.randint(1, 6), rng.randint(0, n - 1)
                for t in range(at + p, n):
                    ids[t] = ids[t - p]
            period, run = rng.randint(1, 8), rng.randint(1, 10)
            self.assertEqual(es2.loop_end(ids, period, run), brute_end(ids, period, run), (ids, period, run))

    def test_a_latex_loop_ends_and_prose_does_not(self):
        loop = [7, 1, 2, 3, 4, 5, 6] * 20  # "\n\usepackage{amssymb}": a period of 7 tokens
        t = es2.loop_end(loop, 32, 48)
        self.assertEqual(t, 7 + 48 - 1, "48 tokens in a row each the one 7 before, after the first 7")
        self.assertEqual(es2.cut_loop(loop, (32, 48)), loop[:t + 1])
        prose = list(range(200))
        self.assertIsNone(es2.loop_end(prose, 32, 48))
        self.assertEqual(es2.cut_loop(prose, (32, 48)), prose)
        self.assertEqual(es2.cut_loop(loop, ()), loop)

    def test_an_answer_ends_right(self):
        eos = [99]
        off, on = Spec(10), Spec(10, (2, 3))
        self.assertTrue(es2.ends_right([1, 2, 99], off, eos))
        self.assertTrue(es2.ends_right(list(range(10)), off, eos))  # at max_tokens
        self.assertFalse(es2.ends_right([1, 2, 3], off, eos))  # neither
        self.assertFalse(es2.ends_right([], off, eos))
        self.assertFalse(es2.ends_right(list(range(11)), off, eos))  # past max_tokens
        self.assertFalse(es2.ends_right([5, 5, 5, 5], off, eos), "a loop isn't an end without loop_stop")
        self.assertTrue(es2.ends_right([5, 5, 5, 5], on, eos), "it loops at its last token: [5] three times over")
        self.assertFalse(es2.ends_right([5, 5, 5, 5, 99], on, eos), "it ran on past where it loops")
        self.assertFalse(es2.ends_right([5, 5, 5, 5, 5], on, eos))
        self.assertTrue(es2.ends_right([1, 2, 1, 2, 1], on, eos), "period 2, three in a row")
        self.assertTrue(es2.ends_right([1, 2, 3, 99], on, eos), "an answer that never loops ends as before")

    def test_without_loop_stop_the_end_is_as_before(self):
        """ends_right with loop_stop off is the check every earlier job ran (es_client.check_many before it)."""
        rng = random.Random(9)
        eos = [3]
        for _ in range(2000):
            spec = Spec(rng.randint(1, 8))
            a = [rng.randrange(5) for _ in range(rng.randint(0, 10))]
            old = not (len(a) > spec.max_tokens or (len(a) < spec.max_tokens and (not a or a[-1] not in eos)))
            self.assertEqual(es2.ends_right(a, spec, eos), old, a)


def spec_json(**es):
    return {"base": "hf:org/model@" + "0" * 40, "manifest": "a" * 64, "model": {"family": "qwen3_5_text"},
            "es": {"rule": 2, "tensors": ["model.layers.*_proj.weight"], "n": 100, "tensors_sha": "b" * 64,
                   "task": "arithmetic", "pairs": 8, "pairs_per_unit": 2, "problems": 4, "max_tokens": 64,
                   "sigma": 0.001, "lr": 0.001, "lr_floor": 0.25, "rounds": 12, "replay": 4, "start": [],
                   "check": 2, "margin": 0.5, "lp_tol": 0.05, "rank": 1, **es}}


class LoopSpecTests(unittest.TestCase):
    def test_off_is_left_out_of_the_spec(self):
        s = es2.Spec2.from_json(spec_json())
        self.assertEqual(s.loop_stop, ())
        self.assertNotIn("loop_stop", s.to_json()["es"])
        self.assertEqual(s.to_json(), es2.Spec2.from_json(s.to_json()).to_json())

    def test_on_round_trips_and_is_checked(self):
        s = es2.Spec2.from_json(spec_json(loop_stop=[32, 48]))
        self.assertEqual(s.loop_stop, (32, 48))
        self.assertEqual(s.to_json()["es"]["loop_stop"], [32, 48])
        self.assertEqual(es2.Spec2.from_json(s.to_json()), s)
        self.assertFalse([p for p in s.problems_list() if "loop_stop" in p])
        for bad in ([0, 48], [65, 48], [32, 0], [32, 65], [32], [32, 48, 1], [32.0, 48]):
            got = es2.Spec2.from_json(spec_json(loop_stop=bad)).problems_list()
            self.assertTrue([p for p in got if "loop_stop" in p], bad)
        with self.assertRaises(TypeError):
            es2.Spec2.from_json(spec_json(loop_stop="32,48"))


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class LoopEngineTests(unittest.TestCase):
    """Qwen3.5's engine (es_hybrid, eager on the CPU: the same steps a CUDA graph records) with loop_stop: every
    answer is the answer without it, cut where es2.loop_end says, and ends as a checker requires."""

    def test_the_engine_stops_a_looping_row_where_the_rule_does(self):
        from vibehash import es2_rounds, es_fast, es_local
        from vibehash.es_client import prompt_rows, round_factors, runtime2

        from tests.test_es_hybrid import tiny_hybrid

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        found = es_local.find(str(tiny_hybrid(Path(tmp.name) / "tiny35", seed=3)))
        es_local.register(found)
        loop = (3, 4)  # small, so a random tiny model's answers loop within 40 tokens
        spec = es_local.spec2_for(found, "arithmetic", pairs=4, pairs_per_unit=2, problems=5, max_tokens=40,
                                  sigma=0.05, check=2, margin=1e-3, lp_tol=0.02, loop_stop=loop)
        self.assertEqual(spec.loop_stop, loop)
        rt = runtime2(spec, "cpu")
        rows = prompt_rows(rt, es2_rounds.items(spec, 777))
        fac = round_factors(rt, spec, 777)
        members = [(j, s) for j in range(spec.pairs) for s in (1, -1)]
        whole = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma)
        cut = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, loop=loop)
        stopped = 0
        for k in range(len(members)):
            for q in range(len(rows)):
                self.assertEqual(cut[k][q], es2.cut_loop(whole[k][q], loop), f"member {k}, prompt {q}")
                self.assertTrue(es2.ends_right(cut[k][q], spec, rt.eos), (k, q, cut[k][q]))
                stopped += len(cut[k][q]) < len(whole[k][q])
        self.assertGreater(stopped, 0, "no answer looped: the test shows nothing")
        # A row stopped for looping leaves stale tokens behind it; the next call's answers are still exact.
        again = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, loop=loop)
        self.assertEqual(again, cut)
        # And a checker teacher-forcing a loop-stopped answer finds it greedy throughout.
        gaps = es_fast.gaps_members(rt, rt.linears, fac, members, rows, cut, spec.sigma, max_tokens=spec.max_tokens)
        self.assertEqual(float(gaps.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
