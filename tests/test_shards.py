"""Sharded units (vibehash/shards.py, SHARDING.md): a model split across workers trains exactly as it does on one
computer, and each cell, one shard by one segment, can be checked holding only that shard."""
import unittest

import numpy as np

from vibehash import shards as sh
from vibehash.model import Dataset, ModelSpec, train_segment_numpy

SEED, STEPS, EVERY, LR, BATCH = 7, 12, 3, 0.1, 16


def setup(precision="float64", widths=(24, 20, 16, 12)):
    spec = ModelSpec(n_in=6, n_hidden=widths, n_out=3, precision=precision)
    data = Dataset.synthetic(spec, seed=1, n_train=256, n_val=64)
    return spec, data, spec.init_params(5)


class SplitTests(unittest.TestCase):
    def test_shards_cover_every_layer_once_and_cut_the_weights_into_slices(self):
        spec, _, theta = setup()
        for n in range(1, len(spec.widths)):
            shards = sh.split(spec, n)
            self.assertEqual(len(shards), n)
            self.assertEqual([s.lo for s in shards[1:]], [s.hi for s in shards[:-1]])
            self.assertTrue(shards[0].first and shards[-1].last)
            spans = [s.span(spec) for s in shards]
            self.assertEqual(spans[0].start, 0)
            self.assertEqual(spans[-1].stop, spec.n_params)
            self.assertEqual([p.start for p in spans[1:]], [p.stop for p in spans[:-1]])
            np.testing.assert_array_equal(sh.assemble(spec, shards, [theta[p] for p in spans]), theta)
        with self.assertRaises(ValueError):
            sh.split(spec, len(spec.widths))

    def test_shards_hold_about_the_same_number_of_weights(self):
        spec = ModelSpec(n_in=64, n_hidden=(64,) * 7, n_out=64)
        sizes = [s.span(spec).stop - s.span(spec).start for s in sh.split(spec, 4)]
        self.assertEqual(sizes, [2 * (64 * 64 + 64)] * 4)


class PipelineTests(unittest.TestCase):
    def test_a_sharded_unit_trains_to_exactly_the_whole_models_weights(self):
        """The split changes who computes what, not what's computed: the same operations in the same order, so the
        bits match and sharded units can be averaged and replayed like whole ones."""
        for precision in ("float64", "float32"):
            spec, data, theta = setup(precision)
            whole = train_segment_numpy(spec, data, theta, SEED, 0, STEPS, LR, BATCH)
            for n in range(1, len(spec.widths)):
                shards = sh.split(spec, n)
                traces = sh.train_sharded(spec, shards, data, theta, SEED, STEPS, EVERY, LR, BATCH)
                final = sh.assemble(spec, shards, [t.checkpoints[-1] for t in traces])
                np.testing.assert_array_equal(final, whole, err_msg=f"{precision}, {n} shards")
                self.assertEqual([len(t.checkpoints) for t in traces], [STEPS // EVERY + 1] * n)

    def test_neighbours_record_the_same_tensors_where_they_meet(self):
        spec, data, theta = setup()
        traces = sh.train_sharded(spec, sh.split(spec, 3), data, theta, SEED, STEPS, EVERY, LR, BATCH)
        for lower, upper in zip(traces, traces[1:]):
            for a, b in zip(lower.tensors, upper.tensors):
                self.assertIs(a["h_out"], b["h_in"])
                self.assertIs(a["g_out"], b["g_in"])
        self.assertTrue(all(t["h_in"] is None for t in traces[0].tensors))  # it reads the dataset
        self.assertTrue(all(t["g_out"] is None for t in traces[-1].tensors))  # it reads the labels


class CellTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.data, self.theta = setup("float32")
        self.shards = sh.split(self.spec, 3)

    def run_unit(self, cheats=None):
        return sh.train_sharded(self.spec, self.shards, self.data, self.theta, SEED, STEPS, EVERY, LR, BATCH, cheats)

    def check(self, trace, seg, root=None, cell=None):
        return sh.check_cell(self.spec, trace.shard, self.data, root or trace.root(), seg,
                             cell or sh.open_cell(trace, seg, EVERY), SEED, STEPS, EVERY, LR, BATCH, 1e-4)

    def edges(self, traces, steps=range(STEPS)):
        return [sh.edges_agree(lo.root(), sh.open_edges(lo, steps), up.root(), sh.open_edges(up, steps),
                               STEPS // EVERY + 1, steps) for lo, up in zip(traces, traces[1:])]

    def failing(self, traces) -> dict[int, list[int]]:
        return {t.shard.index: [s for s in range(STEPS // EVERY) if self.check(t, s)] for t in traces}

    def test_every_honest_cell_checks_out(self):
        traces = self.run_unit()
        self.assertEqual(self.failing(traces), {0: [], 1: [], 2: []})
        self.assertEqual(self.edges(traces), [None, None])

    def test_a_checker_needs_only_its_shards_weights(self):
        traces = self.run_unit()
        for t in traces:
            cell = sh.open_cell(t, 1, EVERY)
            span = t.shard.span(self.spec)
            self.assertEqual(cell["before"].size, span.stop - span.start)
            self.assertLess(cell["before"].size, self.spec.n_params)

    def test_each_cheat_fails_only_the_cheats_own_cells(self):
        """A shard that consumed a cheat's garbage computed honestly on what it was sent, so its cells pass; the
        cheat's fail, whichever way it cheated."""
        for k, cheat in [(1, "noisy_output"), (2, "zero_gradient"), (0, "frozen"), (1, "frozen")]:
            traces = self.run_unit({k: cheat})
            failing = self.failing(traces)
            self.assertEqual(failing[k], list(range(STEPS // EVERY)), cheat)
            self.assertEqual({i: f for i, f in failing.items() if i != k}, {i: [] for i in failing if i != k}, cheat)
            self.assertEqual(self.edges(traces), [None, None], cheat)  # it lied in what it computed, not what it sent

    def test_a_worker_that_commits_to_something_other_than_it_sent_is_caught_at_the_edge(self):
        traces = self.run_unit()
        traces[1].tensors[4] = dict(traces[1].tensors[4], h_out=traces[1].tensors[4]["h_out"] * 1.01)
        self.assertEqual(self.edges(traces)[0], None)
        self.assertIn("activations sent on", self.edges(traces)[1])

    def test_an_opening_must_be_what_was_committed(self):
        traces = self.run_unit()
        t = traces[1]
        cell = sh.open_cell(t, 2, EVERY)
        forged = dict(cell, tensors=[dict(cell["tensors"][0], h_in=cell["tensors"][0]["h_in"] + 1e-3), *cell["tensors"][1:]])
        self.assertIn("committed", self.check(t, 2, cell=forged))
        # A proof of the right value at another position doesn't pass either.
        moved = dict(cell, proofs=cell["proofs"] | {2: sh.open_cell(t, 1, EVERY)["proofs"][1]})
        moved["before"] = t.checkpoints[1]
        self.assertIn("committed", self.check(t, 2, cell=moved))

    def test_one_shard_is_a_whole_model_and_checks_as_one(self):
        spec, data, theta = self.spec, self.data, self.theta
        (only,) = sh.split(spec, 1)
        (trace,) = sh.train_sharded(spec, [only], data, theta, SEED, STEPS, EVERY, LR, BATCH)
        self.assertEqual([self.check(trace, s) for s in range(STEPS // EVERY)], [None] * (STEPS // EVERY))


if __name__ == "__main__":
    unittest.main()
