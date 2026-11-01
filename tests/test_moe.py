"""Mixture-of-experts (vibehash/moe.py, SHARDING.md §7): top-k routing flips under rounding noise, so workers commit to
their routing and checkers replay with it; and each expert is a cell a checker can check holding that expert alone."""
import dataclasses
import unittest

import numpy as np

from vibehash import moe
from vibehash.model import relative_distance

TOL = 1e-4  # a float32 job's tolerance (DESIGN §8)
TOKENS = 48


def setup(precision="float64", **kw):
    spec = moe.MoESpec(d_model=16, d_ff=24, n_experts=8, top_k=2, n_shared=1, precision=precision, **kw)
    rng = np.random.default_rng(4)
    x, g_out = (spec.round(rng.normal(0, 1, (TOKENS, spec.d_model))) for _ in range(2))
    return spec, spec.init_params(3), x, g_out


def claims(spec, params, x, g_out, routing=None):
    """An honest worker's routing, output and gradients, as check_block takes them."""
    out, saved = moe.forward(spec, params, x, routing)
    g_in, grads = moe.backward(spec, params, saved, g_out)
    return saved["routing"], out, {"x": g_in, **grads}


def trace_claims(spec, trace):
    """The output and gradients a trace commits to, as check_block takes them."""
    grads = {"x": trace.dense["g_in"], **moe.unpack(spec, trace.dense["grad"], trainable=True)}
    for e, ex in enumerate(trace.experts):
        grads |= moe.unpack(spec, ex["grad"], e, trainable=True)
    return trace.dense["h_out"], grads


def committed(scores, experts):
    """A routing to `experts`, with the gates honestly computed for them."""
    chosen = np.take_along_axis(scores, experts, axis=1)
    return moe.Routing(experts, chosen / chosen.sum(axis=1, keepdims=True))


def near_ties(spec, params, n, rng):
    """Tokens whose k-th and (k+1)-th best experts tie but for a nudge of 1e-12 in the logits, far below what float32
    resolves: which of the two an honest machine picks is down to how it rounds."""
    w, k, out = params["router"], spec.top_k, []
    while len(out) < n:
        x = rng.normal(0, 1, spec.d_model)
        order = np.argsort(-(x @ w))
        u = w[:, order[k - 1]] - w[:, order[k]]
        x += (1e-12 - x @ u) / (u @ u) * u  # now the k-th's logit is the (k+1)-th's plus 1e-12
        now = np.argsort(-(x @ w))
        if set(now[:k - 1]) == set(order[:k - 1]) and set(now[k - 1:k + 1]) == set(order[k - 1:k + 1]):
            out.append(x)  # the move kept them k-th and (k+1)-th
    return np.array(out)


class LayoutTests(unittest.TestCase):
    def test_each_expert_is_a_flat_vector_of_its_own(self):
        spec, params, _, _ = setup(lora_rank=2)
        self.assertEqual(sorted(params), sorted(n for p in (None, *range(spec.n_experts)) for n in spec.names(p)))
        self.assertEqual(spec.n_params, sum(a.size for a in params.values()))
        for part in (None, 0, 5):
            flat = moe.pack(spec, params, part)
            self.assertEqual(flat.shape, (spec.part_size(part),))
            back = moe.unpack(spec, flat, part)
            self.assertEqual(list(back), spec.names(part))
            for name, a in back.items():
                np.testing.assert_array_equal(a, params[name])
        self.assertTrue(all(n.startswith("expert.5.") for n in spec.names(5)))  # a shard can hold any subset of experts
        self.assertEqual(spec.names(5, trainable=True), [n for n in spec.names(5) if "lora" in n])
        self.assertLess(spec.active_params, spec.n_params / 2)


class GradientTests(unittest.TestCase):
    """Backward against central finite differences, in float64. The routing is held fixed: top-k has no gradient, and
    a nudge across a tie would measure a jump, not a slope."""

    def assert_matches_finite_differences(self, spec, params, x, g_out):
        routing, _, grads = claims(spec, params, x, g_out)

        def loss(p, xx):  # the gradient of this with respect to the block's output is g_out
            return float((moe.forward(spec, p, xx, routing)[0] * g_out).sum())

        rng, eps = np.random.default_rng(0), 1e-6
        for name, grad in grads.items():
            base = x if name == "x" else params[name]
            for flat in rng.choice(base.size, 5, replace=False):
                i = np.unravel_index(flat, base.shape)
                sides = []
                for sign in (1, -1):
                    nudged = base.copy()
                    nudged[i] += sign * eps
                    sides.append(loss(params, nudged) if name == "x" else loss(params | {name: nudged}, x))
                fd = (sides[0] - sides[1]) / (2 * eps)
                self.assertAlmostEqual(grad[i], fd, delta=1e-6 * max(1.0, abs(fd)), msg=f"{name}{i}")
        return grads

    def test_experts_router_shared_expert_and_input(self):
        spec, params, x, g_out = setup()
        grads = self.assert_matches_finite_differences(spec, params, x, g_out)
        self.assertEqual(set(grads), {"x", *params})  # every weight trains
        self.assertGreater(np.abs(grads["router"]).max(), 0)  # through the gates

    def test_lora_adapters_on_the_experts(self):
        spec, params, x, g_out = setup(lora_rank=2)
        rng = np.random.default_rng(1)  # B starts at zero, which would leave A's gradient zero: give it something
        params = {n: rng.normal(0, 0.3, a.shape) if n.endswith(".lora_b") else a for n, a in params.items()}
        grads = self.assert_matches_finite_differences(spec, params, x, g_out)
        self.assertEqual(set(grads), {"x", "router", *(n for n in params if "lora" in n)})  # the bases are frozen


class RoutingFlipTests(unittest.TestCase):
    """Why the routing has to be committed: at a near-tie, honest machines pick different experts, and then a replay
    that routes for itself is nowhere near the worker's output, while one that uses the worker's routing is."""

    def setUp(self):
        self.spec32 = moe.MoESpec(d_model=16, d_ff=24, n_experts=8, top_k=2, n_shared=1, precision="float32")
        self.spec64 = dataclasses.replace(self.spec32, precision="float64")
        self.params = self.spec32.init_params(3)  # float32 values, so both precisions use the same weights
        rng = np.random.default_rng(5)
        self.ties = 32
        self.x = np.vstack([near_ties(self.spec32, self.params, self.ties, rng), rng.normal(0, 1, (16, 16))])
        self.g_out = self.spec32.round(rng.normal(0, 1, self.x.shape))

    @staticmethod
    def flipped(a, b):
        return [t for t in range(len(a.experts)) if set(a.experts[t]) != set(b.experts[t])]

    def test_float32_and_float64_route_near_ties_differently(self):
        """A float32 worker (a GPU) against a float64 checker, standing in for any machine that rounds differently."""
        x = self.spec32.round(self.x)  # a float32 job's activations
        worker, out, grads = claims(self.spec32, self.params, x, self.g_out)
        naive, _ = moe.route(self.spec64, self.params, x)
        flipped = self.flipped(worker, naive)
        self.assertTrue(flipped)
        self.assertTrue(all(t < self.ties for t in flipped))  # only the near-ties
        # A checker that routes for itself finds the honest worker wrong, far beyond the tolerance...
        self.assertGreater(relative_distance(moe.forward(self.spec64, self.params, x)[0], out), 100 * TOL)
        # ...while replaying with the worker's committed routing agrees, and the routing passes its own check.
        self.assertIsNone(moe.check_block(self.spec64, self.params, x, worker, out, grads, TOL, self.g_out))
        self.assertIsNone(moe.check_routing(self.spec64, self.params, x, worker))
        # The margin is what lets it pass: without one, an honest worker would be rejected for a rounding error.
        self.assertIn("left out", moe.check_routing(self.spec64, self.params, x, worker, margin=0))

    def test_tiny_perturbations_flip_naive_routing_too(self):
        """Recomputing the layers below on other hardware moves a block's input by about a rounding error. That flips
        near-ties as well, which is why a checker starts from the committed input and the committed routing."""
        spec, x = self.spec64, self.x
        worker, out, grads = claims(spec, self.params, x, self.g_out)
        nudged = x * (1 + 1e-9 * np.random.default_rng(6).normal(size=x.shape))
        self.assertTrue(self.flipped(worker, moe.route(spec, self.params, nudged)[0]))
        self.assertGreater(relative_distance(moe.forward(spec, self.params, nudged)[0], out), 100 * TOL)
        self.assertIsNone(moe.check_block(spec, self.params, nudged, worker, out, grads, TOL, self.g_out))


class CheatTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.params, self.x, self.g_out = setup()
        self.honest, self.scores = moe.route(self.spec, self.params, self.x)

    def check(self, routing):
        return moe.check_routing(self.spec, self.params, self.x, routing)

    def test_the_honest_routing_passes(self):
        self.assertIsNone(self.check(self.honest))
        np.testing.assert_allclose(self.honest.gates.sum(axis=1), 1)

    def test_sending_every_token_to_the_same_cheap_experts_fails(self):
        self.assertIn("left out", self.check(committed(self.scores, np.tile([0, 1], (TOKENS, 1)))))

    def test_skipping_an_expert_fails(self):
        """Its tokens go to their next-best expert instead, so it never has to run."""
        busiest = np.bincount(self.honest.experts.ravel()).argmax()
        experts = self.honest.experts.copy()
        for t, j in zip(*np.nonzero(experts == busiest)):
            others = np.argsort(-self.scores[t], kind="stable")
            experts[t, j] = next(e for e in others if e not in experts[t])
        self.assertFalse(np.any(experts == busiest))
        self.assertIn("left out", self.check(committed(self.scores, experts)))

    def test_at_a_near_tie_either_expert_passes_but_the_best_cant_be_dropped(self):
        x = near_ties(self.spec, self.params, 8, np.random.default_rng(7))
        k, logits = self.spec.top_k, x @ self.params["router"]
        _, scores = moe.route(self.spec, self.params, x)
        order = np.argsort(-logits, axis=1, kind="stable")
        swapped = order[:, [*range(k - 1), k]]  # the (k+1)-th in place of the k-th: as good, to 1e-12
        self.assertIsNone(moe.check_routing(self.spec, self.params, x, committed(scores, swapped)))
        dropped = order[:, 1:k + 1]  # the best dropped for the (k+1)-th
        kth = np.sort(logits, axis=1)[:, -k]
        # Every chosen expert is still within the margin of the k-th best, so that rule alone would let this through.
        self.assertTrue(np.all(np.take_along_axis(logits, dropped, axis=1) >= kth[:, None] - moe.ROUTING_MARGIN))
        self.assertIn("left out", moe.check_routing(self.spec, self.params, x, committed(scores, dropped)))

    def test_malformed_routings_fail(self):
        e, g = self.honest.experts, self.honest.gates
        self.assertIn("2 experts", self.check(moe.Routing(e[:, :1], g[:, :1])))  # a token's second expert skipped
        self.assertIn("2 experts", self.check(moe.Routing(e[:-1], g[:-1])))  # a token left out altogether
        self.assertIn("twice", self.check(moe.Routing(np.repeat(e[:, :1], 2, axis=1), g)))
        self.assertIn("don't exist", self.check(moe.Routing(np.where(e == 3, 8, e), g)))
        self.assertIn("don't exist", self.check(moe.Routing(e.astype(float), g)))
        self.assertIn("2 experts", self.check(None))

    def test_gates_must_be_the_chosen_scores_normalised(self):
        e, g = self.honest.experts, self.honest.gates
        self.assertIn("gates", self.check(moe.Routing(e, np.full_like(g, 0.5))))
        self.assertIn("gates", self.check(moe.Routing(e, g[:, ::-1])))
        self.assertIn("gates", self.check(moe.Routing(e, np.where(g > 0.9, np.nan, g))))

    def test_the_right_routing_with_wrong_expert_outputs_fails_the_replay(self):
        out, grads = trace_claims(self.spec, moe.trace_block(self.spec, self.params, self.x, self.g_out))
        self.assertIsNone(moe.check_block(self.spec, self.params, self.x, self.honest, out, grads, TOL, self.g_out))
        for cheat in moe.CHEATS:
            trace = moe.trace_block(self.spec, self.params, self.x, self.g_out, cheats={3: cheat})
            self.assertIsNone(self.check(trace.routing), cheat)  # its routing is right...
            out, grads = trace_claims(self.spec, trace)
            self.assertIn("output", moe.check_block(self.spec, self.params, self.x, trace.routing, out, grads, TOL,
                                                    self.g_out), cheat)  # ...what it did with it isn't

    def test_honest_outputs_with_wrong_gradients_fail_the_replay(self):
        routing, out, grads = claims(self.spec, self.params, self.x, self.g_out)
        for name in ("x", "router", "expert.3.down", "shared.0.up"):
            bad = grads | {name: grads[name] * 1.01}
            self.assertIn("gradient", moe.check_block(self.spec, self.params, self.x, routing, out, bad, TOL, self.g_out))
        bad = {n: g for n, g in grads.items() if n != "expert.3.gate"}
        self.assertIn("wrong arrays", moe.check_block(self.spec, self.params, self.x, routing, out, bad, TOL, self.g_out))


class ExpertCellTests(unittest.TestCase):
    """Each expert is a cell, checked holding that expert alone, and the router and shared experts are another; a lie
    fails only the cell it's in, as with shards."""

    def setUp(self):
        self.spec, self.params, self.x, self.g_out = setup()

    def trace(self, **kw):
        return moe.trace_block(self.spec, self.params, self.x, self.g_out, **kw)

    @staticmethod
    def failing(spec, trace):
        """(experts whose cells fail, what's wrong with the router's cell)"""
        root = trace.root()
        experts = [e for e in range(spec.n_experts) if moe.check_expert(spec, root, e, moe.open_expert(trace, e), TOL)]
        return experts, moe.check_router(spec, root, moe.open_router(trace), TOL)

    def test_every_honest_cell_checks_out(self):
        for kw in ({}, {"lora_rank": 2}, {"precision": "float32"}):
            spec, params, x, g_out = setup(**kw)
            self.assertEqual(self.failing(spec, moe.trace_block(spec, params, x, g_out)), ([], None), kw)

    def test_faking_one_expert_fails_only_that_experts_cell(self):
        """Its outputs zeroed (it never ran) or noisy, and combined as if real, so the router's cell is consistent."""
        for cheat in moe.CHEATS:
            self.assertEqual(self.failing(self.spec, self.trace(cheats={3: cheat})), ([3], None), cheat)

    def test_a_bad_routing_fails_only_the_routers_cell(self):
        """The experts computed honestly on the tokens they were sent, however wrongly they were sent them."""
        cheap = committed(moe.route(self.spec, self.params, self.x)[1], np.tile([0, 1], (TOKENS, 1)))
        experts, router = self.failing(self.spec, self.trace(routing=cheap))
        self.assertEqual(experts, [])
        self.assertIn("left out", router)

    def test_a_faked_combination_fails_only_the_routers_cell(self):
        trace = self.trace()
        trace.dense["h_out"] = trace.dense["h_out"] * 1.01
        experts, router = self.failing(self.spec, trace)
        self.assertEqual(experts, [])
        self.assertIn("combined", router)

    def test_an_expert_must_be_sent_the_tokens_its_routing_gave_it(self):
        """Expert 2 run, honestly, on a token the routing didn't send it: its own cell holds up, the router's doesn't."""
        trace = self.trace()
        ex = trace.experts[2]
        x = ex["x"].copy()
        x[0] = self.x[np.flatnonzero(~np.any(trace.routing.experts == 2, axis=1))[0]]
        local = moe.LocalExperts(self.spec, self.params)
        y, saved = local.forward(2, x)
        dx, grads = local.backward(2, saved, ex["dy"])
        ex.update(x=x, y=y, dx=dx, grad=moe.pack(self.spec, grads, 2, trainable=True))
        experts, router = self.failing(self.spec, trace)
        self.assertEqual(experts, [])
        self.assertIn("expert 2 wasn't sent", router)

    def test_an_opening_must_be_what_was_committed(self):
        trace = self.trace()
        root, cell = trace.root(), moe.open_expert(trace, 2)
        self.assertIn("committed", moe.check_expert(self.spec, root, 2, dict(cell, y=cell["y"] * 1.01), TOL))
        # A real leaf's proof at another position doesn't pass either: expert 5's outputs offered as expert 2's.
        other, i = moe.open_expert(trace, 5), moe.leaf_index("y", 2)
        moved = dict(cell, y=other["y"], proofs=cell["proofs"] | {i: other["proofs"][moe.leaf_index("y", 5)]})
        self.assertIn("committed", moe.check_expert(self.spec, root, 2, moved, TOL))
        router = moe.open_router(trace)
        swapped = dict(router, routing=moe.Routing(trace.routing.experts[:, ::-1], trace.routing.gates[:, ::-1]))
        self.assertIn("committed", moe.check_router(self.spec, root, swapped, TOL))

    def test_the_blocks_edges_hash_as_a_shards_do(self):
        """So a neighbouring shard's commitment names the same tensors, and shards.edges_agree carries over."""
        from vibehash.shards import tensor_hash
        trace = self.trace()
        leaves = trace.leaves()
        for kind in ("h_in", "g_out", "h_out", "g_in"):
            self.assertEqual(leaves[moe.leaf_index(kind)], tensor_hash(trace.dense[kind]))


class CostTests(unittest.TestCase):
    def setUp(self):
        self.spec = moe.MoESpec(d_model=16, d_ff=24, n_experts=32, top_k=2, n_shared=1)
        self.params = self.spec.init_params(3)
        rng = np.random.default_rng(4)
        x, g_out = (rng.normal(0, 1, (128, 16)) for _ in range(2))
        self.trace = moe.trace_block(self.spec, self.params, x, g_out)

    def test_checking_an_expert_takes_that_expert_and_its_tokens_alone(self):
        """check_expert takes no model, only the opened cell: one expert's weights (1/32 of the routed weights here)
        and the rows of the tokens routed to it (k/32 of them, on average)."""
        spec, root = self.spec, self.trace.root()
        counts = np.bincount(self.trace.routing.experts.ravel(), minlength=spec.n_experts)
        self.assertEqual(spec.n_params - spec.part_size(), spec.n_experts * spec.part_size(0))
        for e in range(spec.n_experts):
            cell = moe.open_expert(self.trace, e)
            self.assertEqual(list(moe.unpack(spec, cell["weights"], e)), spec.names(e))  # its arrays, no other's
            for kind in ("x", "dy", "y", "dx"):
                self.assertEqual(cell[kind].shape, (counts[e], spec.d_model))
            self.assertIsNone(moe.check_expert(spec, root, e, cell, TOL))

    def test_checking_the_router_takes_no_routed_expert(self):
        cell = moe.open_router(self.trace)
        self.assertEqual(list(moe.unpack(self.spec, cell["weights"])), self.spec.names())
        self.assertFalse(any(n.startswith("expert.") for n in self.spec.names()))
        self.assertTrue(all("weights" not in ex for ex in cell["experts"]))
        self.assertIsNone(moe.check_router(self.spec, self.trace.root(), cell, TOL))


class DeterminismTests(unittest.TestCase):
    def test_the_same_inputs_give_the_same_bits(self):
        for precision in ("float64", "float32"):
            def run():
                spec, params, x, g_out = setup(precision)
                routing, out, grads = claims(spec, params, x, g_out)
                arrays = [routing.experts, routing.gates, out, *(grads[n] for n in sorted(grads))]
                return [a.tobytes() for a in arrays], moe.trace_block(spec, params, x, g_out).root()
            self.assertEqual(run(), run(), precision)

    def test_replaying_a_routing_gives_the_bits_that_made_it(self):
        spec, params, x, g_out = setup("float32")
        routing, out, grads = claims(spec, params, x, g_out)
        again, out2, grads2 = claims(spec, params, x, g_out, routing)
        self.assertEqual(again.hash(), routing.hash())
        self.assertEqual(out2.tobytes(), out.tobytes())
        self.assertEqual({n: g.tobytes() for n, g in grads2.items()}, {n: g.tobytes() for n, g in grads.items()})
        self.assertEqual(moe.trace_block(spec, params, x, g_out, routing).root(),
                         moe.trace_block(spec, params, x, g_out).root())

    def test_exact_ties_go_to_the_lower_index(self):
        spec, params, x, _ = setup()
        params = params | {"router": params["router"].copy()}
        params["router"][:, 5] = params["router"][:, 2]  # experts 2 and 5 score exactly alike for every token
        routing, scores = moe.route(spec, params, x)
        np.testing.assert_array_equal(scores[:, 2], scores[:, 5])
        has2, has5 = (routing.experts == 2).any(axis=1), (routing.experts == 5).any(axis=1)
        self.assertTrue(np.any(has2 & ~has5))  # the tie mattered somewhere...
        self.assertFalse(np.any(has5 & ~has2))  # ...and always went to expert 2


if __name__ == "__main__":
    unittest.main()
