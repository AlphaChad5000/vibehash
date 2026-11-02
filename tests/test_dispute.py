"""Disputes over fine-tunes' claims (dispute.py): the referee (referee.py), the states it reads (merkle.py), the program
both sides follow (program.py) and a worker's side of the game (tracer.py), on tiny random checkpoints.

The rounds these games are part of are in tests/test_finetune.py (NodeFreeRoundTests)."""
import hashlib
import math
import unittest

import numpy as np

from vibehash import dispute, merkle, referee

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tests.finetune_fixtures import CONFIGS, TinyModel
    from tests.test_finetune import tmpdir
    from vibehash import finetune as ft
    from vibehash.program import Op, Program, groups, split_theta
    from vibehash.tracer import Executor, Party

needs_torch = unittest.skipIf(torch is None, "needs PyTorch")
CUDA = torch is not None and torch.cuda.is_available()
ALL = ("q", "k", "v", "o", "gate", "up", "down")


class MathTests(unittest.TestCase):
    """The referee's exp, log, sin and cos are built from + − × ÷, so every machine gets the same bits."""

    def test_close_to_the_c_library(self):
        x = np.random.default_rng(0).uniform(-80, 80, 2000)
        self.assertLess(np.max(np.abs(referee.dexp(x) / np.exp(x) - 1)), 4e-16)
        y = np.abs(x) * 10.0 ** np.random.default_rng(1).uniform(-30, 30, 2000)
        self.assertLess(np.max(np.abs(referee.dlog(y) - np.log(y))), 4e-16 * np.max(np.abs(np.log(y))) + 1e-15)
        a = np.random.default_rng(2).uniform(-3e4, 3e4, 2000)
        s, c = referee.dsincos(a)
        self.assertLess(np.max(np.abs(s - np.sin(a))), 1e-12)
        self.assertLess(np.max(np.abs(c - np.cos(a))), 1e-12)
        self.assertEqual((referee.dexp(np.array([0.0, -np.inf, 1000.0]))).tolist(), [1.0, 0.0, np.inf])

    def test_pinned_values(self):
        """Bit patterns these must keep: a NumPy or platform that changed them would split validators."""
        got = [float(referee.dexp(np.array([1.0]))[0]).hex(), float(referee.dlog(np.array([10.0]))[0]).hex(),
               float(referee.dsincos(np.array([1234.5]))[0][0]).hex()]
        # (e here is an ulp above the correctly rounded one: what matters is that it's the same everywhere)
        self.assertEqual(got, ["0x1.5bf0a8b14576ap+1", "0x1.26bb1bbb55516p+1", "0x1.29c531dfd031bp-3"])
        self.assertAlmostEqual(float.fromhex(got[2]), math.sin(1234.5), places=12)

    def test_float32_bounds(self):
        """A dot product's bound covers any summation order. It's rigorous, so it has room: an order's actual error
        is a small part of it (the room a cheat could use, which DESIGN.md weighs)."""
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(1, 4096)).astype(np.float32), rng.normal(size=(1, 4096)).astype(np.float32)
        e = referee.contract(referee.E(a.astype(np.float64)), b.astype(np.float64))
        for order in (np.arange(4096), rng.permutation(4096), np.argsort(np.abs(a[0] * b[0]))):
            acc = np.float32(0)
            for i in order:
                acc = np.float32(acc + np.float32(a[0, i] * b[0, i]))
            self.assertLessEqual(abs(float(acc) - float(e.v[0, 0])), float(e.e[0, 0]) / 10)


class MerkleTests(unittest.TestCase):
    def test_multiproofs(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(37, 300)).astype(np.float32)
        tree = merkle.levels(merkle.leaf_hashes(a))
        n = merkle.n_leaves(*a.shape)
        flat = a.reshape(-1)
        for idx in ([0], [n - 1], [3, 4, 5], sorted(rng.choice(n, 9, replace=False).tolist()), list(range(n))):
            proof = merkle.multiproof(tree, idx)
            leaves = {i: merkle._leaf(flat[i * merkle.LEAF:(i + 1) * merkle.LEAF].tobytes()) for i in idx}
            self.assertTrue(merkle.verify(tree[-1][0], n, leaves, proof))
            bad = dict(leaves)
            bad[idx[0]] = merkle._leaf(b"\0" * 8)
            self.assertFalse(merkle.verify(tree[-1][0], n, bad, proof))
            self.assertFalse(merkle.verify(tree[-1][0], n, leaves, proof + [b"\0" * 32]))
        self.assertEqual(merkle.root_of(a), merkle.tensor_root(37, 300, tree[-1][0]))

    def test_manifests_are_canonical_and_checked(self):
        m = merkle.manifest_bytes({"b": ("1" * 64, "2" * 64, 3, 4), "a": ("3" * 64, "4" * 64, 1, 1)})
        self.assertEqual(merkle.parse_manifest(m)["a"], ("3" * 64, "4" * 64, 1, 1))
        for bad in (m.replace(b'"a"', b'"a" '), m.replace(b"1,1]", b"0,1]"), b"{}", b"x" * (merkle.MAX_MANIFEST + 1)):
            with self.assertRaises(merkle.MerkleError):
                merkle.parse_manifest(bad)

    def test_containers(self):
        blob = merkle.pack({"x": 1}, {"b": b"22", "a": b"1"})
        self.assertEqual(merkle.unpack(blob, 100), ({"x": 1}, {"a": b"1", "b": b"22"}))
        for bad in (blob + b"!", blob[:-1], blob[:8] + blob[9:]):
            with self.assertRaises(merkle.MerkleError):
                merkle.unpack(bad, 100)


@needs_torch
class _Tiny(unittest.TestCase):
    def tiny(self, family="qwen3", device="cpu", **kw):
        self.model = TinyModel(tmpdir(self), family).use(self)
        self.spec = self.model.spec(targets=ALL, rank=4, **kw)
        self.data = self.model.dataset()
        self.rt = ft.runtime(self.spec, self.data, device)
        if device != "cpu":
            from vibehash import compute
            compute.TorchBackend(device)
        return self


class _Full:
    """referee.judge's view of whole arrays (no proofs needed): dispute.Inputs over a state."""

    def __new__(cls, st, spec, data, rt):
        from vibehash.tracer import TorchWeights
        out = dispute.Outside(spec, data, TorchWeights(rt.model), keep=False)
        return dispute.Inputs({n: tuple(a.shape) for n, a in st.items()}, out, full=st)


class RefereeTests(_Tiny):
    """Every op of a step, every tile of its output: an honest float32 computation passes, one changed by 0.1% doesn't."""

    def judge_step(self, device="cpu", every=1):
        prog = Program(self.spec, self.data, 1e-2, 3, 5)
        theta = np.zeros(self.spec.n_params)
        theta[self.spec.parts[0]] = np.random.default_rng(0).normal(0, 0.05, self.spec.n_adapter)
        st = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in split_theta(self.spec, theta).items()}
        ex, node = Executor(self.rt, self.spec, prog), prog.step(0)
        honest = caught = tampered = 0
        kinds = set()
        for k, op in enumerate(node.ops()):
            before = {n: t.cpu().numpy() for n, t in st.items()}
            ex.run(op, st)
            if not op.out or k % every:
                continue
            kinds.add(op.kind)
            out = st[op.out].cpu().numpy().reshape(-1)
            x = _Full(before, self.spec, self.data, self.rt)
            for leaf in range(merkle.n_leaves(*op.shape)):
                said = out[leaf * merkle.LEAF:(leaf + 1) * merkle.LEAF]
                ok, why = referee.judge(op, x, leaf, said)
                self.assertTrue(ok, why)
                honest += 1
                j = len(said) // 2
                if np.isfinite(said[j]) and said[j] != 0:
                    bad = said.copy()
                    bad[j] *= 1.001
                    tampered += 1
                    caught += not referee.judge(op, x, leaf, bad)[0]
        return honest, caught, tampered, kinds

    def test_every_op_of_every_family(self):
        for family in CONFIGS:
            with self.subTest(family=family):
                honest, caught, tampered, kinds = self.tiny(family).judge_step()
                self.assertGreater(honest, 1000)
                self.assertGreater(caught, 0.98 * tampered)
        self.assertEqual(kinds | {"embed"}, set(referee.OPS) - {"free"})

    @unittest.skipUnless(CUDA, "needs a CUDA GPU")
    def test_a_cuda_step_passes_the_cpu_referee(self):
        honest, caught, tampered, _ = self.tiny("qwen3", "cuda").judge_step("cuda")
        self.assertGreater(honest, 1000)
        self.assertGreater(caught, 0.98 * tampered)

    def test_lora_a_one_matches_lora_a(self):
        self.tiny()
        whole = self.spec.lora_a()
        for k in (0, 3, len(whole) - 1):
            np.testing.assert_array_equal(dispute.lora_a_one(self.spec, k), whole[k])

    def test_rope_tables_match_llm(self):
        from vibehash import llm
        for family in ("qwen3", "llama"):
            arch = llm.Arch.from_hf_config(CONFIGS[family])
            cos, sin = llm.rope_table(arch, 300)
            ours = referee.rope_tables(arch, 300)
            for a, b in zip((cos, sin), ours):
                ulps = np.abs(a.float().numpy().astype(np.float64) - b) / np.spacing(np.abs(b).astype(np.float32) + 1e-30)
                self.assertLessEqual(float(ulps.max()), 1.0)


class _Remote:
    def __init__(self):
        self.blobs = {}

    def put(self, raw):
        k = hashlib.sha256(raw).hexdigest()
        self.blobs[k] = bytes(raw)
        return k

    def get(self, k):
        return self.blobs[k]


def tamper(where, op):
    if where[:1] == ("step 2",) and where[1].startswith("step 2 sequence 1") and where[2:3] == ("layer 1 backward",) \
            and op.kind == "attn_dv":
        return 1.25
    return None


class GameTests(_Tiny):
    """Games played with dispute.play on tiny models: the node's side with NumPy only, the parties' with tracer.Party."""

    def setUp(self):
        self.lr, self.batch, self.seed, self.per = 1e-2, 4, 7, 4

    def unit(self, tamper_fn=None, device="cpu"):
        from vibehash.compute import TorchBackend
        return ft.train_segment(self.spec, self.data, np.zeros(self.spec.n_params), self.seed, 0, self.per, self.lr,
                                self.batch, None if device == "cpu" else TorchBackend(device), tamper=tamper_fn)

    def play(self, after, defender, challenger, cheat=None, policy=None):
        """One game over segment 0, from fresh adapters to `after`. `policy(game, who) -> move | None | "default"`."""
        from vibehash import hub
        prog = Program(self.spec, self.data, self.lr, self.batch, self.seed, 1, self.per)
        remote = _Remote()
        roots = []
        for theta in (np.zeros(self.spec.n_params), after):
            root, m, blobs = dispute.checkpoint_state(self.spec, theta)
            for b in blobs.values():
                remote.put(b)
            remote.put(m)
            roots.append(root)
        g = dispute.Game("g", "u", "D", "C", "dispute", ["segment", 0], *roots, seed=self.seed).start(prog)
        d, c = Party(defender, self.spec, self.data, remote, cheat), Party(challenger, self.spec, self.data, remote)
        weights = dispute.HubWeights(hub.Manifest.from_bytes(self.model.manifest_bytes))
        while g.turn != "done":
            v = g.view()
            who = "C" if g.turn == "pick" else "D"
            move = policy(g, who) if policy else "default"
            if move is None:
                dispute.silence(g)
                break
            if move == "default":
                p = c if who == "C" else d
                move = (p.pick(prog, v) if g.turn == "pick" else {"split": p.split(prog, v)} if g.turn == "split"
                        else {"evidence": remote.put(p.evidence(prog, v))})
            dispute.play(g, prog, who, move, evidence=remote.blobs.get(move.get("evidence")),
                         outside=dispute.Outside(self.spec, self.data, weights),
                         check_states=lambda rs: [remote.get(r) for r in rs])
            g.moves.append(move)
        return g

    def test_a_tampered_op_is_found(self):
        """A worker that scaled one op's output deep in one step (the attention's value gradient, in layer 1's
        backward, for the second sequence of step 2) loses: the game narrows to that op, and the referee finds it."""
        self.tiny()
        g = self.play(self.unit(tamper), self.rt, self.rt, tamper)
        self.assertEqual(g.outcome, "challenger", g.detail)
        self.assertEqual(g.leaf.kind, "attn_dv")
        self.assertTrue(g.detail.startswith("attn_dv → b.dv"), g.detail)
        self.assertLessEqual(len(g.moves), 13)

    def test_an_honest_defender_beats_a_false_challenger(self):
        """A challenger that picks the last child at every level, and the first tile of the op it ends at."""
        self.tiny()
        prog = Program(self.spec, self.data, self.lr, self.batch, self.seed, 1, self.per)

        def false(g, who):
            if who == "D":
                return "default"
            child = len(groups(g.lo, g.hi)) - 1
            g0, g1 = groups(g.lo, g.hi)[child]
            atom = g.node(prog).atoms[g0] if g1 - g0 == 1 else None
            return {"pick": child, **({"tile": 0} if isinstance(atom, Op) and atom.out else {})}
        g = self.play(self.unit(), self.rt, self.rt, policy=false)
        self.assertEqual(g.outcome, "defender", g.detail)

    def test_silence_and_broken_chains_lose(self):
        self.tiny()
        after = self.unit()
        self.assertEqual(self.play(after, self.rt, self.rt, policy=lambda g, who: None if who == "D" else "default").outcome,
                         "challenger")  # a silent defender
        quiet = self.play(after, self.rt, self.rt, policy=lambda g, who: None if who == "C" else "default")
        self.assertEqual((quiet.outcome, quiet.detail), ("defender", "the challenger didn't pick in time"))

        def broken(g, who):  # children that end somewhere other than the state being defended
            if who == "D" and g.turn == "split":
                return {"split": [g.a] + ["0" * 64] * len(groups(g.lo, g.hi))}
            return "default"
        g = self.play(after, self.rt, self.rt, policy=broken)
        self.assertEqual(g.outcome, "challenger")
        self.assertIn("don't chain", g.detail)

    def test_moves_are_checked(self):
        self.tiny()
        prog = Program(self.spec, self.data, self.lr, self.batch, self.seed, 1, self.per)
        g = dispute.Game("g", "u", "D", "C", "dispute", ["segment", 0], "a" * 64, "b" * 64).start(prog)
        n = len(groups(g.lo, g.hi))
        for who, move in [("C", {"split": ["a" * 64] * (n + 1)}),  # not the challenger's to split
                          ("D", {"pick": 0}),  # not the defender's turn to pick
                          ("D", {"split": ["a" * 64] * n}),  # too few roots
                          ("D", {"split": ["a" * 64] + ["xyz"] * n}),  # not hashes
                          ("D", {"split": ["a" * 64] * (n + 1), "tile": 0}),
                          ("D", "split"), ("D", {}), ("D", {"evidence": "a" * 64})]:
            with self.subTest(move=move), self.assertRaises(dispute.MoveRefused):
                dispute.play(g, prog, who, move)
        dispute.play(g, prog, "D", {"split": ["a" * 64] + ["c" * 64] * (n - 1) + ["b" * 64]})
        for move in [{"pick": n}, {"pick": -1}, {"pick": True}, {"pick": 0, "tile": 0}, {"concede": 1}]:
            with self.subTest(move=move), self.assertRaises(dispute.MoveRefused):
                dispute.play(g, prog, "C", move)
        dispute.play(g, prog, "C", {"concede": True})
        self.assertEqual(g.outcome, "defender")  # a checker that concedes loses: it accused

    @unittest.skipUnless(CUDA, "needs a CUDA GPU")
    def test_a_cuda_defender_beats_a_cpu_checker(self):
        """An honest unit trained on CUDA, challenged by a checker that recomputes on the CPU (whose numbers differ
        in their last bits throughout): the CPU-side referee sides with the CUDA worker."""
        self.tiny()
        cuda = ft.runtime(self.spec, self.data, "cuda")
        from vibehash import compute
        compute.TorchBackend("cuda")
        after = self.unit(device="cuda")
        g = self.play(after, cuda, self.rt)
        self.assertEqual(g.outcome, "defender", g.detail)
        self.assertNotEqual(g.leaf.kind, "embed", "the CPU challenger found a difference to chase")


if __name__ == "__main__":
    unittest.main()
