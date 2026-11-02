"""Disputes over fine-tunes' claims (dispute.py): the referee (referee.py), the states it reads (merkle.py), the program
both sides follow (program.py) and a worker's side of the game (tracer.py), on tiny random checkpoints.

The rounds these games are part of are in tests/test_finetune.py (NodeFreeRoundTests)."""
import hashlib
import math
import unittest

import numpy as np

from vibehash import dispute, merkle, referee, verification

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

    def play(self, after, defender, challenger, cheat=None, policy=None, party=None):
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
        d = (party or Party)(defender, self.spec, self.data, remote, cheat)
        c = Party(challenger, self.spec, self.data, remote)
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
        from vibehash.program import dispute_size
        self.assertLessEqual(len(g.moves), dispute_size(self.spec, self.batch, 1, self.per)[0])

    def test_the_dispute_phase_has_room_for_the_deepest_game(self):
        """The second review's finding 6: the phase had room for 14 moves, fewer than real trees take. program.
        dispute_size counts them from the spec alone; on a tiny model it agrees with a walk of every atom of the real
        program's tree."""
        from vibehash.program import Node, dispute_size

        self.tiny()
        prog = Program(self.spec, self.data, self.lr, self.batch, self.seed, 2, self.per)

        def rounds(n):
            return 1 if n <= 64 else 1 + rounds(-(-n // 64))

        def deepest(node):
            return rounds(len(node.atoms)) + max((deepest(a) for a in node.atoms if isinstance(a, Node)), default=0)
        self.assertEqual(dispute_size(self.spec, self.batch, 2, self.per)[0], 2 * deepest(prog.unit()) + 1)

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
        self.assertEqual(g.outcome, "withdrawn")  # a checker that concedes withdraws: unpaid, not slashed

    def test_a_tf32_worker_loses_at_full_width(self):
        """The second review's end-to-end case: every projection computed with TF32-rounded activations (cheaper tensor
        cores), at width 2560, ends 1.7% off; the honest checker rejects it, plays the game, and now wins on the
        projection the cheat changed (it used to lose there, within the old worst-case bound, and be slashed)."""
        import torch

        from vibehash import tracer
        model = TinyModel(tmpdir(self), "qwen3", hidden_size=2560, intermediate_size=2560, num_hidden_layers=1,
                          num_attention_heads=8, num_key_value_heads=4, head_dim=64).use(self)
        self.model, self.spec, self.data = model, model.spec(targets=ALL, rank=4), model.dataset()
        self.rt = ft.runtime(self.spec, self.data, "cpu")
        self.per = 2

        def tf32(t):
            b = t.contiguous().view(torch.int32).to(torch.int64)
            return (((b + 0xFFF + ((b >> 13) & 1)) >> 13) << 13).to(torch.int32).view(torch.float32)

        class Lazy(Executor):
            def run(self, op, st):
                if op.kind in ("linear", "linear_bwd"):
                    view = dict(st)
                    view[op.ins[0]] = tf32(st[op.ins[0]])
                    st[op.out] = tracer.RUN[op.kind](self, op, view).contiguous()
                    for d in op.drops:
                        del st[d]
                else:
                    super().run(op, st)

        class LazyParty(Party):
            def executor(self, prog, honest):
                return Executor(self.rt, self.spec, prog) if honest else Lazy(self.rt, self.spec, prog)

        old = tracer.Executor
        tracer.Executor = Lazy
        try:
            lazy = self.unit()
        finally:
            tracer.Executor = old
        self.assertFalse(verification.reproduces(lazy, self.unit(), np.zeros(self.spec.n_params), ft.TOLERANCE,
                                                 self.spec), "the checker rejects it")
        g = self.play(lazy, self.rt, self.rt, party=LazyParty)
        self.assertEqual(g.outcome, "challenger", g.detail)
        self.assertIn(g.leaf.kind, ("linear", "linear_bwd"))

    def test_an_honest_challenger_withdraws_rather_than_lose(self):
        """A checker that rejected an honest unit (here, told to) finds nothing the referee would refuse when it gets
        to an op, and withdraws: unpaid, not slashed (dispute.play's concede)."""
        self.tiny()
        g = self.play(self.unit(), self.rt, self.rt)
        self.assertEqual((g.outcome, g.loser), ("withdrawn", None))

    @unittest.skipUnless(CUDA, "needs a CUDA GPU")
    def test_a_cuda_defender_beats_a_cpu_checker(self):
        """An honest unit trained on CUDA, challenged by a checker that recomputes on the CPU (whose numbers differ
        in their last bits throughout): every number the CUDA worker posts is within the referee's bounds, so the
        checker, running the referee itself before naming a tile, finds nothing to win on and withdraws. Neither is
        slashed; the CUDA worker's evidence for any tile would pass (checked directly)."""
        self.tiny()
        cuda = ft.runtime(self.spec, self.data, "cuda")
        from vibehash import compute
        compute.TorchBackend("cuda")
        after = self.unit(device="cuda")
        g = self.play(after, cuda, self.rt)
        self.assertEqual(g.outcome, "withdrawn", g.detail)
        self.assertGreater(len(g.moves), 4, "the CPU challenger found differences to chase down to an op")



class _Src:
    """Float32 weights by name, as dispute.Outside reads them."""

    def __init__(self, ws):
        self.ws = ws

    def shape(self, name):
        return "F32", tuple(self.ws[name].shape)

    def read(self, name, r0, r1):
        return np.ascontiguousarray(self.ws[name][r0:r1], dtype="<f4").tobytes()


def _round_bits(a, drop):
    """float32s rounded to `drop` fewer mantissa bits, ties to even (16: bf16; 13: TF32)."""
    b = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    half = (1 << (drop - 1)) - 1
    b = ((b + half + ((b >> drop) & 1)) >> drop) << drop
    return b.astype(np.uint32).view(np.float32)


class BoundTests(unittest.TestCase):
    """The second review's finding 1: sums are bounded by how far honest orders actually land (referee.spread), not
    by the worst case γ_K·Σ|terms|, which let TF32 and bf16-activation workers pass at K ≥ 2560."""

    def test_reduced_precision_fails_every_tile_and_honest_passes(self):
        from vibehash.program import Op

        rng = np.random.default_rng(0)
        for K, O in ((1024, 1024), (2560, 2560), (4096, 512)):
            T = 4
            x = rng.normal(0, 1, (T, K)).astype(np.float32)
            w = _round_bits(rng.normal(0, 1 / np.sqrt(K), (O, K)), 16)  # bf16 weights, as pretrained ones are
            op = Op("linear", "y", ("x",), {"weight": "W", "bias": None}, (), (T, O), ((T, K),))
            out = dispute.Outside(None, None, _Src({"W": w}), keep=False)
            cases = {"honest": x @ w.T, "bf16 activations": _round_bits(x, 16) @ w.T,
                     "tf32": _round_bits(x, 13) @ _round_bits(w, 13).T}
            if torch is not None:
                cases["honest (torch)"] = (torch.from_numpy(x) @ torch.from_numpy(w).T).numpy()
            for name, y in cases.items():
                flat = y.reshape(-1)
                passed = [referee.judge(op, dispute.Inputs({"x": (T, K)}, out, full={"x": x}), leaf,
                                        flat[leaf * merkle.LEAF:(leaf + 1) * merkle.LEAF])[0]
                          for leaf in range(merkle.n_leaves(T, O))]
                if name.startswith("honest"):
                    self.assertTrue(all(passed), (K, name))
                else:
                    self.assertFalse(any(passed), (K, name))
            want = referee.op_linear(op, dispute.Inputs({"x": (T, K)}, out, full={"x": x}), 0, 0, merkle.LEAF)
            allowed = referee.MARGIN * want.e + referee.REL * np.abs(want.v) + referee.FLOOR
            self.assertLess(np.median(allowed / np.abs(want.v)), 1e-4, K)  # (it was ~1.5% at K = 2560)

    def test_honest_sums_land_well_inside_the_model(self):
        """The calibration behind SPREAD_C: float32 dot products, by numpy, by torch on the CPU and CUDA, and by a
        naive sequential loop, on random, same-sign, alternating and heavy-tailed terms, land within 2 of the
        statistical scale u·spread(); the referee allows SPREAD_C · MARGIN = 7.5 of it."""
        rng = np.random.default_rng(1)
        worst = 0.0
        for K in (256, 1024, 3072):
            x = rng.normal(0, 1, (8, K)).astype(np.float32)
            w = rng.normal(0, 1 / np.sqrt(K), (64, K)).astype(np.float32)
            alt = np.abs(x) * np.where(np.arange(K) % 2, 1, -1).astype(np.float32)
            heavy = x.copy()
            heavy[:, ::7] *= 50
            for a, b in ((x, w), (np.abs(x), np.abs(w)), (alt, np.abs(w)), (heavy, w)):
                p = a[:, None, :].astype(np.float64) * b[None, :, :].astype(np.float64)
                exact, scale = referee.fsum_rows(p), referee.U * referee.spread(p)
                outs = [a @ b.T]
                if torch is not None:
                    outs.append((torch.from_numpy(a) @ torch.from_numpy(b).T).numpy())
                    if CUDA:
                        outs.append((torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda().T).cpu().numpy())
                seq = np.zeros((2, 8), dtype=np.float32)
                for i in range(2):
                    for j in range(8):
                        acc = np.float32(0)
                        for k in range(K):
                            acc = np.float32(acc + np.float32(a[i, k] * b[j, k]))
                        seq[i, j] = acc
                worst = max(worst, float(np.max(np.abs(seq - exact[:2, :8]) / scale[:2, :8])))
                for y in outs:
                    worst = max(worst, float(np.max(np.abs(y.astype(np.float64) - exact) / scale)))
        self.assertLess(worst, 2.0)
        self.assertGreaterEqual(referee.SPREAD_C * referee.MARGIN, 3 * 2.0)


if __name__ == "__main__":
    unittest.main()


class EvidenceTests(unittest.TestCase):
    """The second review's finding 4: attention's backward over a long sequence reads every query position of a group's
    heads, more than MAX_EVIDENCE, so an honest defender's evidence was refused. A tile's cap now grows with what the
    referee reads for it (dispute.evidence_cap), and a job's sequences are at most finetune.MAX_SEQ_LEN."""

    def test_long_attention_evidence_is_judged_and_oversized_evidence_isnt(self):
        from vibehash.program import Op

        H, KV, hd, T = 64, 2, 64, 512  # small, but a tile reads 40 MB: 64 heads' columns over 512 positions
        G, c = H // KV, referee.f32(1 / np.sqrt(hd))
        rng = np.random.default_rng(4)
        ds = (rng.normal(0, 0.01, (H * T, T))).astype(np.float32)
        qr = rng.normal(0, 1, (T, H * hd)).astype(np.float32)
        a = (ds * np.float32(c)).reshape(H, T, T)
        dk = np.concatenate([np.concatenate(list(a[g * G:(g + 1) * G]), axis=0).T
                             @ np.concatenate([qr[:, h * hd:(h + 1) * hd] for h in range(g * G, (g + 1) * G)], axis=0)
                             for g in range(KV)], axis=1)
        op = Op("attn_dk", "dk", ("ds", "qr"), {"H": H, "KV": KV, "hd": hd, "c": c}, (), (T, KV * hd),
                ((H * T, T), (T, H * hd)))
        before, out = {"ds": ds, "qr": qr}, dispute.Outside(None, None, None)
        after = {**before, "dk": dk}
        ms = (dispute.state_manifest(before)[0], dispute.state_manifest(after)[0])
        roots = [merkle.state_root(m) for m in ms]
        for tile in (0, merkle.n_leaves(T, KV * hd) - 1):
            evidence, ok, why = dispute.make_evidence(op, tile, before, after, ms, out)
            self.assertTrue(ok, why)
            self.assertGreater(len(evidence), dispute.MAX_EVIDENCE)
            self.assertEqual(dispute.judge_evidence(op, tile, *roots, evidence, out)[0], "defender")
        cap = dispute.evidence_cap(op, tile, out)
        self.assertLess(cap, 3 * len(evidence))
        header, chunks = merkle.unpack(evidence, cap)
        padded = merkle.pack(header, {**chunks, "pad": bytes(cap - len(evidence))})
        verdict, why = dispute.judge_evidence(op, tile, *roots, padded, out)
        self.assertEqual((verdict, "too big" in why), ("challenger", True))

    def test_the_cap_stays_bounded_at_the_longest_sequences(self):
        """The reviewer's sizes (t_evidence_size): at MAX_SEQ_LEN, Qwen2.5-3B's attention tiles read 96 MB, past the old
        32 MB; their cap allows them and stays a few hundred MB."""
        from vibehash import finetune
        from vibehash.program import Op

        T = finetune.MAX_SEQ_LEN
        for H, KV, hd, MB in ((32, 8, 128, 48), (16, 2, 128, 96)):  # Qwen3-4B, Qwen2.5-3B
            shapes = {"ds": (H * T, T), "qr": (T, H * hd)}
            op = Op("attn_dk", "dk", ("ds", "qr"), {"H": H, "KV": KV, "hd": hd, "c": 0.125}, (), (T, KV * hd),
                    (shapes["ds"], shapes["qr"]))
            cap = dispute.evidence_cap(op, 0, dispute.Outside(None, None, None))
            self.assertEqual(cap, dispute.MAX_EVIDENCE + 2 * MB * 2**20)
            self.assertLess(cap, 256 * 2**20)
        self.assertTrue(any("seq_len" in p for p in
                            finetune.LoraSpec("x/y", "r", {}, seq_len=T + finetune.SEQ_STEP).problems()))


class StateCheckTests(unittest.TestCase):
    """The second review's finding 5: the node's cache of tensor roots outlived the blobs, so a split naming a state
    whose tensors were pruned still passed Coordinator._check_states (t_stale_roots)."""

    def test_a_pruned_tensor_is_missing_and_shapes_have_their_own_roots(self):
        import tempfile
        import types

        from vibehash.blobs import BlobStore
        from vibehash.coordinator import Coordinator, _MemoryBlobs
        from vibehash.node import TraceBlobs

        x = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
        z = np.zeros((8, 64), dtype=np.float32)
        manifest, blobs = dispute.state_manifest({"x": x, "z": z, "zt": z.reshape(64, 8)})
        for tb in (TraceBlobs(BlobStore(tempfile.mkdtemp())), _MemoryBlobs()):
            fake = types.SimpleNamespace(trace_blobs=tb, _get_trace=tb.get_bytes)
            for raw in blobs.values():
                tb.put_bytes(raw)
            root = tb.put_bytes(manifest)
            Coordinator._check_states(fake, [root])  # zeros in two shapes: one blob, two roots, both right
            Coordinator._check_states(fake, [root])
            if isinstance(tb, TraceBlobs):
                tb.store.prune(set())  # the round's end
                tb.put_bytes(manifest)  # and later, only the manifest is uploaded again
                with self.assertRaises(dispute.MoveRefused):
                    Coordinator._check_states(fake, [root])
                tb.forget()
                self.assertEqual(tb.roots, {})
