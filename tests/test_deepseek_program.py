"""vibehash/deepseek_program.py and deepseek_tracer.py: a DeepSeek top-layers training unit, op by op, and its disputes
(the tiny config, CPU).

- ProgramTests: a unit trained op by op agrees with deepseek_top's autograd step to float32 rounding, and replays to
  the bit however its steps are split into segments, as a checker replays one.
- RefereeTests: every op of an honest step passes the node's referee (referee_deepseek.py, NumPy), and the same tiles
  bent by 3% are refused.
- GameTests: games played with dispute.play between deepseek_tracer.Party's: a trainer that scaled one op's output
  (the attention's key gradient, deep in step 2's backward) loses, the game narrowed to that op; an honest one beats a
  challenger that picks blindly.
"""
import collections
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek, deepseek_job as dj, deepseek_prep as prep, deepseek_split as split
    from vibehash import deepseek_top as top, deepseek_unit, dispute, merkle, referee, referee_deepseek
    from vibehash.deepseek_job import TokenFile
    from vibehash.deepseek_program import DeepSeekProgram
    from vibehash.deepseek_tracer import Executor, Party
    from vibehash.model import batch_indices
    from vibehash.program import Op, groups, split_theta
    from vibehash.worker import tamper
    from tests.test_deepseek import token_map_for
    from tests.test_deepseek_job import job
    from tests.test_dispute import _Remote
except ImportError:
    torch = None


class _Tiny(unittest.TestCase):
    """A tiny DeepSeek job after its prep: six short sequences, their records in a local bucket."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.spec = spec = dj.spec("tiny", seq_len=64)
        g = np.random.default_rng(0)
        seqs = []
        for _ in range(6):
            n = int(g.integers(10, 30))
            seqs.append((g.integers(0, spec.cfg.vocab_size, n).tolist(), (g.random(n) < 0.5).astype(int).tolist()))
        path = self.tmp / "data.npz"
        prep.save_dataset(path, seqs)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.job = job(spec=spec, job_id="think-shorter-tiny",
                       data={"kind": "deepseek", "sha256": sha, "train": [0, 4], "heldback": [4, 6]})
        self.job.attach_data(TokenFile.from_npz(path), sha)
        store = split.LocalStore(self.tmp / "bucket")
        chunks = self.job.data.tokens.chunks(spec.chunk_tokens)
        m = split.run_local(prep.LocalSource(deepseek.random_weights(spec.cfg, 0)), store, prep.load_dataset(path),
                            chunks, spec.top, split.pieces(spec.cut, spec.pieces), token_map_for(spec.cfg),
                            self.tmp / "work", ranges=spec.row_ranges)
        self.job.data.records = deepseek_unit.Records(store, m, m["root"], self.tmp / "records", spec.top)
        deepseek_unit.unload()
        self.addCleanup(deepseek_unit.unload)
        self.rt = deepseek_unit.runtime(spec, "cpu")
        self.theta = spec.init_params()
        self.theta[spec.parts[0]] = np.random.default_rng(1).normal(0, 0.05, spec.n_adapter)  # adapters that matter
        self.lr, self.batch, self.seed = 1e-3, 2, 1234

    def source(self):
        """The referee's weights and records, as the node reads them."""
        return referee_deepseek.Sources(referee_deepseek.StateWeights(self.rt.weights),
                                        referee_deepseek.RecordSource(self.job.data))

    def train(self, start, n, tamper_fn=None, theta=None):
        return deepseek_unit.train_segment(self.spec, self.job.data, self.theta if theta is None else theta, self.seed,
                                           start, n, self.lr, self.batch, tamper=tamper_fn)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class ProgramTests(_Tiny):
    def test_agrees_with_autograd(self):
        mine = self.train(0, 2)
        ad = deepseek_unit._adapters(self.spec, self.theta, "cpu")
        ad.attach(self.rt.model)
        train = self.job.data.train
        for t in range(2):
            seqs = [train[i] for i in batch_indices(self.seed, t, len(train), self.batch)]
            top.train_step(self.rt.model, self.rt.cut, ad, self.job.data.records.examples(seqs, "cpu"), t + 1, self.lr)
        ref = deepseek_unit._vector(ad)
        for part in self.spec.parts:
            self.assertLess(np.abs(mine[part] - ref[part]).max(), 2e-6 * np.abs(ref[part]).max())
        b = self.spec.parts[0]
        self.assertGreater(np.abs(mine[b] - self.theta[b]).max(), 1e-4)  # it trained

    def test_segments_replay_to_the_bit(self):
        whole = self.train(0, 4)
        again = self.train(2, 2, theta=self.train(0, 2))
        self.assertTrue(np.array_equal(whole, again))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class RefereeTests(_Tiny):
    def test_honest_ops_pass_and_bent_ones_dont(self):
        prog = DeepSeekProgram(self.spec, self.job.data, self.lr, self.batch, self.seed)
        ex = Executor(self.rt, self.spec, prog)
        out = dispute.Outside(self.spec, self.job.data, self.source(), keep=False)
        st = {k: torch.tensor(v) for k, v in split_theta(self.spec, self.theta).items()}
        rng = np.random.default_rng(0)
        kinds, refused, passed_bent = collections.Counter(), [], []
        for op in prog.step(0).ops():
            before = {k: v.numpy() for k, v in st.items()}
            ex.run(op, st)
            if not op.out:
                continue
            after = st[op.out].numpy()
            n = merkle.n_leaves(*op.shape)
            for tile in sorted({0, n - 1, int(rng.integers(0, n))}):
                said = np.ascontiguousarray(after, dtype="<f4").reshape(-1)[tile * merkle.LEAF:(tile + 1) * merkle.LEAF]
                for bend in (False, True):
                    s = said
                    if bend:
                        fin = np.isfinite(said) & (said != 0)
                        if not fin.any() or op.kind in referee.CHOICES:
                            continue
                        s = said.copy()
                        s[fin] *= np.float32(1.03)
                    x = dispute.Inputs({k: v.shape for k, v in before.items()}, out, full=before)
                    x.after = dispute.Inputs({op.out: after.shape}, out, full={op.out: after})
                    ok, why = referee.judge(op, x, tile, s)
                    kinds[op.kind] += 1
                    if not bend and not ok:
                        refused.append((op.describe(), tile, why))
                    if bend and ok:
                        passed_bent.append((op.describe(), tile))
        self.assertEqual(refused, [])
        self.assertEqual(passed_bent, [])
        for kind in ("ds_record", "ds_qlinear", "lora_up", "ds_attn", "ds_picks", "ds_experts", "ds_moe_linear",
                     "ds_moe_sum", "ds_head_grad", "ds_attn_dkv", "ds_moe_linear_bwd", "ds_hc_split_bwd", "lora_grad",
                     "adam_b"):
            self.assertIn(kind, kinds)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class GameTests(_Tiny):
    def play(self, after, cheat=None, policy=None):
        """One game over a 4-step unit's one segment, from self.theta to `after`."""
        prog = DeepSeekProgram(self.spec, self.job.data, self.lr, self.batch, self.seed, 1, 4)
        remote = _Remote()
        roots = []
        for theta in (self.theta, after):
            root, m, blobs = dispute.checkpoint_state(self.spec, theta)
            for b in blobs.values():
                remote.put(b)
            remote.put(m)
            roots.append(root)
        g = dispute.Game("g", "u", "D", "C", "dispute", ["segment", 0], *roots, seed=self.seed).start(prog)
        d = Party(self.rt, self.spec, self.job.data, remote, cheat)
        c = Party(self.rt, self.spec, self.job.data, remote)
        weights = self.source()
        while g.turn != "done":
            v = g.view()
            who = "C" if g.turn == "pick" else "D"
            move = policy(g, who) if policy else "default"
            if move == "default":
                p = c if who == "C" else d
                move = (p.pick(prog, v) if g.turn == "pick" else {"split": p.split(prog, v)} if g.turn == "split"
                        else {"evidence": remote.put(p.evidence(prog, v))})
            dispute.play(g, prog, who, move, evidence=remote.blobs.get(move.get("evidence")),
                         outside=dispute.Outside(self.spec, self.job.data, weights),
                         check_states=lambda rs: [remote.get(r) for r in rs])
            g.moves.append(move)
        return g

    def test_a_tampered_op_is_found(self):
        g = self.play(self.train(0, 4, tamper), tamper)
        self.assertEqual(g.outcome, "challenger", g.detail)
        self.assertEqual(g.leaf.kind, "ds_attn_dkv")
        self.assertLessEqual(len(g.moves), 16)

    def test_an_honest_defender_beats_a_blind_challenger(self):
        prog = DeepSeekProgram(self.spec, self.job.data, self.lr, self.batch, self.seed, 1, 4)

        def blind(g, who):  # the last child at every level, and the op's first tile
            if who == "D":
                return "default"
            child = len(groups(g.lo, g.hi)) - 1
            g0, g1 = groups(g.lo, g.hi)[child]
            atom = g.node(prog).atoms[g0] if g1 - g0 == 1 else None
            return {"pick": child, **({"tile": 0} if isinstance(atom, Op) and atom.out else {})}
        g = self.play(self.train(0, 4), policy=blind)
        self.assertEqual(g.outcome, "defender", g.detail)


if __name__ == "__main__":
    unittest.main()
