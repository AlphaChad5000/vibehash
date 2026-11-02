"""vibehash/deepseek_job.py: a DeepSeek-V4.1 top-layers job's spec, as the protocol reads it.

- SpecTests: the spec survives the chain (posted, parsed back, equal); what's wrong with a bad one is said; a node
  from before DeepSeek jobs refuses one rather than misreading it; Flash's adapters are 2.5M weights of B for K = 4.
- AdapterTests (PyTorch): the spec's projections, their order and their frozen A are deepseek_top.Adapters' own.
- TrainTests (PyTorch, the tiny config): after a prep, a training unit trains from the records, checked against the
  pinned root; its segments replay to the bit however they're split, as checkers replay them; the scorers' held-back
  loss comes from the same records.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vibehash import deepseek_job as dj
from vibehash.tasks import FINETUNE_KEYS, InvalidJob, TrainingJob

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    import numpy as np
    from vibehash import deepseek, deepseek_prep as prep, deepseek_split as split, deepseek_top as top
    from vibehash import deepseek_unit, model
    from vibehash.deepseek_job import TokenFile
    from tests.test_deepseek import token_map_for
except ImportError:
    torch = None

DATA = {"kind": "deepseek", "sha256": "f4d7f822" + "0" * 56, "train": [0, 1500], "heldback": [1500, 2000]}


def job(spec=None, **kw):
    s = spec or dj.spec("flash")
    return TrainingJob(job_id=kw.pop("job_id", "think-shorter-pilot"), requester="r", budget=10, fee_per_round=1,
                       spec=s, data_source=kw.pop("data", DATA), init_seed=s.seed, lr=1e-4, batch_size=8,
                       steps_per_unit=16, checkpoint_every=4, units_per_round=8, memory_mb=8000, tolerance=1e-4,
                       max_update_ratio=1.0, **kw)


class SpecTests(unittest.TestCase):
    def test_round_trip(self):
        j = job()
        back = TrainingJob.from_spec_json(j.spec_json(), "r", 10)
        self.assertTrue(back.deepseek and back.finetune)
        self.assertEqual(back.spec, j.spec)
        self.assertEqual(back.spec_json(), j.spec_json())
        self.assertEqual(json.loads(j.spec_json())["deepseek"],
                         {"cut": 36, "pieces": 4, "chunk_tokens": 32768, "row_ranges": 15})

    def test_sizes(self):
        s = dj.spec("flash")
        self.assertEqual(s.n_adapter, 4 * 634_880)  # DEEPSEEK.md §8: 635K weights of B a layer
        self.assertEqual(s.n_params * 4, 30_474_240)  # [B|m|v] in float32: a 30 MB checkpoint
        self.assertEqual([m[0] for m in s.modules()][:4],
                         ["layers.36.attn.wq_a", "layers.36.attn.wq_b", "layers.36.attn.wkv", "layers.36.attn.wo_b"])
        self.assertEqual(s.peft_config()["layers_to_transform"], [36, 37, 38, 39])

    def test_problems_are_said(self):
        good = dj.spec("tiny")
        self.assertEqual(good.problems(), [])
        for bad, why in ((dj.spec("tiny", cut=1), "kv source"), (dj.spec("tiny", rank=0), "rank"),
                         (dj.spec("tiny", targets=("q",)), "targets"), (dj.spec("tiny", pieces=9), "pieces")):
            self.assertTrue(any(why in p for p in bad.problems()), (why, bad.problems()))
        with self.assertRaises(InvalidJob):
            job(spec=dj.DeepSeekSpec(**{**good.__dict__, "base": dj.BASES["flash"]}))
        with self.assertRaises(InvalidJob):
            job(data={**DATA, "heldback": [1000, 2000]})  # overlaps the training sequences

    def test_an_older_node_refuses_it(self):
        d = json.loads(job().spec_json())
        self.assertNotIn("deepseek", FINETUNE_KEYS)  # what PR #5's fine-tune parser accepts
        with self.assertRaises(InvalidJob):
            TrainingJob._from_finetune(d, "r", 10)  # as a node without DeepSeek jobs parses it: refused, not misread


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class AdapterTests(unittest.TestCase):
    def test_adapters_match(self):
        s = dj.spec("tiny", targets=("wq_a", "wq_b", "wkv", "wo_b", "w1", "w2", "w3"))
        model = deepseek.Model(s.cfg, list(s.top.trained), embed=False, device="meta")
        ad = top.Adapters.fresh(model, s.top, rank=s.rank, alpha=s.alpha, targets=s.targets, seed=s.seed)
        self.assertEqual(ad.names, [m[0] for m in s.modules()])
        self.assertEqual([tuple(b.shape) for b in ad.b], [(out, s.rank) for _, out, _ in s.modules()])
        self.assertEqual(sum(b.numel() for b in ad.b), s.n_adapter)
        for mine, theirs in zip(s.lora_a(), ad.a):
            self.assertTrue(torch.equal(torch.from_numpy(mine), theirs))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class TrainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.spec = dj.spec("tiny", seq_len=64)
        cfg = self.spec.cfg
        g = np.random.default_rng(0)
        seqs = []
        for _ in range(6):
            n = int(g.integers(10, 30))
            seqs.append((g.integers(0, cfg.vocab_size, n).tolist(), (g.random(n) < 0.5).astype(int).tolist()))
        path = self.tmp / "data.npz"
        prep.save_dataset(path, seqs)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.job = job(spec=self.spec, job_id="think-shorter-tiny",
                       data={"kind": "deepseek", "sha256": sha, "train": [0, 4], "heldback": [4, 6]})
        self.job.attach_data(TokenFile.from_npz(path), sha)
        store = split.LocalStore(self.tmp / "bucket")
        chunks = self.job.data.tokens.chunks(self.spec.chunk_tokens)
        m = split.run_local(prep.LocalSource(deepseek.random_weights(cfg, 0)), store, prep.load_dataset(path), chunks,
                            self.spec.top, split.pieces(self.spec.cut, self.spec.pieces), token_map_for(cfg),
                            self.tmp / "work", ranges=self.spec.row_ranges)
        self.job.data.records = deepseek_unit.Records(store, m, m["root"], self.tmp / "records", self.spec.top)
        deepseek_unit.unload()
        self.addCleanup(deepseek_unit.unload)

    def test_a_unit_trains_and_replays(self):
        spec, data = self.spec, self.job.data
        theta = spec.init_params()
        whole = model.train_segment(spec, data, theta, 1234, 0, 4, 1e-3, 2)
        half = model.train_segment(spec, data, theta, 1234, 0, 2, 1e-3, 2)
        again = model.train_segment(spec, data, half, 1234, 2, 2, 1e-3, 2)
        self.assertTrue(np.array_equal(whole, again))  # a checker replays any segment from its checkpoint
        b, m, v = spec.parts
        self.assertGreater(np.abs(whole[b]).max(), 0)  # it trained
        self.assertTrue((whole[v] >= 0).all())
        self.assertTrue(spec.sane(theta, whole, 1e-3, 4, 1.0))  # within Adam's bound, as the protocol checks
        loss0, _ = self.job.evaluate(theta)
        loss4, acc = self.job.evaluate(whole)
        self.assertTrue(np.isfinite(loss0) and np.isfinite(loss4) and 0 <= acc <= 1)
        self.assertNotEqual(loss0, loss4)

    def test_records_are_checked(self):
        records = self.job.data.records
        self.assertEqual(sorted(records.where), list(range(6)))
        bad = deepseek_unit.Records(records.store, records.manifest, "00" * 32, self.tmp / "other", self.spec.top)
        with self.assertRaises(split.Tampered):
            bad.record(0)  # not the root the job pinned


if __name__ == "__main__":
    unittest.main()
