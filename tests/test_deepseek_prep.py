"""vibehash/deepseek_prep.py: the one-time prep of a top-layers DeepSeek-V4.1 job, on the small config, CPU only.

- RowTests: streaming a table for the rows a dataset reads, interrupted and resumed, gives exactly those rows, and
  the slice dequantises them as the engram layer does.
- PrepTests: every record a prep writes is what deepseek_top.precompute makes for that sequence, bit for bit; the
  Merkle root and per-layer hashes are reproducible; and a prep stopped part way, then run again, ends exactly
  where an uninterrupted one does.
- TokenMapTests (needs `tokenizers` and the Hub): the compressed token map has config.json's 99,092 ids.
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek, deepseek_prep as prep, deepseek_top as top
    from tests.test_deepseek import tiny_config, token_map_for
    from tests.test_deepseek_top import split
except ImportError:
    torch = None


def dataset(cfg, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        s = int(torch.randint(10, 30, (1,), generator=g))
        ids = torch.randint(0, cfg.vocab_size, (s,), generator=g)
        out.append((ids.tolist(), (torch.rand(s, generator=g) < 0.5).long().tolist()))
    return out


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class RowTests(unittest.TestCase):
    def test_streamed_rows_resume(self):
        g = torch.Generator().manual_seed(0)
        table = torch.randint(0, 256, (1000, 24), generator=g, dtype=torch.uint8).numpy()
        flat = table.reshape(-1).tobytes()
        wanted = torch.randperm(1000, generator=g)[:137].sort().values.numpy()
        with tempfile.TemporaryDirectory() as d:
            read = (lambda o, n: flat[o:o + n])
            with self.assertRaises(prep.Stop):
                prep.extract_rows(read, 0, 24, wanted, Path(d), piece_rows=50, workers=1, stop_after=3)
            kept = len(list(Path(d).glob("piece-*.npy")))  # the 3, and any piece already under way
            self.assertTrue(3 <= kept < len(set((wanted // 50).tolist())), kept)
            got = prep.extract_rows(read, 0, 24, wanted, Path(d), piece_rows=50)
        self.assertTrue((got == table[wanted]).all())

    def test_slice_dequantises_as_the_engram_layer(self):
        cfg = tiny_config()
        state = deepseek.random_weights(cfg, 0)
        block = deepseek.Model(cfg, [1], embed=False, head=False).load_weights(
            {k: v for k, v in state.items() if k.startswith("layers.1.")}).layers["1"]
        tokens = torch.randint(0, cfg.vocab_size, (1, 20))
        hashes = deepseek.engram_hashes(tokens, token_map_for(cfg), cfg)[:, :, 0]
        wanted = torch.unique(hashes).numpy()
        with tempfile.TemporaryDirectory() as d:
            sl = prep.LocalSource(state).slice(1, wanted, Path(d))
        self.assertTrue(torch.equal(sl.rows(hashes, torch.bfloat16), block.engram.rows(hashes)))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PrepTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cfg = tiny_config()
        self.state = deepseek.random_weights(self.cfg, 3)
        self.cut = top.Cut(self.cfg, 4)  # an engram layer below; compressed keys, index keys, candidates cross
        prep.save_dataset(self.tmp / "data.npz", dataset(self.cfg))
        self.seqs = prep.load_dataset(self.tmp / "data.npz")
        self.tmap = token_map_for(self.cfg)

    def prepare(self, name, **kw):
        return prep.run(prep.LocalSource(self.state), prep.LocalUploader(self.tmp / name / "repo"), self.seqs,
                        self.cut, self.tmp / name / "work", block_tokens=40, token_map=self.tmap, **kw)

    def test_records_are_the_precompute(self):
        manifest = self.prepare("a")
        bottom = top.freeze(deepseek.Model(self.cfg, range(4), head=False).load_weights(split(self.cfg, self.state, 4)[0]))
        repo = self.tmp / "a" / "repo"
        n = 0
        for blk in manifest["blocks"]:
            data = (repo / blk["file"]).read_bytes()
            meta = json.loads((repo / blk["file"].replace(".bin", ".json")).read_text())
            for j, i in enumerate(meta["sequences"]):
                rec = top.unpack(data[meta["offsets"][j]:meta["offsets"][j + 1]])
                ids, mask = self.seqs[i]
                ex, choices = top.precompute(bottom, self.cut, ids[None], mask[None], self.tmap)
                self.assertTrue(torch.equal(rec["x"], ex.x[0]), i)
                self.assertTrue(torch.equal(rec["pre"], ex.pre[0]), i)
                self.assertTrue(torch.equal(rec["compress_kv"], ex.side.compress_kv[0]), i)
                self.assertTrue(torch.equal(rec["index_k"], ex.side.index_k[0]), i)
                self.assertTrue(torch.equal(rec["candidates"].bool(), ex.candidates[0]), i)
                self.assertTrue(torch.equal(rec["mask"].bool(), mask))
                self.assertEqual(len(meta["layer_hashes"][j]), 4)
                n += 1
        self.assertEqual(n, len(self.seqs))
        self.assertGreater(len(manifest["blocks"]), 1)
        self.assertEqual(json.loads((repo / "manifest.json").read_text())["root"], manifest["root"])

    def test_stopped_and_resumed(self):
        whole = self.prepare("whole")
        with self.assertRaises(prep.Stop):
            self.prepare("parts", stop_after_pieces=1)  # in the middle of streaming the engram table
        with self.assertRaises(prep.Stop):
            self.prepare("parts", stop_after_blocks=1)  # after the first block
        again = self.prepare("parts")
        self.assertEqual(again["root"], whole["root"])
        self.assertEqual([b["leaves"] for b in again["blocks"]], [b["leaves"] for b in whole["blocks"]])

    def test_fp8_stream(self):
        manifest = self.prepare("fp8", fp8=True)
        self.assertNotEqual(manifest["root"], self.prepare("bf16")["root"])
        repo, blk = self.tmp / "fp8" / "repo", manifest["blocks"][0]
        meta = json.loads((repo / blk["file"].replace(".bin", ".json")).read_text())
        rec = top.unpack((repo / blk["file"]).read_bytes()[meta["offsets"][0]:meta["offsets"][1]])
        other = json.loads((self.tmp / "bf16" / "repo" / blk["file"].replace(".bin", ".json")).read_text())
        bf16 = top.unpack((self.tmp / "bf16" / "repo" / blk["file"]).read_bytes()[:other["offsets"][1]])
        self.assertTrue(torch.equal(rec["x"], deepseek.round_fp8(bf16["x"])))  # the same sequence, rounded


class TokenMapTests(unittest.TestCase):
    def test_compressed_token_map(self):
        if torch is None:
            self.skipTest("needs PyTorch 2.8 or later")
        import importlib.util
        if importlib.util.find_spec("tokenizers") is None:
            self.skipTest("needs the tokenizers package")
        try:
            from tests import deepseek_reference as reference
            path = reference.fetch_tokenizer()
        except Exception as e:  # the Hub can't be reached
            self.skipTest(f"needs DeepSeek's tokenizer from the Hub: {e}")
        m = prep.compressed_token_map(path)
        self.assertEqual(len(m), 129280)
        self.assertEqual(len(set(m.tolist())), deepseek.Config.flash().engram_compressed_vocab_size)


if __name__ == "__main__":
    unittest.main()
