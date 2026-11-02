"""vibehash/deepseek_split.py: the prep cut into units that miners run, connected by stored hand-offs.

- PlanTests: what each piece of DeepSeek-V4.1-Flash reads and writes, the work graph, and the node's Line
  (claims, optimistic consumption, rollback, deletion of what everyone has read).
- CodecTests: hand-offs decode to exactly what was encoded, and a commitment doesn't depend on the compression.
- PresignTests: the S3 signature is AWS's own worked example's.
- SplitTests (small config, CPU): a prep split into pieces of 1 and 2 layers, the engram table in several row
  ranges, writes exactly deepseek_prep's records and Merkle root, whatever order its units run in.
"""
import datetime
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek, deepseek_prep as prep, deepseek_split as split, deepseek_top as top
    from tests.test_deepseek import tiny_config, token_map_for
    from tests.test_deepseek_prep import dataset
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PlanTests(unittest.TestCase):
    def setUp(self):
        self.cut = top.Cut(deepseek.Config.flash(), 36)

    def test_pieces(self):
        self.assertEqual(split.pieces(36, 9), [(0, 9), (9, 18), (18, 27), (27, 36)])
        self.assertEqual(split.pieces(36, 8)[-1], (32, 36))
        self.assertEqual(split.pieces(36, bounds=[2, 8, 20]), [(0, 2), (2, 8), (8, 20), (20, 36)])

    def test_what_pieces_read_and_write(self):
        cut = self.cut
        self.assertEqual(split.needs(cut, 0, 4), {"engram": [1]})
        self.assertEqual(split.emits(cut, 0, 4), {"compress_kv": 2, "picks": 2})  # layers 4-7 reuse layer 2's
        self.assertEqual(split.needs(cut, 4, 8), {"compress_kv": 2, "picks": 2})
        self.assertEqual(split.needs(cut, 12, 16), {"compress_kv": 8, "picks": 8, "engram": [14]})
        self.assertEqual(split.emits(cut, 20, 24), {"compress_kv": 20, "index_k": 20, "candidates": 20})
        self.assertEqual(split.needs(cut, 24, 28), {"compress_kv": 20, "index_k": 20, "candidates": 20})
        self.assertEqual(split.needs(cut, 21, 24), {"compress_kv": 20, "picks": 20})
        self.assertEqual(split.emits(cut, 32, 36), {})  # the last piece: into the records
        self.assertEqual(split.needs(cut, 32, 36), {"compress_kv": 20, "index_k": 20, "candidates": 20})

    def test_row_ranges(self):
        cfg = self.cut.cfg
        ranges = split.row_ranges(cfg, 1, 15)
        self.assertEqual(len(ranges), 15)
        self.assertEqual((ranges[0][0], ranges[-1][1]), (0, cfg.engram_num_embeddings[0]))
        self.assertTrue(all(lo % split.ROWS_PER_PIECE == 0 for lo, _ in ranges))
        self.assertLess((ranges[0][1] - ranges[0][0]) * 264 / 1e9, 7.0)  # GB a range: a 256-byte row, 8 scales

    def test_line(self):
        graph = split.plan(self.cut, [[0], [1]], split.pieces(36, 9))
        line = split.Line(graph)
        self.assertEqual({u.kind for u in line.claimable()}, {"rows"})  # layer 1 needs its rows first
        for u in line.claimable():
            line.claim(u.name)
            line.done(u.name)
        first = [u.name for u in line.claimable()]
        self.assertEqual(first, ["layers-00-09-c00000", "layers-00-09-c00001"])
        self.assertEqual(line.claimable(holds=range(9, 18)), [])
        line.claim(first[0])
        line.done(first[0])  # optimistic: the next piece can read it before it's accepted
        nxt = [u.name for u in line.claimable(holds=range(9, 18))]
        self.assertEqual(nxt, ["layers-09-18-c00000"])
        line.claim(nxt[0])
        line.done(nxt[0])
        back = line.overturn(first[0])
        self.assertEqual(back[0], first[0])
        self.assertEqual(set(back[1:]), {nxt[0], "layers-18-27-c00000", "layers-27-36-c00000"})  # chunk 0 only
        self.assertEqual(line.state[nxt[0]], "pending")
        for name in (first[0], nxt[0]):
            line.claim(name)
            line.done(name)
        line.accept(first[0])
        gone = line.accept(nxt[0])
        self.assertIn(split.stream_name(0, 9), gone)  # everyone who reads it is past their window
        self.assertNotIn(split.rows_name(0, 1, 0), gone)  # its rows unit isn't accepted yet

    def test_order(self):
        graph = split.plan(self.cut, [[0], [1], [2]], split.pieces(36, 4))
        seen = set()
        for u in graph.order():
            self.assertTrue(graph.deps(u.name) <= seen, u.name)
            seen.add(u.name)
        self.assertEqual(len(seen), 30 + 9 * 3)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class CodecTests(unittest.TestCase):
    def test_round_trip(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(3, 7, 4, 64, generator=g).bfloat16()
        tensors = {"x": x, "pre": torch.rand(3, 7, 4, generator=g), "picks": torch.randint(-1, 9, (7, 5), dtype=torch.int16),
                   "mask": torch.rand(7, 9, generator=g) < 0.5, "fp8": x.float().to(torch.float8_e4m3fn),
                   "empty": torch.zeros(0, 32, dtype=torch.uint8), "ids": torch.arange(5, dtype=torch.int64)}
        back = split.decode(split.encode(tensors))
        self.assertEqual(list(back), list(tensors))
        for k, t in tensors.items():
            self.assertEqual(back[k].dtype, t.dtype, k)
            self.assertTrue(torch.equal(back[k].view(torch.uint8) if t.dtype == torch.float8_e4m3fn else back[k],
                                        t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t), k)
        self.assertEqual(split.digest(back), split.digest(tensors))

    def test_digest_ignores_the_codec(self):
        x = {"x": torch.randn(2, 64).bfloat16()}
        a = split.encode(x)
        old, split.CODEC = split.CODEC, "zlib"
        try:
            b = split.encode(x)
        finally:
            split.CODEC = old
        if old != "zlib":
            self.assertNotEqual(a, b)
        self.assertEqual(split.digest(split.decode(a)), split.digest(split.decode(b)))


class PresignTests(unittest.TestCase):
    def test_aws_example(self):
        # AWS's worked example of a presigned GET (Signature Version 4, query-string authentication).
        when = datetime.datetime(2013, 5, 24, tzinfo=datetime.timezone.utc)
        url = split.presign("GET", "examplebucket.s3.amazonaws.com", "test.txt", "us-east-1", "AKIAIOSFODNN7EXAMPLE",
                            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", 86400, when)
        self.assertTrue(url.startswith("https://examplebucket.s3.amazonaws.com/test.txt?X-Amz-Algorithm="))
        self.assertTrue(url.endswith("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class SplitTests(unittest.TestCase):
    """deepseek_prep's PrepTests setting: cut at 4, an engram layer (1) below it; compressed keys, index keys and
    candidate blocks cross into the records; layer 2 reuses layer 1's keys and picks across a piece boundary."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cfg = tiny_config()
        self.state = deepseek.random_weights(self.cfg, 3)
        self.cut = top.Cut(self.cfg, 4)
        prep.save_dataset(self.tmp / "data.npz", dataset(self.cfg))
        self.seqs = prep.load_dataset(self.tmp / "data.npz")
        self.tmap = token_map_for(self.cfg)
        self.chunks = prep.blocks_of(self.seqs, 40)
        self.whole = prep.run(prep.LocalSource(self.state), prep.LocalUploader(self.tmp / "whole" / "repo"), self.seqs,
                              self.cut, self.tmp / "whole" / "work", block_tokens=40, token_map=self.tmap)

    def split(self, name, bounds, ranges=3):
        store = split.LocalStore(self.tmp / name / "store")
        m = split.run_local(prep.LocalSource(self.state), store, self.seqs, self.chunks, self.cut, bounds, self.tmap,
                            self.tmp / name / "work", ranges=ranges, align=1)
        return m, store

    def assert_same(self, m, store):
        self.assertEqual(m["root"], self.whole["root"])
        self.assertEqual([b["leaves"] for b in m["blocks"]], [b["leaves"] for b in self.whole["blocks"]])
        repo = self.tmp / "whole" / "repo"
        for c in range(len(self.chunks)):
            bin_name, meta_name = split.block_names(c)
            self.assertEqual(store.get(bin_name), (repo / bin_name).read_bytes(), c)
            self.assertEqual(store.get(meta_name), (repo / meta_name).read_bytes(), c)

    def test_one_layer_pieces(self):
        self.assertGreater(len(self.chunks), 1)
        m, store = self.split("one", split.pieces(4, 1))
        self.assert_same(m, store)
        self.assertGreater(len(split.decode(store.get(split.side_name(0, "picks", 1)))), 0)  # crossed into layer 2

    def test_two_layer_pieces(self):
        self.assert_same(*self.split("two", split.pieces(4, 2), ranges=2))

    def test_any_order(self):
        """Units claimed as a Line offers them, one chunk racing ahead of the others, and one unit overturned and
        redone after the next piece read its output: the same records and root."""
        bounds = split.pieces(4, bounds=[0, 1, 3])
        graph = split.plan(self.cut, self.chunks, bounds, 3, align=1)
        line, store = split.Line(graph), split.LocalStore(self.tmp / "order" / "store")
        work, src, last = self.tmp / "order" / "work", prep.LocalSource(self.state), {}
        overturned = False  # until then, every window is still open: nothing is accepted
        while any(s != "accepted" for s in line.state.values()):
            ready = line.claimable()
            u = max(ready, key=lambda u: (u.kind == "rows", u.chunk, u.a))  # the last chunk first, deepest first
            line.claim(u.name)
            if u.kind == "rows":
                split.run_rows(src, store, self.seqs, self.chunks, self.cut, u.a, u.b, self.tmap, work, 3, align=1)
            else:
                commit = split.run_piece(src, store, self.seqs, self.chunks, self.cut, u, self.tmap, work, ranges=3,
                                         align=1)
                if u.b == self.cut.first:
                    last[u.chunk] = commit
            line.done(u.name)
            if u.kind == "layers" and u.a == 1 and not overturned:
                overturned = True
                line.overturn(graph.producer[split.stream_name(u.chunk, 1)])  # its input's maker: redo both
                continue
            if not overturned:
                continue
            for name, s in line.state.items():  # accept whatever is done and whose inputs are accepted
                if s == "done" and all(line.state[d] == "accepted" for d in graph.deps(name)):
                    for gone in line.accept(name):
                        store.delete(gone)
        m = split.manifest(self.cut, self.seqs, self.chunks, list(last.values()))
        self.assert_same(m, store)
        self.assertFalse(store.has(split.stream_name(0, 1)))  # the frontier only: read by everyone, then deleted
        self.assertTrue(store.has(split.block_names(0)[0]))  # the records stay

    def test_commitment_chain(self):
        _, store = self.split("chain", split.pieces(4, 1))
        made = json.loads(store.get(split.commit_name(0, 1)))["outputs"]
        read = json.loads(store.get(split.commit_name(0, 2)))["inputs"]
        for name in (split.stream_name(0, 2), split.side_name(0, "compress_kv", 1)):
            self.assertEqual(read[name], made[name], name)  # what layer 2 read is what layer 1 wrote


if __name__ == "__main__":
    unittest.main()
