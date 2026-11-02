"""vibehash/deepseek_worker.py: a worker's side of a DeepSeek job, on the tiny config (CPU).

- PrepTests: a worker runs the prep's units one by one, as the node hands them out, through the bucket (FakeS3, every
  link signed), each checking its inputs against their writers' digests: the records and root are the one-PC prep's.
  A checker's replay agrees to the bit on the same hardware, passes outputs that differ as other hardware's would,
  and fails ones bent further.
- TakeTests: when a PC can't take DeepSeek work, it's told why and that it keeps earning on other jobs; Flash's
  files for a piece are the layers' own, its embedding and tokenizer only where needed.
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    import numpy as np
    from vibehash import deepseek, deepseek_job as dj, deepseek_prep as prep, deepseek_split as split
    from vibehash import deepseek_worker as dw, modelstore
    from vibehash.deepseek_job import TokenFile
    from tests.test_deepseek_split import FakeS3
except ImportError:
    torch = None


def tiny_job(tmp: Path, n: int = 6):
    spec = dj.spec("tiny", seq_len=64)
    g = np.random.default_rng(0)
    seqs = []
    for _ in range(n):
        k = int(g.integers(10, 30))
        seqs.append((g.integers(0, spec.cfg.vocab_size, k).tolist(), (g.random(k) < 0.5).astype(int).tolist()))
    path = tmp / "data.npz"
    prep.save_dataset(path, seqs)
    return spec, TokenFile.from_npz(path), path


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PrepTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # two chunks, so units hand off to each other: chunk_tokens is small for the tiny text (the spec allows 1024+)
        self.spec = dj.DeepSeekSpec(**{**dj.spec("tiny", seq_len=64, pieces=2).__dict__, "chunk_tokens": 60})
        _, self.tokens, self.path = tiny_job(self.tmp)
        self.weights = dw.Weights(self.spec, "think-shorter-tiny")
        s3 = FakeS3()
        self.addCleanup(s3.close)
        self.store = split.PresignedStore(s3.signer(), part=4096, journal=self.tmp / "journal", wait=lambda s: None)

    def prep_all(self):
        graph = self.spec.graph(self.tokens)
        line, commits, committed = split.Line(graph), {}, {}
        while any(st != "accepted" for st in line.state.values()):
            u = line.claimable()[0]
            line.claim(u.name)
            commits[u.name] = dw.run_unit(self.spec, self.tokens, u, self.store, self.weights, self.tmp / "work",
                                          want=committed)
            committed.update(commits[u.name]["outputs"])
            line.done(u.name)
            line.accept(u.name)
        return graph, commits

    def test_a_whole_prep_through_the_bucket(self):
        graph, commits = self.prep_all()
        self.assertGreater(len(self.tokens.chunks(self.spec.chunk_tokens)), 1)
        last = [c for n, c in commits.items() if graph.units[n].kind == "layers" and graph.units[n].b == self.spec.cut]
        seqs = dw.sequences(self.tokens)
        m = split.manifest(self.spec.top, seqs, self.tokens.chunks(self.spec.chunk_tokens), last)
        whole = prep.run(prep.LocalSource(deepseek.random_weights(self.spec.cfg, 0)),
                         prep.LocalUploader(self.tmp / "one-pc"), prep.load_dataset(self.path), self.spec.top,
                         self.tmp / "one-pc-work", block_tokens=self.spec.chunk_tokens,
                         token_map=self.weights.token_map(self.tmp))
        self.assertEqual(m["root"], whole["root"])  # the one-PC prep's records, to the bit

    def test_checking_a_unit(self):
        graph, commits = self.prep_all()
        name = next(n for n, u in graph.units.items() if u.kind == "layers" and u.a == 0 and u.chunk == 0)
        unit, commit = graph.units[name], commits[name]
        ok, why = dw.check_unit(self.spec, self.tokens, unit, commit, self.store, self.weights, self.tmp / "check")
        self.assertTrue(ok)
        self.assertEqual(why, "the same, to the bit")
        stream = split.stream_name(0, unit.b)
        for scale, holds in ((1.01, True), (1.2, False)):  # as other hardware's arithmetic, then bent
            got = split.decode(self.store.get(stream))
            other = {k: (v.float() * scale).to(v.dtype) if k.endswith(".x") else v for k, v in got.items()}
            alt = split.PresignedStore(self.store.sign, part=4096, wait=lambda s: None)
            alt.put(stream, split.encode(other))
            bent = json.loads(json.dumps(commit))
            bent["outputs"][stream] = split.digest(other)
            ok, why = dw.check_unit(self.spec, self.tokens, unit, bent, alt, self.weights, self.tmp / "check")
            self.assertEqual(ok, holds, why)
            self.assertIn(stream, why)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class TakeTests(unittest.TestCase):
    def test_messages(self):
        gpu = {"name": "RTX 4070 SUPER", "gb": 11.99}
        self.assertEqual(dw.can_take("train", {"gpu": gpu, "ram_gb": 31.8}), (True, ""))
        for mach, why in (({"gpu": None, "ram_gb": 32}, "NVIDIA graphics card"),
                          ({"gpu": {"name": "GTX 1650", "gb": 4}, "ram_gb": 32}, "GTX 1650 has 4 GB"),
                          ({"gpu": gpu, "ram_gb": 8}, "8 GB of memory")):
            ok, msg = dw.can_take("train", mach)
            self.assertFalse(ok)
            self.assertIn(why, msg)
            self.assertIn("keeps earning on other jobs", msg)
        self.assertTrue(dw.can_take("prep", {"gpu": {"name": "RTX 2060", "gb": 5.8}, "ram_gb": 16})[0])
        with tempfile.TemporaryDirectory() as d:
            plan = modelstore.Store(Path(d), cap=10).plan("think-shorter-pilot", {"x": 100})
        ok, msg = dw.can_take("train", {"gpu": gpu, "ram_gb": 32}, plan)
        self.assertFalse(ok)
        self.assertIn("the space you allow", msg)

    def test_flash_files(self):
        wm = {"embed.weight": "model-00002-of-00048.safetensors", "head.weight": "model-00043-of-00048.safetensors",
              "layers.0.attn.wq_a.weight": "model-00003-of-00048.safetensors",
              "layers.1.attn.wq_a.weight": "model-00004-of-00048.safetensors",
              "layers.1.engram.embed.weight": "model-00006-of-00048.safetensors"}
        sizes = {f: 7 * 10**9 for f in set(wm.values())} | {"tokenizer.json": 6_367_257}
        hub = types.SimpleNamespace(list_files=lambda r, c: {f: types.SimpleNamespace(size=s) for f, s in sizes.items()},
                                    weight_map=lambda r, c: wm)
        w = dw.Weights(dj.spec("flash"), "think-shorter-pilot", hub=hub)
        got = w.files([0, 1], embed=True, tokenizer=True)
        self.assertEqual(sorted(n.split("/")[-1] for n in got),
                         ["model-00002-of-00048.safetensors", "model-00003-of-00048.safetensors",
                          "model-00004-of-00048.safetensors", "tokenizer.json"])  # not the 101 GB table
        self.assertTrue(all(n.startswith(f"deepseek/{deepseek.COMMIT}/") for n in got))


if __name__ == "__main__":
    unittest.main()
