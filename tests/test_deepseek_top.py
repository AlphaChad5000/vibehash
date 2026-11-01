"""vibehash/deepseek_top.py: training DeepSeek-V4.1's top layers from precomputed inputs is training the whole model
with its bottom frozen, to the bit.

- ExactTests: the reference's default 5-layer model and the every-feature one, from random weights, cut at several
  layers. A few Adam steps of LoRA on the top layers, (a) through the whole model with the layers below the cut
  frozen, one micro-batch at a time, and (b) layer-major on the top layers alone from what precompute made. Same
  losses, same B, m and v, bit for bit; and the same again from the dataset's packed records.
- CutTests: what crosses the real model's cuts, and the cuts layer-major training refuses.
- RecordTests: records pack and unpack exactly, FP8 streams to round_fp8's values, and their Merkle root.
- PeftTests (needs PEFT): the adapter, saved in PEFT's format under the checkpoint's names, loads into PEFT with
  layers_to_transform and computes what deepseek.py's adapted projections compute.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek
    from vibehash.deepseek import Config, Model, Numerics
    from vibehash import deepseek_top as top
    from vibehash.deepseek_top import Adapters, Cut
    from tests.test_deepseek import tiny_config, token_map_for, tokens_for
except ImportError:
    torch = None


def split(cfg, state, first):
    """The checkpoint's tensors for the frozen part (layers below the cut, the embedding) and the trained part."""
    def layer(k):
        return int(k.split(".")[1]) if k.startswith("layers.") else None
    bottom = {k: v for k, v in state.items() if k.startswith("embed.") or (layer(k) is not None and layer(k) < first)}
    upper = {k: v for k, v in state.items() if k.startswith(("norm.", "head.")) or (layer(k) or -1) >= first}
    return bottom, upper


def models(cfg, state, first, num=None, device="cpu", train_num=None):
    """The whole model (its trained layers and head in train_num, the rest in num), the frozen part and the
    trained part."""
    num = num or Numerics()
    train_num = train_num or num
    cut = Cut(cfg, first)
    state = {k: v.to(device) for k, v in state.items()}
    bottom, upper = split(cfg, state, first)
    whole = top.freeze(Model(cfg, num=num, device=device, layer_num={i: train_num for i in cut.trained},
                             head_num=train_num).load_weights(state))
    low = top.freeze(Model(cfg, range(first), num, device=device, head=False).load_weights(bottom))
    high = top.freeze(Model(cfg, cut.trained, train_num, device=device, embed=False).load_weights(upper))
    return cut, whole, low, high


def learned(tokens, seed):
    """Which positions' successors are learned: about half, never the last."""
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(tokens.shape, generator=g) < 0.5
    m[:, -1] = False
    return m


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class ExactTests(unittest.TestCase):
    def check(self, cfg, first, seq, steps=3, micro=2, tmap=None, seed=0, via_records=False, train_num=None):
        cut, whole, low, high = models(cfg, deepseek.random_weights(cfg, seed), first, train_num=train_num)
        mine, theirs = (Adapters.fresh(m, cut, rank=4, alpha=8.0, seed=7) for m in (high, whole))
        mine.attach(high)
        theirs.attach(whole)
        tokens = [tokens_for(cfg, 1, seq, seed=20 + j) for j in range(micro)]
        masks = [learned(t, 30 + j) for j, t in enumerate(tokens)]
        examples = [top.precompute(low, cut, t, m, tmap)[0] for t, m in zip(tokens, masks)]
        if via_records:  # through the dataset's records and back
            examples = [top.example_of([top.unpack(top.pack(ex, 0))], cut) for ex in examples]
        losses = []
        for step in range(1, steps + 1):
            a = top.whole_model_step(whole, cut, theirs, tokens, masks, step, 1e-2, tmap)
            b = top.train_step(high, cut, mine, examples, step, 1e-2)
            self.assertEqual(a, b, f"step {step}")
            losses.append(a)
        for name, x, y in zip(mine.names, mine.b + mine.m + mine.v, theirs.b + theirs.m + theirs.v):
            self.assertTrue(torch.equal(x, y), name)
        self.assertGreater(float(sum(b.abs().sum() for b in mine.b)), 0)  # it trained
        self.assertLess(losses[-1], losses[0])
        return mine

    def test_default_model_top_layer(self):
        # Layer 4 alone: it reuses layer 3's compressed keys and picks, which cross the cut.
        cfg = Config()
        self.assertEqual(Cut(cfg, 4).crossing(), {"compress_kv": 3, "picks": 3})
        self.check(cfg, 4, seq=160)

    def test_default_model_top_two_layers(self):
        # Layers 3 and 4: layer 3 makes its own compressed keys and picks, and layer 4 reuses them.
        cfg = Config()
        self.assertEqual(Cut(cfg, 3).crossing(), {})
        self.check(cfg, 3, seq=160)

    def test_every_feature(self):
        # Layer 4 picks inside layer 3's candidate blocks with its index keys; an engram layer below the cut.
        cfg = tiny_config()
        self.assertEqual(Cut(cfg, 4).crossing(), {"compress_kv": 3, "index_k": 3, "candidates": 3})
        self.check(cfg, 4, seq=24, tmap=token_map_for(cfg))
        self.check(cfg, 3, seq=24, tmap=token_map_for(cfg))
        with self.assertRaisesRegex(NotImplementedError, "layer 3 is a trained kv source"):
            self.check(cfg, 2, seq=24, tmap=token_map_for(cfg))  # layer 4's gradient would go through layer 3's keys

    def test_float32_training_above_reference_layers(self):
        # As a job runs: the preparer's frozen layers in the reference's bf16 arithmetic, the miners' trained layers
        # and head in float32 without activation rounding (deepseek_top.TRAIN).
        self.check(Config(), 3, seq=160, train_num=top.TRAIN)
        cfg = tiny_config()
        self.check(cfg, 4, seq=24, tmap=token_map_for(cfg), train_num=top.TRAIN)

    def test_from_the_records(self):
        cfg = tiny_config()
        self.check(cfg, 4, seq=24, tmap=token_map_for(cfg), via_records=True)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs a CUDA GPU")
class ReplayTests(unittest.TestCase):
    def test_float32_training_replays_across_devices(self):
        # A checker replays a miner's steps from the same committed inputs on other hardware. In float32 the
        # results agree to 7e-5 of B after four steps (measured); in the reference's bf16 they'd be 4-9% apart.
        deepseek.configure_torch()
        cfg = Config()
        state = deepseek.random_weights(cfg, 0)
        tokens = [tokens_for(cfg, 1, 160, seed=20 + j) for j in range(2)]
        masks = [learned(t, 30 + j) for j, t in enumerate(tokens)]
        cut, _, low, _ = models(cfg, state, 3)
        inputs = [top.precompute(low, cut, t, m)[0] for t, m in zip(tokens, masks)]
        ends = []
        for dev in ("cpu", "cuda"):
            cut, _, _, high = models(cfg, state, 3, device=dev, train_num=top.TRAIN)
            ad = Adapters.fresh(high, cut, seed=7, device=dev)
            ad.attach(high)
            for step in range(1, 5):
                top.train_step(high, cut, ad, [ex.to(dev) for ex in inputs], step, 1e-3)
            ends.append(torch.cat([b.detach().cpu().reshape(-1) for b in ad.b]).double())
        self.assertLess(float((ends[0] - ends[1]).norm() / ends[0].norm()), 1e-3)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class CutTests(unittest.TestCase):
    def test_what_crosses_the_real_cuts(self):
        flash = Config.flash()
        for first in (36, 32, 28, 24):  # 4, 8, 12, 16 layers trained: each starts at an index source
            self.assertEqual(Cut(flash, first).crossing(), {"compress_kv": 20, "index_k": 20, "candidates": 20})
        self.assertEqual(Cut(flash, 37).crossing(), {"compress_kv": 20, "picks": 36})  # 37 reuses 36's picks
        self.assertEqual(Cut(flash, 2).crossing(), {"engram": [14]})  # its table rows would cross
        self.assertEqual(Cut(flash, 1).crossing()["engram"], [1, 14])
        size = Cut(flash, 36).bytes_per_token(seq_len=4096)
        self.assertEqual(size["stream"], 4 * 5120 * 2)
        self.assertEqual((size["compress_kv"], size["index_k"], size["candidates"]), (1024, 256, 0))
        self.assertAlmostEqual(size["total"], 40960 + 16 + 1024 + 256 + 4.125)
        self.assertAlmostEqual(Cut(flash, 36).bytes_per_token(fp8_stream=True)["stream"], 20480 + 640)
        self.assertEqual(Cut(flash, 36).bytes_per_token(seq_len=65536)["candidates"], 4096)  # beyond 16K tokens

    def test_layer_major_refuses_a_trained_kv_source_above_the_cut(self):
        Cut(Config.flash(), 20).check()  # the first trained layer may be one
        with self.assertRaisesRegex(NotImplementedError, "layer 20 is a trained kv source"):
            Cut(Config.flash(), 19).check()


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class RecordTests(unittest.TestCase):
    def test_round_trip_and_root(self):
        cfg = tiny_config()
        cut, _, low, _ = models(cfg, deepseek.random_weights(cfg, 1), 4)
        tokens = tokens_for(cfg, 2, 24)
        ex, frozen = top.precompute(low, cut, tokens, learned(tokens, 1), token_map_for(cfg))
        self.assertEqual(set(frozen.experts), {0, 1, 2, 3})  # the frozen layers' own choices, for spot-checks
        recs = [top.pack(ex, i) for i in range(2)]
        back = [top.unpack(r) for r in recs]
        for i, r in enumerate(back):
            self.assertTrue(torch.equal(r["x"], ex.x[i]))
            self.assertTrue(torch.equal(r["pre"], ex.pre[i]))
            self.assertTrue(torch.equal(r["compress_kv"], ex.side.compress_kv[i]))
            self.assertTrue(torch.equal(r["candidates"].bool(), ex.candidates[i]))
        per_token = (len(recs[0]) - len(recs[0].split(b"\n", 1)[0]) - 1) / 24
        size = cut.bytes_per_token(seq_len=24)
        # pack keeps the candidate mask whole for tiny sequences; the rest is as bytes_per_token counts it
        self.assertAlmostEqual(per_token, size["total"] - size["candidates"] + 24 - 1 / 8 + 1 - 0, delta=1e-9)
        q = top.unpack(top.pack(ex, 0, fp8_stream=True))
        self.assertTrue(torch.equal(q["x"], deepseek.round_fp8(ex.x[0])))
        self.assertEqual(len(top.record_root(recs)), 64)
        self.assertNotEqual(top.record_root(recs), top.record_root(recs[::-1]))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class PeftTests(unittest.TestCase):
    def test_the_adapter_loads_in_peft(self):
        try:
            from peft import PeftModel
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("needs peft and safetensors")
        cfg = Config()
        first = 3
        cut, _, _, high = models(cfg, deepseek.random_weights(cfg, 2), first, Numerics(torch.float64, quantize=False))
        ad = Adapters.fresh(high, cut, rank=4, alpha=8.0, seed=3)
        g = torch.Generator().manual_seed(4)
        ad.b = [torch.randn(b.shape, generator=g) * 0.1 for b in ad.b]  # as if trained
        ad.attach(high)
        # The same projections as plain nn.Linear layers, under the same names: what any model's would be.
        plain = torch.nn.Module()
        plain.layers = torch.nn.ModuleDict()
        mods = high.lora_modules()
        for i in range(cfg.n_layers):
            layer = torch.nn.Module()
            layer.attn = torch.nn.Module()
            for t in deepseek.LORA_TARGETS:
                q = high.lora_modules().get(f"layers.{i}.attn.{t}") if i >= first else None
                w = deepseek.dequant(q.weight, q.scale, torch.float64) if q is not None else torch.zeros(32, 32)
                lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.float64)
                lin.weight.data.copy_(w)
                setattr(layer.attn, t, lin)
            plain.layers[str(i)] = layer
        with tempfile.TemporaryDirectory() as d:
            config = ad.peft_config()
            self.assertEqual(config["layers_to_transform"], [3, 4])
            (Path(d) / "adapter_config.json").write_text(json.dumps(dict(config, task_type=None)))
            save_file({k: v.double() for k, v in ad.peft_state().items()}, str(Path(d) / "adapter_model.safetensors"))
            peft = PeftModel.from_pretrained(plain, d)
        adapted = {n for n, m in peft.named_modules() if n.endswith((".wq_a", ".wq_b", ".wkv", ".wo_b"))
                   and hasattr(m, "lora_A")}
        self.assertEqual({n.split(".", 2)[2] for n in adapted}, set(ad.names))  # these layers only
        for name in ad.names:
            q, lin = mods[name], peft.get_submodule("base_model.model." + name)
            x = torch.randn(3, lin.in_features, generator=g, dtype=torch.float64)
            got, want = lin(x), q(x)
            self.assertLess(float((got - want).abs().max() / want.abs().max()), 1e-12, name)


if __name__ == "__main__":
    unittest.main()
