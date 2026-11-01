"""vibehash/deepseek.py: DeepSeek-V4.1-Flash, differentiable and deterministic, checked against DeepSeek's own code.

What's proven where:
- QuantTests: dequantising FP8 and FP4 weights is exact; activations round bit for bit as the reference's kernels
  round them (their stand-ins, tests/deepseek_reference.py); the stored formats pack and unpack.
- EngramTests: the bucket primes give the real model's table sizes.
- GradientTests: autograd's gradients agree with finite differences (a tiny config in float64, every feature on).
- DeterminismTests: forward and backward twice give the same bits, with PyTorch refusing any nondeterministic op.
- ChoiceTests: given choices are used as they are; missing ones are computed and handed back.
- ReferenceTests (needs DeepSeek's reference code, fetched from the Hub, ~100 KB): the small model the reference's
  ModelArgs defaults to, and a small one with every feature on, layer by layer and end to end, from random weights.
- RealLayerTests (opt-in: needs the real layer 2 and the embedding in ~/.vibehash/cache/deepseek, 8.7 GB, and a CUDA
  GPU): the real layer on real token embeddings. scripts/deepseek_bench.py fetches them and times the layer.
"""
import hashlib
import math
import unittest
from dataclasses import replace
from unittest import mock

from tests.test_hub import HubTestCase, sharded_files

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):  # PyTorch before 2.8 lacks the FP4 dtype the reference uses
        raise ImportError
    from vibehash import deepseek
    from vibehash.deepseek import Choices, Config, Model, Numerics, Side
    from tests import deepseek_reference as reference
except ImportError:  # no PyTorch, or too old: nothing here can run
    torch = None

BF16_EPS = 2 ** -8  # bf16's relative rounding step (8 significant bits)


def tiny_config(**kw):
    """Every feature, small enough for finite differences: a pooling (ratio 2) kv source and a layer reusing it, a
    ratio-1 kv source that's also the candidate source, an index source reading another layer's keys inside its
    candidate blocks, an engram layer, YaRN, the SwiGLU clamp and route scaling."""
    cfg = dict(vocab_size=64, dim=64, moe_inter_dim=32, n_layers=5, n_heads=2, n_routed_experts=4,
               n_activated_experts=2, route_scale=1.5, swiglu_limit=1.0, q_lora_rank=32, head_dim=64, rope_head_dim=16,
               o_groups=2, o_lora_rank=32, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
               kv_source_layers=(1, 3), index_source_layers=(1, 3, 4), original_seq_len=8, rope_factor=4,
               index_n_heads=2, index_head_dim=32, index_topk=3, candidate_source_layer=3, candidate_topk_blocks=2,
               candidate_block_size=2, engram_layer_ids=(1,), engram_max_ngram_size=3, engram_vocab_size=11,
               engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=40)
    cfg.update(kw)
    cfg["engram_num_embeddings"] = (1,) * len(cfg["engram_layer_ids"])  # a placeholder, until the primes are known
    cfg = Config(**cfg)
    if not cfg.engram_layer_ids:
        return cfg
    layout = deepseek.EngramLayout.of(cfg)  # the tables' sizes are what their bucket primes add up to, as the real ones
    return replace(cfg, engram_num_embeddings=tuple(layout.rows(i) for i in range(len(cfg.engram_layer_ids))))


def token_map_for(cfg, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.engram_compressed_vocab_size, (cfg.vocab_size,), generator=g)


def tokens_for(cfg, b, s, seed=2):
    return torch.randint(0, cfg.vocab_size, (b, s), generator=torch.Generator().manual_seed(seed))


def model_for(cfg, state, num=None, device="cpu", **kw):
    return Model(cfg, num=num or Numerics(), device=device, **kw).load_weights({k: v.to(device) for k, v in state.items()})


def rel(a, b):
    """max |a - b| over max |b|."""
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def rms(a, b):
    """|a - b| over |b|, as vectors."""
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class QuantTests(unittest.TestCase):
    def test_dequantising_is_exact(self):
        g = torch.Generator().manual_seed(0)
        codes = torch.arange(256, dtype=torch.uint8)
        codes = codes[(codes & 0x7F) != 0x7F].repeat(260)[:64 * 1024]  # every E4M3 code but the two NaNs
        w = codes[torch.randperm(64 * 1024, generator=g)].view(64, 1024).view(torch.float8_e4m3fn)
        scale = (torch.randint(127 - 30, 127 + 10, (2, 32), generator=g, dtype=torch.uint8)).view(torch.float8_e8m0fnu)
        exact = deepseek.dequant_fp8(w, scale, torch.float64)
        for dtype in (torch.bfloat16, torch.float32):
            self.assertTrue(torch.equal(deepseek.dequant_fp8(w, scale, dtype).double(), exact), dtype)
        want = w.double().view(2, 32, 32, 32) * scale.double()[:, None, :, None]  # the definition, in float64
        self.assertTrue(torch.equal(exact, want.view(64, 1024)))
        packed = torch.randint(-128, 128, (64, 512), generator=g, dtype=torch.int8)
        scale4 = torch.randint(127 - 30, 127 + 10, (64, 32), generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        exact = deepseek.dequant_fp4(packed, scale4, torch.float64)
        for dtype in (torch.bfloat16, torch.float32):
            self.assertTrue(torch.equal(deepseek.dequant_fp4(packed, scale4, dtype).double(), exact), dtype)
        # the stand-in's unpacking (convert.py's table, low nibble first) times its scales
        raw = packed.view(torch.uint8).long()
        table = torch.tensor(reference.FP4_TABLE, dtype=torch.float64)
        vals = torch.stack([table[raw & 15], table[raw >> 4]], -1).flatten(-2)
        self.assertTrue(torch.equal(exact, vals * scale4.double().repeat_interleave(32, 1)))

    def test_formats_round_trip(self):
        g = torch.Generator().manual_seed(1)
        w = torch.randn(64, 96, generator=g) * 0.02
        q, s = deepseek.quantize_fp8(w)
        self.assertEqual((q.dtype, s.dtype, tuple(s.shape)), (torch.float8_e4m3fn, torch.float8_e8m0fnu, (2, 3)))
        d = deepseek.dequant_fp8(q, s, torch.float32)
        self.assertLess(rel(d, w), 2 ** -3)  # within E4M3's precision
        self.assertTrue(torch.equal(deepseek.dequant_fp8(*deepseek.quantize_fp8(d), torch.float32), d))
        q, s = deepseek.quantize_fp4(w)
        self.assertEqual((q.dtype, tuple(q.shape), tuple(s.shape)), (torch.int8, (64, 48), (64, 3)))
        d = deepseek.dequant_fp4(q, s, torch.float32)
        self.assertLess(rel(d, w), 0.25)  # E2M1: two significant bits
        self.assertTrue(torch.equal(deepseek.dequant_fp4(*deepseek.quantize_fp4(d), torch.float32), d))

    def test_fp8_cast_rounds_to_nearest_even(self):
        # deepseek.py and the stand-ins both lean on PyTorch's float -> E4M3 cast: check it rounds as CUDA's does.
        vals = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()  # 0 .. 448, ascending
        mid = (vals[:-1] + vals[1:]) / 2
        got = mid.to(torch.float8_e4m3fn).view(torch.uint8).long()
        even = torch.arange(0, 126) + torch.arange(0, 126) % 2  # of the two neighbours, the even code
        self.assertTrue(torch.equal(got, even))

    def activations(self, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(64, 256, generator=g) * torch.logspace(-6, 3, 64)[:, None]
        x[0] = 0  # all zero: the floors on amax
        x[1, :32] = 448 * 2.0 ** torch.arange(-4, 28).float()[:32].clamp(max=8)  # amax at exact powers of two times 448
        x[2, 5] = 6 * 2.0 ** -3  # exactly at FP4's top with a power-of-two scale
        return x.to(torch.bfloat16)

    def test_activations_round_as_the_kernels_do(self):
        for seed in range(3):
            x = self.activations(seed)
            want = reference.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
            self.assertTrue(torch.equal(deepseek.round_fp8(x), want), seed)
            want = reference.fp4_act_quant(x.clone(), 32, True)
            self.assertTrue(torch.equal(deepseek.round_fp4(x), want), seed)
            want = reference.fp4_act_quant(x.clone(), 16, True, scale_dtype=torch.float8_e4m3fn)
            self.assertTrue(torch.equal(deepseek.round_fp4(x, 16, e4m3_scale=True), want), seed)
        # Every FP4 tie: halfway between two E2M1 values, it goes to the even code.
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        self.assertEqual(deepseek._fp4_grid(ties).tolist(), [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        self.assertEqual(reference._cast_fp4(ties).tolist(), [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class EngramTests(unittest.TestCase):
    def test_bucket_primes_give_the_real_table_sizes(self):
        cfg = Config.flash()
        layout = deepseek.EngramLayout.of(cfg)
        self.assertEqual(tuple(layout.rows(i) for i in range(2)), cfg.engram_num_embeddings)
        primes = [p for layer in layout.primes for size in layer for p in size]
        self.assertEqual(len(set(primes)), 48)
        self.assertTrue(all(16_000_000 <= p < 16_001_000 for p in primes))

    def test_primality(self):
        small = [n for n in range(2, 5000) if all(n % d for d in range(2, int(n ** 0.5) + 1))]
        self.assertEqual([n for n in range(5000) if deepseek._is_prime(n)], small)
        self.assertTrue(deepseek._is_prime(2 ** 61 - 1))
        self.assertFalse(deepseek._is_prime(3215031751))  # a strong pseudoprime to bases 2, 3, 5 and 7


def tiny_setup(seed=0, **kw):
    cfg = tiny_config(**kw)
    return cfg, deepseek.random_weights(cfg, seed), tokens_for(cfg, 2, 12), token_map_for(cfg)


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class GradientTests(unittest.TestCase):
    def test_finite_differences(self):
        cfg, state, tokens, tmap = tiny_setup()
        model = model_for(cfg, state, Numerics(torch.float64, quantize=False))
        _, choices = model(tokens, token_map=tmap)  # the choices, then fixed: the rest is smooth
        names = [n for n, _ in model.named_parameters()]
        params = [p.detach().clone().requires_grad_() for _, p in model.named_parameters()]

        def logits(*ps):
            return torch.func.functional_call(model, dict(zip(names, ps)), (tokens,),
                                              {"choices": choices, "token_map": tmap})[0]

        self.assertTrue(torch.autograd.gradcheck(logits, params, fast_mode=True, eps=1e-6, atol=1e-6, rtol=1e-4))
        # The compressed keys' gradient reaches the kv source's compressor from the layer that reuses them.
        loss = logits(*params).square().sum()
        loss.backward()
        grads = dict(zip(names, (p.grad for p in params)))
        self.assertGreater(float(grads["layers.1.attn.compressor.wgate.weight"].abs().sum()), 0)
        self.assertIsNone(grads["layers.1.attn.indexer.weights_proj.weight"])  # picks are discrete: no gradient

    def test_rounding_passes_gradients_straight_through(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
        Numerics().fp8(x).float().sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class DeterminismTests(unittest.TestCase):
    def setUp(self):
        deepseek.configure_torch()
        before = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(lambda: torch.use_deterministic_algorithms(before[0], warn_only=before[1]))
        torch.use_deterministic_algorithms(True, warn_only=False)  # a nondeterministic op raises

    def twice(self, device):
        cfg, state, tokens, tmap = tiny_setup()
        runs = []
        for _ in range(2):
            model = model_for(cfg, state, device=device)
            logits, _ = model(tokens.to(device), token_map=tmap)
            logits.float().square().mean().backward()
            runs.append((logits.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}))
        (l1, g1), (l2, g2) = runs
        self.assertTrue(torch.equal(l1, l2))
        self.assertEqual(g1.keys(), g2.keys())
        self.assertGreater(len(g1), 20)
        for n in g1:
            self.assertTrue(torch.equal(g1[n], g2[n]), n)

    def test_cpu(self):
        self.twice("cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs a CUDA GPU")
    def test_cuda(self):
        self.twice("cuda")


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class ChoiceTests(unittest.TestCase):
    def test_choices_round_trip_and_are_used(self):
        cfg, state, tokens, tmap = tiny_setup()
        model = model_for(cfg, state)
        with torch.no_grad():
            logits, choices = model(tokens, token_map=tmap)
            self.assertEqual(set(choices.experts), set(range(5)))
            self.assertEqual(set(choices.picks), {1, 3, 4})
            self.assertEqual(tuple(choices.candidates.shape), (2, 12, 12))
            again, used = model(tokens, Choices(dict(choices.experts), dict(choices.picks), choices.candidates),
                                token_map=tmap)
            self.assertTrue(torch.equal(logits, again))
            # Another expert for every token of layer 2: used as given, and it changes the output.
            other = (choices.experts[2] + 1) % cfg.n_routed_experts
            given = Choices({**choices.experts, 2: other}, dict(choices.picks), choices.candidates)
            changed, used = model(tokens, given, token_map=tmap)
            self.assertIs(used.experts[2], other)
            self.assertGreater(rel(changed, logits), 1e-3)
            # Other picks for layer 1, which layer 2 reuses.
            picks = choices.picks[1].clone()
            picks[:, -1] = torch.tensor([0, 1, 2])
            changed, used = model(tokens, Choices(dict(choices.experts), {**choices.picks, 1: picks}, choices.candidates),
                                  token_map=tmap)
            self.assertTrue(torch.equal(used.picks[1], picks))
            self.assertFalse(torch.equal(changed[:, -1], logits[:, -1]))
            self.assertTrue(torch.equal(changed[:, :-1], logits[:, :-1]))  # causal: only the last token's changed

    def test_choice_gap(self):
        scores = torch.tensor([[0.1, 0.5, 0.3, 0.9], [1.0, -math.inf, 2.0, 0.0]], dtype=torch.float64)
        self.assertEqual(deepseek.choice_gap(scores, torch.tensor([[3, 1], [2, 0]])).tolist(), [-0.2, -1.0])
        gap = deepseek.choice_gap(scores, torch.tensor([[3, 2], [2, -1]]))
        self.assertAlmostEqual(float(gap[0]), 0.2)
        self.assertEqual(float(gap[1]), math.inf)  # a free slot while position 0 (finite) is left out
        # -1 is padding only where fewer than k options are open: 3 finite scores, 4 slots
        self.assertEqual(float(deepseek.choice_gap(scores[1:], torch.tensor([[2, 0, 3, -1]]))[0]), -math.inf)
        # Naming the top expert twice, or dropping one, is no top-2: it would skip the second expert's work.
        top3 = torch.tensor([[0.9, 0.8, 0.1]] * 5, dtype=torch.float64)
        gap = deepseek.choice_gap(top3, torch.tensor([[0, 1], [0, 0], [0, -1], [0, 3], [-2, 1]]))
        self.assertAlmostEqual(float(gap[0]), -0.7)
        self.assertEqual(gap[1:].tolist(), [math.inf] * 4)

    def test_candidate_gap(self):
        # 8 positions in blocks of 2; the query sees 7, so block 3 (positions 6, 7) is its newest: always kept.
        score = torch.tensor([[[0.5, 0.1, 0.9, 0.2, 0.3, 0.0, 0.4, -math.inf]]], dtype=torch.float64)
        visible = torch.tensor([[7]])
        keep = deepseek.candidate_blocks(score, visible, 2, 2)
        self.assertEqual(keep[0, 0].tolist(), [False, False, True, True, False, False, True, True])
        self.assertLessEqual(float(deepseek.candidate_gap(score, visible, keep, 2, 2)), 0)
        other = torch.tensor([[[True, True, False, False, False, False, True, True]]])  # block 0 (0.5) over 1 (0.9)
        self.assertAlmostEqual(float(deepseek.candidate_gap(score, visible, other, 2, 2)), 0.4)
        for bad in ([True, True, True, True, False, False, False, False],  # drops the pinned block
                    [False, False, True, False, False, False, True, True],  # half a block
                    [True, True, True, True, False, False, True, True],  # three blocks
                    [False, False, False, False, False, False, True, True]):  # an empty slot, open blocks left
            mask = torch.tensor([[bad]])
            self.assertEqual(float(deepseek.candidate_gap(score, visible, mask, 2, 2)), math.inf, bad)

    def test_malformed_choices_are_refused(self):
        cfg, state, tokens, tmap = tiny_setup()
        model = model_for(cfg, state)
        with torch.no_grad():
            _, choices = model(tokens, token_map=tmap)
            experts = choices.experts[2].clone()
            experts[0, 1] = experts[0, 0]  # the top expert twice
            for bad in (experts, choices.experts[2].clamp(max=-1), choices.experts[2] + cfg.n_routed_experts,
                        choices.experts[2][:, :1]):
                with self.assertRaisesRegex(ValueError, "routing"):
                    model(tokens, Choices({**choices.experts, 2: bad}, dict(choices.picks), choices.candidates),
                          token_map=tmap)
            picks = choices.picks[1]  # layer 1: ratio 2, 6 compressed tokens of 12, 3 picks each
            unsorted, dup, unseen, dropped = (picks.clone() for _ in range(4))
            unsorted[:, -1] = torch.tensor([2, 1, 0])
            dup[:, -1] = torch.tensor([0, 0, 1])
            unseen[:, 3] = torch.tensor([0, 1, 2])  # token 3 sees only compressed tokens 0 and 1
            dropped[:, -1] = torch.tensor([0, 1, -1])  # a slot left empty while 6 are visible
            for bad in (unsorted, dup, unseen, dropped):
                with self.assertRaisesRegex(ValueError, "picks"):
                    model(tokens, Choices(dict(choices.experts), {**choices.picks, 1: bad}, choices.candidates),
                          token_map=tmap)

    def test_a_shard_needs_what_its_layers_reuse(self):
        cfg, state, tokens, tmap = tiny_setup()
        model = model_for(cfg, state)
        with torch.no_grad():
            logits, choices = model(tokens, token_map=tmap)
            hashes = deepseek.engram_hashes(tokens, tmap, cfg)
            x, pre = model.start(tokens)
            side = Side()
            for i in (0, 1):
                x, pre = model.layers[str(i)](x, pre, side, choices, hashes[:, :, 0] if i == 1 else None)
            rest = {k: v for k, v in state.items() if not k.startswith(("layers.0.", "layers.1.", "embed."))}
            shard = model_for(cfg, rest, layers=[2, 3, 4], embed=False)
            # Given what layers 0 and 1 handed on, the shard is the rest of the model, to the bit.
            y, ypre = shard.run(x, pre, Side(side.compress_kv, side.index_k, side.picks), choices, hashes)
            self.assertTrue(torch.equal(shard.logits(y, ypre), logits))
            # Without layer 1's compressed keys, layer 2 refuses rather than attend to its window alone.
            with self.assertRaisesRegex(ValueError, "layer 2 reuses layer 1's compress_kv"):
                shard.run(x, pre, Side(), choices, hashes)
            with self.assertRaisesRegex(ValueError, "layer 2 reuses layer 1's picks"):
                shard.run(x, pre, Side(side.compress_kv, side.index_k), choices, hashes)
            # Layer 4 picks with layer 3's compressed keys and index keys, inside layer 3's candidate blocks.
            tail = model_for(cfg, {k: v for k, v in state.items() if k.startswith("layers.4.")}, layers=[4],
                             embed=False, head=False)
            three = Side()
            x3, pre3 = shard.layers["2"](x, pre, Side(side.compress_kv, side.index_k, side.picks), choices)
            x3, pre3 = shard.layers["3"](x3, pre3, three, choices, hashes[:, :, 0])
            with self.assertRaisesRegex(ValueError, "layer 4 reuses layer 3's compress_kv"):
                tail.run(x3, pre3, Side(), Choices(), hashes)
            with self.assertRaisesRegex(ValueError, "layer 4 reuses layer 3's index_k"):
                tail.run(x3, pre3, Side(three.compress_kv), Choices(), hashes)
            with self.assertRaisesRegex(ValueError, "candidate"):
                tail.run(x3, pre3, Side(three.compress_kv, three.index_k), Choices(), hashes)
            own = Choices(candidates=choices.candidates)
            tail.run(x3, pre3, Side(three.compress_kv, three.index_k), own, hashes)
            self.assertTrue(torch.equal(own.picks[4], choices.picks[4]))  # the same picks as the whole model's

    def test_computed_choices_have_no_gap(self):
        cfg, state, tokens, tmap = tiny_setup()
        with torch.no_grad():
            _, choices = model_for(cfg, state)(tokens, Choices(keep_scores=True), token_map=tmap)
        for (kind, layer), scores in choices.scores.items():
            chosen = choices.experts[layer] if kind == "experts" else choices.picks[layer]
            self.assertLessEqual(float(deepseek.choice_gap(scores, chosen).max()), 0, (kind, layer))


def compare_layers(test, cfg, state, tokens, tmap, device="cpu"):
    """Each of deepseek.py's Blocks on the reference's input to that layer, with the reference's side channels and
    choices, against the reference's output. Returns {layer: (stream error, its rms, pre_mix error)} and the
    choices each Block made for itself on the same input, with their scores."""
    r = reference.run(cfg, state, tokens, tmap, device)
    model = model_for(cfg, state, device=device)
    errors, own = {}, {}
    with torch.no_grad():
        for i, block in model.layers.items():
            i = int(i)
            x, pre = r.block_in[i]
            x = r.engram_in.get(i, x)
            given = Choices({i: r.experts[i]}, {i: r.picks[i]} if i in r.picks else {}, r.candidates)
            # The candidate source makes its own blocks; the layers after it pick inside the reference's.
            mine = Choices(keep_scores=True, candidates=None if i == cfg.candidate_source_layer else r.candidates)
            hashes = r.hashes.get(i)
            y, ypre = block(x, pre, reference.side_for(cfg, r, i), given, hashes)
            block(x, pre, reference.side_for(cfg, r, i), mine, hashes)
            errors[i] = rel(y, r.block_out[i][0]), rms(y, r.block_out[i][0]), rel(ypre, r.block_out[i][1])
            own[i] = mine
    return r, model, errors, own


def choice_report(cfg, r, own):
    """{(kind, layer): (fraction of tokens choosing exactly as the reference, the reference's worst gap under our
    scores, relative to the row's largest score)}"""
    out = {}
    for i, mine in own.items():
        for kind, theirs, ours in (("experts", r.experts.get(i), mine.experts.get(i)),
                                   ("picks", r.picks.get(i), mine.picks.get(i))):
            if theirs is None or ours is None:
                continue
            same = (theirs.sort(-1).values == ours.sort(-1).values).all(-1).float().mean()
            scores = mine.scores[(kind, i)].float()
            size = scores.masked_fill(~scores.isfinite(), 0).abs().amax(-1).clamp_min(1e-30)
            gap = deepseek.choice_gap(scores, theirs.to(ours.device)) / size
            out[(kind, i)] = float(same), float(gap.max())
        if i == cfg.candidate_source_layer:
            scores = mine.scores[("picks", i)].float()
            size = scores.masked_fill(~scores.isfinite(), 0).abs().amax(-1).clamp_min(1e-30)
            visible = (torch.arange(1, scores.size(1) + 1, device=scores.device) // cfg.ratio(i)).unsqueeze(-1)
            theirs = r.candidates.to(scores.device)
            gap = deepseek.candidate_gap(scores, visible, theirs, cfg.candidate_topk_blocks,
                                         cfg.candidate_block_size) / size
            out[("candidates", i)] = float((mine.candidates == theirs).all(-1).float().mean()), float(gap.max())
    return out


class ReferenceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("needs PyTorch 2.8 or later")
        import os
        from vibehash.hub import HubError
        try:
            reference.load()
        except (HubError, OSError) as e:
            if os.environ.get("CI"):  # in CI the comparison is the point: an outage mustn't turn it green
                raise
            raise unittest.SkipTest(f"DeepSeek's reference code isn't cached and can't be fetched: {e}")

    # Measured (2026-11-01, CPU and an RTX 4070 SUPER), with about 2x headroom. The yardstick: the reference itself,
    # on the CPU and on CUDA, differs at layer 0 (the same input to both) by 1.4e-3 rms, 1.3e-2 max; this code against
    # it, per layer, by 0.6-2e-3 rms, at most 1.7e-2 max. Per layer is where the proof is: with random weights the
    # model is chaotic, the FP8 and FP4 activation rounding turning one rounding step into whole FP8 steps, so a 1e-3
    # rms nudge after layer 0 moves the final logits 6% rms (with every choice held). End to end, this code against
    # the reference is 5% rms, 7% max: at that floor. A wiring mistake would be O(1).
    LAYER_RMS, LAYER_MAX, PRE_MAX, LOGITS_RMS, LOGITS_MAX = 4e-3, 3e-2, 1e-2, 0.1, 0.15
    # A reference choice may lose to one it left out, under this code's scores on the same block input, by this much
    # relative to the row's best score. Routing is float32, but its input comes after FP8-rounded attention: measured
    # at most 7e-4 (one token in 320). Index scores are bf16: measured 0.
    MARGIN = {"experts": 2e-3, "picks": 2 * BF16_EPS, "candidates": 2 * BF16_EPS}

    def check(self, cfg, state, tokens, tmap=None, device="cpu"):
        r, model, errors, own = compare_layers(self, cfg, state, tokens, tmap, device)
        for i, (e, erms, epre) in errors.items():
            self.assertLess(erms, self.LAYER_RMS, f"layer {i}'s stream, rms")
            self.assertLess(e, self.LAYER_MAX, f"layer {i}'s stream")
            self.assertLess(epre, self.PRE_MAX, f"layer {i}'s pre_mix")
        report = choice_report(cfg, r, own)
        for (kind, i), (same, gap) in report.items():
            # The reference's own choice, scored by this code on the same input: a top-k up to rounding.
            self.assertLessEqual(gap, self.MARGIN[kind], (kind, i, same))
            self.assertGreater(same, 0.99, (kind, i))
        c = cfg.candidate_source_layer
        if 0 <= c < cfg.n_layers:
            self.assertIn(("candidates", c), report)
            # The block choice itself, on the same scores: the reference's select_candidate_blocks, to the bit.
            scores = own[c].scores[("picks", c)]
            visible = (torch.arange(1, scores.size(1) + 1, device=scores.device) // cfg.ratio(c)).unsqueeze(-1)
            want = reference.load().select_candidate_blocks(scores, visible, cfg.candidate_topk_blocks,
                                                            cfg.candidate_block_size)
            got = deepseek.candidate_blocks(scores, visible, cfg.candidate_topk_blocks, cfg.candidate_block_size)
            self.assertTrue(torch.equal(got, want))
            self.assertTrue(torch.equal(got, own[c].candidates))
        with torch.no_grad():
            logits, _ = model(tokens.to(device), Choices(r.experts, r.picks, r.candidates), token_map=tmap)
        self.assertLess(rms(logits, r.logits), self.LOGITS_RMS)
        self.assertLess(rel(logits, r.logits), self.LOGITS_MAX)
        self.assertLess(rel(logits[:, -1], r.last_logits), self.LOGITS_MAX)
        return r, errors, logits


class ReferenceTests(ReferenceCase):
    def test_config_files(self):
        import json
        base = reference.folder()
        hf = json.loads((base / "config.json").read_text())
        inference = json.loads((base / "inference" / "config.json").read_text())
        self.assertEqual(Config.from_hf(hf), Config.flash())
        self.assertEqual(Config.from_dict(inference), Config.flash())
        ref = reference.load()
        args = ref.ModelArgs()
        self.assertEqual(Config(), Config(**{f: getattr(args, f) for f in Config.__dataclass_fields__}))

    def test_default_small_model(self):
        cfg = Config()  # the reference's ModelArgs defaults: a 5-layer V4.1
        state = deepseek.random_weights(cfg, 3)
        self.check(cfg, state, tokens_for(cfg, 2, 256))

    def test_every_feature(self):
        cfg = tiny_config(vocab_size=1000, dim=256, moe_inter_dim=128, n_heads=8, head_dim=128, rope_head_dim=32,
                          q_lora_rank=128, o_lora_rank=64, o_groups=4, n_routed_experts=16, n_activated_experts=4,
                          window_size=32, index_topk=16, index_head_dim=64, index_n_heads=4,
                          candidate_topk_blocks=8, candidate_block_size=4, original_seq_len=64, rope_factor=8,
                          swiglu_limit=2.0, engram_layer_ids=(1, 3), engram_n_heads=3, engram_head_dim=64,
                          engram_vocab_size=50, engram_compressed_vocab_size=500)
        state = deepseek.random_weights(cfg, 4)
        tokens, tmap = tokens_for(cfg, 2, 160), token_map_for(cfg)
        r, _, _ = self.check(cfg, state, tokens, tmap)
        hashes = deepseek.engram_hashes(tokens, tmap, cfg)
        for j, i in enumerate(cfg.engram_layer_ids):
            self.assertTrue(torch.equal(hashes[:, :, j], r.hashes[i]))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs a CUDA GPU")
    def test_default_small_model_on_cuda(self):
        deepseek.configure_torch()
        cfg = Config()
        self.check(cfg, deepseek.random_weights(cfg, 3), tokens_for(cfg, 2, 256), device="cuda")


class FetchTests(HubTestCase):
    """deepseek.fetch_file against tests/test_hub.py's fake Hub and CDN: in pieces, resumed, checked."""

    def setUp(self):
        if torch is None:
            self.skipTest("needs PyTorch 2.8 or later")
        super().setUp()  # also points ~/.vibehash at a temporary folder
        self.commit = self.sharded()
        self.path = "model-00001-of-00002.safetensors"
        self.want = sharded_files()[self.path]
        patch = mock.patch.object(deepseek, "PIECE", 1000)  # a 6 KB file in 7 pieces
        patch.start()
        self.addCleanup(patch.stop)

    def fetch(self, **kw):
        return deepseek.fetch_file(self.path, self.hub, repo="acme/sharded", commit=self.commit, **kw)

    def test_pieces_resume_and_are_checked(self):
        from vibehash.hub import IntegrityError
        self.assertEqual(self.fetch().read_bytes(), self.want)
        self.assertEqual(self.fetch(), deepseek.cache_dir(self.commit) / self.path)  # cached: nothing fetched
        # A restart: pieces 0 and 2 already written; only the rest come down.
        dest = deepseek.cache_dir(self.commit) / self.path
        dest.unlink()
        part, listed = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".done")
        part.write_bytes(self.want[:1000] + bytes(1000) + self.want[2000:3000] + bytes(len(self.want) - 3000))
        listed.write_text("[0, 2]")
        before = self.fake.cdn_bytes
        self.assertEqual(self.fetch(workers=2).read_bytes(), self.want)
        self.assertEqual(self.fake.cdn_bytes - before, len(self.want) - 2000)
        self.assertFalse(part.exists() or listed.exists())
        # A CDN that alters a byte: refused, nothing kept.
        dest.unlink()
        sha = hashlib.sha256(self.want).hexdigest()
        bad = bytearray(self.want)
        bad[-1] ^= 1
        self.fake.lfs[sha] = bytes(bad)
        with self.assertRaises(IntegrityError):
            self.fetch()
        self.assertFalse(dest.exists() or part.exists() or listed.exists())


REAL_FILES = ("model-00005-of-00048.safetensors", "model-00002-of-00048.safetensors")  # layer 2, the embedding


def readme_tokens(n):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / "data" / "deepseek_readme_tokens.json").read_text())["ids"][:n]


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs a CUDA GPU")
class RealLayerTests(ReferenceCase):
    """DeepSeek's real layer 2 (a kv and index source: the pooling compressor, its own indexer keys, YaRN) on real
    token embeddings: README.md's first 2048 tokens, entering as layer 0's input does (four equal copies). 2048
    tokens, so the indexer really chooses: 512 of up to 1024 compressed tokens. Opt-in: the weights must be cached."""
    TOKENS = 2048

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        paths = [deepseek.cache_dir() / f for f in REAL_FILES]
        if not all(p.exists() for p in paths):
            raise unittest.SkipTest("opt-in: fetch the real weights (8.7 GB) with python scripts/deepseek_bench.py "
                                    "--fetch-only")
        deepseek.configure_torch()
        cls.cfg = Config.flash()
        cls.state = deepseek.read_safetensors(paths[0], device="cuda")
        embed = deepseek.read_safetensors(paths[1], {"embed.weight"})["embed.weight"]
        x = embed[torch.tensor(readme_tokens(cls.TOKENS))].cuda()
        cls.x = x[None, :, None].repeat(1, 1, cls.cfg.hc_mult, 1)
        cls.pre = torch.zeros(1, cls.TOKENS, cls.cfg.hc_mult, device="cuda")
        cls.pre[..., 0] = 1

    @classmethod
    def tearDownClass(cls):
        cls.state = cls.x = None
        torch.cuda.empty_cache()

    def test_weights_dequantise_exactly(self):
        n = 0
        for name, w in self.state.items():
            if w.dtype not in (torch.float8_e4m3fn, torch.int8):
                continue
            scale = self.state[name[:-len("weight")] + "scale"]
            self.assertFalse(bool((scale.view(torch.uint8) == 255).any()), name)  # 255 is E8M0's NaN
            exact = deepseek.dequant(w, scale, torch.float32)
            self.assertTrue(bool(exact.isfinite().all()), name)
            self.assertTrue(torch.equal(deepseek.dequant(w, scale, torch.bfloat16).float(), exact), name)
            n += 1
        self.assertEqual(n, 384 * 3 + 3 + 5 + 1)  # experts, the shared expert, attention, the indexer's wq_b

    def test_matches_reference(self):
        cfg = self.cfg
        blk = reference.block(cfg, 2, self.state, "cuda", self.TOKENS)
        y_ref, pre_ref, experts, picks = reference.block_run(blk, self.x, self.pre)
        del blk
        mine = Model(cfg, [2], embed=False, head=False, device="meta").load_weights(self.state).layers["2"]
        with torch.no_grad():
            y, pre = mine(self.x, self.pre, Side(), Choices({2: experts}, {2: picks}))
            own = Choices(keep_scores=True)
            mine(self.x, self.pre, Side(), own)
        print(f"\nreal layer 2, {self.TOKENS} tokens: stream {rms(y, y_ref):.2e} rms, {rel(y, y_ref):.2e} max; "
              f"pre_mix {rel(pre, pre_ref):.2e} max")
        self.assertLess(rms(y, y_ref), self.LAYER_RMS)
        self.assertLess(rel(y, y_ref), self.LAYER_MAX)
        self.assertLess(rel(pre, pre_ref), self.PRE_MAX)
        for kind, theirs, ours in (("experts", experts, own.experts[2]), ("picks", picks, own.picks[2])):
            same = float((theirs.sort(-1).values == ours.sort(-1).values).all(-1).float().mean())
            scores = own.scores[(kind, 2)].float()
            size = scores.masked_fill(~scores.isfinite(), 0).abs().amax(-1).clamp_min(1e-30)
            gap = float((deepseek.choice_gap(scores, theirs) / size).max())
            print(f"  {kind}: {same:.4f} of tokens choose exactly as the reference; its worst relative gap {gap:.1e}")
            self.assertLessEqual(gap, self.MARGIN[kind], kind)
        self.assertGreater(int((picks >= 0).sum(-1).max()), 511)  # the indexer did choose: 512 of up to 1024


if __name__ == "__main__":
    unittest.main()
