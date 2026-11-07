"""Evo v2 on Qwen3.5's hybrid layers (vibehash/es_hybrid.py, routed from es_fast): a tiny random Qwen3.5 on the CPU
(the engine computes the model, twists where transformers' modules would, and a member checked alone finds its own
answers exactly greedy) and, with a CUDA GPU and Qwen/Qwen3.5-0.8B in the Hugging Face cache, the same exactness on the
real model in bf16.
"""
import tempfile
import unittest
from pathlib import Path

from vibehash import es

from tests.test_es import HAVE_TORCH, TEMPLATE


def tiny_hybrid(folder: Path, seed: int = 0) -> Path:
    """A random four-layer Qwen3.5 text model (three Gated DeltaNet layers, one gated full attention; twice as many
    value heads as key heads, as the 4B has) with test_es's byte-level tokenizer and chat template."""
    import json

    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    folder.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(folder)
    ids = {s: tok.convert_tokens_to_ids(s) for s in specials}
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=len(vocab) + len(specials), hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=2048, tie_word_embeddings=True,
        eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"])
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():  # norms away from their start, so the model isn't flat
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.3)
    model.save_pretrained(folder)
    (folder / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [ids["<|im_end|>"]], "pad_token_id": ids["<|endoftext|>"]}))
    return folder


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class HybridCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vibehash import es2_rounds, es_local
        from vibehash.es_client import prompt_rows, round_factors, runtime2

        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        folder = tiny_hybrid(Path(cls.tmp.name) / "tiny35", seed=3)
        found = es_local.find(str(folder))
        es_local.register(found)
        cls.spec = es_local.spec2_for(found, "arithmetic", pairs=4, pairs_per_unit=2, problems=5, max_tokens=10,
                                      sigma=0.05, check=2, margin=1e-3, lp_tol=0.02)
        cls.rt = runtime2(cls.spec, "cpu")
        cls.seed = 777
        cls.rows = prompt_rows(cls.rt, es2_rounds.items(cls.spec, cls.seed))
        cls.fac = round_factors(cls.rt, cls.spec, cls.seed)
        cls.members = [(j, s) for j in range(cls.spec.pairs) for s in (1, -1)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_spec_twists_qwen35s_projections(self):
        from vibehash import es_hybrid
        self.assertEqual(self.spec.tensors, es_hybrid.TENSORS)
        names = [n for n, _ in self.rt.linears]
        self.assertEqual(len(names), 3 * 6 + 1 * 7)  # in_proj_qkv, in_proj_z, out_proj + MLP; q/k/v/o + MLP
        self.assertTrue(any(n.endswith("linear_attn.in_proj_qkv") for n in names))
        self.assertFalse(any(n.endswith(("in_proj_a", "in_proj_b")) for n in names))
        self.assertEqual(self.spec.model["family"], "qwen3_5_text")

    def test_the_engine_computes_the_twisted_model(self):
        """A member's answer read by transformers itself (the prompt's shared part untwisted, the rest through
        es_fast.Twist on transformers' modules, Gated DeltaNet's chunked algorithm) is that member's greedy answer
        within float32 rounding, with the log-probabilities the engine gives."""
        import torch
        from vibehash import es_fast

        rt, rows, fac = self.rt, self.rows, self.fac
        member = self.members[3]
        got = es_fast.answer_members(rt, rt.linears, fac, [member], rows, self.spec.max_tokens, self.spec.sigma)[0]
        lps = es_fast.score_items(rt, rt.linears, fac, [member], list(zip(rows, got)), self.spec.sigma)[0]
        sub = {n: (A[[member[0]]], B[[member[0]]]) for n, (A, B) in fac.items()}
        coef = torch.tensor([member[1] * self.spec.sigma])
        for q, (r, a) in enumerate(zip(rows, got)):
            with torch.no_grad():
                out = rt.model(input_ids=torch.tensor([r[:-1]]), use_cache=True)
                with es_fast.Twist(rt.linears, sub, coef):
                    lg = rt.model(input_ids=torch.tensor([[r[-1]] + a[:-1]]), past_key_values=out.past_key_values,
                                  use_cache=True).logits[0].float()
            want = torch.tensor(a)
            gap = (lg.max(-1).values - lg.gather(1, want[:, None])[:, 0]).max()
            self.assertLess(float(gap), 1e-3, f"prompt {q}")
            ref = torch.log_softmax(lg, -1).gather(1, want[:, None])[:, 0].mean()
            self.assertAlmostEqual(float(lps[q]), float(ref), delta=1e-4)

    def test_a_member_checked_alone_rounds_as_it_did_among_the_others(self):
        from vibehash import es_fast

        rt, rows, fac, spec = self.rt, self.rows, self.fac, self.spec
        got = es_fast.answer_members(rt, rt.linears, fac, self.members, rows, spec.max_tokens, spec.sigma)

        def gaps(k, answers):
            return es_fast.gaps_items(rt, rt.linears, fac, [self.members[k]], list(zip(rows, answers)), spec.sigma,
                                      max_tokens=spec.max_tokens)[0].tolist()
        for k in range(len(self.members)):
            self.assertEqual(gaps(k, got[k]), [0.0] * len(rows), f"member {k}")
        differ = [q for q in range(len(rows)) if got[0][q] != got[1][q]]
        self.assertTrue(differ, "the twins answered alike everywhere")
        claimed = gaps(1, got[0])
        self.assertTrue(all(claimed[q] > 0 for q in differ), [claimed[q] for q in differ])
        # Scores: each member alone as among all (the replay fitness a checker re-works out).
        items = list(zip(rows, got[0]))
        every = es_fast.score_items(rt, rt.linears, fac, self.members, items, spec.sigma)
        for k in (0, 5):
            alone = es_fast.score_items(rt, rt.linears, fac, [self.members[k]], items, spec.sigma)[0]
            self.assertEqual(alone.tolist(), every[k].tolist())

    def test_a_rounds_members_checked_together_get_their_gaps_alone(self):
        """es_fast.gaps_members (a checker's whole round at once, each member with its own answers): every member's
        gaps exactly as gaps_items gives them alone, an honest one's 0 and a twin-swapped one's above 0."""
        from vibehash import es_fast

        rt, rows, fac, spec = self.rt, self.rows, self.fac, self.spec
        got = es_fast.answer_members(rt, rt.linears, fac, self.members, rows, spec.max_tokens, spec.sigma)
        claims = [got[k ^ 1] if k in (2, 5) else got[k] for k in range(len(self.members))]  # 2 and 5: the twin's
        together = es_fast.gaps_members(rt, rt.linears, fac, self.members, rows, claims, spec.sigma,
                                        max_tokens=spec.max_tokens)
        for k in range(len(self.members)):
            alone = es_fast.gaps_items(rt, rt.linears, fac, [self.members[k]], list(zip(rows, claims[k])), spec.sigma,
                                       max_tokens=spec.max_tokens)[0]
            self.assertEqual(together[k].tolist(), alone.tolist(), f"member {k}")
        self.assertEqual(float(together[0].max()), 0.0)
        self.assertGreater(float(together[2].max()), 0.0)


def _qwen35_on_cuda() -> bool:
    if not HAVE_TORCH:
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded
        return False
    return True


@unittest.skipUnless(_qwen35_on_cuda(), "needs a CUDA GPU and Qwen/Qwen3.5-0.8B in the Hugging Face cache")
class HybridEngineCheckTests(unittest.TestCase):
    def test_qwen35_members_checked_alone_round_as_among_hundreds(self):
        """Qwen3.5-0.8B in bf16 on CUDA: 32 members x 16 arithmetic prompts answered in one batch (512 rows through
        passes of es_hybrid.rows_per_pass rows), and every member teacher-forced alone (16 rows and padding) finds each
        token exactly its greedy choice; a twin's answers fail where they differ."""
        from vibehash import es2_rounds, es_fast, es_hybrid, es_local
        from vibehash.es_client import prompt_rows, round_factors, runtime2

        found = es_local.find("Qwen/Qwen3.5-0.8B", download=False)
        es_local.register(found)
        spec = es_local.spec2_for(found, "arithmetic", pairs=16, pairs_per_unit=8, problems=16, max_tokens=48)
        rt = runtime2(spec, "cuda")
        self.assertTrue(es_hybrid.is_hybrid(rt.model))
        seed = 4242
        rows = prompt_rows(rt, es2_rounds.items(spec, seed))
        fac = round_factors(rt, spec, seed)
        members = [(j, sign) for u in range(spec.units) for j, _, sign in es.unit_members(spec, seed, u)]
        got = es_fast.answer_members(rt, rt.linears, fac, members, rows, spec.max_tokens, spec.sigma, spec.rank)

        def gaps(k, answers):
            return es_fast.gaps_items(rt, rt.linears, fac, [members[k]], list(zip(rows, answers)), spec.sigma,
                                      spec.rank, max_tokens=spec.max_tokens)[0].tolist()
        for k in (0, 1, 13, len(members) - 1):
            self.assertEqual(gaps(k, got[k]), [0.0] * len(rows), f"member {k}")
        differ = [q for q in range(len(rows)) if got[0][q] != got[1][q]]
        self.assertTrue(differ, "the twins answered alike everywhere")
        claimed = gaps(1, got[0])
        self.assertTrue(all(claimed[q] > 0 for q in differ), [claimed[q] for q in differ])


@unittest.skipUnless(HAVE_TORCH, "needs PyTorch and transformers")
class ExportLayoutTests(unittest.TestCase):
    """`vibehash evo export` of a Qwen3.5 job: Qwen3.5's checkpoints are image-text-to-text models (a vision tower and
    a multi-token-prediction head beside the language model, the text model's dims in text_config), and the workers
    load and train the text model alone. es_export.base_layout puts the export back in the base's own layout."""

    def test_a_multimodal_base_is_exported_whole(self):
        import json

        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        from vibehash import es_export

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        text = tiny_hybrid(root / "text", seed=5)  # a text-only base: nothing to put back
        torch.manual_seed(1)
        tcfg = json.loads((text / "config.json").read_text())
        config = Qwen3_5Config(
            text_config={k: v for k, v in tcfg.items() if k not in ("architectures", "model_type", "transformers_version")},
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=2, patch_size=4,
                               temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=64,
                               num_position_embeddings=16),
            tie_word_embeddings=True)
        base = root / "base"
        Qwen3_5ForConditionalGeneration(config).save_pretrained(base)
        bf = base / "model.safetensors"
        with safe_open(bf, "pt") as fh:  # Qwen's checkpoints carry a multi-token-prediction head transformers skips
            tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        tensors["mtp.fc.weight"] = torch.randn(64, 128)
        save_file(tensors, bf, metadata={"format": "pt"})
        (base / "preprocessor_config.json").write_text('{"patch_size": 4}')
        (base / "LICENSE").write_text("Apache-2.0")
        (base / "README.md").write_text("the base's own model card")

        # what export does: the text model loaded as a worker loads it, a weight trained, saved as transformers saves it
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        q = model.model.layers[3].self_attn.q_proj
        with torch.no_grad():
            q.weight.add_(0.5)
        trained = q.weight.detach().clone()
        out = root / "out"
        model.save_pretrained(out)
        self.assertFalse(es_export.base_layout(text, out, "float32"), "a text-only base has nothing to add")
        self.assertTrue(es_export.base_layout(base, out, "float32"))

        cfg = json.loads((out / "config.json").read_text())
        self.assertIn("vision_config", cfg)
        self.assertEqual(cfg.get("dtype"), "float32")
        self.assertTrue((out / "preprocessor_config.json").is_file() and (out / "LICENSE").is_file())
        self.assertFalse((out / "README.md").exists(), "the base's model card isn't the export's")
        index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(set(index), set(tensors), "every tensor of the base, no more")
        self.assertFalse((out / "model.safetensors").exists(), "a lone file would shadow the index")

        lm = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
        torch.testing.assert_close(lm.model.layers[3].self_attn.q_proj.weight, trained, rtol=0, atol=0)
        full, info = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32, output_loading_info=True)
        self.assertFalse(info["missing_keys"])
        torch.testing.assert_close(full.model.language_model.layers[3].self_attn.q_proj.weight, trained, rtol=0,
                                   atol=0)
        was = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
        for (n, p), (_, b) in zip(full.model.visual.named_parameters(), was.model.visual.named_parameters()):
            torch.testing.assert_close(p, b, rtol=0, atol=0, msg=n)
        with safe_open(out / index["mtp.fc.weight"], "pt") as fh:
            torch.testing.assert_close(fh.get_tensor("mtp.fc.weight"), tensors["mtp.fc.weight"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
