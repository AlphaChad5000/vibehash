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


if __name__ == "__main__":
    unittest.main()
