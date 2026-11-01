"""Fine-tuning: LoRA jobs on pretrained models (finetune.py, llm.py).

Everything here runs offline on tiny random checkpoints (tests/finetune_fixtures.py) with CPU PyTorch. Checks against
Hugging Face transformers and PEFT run when those are installed. Real models are opt-in:
    VIBEHASH_REAL_MODELS=1 python -m unittest tests.test_finetune
fetches Qwen3-0.6B (1.5 GB, cached in ~/.vibehash/cache) and compares it with transformers, measures CPU against CUDA
and loads an exported adapter with PEFT.
"""
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from vibehash import finetune as ft
from vibehash.blobs import file_sha256
from vibehash.coordinator import CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import Chain
from vibehash.model import Dataset, ModelSpec
from vibehash.tasks import MAX_SPEC_BYTES, InvalidJob, TrainingJob, load_dataset
from vibehash.verification import rescale_update, reproduces

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tests.finetune_fixtures import CONFIGS, TinyModel
    from vibehash import llm

HAVE_HF = torch is not None and all(importlib.util.find_spec(m) for m in ("transformers", "peft"))
HAVE_TOKENIZERS = all(importlib.util.find_spec(m) for m in ("tokenizers", "jinja2"))
REAL = os.environ.get("VIBEHASH_REAL_MODELS") == "1"
needs_torch = unittest.skipIf(torch is None, "needs PyTorch")
VIBE = 10**8


def tmpdir(test) -> Path:
    d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(d.cleanup)
    return Path(d.name)


def spec_of(**kw) -> ft.LoraSpec:
    """A spec shaped like Qwen3-0.6B, without needing its files."""
    dims = {"family": "qwen3", "vocab": 151936, "hidden": 1024, "layers": 28, "heads": 16, "kv_heads": 8,
            "head_dim": 128, "ffn": 3072, "tied": True}
    return ft.LoraSpec(base="hf:Qwen/Qwen3-0.6B@" + "c" * 40, manifest="a" * 64, dims=dims, **kw)


def job_of(spec=None, **kw) -> TrainingJob:
    args = dict(budget=10 * VIBE, fee_per_round=VIBE, spec=spec or spec_of(), data_source={"kind": "tokens", "sha256": "b" * 64},
                lr=1e-3, batch_size=8, steps_per_unit=16, checkpoint_every=4, tolerance=1e-3, max_update_ratio=1.0)
    return TrainingJob("ft", "", **{**args, **kw})


class SpecTests(unittest.TestCase):
    def test_round_trip(self):
        job = job_of(compute_s=60.0)
        again = TrainingJob.from_spec_json(job.spec_json(), "someone", 5)
        self.assertTrue(again.finetune)
        self.assertEqual(again.spec_json(), job.spec_json())
        self.assertEqual(again.spec, job.spec)
        self.assertLess(len(job.spec_json()), MAX_SPEC_BYTES)
        d = json.loads(job.spec_json())
        self.assertEqual(d["base"], job.spec.base)  # where trainui.parse_job reads it, for the forge's emblem
        self.assertEqual(d["lora"], {"rank": 16, "alpha": 32, "targets": ["q", "k", "v", "o"]})

    def test_mlp_specs_are_unchanged(self):
        """Fine-tunes have a spec of their own; MLP jobs' specs keep exactly the fields they had."""
        mlp = TrainingJob("j", "", 1, 1, spec=ModelSpec(4, (8,), 3, "float32"))
        self.assertFalse(mlp.finetune)
        self.assertEqual(set(json.loads(mlp.spec_json())), {
            "id", "fee_per_round", "model", "precision", "data", "init_seed", "lr", "batch_size", "steps_per_unit",
            "checkpoint_every", "units_per_round", "memory_mb", "tolerance", "max_update_ratio"})

    def test_bad_specs_are_refused(self):
        good = json.loads(job_of().spec_json())
        edits = {
            "unknown field": lambda d: d.update(colour="red"),
            "base not pinned": lambda d: d.update(base="hf:Qwen/Qwen3-0.6B@main"),
            "manifest": lambda d: d.update(manifest="xyz"),
            "family": lambda d: d["model"].update(family="gemma3"),
            "a missing dimension": lambda d: d["model"].pop("ffn"),
            "rank": lambda d: d["lora"].update(rank=0),
            "targets": lambda d: d["lora"].update(targets=["q", "lm_head"]),
            "no targets": lambda d: d["lora"].update(targets=[]),
            "seq_len": lambda d: d.update(seq_len=8),
            "optimizer": lambda d: d.update(optimizer="sgd"),
            "precision": lambda d: d.update(precision="float64"),
            "data kind": lambda d: d.update(data={"kind": "npz", "sha256": "b" * 64}),
            "lr": lambda d: d.update(lr=2.0),
            "segments": lambda d: d.update(checkpoint_every=5),
        }
        for what, edit in edits.items():
            d = copy.deepcopy(good)
            edit(d)
            with self.subTest(what), self.assertRaises(InvalidJob):
                TrainingJob.from_spec_json(json.dumps(d), "", 0)
        with self.assertRaises(InvalidJob):
            job_of(shards=2)

    def test_sizes(self):
        s = spec_of()
        # rank 16 on q, k, v, o of Qwen3-0.6B: 28 layers × 16 × (2048 + 1024 + 1024 + 1024)
        self.assertEqual(s.n_adapter, 28 * 16 * 5120)
        self.assertEqual(s.n_params, 3 * s.n_adapter)
        self.assertEqual([p.stop - p.start for p in s.parts], [s.n_adapter] * 3)
        self.assertGreater(s.flops_per_step(8), s.flops_per_step(4))
        job = job_of()
        self.assertEqual(job.checkpoint_bytes, 4 * s.n_params)
        lengths = job.phase_lengths()
        self.assertGreater(lengths["compute"], job.unit_flops / 5e12)  # a gaming GPU's rate, not the laptop CPU's
        self.assertLess(lengths["compute"], 600)
        self.assertGreater(ft.memory_mb(s, 8, 1_500_000_000), 2000)

    def test_frozen_a_is_drawn_like_peft_and_pinned(self):
        s = spec_of(targets=("q", "o"))
        a = s.lora_a()
        self.assertEqual([x.shape for x in a[:2]], [(16, 1024), (16, 2048)])
        for x, (_, _, inp) in zip(a, s.modules()):
            self.assertLessEqual(np.abs(x).max(), 1 / np.sqrt(inp))
            self.assertGreater(np.abs(x).max(), 0.99 / np.sqrt(inp))
        self.assertEqual(a[0].dtype, np.float32)
        # Every worker must draw the same A; PCG64's raw stream is stable across numpy versions, and this pins it.
        self.assertEqual(hashlib.sha256(np.concatenate([x.ravel() for x in a]).tobytes()).hexdigest()[:16],
                         PINNED_A)
        self.assertFalse(np.array_equal(spec_of(seed=12).lora_a()[0], a[0]))

    def test_adam_bound_holds_for_any_gradients(self):
        """No sequence of gradients moves a weight further than max_step says, and honest-looking ones come close."""
        s = spec_of()
        b1, b2, eps = ft.OPTIMIZERS["adam"]
        rng = np.random.default_rng(0)
        for steps in (1, 4, 16, 64):
            bound, worst = s.max_step(1.0, steps), 0.0
            for trial in range(300):
                kind = trial % 3
                g = (rng.normal(0, 1, steps) * np.logspace(rng.uniform(-8, 0), rng.uniform(-8, 0), steps) if kind == 0
                     else np.r_[np.full(steps - 1, 1e-6), 1.0] if kind == 1 else np.ones(steps))
                m = v = x = 0.0
                for t in range(1, steps + 1):
                    m, v = b1 * m + (1 - b1) * g[t - 1], b2 * v + (1 - b2) * g[t - 1] ** 2
                    x -= (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
                worst = max(worst, abs(x))
            self.assertLessEqual(worst, bound)
            self.assertGreater(worst, 0.95 * steps)
        self.assertAlmostEqual(s.max_step(1.0, 1), 1.0)

    def test_fresh_zeroes_only_the_moments(self):
        s = spec_of(targets=("q",))
        theta = np.arange(s.n_params, dtype=np.float64)
        out = s.fresh(theta)
        np.testing.assert_array_equal(out[s.parts[0]], theta[s.parts[0]])
        self.assertFalse(out[s.n_adapter:].any())
        np.testing.assert_array_equal(s.init_params(), np.zeros(s.n_params))

    def test_peft_config(self):
        c = spec_of(rank=8, alpha=16.0, targets=("q", "v", "down")).peft_config()
        self.assertEqual((c["peft_type"], c["r"], c["lora_alpha"], c["target_modules"]),
                         ("LORA", 8, 16, ["q_proj", "v_proj", "down_proj"]))
        self.assertEqual((c["base_model_name_or_path"], c["revision"]), ("Qwen/Qwen3-0.6B", "c" * 40))


PINNED_A = "13d870abcba58031"  # sha256 of spec_of(targets=("q", "o")).lora_a(), first 16 hex digits


@needs_torch
class DataTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir(self)
        self.tiny = TinyModel(self.dir)
        self.data = self.tiny.dataset()

    def test_round_trip_and_sniffing(self):
        path = self.data.save(self.dir / "d.npz")
        back = load_dataset(path)
        self.assertIsInstance(back, ft.TokenDataset)
        for name in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask"):
            np.testing.assert_array_equal(getattr(back, name), getattr(self.data, name))
        self.assertEqual((back.manifest, back.config), (self.data.manifest, self.data.config))
        back.check(self.tiny.spec())
        mlp = self.dir / "t.npz"
        d = Dataset.synthetic(ModelSpec(), 1, 16, 4)
        np.savez(mlp, x_train=d.x_train, y_train=d.y_train, x_val=d.x_val, y_val=d.y_val)
        self.assertIsInstance(load_dataset(mlp), Dataset)

    def test_batches_are_cut_to_their_longest(self):
        ids, mask = self.data.batch([0, 1])
        self.assertEqual(ids.shape[1], max(self.data.train_len[:2]))
        self.assertEqual(ids.dtype, np.int64)

    def test_malformed_datasets_are_refused(self):
        d = self.data

        def broken(**changes):
            arrays = {k: getattr(d, k).copy() for k in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask")}
            for k, f in changes.items():
                f(arrays[k])
            return ft.TokenDataset(**arrays, manifest=d.manifest, config=d.config)
        cases = {
            "learning position 0": dict(train_mask=lambda m: m.__setitem__((0, 0), 1)),
            "learning past the end": dict(train_mask=lambda m: m.__setitem__((0, -1), 1), train_len=lambda n: n.__setitem__(0, 5)),
            "tokens past the end": dict(train_ids=lambda x: x.__setitem__((0, -1), 7), train_len=lambda n: n.__setitem__(0, 5)),
            "nothing to learn": dict(val_mask=lambda m: m.__setitem__(0, 0)),
            "negative ids": dict(train_ids=lambda x: x.__setitem__((0, 1), -3)),
            "too long": dict(train_len=lambda n: n.__setitem__(0, 999)),
        }
        for what, change in cases.items():
            with self.subTest(what), self.assertRaises(ValueError):
                broken(**change).check_shape()
        with self.assertRaises(ValueError):
            ft.TokenDataset(d.train_ids, d.train_len, d.train_mask, d.val_ids, d.val_len, d.val_mask,
                            manifest=b"{}", config=d.config).check_shape()
        path = self.dir / "wrong-dtype.npz"
        arrays = {k: getattr(d, k) for k in ("train_ids", "train_len", "train_mask", "val_ids", "val_len", "val_mask")}
        np.savez(path, **{**arrays, "train_ids": d.train_ids.astype(np.int64)},
                 manifest=np.frombuffer(d.manifest, np.uint8), config=np.frombuffer(d.config, np.uint8))
        with self.assertRaisesRegex(ValueError, "int32"):
            load_dataset(path)

    def test_dataset_must_suit_the_job(self):
        spec = self.tiny.spec()
        other = TinyModel(self.dir / "other", seed=5)
        with self.assertRaisesRegex(ValueError, "manifest isn't the one"):
            other.dataset().check(spec)
        with self.assertRaisesRegex(ValueError, "longer than"):
            self.data.check(self.tiny.spec(seq_len=16))
        bad = self.tiny.dataset()
        bad.train_ids[0, 1] = self.tiny.vocab
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            bad.check(spec)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.data.check(ft.LoraSpec(spec.base, spec.manifest, {**spec.model, "ffn": 99}, seq_len=32))
        job = job_of(spec)
        path = self.data.save(self.dir / "d.npz")
        with self.assertRaisesRegex(InvalidJob, "hash"):
            job.attach_data(load_dataset(path), "0" * 64)
        job.data_source = {"kind": "tokens", "sha256": file_sha256(path)}
        job.attach_data(load_dataset(path), file_sha256(path))
        with self.assertRaisesRegex(InvalidJob, "token"):
            job.attach_data(Dataset.synthetic(ModelSpec(), 1, 8, 2), file_sha256(path))


@unittest.skipUnless(HAVE_TOKENIZERS, "needs the tokenizers and jinja2 packages")
class ChatTests(unittest.TestCase):
    TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def tokenizer(self, template=TEMPLATE) -> ft.ChatTokenizer:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(["hello there how are you", "I am fine thanks", "user assistant system"] * 20,
                                trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"],
                                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        return ft.ChatTokenizer(tok.to_str().encode(), {"eos_token": "<|im_end|>"}, template)

    def test_only_the_replies_and_their_end_are_learned(self):
        t = self.tokenizer()
        chat = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "I am fine"}, {"role": "user", "content": "how are you"},
                {"role": "assistant", "content": "fine thanks"}]
        ids, mask = t.chat(chat)
        learned = t.tok.decode([i for i, m in zip(ids, mask) if m], skip_special_tokens=False)
        self.assertEqual(learned.replace(" ", ""), "Iamfine<|im_end|>finethanks<|im_end|>")
        self.assertEqual(t.tok.decode(ids, skip_special_tokens=False), t.render(chat))

    def test_a_template_that_changes_the_reply_is_caught(self):
        t = self.tokenizer(self.TEMPLATE.replace("{{ m.content }}", "{{ m.content | upper }}"))
        with self.assertRaisesRegex(ft.DataError, "message 2"):
            t.chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}])

    def test_reading_examples(self):
        d = tmpdir(self)
        good = d / "chats.jsonl"
        good.write_text("\n".join(json.dumps(r) for r in [
            {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello there"}]},
            {"text": "how are you"}, {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "I am fine"}]}]))
        self.assertEqual(len(ft.read_examples(good)), 3)
        bad = d / "data.jsonl"
        for lines, message in ((['{"text": "a"}', '{"text": "b"}', '{"prompt": "c"}'], "Line 3 of data.jsonl has no `text` or `messages`"),
                               (['{"messages": [{"role": "user", "content": "hi"}]}'], "no assistant message"),
                               (["{not json"], "line 1 of data.jsonl isn't JSON")):
            bad.write_text("\n".join(lines))
            with self.subTest(message), self.assertRaisesRegex(ft.DataError, message):
                ft.read_examples(bad)

    @needs_torch
    def test_prepare_holds_some_back_and_sizes_sequences(self):
        d = tmpdir(self)
        tiny = TinyModel(d)
        path = d / "chats.jsonl"
        path.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": "hello there " * (i % 5 + 1)},
                                                         {"role": "assistant", "content": "I am fine " * (i % 7 + 1)}]})
                                  for i in range(40)))
        p = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=1024)
        self.assertEqual((p.examples, p.kept, p.cut, p.chats), (40, 40, 0, 40))
        self.assertEqual(len(p.data.val_len), 4)
        self.assertEqual(p.seq_len % ft.SEQ_STEP, 0)
        self.assertGreaterEqual(p.seq_len, int(p.data.train_len.max()))
        short = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=20)
        self.assertGreater(short.cut, 0)
        self.assertEqual(short.seq_len, 20)
        again = ft.prepare(path, self.tokenizer(), tiny.manifest_bytes, tiny.config_bytes, max_len=1024)
        np.testing.assert_array_equal(again.data.val_ids, p.data.val_ids)  # the split follows the file


@needs_torch
class ModelTests(unittest.TestCase):
    def test_matches_the_numpy_reference_in_float64(self):
        """llm.py and transformer.py are written apart; on a Llama they agree to float64 rounding, loss and LoRA
        gradient both."""
        from vibehash import transformer as tr

        config = {k: v for k, v in CONFIGS["llama"].items() if k != "rope_scaling"} | {"rope_theta": 10000.0}
        tiny = TinyModel(tmpdir(self), "llama", **{k: config[k] for k in ("rope_theta",)}, rope_scaling=None)
        arch = llm.Arch.from_hf_config(tiny.config)
        tensors = {n: _array(*t) for n, t in tiny.tensors.items()}
        rspec = tr.TransformerSpec.from_hf_config(tiny.config, seq_len=12, precision="float64", lora_rank=4,
                                                  lora_alpha=8.0, lora_targets=("q", "v", "down"))
        base = tr.pack(tr.layout(rspec), tensors)
        rng = np.random.default_rng(3)
        adapters = tr.init_adapters(rspec, 5)
        adapters += rng.normal(0, 0.1, adapters.shape) * np.array(["lora_B" in n for n, s in tr.adapter_layout(rspec)
                                                                  for _ in range(int(np.prod(s)))])
        tokens = rng.integers(0, arch.vocab, (3, 13))
        want_loss = tr.loss(rspec, base, adapters, tokens)
        want_grad = tr.unpack(tr.adapter_layout(rspec), tr.gradient(rspec, base, adapters, tokens))

        model = llm.Model(arch, {n: torch.from_numpy(a) for n, a in tensors.items()}, dtype=torch.float64)
        named = tr.unpack(tr.adapter_layout(rspec), adapters)
        ad = {}
        for name in {n.rsplit(".lora_", 1)[0] for n in named}:
            ad[name] = (torch.from_numpy(named[name + ".lora_A.weight"].copy()),
                        torch.from_numpy(named[name + ".lora_B.weight"].copy()).requires_grad_())
        mask = torch.ones(tokens.shape, dtype=torch.uint8)
        mask[:, 0] = 0
        total, count = model.loss(torch.from_numpy(tokens), mask, ad, rspec.scaling)
        (total / count).backward()
        total = total.detach()
        self.assertAlmostEqual((total / count).item(), want_loss, delta=1e-12)
        for name, (_, b) in ad.items():
            np.testing.assert_allclose(b.grad.numpy(), want_grad[name + ".lora_B.weight"], rtol=1e-9, atol=1e-13)

    def test_fused_head_and_loss_match_autograd(self):
        torch.manual_seed(0)
        h = torch.randn(1100, 24, dtype=torch.float64, requires_grad=True)  # more than one chunk
        w = torch.randn(50, 24, dtype=torch.float64)
        t = torch.randint(0, 50, (1100,))
        llm._HeadLoss.apply(h, w, t).backward()
        h2 = h.detach().clone().requires_grad_()
        torch.nn.functional.cross_entropy(h2 @ w.T, t, reduction="sum").backward()
        torch.testing.assert_close(h.grad, h2.grad)

    def test_recomputing_layers_changes_no_bits(self):
        tiny = TinyModel(tmpdir(self)).use(self)
        spec, data = tiny.spec(), tiny.dataset()
        rt = ft.runtime(spec, data, "cpu")
        ids, mask = (torch.from_numpy(x) for x in data.batch(np.arange(8)))
        grads = []
        for recompute in (True, False, True):
            rt.model.recompute = recompute
            b = [torch.full(shape, 0.01).requires_grad_() for _, _, _, shape in rt.lora[spec]]
            total, _ = rt.model.loss(ids, mask, {n: (a, x) for (n, a, _, _), x in zip(rt.lora[spec], b)}, spec.scaling)
            total.backward()
            grads.append(torch.cat([x.grad.ravel() for x in b]))
        self.assertTrue(torch.equal(grads[0], grads[1]) and torch.equal(grads[0], grads[2]))

    def test_unsupported_models_are_refused(self):
        for config, why in (({**CONFIGS["qwen3"], "model_type": "gemma3"}, "gemma3"),
                            ({**CONFIGS["llama"], "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}, "yarn"),
                            ({**CONFIGS["llama"], "hidden_act": "gelu"}, "gelu")):
            with self.subTest(why), self.assertRaisesRegex(ValueError, why):
                llm.Arch.from_hf_config(config)
        arch = llm.Arch.from_hf_config({**CONFIGS["qwen3"], "model_type": "mistral", "sliding_window": 64})
        with self.assertRaisesRegex(ValueError, "sliding window"):
            arch.check(128)
        self.assertEqual(llm.Arch.from_hf_config(CONFIGS["smollm3"]).nope, (3,))


def _array(dtype, shape, raw) -> np.ndarray:
    """bf16 bytes to float64."""
    assert dtype == "BF16"
    return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)


@unittest.skipUnless(HAVE_HF, "needs transformers and peft")
class HuggingFaceTests(unittest.TestCase):
    """The same tiny checkpoints through Hugging Face transformers, and adapters through PEFT."""

    def load_hf(self, folder):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, attn_implementation="eager").eval()

    def test_every_family_matches_transformers(self):
        for family in CONFIGS:
            with self.subTest(family):
                tiny = TinyModel(tmpdir(self), family)
                ours = llm.Model(llm.Arch.from_hf_config(tiny.config),
                                 {n: torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(s)
                                  for n, (_, s, raw) in tiny.tensors.items()})
                tokens = torch.randint(0, tiny.vocab, (2, 96), generator=torch.Generator().manual_seed(1))
                with torch.no_grad():
                    want = self.load_hf(tiny.folder)(tokens).logits
                    got = ours.logits(tokens)
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_llama3_rotary_scaling_matches_transformers(self):
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        config = AutoConfig.for_model(**CONFIGS["llama"])
        inv, _ = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
        cos, _ = llm.rope_table(llm.Arch.from_hf_config(CONFIGS["llama"]), 2)
        theirs = torch.cos(inv.double())
        torch.testing.assert_close(cos[1, :len(inv)], theirs, rtol=1e-6, atol=1e-7)

    def test_an_exported_adapter_loads_in_peft_and_gives_our_outputs(self):
        from peft import PeftModel

        d = tmpdir(self)
        tiny = TinyModel(d).use(self)
        spec, data = tiny.spec(targets=("q", "v", "up")), tiny.dataset()
        theta = ft.train_segment(spec, data, spec.init_params(), 3, 0, 8, 1e-2, 8)
        out = ft.export_peft(spec, theta, d / "adapter", {"job_id": "test"})
        self.assertEqual(json.loads((out / "adapter_config.json").read_text())["target_modules"], ["q_proj", "v_proj", "up_proj"])
        self.assertIn(spec.commit, (out / "README.md").read_text())
        model = PeftModel.from_pretrained(self.load_hf(tiny.folder), out).eval()
        rt = ft.runtime(spec, data, "cpu")
        b = torch.from_numpy(theta[spec.parts[0]].astype(np.float32))
        ad = {n: (a, b[s].view(shape)) for n, a, s, shape in rt.lora[spec]}
        ids = torch.from_numpy(data.batch(np.arange(4))[0])
        with torch.no_grad():
            want, got = model(ids).logits, rt.model.logits(ids, ad, spec.scaling)
            base = rt.model.logits(ids)
        self.assertGreater(float((got - base).abs().max()), 1e-3)  # the adapter does something
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@needs_torch
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tiny = TinyModel(tmpdir(self)).use(self)
        self.spec, self.data = self.tiny.spec(), self.tiny.dataset()

    def test_deterministic_and_resumable(self):
        s, d = self.spec, self.data
        a = ft.train_segment(s, d, s.init_params(), 7, 0, 8, 1e-2, 8)
        np.testing.assert_array_equal(a, ft.train_segment(s, d, s.init_params(), 7, 0, 8, 1e-2, 8))
        half = ft.train_segment(s, d, s.init_params(), 7, 0, 4, 1e-2, 8)
        np.testing.assert_array_equal(a, ft.train_segment(s, d, half, 7, 4, 4, 1e-2, 8))
        self.assertTrue(np.all(a[s.parts[2]] >= 0) and np.any(a[s.parts[1]]))
        self.assertFalse(np.array_equal(a, ft.train_segment(s, d, s.init_params(), 8, 0, 8, 1e-2, 8)))

    def test_averaging_b_averages_the_change_to_the_model(self):
        """With A frozen, the network's average of units' B gives exactly the average of their changes to every weight."""
        s, d = self.spec, self.data
        finals = [ft.train_segment(s, d, s.init_params(), u, 0, 8, 1e-2, 8) for u in range(3)]
        mean_b = np.mean([f[s.parts[0]] for f in finals], axis=0)

        def deltas(b):
            out, i = [], 0
            for (_, rows, _), a in zip(s.modules(), s.lora_a()):
                out.append(s.scaling * b[i:i + rows * s.rank].reshape(rows, s.rank) @ a.astype(np.float64))
                i += rows * s.rank
            return out
        for got, *each in zip(deltas(mean_b), *(deltas(f[s.parts[0]]) for f in finals)):
            np.testing.assert_allclose(got, np.mean(each, axis=0), rtol=1e-12, atol=1e-15)

    def test_training_learns(self):
        s, d = self.tiny.spec(), self.tiny.dataset(task="copy", n_train=64)
        theta = s.init_params()
        before, _ = ft.evaluate(s, d, theta)
        for r in range(6):
            theta = s.fresh(np.mean([ft.train_segment(s, d, theta, 10 * r + u, 0, 16, 1e-2, 8) for u in range(2)], axis=0))
        after, _ = ft.evaluate(s, d, theta)
        self.assertLess(after, before - 0.1)

    def test_checks_catch_doctored_results(self):
        s, d, lr = self.spec, self.data, 1e-2
        start = ft.train_segment(s, d, s.init_params(), 1, 0, 4, lr, 8)
        honest = ft.train_segment(s, d, start, 1, 4, 4, lr, 8)
        self.assertTrue(reproduces(honest, honest, start, 1e-3, s))
        self.assertTrue(s.sane(s.init_params(), honest, lr, 8, 1.0))
        self.assertFalse(s.sane(s.init_params(), s.init_params() + 3 * honest, lr, 8, 1.0), "a boosted update")
        negative = honest.copy()
        negative[s.parts[2].start] = -1e-9
        self.assertFalse(s.sane(s.init_params(), negative, lr, 8, 1.0))
        from vibehash.crypto import Draw
        canary = rescale_update(s, start, honest, Draw(b"t"))
        self.assertFalse(reproduces(canary, honest, start, 1e-3, s))

    def test_moments_are_checked_on_their_own(self):
        """In a real model B grows round after round while the moments stay gradient-sized, so in one norm a worker
        could point its momentum anywhere and pass; part by part, it can't."""
        from vibehash.verification import _reproduces
        s, rng = self.spec, np.random.default_rng(0)
        b, m, v = s.parts
        start = np.zeros(s.n_params)
        start[b] = rng.normal(0, 1e-2, s.n_adapter)
        honest = start.copy()
        honest[b] += rng.normal(0, 1e-3, s.n_adapter)
        honest[m] = rng.normal(0, 1e-6, s.n_adapter)
        honest[v] = honest[m] ** 2 * 10
        doctored = honest.copy()
        doctored[m] = -honest[m]  # momentum the other way
        self.assertTrue(_reproduces(doctored, honest, start, 1e-3, s))
        self.assertFalse(reproduces(doctored, honest, start, 1e-3, s))
        self.assertTrue(reproduces(honest, honest, start, 1e-3, s))


@needs_torch
class ProtocolTests(unittest.TestCase):
    """Fine-tune rounds through the same Coordinator a node runs, with cheaters, then replayed and rechecked."""

    def test_rounds_train_pay_catch_cheats_and_replay(self):
        from vibehash.protocol import Protocol
        from vibehash.replay import replay_block
        from vibehash.worker import HardwareProfile, Worker

        d = tmpdir(self)
        tiny = TinyModel(d).use(self)
        path = tiny.dataset().save(d / "data.npz")
        sha = file_sha256(path)
        behaviors = [("honest", "honest")] * 4 + [("lazy", "honest"), ("freezer", "honest"), ("poisoner", "honest"),
                                                  ("shortcut", "honest")]
        hw = HardwareProfile("gpu", 1e12, 16384, 1.0)
        workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, *b, fp_noise=0) for i, b in enumerate(behaviors)]
        req = KeyPair.generate(b"req")
        chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers})
        proto = Protocol(chain, workers, CoordinatorConfig(), seed=3)
        job = job_of(tiny.spec(), data_source={"kind": "tokens", "sha256": sha}, units_per_round=12, lr=1e-2,
                     memory_mb=2048, budget=100 * VIBE, fee_per_round=10 * VIBE)
        job.attach_data(load_dataset(path), sha)
        proto.post_job(req, job)
        before = copy.deepcopy(proto.chain)
        reports = [proto.run_round() for _ in range(2)]
        by = {w.address: w for w in workers}
        for rep in reports:
            for res in rep.results:
                honest = by[res.worker].behavior == "honest"
                self.assertEqual(res.verdict.value == "accepted", honest, (by[res.worker].behavior, res.verdict))
            self.assertTrue(np.isfinite(rep.val_loss))
            self.assertFalse(rep.params[tiny.spec().n_adapter:].any(), "the next round starts Adam afresh")
        self.assertFalse([s for rep in reports for s in rep.slashes if s[2] in ("approved a canary", "false accusation")])
        for w in workers[4:8]:
            self.assertLess(proto.chain.state.stake(w.address), 10 * VIBE, w.behavior)

        class Recheck:
            def array(self, bid, job):
                return proto.checkpoints.array(bid, job)

            def attach(self, j):
                j.attach_data(load_dataset(path), sha)
        chain, stats = copy.deepcopy(before), {}
        for rep in reports:
            self.assertEqual(replay_block(chain, rep.block, copy.deepcopy(rep.record), Recheck(), stats), [])
            chain.append(rep.block)
        self.assertGreater(stats["rechecked"], 10)
        self.assertEqual(stats["unchecked"], 0)


    def test_canaries_of_fine_tunes_fail_replay(self):
        """A canary copies a real reveal's challenged checkpoints with one segment's update rescaled: an honest checker
        replaying it rejects it, as it accepts the real one."""
        from vibehash.crypto import Draw
        from vibehash.tasks import WorkUnit
        from vibehash.verification import challenge_segments, plant_canary, replay_openings
        from vibehash.worker import HardwareProfile, Worker

        tiny = TinyModel(tmpdir(self)).use(self)
        job = job_of(tiny.spec(), lr=1e-2)
        job._data = tiny.dataset()
        w = Worker("w", KeyPair.generate(b"w"), HardwareProfile("gpu", 1e12, 16384), fp_noise=0)
        start = job.spec.init_params()
        from vibehash.model import params_hash
        unit = WorkUnit.make(job, 1, 0, params_hash(start))
        w.compute(unit, job, start)
        rv = w.reveal(unit, job, b"beacon", 3)
        segs = challenge_segments(b"beacon", unit.unit_id, w.address, job.n_segments, 3)
        real = {i: rv.openings[i] for sg in segs for i in (sg, sg + 1)}
        self.assertEqual(replay_openings(job, unit, real, segs), (True, None))
        canary = plant_canary(job, real, segs, Draw(b"canary"))
        ok, bad = replay_openings(job, unit, canary, segs)
        self.assertFalse(ok)
        self.assertIn(bad, segs)


CHATML = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
WORDS = "hello there how are you I am fine thanks what is the weather like today it is sunny and warm good".split()


def word_tokenizer(vocab: int) -> bytes:
    """tokenizer.json of a word-level tokenizer small enough for a tiny model's vocabulary, with ChatML's tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([" ".join(WORDS + ["user", "assistant", "system"])] * 3,
                            trainers.WordLevelTrainer(vocab_size=vocab, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"]))
    return tok.to_str().encode()


@unittest.skipUnless(torch is not None and HAVE_TOKENIZERS, "needs PyTorch, tokenizers and jinja2")
class LiveNetworkTests(unittest.TestCase):
    """The whole thing over HTTP, offline: a fake Hugging Face (tests/test_hub.py) serving a tiny model and its
    tokenizer, a node and three workers, `vibehash train chats.jsonl --base`, rounds, and `vibehash job download`."""

    def test_post_train_and_download_an_adapter(self):
        import contextlib
        import io
        import logging
        from unittest import mock

        from tests.test_hub import FakeHub
        from tests.test_network import run_until, start_network
        from vibehash import cli
        from vibehash import wallet as wallets

        d = tmpdir(self)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(d / "home"), "HF_HOME": str(d / "hf")})
        env.start()
        self.addCleanup(env.stop)
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(k, None)
        fake = FakeHub()
        self.addCleanup(fake.close)
        os.environ["HF_ENDPOINT"] = fake.endpoint
        logging.disable(logging.INFO)
        self.addCleanup(logging.disable, logging.NOTSET)
        ft.unload()
        self.addCleanup(ft.unload)

        tiny = TinyModel(d / "src")  # its own cache, under another home: the workers fetch from the fake Hub
        files = {p.name: p.read_bytes() for p in tiny.folder.iterdir()}
        files["tokenizer.json"] = word_tokenizer(tiny.vocab)
        files["tokenizer_config.json"] = json.dumps({"chat_template": CHATML, "eos_token": "<|im_end|>"}).encode()
        fake.add_commit("tiny/qwen3", files)
        rng = np.random.default_rng(0)
        chats = d / "chats.jsonl"
        chats.write_text("\n".join(json.dumps({"messages": [
            {"role": "user", "content": " ".join(rng.choice(WORDS, 4))},
            {"role": "assistant", "content": "I am fine " + " ".join(rng.choice(WORDS, 3))}]}) for _ in range(60)))

        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 3)
        wallets.save(req, d / "requester.json", None)

        def run(*argv) -> str:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                try:
                    cli.main(list(argv))
                except SystemExit as e:
                    if e.code not in (None, 0):
                        self.fail(f"vibehash {' '.join(argv)} exited with {e.code!r}\n{out.getvalue()}")
            return out.getvalue()
        said = run("train", str(chats), "--base", "tiny/qwen3", "--node", node.url, "--wallet", str(d / "requester.json"),
                   "--yes", "--plain", "--budget", "100", "--fee", "10", "--units", "6", "--id", "chats-1")
        self.assertIn("Model     tiny/qwen3", said)
        self.assertIn("60 examples (60 chats", said)
        self.assertIn("Posted job chats-1", said)

        run_until(node, clients, stop, 4)  # the job's block and three rounds
        info = api.get("/job/chats-1")
        spec = TrainingJob.from_spec_json(json.dumps(info["spec"]), "", 0).spec
        self.assertEqual(spec.repo, "tiny/qwen3")
        self.assertGreaterEqual(len(info["rounds"]), 2)
        self.assertTrue(all(r["accepted_units"] > 0 and np.isfinite(r["val_loss"]) for r in info["rounds"]))
        self.assertLess(node.chain.state.escrow["chats-1"], 100 * VIBE)
        self.assertTrue(any(node.chain.state.balance(w.address) for w in workers), "workers were paid")

        out = d / "adapter"
        said = run("job", "download", "chats-1", "--node", node.url, "--out", str(out))
        self.assertIn("a LoRA adapter for tiny/qwen3", said)
        from vibehash import hub
        raw = (out / "adapter_model.safetensors").read_bytes()
        header = hub.parse_header(raw[8:8 + int.from_bytes(raw[:8], "little")], len(raw))
        self.assertEqual(len(header.tensors), 2 * len(spec.modules()))
        self.assertIn("base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight", header.tensors)
        self.assertEqual(json.loads((out / "adapter_config.json").read_text())["revision"], spec.commit)


@unittest.skipUnless(REAL and torch is not None, "set VIBEHASH_REAL_MODELS=1 to fetch and check Qwen3-0.6B")
class RealModelTests(unittest.TestCase):
    """Qwen3-0.6B, fetched from Hugging Face through hub.py and checked against its manifest."""
    REPO, COMMIT = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"

    @classmethod
    def setUpClass(cls):
        from vibehash import hub
        cls.mbytes = ft.model_manifest(cls.REPO, cls.COMMIT, hub.Hub())
        cls.manifest = hub.Manifest.from_bytes(cls.mbytes)
        cls.config = hub.fetch_file(cls.manifest, "config.json")
        cls.spec = ft.LoraSpec(base=f"hf:{cls.REPO}@{cls.COMMIT}", manifest=hashlib.sha256(cls.mbytes).hexdigest(),
                               dims=ft.arch_dims(json.loads(cls.config)), seq_len=256)
        rng = np.random.default_rng(0)

        def make(n):
            lens = rng.integers(64, 257, n).astype(np.int32)
            ids = np.zeros((n, 256), np.int32)
            mask = np.zeros((n, 256), np.uint8)
            for i, k in enumerate(lens):
                ids[i, :k] = rng.integers(0, 150000, k)
                mask[i, k // 2:k] = 1
            return ids, lens, mask
        cls.data = ft.TokenDataset(*make(32), *make(8), manifest=cls.mbytes, config=cls.config)
        cls.data.check(cls.spec)

    def test_matches_transformers(self):
        from transformers import AutoModelForCausalLM
        from vibehash import hub

        rt = ft.runtime(self.spec, self.data, "cuda:0" if torch.cuda.is_available() else "cpu")
        paths = hub.fetch_tensors(self.manifest, [n for n in self.manifest.tensors])
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, ignore_errors=True))
        (folder / "config.json").write_bytes(self.config)
        from tests.finetune_fixtures import write_checkpoint
        write_checkpoint(folder, json.loads(self.config), {n: (e.dtype, list(e.shape), paths[n].read_bytes())
                                                           for n, e in self.manifest.tensors.items()})
        ref = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32, attn_implementation="eager").to(rt.device)
        ids = torch.from_numpy(self.data.batch(np.arange(2))[0])
        with torch.no_grad():
            got, want = rt.model.logits(ids), ref(ids.to(rt.device)).logits
        err = float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt())
        print(f"\nQwen3-0.6B vs transformers: {err:.2e} rms relative")
        self.assertLess(err, 1e-5)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "needs CUDA")
    def test_cpu_and_cuda_replays_agree_within_tolerance(self):
        """A segment trained on CUDA, replayed on the CPU, passes; the gap is far inside the job's tolerance and the
        1% rule, and a rescaled (canary) segment fails."""
        from vibehash import compute
        from vibehash.crypto import Draw

        gpu, cpu = compute.TorchBackend("cuda:0"), compute.TorchBackend("cpu")
        s, d, lr = self.spec, self.data, 5e-4
        start = s.round(ft.train_segment(s, d, s.init_params(), 1, 0, 4, lr, 4, gpu))
        t0 = time.time()
        claim = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, gpu))
        again = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, gpu))
        np.testing.assert_array_equal(claim, again)
        replay = s.round(ft.train_segment(s, d, start, 1, 4, 4, lr, 4, cpu))
        print(f"\nCPU replay of 4 steps: {time.time() - t0:.0f} s")
        for name, part in zip("Bmv", s.parts):
            gap = np.linalg.norm(claim[part] - replay[part])
            print(f"  {name}: replay differs by {gap / np.linalg.norm(replay[part]):.1e} of its size, "
                  f"{gap / np.linalg.norm(replay[part] - start[part]):.1e} of the segment's change")
        self.assertTrue(reproduces(claim, replay, start, ft.TOLERANCE, s))
        self.assertFalse(reproduces(rescale_update(s, start, claim, Draw(b"x")), replay, start, ft.TOLERANCE, s))


if __name__ == "__main__":
    unittest.main()
