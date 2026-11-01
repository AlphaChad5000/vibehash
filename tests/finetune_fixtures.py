"""Tiny pretrained-looking models for fine-tuning tests, with no network: a random checkpoint written as a Hugging Face
repository would have it (config.json, model.safetensors in bf16), its hub.Manifest, the tensors put straight into
the tensor cache under $VIBEHASH_HOME, and token datasets for it. Real models are in the opt-in tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from vibehash import hub
from vibehash.finetune import LoraSpec, TokenDataset, arch_dims

CONFIGS = {
    "qwen3": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 96, "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48,
              "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True, "hidden_act": "silu",
              "max_position_embeddings": 512, "attention_bias": False},
    "llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 80, "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 1, "intermediate_size": 40, "rms_norm_eps": 1e-5,
              "rope_theta": 500000.0, "tie_word_embeddings": False, "hidden_act": "silu",
              "max_position_embeddings": 512,
              "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                               "original_max_position_embeddings": 64}},
    "qwen2": {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 72, "hidden_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 40, "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0, "tie_word_embeddings": True, "hidden_act": "silu", "max_position_embeddings": 512,
              "use_sliding_window": False},
    "smollm3": {"architectures": ["SmolLM3ForCausalLM"], "model_type": "smollm3", "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 4,
                "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 40, "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0, "tie_word_embeddings": True, "hidden_act": "silu",
                "max_position_embeddings": 512, "no_rope_layer_interval": 4, "pad_token_id": 0},
}


def bf16(a: np.ndarray) -> bytes:
    """float32 values rounded to bfloat16 (to nearest, ties to even), as little-endian bytes."""
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bits.astype("<u2").tobytes()


def weights(config: dict, seed: int = 0) -> dict[str, tuple[str, list[int], bytes]]:
    """Random weights shaped like the config's model, at the scale pretrained ones have: matrices N(0, 1/in), norms
    near 1, in bf16, keyed by Hugging Face's names: {name: (dtype, shape, bytes)}."""
    from vibehash.llm import Arch, MODULES, TARGETS

    arch = Arch.from_hf_config(config)
    rng = np.random.default_rng(seed)
    out = {}

    def put(name, shape, scale=None, around=0.0):
        a = rng.normal(around, scale if scale is not None else 1 / np.sqrt(shape[-1]), shape)
        out[name] = ("BF16", list(shape), bf16(a))
    put("model.embed_tokens.weight", (arch.vocab, arch.hidden))
    for i in range(arch.layers):
        p = f"model.layers.{i}."
        for t in TARGETS:
            put(p + MODULES[t] + ".weight", arch.shape(t))
        put(p + "input_layernorm.weight", (arch.hidden,), 0.1, 1.0)
        put(p + "post_attention_layernorm.weight", (arch.hidden,), 0.1, 1.0)
        if arch.qkv_bias:
            for t in ("q", "k", "v"):
                put(p + MODULES[t] + ".bias", (arch.shape(t)[0],), 0.1)
        if arch.qk_norm:
            put(p + "self_attn.q_norm.weight", (arch.head_dim,), 0.1, 1.0)
            put(p + "self_attn.k_norm.weight", (arch.head_dim,), 0.1, 1.0)
    put("model.norm.weight", (arch.hidden,), 0.1, 1.0)
    if not arch.tied:
        put("lm_head.weight", (arch.vocab, arch.hidden))
    return out


def write_checkpoint(folder: Path, config: dict, tensors: dict) -> None:
    """config.json and model.safetensors, as a Hugging Face repository has them."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(folder / "model.safetensors", "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head + b"".join(blobs))


def listing(folder: Path) -> dict[str, hub.FileInfo]:
    """What the Hub's tree API would say about the folder: Git blob ids for small files, LFS sha256 for weights."""
    out = {}
    for p in folder.iterdir():
        data = p.read_bytes()
        if p.suffix == ".safetensors":
            out[p.name] = hub.FileInfo(p.name, len(data), "0" * 40, hashlib.sha256(data).hexdigest())
        else:
            out[p.name] = hub.FileInfo(p.name, len(data), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), None)
    return out


class TinyModel:
    """A tiny checkpoint of `family`, its manifest, and its tensors in the cache under `home` (set $VIBEHASH_HOME to
    it, as `use` does), so finetune.runtime loads it without asking the Hub."""

    def __init__(self, home: Path, family: str = "qwen3", seed: int = 0, **overrides):
        self.home = Path(home)
        self.config = {**CONFIGS[family], **overrides}
        self.folder = self.home / "src" / f"{family}-{seed}"
        self.tensors = weights(self.config, seed)
        write_checkpoint(self.folder, self.config, self.tensors)
        self.commit = hashlib.sha1((self.folder / "model.safetensors").read_bytes()).hexdigest()
        self.repo = f"tiny/{family}"
        self.manifest = hub.build_manifest(self.repo, self.commit, local_dir=self.folder, listing=listing(self.folder))
        self.manifest_bytes = self.manifest.to_bytes()
        self.config_bytes = (self.folder / "config.json").read_bytes()
        root = self.home / "cache" / "hub" / self.commit
        root.mkdir(parents=True, exist_ok=True)
        for name, (_, _, raw) in self.tensors.items():
            (root / self.manifest.tensors[name].sha256).write_bytes(raw)

    def use(self, test) -> "TinyModel":
        """Point $VIBEHASH_HOME at the model's home for the test, and forget models loaded before."""
        from vibehash import finetune

        old = os.environ.get("VIBEHASH_HOME")
        os.environ["VIBEHASH_HOME"] = str(self.home)
        test.addCleanup(lambda: os.environ.pop("VIBEHASH_HOME") if old is None else os.environ.update(VIBEHASH_HOME=old))
        finetune.unload()
        test.addCleanup(finetune.unload)
        return self

    @property
    def vocab(self) -> int:
        return self.config["vocab_size"]

    def spec(self, seq_len: int = 32, **kw) -> LoraSpec:
        return LoraSpec(base=f"hf:{self.repo}@{self.commit}", manifest=hashlib.sha256(self.manifest_bytes).hexdigest(),
                        dims=arch_dims(self.config), seq_len=seq_len, **kw)

    def dataset(self, n_train: int = 48, n_val: int = 8, seq_len: int = 32, seed: int = 1, task: str = "copy") -> TokenDataset:
        """Sequences with something learnable. "copy": a random prompt, a separator, then the prompt again, learning
        only the copy (the reply). Lengths vary, so batches are padded and cut."""
        rng = np.random.default_rng(seed)
        v = self.vocab

        def make(n):
            ids = np.zeros((n, seq_len), dtype=np.int32)
            mask = np.zeros((n, seq_len), dtype=np.uint8)
            lens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                k = int(rng.integers(3, (seq_len - 1) // 2 + 1))
                prompt = rng.integers(2, v, k)
                row = np.concatenate([prompt, [1], prompt]) if task == "copy" else rng.integers(2, v, 2 * k + 1)
                ids[i, :len(row)], lens[i] = row, len(row)
                mask[i, k + 1:len(row)] = 1
            return ids, lens, mask
        return TokenDataset(*make(n_train), *make(n_val), manifest=self.manifest_bytes, config=self.config_bytes)
