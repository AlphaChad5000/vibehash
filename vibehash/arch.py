"""What a pretrained model's config.json says about its shape and arithmetic, without PyTorch.

llm.py computes these models in PyTorch; the node, which runs without it, still needs a model's shape to referee one
step of a fine-tune's training (program.py, referee.py), so the description lives here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

FAMILIES = ("llama", "mistral", "qwen2", "qwen3", "smollm3")
TARGETS = ("q", "k", "v", "o", "gate", "up", "down")  # what LoRA can adapt, in their order within a layer (transformer.py's)
MODULES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
           "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
EMBED, NORM, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
HEAD_CHUNK = 512  # tokens whose logits are made at once: 512 × a 152k vocabulary is 311 MB of float32


@dataclass(frozen=True)
class Arch:
    """What a checkpoint's config.json says about its shape and arithmetic, for the families above. `dims()` is the
    part a job pins in its spec: enough to size the adapters and price the work without fetching anything."""
    family: str
    vocab: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    tied: bool
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: tuple = ()  # Llama 3's: (factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    qkv_bias: bool = False
    qk_norm: bool = False
    nope: tuple[int, ...] = ()  # layers without rotary positions (SmolLM3)
    window: int = 0  # a sliding window over the keys, 0 if none
    max_positions: int = 0  # the longest sequence it was trained for, 0 if the config doesn't say

    @classmethod
    def from_hf_config(cls, config: dict) -> "Arch":
        """The arch of a Hugging Face checkpoint, from its config.json. Refuses what this module doesn't compute
        rather than train a different model under the checkpoint's name."""
        if not isinstance(config, dict):
            raise ValueError("config.json isn't a JSON object")
        family = config.get("model_type")
        if family not in FAMILIES:
            raise ValueError(f"model_type {family!r} isn't supported yet (supported: {', '.join(FAMILIES)})")
        rope = dict(config.get("rope_parameters") or config.get("rope_scaling") or {})
        kind = rope.get("rope_type", rope.get("type", "default"))
        problems = [why for bad, why in [
            (config.get("hidden_act", "silu") != "silu", f"activation {config.get('hidden_act')!r}"),
            (kind not in ("default", "llama3"), f"rotary scaling {kind!r}"),
            (bool(config.get("mlp_bias")), "biases in the MLP"),
            (family not in ("qwen2",) and bool(config.get("attention_bias")), "attention biases"),
        ] if bad]
        if problems:
            raise ValueError(f"not supported yet: {', '.join(problems)}")
        layers, heads = int(config["num_hidden_layers"]), int(config["num_attention_heads"])
        scaling = ()
        if kind == "llama3":
            scaling = (float(rope["factor"]), float(rope["low_freq_factor"]), float(rope["high_freq_factor"]),
                       int(rope["original_max_position_embeddings"]))
        nope = ()
        if family == "smollm3":
            use = config.get("no_rope_layers") or [int((i + 1) % int(config.get("no_rope_layer_interval", 4)) != 0)
                                                  for i in range(layers)]
            nope = tuple(i for i, u in enumerate(use) if not u)
        window = 0
        if config.get("sliding_window") and (family == "mistral" or config.get("use_sliding_window")):
            window = int(config["sliding_window"])
        return cls(family=family, vocab=int(config["vocab_size"]), hidden=int(config["hidden_size"]), layers=layers,
                   heads=heads, kv_heads=int(config.get("num_key_value_heads") or heads),
                   head_dim=int(config.get("head_dim") or int(config["hidden_size"]) // heads),
                   ffn=int(config["intermediate_size"]), tied=bool(config.get("tie_word_embeddings", False)),
                   norm_eps=float(config.get("rms_norm_eps", 1e-6)),
                   rope_theta=float(rope.get("rope_theta", config.get("rope_theta", 10000.0))), rope_scaling=scaling,
                   qkv_bias=family == "qwen2" or bool(config.get("attention_bias")), qk_norm=family == "qwen3",
                   nope=nope, window=window, max_positions=int(config.get("max_position_embeddings") or 0))

    def dims(self) -> dict:
        """What a fine-tune's spec pins about the model, checked against config.json when the weights load."""
        return {k: v for k, v in asdict(self).items()
                if k in ("family", "vocab", "hidden", "layers", "heads", "kv_heads", "head_dim", "ffn", "tied")}

    def shape(self, target: str) -> tuple[int, int]:
        """A projection's weight shape as Hugging Face stores it, [out, in]."""
        d, f, q, kv = self.hidden, self.ffn, self.heads * self.head_dim, self.kv_heads * self.head_dim
        return {"q": (q, d), "k": (kv, d), "v": (kv, d), "o": (d, q), "gate": (f, d), "up": (f, d), "down": (d, f)}[target]

    @property
    def n_weights(self) -> int:
        """Weights in all, the embedding and head included."""
        return self.layers * (sum(math.prod(self.shape(t)) for t in TARGETS) + 2 * self.hidden) + \
            self.vocab * self.hidden * (1 if self.tied else 2) + self.hidden

    def tensor_names(self) -> list[str]:
        """Every tensor the model reads, in Hugging Face's names."""
        names = [EMBED]
        for i in range(self.layers):
            p = f"model.layers.{i}."
            names += [p + MODULES[t] + ".weight" for t in TARGETS]
            names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
            if self.qkv_bias:
                names += [p + MODULES[t] + ".bias" for t in ("q", "k", "v")]
            if self.qk_norm:
                names += [p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"]
        names.append(NORM)
        if not self.tied:
            names.append(HEAD)
        return names

    def check(self, seq_len: int) -> None:
        if self.window and self.window < seq_len:
            raise ValueError(f"its sliding window ({self.window} tokens) is shorter than the sequences ({seq_len}); "
                             "that isn't supported yet")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("query heads must be a multiple of key/value heads, and head_dim even")
