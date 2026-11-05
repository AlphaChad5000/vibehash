"""The base models the agents may fine-tune: our short name -> the Hugging Face repo with its weights.

Small models only, so a demo trains on a gaming GPU in minutes. Llama's repo is gated: accept its licence on
Hugging Face and log in (`huggingface-cli login`) before using it.
"""
BASES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}

# The exact snapshot of each repo to download, so a later change to it on Hugging Face (or a tampered upload)
# can't silently change what's trained and served. None follows the repo's main branch.
REVISIONS = {
    "qwen3-0.6b": "c1899de289a04d12100db370d81485cdf75e47ca",
    "llama-3.2-1b": None,
}

# Each base's depth and width, from its config.json: `vibehash train` picks its batch (chats a step) by model size
# from these (finetune.settings_for), and the architect needs that batch to work out a budget without downloading.
DIMS = {
    "qwen3-0.6b": {"layers": 28, "hidden": 1024},
    "llama-3.2-1b": {"layers": 16, "hidden": 2048},
}


def source(base: str) -> dict:
    """What to pass to from_pretrained for a base model: its repo and pinned snapshot."""
    return {"pretrained_model_name_or_path": BASES[base], "revision": REVISIONS.get(base)}


def chat_prompt(tok, user: str | list[dict]) -> str:
    """The text a model sees before it replies to `user` (one message, or a conversation of {role, content}
    messages ending with the user's), in its own chat format. Thinking is switched off (Qwen3 would otherwise spend
    its reply budget on a <think> block); templates without that switch ignore it."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
