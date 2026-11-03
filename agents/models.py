"""The base models the agents may fine-tune: our short name -> the Hugging Face repo with its weights.

Small models only, so a demo trains on a gaming GPU in minutes. Llama's repo is gated: accept its licence on
Hugging Face and log in (`huggingface-cli login`) before using it.
"""
BASES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def chat_prompt(tok, user: str | list[dict]) -> str:
    """The text a model sees before it replies to `user` (one message, or a conversation of {role, content}
    messages ending with the user's), in its own chat format. Thinking is switched off (Qwen3 would otherwise spend
    its reply budget on a <think> block); templates without that switch ignore it."""
    messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
