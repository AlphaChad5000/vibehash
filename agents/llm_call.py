"""One function both agents share: send messages to a hosted chat model, get its reply back.

An LLM API is just an HTTP POST. This uses the "chat completions" shape most providers accept (OpenAI's, and the
OpenAI-compatible endpoints of Meta's Llama API and others), so switching providers means changing three settings,
not code:

    LLM_BASE_URL   the provider's base URL, from the hackathon / provider docs (the part before /chat/completions)
    LLM_API_KEY    your key: kept in the environment, never in a file that gets committed
    LLM_MODEL      the model's name as the provider spells it

No SDK on purpose, so nothing hides what goes over the wire.
"""
import json
import os
import urllib.request


def chat(messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    body = {"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": temperature}
    if json_mode:  # ask for one JSON object back; providers that don't support this ignore it or error
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.load(r)
    return reply["choices"][0]["message"]["content"]


def chat_json(messages: list[dict]) -> dict:
    """chat(), parsed. Models sometimes wrap JSON in ``` fences, so strip those first."""
    text = chat(messages, json_mode=True).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)
