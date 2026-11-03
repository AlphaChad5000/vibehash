"""One function both agents share: send messages to a hosted chat model, get its reply back.

An LLM API is just an HTTP POST. This uses the "chat completions" shape most providers accept (OpenAI's, and the
OpenAI-compatible endpoint of Meta's Model API and others), so switching providers means changing three settings,
not code:

    LLM_BASE_URL   the provider's base URL, from the hackathon / provider docs (the part before /chat/completions)
    LLM_API_KEY    your key: kept in the environment, never in a file that gets committed
    LLM_MODEL      the model's name as the provider spells it

For Meta's Model API, MODEL_API_KEY (the name Meta's docs use) alone is enough: the URL and model default to
https://api.meta.ai/v1 and muse-spark-1.3 (LLM_MODEL picks another, e.g. a cheaper Contributor tier).

Settings can also go in .env.local (or .env) at the top of the repository, one KEY=value a line; git ignores both.
The environment wins over the file. To check the settings with one short call:

    python agents/llm_call.py

No SDK on purpose, so nothing hides what goes over the wire.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

RETRIES = 5  # rate limits and overloads (429, 5xx): wait and try again, this many times


def load_env_file() -> None:
    """KEY=value lines from .env.local or .env at the repository's top, for whatever the environment doesn't set."""
    for name in (".env.local", ".env"):
        path = Path(__file__).resolve().parent.parent / name
        if path.exists():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip().removeprefix("export ").removeprefix("$env:")  # bash or PowerShell lines too
                key, sep, value = line.partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file()

META = {"LLM_BASE_URL": "https://api.meta.ai/v1", "LLM_MODEL": "muse-spark-1.3"}


def settings() -> dict:
    """The LLM_* settings: a key (LLM_API_KEY, or Meta's MODEL_API_KEY), with Meta's URL and model unless set."""
    s = {k: os.environ.get(k) or META[k] for k in ("LLM_BASE_URL", "LLM_MODEL")}
    s["LLM_API_KEY"] = os.environ.get("LLM_API_KEY") or os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not s["LLM_API_KEY"]:
        raise SystemExit("set MODEL_API_KEY to a Meta key, in .env.local or the environment (see the top of "
                         "agents/llm_call.py); in PowerShell: $env:MODEL_API_KEY = \"...\"")
    return s


def chat(messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2) -> str:
    s = settings()
    base = s["LLM_BASE_URL"].rstrip("/")
    body = {"model": s["LLM_MODEL"], "messages": messages, "temperature": temperature}
    if json_mode:  # ask for one JSON object back; providers that don't support this ignore it or error
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {s['LLM_API_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:  # a local model writing 40 chats takes minutes
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:  # the provider's error body says why (no credits, bad model name, ...)
            if (e.code == 429 or e.code >= 500) and attempt < RETRIES:
                wait = float(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt  # 5, 10, 20, 40, 80 s
                print(f"{base} answered {e.code}; trying again in {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            raise SystemExit(f"{base} answered {e.code} {e.reason}: {e.read().decode(errors='replace')[:500]}")


def chat_json(messages: list[dict]) -> dict:
    """chat(), parsed. Models sometimes wrap the JSON in ``` fences or put a <think> block before it, so parse only
    the outermost {...}."""
    text = chat(messages, json_mode=True).split("</think>")[-1]
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def model_name() -> str:
    """The model the agents are talking to, recorded with each run so the results page names it."""
    return settings()["LLM_MODEL"]


if __name__ == "__main__":
    s = settings()
    print(f"{s['LLM_MODEL']} at {s['LLM_BASE_URL']}:",
          chat_json([{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}]))
