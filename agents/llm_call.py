"""One function both agents share: send messages to a hosted chat model, get its reply back.

An LLM API is just an HTTP POST to `<base URL>/chat/completions`, in the shape OpenAI made and most providers copy.
Meta's Model API speaks it, and it's the default here: Muse Spark at https://api.meta.ai/v1, with a bearer key.

    MODEL_API_KEY   your Meta key, from the API keys tab of the Model API dashboard at https://dev.meta.ai (the name
                    Meta's own docs use; META_API_KEY, LiteLLM's name, works too). That's all Muse Spark needs.
    LLM_MODEL       another model: muse-spark-1.3 is the default, the one Meta's docs recommend for new work (1.1
                    and 1.2 cost the same).

Any other OpenAI-compatible provider works by setting all three of these, which also override the Meta defaults:

    LLM_BASE_URL    the provider's base URL, the part before /chat/completions
    LLM_API_KEY     its key (a Meta key is only ever sent to Meta)
    LLM_MODEL       the model's name as the provider spells it

and LLM_REASONING_EFFORT (minimal, low, medium, high, ...) is sent as `reasoning_effort` when it's set. Keys stay in
the environment, never in a file that gets committed, and never in anything this prints or raises.

Replies from reasoning models are messy, so `chat_json` takes the first complete JSON object out of whatever comes
back (prose, ``` fences, a <think> block first). A provider that refuses `response_format` or `temperature` gets the
request again without it; 429s and 5xx answers are retried a few times with backoff. Everything else becomes an
LLMError whose message says what to do. No SDK on purpose, so nothing hides what goes over the wire.
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

META_BASE = "https://api.meta.ai/v1"
META_MODEL = "muse-spark-1.3"
META_KEYS = ("MODEL_API_KEY", "META_API_KEY")  # Meta's docs' name, then LiteLLM's
TIMEOUT_S = 600  # a reasoning model writing 60 examples can think for minutes before it sends a byte
RETRIES = 3  # after the first try, for 429 and 5xx (Meta's docs: 3-5 attempts, exponential backoff with jitter)
MAX_WAIT_S = 60.0  # the longest a Retry-After is honoured for
OPTIONAL = ("response_format", "temperature", "reasoning_effort")  # dropped, once each, if the provider refuses them
_sleep = time.sleep  # tests make waiting instant


class LLMError(Exception):
    """The model couldn't be asked, or its answer can't be used. The message says why, and never holds the key."""


class NotJSON(LLMError):
    """The model answered, twice, without a JSON object: this answer is unusable, but the API itself works."""


def settings(env=None) -> dict:
    """Where requests go: {"base_url", "model", "key", "provider"}. LLM_* win; with no LLM_BASE_URL it's Meta's Model
    API, keyed by MODEL_API_KEY (or META_API_KEY), and Muse Spark unless LLM_MODEL says otherwise."""
    env = os.environ if env is None else env
    base = (env.get("LLM_BASE_URL") or META_BASE).strip().rstrip("/")
    if base.endswith("/chat/completions"):  # the whole endpoint, pasted from a provider's docs
        base = base[:-len("/chat/completions")]
    meta = urllib.parse.urlsplit(base).hostname == urllib.parse.urlsplit(META_BASE).hostname
    key = env.get("LLM_API_KEY") or (next((env[k] for k in META_KEYS if env.get(k)), "") if meta else "")
    model = env.get("LLM_MODEL") or (META_MODEL if meta else "")
    return {"base_url": base, "model": model, "key": key.strip(), "provider": "meta" if meta else "openai-compatible"}


def describe(env=None) -> dict:
    """What a result should record about the model that made it: the provider and model, never the key."""
    s = settings(env)
    return {"provider": s["provider"], "base_url": s["base_url"], "model": s["model"]}


def _problems(s: dict) -> list[str]:
    out = []
    parts = urllib.parse.urlsplit(s["base_url"])
    if parts.scheme not in ("http", "https") or not parts.hostname:
        out.append(f"LLM_BASE_URL {s['base_url']!r} isn't an http(s) address")
    elif parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1") and s["key"]:
        out.append(f"LLM_BASE_URL is plain http to {parts.hostname}: the key would cross the internet unencrypted "
                   "(use https)")
    if s["provider"] == "meta" and not s["key"]:
        out.append("no Meta API key: make one in the API keys tab at https://dev.meta.ai, then set MODEL_API_KEY "
                   "(PowerShell: $env:MODEL_API_KEY = \"...\"; bash: export MODEL_API_KEY=...). agents/README.md "
                   "has the steps")
    if not s["model"]:
        out.append("set LLM_MODEL to the model's name as your provider spells it")
    return out


def _redact(text: str, key: str) -> str:
    """An error body with the key taken out, in case a provider echoes it back."""
    return text.replace(key, "***") if key and len(key) >= 4 else text


def _error_detail(body: bytes) -> tuple[str, str | None, str | None]:
    """(message, type or code, param) from an OpenAI-style error body, else the start of the body as text."""
    text = body.decode("utf-8", errors="replace")
    try:
        err = json.loads(text).get("error")
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict):
        return str(err.get("message") or text[:300]), err.get("code") or err.get("type"), err.get("param")
    if isinstance(err, str):
        return err, None, None
    return text.strip()[:300] or "(no details)", None, None


def _advice(status: int, message: str, code, s: dict) -> str:
    """What a failed request means, in plain words: Meta's documented codes, which other providers mostly share."""
    where, model = s["base_url"], s["model"]
    low = f"{message} {code or ''}".lower()
    if status == 401:
        return (f"{where} refused the key (401): it's missing, mistyped or revoked. Make a new one at "
                "https://dev.meta.ai and set MODEL_API_KEY (or LLM_API_KEY for another provider)")
    if status == 402 or "billing" in low or "insufficient" in low or "credit" in low:
        return (f"{where} says the account can't pay ({status}): add a payment method or credits to the account "
                "the key belongs to")
    if status == 403:
        return (f"{where} won't let this key use {model} (403). Meta's Model API is a public preview whose "
                "availability depends on the country you call from (US developers first), and team settings can "
                "block a model too")
    if status == 404 or "model_not_found" in low:
        return (f"{where} has no model called {model!r}, or no /chat/completions (404): check LLM_MODEL against the "
                "provider's model list (Meta's: muse-spark-1.1, muse-spark-1.2, muse-spark-1.3), and LLM_BASE_URL")
    if status == 429:
        return (f"{where} is rate limiting this key (429) and kept doing so after {RETRIES} retries: wait a minute "
                "and run it again")
    if status == 504:
        return (f"{where} timed out making the reply (504). Ask for less (fewer examples), or set "
                "LLM_REASONING_EFFORT=low")
    if status >= 500:
        return f"{where} kept failing ({status}) after {RETRIES} retries: the provider is having trouble; try later"
    return f"{where} refused the request ({status})"


def _retry_after(headers) -> float | None:
    try:
        return max(0.0, min(MAX_WAIT_S, float(headers.get("Retry-After"))))
    except (TypeError, ValueError):
        return None


def chat(messages: list[dict], *, json_mode: bool = False, temperature: float | None = 0.2, env=None) -> str:
    """The model's reply to `messages` ({"role", "content"} dicts), as text. LLMError if it can't be had."""
    s = settings(env)
    if problems := _problems(s):
        raise LLMError("; ".join(problems))
    env = os.environ if env is None else env
    body = {"model": s["model"], "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode:  # one JSON object back; Meta's schema lists json_object as its older JSON mode
        body["response_format"] = {"type": "json_object"}
    if env.get("LLM_REASONING_EFFORT"):
        body["reasoning_effort"] = env["LLM_REASONING_EFFORT"]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if s["key"]:
        headers["Authorization"] = f"Bearer {s['key']}"
    url, waits, dropped = f"{s['base_url']}/chat/completions", 0, set()
    while True:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            detail = e.read()
            e.close()
            message, code, param = _error_detail(detail)
            message = _redact(message, s["key"])
            if e.code == 400:  # a parameter this provider doesn't take: send it again without, once
                text = f"{message} {param or ''}".lower()
                drop = next((p for p in OPTIONAL if p in body and p not in dropped
                             and (param == p or p in text or (p == "response_format" and "json" in text))), None)
                if drop:
                    dropped.add(drop)
                    del body[drop]
                    continue
            if (e.code == 429 or (e.code >= 500 and e.code != 504)) and waits < RETRIES:
                wait = _retry_after(e.headers)
                _sleep(wait if wait is not None else min(MAX_WAIT_S, 2 ** waits + random.random()))
                waits += 1
                continue
            raise LLMError(f"{_advice(e.code, message, code, s)}. It said: {message}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (socket.timeout, TimeoutError)) or isinstance(e, (socket.timeout, TimeoutError)):
                raise LLMError(f"{s['base_url']} sent nothing back for {TIMEOUT_S} s; try again, ask for less, or set "
                               "LLM_REASONING_EFFORT=low") from None
            if waits < 1:  # a dropped connection: once more
                waits += 1
                _sleep(1.0)
                continue
            raise LLMError(f"can't reach {s['base_url']}: {_redact(str(reason), s['key'])}. Check the address and "
                           "your connection") from None
    try:
        reply = json.loads(raw)
        choice = reply["choices"][0]
        message = choice.get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise LLMError(f"{s['base_url']} answered with something that isn't a chat completion: "
                       f"{_redact(raw.decode('utf-8', errors='replace')[:200], s['key'])!r}") from None
    content = message.get("content")
    if isinstance(content, list):  # content parts, as some providers send them
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str) or not content.strip():
        why = choice.get("finish_reason")
        if message.get("refusal"):
            raise LLMError(f"{s['model']} refused: {message['refusal']}")
        if why == "length":
            raise LLMError(f"{s['model']} spent its whole token allowance before answering (much of it thinking): "
                           "ask for less, or set LLM_REASONING_EFFORT=low")
        raise LLMError(f"{s['model']} sent an empty reply (finish_reason {why!r})")
    if choice.get("finish_reason") == "length" and json_mode:
        raise LLMError(f"{s['model']}'s reply was cut off at its token limit, so its JSON is incomplete: ask for "
                       "fewer examples, or set LLM_REASONING_EFFORT=low")
    return content


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def first_json_object(text: str) -> dict:
    """The first complete JSON object in a reply: bare, in ``` fences, or in prose, after any <think> block. ValueError
    if there's none."""
    text = _THINK.sub("", text)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in the reply")


def chat_json(messages: list[dict], *, temperature: float | None = 0.2, env=None) -> dict:
    """chat(), asking for JSON and parsed. One more try, saying what was wrong, if the reply holds no JSON object."""
    text = chat(messages, json_mode=True, temperature=temperature, env=env)
    try:
        return first_json_object(text)
    except ValueError:
        pass
    text = chat(messages + [{"role": "assistant", "content": text},
                            {"role": "user", "content": "That had no JSON object in it. Reply with only the JSON "
                                                        "object, nothing else."}],
                json_mode=True, temperature=temperature, env=env)
    try:
        return first_json_object(text)
    except ValueError:
        raise NotJSON(f"{settings(env)['model']} didn't reply with JSON, twice. It began: {text[:200]!r}") from None
