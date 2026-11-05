"""The Muse agents (agents/): the Job Architect, the Quality Judge, the rules they score with, and the LLM client.

Offline, on the CPU, with no API key: a fake OpenAI-compatible server on 127.0.0.1 plays Muse Spark, answering from a
script (plans, fenced and prose-wrapped JSON, refusals of response_format and temperature, rate limits, bad keys).
The architect's network files go through `vibehash train` itself up to its pre-flight, against tests/test_hub.py's
fake Hugging Face, where PyTorch, tokenizers and jinja2 are installed (CI has them).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

from agents import architect, checks, judge, llm_call
from vibehash import cli
from vibehash import finetune as ft
from vibehash.tasks import TrainingJob

KEY = "sk-test-NEVER-SHOWN-7f3a9c"
MODEL = "fake-muse"
HAVE = {m: importlib.util.find_spec(m) is not None for m in ("torch", "tokenizers", "jinja2", "transformers", "peft")}
QWEN3ISH = ("{% for m in messages %}{% if m.role == 'assistant' %}<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "{{ m.content }}<|im_end|>\n{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}"
            "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
WORDS = "how do i fix my what is the a an to for it you can try please thanks short friendly answer question".split()


def word_tokenizer(vocab: int) -> str:
    """tokenizer.json of a word-level tokenizer with exactly `vocab` ids (so any id a tiny model makes decodes to a
    word), with ChatML's and Qwen3's think tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([" ".join(WORDS + [f"w{i}" for i in range(vocab)])] * 3, trainers.WordLevelTrainer(
        vocab_size=vocab, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]))
    return tok.to_str()


def tmpdir(test) -> Path:
    d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(d.cleanup)
    return Path(d.name)


def completion(content, finish="stop", **message) -> dict:
    return {"id": "c1", "object": "chat.completion", "model": MODEL, "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, **message}}]}


def ok(content) -> tuple:
    return 200, completion(content if isinstance(content, str) else json.dumps(content)), {}


def refused(status, message, param=None, kind="invalid_request_error", code=None, headers=None) -> tuple:
    return status, {"error": {"message": message, "type": kind, "param": param, "code": code}}, headers or {}


class FakeLLM:
    """An OpenAI-compatible POST /v1/chat/completions on 127.0.0.1. Each request takes the next entry of `script`:
    (status, body, headers), or a function of the request body that returns one. Every request is kept."""

    def __init__(self, test):
        self.script: list = []
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                step = fake.script.pop(0) if fake.script else refused(500, "the test's script ran out", kind="server_error")
                status, payload, headers = step(body) if callable(step) else step
                data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, str(v))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        test.addCleanup(self.server.server_close)
        test.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def env(self, **extra) -> dict:
        return {"LLM_BASE_URL": self.url, "LLM_API_KEY": KEY, "LLM_MODEL": MODEL, **extra}


class AgentTest(unittest.TestCase):
    """A clean environment: no LLM or Hugging Face settings from this machine, VIBEHASH_HOME in a temporary folder,
    and backoff waits recorded instead of slept."""

    def setUp(self):
        self.dir = tmpdir(self)
        env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.dir / "home")})
        env.start()
        self.addCleanup(env.stop)
        for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_REASONING_EFFORT", *llm_call.META_KEYS, "HF_TOKEN",
                  "HUGGING_FACE_HUB_TOKEN", "VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(k, None)
        self.waits: list[float] = []
        sleep = mock.patch.object(llm_call, "_sleep", self.waits.append)
        sleep.start()
        self.addCleanup(sleep.stop)
        self.fake = FakeLLM(self)

    def use_fake(self, **extra):
        os.environ.update(self.fake.env(**extra))


class LLMCallTests(AgentTest):
    def test_a_meta_key_alone_means_muse_spark(self):
        for name in llm_call.META_KEYS:
            with self.subTest(name):
                s = llm_call.settings({name: "k1"})
                self.assertEqual((s["base_url"], s["model"], s["key"], s["provider"]),
                                 ("https://api.meta.ai/v1", "muse-spark-1.3", "k1", "meta"))
        s = llm_call.settings({"MODEL_API_KEY": "k1", "LLM_MODEL": "muse-spark-1.1", "LLM_API_KEY": "k2"})
        self.assertEqual((s["model"], s["key"]), ("muse-spark-1.1", "k2"))  # LLM_* still win
        s = llm_call.settings({"MODEL_API_KEY": "k1", "LLM_BASE_URL": "https://api.meta.ai/v1/chat/completions/"})
        self.assertEqual((s["base_url"], s["key"], s["provider"]), ("https://api.meta.ai/v1", "k1", "meta"))
        self.assertNotIn("key", llm_call.describe({"MODEL_API_KEY": "k1"}))

    def test_a_meta_key_never_goes_to_another_provider(self):
        self.assertEqual(llm_call.settings({"MODEL_API_KEY": "k1", "LLM_BASE_URL": "https://other.example/v1"})["key"], "")
        os.environ.update({"LLM_BASE_URL": self.fake.url, "LLM_MODEL": MODEL, "MODEL_API_KEY": KEY})
        self.fake.script = [ok("{}")]
        llm_call.chat_json([{"role": "user", "content": "hi"}])
        self.assertNotIn("Authorization", self.fake.requests[0]["headers"])

    def test_no_key_or_model_says_what_to_set(self):
        with self.assertRaisesRegex(llm_call.LLMError, "dev.meta.ai.*MODEL_API_KEY"):
            llm_call.chat([{"role": "user", "content": "hi"}], env={})
        with self.assertRaisesRegex(llm_call.LLMError, "LLM_MODEL"):
            llm_call.chat([{"role": "user", "content": "hi"}], env={"LLM_BASE_URL": "https://other.example/v1"})
        with self.assertRaisesRegex(llm_call.LLMError, "unencrypted"):
            llm_call.chat([{"role": "user", "content": "hi"}],
                          env={"LLM_BASE_URL": "http://other.example/v1", "LLM_API_KEY": KEY, "LLM_MODEL": MODEL})
        self.assertEqual(self.fake.requests, [])

    def test_the_request_is_a_chat_completion(self):
        self.use_fake(LLM_REASONING_EFFORT="low")
        self.fake.script = [ok({"a": 1})]
        self.assertEqual(llm_call.chat_json([{"role": "user", "content": "hi"}]), {"a": 1})
        r = self.fake.requests[0]
        self.assertEqual(r["path"], "/v1/chat/completions")
        self.assertEqual(r["headers"]["Authorization"], f"Bearer {KEY}")
        self.assertEqual(r["body"], {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2,
                                     "response_format": {"type": "json_object"}, "reasoning_effort": "low"})

    def test_json_in_fences_prose_or_after_thinking(self):
        self.use_fake()
        want = {"winner": "A", "reason": "it {really} is"}
        for reply in (f"```json\n{json.dumps(want, indent=2)}\n```",
                      f"Sure! Here's my verdict: {json.dumps(want)} Hope that helps {{not json}}.",
                      f"{{broken {json.dumps(want)}",
                      f"<think>Maybe {{\"winner\": \"B\"}}? No.</think>\n{json.dumps(want)}"):
            with self.subTest(reply[:20]):
                self.fake.script = [ok(reply)]
                self.assertEqual(llm_call.chat_json([{"role": "user", "content": "x"}]), want)

    def test_a_reply_without_json_is_asked_again_once(self):
        self.use_fake()
        self.fake.script = [ok("I think A is better."), ok('{"winner": "A"}')]
        self.assertEqual(llm_call.chat_json([{"role": "user", "content": "x"}]), {"winner": "A"})
        self.assertIn("only the JSON object", self.fake.requests[1]["body"]["messages"][-1]["content"])
        self.fake.script = [ok("A."), ok("Still A.")]
        with self.assertRaises(llm_call.NotJSON):
            llm_call.chat_json([{"role": "user", "content": "x"}])

    def test_refused_response_format_then_temperature_are_dropped(self):
        self.use_fake()
        self.fake.script = [refused(400, "response_format of type json_object is not supported for this model",
                                    param="response_format"),
                            refused(400, "Unsupported value: 'temperature' does not support 0.2 with this model."),
                            ok('{"done": true}')]
        self.assertEqual(llm_call.chat_json([{"role": "user", "content": "x"}]), {"done": True})
        bodies = [r["body"] for r in self.fake.requests]
        self.assertEqual([("response_format" in b, "temperature" in b) for b in bodies],
                         [(True, True), (False, True), (False, False)])
        self.fake.script = [refused(400, "messages must not be empty", param="messages")]  # not ours to drop: an error
        with self.assertRaisesRegex(llm_call.LLMError, r"refused the request \(400\).*must not be empty"):
            llm_call.chat_json([{"role": "user", "content": "x"}])

    def test_rate_limits_and_server_errors_are_retried_with_backoff(self):
        self.use_fake()
        self.fake.script = [refused(429, "slow down", kind="rate_limit_error", code="rate_limit_exceeded",
                                    headers={"Retry-After": "0"}),
                            refused(503, "overloaded", kind="server_error", code="service_overloaded"),
                            ok('{"ok": 1}')]
        self.assertEqual(llm_call.chat_json([{"role": "user", "content": "x"}]), {"ok": 1})
        self.assertEqual(len(self.waits), 2)
        self.assertEqual(self.waits[0], 0.0)  # what Retry-After said
        self.assertGreaterEqual(self.waits[1], 1.0)  # no header: exponential backoff with jitter
        self.waits.clear()
        self.fake.script = [refused(500, "oops", kind="server_error")] * (llm_call.RETRIES + 1)
        with self.assertRaisesRegex(llm_call.LLMError, f"kept failing \\(500\\) after {llm_call.RETRIES} retries"):
            llm_call.chat([{"role": "user", "content": "x"}])
        self.assertEqual(len(self.waits), llm_call.RETRIES)
        self.fake.script = [refused(504, "too slow", kind="server_error", code="gateway_timeout")]
        with self.assertRaisesRegex(llm_call.LLMError, "timed out.*LLM_REASONING_EFFORT"):
            llm_call.chat([{"role": "user", "content": "x"}])

    def test_errors_say_what_to_do_and_never_show_the_key(self):
        self.use_fake()
        for status, kind, code, said in (
                (401, "authentication_error", "invalid_api_key", "refused the key"),
                (402, "billing_error", "billing_not_configured", "can't pay"),
                (403, "permission_error", None, "country"),
                (404, "invalid_request_error", "model_not_found", "no model called 'fake-muse'")):
            with self.subTest(status):
                self.fake.script = [refused(status, f"Incorrect API key provided: {KEY}", kind=kind, code=code)]
                with self.assertRaises(llm_call.LLMError) as caught:
                    llm_call.chat([{"role": "user", "content": "x"}])
                self.assertIn(said, str(caught.exception))
                self.assertNotIn(KEY, str(caught.exception))
                self.assertIn("***", str(caught.exception))  # the echo, redacted
        self.assertEqual(self.waits, [])  # none of those is worth retrying

    def test_a_reasoning_model_that_never_answers(self):
        self.use_fake()
        self.fake.script = [(200, completion(None, finish="length", reasoning_content="hmm"), {})]
        with self.assertRaisesRegex(llm_call.LLMError, "token allowance"):
            llm_call.chat([{"role": "user", "content": "x"}])
        self.fake.script = [(200, completion('{"examples": [', finish="length"), {})]
        with self.assertRaisesRegex(llm_call.LLMError, "cut off"):
            llm_call.chat_json([{"role": "user", "content": "x"}])
        self.fake.script = [(200, "<html>a proxy's error page</html>", {})] * (llm_call.RETRIES + 1)
        with self.assertRaisesRegex(llm_call.LLMError, "isn't a chat completion"):  # retried first: proxies blink
            llm_call.chat([{"role": "user", "content": "x"}])
        self.fake.script = [(200, "<html>a proxy's error page</html>", {}), ok("fine")]
        self.assertEqual(llm_call.chat([{"role": "user", "content": "x"}]), "fine")

    def test_an_unreachable_provider(self):
        port = self.fake.server.server_port
        self.fake.server.shutdown()
        self.fake.server.server_close()
        os.environ.update({"LLM_BASE_URL": f"http://127.0.0.1:{port}/v1", "LLM_API_KEY": KEY, "LLM_MODEL": MODEL})
        with self.assertRaisesRegex(llm_call.LLMError, "can't reach"):
            llm_call.chat([{"role": "user", "content": "x"}])


class ModelOverrideTests(AgentTest):
    def test_a_call_can_name_another_model_on_the_same_provider(self):
        """The second judge: the same key and URL, another model."""
        self.use_fake()
        self.fake.script = [ok({"a": 1})]
        llm_call.chat_json([{"role": "user", "content": "x"}], model="muse-wonder-1.0")
        self.assertEqual(self.fake.requests[0]["body"]["model"], "muse-wonder-1.0")
        self.assertEqual(llm_call.model_name(), MODEL)
        self.assertTrue(issubclass(llm_call.NotJSON, ValueError))  # what the agents' callers catch


def long_text(i, lead="Act as a senior support lead") -> str:
    """A long, demanding message (architect.LONG+ characters): a role, numbered parts, a format."""
    text = (f"{lead}. Ticket {i}: a customer says the export button does nothing. Give me 1) the likely cause, "
            "2) the fix, 3) what to tell them, in a short table with a heading, under 120 words, and keep the tone "
            "warm but direct, because this customer has written in three times already this week about it.")
    return text if len(text) >= architect.LONG else text + " Be thorough: check every step twice before you answer."


def good_plan(**changes) -> dict:
    """A plan that passes every check: 46 chats (10 long, 8 casual), 20 test prompts (5 long, 3 casual), each reply
    one sentence that opens its own way, and a rule they all meet."""
    users = ([f"How do I fix problem number {i}?" for i in range(28)] + [f"fix {i} pls" for i in range(8)]
             + [long_text(i) for i in range(10)])
    plan = {"job_id": "support-bot", "base": "qwen3-0.6b", "rank": 8, "epochs": 3, "caveat": "",
            "reasoning": "A small model learns a short style quickly.",
            "examples": [{"user": u, "assistant": f"Fix {i} usually works, so try it and restart."}
                         for i, u in enumerate(users)],
            "eval_prompts": [f"What should I do about issue {c}?" for c in "abcdefghijkl"]
                            + [long_text(c, "Act as a careful reviewer") for c in "mnopq"]
                            + [f"issue {c} help" for c in "rst"],
            "checks": [{"check": "max_sentences", "arg": 2}],
            "scope": {"narrow": False, "decline": ""}, "off_scope_prompts": []}
    return {**plan, **changes}


def routed(designs: list, audit_meets: bool = True, allowed: bool = True, more=None):
    """A fake Muse Spark that answers by which agent is asking (the system prompt): the screen, the designer (the
    next of `designs` each time), the audit, and the second batch."""
    def step(body):
        system = body["messages"][0]["content"]
        if system.startswith("You screen"):
            return ok({"allowed": allowed, "reason": "fine" if allowed else "it would exist to scam people"})
        if system.startswith("You design"):
            return ok(designs.pop(0))
        if system.startswith("You audit"):
            return ok({"verdicts": [{"n": n, "meets": audit_meets, "why": "ok"} for n in range(1, 13)]})
        if system.startswith("You add"):
            return ok(more if more is not None else {"examples": []})
        return refused(400, f"unexpected system prompt {system[:40]!r}")
    return step


class ArchitectTests(AgentTest):
    def test_a_good_plan_passes_on_both_backends(self):
        self.assertEqual(architect.check(good_plan()), [])
        self.assertEqual(architect.check(good_plan(), "vibehash"), [])

    def test_check_refuses_each_rule(self):
        fine = good_plan()
        leaked = fine["eval_prompts"][:-1] + ["How do I FIX problem number 3??"]
        cases = {
            "id upper case": ({"job_id": "Support Bot"}, "lowercase letters, digits and hyphens"),
            "id a path": ({"job_id": "../evil"}, "lowercase letters, digits and hyphens"),
            "id a device name": ({"job_id": "con"}, "device name"),
            "base not allowed": ({"base": "llama-3.1-8b"}, "base 'llama-3.1-8b' isn't one of"),
            "rank not allowed": ({"rank": 12}, "rank 12 isn't 8 or 16"),
            "epochs too many": ({"epochs": 9}, "epochs 9 isn't 2 to 5"),
            "too few examples": ({"examples": fine["examples"][:39]}, "only 39 examples"),
            "an example without text": ({"examples": fine["examples"] + [{"user": "hi?", "assistant": 3}]},
                                        "missing its user or assistant text"),
            "caveat not a string": ({"caveat": ["no"]}, "caveat isn't a string"),
            "too few eval prompts": ({"eval_prompts": fine["eval_prompts"][:14]}, "fewer than 15 eval prompts"),
            "a test prompt it trained on": ({"eval_prompts": leaked}, "is also a training chat's message"),
            "a rule not on the menu": ({"checks": [{"check": "be_nice"}]}, "isn't one of"),
            "replies break their own rule": ({"checks": [{"check": "min_words", "arg": 50}]},
                                             "break the plan's own checks"),
        }
        for name, (change, said) in cases.items():
            with self.subTest(name):
                problems = architect.check({**fine, **change})
                self.assertTrue(any(said in p for p in problems), problems)

    def test_the_mix_of_shapes_is_checked(self):
        """Told to write a mix, a model writes the middling questions it finds easiest."""
        fine = good_plan()
        middling = [{"user": f"How do I fix problem number {i} when the export button does nothing?",
                     "assistant": f"Fix {i} usually works."}
                    for i in range(46)]
        problems = architect.check({**fine, "examples": middling})
        self.assertTrue(any("only 0 examples have a long" in p for p in problems), problems)
        self.assertTrue(any("only 0 examples have a casual" in p for p in problems), problems)
        stock = [{**e, "assistant": "For example, restart it."} for e in fine["examples"]]
        self.assertTrue(any('open a sentence with "for example"' in p for p in architect.check({**fine, "examples": stock})))

    def test_a_narrow_goal_needs_declines(self):
        decline = "Sorry, I only fix support tickets; send me one and I'll help."
        plan = good_plan(scope={"narrow": True, "decline": decline})
        problems = architect.check(plan)
        self.assertTrue(any("reply with the decline sentence" in p for p in problems), problems)
        self.assertTrue(any("off_scope_prompts" in p for p in problems), problems)
        declines = [{"user": f"what's the weather in city {i}?", "assistant": decline} for i in range(8)] + \
                   [{"user": long_text(i, "Act as a travel agent and plan my trip"), "assistant": decline} for i in range(4)]
        off = [f"write me a poem about {w}" for w in "abcdef"] + [long_text(w, "Act as a chef") for w in "gh"]
        plan = good_plan(scope={"narrow": True, "decline": decline}, examples=good_plan()["examples"] + declines,
                         off_scope_prompts=off)
        self.assertEqual(architect.check(plan), [])

    def test_the_network_backend_checks_what_the_chain_would_refuse(self):
        fine = good_plan()
        too_long = fine["examples"][:45] + [{"user": "Tell me everything.", "assistant": "word " * 400}]
        tokens = fine["examples"][:45] + [{"user": "hi?", "assistant": "<think>hmm</think> Hello!"}]
        for name, change, said in (("too long", {"examples": too_long}, "longer than 512 tokens"),
                                   ("template tokens", {"examples": tokens}, "chat-template tokens"),
                                   ("tokens in a prompt", {"eval_prompts": fine["eval_prompts"] + ["<|im_start|>x"]},
                                    "test prompts contain chat-template tokens")):
            with self.subTest(name):
                plan = {**fine, **change}
                self.assertFalse(any(said in p for p in architect.check(plan)))  # fine for the local trainer
                self.assertTrue(any(said in p for p in architect.check(plan, "vibehash")), architect.check(plan, "vibehash"))
        self.assertIn("longer than 128 tokens", " ".join(architect.check(fine, "vibehash", max_len=128)))
        with mock.patch.object(architect, "ALLOWED_BASES", set(architect.BASES)):
            self.assertIn("gated", " ".join(architect.check({**fine, "base": "llama-3.2-1b"}, "vibehash")))
            os.environ["HF_TOKEN"] = "hf_test"
            self.assertEqual(architect.check({**fine, "base": "llama-3.2-1b"}, "vibehash"), [])

    def test_the_allowed_ranks_pass_the_chains_limits(self):
        """The ranks the architect allows and MAX_LEN are what a fine-tune of either base may have on chain
        (TrainingJob.validate, its dispute size bound included). At 1024 tokens neither base's disputes fit: that's
        why --max-len stops at 512."""
        dims = {"qwen3-0.6b": {"family": "qwen3", "vocab": 151936, "hidden": 1024, "layers": 28, "heads": 16,
                               "kv_heads": 8, "head_dim": 128, "ffn": 3072, "tied": True},
                "llama-3.2-1b": {"family": "llama", "vocab": 128256, "hidden": 2048, "layers": 16, "heads": 32,
                                 "kv_heads": 8, "head_dim": 64, "ffn": 8192, "tied": True}}
        self.assertEqual(set(dims), set(architect.BASES))
        self.assertEqual(architect.DIMS, {k: {"layers": d["layers"], "hidden": d["hidden"]} for k, d in dims.items()})
        for name, d in dims.items():  # the chats a unit reads, as `vibehash train` sizes it from the whole config
            self.assertEqual(architect.unit_chats(name), ft.UNIT_STEPS * ft.settings_for(d)["batch_size"])

        def job(d, rank, seq_len):
            spec = ft.LoraSpec(base="hf:org/name@" + "c" * 40, manifest="a" * 64, dims=d, rank=rank, alpha=2.0 * rank,
                               seq_len=seq_len, seed=11)
            s = ft.settings_for(d)
            return TrainingJob("support-bot-1a2b", "", budget=50 * 10**8, fee_per_round=10 * 10**8, spec=spec,
                               data_source={"kind": "tokens", "sha256": "0" * 64}, init_seed=11, lr=s["lr"],
                               batch_size=s["batch_size"], steps_per_unit=ft.UNIT_STEPS,
                               checkpoint_every=ft.SEGMENT_STEPS, units_per_round=8,
                               memory_mb=ft.memory_mb(spec, s["batch_size"], 2 * 10**9), tolerance=ft.TOLERANCE,
                               max_update_ratio=1.0)
        from vibehash.tasks import InvalidJob
        for name, d in dims.items():
            for rank in (8, 16):
                with self.subTest(name, rank=rank):
                    job(d, rank, architect.MAX_LEN).validate()
            with self.subTest(name, seq_len=1024), self.assertRaisesRegex(InvalidJob, "dispute"):
                job(d, 16, 1024)

    def test_budgets_ids_and_token_guesses(self):
        def plan(n, epochs, base="qwen3-0.6b"):
            return {"base": base, "examples": [0] * n, "epochs": epochs}
        # a unit reads 16 steps of 8 chats: 80 chats (72 trained) at 3 epochs is 2 units in one round, not 2 rounds
        # of 8 units (25 epochs)
        self.assertEqual(architect.schedule(plan(80, 3)), {"units": 2, "budget": 10, "network_epochs": 3.6})
        self.assertEqual(architect.schedule(plan(46, 3)), {"units": 1, "budget": 10, "network_epochs": 3.0})
        self.assertEqual(architect.schedule(plan(46, 3, "llama-3.2-1b"))["units"], 1)
        self.assertEqual(architect.budget(plan(10, 2)), architect.FEE)  # at least a round
        self.assertEqual(architect.schedule(plan(4000, 5)), {"units": 8, "budget": 190, "network_epochs": 5.0})
        capped = architect.schedule(plan(20000, 5))  # what MAX_BUDGET buys, and fewer epochs than asked
        self.assertEqual((capped["units"], capped["budget"]), (8, architect.MAX_BUDGET))
        self.assertLess(capped["network_epochs"], 5)
        self.assertEqual(architect.schedule(plan(46, 3), budget=30), {"units": 1, "budget": 30, "network_epochs": 9.1})
        self.assertEqual(architect.trained(46), 42)  # train holds back a tenth
        self.assertEqual(architect.trained(5000), 5000 - ft.MAX_VAL)
        self.assertLessEqual(architect.MAX_BUDGET, 200)
        (Path(os.environ["VIBEHASH_HOME"]) / "jobs" / "bot-aaaa").mkdir(parents=True)
        with mock.patch.object(architect.secrets, "token_hex", side_effect=["aaaa", "bbbb"]):
            self.assertEqual(architect.chain_id("bot"), "bot-bbbb")  # taken here already: drawn again
        self.assertRegex(architect.chain_id("x" * 40), r"^x{40}-[0-9a-f]{4}$")
        self.assertEqual(architect.tokens_about("abcdef"), 2)
        self.assertEqual(architect.tokens_about("你好吗"), 3)

    def test_the_train_commands_are_exactly_these(self):
        plan = good_plan(job_id="support-bot-1a2b", budget=50, units=2)
        out = self.dir / "agent-jobs"
        chats = out / "support-bot-1a2b.jsonl"
        self.assertEqual(architect.train_command(plan, chats, out),
                         [sys.executable, str(Path(architect.__file__).with_name("train_local.py")), str(chats),
                          "--base", "qwen3-0.6b", "--rank", "8", "--epochs", "3", "--out", str(out / "adapters")])
        cmd = architect.train_command(plan, chats, out, "vibehash")
        self.assertEqual(cmd, [sys.executable, "-m", "vibehash", "train", str(chats.resolve()), "--base", "qwen3-0.6b",
                               "--rank", "8", "--budget", "50", "--units", "2", "--max-len", "512",
                               "--id", "support-bot-1a2b", "--plain"])
        full = architect.train_command(plan, chats, out, "vibehash", max_len=256, node="http://127.0.0.1:8545",
                                       wallet="req.json", yes=True)
        a = cli.parser().parse_args(full[3:])  # what `vibehash train` makes of it
        self.assertEqual((a.data, a.base, a.rank, a.budget, a.units, a.max_len, a.id, a.plain, a.yes, a.node,
                          a.wallet),
                         (str(chats.resolve()), "qwen3-0.6b", 8, 50 * 10**8, 2, 256, "support-bot-1a2b", True,
                          True, "http://127.0.0.1:8545", "req.json"))
        self.assertIs(a.fn, cli.cmd_train)

    def test_plan_job_screens_designs_retries_and_audits(self):
        """The first design has too few chats; the model is told so, and its second passes, then the audit. The
        second batch keeps failing its checks, so the job trains on the first."""
        self.use_fake()
        self.fake.script = [routed([good_plan(examples=good_plan()["examples"][:30]), good_plan()])] * 6
        logged = []
        with contextlib.redirect_stderr(io.StringIO()):
            plan = architect.plan_job("answer support questions in a sentence", log=logged.append)
        systems = [r["body"]["messages"][0]["content"][:12] for r in self.fake.requests]
        self.assertEqual(systems, ["You screen r", "You design f", "You design f", "You audit tr",
                                   "You add trai", "You add trai"])
        retry = self.fake.requests[2]["body"]["messages"]
        self.assertEqual([m["role"] for m in retry], ["system", "user", "assistant", "user"])
        self.assertIn("only 30 examples", retry[-1]["content"])
        self.assertIn("only 30 examples", logged[0])
        self.assertTrue(any("training on the first" in m for m in logged))
        self.assertEqual((plan["goal"], len(plan["examples"]), plan["audit"]["met"], plan["audit"]["sampled"]),
                         ("answer support questions in a sentence", 46, 12, 12))
        gated = self.fake.requests[0]["body"]["messages"][1]["content"]
        self.assertIn("<<<\nanswer support questions in a sentence\n>>>", gated)  # the goal, fenced as data

    def test_a_refused_goal_designs_nothing(self):
        self.use_fake()
        self.fake.script = [routed([], allowed=False)]
        with self.assertRaisesRegex(architect.Refused, "turned this goal down: it would exist to scam people"):
            architect.plan_job("write scam texts")
        self.assertEqual(len(self.fake.requests), 1)

    def test_a_failed_audit_sends_the_plan_back(self):
        self.use_fake()
        self.fake.script = [routed([good_plan(), good_plan(), good_plan()], audit_meets=False)] * 9
        logged = []
        with contextlib.redirect_stderr(io.StringIO()):
            plan = architect.plan_job("support", log=logged.append)
        self.assertIn("the Quality Judge read 12 of the chats and 12 don't do what the goal asks", logged[0])
        self.assertTrue(any("training anyway" in m for m in logged))  # three tries, then the audit is published
        self.assertEqual(plan["audit"]["met"], 0)

    def run_main(self, *argv) -> tuple[str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = architect.main(list(argv))
        self.assertEqual(code, 0)
        return out.getvalue(), err.getvalue()

    def test_main_writes_a_network_job_and_posts_nothing(self):
        self.use_fake()
        self.fake.script = [routed([good_plan()])] * 5
        out_dir = self.dir / "agent-jobs"
        said, _ = self.run_main("answer support questions in a sentence", "--out", str(out_dir))
        chats = list(out_dir.glob("*.jsonl"))
        self.assertEqual(len(chats), 1)
        job_id = chats[0].stem
        self.assertRegex(job_id, r"^support-bot-[0-9a-f]{4}$")  # new on the chain
        records = ft.read_examples(chats[0])  # `vibehash train`'s own reader
        self.assertEqual(len(records), 46)
        self.assertEqual([m["role"] for m in records[0]["messages"]], ["user", "assistant"])
        saved = json.loads((out_dir / f"{job_id}.plan.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["backend"], saved["budget"], saved["units"], saved["network_epochs"], saved["model"],
                          saved["examples"]), ("vibehash", 10, 1, 3.0, MODEL, 46))
        evals = json.loads((out_dir / f"{job_id}.eval.json").read_text(encoding="utf-8"))
        self.assertEqual((evals["base"], len(evals["prompts"])), ("qwen3-0.6b", 20))
        for f in out_dir.iterdir():
            self.assertNotIn(KEY, f.read_text(encoding="utf-8"))
        self.assertIn("--budget 10 --units 1", said)
        self.assertIn("10 VIBE: 1 round at 10 VIBE of 1 unit, about 3.0 epochs over the 42 chats", said)
        self.assertIn("Nothing was posted, and no VIBE spent", said)

    def test_post_runs_train_and_reads_its_receipt(self):
        self.use_fake()
        ran = []

        def fake_run(cmd, env=None, **kw):
            ran.append((cmd, env))
            if posts:  # as `vibehash train` does once it has posted
                receipt = Path(os.environ["VIBEHASH_HOME"]) / "jobs" / cmd[cmd.index("--id") + 1] / "job.json"
                receipt.parent.mkdir(parents=True)
                receipt.write_text("{}")
            return mock.Mock(returncode=0)
        for posts, said in ((True, "Posted support-bot-"), (False, "Nothing was posted, so nothing was spent")):
            with self.subTest(posts=posts), mock.patch.object(architect.subprocess, "run", fake_run):
                self.fake.script = [routed([good_plan()])] * 5
                out, _ = self.run_main("support bot", "--out", str(self.dir / f"jobs{int(posts)}"), "--post",
                                       "--node", "http://127.0.0.1:9", "--wallet", "req.json", "--budget", "30")
                cmd, env = ran[-1]
                self.assertEqual(cmd[cmd.index("--budget") + 1], "30")
                self.assertEqual(cmd[cmd.index("--node") + 1], "http://127.0.0.1:9")
                self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(architect.REPO))
                self.assertIn("Posting to http://127.0.0.1:9", out)
                self.assertIn(said, out)
                if posts:
                    self.assertIn("job download support-bot-", out)
                    self.assertIn("judge.py grade", out)

    def test_bad_flags_stop_before_anything_is_paid_for(self):
        self.use_fake()
        (Path(os.environ["VIBEHASH_HOME"]) / "jobs" / "mine").mkdir(parents=True)
        (self.dir / "taken.jsonl").write_text("")
        for argv, said in ((["goal", "--budget", "5"], "--budget must be"), (["goal", "--max-len", "4096"], "--max-len"),
                           (["goal", "--fee", "0"], "--fee must be"), (["goal", "--id", "Bad Id"], "lowercase"),
                           (["goal", "--id", "mine"], "already have a job called mine"),
                           (["goal", "--id", "taken", "--out", str(self.dir)], "already exists")):
            with self.subTest(argv), self.assertRaises(SystemExit) as caught:
                architect.main(argv)
            self.assertIn(said, str(caught.exception.code))
        self.assertEqual(self.fake.requests, [])

    def test_the_requester_can_name_the_job_pick_the_base_and_the_fee(self):
        """Cole's flags: --id is used as given (no hex digits), --base rides into the design prompt and the plan
        whatever the LLM wrote, --fee goes to train and sizes the budget."""
        self.use_fake()
        self.fake.script = [routed([good_plan(base="llama-3.2-1b")])] * 5
        os.environ["HF_TOKEN"] = "hf_test"
        out_dir = self.dir / "jobs"
        with mock.patch.object(architect, "ALLOWED_BASES", {"qwen3-0.6b"}):
            said, _ = self.run_main("support bot", "--out", str(out_dir), "--id", "my-bot", "--base", "qwen3-0.6b",
                                    "--fee", "5")
        self.assertIn("(Use base qwen3-0.6b.)", self.fake.requests[1]["body"]["messages"][1]["content"])
        plan = json.loads((out_dir / "my-bot.plan.json").read_text(encoding="utf-8"))
        self.assertEqual((plan["job_id"], plan["base"], plan["budget"]), ("my-bot", "qwen3-0.6b", 5))  # a round at 5
        self.assertIn("--fee 5", said)
        self.assertIn("--id my-bot", said)
        self.assertEqual(architect.budget({"base": "qwen3-0.6b", "examples": [0] * 600, "epochs": 3}, fee=5), 10)
        cmd = architect.train_command(good_plan(job_id="b", budget=10, units=1), out_dir / "b.jsonl", out_dir,
                                      "vibehash", fee=5)
        self.assertEqual(cli.parser().parse_args(cmd[3:]).fee, 5 * 10**8)

    @unittest.skipUnless(HAVE["torch"] and HAVE["tokenizers"] and HAVE["jinja2"], "needs PyTorch, tokenizers and jinja2")
    def test_vibehash_train_takes_the_files_up_to_its_preflight(self):
        """The command the architect builds, run by `vibehash train` itself against a fake Hugging Face serving a
        tiny Qwen3 with a Qwen3-style template (an empty think block before each reply), with --base pointed at it:
        it reads the chats with the model's tokenizer and template, finds every reply, sizes the job, shows the
        pre-flight, and is told no at "Post it?". Nothing reaches a node."""
        from tests.finetune_fixtures import TinyModel
        from tests.test_hub import FakeHub

        self.use_fake()
        self.fake.script = [routed([good_plan()])] * 5
        out_dir = self.dir / "jobs"
        self.run_main("support bot", "--out", str(out_dir), "--max-len", "256")
        chats = next(out_dir.glob("*.jsonl"))
        plan = json.loads(chats.with_suffix(".plan.json").read_text(encoding="utf-8"))
        cmd = architect.train_command(plan, chats, out_dir, "vibehash", max_len=256)

        hub = FakeHub()
        self.addCleanup(hub.close)
        tiny = TinyModel(self.dir / "src")
        files = {p.name: p.read_bytes() for p in tiny.folder.iterdir()}
        files["tokenizer.json"] = word_tokenizer(tiny.vocab).encode()
        files["tokenizer_config.json"] = json.dumps({"chat_template": QWEN3ISH, "eos_token": "<|im_end|>"}).encode()
        hub.add_commit("tiny/qwen3", files)
        os.environ.update({"HF_ENDPOINT": hub.endpoint, "HF_HOME": str(self.dir / "hf")})
        ft.unload()
        self.addCleanup(ft.unload)

        argv = cmd[3:]
        argv[argv.index("--base") + 1] = "tiny/qwen3"
        asked = []

        def ask(question, assume_yes=False):
            asked.append(question)
            return question.startswith("Check")  # stream the tiny weights to pin them; then don't post
        out = io.StringIO()
        with mock.patch.object(cli, "ask", ask), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.main(argv + ["--node", "http://127.0.0.1:9"])
        said = out.getvalue()
        self.assertEqual([q.split()[0] for q in asked], ["Check", "Post"])
        self.assertIn("Model     tiny/qwen3", said)
        self.assertIn("46 examples (46 chats, in the model's chat template)", said)
        self.assertIn("LoRA rank 8", said)
        self.assertIn("a budget of 10 VIBE buys 1 rounds", said)
        # the epochs the architect paid for are the ones train's own pre-flight works out (the tiny model takes the
        # same 8 chats a step as Qwen3-0.6B)
        kept = int(re.search(r"([\d,]+) to train", said)[1].replace(",", ""))
        batch, steps, units = map(int, re.search(r"(\d+) sequences a step, (\d+) steps a unit, (\d+) units a round",
                                                 said).groups())
        self.assertEqual((kept, units), (architect.trained(46), plan["units"]))
        self.assertEqual(round(plan["budget"] // architect.FEE * units * steps * batch / kept, 1),
                         plan["network_epochs"])
        self.assertIn("Nothing was posted, and nothing was spent.", said)
        self.assertFalse((Path(os.environ["VIBEHASH_HOME"]) / "jobs" / plan["job_id"]).exists())


class ChecksTests(unittest.TestCase):
    def test_rules_score_replies(self):
        two = {"check": "exact_sentences", "arg": 2}
        self.assertTrue(checks.passes(two, "Yes. No."))
        self.assertFalse(checks.passes(two, "Yes."))
        self.assertTrue(checks.passes({"check": "json_keys", "arg": ["a", "b"]}, '{"a": 1, "b": 2}'))
        self.assertFalse(checks.passes({"check": "starts_with", "arg": "="}, None))  # nothing to apply it to: a fail
        self.assertEqual(checks.failed([two, {"check": "no_bullets"}], "- one\n- two"), ["exact_sentences 2", "no_bullets"])
        self.assertEqual(checks.failed([two], "Sorry, I only do that.", exempt=lambda r: r.startswith("Sorry")), [])

    def test_valid_names_the_problem(self):
        self.assertEqual(checks.valid([{"check": "max_words", "arg": 50}]), [])
        self.assertIn("isn't one of", checks.valid([{"check": "rhymes"}])[0])
        self.assertIn("needs number", checks.valid([{"check": "max_words", "arg": "50"}])[0])
        self.assertIn("needs no", checks.valid([{"check": "one_line", "arg": 1}])[0])
        self.assertIn("pick the 5", checks.valid([{"check": "one_line"}] * 6)[0])


class JudgeTests(AgentTest):
    def setUp(self):
        super().setUp()
        second = mock.patch.object(judge, "SECOND", "")  # the second judge has its own test
        second.start()
        self.addCleanup(second.stop)

    @staticmethod
    def picks_good(body):
        """A fake judge that always prefers the reply that says GOOD (the fine-tune's)."""
        text = body["messages"][1]["content"]
        a = re.search(r"Reply A:\n<<<\n(.*?)\n>>>", text, re.DOTALL).group(1)
        return ok(f"Verdict: {json.dumps({'winner': 'A' if a.startswith('GOOD') else 'B', 'reason': 'Reply A is kinder'})}")

    def test_a_verdict_needs_both_readings_and_is_blind(self):
        self.use_fake()
        pair = {"prompt": "question?", "base": "BAD reply", "tuned": "GOOD reply"}
        self.fake.script = [self.picks_good] * 2
        v = judge.verdict("two friendly sentences", pair)
        self.assertEqual((v["tuned_won"], v["winner"], v["readings"]), (True, "tuned", ["tuned", "tuned"]))
        self.assertIn("the fine-tune is kinder", v["reason"].lower())  # unblinded once decided
        firsts = [re.search(r"Reply A:\n<<<\n(.*?)\n", r["body"]["messages"][1]["content"]).group(1)
                  for r in self.fake.requests]
        self.assertEqual(firsts, ["GOOD reply", "BAD reply"])  # once in each order
        for r in self.fake.requests:  # blind: nothing says which reply is which
            self.assertNotRegex(r["body"]["messages"][1]["content"], r"tuned|base|fine-tune")
        self.fake.script = [ok({"winner": "A", "reason": "first is best"})] * 2  # a judge that favours position
        v = judge.verdict("two friendly sentences", pair)
        self.assertEqual((v["tuned_won"], v["winner"]), (False, "tie"))  # the readings split: a tie, not a win

    def test_grading_a_file(self):
        self.use_fake()
        path = self.dir / "support-bot.eval.answers.json"
        path.write_text(json.dumps({"goal": "two friendly sentences", "checks": [{"check": "max_sentences", "arg": 2}],
                                    "pairs": [{"prompt": f"question {i}?", "base": f"BAD {i}",
                                               "prompted": f"BAD told {i}", "tuned": f"GOOD {i}. Yes. Really."
                                               if i == 0 else f"GOOD {i}."} for i in range(4)]}), encoding="utf-8")
        self.fake.script = [self.picks_good] * 16  # 4 pairs, 2 rivals, 2 readings
        seen = []
        with contextlib.redirect_stdout(io.StringIO()) as out:
            judge.grade(path, on_verdict=lambda i, pair: seen.append(i))
        data = json.loads(path.with_suffix(".graded.json").read_text(encoding="utf-8"))
        s = data["summary"]
        self.assertEqual((s["prompts"], s["tuned_wins"], s["ties"], s["base_wins"]), (4, 4, 0, 0))
        self.assertEqual(s["vs_prompted"]["tuned_wins"], 4)
        self.assertEqual(data["model"], MODEL)
        self.assertEqual(data["pairs"][0]["fails"], ["max_sentences 2"])  # scored by code, not the judge
        self.assertEqual(s["checks"]["all"]["tuned"], 0.75)
        self.assertEqual(sorted(seen), [0, 1, 2, 3])
        self.assertIn("fine-tune preferred on 4 of 4 prompts", out.getvalue())
        self.assertNotIn(KEY, path.with_suffix(".graded.json").read_text(encoding="utf-8"))

    def test_an_api_error_stops_grading(self):
        self.use_fake()
        path = self.dir / "bot.eval.answers.json"
        path.write_text(json.dumps({"goal": "g", "pairs": [{"prompt": "q?", "base": "b", "tuned": "t"}]}))
        self.fake.script = [refused(402, "Your balance is too low", kind="billing_error", code="billing_not_configured")]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(llm_call.LLMError, "can't pay"):
            judge.grade(path)

    def test_the_second_judge_is_another_model_and_never_ends_a_run(self):
        self.use_fake()
        pair = {"prompt": "q?", "base": "BAD", "prompted": "BAD told", "tuned": "GOOD"}
        with mock.patch.object(judge, "SECOND", "muse-wonder-1.0"):
            self.fake.script = [self.picks_good] * 6
            out = judge.judged("g", pair)
            self.assertEqual([r["body"]["model"] for r in self.fake.requests], [MODEL] * 4 + ["muse-wonder-1.0"] * 2)
            self.assertTrue(out["second"]["tuned_won"])
            self.fake.script = [self.picks_good] * 4 + [refused(404, "no such model", code="model_not_found")]
            with contextlib.redirect_stdout(io.StringIO()) as said:
                out = judge.judged("g", pair)
            self.assertNotIn("second", out)
            self.assertIn("second judge (muse-wonder-1.0) failed", said.getvalue())

    def test_scores_and_their_uncertainty(self):
        self.assertEqual(judge.chance(19, 1), 2e-05)
        self.assertEqual(judge.chance(10, 6), 0.227)
        self.assertEqual(judge.interval(19, 20), [0.764, 0.991])
        self.assertIsNone(judge.kappa(["tuned"] * 3, ["tuned"] * 3))
        self.assertEqual(judge.kappa(["tuned", "base"], ["tuned", "base"]), 1.0)

    def test_locating_a_base_and_an_untrained_adapter(self):
        self.assertEqual(judge.located("org/name@abc"), {"pretrained_model_name_or_path": "org/name", "revision": "abc"})
        self.assertEqual(judge.located("org/name"), {"pretrained_model_name_or_path": "org/name", "revision": None})
        self.assertEqual(judge.located(str(self.dir))["revision"], None)
        folder = self.dir / "lora"
        folder.mkdir()
        self.assertIsNone(judge.trained(folder))  # nothing to read

        def safetensors(b_bytes: bytes) -> bytes:
            header = json.dumps({"base.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                                 "base.lora_B.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}).encode()
            return len(header).to_bytes(8, "little") + header + b"\x01\x00\x00\x00" + b_bytes
        (folder / "adapter_model.safetensors").write_bytes(safetensors(b"\x00" * 4))
        self.assertFalse(judge.trained(folder))
        (folder / "adapter_model.safetensors").write_bytes(safetensors(b"\x00\x00\x80\x3f"))
        self.assertTrue(judge.trained(folder))
        a = judge.parser().parse_args(["answer", "e.json", "lora", "--base", "org/name", "--device", "cpu"])
        self.assertEqual((a.step, a.base, a.device), ("answer", "org/name", "cpu"))
        self.assertEqual(judge.parser().parse_args(["grade", "x.json"]).step, "grade")

    def test_an_adapter_runs_on_the_base_it_was_trained_on(self):
        folder = self.dir / "lora"
        folder.mkdir()
        self.assertEqual(judge.pinned(folder, "qwen3-0.6b"), {})  # no config: the base's own pin
        config = {"base_model_name_or_path": "Qwen/Qwen3-0.6B", "revision": "e" * 40, "peft_type": "LORA"}
        (folder / "adapter_config.json").write_text(json.dumps(config))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(judge.pinned(folder, "qwen3-0.6b"), {"revision": "e" * 40})
        self.assertIn("loading that", err.getvalue())
        (folder / "adapter_config.json").write_text(json.dumps({**config, "base_model_name_or_path": "other/model"}))
        self.assertEqual(judge.pinned(folder, "qwen3-0.6b"), {})


if __name__ == "__main__":
    unittest.main()
