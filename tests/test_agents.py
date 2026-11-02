"""The Muse agents (agents/): the Job Architect, the Semantic Judge, and the LLM client they share.

Offline, on the CPU, with no API key: a fake OpenAI-compatible server on 127.0.0.1 plays the model, answering from a
script (plans, fenced and prose-wrapped JSON, refusals of response_format and temperature, rate limits, bad keys).
The architect's files go through `vibehash train` itself up to its pre-flight, against tests/test_hub.py's fake
Hugging Face, where PyTorch, tokenizers and jinja2 are installed (CI has them). The judge's `answer` step runs a tiny
random model and an adapter `finetune.export_peft` wrote, where transformers and peft are installed too.
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

from agents import architect, judge, llm_call
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


def good_plan(n: int = 45, **changes) -> dict:
    plan = {"job_id": "Support Bot!", "base": "qwen3-0.6b", "rank": 8, "budget": 50,
            "reasoning": "A small model learns a short style quickly.",
            "examples": [{"user": f"How do I fix problem number {i}?", "assistant": f"Try step {i}, then restart it. "
                                                                                   "Happy to help!"} for i in range(n)],
            "eval_prompts": [f"What should I do about issue {c}?" for c in "abcdef"]}
    return {**plan, **changes}


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
                                 ("https://api.meta.ai/v1", "muse-spark-1.1", "k1", "meta"))
        s = llm_call.settings({"MODEL_API_KEY": "k1", "LLM_MODEL": "muse-spark-1.3", "LLM_API_KEY": "k2"})
        self.assertEqual((s["model"], s["key"]), ("muse-spark-1.3", "k2"))  # LLM_* still win
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
        self.fake.script = [(200, "<html>a proxy's error page</html>", {})]
        with self.assertRaisesRegex(llm_call.LLMError, "isn't a chat completion"):
            llm_call.chat([{"role": "user", "content": "x"}])

    def test_an_unreachable_provider(self):
        port = self.fake.server.server_port
        self.fake.server.shutdown()
        self.fake.server.server_close()
        os.environ.update({"LLM_BASE_URL": f"http://127.0.0.1:{port}/v1", "LLM_API_KEY": KEY, "LLM_MODEL": MODEL})
        with self.assertRaisesRegex(llm_call.LLMError, "can't reach"):
            llm_call.chat([{"role": "user", "content": "x"}])


class ArchitectTests(AgentTest):
    def checked(self, plan, **kw) -> list[str]:
        return architect.check(plan, **kw)

    def test_a_good_plan_passes(self):
        plan = good_plan(job_id="support-bot-1a2b")
        self.assertEqual(self.checked(plan), [])

    def test_check_refuses_each_rule(self):
        fine = good_plan(job_id="support-bot-1a2b")
        out = self.dir / "jobs"
        out.mkdir()
        (out / "taken-1.jsonl").write_text("")
        (Path(os.environ["VIBEHASH_HOME"]) / "jobs" / "posted-1").mkdir(parents=True)
        dup = [dict(e) for e in fine["examples"]]
        dup[7]["user"] = "  how do I FIX problem number 3??"
        cases = {
            "not an object": ([1, 2], "one JSON object"),
            "id upper case": ({"job_id": "Support Bot"}, "the chain's rule"),
            "id too long": ({"job_id": "a" * 65}, "the chain's rule"),
            "id a device name": ({"job_id": "con.v2"}, "device name"),
            "id ends in a dot": ({"job_id": "bot."}, "ends in a dot"),
            "id already posted here": ({"job_id": "posted-1"}, "already have a job called posted-1"),
            "files already there": ({"job_id": "taken-1"}, "already exists"),
            "base not allowed": ({"base": "llama-3.1-8b"}, "base 'llama-3.1-8b' must be one of"),
            "base gated, no token": ({"base": "llama-3.2-1b"}, "gated"),
            "rank not allowed": ({"rank": 12}, "rank 12 must be one of"),
            "rank a boolean": ({"rank": True}, "rank True"),
            "rank a string": ({"rank": "8"}, "rank '8'"),
            "budget below a round's fee": ({"budget": 5}, "from 10 (one round's fee)"),
            "budget too big": ({"budget": 500}, "to 200 VIBE"),
            "budget fractional": ({"budget": 50.5}, "whole number"),
            "budget a boolean": ({"budget": True}, "whole number"),
            "budget a string": ({"budget": "50"}, "whole number"),
            "budget NaN": ({"budget": float("nan")}, "whole number"),
            "too few examples": ({"examples": fine["examples"][:10]}, "there are 10 examples; it takes 40"),
            "examples not a list": ({"examples": "lots"}, "must be a list"),
            "an example not an object": ({"examples": fine["examples"][:44] + [["hi", "there"]]}, "examples 45 aren't"),
            "an empty reply": ({"examples": fine["examples"][:44] + [{"user": "hi?", "assistant": "   "}]},
                               "examples 45 have an empty"),
            "an example too long": ({"examples": fine["examples"][:44] + [{"user": "Tell me everything.",
                                                                            "assistant": "word " * 400}]},
                                    "examples 45 may be longer than 512 tokens"),
            "an example with template tokens": ({"examples": fine["examples"][:44] + [
                {"user": "hi?", "assistant": "<think>hmm</think> Hello!"}]}, "chat-template tokens"),
            "a duplicate example": ({"examples": dup}, "examples 4 and 8 have the same user message"),
            "too few eval prompts": ({"eval_prompts": ["a?", "b?"]}, "there are 2 eval prompts"),
            "eval prompts not strings": ({"eval_prompts": ["a?", 3, "c?", "d?", "e?"]}, "non-empty strings"),
            "a repeated eval prompt": ({"eval_prompts": fine["eval_prompts"] + ["what should i do about issue A"]},
                                       "eval prompts 7 repeat"),
            "an eval prompt it trained on": ({"eval_prompts": fine["eval_prompts"] + ["How do I fix problem number 9"]},
                                             "eval prompts 7 are also examples' user messages"),
        }
        for name, (change, said) in cases.items():
            with self.subTest(name):
                plan = change if not isinstance(change, dict) else {**fine, **change}
                problems = self.checked(plan, out=out)
                self.assertTrue(any(said in p for p in problems), problems)
                if isinstance(change, dict) and name != "a duplicate example":
                    self.assertEqual(len(problems), 1, problems)  # just the one rule broken
        os.environ["HF_TOKEN"] = "hf_test"
        self.assertEqual(self.checked({**fine, "base": "llama-3.2-1b"}), [])

    def test_fee_and_max_len_move_the_limits(self):
        plan = good_plan(job_id="support-bot-1a2b", budget=20)
        self.assertEqual(self.checked(plan), [])
        self.assertIn("from 25", " ".join(self.checked(plan, fee="25")))
        longish = good_plan(job_id="support-bot-1a2b", examples=[
            {"user": f"Question {i}?", "assistant": "x" * 500} for i in range(45)])
        self.assertEqual(self.checked(longish), [])
        self.assertIn("longer than 128 tokens", " ".join(self.checked(longish, max_len=128)))

    def test_the_allowed_ranks_pass_the_chains_limits(self):
        """RANKS and MAX_LEN are what a fine-tune of either base may have on chain (TrainingJob.validate, its dispute
        size bound included). At 1024 tokens neither base's disputes fit: that's why --max-len stops at 512."""
        dims = {"qwen3-0.6b": {"family": "qwen3", "vocab": 151936, "hidden": 1024, "layers": 28, "heads": 16,
                               "kv_heads": 8, "head_dim": 128, "ffn": 3072, "tied": True},
                "llama-3.2-1b": {"family": "llama", "vocab": 128256, "hidden": 2048, "layers": 16, "heads": 32,
                                 "kv_heads": 8, "head_dim": 64, "ffn": 8192, "tied": True}}
        self.assertEqual(set(dims), set(architect.BASES))

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
            for rank in architect.RANKS:
                with self.subTest(name, rank=rank):
                    job(d, rank, architect.MAX_LEN).validate()
            with self.subTest(name, seq_len=1024), self.assertRaisesRegex(InvalidJob, "dispute"):
                job(d, 16, 1024)

    def test_job_ids_and_token_guesses(self):
        self.assertRegex(architect.job_id_for("Support Bot!"), r"^support-bot-[0-9a-f]{4}$")
        self.assertRegex(architect.job_id_for(None), r"^job-[0-9a-f]{4}$")
        self.assertRegex(architect.job_id_for("x" * 90), r"^x{40}-[0-9a-f]{4}$")
        self.assertEqual(architect.tokens_about("abcdef"), 2)
        self.assertEqual(architect.tokens_about("你好吗"), 3)

    def test_the_train_command_is_exactly_this(self):
        plan = good_plan(job_id="support-bot-1a2b")
        chats = self.dir / "agent-jobs" / "support-bot-1a2b.jsonl"
        cmd = architect.train_command(plan, chats)
        self.assertEqual(cmd, [sys.executable, "-m", "vibehash", "train", str(chats.resolve()), "--base", "qwen3-0.6b",
                               "--rank", "8", "--budget", "50", "--max-len", "512", "--id", "support-bot-1a2b",
                               "--plain"])
        full = architect.train_command(plan, chats, max_len=256, fee="5", node="http://127.0.0.1:8545",
                                       wallet="req.json", yes=True)
        self.assertEqual(full[len(cmd) - 1:], ["--plain", "--fee", "5", "--node", "http://127.0.0.1:8545",
                                               "--wallet", "req.json", "--yes"])
        self.assertIn("256", full)
        a = cli.parser().parse_args(full[3:])  # what `vibehash train` makes of it
        self.assertEqual((a.data, a.base, a.rank, a.budget, a.fee, a.max_len, a.id, a.plain, a.yes, a.node, a.wallet),
                         (str(chats.resolve()), "qwen3-0.6b", 8, 50 * 10**8, 5 * 10**8, 256, "support-bot-1a2b", True,
                          True, "http://127.0.0.1:8545", "req.json"))
        self.assertIs(a.fn, cli.cmd_train)
        self.assertEqual(cli.parser().parse_args(cmd[3:]).fee, 10 * 10**8)  # FEE is train's default

    def run_main(self, *argv) -> tuple[str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = architect.main(list(argv))
        self.assertEqual(code, 0)
        return out.getvalue(), err.getvalue()

    def test_end_to_end_with_a_retry(self):
        """The first plan breaks a rule (30 examples, and in prose); the model is told what, and its fenced second
        plan is written out in the shape `vibehash train` reads."""
        self.use_fake()
        self.fake.script = [ok(f"Here is the job: {json.dumps(good_plan(30))} Good luck!"),
                            ok(f"```json\n{json.dumps(good_plan())}\n```")]
        out_dir = self.dir / "agent-jobs"
        said, warned = self.run_main("make it answer support questions in two friendly sentences", "--out", str(out_dir))
        self.assertIn("there are 30 examples; it takes 40", warned)
        self.assertEqual(len(self.fake.requests), 2)
        retry = self.fake.requests[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in retry], ["system", "user", "assistant", "user"])
        self.assertIn("there are 30 examples", retry[-1]["content"])
        self.assertIn("make it answer support questions", self.fake.requests[0]["body"]["messages"][1]["content"])

        chats = list(out_dir.glob("*.jsonl"))
        self.assertEqual(len(chats), 1)
        job_id = chats[0].stem
        self.assertRegex(job_id, r"^support-bot-[0-9a-f]{4}$")
        records = ft.read_examples(chats[0])  # `vibehash train`'s own reader
        self.assertEqual(len(records), 45)
        self.assertEqual([m["role"] for m in records[0]["messages"]], ["user", "assistant"])
        self.assertEqual(records[3]["messages"][1]["content"], "Try step 3, then restart it. Happy to help!")
        self.assertNotIn(b"\r", chats[0].read_bytes())
        evals = json.loads((out_dir / f"{job_id}.eval.json").read_text(encoding="utf-8"))
        self.assertEqual((evals["job_id"], evals["base"], len(evals["prompts"])), (job_id, "qwen3-0.6b", 6))
        self.assertEqual(evals["designed_by"]["model"], MODEL)
        for f in out_dir.iterdir():
            self.assertNotIn(KEY, f.read_text(encoding="utf-8"))
        self.assertIn(architect.shown(architect.train_command({**good_plan(), "job_id": job_id}, chats[0])), said)
        self.assertIn("Nothing was posted, and no VIBE spent", said)
        self.assertNotIn(KEY, said + warned)

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
        for posts, said in ((True, "Posted bot-1"), (False, "Nothing was posted, so nothing was spent")):
            with self.subTest(posts=posts), mock.patch.object(architect.subprocess, "run", fake_run):
                job_id = f"bot-{int(posts)}"
                self.fake.script = [ok(good_plan())]
                out, _ = self.run_main("support bot", "--out", str(self.dir / "jobs"), "--id", job_id, "--post",
                                       "--node", "http://127.0.0.1:9", "--wallet", "req.json")
                cmd, env = ran[-1]
                self.assertEqual(cmd, architect.train_command({**good_plan(), "job_id": job_id},
                                                              self.dir / "jobs" / f"{job_id}.jsonl",
                                                              node="http://127.0.0.1:9", wallet="req.json"))
                self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(architect.REPO))
                self.assertIn("Posting to http://127.0.0.1:9", out)
                self.assertIn(said, out)
                if posts:
                    self.assertIn(f"job download {job_id} --out {job_id}-lora", out)

    def test_nothing_is_asked_when_the_id_is_already_taken(self):
        (Path(os.environ["VIBEHASH_HOME"]) / "jobs" / "mine").mkdir(parents=True)
        self.use_fake()
        with self.assertRaises(SystemExit) as caught:
            architect.main(["goal", "--id", "mine", "--out", str(self.dir / "jobs")])
        self.assertIn("already have a job called mine", str(caught.exception.code))
        with self.assertRaises(SystemExit) as caught:
            architect.main(["goal", "--base", "llama-3.2-1b", "--out", str(self.dir / "jobs")])
        self.assertIn("gated", str(caught.exception.code))
        self.assertEqual(self.fake.requests, [])  # nothing paid for

    @unittest.skipUnless(HAVE["torch"] and HAVE["tokenizers"] and HAVE["jinja2"], "needs PyTorch, tokenizers and jinja2")
    def test_vibehash_train_takes_the_files_up_to_its_preflight(self):
        """The command the architect builds, run by `vibehash train` itself against a fake Hugging Face serving a
        tiny Qwen3 with a Qwen3-style template (an empty think block before each reply), with --base pointed at it:
        it reads the chats with the model's tokenizer and template, finds every reply, sizes the job, shows the
        pre-flight, and is told no at "Post it?". Nothing reaches a node."""
        from tests.finetune_fixtures import TinyModel
        from tests.test_hub import FakeHub

        self.use_fake()
        self.fake.script = [ok(good_plan())]
        self.run_main("support bot", "--out", str(self.dir / "jobs"), "--id", "support-bot-1a2b", "--max-len", "256")
        chats = self.dir / "jobs" / "support-bot-1a2b.jsonl"
        cmd = architect.train_command({**good_plan(), "job_id": "support-bot-1a2b"}, chats, max_len=256)

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
        self.assertIn("45 examples (45 chats, in the model's chat template)", said)
        self.assertIn("LoRA rank 8", said)
        self.assertIn("a budget of 50 VIBE buys 5 rounds", said)
        self.assertIn("Nothing was posted, and nothing was spent.", said)
        self.assertFalse((Path(os.environ["VIBEHASH_HOME"]) / "jobs" / "support-bot-1a2b").exists())


class JudgeTests(AgentTest):
    def answers(self, n: int = 6) -> Path:
        path = self.dir / "support-bot-1a2b.answers.json"
        path.write_text(json.dumps({"goal": "two friendly sentences", "pairs": [
            {"prompt": f"question {i}?", "base": f"BAD {i}", "tuned": f"GOOD {i}"} for i in range(n)]}),
            encoding="utf-8")
        return path

    def test_file_names_and_lenient_verdicts(self):
        d = Path("x")
        self.assertEqual(judge.sibling(d / "bot.eval.json", ".answers.json"), d / "bot.answers.json")
        self.assertEqual(judge.sibling(d / "bot.answers.json", ".graded.json"), d / "bot.graded.json")
        self.assertEqual(judge.sibling(d / "v1.2.json", ".answers.json"), d / "v1.2.answers.json")
        for said, want in (("A", "A"), (" b. ", "B"), ("Reply A", "A"), ("TIE", "tie"), ("draw", "tie"),
                           ("maybe", None), (1, None), (None, None)):
            self.assertEqual(judge.winner_of({"winner": said}), want, said)
        self.assertIsNone(judge.winner_of(["A"]))

    def test_grading_is_blind_seeded_and_unshuffled(self):
        """The fake judge always picks the reply that says GOOD (the fine-tune's), except one tie and one reply it
        won't make A, B or tie even when asked again. The summary has to come out the same whichever letter the
        fine-tune was given, and the letters follow the seed."""
        self.use_fake()

        def verdict(body):
            text = body["messages"][1]["content"]
            if "question 4?" in text:
                return ok({"winner": "maybe", "reason": "unsure"})
            if "question 5?" in text:
                return ok({"winner": "tie", "reason": "both fine"})
            a = re.search(r"<reply_a>\n(.*?)\n</reply_a>", text, re.DOTALL).group(1)
            return ok(f"Verdict: {json.dumps({'winner': 'A' if a.startswith('GOOD') else 'B', 'reason': 'kinder'})}")
        self.fake.script = [verdict] * 8
        path = self.answers()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            graded = judge.grade(path, seed=3)
        self.assertEqual(graded, self.dir / "support-bot-1a2b.graded.json")
        data = json.loads(graded.read_text(encoding="utf-8"))
        s = data["summary"]
        self.assertEqual((s["prompts"], s["graded"], s["tuned_wins"], s["base_wins"], s["ties"], s["unjudged"]),
                         (6, 6, 4, 0, 1, 1))
        self.assertEqual((s["judge"]["model"], s["judge"]["base_url"], s["seed"]), (MODEL, self.fake.url, 3))
        rng = random.Random(3)
        self.assertEqual([p["verdict"]["tuned_was"] for p in data["pairs"]],
                         ["A" if rng.random() < 0.5 else "B" for _ in range(6)])
        self.assertEqual(len({p["verdict"]["tuned_was"] for p in data["pairs"]}), 2)  # both orders were used
        self.assertEqual(data["pairs"][4]["verdict"]["outcome"], "unjudged")
        self.assertEqual(data["pairs"][4]["verdict"]["said"], {"winner": "maybe", "reason": "unsure"})
        self.assertEqual(len(self.fake.requests), 7)  # pair 4 asked twice
        self.assertIn('exactly "A", "B" or "tie"', self.fake.requests[5]["body"]["messages"][-1]["content"])
        self.assertTrue(all(r["body"]["temperature"] == 0 for r in self.fake.requests))
        for r in self.fake.requests:  # blind: nothing says which reply is which
            self.assertNotRegex(r["body"]["messages"][1]["content"], r"tuned|base|fine-tune")
        self.assertNotIn(KEY, graded.read_text(encoding="utf-8"))
        self.assertIn("fine-tune preferred on 4 of 6 prompts", out.getvalue())

    def test_grading_stops_on_an_api_error_and_keeps_what_it_graded(self):
        self.use_fake()
        self.fake.script = [ok({"winner": "A", "reason": "r"}), ok({"winner": "B", "reason": "r"}),
                            refused(402, "Your balance is too low", kind="billing_error", code="billing_not_configured")]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(llm_call.LLMError, "2 of 6 graded"):
            judge.grade(self.answers())
        data = json.loads((self.dir / "support-bot-1a2b.graded.json").read_text(encoding="utf-8"))
        self.assertEqual(data["summary"]["graded"], 2)
        self.assertIn("can't pay", data["summary"]["stopped"])

    def adapter(self, trained: bool = True, seed: int = 0) -> tuple[Path, ft.LoraSpec]:
        """An adapter folder as `vibehash job download` writes one (finetune.export_peft), for a tiny Qwen3."""
        from tests.finetune_fixtures import CONFIGS

        spec = ft.LoraSpec(base="hf:tiny/qwen3@" + "b" * 40, manifest="a" * 64, dims=ft.arch_dims(CONFIGS["qwen3"]),
                           rank=4, alpha=8.0, seq_len=64)
        theta = np.zeros(spec.n_params)
        if trained:
            theta[:spec.n_adapter] = np.random.default_rng(seed).normal(0, 0.5, spec.n_adapter)
        return ft.export_peft(spec, theta, self.dir / f"lora-{int(trained)}", {"job_id": "support-bot-1a2b"}), spec

    def test_reading_an_adapter_folder(self):
        folder, _ = self.adapter()
        a = judge.read_adapter(folder)
        self.assertEqual((a["repo"], a["revision"], a["rank"], a["trained"]), ("tiny/qwen3", "b" * 40, 4, True))
        self.assertFalse(judge.read_adapter(self.adapter(trained=False)[0])["trained"])
        with self.assertRaisesRegex(SystemExit, "isn't an adapter folder"):
            judge.read_adapter(self.dir)
        evals = self.dir / "bot.eval.json"
        evals.write_text(json.dumps({"goal": "g", "base": "qwen3-0.6b", "prompts": ["hi?"]}))
        with self.assertRaisesRegex(SystemExit, "is for Qwen/Qwen3-0.6B, but .* is an adapter for tiny/qwen3"):
            judge.answer(evals, folder)

    @unittest.skipUnless(all(HAVE.values()), "needs PyTorch, transformers, peft, tokenizers and jinja2")
    def test_answer_runs_base_and_fine_tune(self):
        """A tiny random Qwen3 saved as a model folder, its tokenizer with a Qwen3-style template, and a trained
        adapter for it: both answer every prompt on the CPU, with no accelerate and no device_map, and the adapter
        changes the answers."""
        from tests.finetune_fixtures import TinyModel

        tiny = TinyModel(self.dir / "src")
        (tiny.folder / "tokenizer.json").write_text(word_tokenizer(tiny.vocab), encoding="utf-8")
        (tiny.folder / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "chat_template": QWEN3ISH, "eos_token": "<|im_end|>",
            "pad_token": "[UNK]"}), encoding="utf-8")
        folder, _ = self.adapter()
        evals = self.dir / "support-bot-1a2b.eval.json"
        evals.write_text(json.dumps({"goal": "short answers", "job_id": "support-bot-1a2b",
                                     "prompts": ["how do i fix it", "what is the answer", "can you try please"]}),
                         encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            out = judge.answer(evals, folder, base=str(tiny.folder), max_new_tokens=6, device="cpu")
        self.assertEqual(out, self.dir / "support-bot-1a2b.answers.json")
        self.assertIn("a commit this can not confirm", err.getvalue())  # a local folder without hf's notes
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([p["prompt"] for p in data["pairs"]], ["how do i fix it", "what is the answer",
                                                                "can you try please"])
        self.assertEqual((data["generation"]["device"], data["generation"]["dtype"]), ("cpu", "float32"))
        self.assertTrue(data["adapter"]["trained"])
        self.assertTrue(any(p["base"] != p["tuned"] for p in data["pairs"]), data["pairs"])
        self.assertNotIn("<think>", out.read_text(encoding="utf-8"))
        prompt = judge.Renderer(__import__("transformers").AutoTokenizer.from_pretrained(str(tiny.folder)))("hi")
        self.assertEqual(prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


if __name__ == "__main__":
    unittest.main()
