"""The model catalog (vibehash/catalog.py), `vibehash models` and `vibehash train --base`, against the fake Hub in
test_hub.py. Nothing here reaches huggingface.co: $HF_ENDPOINT points at the fake, and the default endpoint at a port
nothing listens on."""
import contextlib
import dataclasses
import functools
import importlib.util
import io
import json
import logging
import os
import re
import socket
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.test_hub import FakeHub, HubTestCase, shard1, shard2, sharded_files, single_files
from vibehash import catalog, cli, hub
from vibehash.catalog import Facts, Model

ROOT = Path(__file__).resolve().parent.parent
NOWHERE = "http://127.0.0.1:9"  # the discard port: refused at once


def facts(weight_bytes: int, layers: int | None, **kw) -> Facts:
    base = dict(repo="acme/x", commit="0" * 40, parameters=None, weight_bytes=weight_bytes, weight_files=1,
                estimated=False, licence="mit", gated=False, architecture=None, dtype=None, layers=layers, experts=None,
                experts_per_token=None, has_tokenizer=True, pickle_only=False, checked=time.time())
    return Facts(**{**base, **kw})


class CatalogTests(unittest.TestCase):
    def test_the_catalog(self):
        names = [m.name for m in catalog.CATALOG]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(8 <= len(names) <= 20, names)
        for m in catalog.CATALOG:
            self.assertEqual(m.name, m.name.lower())
            self.assertEqual(hub.check_repo(m.repo), m.repo)
            self.assertEqual(catalog.find(m.name.upper()), m)
            self.assertEqual(catalog.find(m.repo.lower()), m)
            self.assertEqual(catalog.resolve(m.url), (m.repo, m))
        for name in ("qwen3-0.6b", "llama-3.2-1b", "qwen3.8-27b", "glm-5.3", "glm-5.3-flash", "deepseek-v4.1-flash"):
            self.assertIn(name, names)
        self.assertEqual(names[0], "qwen3-0.6b")  # one gaming GPU's worth first, as DEEPSEEK.md's build order has it
        self.assertEqual(names[-1], "deepseek-v4.1-flash")

    def test_what_people_type_as_a_base(self):
        deepseek = catalog.find("deepseek-v4.1-flash")
        self.assertEqual(catalog.resolve("DeepSeek-V4.1-Flash"), (deepseek.repo, deepseek))
        self.assertEqual(catalog.resolve("deepseek-ai/DeepSeek-V4.1-Flash"), (deepseek.repo, deepseek))
        self.assertEqual(catalog.resolve("hf:acme/tiny"), ("acme/tiny", None))
        self.assertEqual(catalog.resolve("https://huggingface.co/acme/tiny/tree/v2"), ("acme/tiny@v2", None))
        self.assertEqual(catalog.resolve("deepseek-ai/DeepSeek-V4.1-Flash@main"), (deepseek.repo + "@main", None))
        with self.assertRaisesRegex(ValueError, "did you mean qwen3-8b, .*`vibehash models` lists them"):
            catalog.resolve("qwen3")
        with self.assertRaisesRegex(ValueError, "no model called 'nothing-like-it'"):
            catalog.resolve("nothing-like-it")
        with self.assertRaises(ValueError):
            catalog.resolve("https://example.com/acme/tiny")

    def test_what_a_team_needs_is_deepseek_mds_maths(self):
        deepseek = catalog.find("deepseek-v4.1-flash")
        f = facts(510_296_708_312, 40)  # DEEPSEEK.md §1: 510.3 GB, 40 layers
        self.assertEqual(round(catalog.held_bytes(f, deepseek) / 1e9, 1), 298.3)  # less the tables and draft layers
        self.assertEqual(catalog.team(f, deepseek), {12: 40, 24: 20, 32: 14})  # DEEPSEEK.md §2's table
        self.assertEqual(catalog.team(facts(1_503_300_328, 28)), {12: 1, 24: 1, 32: 1})  # Qwen3 0.6B on any card
        self.assertEqual(catalog.team(facts(755_632_050_320, 78)), {12: None, 24: 78, 32: 39})  # a layer > 9 GB
        self.assertEqual(catalog.gpus(30 * 10**9, None, 12), 4)  # no layer count: shared out evenly
        self.assertEqual(catalog.team(facts(None, 4)), {})
        self.assertEqual(catalog.team_text({12: None, 24: 78, 32: 39}), "- / 78 / 39")


class CatalogTestCase(HubTestCase):
    """HubTestCase's fake Hub as $HF_ENDPOINT, so every Hub() the catalog makes talks to it, and a catalog of models
    that live there."""

    def setUp(self):
        super().setUp()  # its environment patch restores everything set here
        os.environ["HF_ENDPOINT"] = self.fake.endpoint
        for patch in (mock.patch.object(hub, "DEFAULT_ENDPOINT", NOWHERE),
                      mock.patch.object(catalog, "Hub", functools.partial(hub.Hub, backoff=0.001)),
                      mock.patch.object(catalog, "CATALOG", (Model("tiny", "Tiny", "acme/tiny"),
                                                             Model("moe", "Sharded MoE", "acme/sharded")))):
            patch.start()
            self.addCleanup(patch.stop)
        self.single()
        self.sharded()
        logging.disable(logging.INFO)  # cli.main turns INFO logging on; the Hub logs its retries there
        self.addCleanup(logging.disable, logging.NOTSET)

    def gated_with_hidden_hashes(self) -> str:
        """A gated model as the real Hub lists it to an account that hasn't accepted the licence: every LFS file's
        sha256 (and its xet hash) replaced by asterisks."""
        self.fake.add_commit("acme/gated", sharded_files(), gated=True, licence="acme-community")
        tree = FakeHub._tree

        def masked(fake, files):
            out = tree(fake, files)
            for e in out:
                if "lfs" in e:
                    e["lfs"]["oid"] = e["xetHash"] = "*" * 64
            return out
        patch = mock.patch.object(FakeHub, "_tree", masked)
        patch.start()
        self.addCleanup(patch.stop)
        return "acme/gated"

    def run_cli(self, *argv, stdin: str | None = None) -> tuple[object, str, str]:
        """(exit code or message, stdout, stderr). `stdin` makes both ends look like a terminal, with that typed."""
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        out, err = (Terminal() if stdin is not None else io.StringIO()), io.StringIO()
        typed = Terminal(stdin) if stdin is not None else io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch("sys.stdin", typed):
            try:
                cli.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()


class DescribeTests(CatalogTestCase):
    def test_facts_from_the_hub(self):
        f = catalog.describe("acme/tiny")
        self.assertEqual((f.repo, f.licence, f.gated, f.estimated, f.weight_files), ("acme/tiny", "mit", False, False, 1))
        self.assertEqual(f.parameters, 2048 + 1024 + 1 + 0 + 2048 + 32768 + 128 + 32)
        self.assertEqual(f.weight_bytes, len(single_files()["model.safetensors"]))
        self.assertEqual((f.layers, f.experts, f.experts_per_token), (2, 8, 2))
        self.assertEqual(Facts.from_dict(json.loads(json.dumps(dataclasses.asdict(f)))), f)  # survives the cache

    def test_a_language_model_inside_a_multimodal_config(self):
        """Gemma 4, Qwen 3.5 and DeepSeek-V4.1 keep their shape in config.json's text_config."""
        files = single_files()
        files["config.json"] = json.dumps({"architectures": ["AcmeForConditionalGeneration"], "text_config": {
            "num_hidden_layers": 5, "num_experts": 16, "num_experts_per_tok": 4, "dtype": "bfloat16"}}).encode()
        self.fake.add_commit("acme/vision", files)
        f = catalog.describe("acme/vision")
        self.assertEqual((f.layers, f.experts, f.experts_per_token, f.dtype), (5, 16, 4, "bfloat16"))

    def test_a_gated_model_whose_files_are_hidden(self):
        repo = self.gated_with_hidden_hashes()
        f = catalog.describe(repo)
        self.assertEqual((f.gated, f.licence), (True, "acme-community"))
        self.assertEqual(f.parameters, 2048 + 1024 + 1 + 0 + 2048 + 32768 + 128 + 32)
        if f.estimated:  # until Hub.describe copes with hidden hashes itself: from the Hub's count per dtype
            self.assertEqual(f.weight_bytes, sum(len(raw) for _, _, raw in {**shard1(), **shard2()}.values()))
        out = catalog.preflight(repo, None, f)
        self.assertIn(f"gated: accept it at https://huggingface.co/{repo}, then set HF_TOKEN", out)

    def test_a_missing_model_says_what_to_check(self):
        with self.assertRaisesRegex(hub.AccessError, "no model acme/missing"):
            catalog.describe("acme/missing")


class LookupTests(CatalogTestCase):
    def test_answers_are_kept_for_a_day(self):
        refs = ["acme/tiny", "acme/sharded"]
        facts, errors = catalog.lookup(refs)
        self.assertEqual((set(facts), errors), (set(refs), {}))
        asked = len(self.fake.hub_auth)
        self.assertEqual(catalog.lookup(refs)[0], facts)
        self.assertEqual(len(self.fake.hub_auth), asked, "asked the Hub again within the day")
        catalog.lookup(refs, refresh=True)
        self.assertGreater(len(self.fake.hub_auth), asked)
        cached = json.loads(catalog.cache_path().read_text())
        cached["models"]["acme/tiny"]["checked"] -= 2 * catalog.CACHE_TTL
        catalog.cache_path().write_text(json.dumps(cached))
        asked = len(self.fake.hub_auth)
        facts, _ = catalog.lookup(refs)
        self.assertGreater(len(self.fake.hub_auth), asked)
        self.assertGreater(facts["acme/tiny"].checked, time.time() - 60)
        catalog.cache_path().write_text("{not json")
        self.assertEqual(set(catalog.lookup(refs)[0]), set(refs))  # a broken cache is just asked again

    def test_without_the_hub_the_cache_is_used_however_old(self):
        catalog.lookup(["acme/tiny"])
        cached = json.loads(catalog.cache_path().read_text())
        cached["models"]["acme/tiny"]["checked"] -= 30 * catalog.CACHE_TTL
        catalog.cache_path().write_text(json.dumps(cached))
        os.environ["HF_ENDPOINT"] = NOWHERE
        facts, errors = catalog.lookup(["acme/tiny", "acme/sharded"])
        self.assertEqual(set(facts), {"acme/tiny"})  # the old answer, and nothing for the one never asked about
        self.assertEqual(set(errors), {"acme/tiny", "acme/sharded"})
        self.assertIn("can't reach Hugging Face", errors["acme/sharded"])
        text = catalog.table(facts, errors)
        self.assertIn("tiny: couldn't ask Hugging Face (can't reach", text)
        self.assertIn("showing what it said on", text)


    def test_a_hub_out_of_reach_costs_seconds_not_minutes(self):
        """A Hub that takes connections and never answers, as a blocking network does: one short timeout, no retries,
        and once that's seen, the models not yet asked about aren't."""
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen(64)  # connections are accepted by the system, and then nothing is ever said
        self.addCleanup(silent.close)
        os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{silent.getsockname()[1]}"
        refs = [f"acme/model-{i}" for i in range(17)]
        with mock.patch.object(catalog, "TIMEOUT_S", 0.5):
            start = time.monotonic()
            facts, errors = catalog.lookup(refs)
            took = time.monotonic() - start
        self.assertEqual((facts, set(errors)), ({}, set(refs)))
        self.assertTrue(all("can't reach Hugging Face" in e for e in errors.values()), errors)
        self.assertLess(took, 3.0)  # 17 models, 8 at a time: three rounds of timeouts with retries took minutes


class ModelsCommandTests(CatalogTestCase):
    def test_models_lists_the_catalog(self):
        os.environ["COLUMNS"] = "200"
        code, out, _ = self.run_cli("models")
        self.assertEqual(code, 0, out)
        self.assertIn("Status: today the network trains small models sized to a CSV file only", out)
        self.assertIn("Fine-tuning these open-weight models is coming, not live yet.", out)
        self.assertRegex(out, r"name\s+Hugging Face\s+parameters\s+experts\s+weights\s+licence\s+gated\s+GPUs at 12 / 24 / 32 GB")
        self.assertRegex(out, r"tiny\s+acme/tiny\s+38,049\s+2 of 8\s+\S+ kB\s+mit\s+no\s+1 / 1 / 1")
        self.assertIn("moe ", out)
        self.assertIn("train --base NAME` shows what training one would take; it posts nothing", out)
        os.environ["COLUMNS"] = "70"
        out = self.run_cli("models")[1]
        self.assertNotIn("acme/tiny", out)  # too narrow for the repositories
        self.assertIn("models --json` gives each model's Hugging Face page", out)

    def test_models_as_json(self):
        code, out, _ = self.run_cli("models", "--json")
        d = json.loads(out)
        self.assertEqual(code, 0)
        self.assertIn("CSV", d["today"])
        self.assertEqual(d["gpu_gb"], [12, 24, 32])
        self.assertEqual([m["name"] for m in d["models"]], ["tiny", "moe"])
        tiny = d["models"][0]
        self.assertEqual((tiny["repo"], tiny["status"], tiny["licence"], tiny["gated"], tiny["experts"], tiny["error"]),
                         ("acme/tiny", "coming", "mit", False, 8, None))
        self.assertEqual(tiny["team"], {"12": 1, "24": 1, "32": 1})
        self.assertEqual(len(tiny["commit"]), 40)

    def test_with_no_hub_and_nothing_cached_it_says_so(self):
        os.environ["HF_ENDPOINT"] = NOWHERE
        code, out, _ = self.run_cli("models")
        self.assertEqual(code, 1)
        self.assertIn("tiny: couldn't ask Hugging Face (can't reach Hugging Face", out)


class TrainBaseTests(CatalogTestCase):
    def setUp(self):
        super().setUp()
        # Anything on the way to posting a job or spending VIBE fails the test.
        for name in ("NodeClient", "load_wallet", "wallet_of"):
            patch = mock.patch.object(cli, name, side_effect=AssertionError(f"train --base reached {name}"))
            patch.start()
            self.addCleanup(patch.stop)

    def assert_nothing_posted(self, code, out):
        self.assertEqual(code, 1, out)
        self.assertIn("Not yet: fine-tuning open-weight models isn't live on the network, so nothing was posted and "
                      "nothing was spent.", out)
        self.assertIn("What works today: a small model sized to a CSV file", out)
        self.assertFalse((self.tmp / "home" / "jobs").exists())

    def test_a_catalog_model(self):
        code, out, _ = self.run_cli("train", "--base", "tiny")
        self.assert_nothing_posted(code, out)
        self.assertIn("Model     tiny: acme/tiny", out)
        self.assertIn("TinyForCausalLM, 38,049 parameters, a mixture of 8 experts (2 per token), 2 layers", out)
        self.assertRegex(out, r"Weights   \S+ kB in 1 safetensors file \(bfloat16, fp8 \(e4m3\)\)")
        self.assertIn("Licence   mit, not gated", out)
        self.assertIn("GPUs      one 12 GB card holds its", out)

    def test_any_hugging_face_model_and_the_data(self):
        self.fake.add_commit("acme/other", single_files(), licence="apache-2.0")
        chats = self.tmp / "chats.jsonl"
        chats.write_bytes(b'{"messages": [{"role": "user", "content": "hi"}]}\n' * 100)  # 5,000 bytes
        for base in ("acme/other", f"{self.fake.endpoint}/acme/other", "https://huggingface.co/acme/other", "hf:acme/other"):
            code, out, _ = self.run_cli("train", str(chats), "--base", base)
            self.assert_nothing_posted(code, out)
            self.assertIn("Model     acme/other\n", out)
            self.assertIn("Licence   apache-2.0, not gated", out)
            self.assertIn("Data      chats.jsonl: 5.0 kB of JSON lines", out)

    def test_a_gated_model(self):
        repo = self.gated_with_hidden_hashes()
        code, out, _ = self.run_cli("train", "--base", repo)
        self.assert_nothing_posted(code, out)
        self.assertIn(f"Licence   acme-community, gated: accept it at https://huggingface.co/{repo}", out)

    def test_names_it_cant_find(self):
        code, out, err = self.run_cli("train", "--base", "tiyn")
        self.assertIn("no model called 'tiyn' in the catalog: did you mean tiny?", str(code))
        code, out, err = self.run_cli("train", "--base", "acme/missing")
        self.assertIn("no model acme/missing", str(code))
        code, out, err = self.run_cli("train", "missing.jsonl", "--base", "tiny")
        self.assertIn("missing.jsonl doesn't exist", str(code))

    def test_the_menu_on_a_terminal(self):
        code, out, _ = self.run_cli("train", stdin="3\n")
        self.assertIn("Which model?", out)
        self.assertRegex(out, r"1  a small model sized to your CSV data\s+works today")
        self.assertRegex(out, r"2  tiny\s+coming")
        self.assertRegex(out, r"3  moe\s+coming")
        self.assert_nothing_posted(code, out)
        self.assertIn("Model     moe: acme/sharded", out)  # looked up now that it's chosen
        code, out, _ = self.run_cli("train", stdin="99\nTINY\n")
        self.assertIn("Pick a number from 1 to 3.", out)
        self.assertIn("Model     tiny: acme/tiny", out)

    def test_the_menu_asks_no_one(self):
        """It opens at once, even where Hugging Face is out of reach: it shows what's cached, and only the model
        picked is looked up, in its pre-flight."""
        code, out, _ = self.run_cli("train", stdin="1\n\n")
        self.assertIn("Which model?", out)
        self.assertEqual(self.fake.hub_auth, [], "the menu asked Hugging Face")
        catalog.lookup(["acme/tiny"])
        out = self.run_cli("train", stdin="1\n\n")[1]
        self.assertRegex(out, r"2  tiny\s+38,049, \S+ kB, mit\s+coming")  # what the cache knows
        self.assertRegex(out, r"3  moe\s+coming")
        code, out, _ = self.run_cli("train", stdin="")  # no answer at all (Windows counts NUL as a terminal)
        self.assertIn("no answer: `", str(code))

    def test_the_menu_leads_to_todays_model(self):
        code, out, _ = self.run_cli("train", stdin="\n\n")  # Enter: the small CSV model, then no file
        self.assertIn("CSV file to train on:", out)
        self.assertIn("example` writes a sample, fruit.csv, to try", str(code))
        code, out, _ = self.run_cli("train", stdin="1\n" + str(self.tmp / "no-such.csv") + "\n")
        self.assertIn("Which model?", out)
        self.assertIn("no-such.csv", str(code))  # on to today's CSV training, which looks for the file

    def test_a_data_file_goes_to_todays_training_as_before(self):
        """Without --base, any data file is read as a table, whatever its name, on a terminal or not: no menu."""
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["example", "--dir", str(self.tmp)])
        export = self.tmp / "export.txt"
        export.write_bytes((self.tmp / "fruit.csv").read_bytes().replace(b"fruit", b"species", 1))
        for argv, stdin in (([str(self.tmp / "fruit.csv")], ""), ([str(export), "--label", "species"], None),
                            ([str(export), "--label", "species"], ""), ([str(export)], "")):
            with self.subTest(argv=argv, terminal=stdin is not None), \
                    mock.patch.object(cli, "_pick_model", side_effect=AssertionError("showed the menu")), \
                    self.assertRaisesRegex(AssertionError, "reached wallet_of"):  # read, and on to posting it
                self.run_cli("train", *argv, stdin=stdin)

    def test_text_that_isnt_a_table_hints_at_fine_tuning(self):
        notes = self.tmp / "notes.txt"
        notes.write_text("hello\n")
        code, out, _ = self.run_cli("train", str(notes))
        self.assertRegex(str(code), r"^error: ")  # the table reader's own error, as before...
        self.assertIn("notes.txt may be text rather than a table. Training on text means fine-tuning an open-weight "
                      "model, which is coming: `", str(code))  # ...and a hint
        code, out, _ = self.run_cli("train")
        self.assertRegex(str(code), r"which data\? .*train data.csv` trains a small model sized to a CSV file")


class GuideTests(unittest.TestCase):
    def test_the_catalog_is_baked_into_the_guide(self):
        """scripts/bake_models.py writes the catalog into the page; this fails when the catalog changes without it."""
        page = (ROOT / "docs" / "guide.html").read_text(encoding="utf-8")
        block = re.search(r"// models:begin\n(.*?)\n *// models:end", page, re.S)[1]
        rows = [json.loads(line.strip().rstrip(",")) for line in block.splitlines() if line.strip().startswith("{")]
        self.assertEqual([(r["name"], r["repo"]) for r in rows], [(m.name, m.repo) for m in catalog.CATALOG])
        for r in rows:
            self.assertEqual(len(r["team"]), len(catalog.GPU_GB), r)
            self.assertTrue(r["params"] and r["weights"] and r["licence"], r)
        self.assertRegex(block, r'MODELS_CHECKED = "\d{4}-\d\d-\d\d"')
        self.assertIn("A small model sized to your data (works today)", page)

    def test_what_the_hub_says_cant_break_out_of_the_page_script(self):
        """A licence name is anyone's text: a `</script>` in it must not end the guide's inline script."""
        spec = importlib.util.spec_from_file_location("bake_models", ROOT / "scripts" / "bake_models.py")
        bake = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bake)
        hostile = "</script><script>alert(1)</script> & <!-- é"
        lines = bake.block({m.repo: facts(10**9, 4, repo=m.repo, licence=hostile) for m in catalog.CATALOG})
        text = "\n".join(lines)
        for bad in ("<", ">", "&"):
            self.assertNotIn(bad, text)
        rows = [json.loads(line.strip().rstrip(",")) for line in lines if line.strip().startswith("{")]
        self.assertEqual({r["licence"] for r in rows}, {hostile})  # the same text, once the script reads it


if __name__ == "__main__":
    unittest.main()
