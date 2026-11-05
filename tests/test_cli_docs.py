"""docs/CLI.md is written from the parser by scripts/cli_docs.py; this fails when the two drift apart."""
import importlib.util
import unittest
from pathlib import Path

from vibehash import cli

ROOT = Path(__file__).resolve().parent.parent


def load_generator():
    spec = importlib.util.spec_from_file_location("cli_docs", ROOT / "scripts" / "cli_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliDocs(unittest.TestCase):
    def setUp(self):
        self.gen = load_generator()
        self.text = self.gen.render()

    def test_docs_match_parser(self):
        current = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(current, self.text, "docs/CLI.md is out of date: run python scripts/cli_docs.py")

    def test_every_command_documented(self):
        for name, _ in self.gen.leaves(cli.parser()):
            self.assertIn(f"# vibehash {name}\n", self.text, f"`vibehash {name}` has no section")

    def test_nothing_local_leaks(self):
        self.assertNotIn(str(Path.home()), self.text)

    def test_cache_choices_each_get_a_line(self):
        for choice in ("show", "clear", "sweep"):
            self.assertIn(f"  - `{choice}`: ", self.text)

    def test_examples_name_real_commands(self):
        names = {name for name, _ in self.gen.leaves(cli.parser())}
        for name in self.gen.EXAMPLES:
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
