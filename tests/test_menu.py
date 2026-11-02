import contextlib
import io
import re
import unittest
from unittest import mock

from vibehash import cli, menu, termui


def no_wallet():
    return mock.patch.object(menu.settings, "wallet_path", return_value=None)


class FakeKeys:
    def __init__(self, keys):
        self.keys = list(keys)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def read(self, timeout):
        return self.keys.pop(0) if self.keys else "quit"


@contextlib.contextmanager
def screen():
    yield io.StringIO()


class MenuTests(unittest.TestCase):
    def pick(self, keys, start=0):
        with no_wallet(), mock.patch.object(menu, "Keys", lambda: FakeKeys(keys)), \
                mock.patch.object(menu.termui, "alternate_screen", screen):
            return menu.pick(start, menu.Status(), "vibehash")

    def test_arrows_numbers_and_quit(self):
        labels = [i.label for i in menu.items()]
        self.assertEqual(labels[self.pick(["down", "down", "enter"])], "What can this computer train on?")
        self.assertIsNone(self.pick(["up", "enter"]))  # up from the top wraps to Quit
        self.assertEqual(labels[self.pick(["up", "up", "enter"])], "More commands")
        self.assertEqual(self.pick(["4", "enter"]), 3)
        self.assertIsNone(self.pick(["end", "enter"]))  # Quit
        self.assertIsNone(self.pick(["down", "quit"]))
        self.assertEqual(self.pick([None, "enter"], start=2), 2)  # a timeout just redraws

    def test_frame_shows_every_item_and_the_command_behind_the_chosen_one(self):
        with no_wallet():
            status = menu.Status()
        plain = lambda text: re.sub(r"\x1b\[[0-9;]*[mKJH]", "", text)  # noqa: E731
        text = plain(menu.frame(0, status, "python -m vibehash", 120, termui.ASCII))
        for item in menu.items():
            self.assertIn(item.label, text)
        self.assertIn("same as:  python -m vibehash worker", text)
        self.assertIn("No wallet yet", text)
        quit_at = len(menu.items()) - 1
        self.assertIn("same as:  —", plain(menu.frame(quit_at, status, "vibehash", 120, termui.ASCII)))

    def test_more_commands_is_a_page_of_its_own(self):
        ran, main, more = [], [i.label for i in menu.items()], [i.label for i in menu.items(menu.MORE, menu.BACK)]
        picks = iter([main.index("More commands"), more.index("Send VIBE"), None, None])  # in, send, Back, Quit
        answers = iter(["vibe1friend", "2.5", "y", ""])  # to, amount, confirm, then "back to the menu"
        with no_wallet(), mock.patch.object(menu, "pick", lambda *a: next(picks)),                 mock.patch.object(menu.termui, "supported", return_value=True),                 mock.patch("builtins.input", lambda *a: next(answers)), contextlib.redirect_stdout(io.StringIO()):
            menu.run(ran.append, "vibehash")
        self.assertEqual(ran, [["send", "--to", "vibe1friend", "--amount", "2.5"]])
        for label in ("Stake more", "Unstake", "Restore a wallet", "Predict with my model", "Validators",
                      "Verify the whole chain", "Install GPU support", "Every command"):
            self.assertIn(label, more)
        with no_wallet():
            page = menu.frame(more.index("Send VIBE"), menu.Status(), "vibehash", 120, termui.ASCII, menu.MORE, menu.BACK)
        page = re.sub(r"\x1b\[[0-9;]*[mKJH]", "", page)
        self.assertIn("same as:  vibehash send --to ADDRESS --amount VIBE", page)
        self.assertIn("q back", page)

    def test_check_this_computer_runs_the_selftest(self):
        labels = [i.label for i in menu.items()]
        item = menu.items()[labels.index("Check this computer")]
        self.assertEqual(item.argv, ("selftest",))
        self.assertIn("Mine", [title for title, group in menu.SECTIONS if item in group])
        with no_wallet():
            page = menu.frame(labels.index("Check this computer"), menu.Status(), "vibehash", 120, termui.ASCII)
        self.assertIn("same as:  vibehash selftest", re.sub(r"\x1b\[[0-9;]*[mKJH]", "", page))

    def test_fine_tune_asks_for_the_chats_and_a_model(self):
        answers = iter(["my-chats.jsonl", "", ""])
        with mock.patch("builtins.input", lambda *a: next(answers)):
            self.assertEqual(menu._finetune(), [["train", "my-chats.jsonl", "--base", "qwen3-0.6b", "--budget", "100"]])
        with mock.patch("builtins.input", lambda *a: ""):
            self.assertIsNone(menu._finetune())  # no file, nothing posted

    def test_nothing_is_sent_without_a_yes(self):
        answers = iter(["vibe1friend", "2.5", ""])
        with mock.patch("builtins.input", lambda *a: next(answers)):
            self.assertIsNone(menu._send())

    def test_keys_by_name(self):
        self.assertEqual([menu.Keys.name(c) for c in "\r\nqk j7x"], ["enter", "enter", "quit", "up", None, "down", "7", None])
        with self.assertRaises(KeyboardInterrupt):
            menu.Keys.name("\x03")

    def test_run_executes_the_chosen_commands_and_comes_back(self):
        ran, labels = [], [i.label for i in menu.items()]
        train = labels.index("Train a model on a CSV file")
        picks = iter([0, train, None])
        answers = iter(["", "", "", ""])  # train: sample data, default budget; then "back to the menu" twice
        with no_wallet(), mock.patch.object(menu, "pick", lambda *a: next(picks)), \
                mock.patch.object(menu.termui, "supported", return_value=True), \
                mock.patch("builtins.input", lambda *a: next(answers)), contextlib.redirect_stdout(io.StringIO()) as out:
            menu.run(ran.append, "vibehash")
        self.assertEqual(ran, [["worker"], ["example"], ["train", "fruit.csv", "--budget", "100"]])
        self.assertIn("> vibehash train fruit.csv --budget 100", out.getvalue())

    def test_plain_vibehash_opens_the_menu_only_at_a_terminal(self):
        with mock.patch.object(menu, "run") as run, mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch("sys.stdout.isatty", return_value=True):
            cli.main([])
        run.assert_called_once()
        with mock.patch.object(menu, "run") as run, mock.patch("sys.stdin.isatty", return_value=False), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main([])  # piped or scripted: the usual usage error, never a menu waiting for keys
        run.assert_not_called()

    def test_a_failing_command_doesnt_end_the_menu(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as out:
            cli.run_command(["no-such-command"])
        self.assertIn("stopped with exit code 2", out.getvalue())


if __name__ == "__main__":
    unittest.main()
