"""The app's window (window.py): the dashboard page in a web view of its own, the app behind it through Api, the
browser as the fallback."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vibehash import app, settings, window


def fake_webview(start_raises=None):
    """A stand-in for the pywebview package: records the window asked for and how start was called."""
    m = mock.MagicMock()
    m.windows = []

    def create_window(title, html=None, **kw):
        w = SimpleNamespace(title=title, html=html, kw=kw)
        m.windows.append(w)
        return w

    def start(**kw):
        m.started = kw
        if start_raises:
            raise start_raises

    m.create_window, m.start = create_window, start
    return m


class Home(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name, "DISPLAY": ":0",
                                              "APPDATA": self.tmp.name + "/AppData"})  # never this machine's Startup folder
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("VIBEHASH_LAUNCHER", None)

    def alive(self, **more):
        app.app_dir().mkdir(parents=True, exist_ok=True)
        state = {"beat": time.time(), "started": time.time() - 5, "pid": 1, "paused": False, "watch": "http://127.0.0.1:7545/",
                 "worker": {"running": True}, "setup": {"step": "mining", "fraction": 1.0}, "events": []} | more
        app.state_path().write_text(json.dumps(state))
        return state


class WindowTests(Home):
    def test_the_command_is_this_python_or_the_launcher(self):
        self.assertEqual(window.command(7000, env={}), [sys.executable, "-m", "vibehash", "window", "--watch", "7000"])
        exe = Path(self.tmp.name) / "VibeHash.exe"
        exe.write_text("")
        self.assertEqual(window.command(7000, env={"VIBEHASH_LAUNCHER": str(exe)}), [str(exe), "--window", "7000"])
        self.assertEqual(window.command(7000, env={"VIBEHASH_LAUNCHER": str(exe) + ".gone"})[1:3], ["-m", "vibehash"])  # not there: python
        self.assertEqual(window.dashboard_url(7000), "http://127.0.0.1:7000/dashboard")

    def test_the_page_is_the_dashboard_told_where_the_feed_is(self):
        text = window.page(7000)
        self.assertIn('<script>window.FEED = "http://127.0.0.1:7000";</script>', text)
        self.assertIn('fetch(FEED + "/"', text)
        self.assertIn("pywebviewready", text)
        self.assertIn('id="welcome"', text)
        self.assertIn('id="set-avatar"', text)

    def test_the_window_is_the_page_with_the_api_served_from_a_port_of_its_own(self):
        wv = fake_webview()
        said = []
        with mock.patch.dict(sys.modules, {"webview": wv}):
            how = window.show(7000, fallback=lambda url: said.append(url), say=said.append)
        self.assertEqual(how, "window")
        self.assertEqual(said, [])
        w = wv.windows[0]
        self.assertEqual(w.title, window.TITLE)
        self.assertIn('window.FEED = "http://127.0.0.1:7000"', w.html)
        self.assertIsInstance(w.kw["js_api"], window.Api)
        self.assertEqual(w.kw["js_api"].port, 7000)
        self.assertEqual((w.kw["width"], w.kw["height"], w.kw["min_size"]), (window.SIZE[0], window.SIZE[1], window.MIN_SIZE))
        self.assertTrue(wv.started["http_server"])  # an http origin, which the feed's CORS allows
        self.assertTrue(window.Lock(7000).acquire())  # the lock was let go when the window closed

    def test_without_pywebview_the_browser_gets_the_feeds_page(self):
        opened, said = [], []
        with mock.patch.dict(sys.modules, {"webview": None}):  # import webview -> ImportError
            how = window.show(7545, fallback=opened.append, say=said.append)
        self.assertEqual(how, "browser")
        self.assertEqual(opened, ["http://127.0.0.1:7545/dashboard"])
        self.assertIn("pywebview", said[0])

    def test_a_web_view_that_cannot_start_falls_back_to_the_browser(self):
        opened, said = [], []
        wv = fake_webview(start_raises=RuntimeError("You must have either QT or GTK with Python extensions installed"))
        with mock.patch.dict(sys.modules, {"webview": wv}):
            how = window.show(7545, fallback=opened.append, say=said.append)
        self.assertEqual(how, "browser")
        self.assertEqual(opened, ["http://127.0.0.1:7545/dashboard"])
        self.assertIn("QT or GTK", said[0])

    def test_one_window_per_feed(self):
        first = window.Lock(7545)
        self.assertTrue(first.acquire())
        self.assertFalse(window.Lock(7545).acquire())  # the same feed: no
        self.assertTrue(window.Lock(7546).acquire())  # another feed: fine
        opened, said = [], []
        with mock.patch.dict(sys.modules, {"webview": fake_webview()}):
            self.assertEqual(window.show(7545, fallback=opened.append, say=said.append), "open")
        self.assertEqual((opened, said), ([], ["the window is already open"]))
        first.release()
        self.assertTrue(window.Lock(7545).acquire())  # let go: the next one gets it

    def test_no_screen_on_linux_means_no_window_and_no_console_browser(self):
        opened, said = [], []
        with mock.patch.object(window.sys, "platform", "linux"), \
                mock.patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}), \
                mock.patch.dict(sys.modules, {"webview": None}):
            self.assertEqual(window.show(7545, fallback=opened.append, say=said.append), "none")
        self.assertEqual(opened, [])
        self.assertIn("needs a screen", said[0])

    def test_the_cli_has_it(self):
        from vibehash import cli
        a = cli.parser().parse_args(["window", "--watch", "7000"])
        self.assertEqual((a.fn, a.watch), (cli.cmd_window, 7000))
        self.assertEqual(cli.parser().parse_args(["window"]).watch, window.DEFAULT_PORT)
        self.assertTrue(cli.parser().parse_args(["app", "--no-window"]).no_window)
        self.assertFalse(cli.parser().parse_args(["app", "start"]).no_window)


class ApiTests(Home):
    def setUp(self):
        super().setUp()
        self.api = window.Api(7545)

    def test_state_is_the_apps_with_running_added(self):
        self.assertEqual(self.api.state(), {"running": False})
        self.alive()
        s = self.api.state()
        self.assertTrue(s["running"])
        self.assertEqual(s["setup"]["step"], "mining")
        self.assertNotIn("pid", s)

    def test_orders_go_to_the_running_app_only(self):
        self.assertEqual(self.api.order("pause")["ok"], False)  # not running
        self.alive()
        self.assertEqual(self.api.order("pause"), {"ok": True, "order": "pause"})
        self.assertEqual(app.take_order(), "pause")
        self.assertEqual(self.api.order("avatar-off")["ok"], True)
        self.assertEqual(app.take_order(), "avatar-off")
        self.assertFalse(self.api.order("format-disk")["ok"])
        self.assertIsNone(app.take_order())

    def test_the_wallet_and_its_words_come_from_setup(self):
        self.assertEqual(self.api.wallet(), {"words": []})
        words = Path(self.tmp.name) / "words.txt"
        words.write_text("Vibe Hash recovery words for 0xabc\nwallet file: w.json\n\n 1. apple\n 2. brick\n 3. cloud\n\nKeep them.\n")
        self.alive(wallet={"address": "0xabc", "path": "w.json", "words_file": str(words)})
        w = self.api.wallet()
        self.assertEqual((w["address"], w["path"], w["words"]), ("0xabc", "w.json", ["apple", "brick", "cloud"]))
        words.unlink()  # the user deleted the file, as told: no words, no error
        w = self.api.wallet()
        self.assertEqual((w["words"], w["words_file"]), ([], None))

    def test_settings_the_avatar_and_the_welcome(self):
        with mock.patch.object(app, "_winreg", return_value=None):
            c = self.api.settings()
            self.assertEqual((c["avatar"], c["welcomed"], c["startup"]), (True, False, False))
            self.assertEqual(c["home"], self.tmp.name)
            self.assertEqual(self.api.set_avatar(False), {"ok": True, "avatar": False})
            self.assertIs(settings.load()["avatar"], False)
            self.assertIsNone(app.take_order())  # no app running: the setting is enough
            self.alive()
            self.api.set_avatar(True)
            self.assertEqual(app.take_order(), "avatar-on")  # running: told now as well
            self.api.welcomed()
            self.assertTrue(self.api.settings()["welcomed"])
            self.api.welcome_again()
            self.assertFalse(self.api.settings()["welcomed"])

    def test_start_at_login_through_the_window(self):
        calls = []
        with mock.patch.object(Path, "home", return_value=Path(self.tmp.name)), \
                mock.patch.object(app, "_winreg", return_value=None), \
                mock.patch.object(app.platform, "system", return_value="Linux"), \
                mock.patch.object(app.subprocess, "run", side_effect=lambda cmd, **kw: calls.append(cmd)):
            on = self.api.set_startup(True)
            self.assertTrue(on["ok"])
            self.assertTrue(self.api.settings()["startup"])
            self.assertTrue(Path(on["where"]).exists())
            self.assertTrue(self.api.set_startup(False)["ok"])
            self.assertFalse(self.api.settings()["startup"])


if __name__ == "__main__":
    unittest.main()
