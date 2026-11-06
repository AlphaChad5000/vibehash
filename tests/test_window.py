"""The app's window (window.py): the dashboard in a web view of its own, with the browser as the fallback."""
import http.client
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from vibehash import window


class Hook(list):
    """pywebview's `window.events.closed += fn`."""

    def __iadd__(self, fn):
        self.append(fn)
        return self


class FakeWindow:
    def __init__(self):
        self.loaded = []
        self.events = SimpleNamespace(closed=Hook())

    def load_url(self, url):
        self.loaded.append(url)


def fake_webview(start_raises=None):
    """A stand-in for the pywebview package: records the window asked for, and runs start's function at once."""
    m = mock.MagicMock()
    m.windows = []

    def create_window(title, html=None, **kw):
        w = FakeWindow()
        m.windows.append((title, html, kw, w))
        return w

    def start(fn=None, args=None):
        m.started = args
        if start_raises:
            raise start_raises
        if fn:
            fn(*args)

    m.create_window, m.start = create_window, start
    return m


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"VIBEHASH_HOME": self.tmp.name, "DISPLAY": ":0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_the_command_is_this_python_running_the_window_for_a_port(self):
        self.assertEqual(window.command(7000), [sys.executable, "-m", "vibehash", "window", "--watch", "7000"])
        self.assertEqual(window.dashboard_url(7000), "http://127.0.0.1:7000/dashboard")

    def test_the_window_shows_starting_then_the_dashboard_once_the_feed_answers(self):
        wv = fake_webview()
        answers = iter([False, False, True])
        said = []
        with mock.patch.dict(sys.modules, {"webview": wv}), \
                mock.patch.object(window, "feed_ready", side_effect=lambda url, timeout=1.0: next(answers)), \
                mock.patch.object(window.time, "sleep") as sleep:
            how = window.show(7000, fallback=lambda url: said.append(url), say=said.append)
        self.assertEqual(sleep.call_count, 2)  # the two misses, without waiting for real
        self.assertEqual(how, "window")
        self.assertEqual(said, [])
        title, html, kw, w = wv.windows[0]
        self.assertEqual(title, window.TITLE)
        self.assertIn("starting", html)
        self.assertEqual((kw["width"], kw["height"], kw["min_size"]), (window.SIZE[0], window.SIZE[1], window.MIN_SIZE))
        self.assertEqual(w.loaded, ["http://127.0.0.1:7000/dashboard"])  # after the two misses
        self.assertEqual(len(w.events.closed), 1)  # closing it ends the wait: the hook sets the event the wait polls
        closed = wv.started[2]
        self.assertFalse(closed.is_set())
        w.events.closed[0]()
        self.assertTrue(closed.is_set())
        self.assertTrue(window.Lock(7000).acquire())  # the lock was let go when the window closed

    def test_closing_the_window_while_waiting_stops_the_wait(self):
        w = FakeWindow()
        closed = threading.Event()
        looks = []

        def ready(url, timeout=1.0):
            looks.append(url)
            if len(looks) == 3:
                closed.set()  # the user closed it
            return False

        self.assertFalse(window.wait_then_load(w, "http://127.0.0.1:7545/dashboard", closed, ready=ready, sleep=lambda s: None))
        self.assertEqual(len(looks), 3)
        self.assertEqual(w.loaded, [])

    def test_without_pywebview_the_browser_gets_the_dashboard(self):
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

    def test_feed_ready_asks_the_feeds_events_which_need_no_node(self):
        with mock.patch.object(window._LOCAL, "open") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            self.assertTrue(window.feed_ready("http://127.0.0.1:7545/dashboard"))
            self.assertEqual(urlopen.call_args.args[0], "http://127.0.0.1:7545/events?after=0")
            urlopen.side_effect = OSError("refused")
            self.assertFalse(window.feed_ready("http://127.0.0.1:7545/dashboard"))
            urlopen.side_effect = http.client.IncompleteRead(b"")  # a server just starting: not yet, not a crash
            self.assertFalse(window.feed_ready("http://127.0.0.1:7545/dashboard"))

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


if __name__ == "__main__":
    unittest.main()
