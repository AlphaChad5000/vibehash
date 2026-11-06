"""`vibehash worker --watch`: what the dashboard shows, as JSON on this computer only (watch.py)."""
import json
import logging
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

from vibehash import watch
from vibehash.client import NodeError, WorkerStats
from vibehash.tasks import SYN_PER_VIBE, TrainingJob


def fake_client(address="vibe1fake", device="CPU (NumPy, 4 threads)"):
    stats = WorkerStats(round=7, phase="compute", activity="training unit 2 of 3", busy=True, assigned=3, trained=1,
                        rounds=4, units=11, checks=6, flops_per_sec=2.5e9,
                        job=TrainingJob("fruit-demo", "vibe1req", budget=100 * SYN_PER_VIBE, fee_per_round=SYN_PER_VIBE))
    return SimpleNamespace(stats=stats, keys=SimpleNamespace(address=address, secret=b"never shown"),
                           backend=SimpleNamespace(describe=lambda: device),
                           hardware_class={"vendor": "cpu", "arch": "amd64", "memory_gb": 16, "label": "CPU 4 threads"})


class EventsTests(unittest.TestCase):
    def test_keeps_the_latest_numbered_and_clean(self):
        ev = watch.Events(keep=3)
        lg = logging.getLogger("vibehash.worker.test")
        lg.addHandler(ev)
        try:
            for i in range(5):
                lg.warning("round %d: \x1b[31mregistered\x1b[0m\nsecond line", i)
        finally:
            lg.removeHandler(ev)
        kept = ev.since(0)
        self.assertEqual([e["id"] for e in kept], [3, 4, 5])  # the last three, numbered from the first
        self.assertEqual(kept[-1]["text"], "round 4: registered")  # no escape codes, first line only
        self.assertEqual(kept[-1]["level"], "warning")
        self.assertEqual(ev.since(4), [kept[-1]])
        self.assertEqual(ev.last, 5)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.client = fake_client()
        self.api = mock.Mock()
        self.api.status.return_value = {"chain_id": "test-net"}
        self.api.account.return_value = {"balance": 5 * SYN_PER_VIBE, "stake": 10 * SYN_PER_VIBE, "reputation": 0.9,
                                         "unbonding": [], "nonce": 3, "secret": "not a field, but never copied"}
        with mock.patch.object(watch, "NodeClient", return_value=self.api):
            self.w = watch.Watch([("you", self.client)], "http://127.0.0.1:8545/", events=watch.Events().attach())
        self.port = self.w.serve(0)
        self.addCleanup(self.w.stop)

    def get(self, path, origin=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers={"Origin": origin} if origin else {})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()), resp.headers

    def test_the_snapshot_says_what_the_dashboard_shows(self):
        body, headers = self.get("/")
        [w] = body["workers"]
        self.assertEqual((w["name"], w["address"], w["device"]), ("you", "vibe1fake", "CPU (NumPy, 4 threads)"))
        self.assertEqual((w["round"], w["phase"], w["activity"], w["busy"]), (7, "compute", "training unit 2 of 3", True))
        self.assertEqual(w["this_round"], {"assigned": 3, "trained": 1, "committed": 0, "checked": 0})
        self.assertEqual((w["session"]["rounds"], w["session"]["units"], w["session"]["checks"]), (4, 11, 6))
        self.assertEqual(w["job"], {"id": "fruit-demo", "kind": "mlp"})
        self.assertEqual(w["hardware"]["label"], "CPU 4 threads")
        self.assertEqual(w["account"], {"balance": 5 * SYN_PER_VIBE, "stake": 10 * SYN_PER_VIBE, "reputation": 0.9, "unbonding": []})
        self.assertEqual((body["chain_id"], body["node"], body["syn_per_vibe"]), ("test-net", "http://127.0.0.1:8545", SYN_PER_VIBE))
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("never shown", json.dumps(body))  # nothing of the keys

    def test_the_account_is_asked_once_in_a_while_and_kept_when_the_node_is_away(self):
        self.get("/")
        self.get("/")
        self.assertEqual(self.api.account.call_count, 1)
        self.w._accounts.clear()
        self.api.account.side_effect = NodeError("away")
        body, _ = self.get("/")
        self.assertIsNone(body["workers"][0]["account"])  # nothing known yet, and the node can't say
        self.api.account.side_effect = None
        self.w._accounts.clear()
        self.get("/")
        self.w._accounts = {k: (0.0, v) for k, (_, v) in self.w._accounts.items()}  # old enough to ask again
        self.api.account.side_effect = NodeError("away")
        body, _ = self.get("/")
        self.assertEqual(body["workers"][0]["account"]["stake"], 10 * SYN_PER_VIBE)  # what it last said

    def test_events_after_a_number(self):
        lg = logging.getLogger("vibehash.worker")
        before = lg.level
        lg.setLevel(logging.INFO)  # as `vibehash worker` sets it (cli.main); the feed takes what the logger lets through
        self.addCleanup(lg.setLevel, before)
        lg.info("round 7: registered")
        lg.info("round 7: computed 3 units, committed 3")
        body, _ = self.get("/events?after=0")
        texts = [e["text"] for e in body["events"]]
        self.assertEqual(texts[-2:], ["round 7: registered", "round 7: computed 3 units, committed 3"])
        last = body["last"]
        body, _ = self.get(f"/events?after={last}")
        self.assertEqual((body["events"], body["last"]), ([], last))
        body, _ = self.get("/events?after=junk")
        self.assertEqual(body["last"], last)

    def test_only_pages_from_this_computer_get_cors(self):
        _, headers = self.get("/", origin="http://127.0.0.1:5173")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:5173")
        _, headers = self.get("/", origin="tauri://localhost")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "tauri://localhost")
        _, headers = self.get("/", origin="https://evil.example")
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_the_window_is_a_page_that_reads_the_feed_and_the_node(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/dashboard")
        with urllib.request.urlopen(req, timeout=5) as resp:
            page, headers = resp.read().decode("utf-8"), resp.headers
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertIn("connect-src 'self' http://127.0.0.1:8545", headers["Content-Security-Policy"])  # the node, and nothing else
        for needed in ('fetch("/"', '"/events?after="', '"/network"', 'id="feed"', 'id="census"', 'id="balance"'):
            self.assertIn(needed, page)
        self.assertNotIn("https://", page.replace("https://vibehash.org", ""))  # nothing fetched from the wider web
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_anything_else_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/wallet.json")
        self.assertEqual(ctx.exception.code, 404)

    def test_stop_frees_the_port_and_detaches_the_feed(self):
        ev = self.w.events
        self.w.stop()
        self.assertNotIn(ev, logging.getLogger("vibehash.worker").handlers)
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=2)
        self.w._server = None  # cleanup's stop() has nothing left to do


class CommandTests(unittest.TestCase):
    def test_worker_and_testnet_take_the_flag(self):
        from vibehash import cli
        a = cli.parser().parse_args(["worker", "--watch"])
        self.assertEqual(a.watch, watch.DEFAULT_PORT)
        a = cli.parser().parse_args(["testnet", "--watch", "7000"])
        self.assertEqual(a.watch, 7000)
        self.assertIsNone(cli.parser().parse_args(["worker"]).watch)

    def test_a_taken_port_is_a_warning_not_a_stop(self):
        import io
        from vibehash import cli
        # (Windows lets a second server bind a port in use, so the bind failure is played rather than provoked)
        with mock.patch.object(watch.Watch, "serve", side_effect=OSError("address already in use")), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertIsNone(cli._watch(SimpleNamespace(watch=7545), [], "http://127.0.0.1:8545", None))
        self.assertIn("can't serve --watch", err.getvalue())
        self.assertIsNone(cli._watch(SimpleNamespace(watch=None), [], "http://127.0.0.1:8545", None))


if __name__ == "__main__":
    unittest.main()
