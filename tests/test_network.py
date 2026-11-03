import contextlib
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from vibehash import cli, crypto
from vibehash import wallet as wallets
from vibehash.blobs import CHUNK, BlobError, BlobStore, file_sha256
from vibehash.client import NodeClient, NodeError, WorkerClient
from vibehash.coordinator import Coordinator, CoordinatorConfig
from vibehash.crypto import KeyPair
from vibehash.ledger import Block, Chain
from vibehash.model import Dataset, ModelSpec
from vibehash.node import Node, PhaseTimes
from vibehash.replay import replay_block
from vibehash.storage import Store
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.wire import WireError, arr_in, arr_out, open_envelope, seal

VIBE = SYN_PER_VIBE


class WireTests(unittest.TestCase):
    def test_array_round_trip_and_length_check(self):
        a = np.arange(5, dtype=np.float64) / 3
        np.testing.assert_array_equal(arr_in(arr_out(a), 5), a)
        with self.assertRaises(WireError):
            arr_in(arr_out(a), 6)

    def test_envelope(self):
        k = KeyPair.generate(b"w")
        env = seal(k, "commit", {"round": 3}, "net-a")
        self.assertEqual(open_envelope(env, "commit", "net-a"), (k.address, {"round": 3}))
        for bad in ({**env, "body": {"round": 4}}, {**env, "kind": "reveal"}):
            with self.assertRaises(WireError):
                open_envelope(bad, "commit", "net-a")
        with self.assertRaises(WireError):
            open_envelope(env, "commit", "net-b")  # replay on another network
        with self.assertRaises(WireError):
            open_envelope(seal(k, "commit", {}, "net-a", now=time.time() - 3600), "commit", "net-a")  # stale


class WalletTests(unittest.TestCase):
    def test_plaintext_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            k = KeyPair.generate()
            wallets.save(k, Path(d) / "w.json", None)
            self.assertEqual(wallets.load(Path(d) / "w.json", None), k)
            with self.assertRaises(wallets.WalletError):
                wallets.save(KeyPair.generate(), Path(d) / "w.json", None)

    @unittest.skipUnless(crypto.BACKEND == "openssl", "needs the cryptography package")
    def test_encrypted_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            k = KeyPair.generate()
            wallets.save(k, Path(d) / "w.json", "hunter2")
            self.assertNotIn(k.secret.hex(), (Path(d) / "w.json").read_text())
            self.assertEqual(wallets.load(Path(d) / "w.json", "hunter2"), k)
            with self.assertRaises(wallets.WalletError):
                wallets.load(Path(d) / "w.json", "wrong")


def write_npz(path: Path, spec: ModelSpec, seed: int, n_train: int = 2048) -> Path:
    data = Dataset.synthetic(spec, seed, n_train=n_train, n_val=512)
    np.savez(path, x_train=data.x_train, y_train=data.y_train, x_val=data.x_val, y_val=data.y_val)
    return path


class BlobStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.store = BlobStore(Path(self.tmp.name) / "b")

    def test_chunked_resumable_and_hash_checked(self):
        data = np.arange(3 * 2**20, dtype="<f4").tobytes()  # 12 MB
        bid = hashlib.sha256(data).hexdigest()
        step = 5 * 2**20
        got = self.store.write_chunk(bid, 0, data[:step], len(data))
        self.assertEqual(self.store.received(bid), step)
        self.assertEqual(self.store.write_chunk(bid, 0, data[:step], len(data)), step)  # a retried chunk is harmless
        with self.assertRaises(BlobError):
            self.store.write_chunk(bid, 2 * step + 1, data[:10], len(data))  # out of order
        for off in range(step, len(data), step):
            got = self.store.write_chunk(bid, off, data[off:off + step], len(data))
        self.assertEqual(got, len(data))
        self.assertTrue(self.store.has(bid))
        self.assertEqual(self.store.load(bid, "<f4", 3 * 2**20)[-1], 3 * 2**20 - 1)

    def test_rejects_bytes_that_dont_match_the_id(self):
        bid = hashlib.sha256(b"real").hexdigest()
        with self.assertRaises(BlobError):
            self.store.write_chunk(bid, 0, b"fake", 4)
        self.assertFalse(self.store.has(bid))

    def test_put_array_dedupes_and_prune(self):
        a = np.linspace(0, 1, 1000)
        bid = self.store.put_array(a, "<f8")
        self.assertEqual(self.store.put_array(a.copy(), "<f8"), bid)
        np.testing.assert_array_equal(self.store.load(bid, "<f8", 1000), a)
        with self.assertRaises(BlobError):
            self.store.load(bid, "<f8", 999)
        self.store.prune()
        self.assertFalse(self.store.has(bid))


def start_network(test, n_workers, behaviors=None, times=None, config=None, keep_rounds=10):
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows keeps memory-mapped files locked
    test.addCleanup(tmp.cleanup)
    node_keys, req = KeyPair.generate(b"node"), KeyPair.generate(b"requester")
    workers = [KeyPair.generate(f"worker{i}".encode()) for i in range(n_workers)]
    genesis = {"chain_id": "vibehash-localtest", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
               "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}}
    node = Node(Path(tmp.name) / "node", genesis, node_keys, port=0,
                times=times or PhaseTimes(register_s=1.0, compute_s=4.0, reveal_s=2.0, verify_s=3.0, idle_s=0.3),
                config=config or CoordinatorConfig(canaries_per_round=1, newcomer_units=6), keep_rounds=keep_rounds).start()
    state = {"stopped": False}

    def stop_node():
        if not state["stopped"]:
            state["stopped"] = True
            node.stop()
    test.addCleanup(stop_node)
    stop = threading.Event()
    behaviors = behaviors or ["honest"] * n_workers
    clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e9, poll_s=0.1, behavior=b,
                            cache_dir=Path(tmp.name) / f"cache{i}") for i, (k, b) in enumerate(zip(workers, behaviors))]
    test.addCleanup(stop.set)
    return node, NodeClient(node.url), req, workers, clients, stop, stop_node, genesis, Path(tmp.name)


def run_until(node, clients, stop, height, seconds=120):
    threads = [threading.Thread(target=c.run, args=(stop,), daemon=True) for c in clients]
    for t in threads:
        t.start()
    deadline = time.time() + seconds
    while node.chain.tip.height < height and time.time() < deadline:
        time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(timeout=30)


def wait_for(condition, seconds=30):
    deadline = time.time() + seconds
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.1)


class CommandLineTests(unittest.TestCase):
    """`vibehash setup` and `vibehash testnet`, the way a newcomer runs them, against real nodes."""

    def setUp(self):
        home = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(home.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": home.name, "VIBEHASH_PASSWORD": "test-only"})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET"):
            os.environ.pop(var, None)
        logging.disable(logging.INFO)  # cli.main turns on INFO logging for the whole process
        self.addCleanup(logging.disable, logging.NOTSET)
        self.home = Path(home.name)

    def run_cli(self, *argv) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                cli.main(list(argv))
            except SystemExit as e:
                if e.code not in (None, 0):
                    self.fail(f"vibehash {' '.join(argv)} exited with {e.code!r}\n{out.getvalue()}")
        return out.getvalue()

    def run_cli_status(self, *argv) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                cli.main(list(argv))
                code = 0
            except SystemExit as e:
                code = e.code or 0
        return code, out.getvalue()

    def test_censored_commitment_is_proven(self):
        """A node that drops a worker's commitment after accepting it: replays can't tell, the worker's receipts can."""
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 3)
        job = TrainingJob("censor-job", req.address, budget=60 * VIBE, fee_per_round=10 * VIBE, units_per_round=6)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        victim = workers[0].address
        honest = Coordinator.start_reveal

        def censor(c):  # after the node has signed receipts for the victim's commitments, drop them
            r = c.r
            for key in [k for k in r.commitments if k[1] == victim]:
                del r.commitments[key]
            r.record["commitments"] = [d for d in r.record["commitments"] if d["worker"] != victim]
            return honest(c)
        with mock.patch.object(Coordinator, "start_reveal", censor):
            run_until(node, clients, stop, node.chain.tip.height + 4)
        self.assertTrue(clients[0].censored, "the victim's worker didn't notice")
        self.assertIn("commit", clients[0].censored[0])
        self.assertFalse(clients[1].censored)

        code, out = self.run_cli_status("chain", "verify", "--node", node.url, "--receipts", str(tmp / "cache0" / "receipts.jsonl"))
        self.assertEqual(code, 1, out)
        self.assertIn("every decision follows from the signed messages", out)  # the record itself is consistent...
        self.assertRegex(out, r"[1-9]\d* censored")  # ...but the receipts show what's missing from it
        code, out = self.run_cli_status("chain", "verify", "--node", node.url, "--receipts", str(tmp / "cache1" / "receipts.jsonl"))
        self.assertEqual(code, 0, out)
        self.assertNotIn("censored", out)

    def test_a_round_secret_swapped_after_the_commitments_is_noticed(self):
        """A leader that picks its round secret once it has seen the workers' commitments can choose the beacon. If it
        rewrites the commitment in the record to match, a replay can't tell; the workers saw the original in /status."""
        from vibehash.coordinator import secret_commitment

        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 2)
        job = TrainingJob("grind-job", req.address, budget=60 * VIBE, fee_per_round=10 * VIBE, units_per_round=4)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        honest = Coordinator.start_reveal

        def grind(c):
            c.r.secret = os.urandom(32)
            c.r.record["beacon_commit"] = secret_commitment(c.r.secret).hex()
            return honest(c)
        with mock.patch.object(Coordinator, "start_reveal", grind):
            run_until(node, clients, stop, node.chain.tip.height + 3)
        for client in clients:
            self.assertTrue(any("round secret" in d for d in client.censored), client.censored)

    def test_checkpoints_kept_for_the_challenge_window(self):
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 3, keep_rounds=2)
        job = TrainingJob("window-job", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=6)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        run_until(node, clients, stop, node.chain.tip.height + 5)
        node.stopping.set()  # stop sealing rounds (the API stays up), or the window moves on while we look
        node.threads[1].join(timeout=30)
        tip = node.chain.tip.height
        recent, everything = node.store.evidence(tip, 2), node.store.evidence(tip, 100)
        self.assertTrue(recent and all(node.blobs.has(b) for b in recent), "the window's checkpoints were deleted")
        older = everything - recent
        self.assertTrue(older and not any(node.blobs.has(b) for b in older), "checkpoints outlived the window")
        self.assertEqual(api.status()["keeps_checkpoints_for"], 2)
        out = self.run_cli("chain", "verify", "--node", node.url)
        self.assertRegex(out, r"re-ran [1-9]\d* weight-based checks in the last 2 rounds from their checkpoints: every one matches its record")
        self.assertIn("Earlier rounds' checkpoints are pruned", out)

    def test_setup_then_commands_need_no_flags(self):
        node, api, req, *_ = start_network(self, 0)
        out = self.run_cli("setup", "--node", node.url, "--gpu", "skip")
        wallet = self.home / "wallet.json"
        address = json.loads(wallet.read_text())["address"]
        self.assertTrue(wallets.is_encrypted(wallet))
        self.assertIn("Ask the node's operator", out)  # nothing to stake with yet
        self.assertIn("Almost there", out)
        api.send_tx(req, "transfer", address, 20 * VIBE)
        wait_for(lambda: node.chain.state.balance(address) == 20 * VIBE)
        started = []  # say yes to "Start mining now?", and catch the worker it starts
        with mock.patch.object(cli, "ask", lambda question, assume_yes=False: True), \
                mock.patch.object(cli, "cmd_worker", lambda a, keys=None: started.append((a, keys))):
            self.run_cli("setup", "--stake", "10", "--gpu", "skip")  # re-run keeps the wallet
        [(a, keys)] = started
        self.assertEqual((a.node, a.wallet, keys.address), (node.url, str(wallet), address))  # unlocked just once
        wait_for(lambda: node.chain.state.stake(address) == 10 * VIBE)
        with mock.patch("sys.stdin", io.StringIO()):  # no terminal: no question
            self.assertIn("All set. Start earning with", self.run_cli("setup", "--gpu", "skip"))
        out = self.run_cli("balance")
        self.assertIn(address, out)
        self.assertIn("stake:      10 VIBE", out)

    def test_train_a_model_from_a_csv(self):
        """The job poster's whole path: a CSV in, rounds of training, a model file out, predictions."""
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 3)
        wallets.save(req, tmp / "requester.json", None)
        self.assertIn("wrote", self.run_cli("example", "--dir", str(tmp)))
        fruit = tmp / "fruit.csv"
        out = self.run_cli("train", str(fruit), "--budget", "40", "--node", node.url, "--wallet", str(tmp / "requester.json"))
        job_id = re.search(r"Posted job (\S+)", out)[1]
        self.assertIn("5 columns: weight_g, length_cm, width_cm, sweetness, colour_hue", out)
        self.assertIn("predicts  fruit: apple, banana, lemon, orange", out)
        run_until(node, clients, stop, node.chain.tip.height + 4)
        status = self.run_cli("job", "status")  # no job id or node: the last job, and the node it went to
        self.assertIn(f"job {job_id}", status)
        self.assertRegex(status, r"rounds     [1-9]\d* done")
        self.check_job_history(node, job_id, tmp)
        self.run_cli("job", "download", "--out", str(tmp / "fruit-model.npz"))
        page = (tmp / "fruit-model-TRAINING.md").read_text(encoding="utf-8")  # how it was trained, beside it
        self.assertIn("fruit.csv: 256 rows to train on, 64 held back", page)
        self.assertIn(f"on {node.url}, chain `vibehash-localtest`", page)
        self.assertRegex(page, r"[1-9]\d* of them trained it, with [1-9]\d* units of work accepted")
        self.assertIn("posted with a budget of 40 VIBE", page)
        self.assertTrue((tmp / "fruit-model-training-history.csv").read_text().startswith("round,time,units_accepted"))
        result = self.run_cli("predict", str(fruit), "--rows", "3")  # no model named: the one just downloaded
        self.assertGreater(float(re.search(r"([\d.]+)% right", result)[1]), 80)  # four rounds on 256 rows
        self.run_cli("predict", str(tmp / "fruit-model.npz"), str(tmp / "fruit-new.csv"), "--out", str(tmp / "p.csv"))
        self.assertEqual((tmp / "p.csv").read_text().splitlines()[0].split(",")[-2:], ["predicted", "confidence"])
        audit = self.run_cli("chain", "verify", "--node", node.url)  # and anyone can replay how it was trained
        self.assertRegex(audit, r"replayed [1-9]\d* rounds from their records: every decision follows from the signed messages")

    def check_job_history(self, node, job_id, tmp):
        """`job history`: the rounds from the node, printed and saved in ~/.vibehash/jobs/<job>, for a job posted here
        or anywhere else."""
        out = self.run_cli("job", "history")  # the last job, on the node it went to
        self.assertRegex(out, rf"job {job_id} on {re.escape(node.url)}: [1-9]\d* rounds scored, [1-9]\d* of them trained it")
        self.assertRegex(out, r"\n +\d+  20\d\d-\d\d-\d\d \d\d:\d\d:\d\d +[1-9]\d* +[\d.]+ +[\d.]+% +[\d.]+ +[\d.]+\n")
        saved = self.home / "jobs" / job_id / "history.csv"
        self.assertIn(f"wrote {saved}", out)
        rows = saved.read_text().splitlines()
        self.assertEqual(rows[0], "round,time,units_accepted,loss,accuracy,paid_vibe,budget_left_vibe")
        trained = [r.split(",") for r in rows[1:] if r.split(",")[2] != "0"]
        self.assertTrue(trained and all(r[1] and float(r[5]) > 0 for r in trained), rows)  # each with its time and fee
        self.assertTrue((self.home / "jobs" / job_id / "history.jsonl").exists())
        dump = json.loads(self.run_cli("job", "history", job_id, "--json", "--csv", str(tmp / "h.csv")))
        self.assertEqual((dump["job"], dump["chain_id"]), (job_id, "vibehash-localtest"))
        first = dump["rounds"][0]
        self.assertAlmostEqual(first["budget_left_vibe"] + first["paid_vibe"], 40.0)  # before it, all of the budget
        self.assertEqual((tmp / "h.csv").read_text().splitlines()[0], rows[0])
        other = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # another computer: no receipt, no saved node
        self.addCleanup(other.cleanup)
        with mock.patch.dict(os.environ, {"VIBEHASH_HOME": other.name}):
            out = self.run_cli("job", "history", job_id, "--node", node.url)
            self.assertTrue((Path(other.name) / "jobs" / job_id / "history.csv").exists(), out)
            code, out = self.run_cli_status("job", "history", "no-such-job", "--node", node.url)
        self.assertEqual(code, f"error: {node.url} has no job called no-such-job")

    def test_a_worker_keeps_records_and_sums_up(self):
        """`vibehash worker --plain` says where its logs go, writes a line per round, and sums the session up when it's
        stopped (Ctrl+C, here raised once two rounds are sealed)."""
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 1)
        job = TrainingJob("records-job", req.address, budget=60 * VIBE, fee_per_round=10 * VIBE, units_per_round=4)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        wallets.save(workers[0], tmp / "w.json", None)

        def run(client, stop=None):
            deadline = time.time() + 90
            while time.time() < deadline and client.record.rounds < 2:
                try:
                    client.step()
                except NodeError:
                    pass
                time.sleep(0.1)
            raise KeyboardInterrupt
        with mock.patch.object(WorkerClient, "run", run):
            out = self.run_cli("worker", "--plain", "--node", node.url, "--wallet", str(tmp / "w.json"), "--device", "cpu",
                               "--flops", "1e9", "--memory-mb", "8192")
        logs = self.home / "logs"
        self.assertIn(f"recording to {logs}: worker-", out)
        self.assertRegex(out, r"Recorded 2 sealed rounds: [1-9]\d* units accepted, \+[1-9][\d.]* VIBE earned, as the "
                              r"blocks paid it")
        lines = [json.loads(x) for x in (logs / "rounds.jsonl").read_text().splitlines()]
        sealed = [x for x in lines if x["sealed"]]
        self.assertEqual(len(sealed), 2, lines)
        self.assertTrue(all(x["worker"] == workers[0].address and x["node"] == node.url and x["job"] == "records-job"
                            and x["chain_id"] == "vibehash-localtest" and x["committed"] for x in sealed), sealed)
        s = node.chain.state  # what the lines say it earned is what the chain paid it, beyond its genesis stake
        self.assertEqual(round(sum(x["earned_vibe"] for x in sealed) * VIBE),
                         s.balance(workers[0].address) + s.stake(workers[0].address) - 10 * VIBE)

    def test_testnet_runs_and_resumes(self):
        # The small job: this tests resuming, not how much a busy CPU can train in 3 seconds.
        fast = ["--dir", str(self.home / "tn"), "--port", "0", "--device", "cpu", "--job", "small",
                "--register-s", "3", "--compute-s", "3", "--reveal-s", "2", "--verify-s", "2"]
        # With the node's INFO lines on (setUp turns them off), so its log file has something to keep; they go to a
        # scratch stream rather than the test's output.
        logging.disable(logging.NOTSET)
        with mock.patch.object(logging.root, "handlers", []), contextlib.redirect_stderr(io.StringIO()):
            out = self.run_cli("testnet", "--rounds", "2", *fast)
        logging.disable(logging.INFO)
        self.assertIn("posted job demo-1", out)
        earned = [float(x) for x in re.findall(r"worker-1 ([\d.]+)", out)]
        self.assertEqual(len(earned), 2)
        self.assertGreater(earned[-1], 0)
        # Its records: the node's log in the testnet's folder, and each worker's rounds, a JSON line each.
        self.assertIn("node up at", (self.home / "tn" / "node.log").read_text(encoding="utf-8"))
        self.assertRegex(out, r"Recorded [1-9]\d* sealed rounds? of 2 workers: \d+ units? accepted, \+[\d.]+ VIBE earned")
        lines = [json.loads(x) for x in (self.home / "logs" / "rounds.jsonl").read_text().splitlines()]
        sealed = [x for x in lines if x["sealed"]]
        self.assertTrue(sealed, lines)
        workers = {json.loads((self.home / "tn" / f"worker-{i}.json").read_text())["address"] for i in (1, 2)}
        self.assertEqual({x["worker"] for x in lines}, workers)
        self.assertTrue(all(x["job"] == "demo-1" and x["kind"] == "mlp" and x["chain_id"].startswith("vibehash-local-")
                            for x in lines), lines)
        self.assertTrue(any(x["committed"] and x["accepted"] and x["earned_vibe"] > 0 and x["paid"].get("work", 0) > 0
                            and x["train_s"] > 0 for x in sealed), sealed)
        log = next((self.home / "logs").glob("worker-*.log")).read_text(encoding="utf-8")
        self.assertIn("round", log)  # the feed's events: the node's rounds, in a testnet
        out = self.run_cli("testnet", "--rounds", "1", *fast)  # same chain and job, picked up from disk
        self.assertNotIn("posted job", out)
        self.assertGreater(float(re.findall(r"worker-1 ([\d.]+)", out)[-1]), 0, out)  # the first round back gets worked


class DistantWorkerTest(unittest.TestCase):
    def test_many_units_are_committed_and_revealed_in_time_over_a_slow_link(self):
        """A worker 50 ms from the node with 12 units of 9 checkpoints each, as the public network showed on 2026-11-01:
        - revealed one at a time, each asking again about checkpoints it uploaded during compute, they took longer than
          the reveal phase, and missing it forfeits the round. Reveals now go out together, as one message each;
        - the link drops two reveals, one on the way and one's reply: each is sent again, and not lost;
        - uploaded one after another, a unit's checkpoints took so long the last unit missed the compute deadline.
          They now go up together."""
        node, api, req, workers, clients, stop, *_ = start_network(
            self, 1, times=PhaseTimes(register_s=1.0, compute_s=6.0, reveal_s=1.5, verify_s=2.0, idle_s=0.3),
            config=CoordinatorConfig(canaries_per_round=0, newcomer_units=12))
        far = clients[0].node
        drops, lock = ["on the way", "reply"], threading.Lock()

        def far_open(request, near=far._open):
            time.sleep(0.05)
            with lock:
                drop = drops.pop(0) if request.full_url.endswith("/reveal") and drops else None
            if drop == "on the way":
                raise ConnectionResetError("dropped on the way")
            reply = near(request)
            if drop == "reply":
                reply.close()
                raise ConnectionResetError("reply dropped")
            return reply
        far._open = far_open
        job = TrainingJob("far-job", req.address, budget=60 * SYN_PER_VIBE, fee_per_round=10 * SYN_PER_VIBE,
                          units_per_round=12, steps_per_unit=40, checkpoint_every=5)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        start = node.chain.tip.height
        run_until(node, clients, stop, start + 3)
        worked = [b.summary for b in node.chain.blocks[start + 1:] if b.summary and (b.summary.accepted_units or b.summary.missed)]
        self.assertTrue(worked, "no round was worked")
        self.assertEqual([s.missed for s in worked], [[]] * len(worked))
        self.assertIn(12, [s.accepted_units for s in worked])
        self.assertEqual(drops, [])


    def test_a_far_checker_gets_its_verdicts_in_before_the_verify_phase_ends(self):
        """A RunPod pod checking fruit-demo on test night (2026-11-02): "the verify phase ended with 7 tasks left
        unchecked", every round. It fetched each task's checkpoints one at a time and then sent its verdict, a round
        trip after another, and 9 tasks didn't fit the 10 s phase. They now come down together, and verdicts go while
        the next task is checked."""
        node, api, req, workers, clients, stop, *_ = start_network(
            self, 3, times=PhaseTimes(register_s=1.0, compute_s=5.0, reveal_s=1.5, verify_s=4.0, idle_s=0.3),
            config=CoordinatorConfig(canaries_per_round=0, newcomer_units=12))
        far = clients[0]

        def far_open(request, near=far.node._open):
            time.sleep(0.1)
            return near(request)
        far.node._open = far_open
        given, answer = [], far._answer
        far._answer = lambda rnd, job, tasks: (given.append(len(tasks)), answer(rnd, job, tasks))[1]
        late = []

        class Late(logging.Handler):
            def emit(self, record):
                if "left unchecked" in record.getMessage():
                    late.append(record.getMessage())
        handler = Late()
        logging.getLogger("vibehash.worker").addHandler(handler)
        self.addCleanup(logging.getLogger("vibehash.worker").removeHandler, handler)
        job = TrainingJob("far-checks", req.address, budget=60 * SYN_PER_VIBE, fee_per_round=10 * SYN_PER_VIBE,
                          units_per_round=12, steps_per_unit=40, checkpoint_every=5)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        run_until(node, clients, stop, node.chain.tip.height + 3)
        self.assertGreaterEqual(sum(given), 4, "the far worker was given too few checks to tell")
        self.assertEqual(late, [])


class RerunRoundTest(unittest.TestCase):
    def test_a_round_run_again_gets_its_checkpoints_again(self):
        """What a worker remembers the node has is for one attempt at a round: another leader, or the node after a
        restart, runs the round again without them, and a reveal naming them would leave the units withheld."""
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        c = WorkerClient("http://127.0.0.1:9", KeyPair.generate(b"rerun"), device="cpu", cache_dir=tmp.name)
        status = {"genesis_hash": "g", "height": 4, "round": 5, "attempt": "a", "phase": "reveal", "phase_ends": 0}
        c._follow_leader = lambda: dict(status)
        c.done[(5, "a", "reveal")] = c.done[(5, "b", "reveal")] = True  # nothing else to do in either
        c.step()
        c.uploaded[5] = {"blob-sent-in-attempt-a"}
        c.step()
        self.assertEqual(c.uploaded[5], {"blob-sent-in-attempt-a"})
        status["attempt"] = "b"
        c.step()
        self.assertNotIn(5, c.uploaded)


class LiveNetworkTest(unittest.TestCase):
    """A real node on localhost, real HTTP, worker clients (one cheating), uploaded data."""

    def test_end_to_end(self):
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(
            self, 4, ["honest", "honest", "honest", "lazy"])

        # Requester uploads a dataset and posts a job that trains on it.
        job = TrainingJob("e2e-job", req.address, budget=200 * VIBE, fee_per_round=10 * VIBE, units_per_round=8)
        sha = api.upload_dataset(req, write_npz(tmp / "data.npz", job.spec, 5))
        job.data_source = {"kind": "npz", "sha256": sha}
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())

        run_until(node, clients, stop, 4)  # job block + 3 rounds
        self.assertGreaterEqual(node.chain.tip.height, 4, "network didn't produce blocks in time")
        honest = clients[0].stats  # what the mining dashboard shows
        self.assertGreater(honest.units, 0)
        self.assertGreater(honest.checks + sum(c.stats.checks for c in clients[1:]), 0)
        self.assertGreaterEqual(honest.rounds, 2)
        self.assertEqual(honest.training_flops, honest.units * job.unit_flops)

        st = node.chain.state
        for k in workers[:3]:
            self.assertGreater(st.balance(k.address), 0, "honest worker wasn't paid")
            self.assertEqual(st.stake(k.address), 10 * VIBE)
        self.assertLess(st.stake(workers[3].address), 10 * VIBE, "cheater wasn't slashed")
        self.assertEqual(st.balance(workers[3].address), 0)
        self.assertLess(st.escrow["e2e-job"], job.budget)

        # The website's live panel: who mined lately and how the job is doing, readable from any web page.
        with urllib.request.urlopen(node.url + "/network", timeout=10) as resp:
            self.assertEqual(resp.headers["Access-Control-Allow-Origin"], "*")
            net = json.loads(resp.read())
        self.assertGreaterEqual(net["miners"]["online"], 3)
        self.assertEqual(net["height"], node.chain.tip.height)
        j = next(j for j in net["jobs"] if j["id"] == "e2e-job")
        self.assertGreaterEqual(j["rounds_trained"], 1)
        self.assertEqual(j["metric"], "accuracy")
        self.assertTrue(j["history"])
        post = urllib.request.Request(node.url + "/tx", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as bad:  # a malformed tx; what matters is its headers
            urllib.request.urlopen(post, timeout=10)
        with bad.exception as e:
            self.assertIsNone(e.headers["Access-Control-Allow-Origin"])  # only GETs are open to other sites

        # An outside auditor downloads every block, re-validates from genesis, and replays every round from its
        # record: the slashed cheater, the paid workers and everything else must follow from the signed messages.
        audit, replayed = Chain.from_genesis(api.get("/genesis")), 0
        for d in api.get(f"/blocks?from=1&to={node.chain.tip.height}"):
            block = Block.from_dict(d)
            if block.summary is not None:
                self.assertEqual(replay_block(audit, block, api.get(f"/round/{block.height}/record")), [], f"block {block.height}")
                replayed += 1
            audit.append(block)
        self.assertTrue(audit.validate())
        self.assertEqual(audit.tip.hash, node.chain.tip.hash)
        self.assertGreaterEqual(replayed, 3)

        # Restart from disk: chain is re-validated and model weights survive.
        height, params = node.chain.tip.height, node.coord.params["e2e-job"].copy()
        stop_node()
        store = Store(tmp / "node")
        self.addCleanup(store.close)
        reloaded = store.open_chain(genesis)
        self.assertGreaterEqual(reloaded.tip.height, height)
        np.testing.assert_array_equal(store.load_params("e2e-job"), params)
        with self.assertRaises(ValueError):
            store.open_chain({**genesis, "chain_id": "another-net"})

    def test_large_dataset_moves_in_chunks(self):
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 1)
        path = write_npz(tmp / "big.npz", ModelSpec(), 3, n_train=320_000)  # ~20 MB, well past one chunk
        self.assertGreater(path.stat().st_size, 2 * CHUNK)
        sha = api.upload_dataset(req, path)
        self.assertEqual(sha, file_sha256(path))
        local = BlobStore(tmp / "dl")
        got = api.download("dataset", sha, local)
        self.assertEqual(file_sha256(got), sha)
        stranger = KeyPair.generate(b"no-balance")
        with self.assertRaises(NodeError):  # no VIBE, no uploads: stops strangers filling the disk
            api.upload_dataset(stranger, write_npz(tmp / "other.npz", ModelSpec(), 4))

    def test_checkpoints_larger_than_8_mb(self):
        """A 2.2M-weight float32 model: every checkpoint is ~8.8 MB, past the old JSON limit."""
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(
            self, 3, times=PhaseTimes(register_s=1.0, compute_s=4.0, reveal_s=3.0, verify_s=4.0, idle_s=0.3),
            config=CoordinatorConfig(canaries_per_round=0, replication_rate=0, newcomer_units=2))
        spec = ModelSpec(8, (1500, 1460), 4, "float32")
        job = TrainingJob("big-job", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, spec=spec, lr=0.01,
                          batch_size=8, steps_per_unit=2, checkpoint_every=1, units_per_round=3, tolerance=1e-4)
        self.assertGreater(spec.n_params * 4, 8 * 2**20)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        run_until(node, clients, stop, 3)
        self.assertGreaterEqual(node.chain.tip.height, 3)
        summary = node.chain.blocks[2].summary
        self.assertEqual(summary.accepted_units, 3, summary)
        self.assertEqual(summary.slashes, [])
        self.assertTrue(node.chain.validate())


if __name__ == "__main__":
    unittest.main()
