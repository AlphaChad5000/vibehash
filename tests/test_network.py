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
        self.assertIn("All set", self.run_cli("setup", "--stake", "10", "--gpu", "skip"))  # re-run keeps the wallet
        wait_for(lambda: node.chain.state.stake(address) == 10 * VIBE)
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
        self.run_cli("job", "download", "--out", str(tmp / "fruit-model.npz"))
        result = self.run_cli("predict", str(fruit), "--rows", "3")  # no model named: the one just downloaded
        self.assertGreater(float(re.search(r"([\d.]+)% right", result)[1]), 80)  # four rounds on 256 rows
        self.run_cli("predict", str(tmp / "fruit-model.npz"), str(tmp / "fruit-new.csv"), "--out", str(tmp / "p.csv"))
        self.assertEqual((tmp / "p.csv").read_text().splitlines()[0].split(",")[-2:], ["predicted", "confidence"])
        audit = self.run_cli("chain", "verify", "--node", node.url)  # and anyone can replay how it was trained
        self.assertRegex(audit, r"replayed [1-9]\d* rounds from their records: every decision follows from the signed messages")

    def test_testnet_runs_and_resumes(self):
        # The small job: this tests resuming, not how much a busy CPU can train in 3 seconds.
        fast = ["--dir", str(self.home / "tn"), "--port", "0", "--device", "cpu", "--job", "small",
                "--register-s", "3", "--compute-s", "3", "--reveal-s", "2", "--verify-s", "2"]
        out = self.run_cli("testnet", "--rounds", "2", *fast)
        self.assertIn("posted job demo-1", out)
        earned = [float(x) for x in re.findall(r"worker-1 ([\d.]+)", out)]
        self.assertEqual(len(earned), 2)
        self.assertGreater(earned[-1], 0)
        out = self.run_cli("testnet", "--rounds", "1", *fast)  # same chain and job, picked up from disk
        self.assertNotIn("posted job", out)
        self.assertGreater(float(re.findall(r"worker-1 ([\d.]+)", out)[-1]), 0, out)  # the first round back gets worked


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
