"""Denial of service and robustness in the network layer: what an anonymous client, a lying peer or a relay can make a
node or worker do. Each test reproduces one way in, mostly against a real node on a free localhost port."""
import contextlib
import copy
import hashlib
import json
import logging
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import numpy as np

from vibehash import cli, consensus, ledger, node as node_module
from vibehash.blobs import MAX_DATASET, BlobStore
from vibehash.client import NodeClient, NodeError, WorkerClient
from vibehash.consensus import Engine, Timeouts, proposal, proposer_order, vote
from vibehash.crypto import KeyPair, address_of
from vibehash.ledger import Block, Chain, InvalidBlock, Tx, apply_block, canonical
from vibehash.model import Dataset, ModelSpec
from vibehash.node import MAX_JSON, Node, PhaseTimes
from vibehash.ratelimit import Limits, bucket_of
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.wire import WireError, seal
from tests.test_network import run_until, start_network, write_npz

VIBE = SYN_PER_VIBE
REQ, NODE = KeyPair.generate(b"dos-requester"), KeyPair.generate(b"dos-node")
SOLO = {"chain_id": "dos-net", "timestamp": 0, "allocations": {REQ.address: 500 * VIBE}, "stakes": {NODE.address: 10 * VIBE}}
FAST = PhaseTimes(register_s=0.5, compute_s=1.0, reveal_s=0.5, verify_s=0.5, idle_s=0.2)


def start_node(test, genesis=SOLO, keys=NODE, **kw) -> Node:
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(tmp.cleanup)
    node = Node(Path(tmp.name) / "node", genesis, keys, port=kw.pop("port", 0), times=kw.pop("times", FAST), **kw).start()
    test.addCleanup(node.stop)
    return node


def wait_for(condition, seconds=30):
    deadline = time.time() + seconds
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.05)


def raw(url: str, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> tuple[int, bytes]:
    """One request, no retries: (status, body)."""
    u = urlparse(url)
    import http.client
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=20)
    try:
        c.request(method, path, body, headers or {})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def posted_job(test, api: NodeClient, node: Node, job: TrainingJob) -> None:
    """A job on chain. Nodes don't relay a job whose budget can't pay for a round (ledger._relayable), but that's
    relay policy, not a block rule: one can still arrive in a block another validator made. So an unfunded job, which
    no round ever trains, gets past this node's relay check here, as it would reach the chain that way."""
    with mock.patch.object(ledger, "_relayable", lambda tx: None) if not job.budget else contextlib.nullcontext():
        api.send_tx(REQ, "post_job", "system", job.budget, job.spec_json())
    wait_for(lambda: job.job_id in node.chain.state.jobs)


class Fake(BaseHTTPRequestHandler):
    """A node that says whatever `routes` (path -> JSON-able, or bytes) tells it to, and counts what it's asked."""
    routes: dict = {}
    hits: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        self.hits[path] = self.hits.get(path, 0) + 1
        out = self.routes.get(path)
        if out is None:
            self.send_response(404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        body = out if isinstance(out, bytes) else json.dumps(out(path) if callable(out) else out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET


def fake_node(test, routes: dict) -> tuple[str, dict]:
    handler = type("F", (Fake,), {"routes": routes, "hits": {}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    test.addCleanup(srv.server_close)
    test.addCleanup(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}", handler.hits


# --- 1. consensus messages for later heights ------------------------------------------------------------------------

class FutureMessageTests(unittest.TestCase):
    def setUp(self):
        self.keys = [KeyPair.generate(f"fv{i}".encode()) for i in range(4)]
        self.behind = []
        self.e = Engine("future-net", self.keys[0], Timeouts(), validators=lambda: {k.address: 10 for k in self.keys},
                        validate=lambda v: True, commit=lambda *a: None, broadcast=lambda m: None,
                        behind=self.behind.append, block_id=lambda v: v["block"]["id"])
        self.e.start(5, now=0)

    def test_unsigned_or_outsider_messages_for_later_heights_are_dropped(self):
        forged = vote(self.keys[1], "future-net", "prevote", 9, 0, None)
        forged["sig"] = "00" * 64
        outsider = vote(KeyPair.generate(b"outsider"), "future-net", "prevote", 9, 0, None)
        unsigned = {"type": "prevote", "chain_id": "future-net", "height": 999999999, "round": 0, "block": None,
                    "pubkey": "00", "sig": "00", "pad": "x" * 10_000}
        for m in (forged, outsider, unsigned):
            self.e.receive(m, now=1)
        self.assertEqual(self.e.future, {})
        self.assertEqual(self.behind, [], "nor do they send us looking for blocks")
        self.e.receive(vote(self.keys[1], "future-net", "prevote", 7, 0, None), now=1)
        self.assertEqual(len(self.e.future), 1)
        self.assertEqual(self.behind, [7])

    def test_kept_messages_are_bounded_by_height_sender_and_size(self):
        far = vote(self.keys[1], "future-net", "prevote", 5 + consensus.FUTURE_HEIGHTS + 1, 0, None)
        self.e.receive(far, now=1)
        self.assertEqual((self.e.future, self.behind), ({}, [far["height"]]), "too far ahead to keep, but worth a sync")
        for r in range(consensus.FUTURE_PER_SENDER + 10):
            self.e.receive(vote(self.keys[1], "future-net", "prevote", 6, r, None), now=1)
            self.e.receive(vote(self.keys[1], "future-net", "prevote", 6, r, None), now=1)  # re-sent: kept once
        self.assertEqual(len(self.e.future), consensus.FUTURE_PER_SENDER)
        self.e.receive(vote(self.keys[2], "future-net", "prevote", 6, 0, None), now=1)
        self.assertEqual(len(self.e.future), consensus.FUTURE_PER_SENDER + 1, "one validator's flood doesn't crowd out another's")
        with mock.patch.object(consensus, "FUTURE_BYTES", 2000):
            for r in range(10):
                self.e.receive(vote(self.keys[3], "future-net", "precommit", 6, r, None), now=1)
        self.assertLessEqual(sum(size for k, (_, size) in self.e.future.items() if k[0] == self.keys[3].address), 2000)
        self.e.start(6, now=2)  # we caught up: what we kept counts now
        self.assertEqual(len(self.e.votes["prevote"].get(0, {})), 2)

    def test_node_drops_an_anonymous_flood(self):
        genesis = {**SOLO, "chain_id": "future-node", "validators": {NODE.address: "http://127.0.0.1:1"}}
        node = start_node(self, genesis)
        body = {"type": "prevote", "chain_id": "future-node", "height": 10**9, "round": 0, "block": None, "pubkey": "00", "sig": "00"}
        for _ in range(5):
            NodeClient(node.url).post("/consensus", body)
        self.assertEqual(node.engine.future, {})


# --- 2. Content-Length --------------------------------------------------------------------------------------------

class ContentLengthTests(unittest.TestCase):
    def test_negative_or_odd_lengths_get_400_at_once(self):
        node = start_node(self)
        host, port = node.server.server_address[:2]
        for length in (b"-1", b"abc", b"+5", b"1_0"):
            with socket.create_connection((host, port), timeout=5) as s:
                s.sendall(b"POST /consensus HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                          + length + b"\r\n\r\n{}")
                self.assertIn(b" 400 ", s.recv(200), length)  # without waiting for the client to hang up


# --- 3. what goes into round records, and how big a proposal can be ----------------------------------------------

class PaddedRegistrations(WorkerClient):
    def _signed(self, kind, body, entry=None):
        if kind == "register":
            env = seal(self.keys, kind, body, self.node.chain_id)
            env["pad"] = "x" * 2000  # outside the signature
            return self.node.post(f"/{kind}", env)
        return super()._signed(kind, body, entry)


class PaddedRevealsAndVerdicts(WorkerClient):
    def _signed(self, kind, body, entry=None):
        if kind == "reveal":
            body = {**body, "reveal": {**body["reveal"], "pad": "x" * 2000}}
            entry = body["reveal"]
        if kind == "verdict":
            body = {**body, "pad": "x" * 2000}  # inside the signature
        return super()._signed(kind, body, entry)


class RecordTests(unittest.TestCase):
    def test_records_hold_only_what_the_protocol_sends(self):
        node, api, req, workers, clients, stop, stop_node, genesis, tmp = start_network(self, 3)
        for i, cls in ((1, PaddedRegistrations), (2, PaddedRevealsAndVerdicts)):
            clients[i] = cls(node.url, workers[i], memory_mb=8192, flops_override=1e9, poll_s=0.1, cache_dir=tmp / f"pad{i}")
        job = TrainingJob("records-job", req.address, budget=100 * VIBE, fee_per_round=10 * VIBE, units_per_round=6)
        api.send_tx(req, "post_job", "system", job.budget, job.spec_json())
        logging.disable(logging.WARNING)  # the padders' complaints
        self.addCleanup(logging.disable, logging.NOTSET)
        run_until(node, clients, stop, 3)
        records = [node.store.load_record(h) for h in range(2, node.chain.tip.height + 1)]
        records = [r for r in records if r]
        self.assertTrue(records)
        regs = [e for r in records for e in r["registrations"]]
        self.assertTrue(regs)
        for env in regs:
            self.assertEqual(set(env), {"kind", "body", "ts", "chain_id", "pubkey", "sig"})  # what wire.seal makes
        self.assertNotIn(workers[1].address, {address_of(bytes.fromhex(e["pubkey"])) for e in regs}, "a padded one is refused")
        reveals = [d for r in records for d in r["reveals"]]
        self.assertIn(workers[2].address, {d["worker"] for d in reveals}, "a reveal is kept as parsed")
        for d in reveals:
            self.assertEqual(set(d), {"unit_id", "worker", "merkle_root", "salt", "final", "openings"})
        for env in (e for r in records for e in r["verdicts"]):
            self.assertLessEqual(set(env["body"]), {"round", "task_id", "ok", "bad_segment"})
        kinds = {json.loads(line)["kind"] for line in clients[0].receipt_file.read_text().splitlines()}
        self.assertLessEqual({"register", "commit", "reveal"}, kinds, "an honest worker's receipts still vouch for it")

    def test_message_size_and_fields(self):
        node = start_node(self)
        env = seal(REQ, "register", {"round": 1, "flops_per_sec": 1e9, "memory_mb": 1024, "pad": "x"}, "dos-net")
        with self.assertRaisesRegex(WireError, "only"):
            node.handle("register", env)
        env = seal(REQ, "register", {"round": 1, "flops_per_sec": " " * 5000 + "1e9", "memory_mb": 1024}, "dos-net")
        with self.assertRaisesRegex(WireError, "too large"):
            node.handle("register", env)

    def test_a_large_round_fits_a_proposal(self):
        peer = KeyPair.generate(b"other-validator")
        genesis = {**SOLO, "chain_id": "big-net", "stakes": {NODE.address: 10 * VIBE, peer.address: 10 * VIBE},
                   "validators": {NODE.address: "http://127.0.0.1:1", peer.address: "http://127.0.0.1:2"}}
        node = start_node(self, genesis)
        big = {"type": "proposal", "chain_id": "big-net", "height": 1, "round": 0, "valid_round": -1, "pubkey": "00",
               "sig": "00", "block": {}, "record": {"reveals": ["x" * 1000] * 6000}}  # ~6 MB: a round of ~1,500 units
        self.assertGreater(len(json.dumps(big)), MAX_JSON)
        self.assertEqual(NodeClient(node.url).post("/consensus", big), {"ok": True})
        solo = start_node(self)  # a chain's only validator never needs to read one (refused: 400, or a reset mid-send)
        with self.assertRaises(NodeError):
            NodeClient(solo.url).post("/consensus", {**big, "chain_id": "dos-net"})


# --- 4. transactions are passed on as parsed ------------------------------------------------------------------------

class TxRelayTests(unittest.TestCase):
    def test_only_the_tx_is_passed_on(self):
        node = start_node(self)
        sent = []
        node.peers.broadcast = lambda path, body: sent.append((path, body))
        tx = Tx.signed(REQ, "transfer", NODE.address, VIBE, 0, "", "dos-net")
        NodeClient(node.url).post("/tx", {**tx.to_dict(), "pad": "x" * 100_000})
        self.assertEqual(sent, [("/tx", tx.to_dict())])


# --- 5. datasets ----------------------------------------------------------------------------------------------------

def bomb(path: Path, n: int) -> Path:
    np.savez_compressed(path, x_train=np.zeros((n, 1), np.uint8), y_train=np.zeros(n, np.uint8),
                        x_val=np.zeros((1, 1), np.uint8), y_val=np.zeros(1, np.uint8))
    return path


class DatasetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_a_small_file_cant_unpack_into_gigabytes(self):
        path = bomb(self.dir / "bomb.npz", 40 * 2**20)  # ~40 KB on disk, ~700 MB once loaded
        self.assertLess(path.stat().st_size, 2**20)
        with self.assertRaisesRegex(ValueError, "MB of memory"):
            Dataset.from_npz(path)
        honest = write_npz(self.dir / "ok.npz", ModelSpec(), 1)
        self.assertEqual(Dataset.from_npz(honest).x_train.shape, (2048, 8))
        with self.assertRaisesRegex(ValueError, "MB of memory"):
            Dataset.from_npz(honest, max_bytes=2**10)
        np.savez(self.dir / "odd.npz", x_train=np.array(["a"]), y_train=np.zeros(1), x_val=np.zeros((1, 1)), y_val=np.zeros(1))
        with self.assertRaises(ValueError):
            Dataset.from_npz(self.dir / "odd.npz")
        (self.dir / "junk.npz").write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            Dataset.from_npz(self.dir / "junk.npz")

    def test_only_the_upload_that_completes_a_dataset_checks_it(self):
        node = start_node(self)
        api = NodeClient(node.url)
        sha = api.upload_dataset(REQ, write_npz(self.dir / "d.npz", ModelSpec(), 2))
        self.assertTrue(node.store.datasets.has(sha))
        loads = []

        def failing(path, *a, **kw):
            loads.append(path)
            raise MemoryError("a check that fails for some other reason")
        with mock.patch("vibehash.storage.Dataset.from_npz", side_effect=failing):
            for _ in range(3):  # PUTs to the finished dataset, claiming it's 1 byte
                ts = int(time.time())
                headers = {"X-VibeHash-Pubkey": REQ.public.hex(), "X-VibeHash-Ts": str(ts), "X-Total": "1", "X-Offset": "0",
                           "X-VibeHash-Sig": REQ.sign(canonical(["dos-net", "dataset", sha, 1, ts])).hex()}
                status, body = raw(node.url, "PUT", f"/dataset/{sha}", b"", headers)
                self.assertEqual(status, 200, body)
                self.assertEqual(json.loads(body)["received"], node.store.datasets.size(sha))
        self.assertEqual(loads, [], "a finished dataset isn't loaded again")
        self.assertTrue(node.store.datasets.has(sha), "nor ever deleted")
        bad = bomb(self.dir / "b.npz", 40 * 2**20)
        bad_sha = hashlib.sha256(bad.read_bytes()).hexdigest()
        with self.assertRaisesRegex(NodeError, "not a valid dataset"):  # a bad upload is still refused, and not kept
            api.upload(REQ, "dataset", bad, bad_sha)
        self.assertFalse(node.store.datasets.has(bad_sha))
        self.assertEqual(list(node.store.datasets.root.glob("*.part")), [])


# --- 6. job models -------------------------------------------------------------------------------------------------

class JobModelTests(unittest.TestCase):
    def test_an_untrained_job_has_no_weights_to_build(self):
        node = start_node(self)
        api = NodeClient(node.url)
        spec = ModelSpec(64, (1024, 1024), 10, "float32")
        job = TrainingJob("idle-job", REQ.address, budget=0, fee_per_round=0, spec=spec)
        posted_job(self, api, node, job)
        with mock.patch.object(ModelSpec, "init_params", autospec=True, side_effect=ModelSpec.init_params) as built:
            info = api.get("/job/idle-job")
            with self.assertRaisesRegex(NodeError, "hasn't been trained"):
                api._req("GET", "/job/idle-job/model?offset=0")
            self.assertEqual(built.call_count, 0, "the node built nothing")
        self.assertEqual(info["model"], {"height": 0, "bytes": job.checkpoint_bytes, "sha256": None})
        _, weights = api.download_model("idle-job")  # what `job download` does: the poster builds them
        self.assertEqual(weights, np.asarray(spec.init_params(job.init_seed), dtype="<f4").tobytes())
        with self.assertRaises(NodeError):
            api.download_model("idle-job", untrained=False)

    def test_relay_policy_on_job_size(self):
        node = start_node(self, max_job_bytes=2**20)
        api = NodeClient(node.url)
        big = TrainingJob("big-job", REQ.address, budget=0, fee_per_round=0, spec=ModelSpec(8, (512, 512), 4, "float32"))
        self.assertGreater(big.checkpoint_bytes, 2**20)
        with self.assertRaisesRegex(NodeError, "max-job-mb"):
            api.send_tx(REQ, "post_job", "system", 0, big.spec_json())
        posted_job(self, api, node, TrainingJob("small-job", REQ.address, budget=0, fee_per_round=0))

    def test_node_command_flag(self):
        made = {}

        class Stop(Exception):
            pass

        def fake(*a, **kw):
            made.update(kw)
            raise Stop
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        genesis = Path(tmp.name) / "genesis.json"
        genesis.write_text(json.dumps(SOLO))
        args = ["node", "--genesis", str(genesis), "--wallet", "w", "--data-dir", tmp.name]
        self.addCleanup(logging.disable, logging.NOTSET)
        logging.disable(logging.INFO)  # cli.main turns on INFO logging for the whole process
        with mock.patch("vibehash.node.Node", side_effect=fake), mock.patch.object(cli, "load_wallet", return_value=NODE):
            with self.assertRaises(Stop):
                cli.main([*args, "--max-job-mb", "8"])
            self.assertEqual(made["max_job_bytes"], 8 * 2**20)
            with self.assertRaises(Stop):
                cli.main(args)
            self.assertEqual(made["max_job_bytes"], node_module.MAX_JOB_BYTES)


# --- 7. rate limits -------------------------------------------------------------------------------------------------

class RateLimitPathTests(unittest.TestCase):
    def test_bucket_follows_the_route(self):
        for path in ("/tx;x", "/tx;x?y=1", "http://example/tx"):
            self.assertEqual(bucket_of("POST", path), "tx", path)
        node = start_node(self, limits=Limits.parse("tx=0.001/1", local=True))
        codes = [raw(node.url, "POST", "/tx;x", b"{}")[0] for _ in range(2)]
        self.assertEqual(codes[1], 429, "the second transaction is over the tx budget, however the path is spelled")


# --- 8. sync --------------------------------------------------------------------------------------------------------

class SyncTests(unittest.TestCase):
    def setUp(self):
        self.port = cli._free_ports(1)[0]
        self.url = f"http://127.0.0.1:{self.port}"
        self.genesis = {**SOLO, "chain_id": "sync-net", "validators": {NODE.address: self.url}}
        self.leader = start_node(self, self.genesis, port=self.port)
        tx = Tx.signed(REQ, "transfer", NODE.address, VIBE, 0, "", "sync-net")
        NodeClient(self.leader.url).post("/tx", tx.to_dict())
        wait_for(lambda: self.leader.chain.tip.height >= 1)

    def follower(self, peers: list[str]) -> Node:
        """A node that doesn't vote, not started: the test drives its sync."""
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        f = Node(Path(tmp.name) / "f", self.genesis, KeyPair.generate(b"follower"), port=0, times=FAST, peers=peers)
        self.addCleanup(f.store.close)
        self.addCleanup(f.peers.close)
        self.addCleanup(f.server.server_close)
        return f

    def test_a_peer_that_never_sends_anything_new_doesnt_trap_sync(self):
        block0 = NodeClient(self.leader.url).get("/blocks?from=0&to=0")
        evil, hits = fake_node(self, {"/status": {"genesis_hash": self.leader.chain.genesis_hash, "height": 10**9},
                                      "/blocks": block0})
        f = self.follower([evil, self.leader.url])
        done = threading.Thread(target=f._sync, daemon=True)
        done.start()
        done.join(timeout=20)
        self.assertFalse(done.is_alive(), "sync spun on a peer that serves the same old block forever")
        self.assertLessEqual(hits.get("/blocks", 0), 2)
        self.assertEqual(f.chain.tip.height, self.leader.chain.tip.height, "and the next peer was asked")

    def test_block_pages_stop_at_a_byte_budget_and_sync_still_catches_up(self):
        tx = Tx.signed(REQ, "transfer", NODE.address, VIBE, 1, "", "sync-net")
        NodeClient(self.leader.url).post("/tx", tx.to_dict())
        wait_for(lambda: self.leader.chain.tip.height >= 2)
        with mock.patch.object(node_module, "MAX_BLOCKS_REPLY", 1):  # every block is over it: one a page
            self.assertEqual([d["height"] for d in NodeClient(self.leader.url).get("/blocks?from=0&to=99")], [0])
            f = self.follower([self.leader.url])
            f._sync()
        self.assertEqual(f.chain.tip.hash, self.leader.chain.tip.hash)

    def test_malformed_replies_and_leader_hints_from_anyone(self):
        gh = self.leader.chain.genesis_hash
        junk, _ = fake_node(self, {"/status": [1, 2, 3]})
        holes, _ = fake_node(self, {"/status": {"genesis_hash": gh, "consensus": {}, "leader": "http://evil.example"}})
        liar, _ = fake_node(self, {"/status": {"genesis_hash": gh, "height": 5, "consensus": {}, "leader": "http://evil.example"},
                                   "/blocks": [{"height": "x"}]})
        f = self.follower([junk, holes, liar, self.leader.url])
        f._sync()
        self.assertEqual(f.chain.tip.height, self.leader.chain.tip.height, "the honest peer was still asked")
        self.assertEqual(f.leader_hint, self.url, "only a validator the chain lists is followed")


# --- 9. what a client reads from a node -----------------------------------------------------------------------------

class ClientLimitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_replies_are_capped(self):
        url, _ = fake_node(self, {"/status": b"[" + b"0," * 2**20 + b"0]", "/deep": b"[" * 100_000})
        with mock.patch("vibehash.client.MAX_REPLY", 2**20):
            with self.assertRaisesRegex(NodeError, "larger than"):
                NodeClient(url).get("/status")
        with self.assertRaises(NodeError):
            NodeClient(url).get("/deep")

    def test_downloads_take_only_the_size_they_can_be(self):
        bid = "ab" * 32
        url, hits = fake_node(self, {f"/dataset/{bid}/status": {"complete": True, "size": MAX_DATASET + 1, "received": 0},
                                     f"/blob/{bid}/status": {"complete": True, "size": 10**6, "received": 0},
                                     f"/dataset/{bid}": b"", f"/blob/{bid}": b""})
        store = BlobStore(self.dir / "s")
        for kind, size in (("dataset", None), ("blob", 4000)):
            with self.assertRaises(NodeError):
                NodeClient(url).download(kind, bid, store, size=size)
            self.assertEqual(hits.get(f"/{kind}/{bid}", 0), 0, f"no {kind} chunk was fetched")
        spec = {"id": "j", "fee_per_round": 0, "model": {"n_in": 8, "n_hidden": 32, "n_out": 4}, "precision": "float64",
                "data": {"kind": "synthetic", "seed": 7}, "init_seed": 11, "lr": 0.1, "batch_size": 32, "steps_per_unit": 40,
                "checkpoint_every": 5, "units_per_round": 16, "memory_mb": 1024, "tolerance": 1e-6, "max_update_ratio": 0.5}
        url, hits = fake_node(self, {"/job/j": {"spec": spec, "owner": "a", "model": {"height": 3, "bytes": 10**10, "sha256": "00"}},
                                     "/job/j/model": b""})
        with self.assertRaisesRegex(NodeError, "spec makes it"):
            NodeClient(url).download_model("j")
        self.assertEqual(hits.get("/job/j/model", 0), 0)

    def test_stale_partial_downloads_are_pruned(self):
        store = BlobStore(self.dir / "d")
        old, new, keep = (store.root / f"{c * 64}.part" for c in "abc")
        for p in (old, new, keep):
            p.write_bytes(b"x")
        import os
        os.utime(old, (time.time() - 2 * 86400,) * 2)
        self.assertEqual(store.prune_partial(older_than=86400), 1)
        self.assertEqual(store.prune_partial(keep={"c" * 64}), 1)
        self.assertEqual([p.name for p in store.root.iterdir()], [keep.name])

    def test_a_worker_keeps_only_the_partial_download_it_needs(self):
        node = start_node(self)
        api = NodeClient(node.url)
        sha = api.upload_dataset(REQ, write_npz(self.dir / "d.npz", ModelSpec(), 2))
        job = TrainingJob("data-job", REQ.address, budget=0, fee_per_round=0, data_source={"kind": "npz", "sha256": sha})
        posted_job(self, api, node, job)
        w = WorkerClient(node.url, KeyPair.generate(b"w"), memory_mb=1024, flops_override=1e9, cache_dir=self.dir / "cache")
        stray = w.datasets.root / ("f" * 64 + ".part")
        stray.write_bytes(b"x" * 1000)  # a download some other job's node never finished
        w.job("data-job")
        self.assertFalse(stray.exists())
        self.assertTrue(w.datasets.has(sha))


# --- 10. bad numbers --------------------------------------------------------------------------------------------------

class BadNumberTests(unittest.TestCase):
    def test_bad_numbers_get_400_not_500(self):
        node = start_node(self)
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        sha = node.store.add_dataset(write_npz(Path(tmp.name) / "d.npz", ModelSpec(), 2))
        for path in (f"/dataset/{sha}?offset=-1", f"/dataset/{sha}?offset={10**20}", f"/blocks?from={10**20}",
                     "/blocks?from=-5", "/blocks?to=x", f"/round/{10**20}/record", "/round/-1/record"):
            self.assertEqual(raw(node.url, "GET", path)[0], 400, path)
        self.assertEqual(raw(node.url, "GET", f"/dataset/{sha}?offset=0")[0], 200)


# --- 11. a relay altering a proposal ------------------------------------------------------------------------------------

class MalleabilityTests(unittest.TestCase):
    """A relayed copy of a proposal with its last transaction repeated, or its record swapped, keeps the block id and
    the proposer's signature. It mustn't make validators turn down the real one."""

    def setUp(self):
        self.keys = [KeyPair.generate(f"mv{i}".encode()) for i in range(4)]
        a, b = KeyPair.generate(b"ma"), KeyPair.generate(b"mb")
        self.chain = Chain({a.address: 100 * VIBE, b.address: 100 * VIBE}, {}, chain_id="mall-net",
                           validators={k.address: f"http://v{i}:1" for i, k in enumerate(self.keys)},
                           bootstrap={k.address: 10 * VIBE for k in self.keys})
        self.vals = self.chain.state.validator_set()
        self.leader = next(k for k in self.keys if k.address == proposer_order(self.vals, 1)[0])
        self.victim = next(k for k in self.keys if k is not self.leader)
        txs = [Tx.signed(a, "transfer", b.address, 1, 0, "", "mall-net"), Tx.signed(a, "transfer", b.address, 2, 1, "", "mall-net"),
               Tx.signed(b, "transfer", a.address, 3, 0, "", "mall-net")]
        block, _ = self.chain.build_block(None, txs, 1)
        block.sign(self.leader)
        self.value = {"block": block.to_dict(), "record": None}
        self.bid = block.hash.hex()
        self.genuine = proposal(self.leader, "mall-net", 1, 0, -1, self.value, self.bid)
        self.repeated = copy.deepcopy(self.genuine)
        self.repeated["block"]["txs"].append(self.repeated["block"]["txs"][-1])
        self.swapped = {**copy.deepcopy(self.genuine), "record": {"junk": "x" * 1000}}
        self.sent, self.committed = [], []
        self.e = Engine("mall-net", self.victim, Timeouts(propose=30, vote=1, grow=1, regossip=1), validators=lambda: dict(self.vals),
                        validate=self.validate, commit=lambda h, v, c: self.committed.append(v), broadcast=self.sent.append)
        self.e.start(1, now=0)

    def validate(self, value):
        if value["record"] is not None:  # (a block without a round has no record: node.py replays one that has)
            return False
        try:
            block = Block.from_dict(value["block"])
            if len({t.txid for t in block.txs}) != len(block.txs):
                return False
            apply_block(self.chain.state, block, "mall-net", self.chain.tip)
            return True
        except InvalidBlock:
            return False

    def prevotes(self):
        return [m["block"] for m in self.e.mine if m["type"] == "prevote"]  # (self.sent repeats them every few seconds)

    def test_a_bad_copy_first_doesnt_cost_the_real_ones_vote(self):
        for bad in (self.repeated, self.swapped):
            self.assertEqual(Block.from_dict(bad["block"]).hash.hex(), self.bid)  # the same id, so the same signature...
            self.assertFalse(self.validate({"block": bad["block"], "record": bad["record"]}))  # ...on a bad block
            self.setUp()
            self.e.receive(copy.deepcopy(bad), now=0.1)
            self.e.receive(copy.deepcopy(self.genuine), now=0.2)
            self.assertEqual(self.prevotes(), [self.bid])

    def test_the_real_one_arriving_late_still_gets_decided(self):
        self.e.receive(copy.deepcopy(self.repeated), now=0.1)
        self.e.tick(now=5)  # no good copy in time: nil
        self.assertEqual(self.prevotes(), [None])
        self.e.receive(copy.deepcopy(self.genuine), now=5.1)  # the proposer re-sends its own
        for k in self.keys:
            if k is not self.victim:
                self.e.receive(vote(k, "mall-net", "precommit", 1, 0, self.bid), now=5.2)
        self.assertEqual(len(self.committed), 1, "the block the others finalised is final here too")
        self.assertEqual(self.committed[0], self.value)


if __name__ == "__main__":
    unittest.main()
