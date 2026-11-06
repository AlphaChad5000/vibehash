"""What a node tells clients, exchanges and listing sites about itself (EXCHANGES.md): its versions in /status and the
nudges that follow from them, the supply endpoints and that their numbers add up, block and tx ids in /blocks, and
the explorer page with its Content-Security-Policy. Live nodes on free localhost ports; no PyTorch."""
import contextlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import unittest
import urllib.request
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import vibehash
from vibehash import advice, cli, client, ledger, node as node_module, selftest, supply
from vibehash import economics as econ
from vibehash import wallet as wallets
from vibehash.client import MIN_CLIENT, NodeClient, NodeError
from vibehash.crypto import KeyPair, checksummed
from vibehash.ledger import Tx
from vibehash.ratelimit import Limits, bucket_of
from vibehash.tasks import SYN_PER_VIBE
from tests.test_core import network
from tests.test_dos import NODE, REQ, SOLO, raw, start_node, wait_for

VIBE = SYN_PER_VIBE


class Recorder(BaseHTTPRequestHandler):
    """A node from the outside: answers `routes` (path -> JSON) and keeps each request's User-Agent."""
    routes: dict = {}
    agents: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.agents.append(self.headers.get("User-Agent"))
        out = self.routes.get(self.path.split("?")[0])
        body = json.dumps(out if out is not None else {"error": "not found"}).encode()
        self.send_response(200 if out is not None else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET


def recorder(test, routes: dict) -> tuple[str, list]:
    handler = type("R", (Recorder,), {"routes": routes, "agents": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    test.addCleanup(srv.server_close)
    test.addCleanup(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}", handler.agents


# What a node from before versions answers /status with (0.5.15's keys), and the same from a node on newer rules.
OLD_STATUS = {"chain_id": "old-net", "genesis_hash": "ab" * 32, "height": 5, "tip": "cd" * 32, "phase": "idle",
              "phase_ends": 0, "progress": None, "round": None, "job_id": None, "attempt": None, "beacon": None,
              "beacon_commit": None, "challenge_segments": 3, "min_stake": 5 * VIBE, "mempool": 0, "chunk_bytes": 2**23,
              "keeps_checkpoints_for": 10, "economics": econ.TESTNET.to_dict(), "subsidy": 50 * VIBE, "min_fee": 0,
              "earn_in": True, "waiting": {}, "halted": None}
NEWER = OLD_STATUS | {"version": "9.0.0", "block_version": ledger.VERSION + 1, "tip_version": ledger.VERSION + 1,
                      "min_client": MIN_CLIENT, "paid_rounds": None}


class VersionTests(unittest.TestCase):
    def setUp(self):
        client._NUDGED.clear()  # (each test's nudges start unsaid)

    def test_status_names_the_versions(self):
        node = start_node(self)
        s = NodeClient(node.url).status()
        self.assertEqual(s["version"], vibehash.__version__)
        self.assertRegex(s["version"], r"^\d+\.\d+\.\d+")
        self.assertEqual((s["block_version"], s["tip_version"]), (ledger.VERSION, 1))  # (the genesis counts as 1)
        self.assertEqual(s["min_client"], MIN_CLIENT)
        self.assertIsNone(s["paid_rounds"], "a genesis that counts blocks has no paid-round clock")
        self.assertIsNone(client.too_old(s), "a node of this release is no problem")

    def test_paid_rounds_on_a_genesis_that_counts_them(self):
        genesis = SOLO | {"chain_id": "paid-net", "economics": econ.NEW_SCHEDULES["testnet"].to_dict()}
        node = start_node(self, genesis)
        self.assertEqual(NodeClient(node.url).status()["paid_rounds"], 0)
        self.assertEqual(NodeClient(node.url).get("/supply")["paid_rounds"], 0)

    def test_min_client_is_the_nodes_to_set(self):
        node = start_node(self, min_client="0.9.1")
        self.assertEqual(NodeClient(node.url).status()["min_client"], "0.9.1")
        with self.assertRaises(ValueError):
            start_node(self, min_client="latest")
        a = cli.parser().parse_args(["node", "--genesis", "g.json", "--wallet", "w.json", "--data-dir", "d",
                                     "--min-client", "0.6.0"])
        self.assertEqual(a.min_client, "0.6.0")
        self.assertEqual(cli.parser().parse_args(["node", "--genesis", "g", "--wallet", "w", "--data-dir", "d"]).min_client,
                         MIN_CLIENT)

    def test_every_request_names_the_release(self):
        url, agents = recorder(self, {"/status": OLD_STATUS, "/tx": {"txid": "00"}, "/account/x": {"nonce": 0}})
        api = NodeClient(url)
        api.status()
        api.get("/account/x")
        api.post("/tx", {})
        selftest._get(api, "/status")  # selftest builds its own request and opens it through NodeClient
        self.assertEqual(agents, [f"vibehash/{vibehash.__version__}"] * 4)
        self.assertEqual(client.USER_AGENT, f"vibehash/{vibehash.__version__}")

    def test_a_node_from_before_versions_works_as_before(self):
        url, _ = recorder(self, {"/status": OLD_STATUS, "/validators": {}})
        self.assertIsNone(client.too_old(OLD_STATUS))
        with self.assertNoLogs("vibehash.worker", logging.WARNING):
            self.assertEqual(NodeClient(url).status()["chain_id"], "old-net")
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertNoLogs("vibehash.worker", logging.WARNING):
            cli.main(["validator", "list", "--node", url])
        self.assertIn("single node", out.getvalue())

    def test_newer_rules_are_said_once_and_commands_carry_on(self):
        url, _ = recorder(self, {"/status": NEWER, "/validators": {}})
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertLogs("vibehash.worker", logging.WARNING) as logs:
            cli.main(["validator", "list", "--node", url])
            NodeClient(url).status()  # the same advice for the same node isn't said again
        self.assertIn("single node", out.getvalue(), "the command carried on")
        self.assertEqual(len(logs.records), 1)
        line = logs.records[0].getMessage()
        self.assertNotIn("\n", line)
        self.assertIn(f"version {ledger.VERSION + 1} blocks", line)
        self.assertIn("pip install -U vibehash", line)

    def test_a_newer_min_client_is_said_too(self):
        problem = client.too_old(OLD_STATUS | {"tip_version": 1, "min_client": "99.0", "version": "99.0.1"}, mine="0.5.15")
        self.assertIn("works with Vibe Hash 99.0 or newer", problem)
        self.assertIn("99.0.1", problem)
        self.assertIsNone(client.too_old(OLD_STATUS | {"tip_version": 1, "min_client": "0.5.15"}, mine="0.5.15"))
        self.assertIsNone(client.too_old(OLD_STATUS | {"min_client": "0.6"}, mine="unknown"), "an unknown release isn't judged")
        # Node text reaches a terminal only when it looks like a version.
        sneaky = client.too_old(OLD_STATUS | {"min_client": "99\x1b[2J", "version": "\x1b]0;x\x07"}, mine="0.5.15")
        self.assertNotIn("\x1b", sneaky)

    def test_versions_compare_as_numbers(self):
        self.assertEqual(client.parse_version("0.5.15"), (0, 5, 15))
        self.assertEqual(client.parse_version("0.5.16rc1"), (0, 5, 16))
        self.assertEqual(client.parse_version("unknown"), ())
        self.assertLess(client.parse_version("0.5.9"), client.parse_version("0.5.10"))
        self.assertEqual(client.agent_version("vibehash/0.5.16"), (0, 5, 16))
        self.assertEqual(client.agent_version("Python-urllib/3.12"), ())
        self.assertEqual(client.agent_version(None), ())

    def test_a_worker_refuses_to_start_on_newer_rules(self):
        url, _ = recorder(self, {"/status": NEWER})
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        wallets.save(KeyPair.generate(b"too old"), tmp / "w.json", None)
        with mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(tmp / "home")}), \
                mock.patch.object(cli, "WorkerClient", side_effect=AssertionError("got as far as the GPU")), \
                self.assertRaises(SystemExit) as e:
            cli.main(["worker", "--node", url, "--wallet", str(tmp / "w.json"), "--device", "cpu", "--plain",
                      "--cache-dir", str(tmp / "cache")])
        self.assertIn("newer rules", str(e.exception.code))
        self.assertIn("pip install -U vibehash", str(e.exception.code))

    def test_the_node_turns_away_a_register_from_an_older_release(self):
        node = start_node(self, min_client="0.6.0")
        status, body = raw(node.url, "POST", "/register", b"{}", {"User-Agent": "vibehash/0.5.20",
                                                                   "Content-Type": "application/json"})
        self.assertEqual(status, 400)
        error = json.loads(body)["error"]
        self.assertIn("0.6.0 or newer", error)
        self.assertIn("pip install -U vibehash", error)
        self.assertEqual(advice.explain(NodeError(f"POST /register: {error}", 400)), error)
        # A worker from before releases named themselves (urllib's own User-Agent) is told to upgrade too.
        status, body = raw(node.url, "POST", "/register", b"{}", {"User-Agent": "Python-urllib/3.12"})
        self.assertEqual(status, 400)
        self.assertIn("0.6.0 or newer", json.loads(body)["error"])
        # The release it asks for, a newer one, another client, or none, get as far as the message's own checks.
        for agent in ("vibehash/0.6.0", "vibehash/1.0", "curl/8.4", None):
            status, body = raw(node.url, "POST", "/register", b"{}", {"User-Agent": agent} if agent else {})
            self.assertEqual(status, 400)
            self.assertIn("malformed envelope", json.loads(body)["error"], agent)

    def test_selftest_says_when_this_computer_is_too_old(self):
        url, _ = recorder(self, {"/status": NEWER, "/network": {"jobs": []}})
        rep = selftest.Report(quiet=True)
        with mock.patch.object(selftest, "_speed", lambda *a: None):
            selftest.network(rep, url, None, offline=False)
        self.assertTrue(rep.failed("versions"))
        self.assertIn("pip install -U vibehash", rep.get("versions").text)


class SupplyTests(unittest.TestCase):
    def check(self, s, height):
        """/supply's numbers for State `s` add up, every way they should."""
        got = supply.supply(s, height)
        parts = got["circulating"] + got["unspendable"] + got["staked"] + got["unbonding"] + got["escrow"]
        self.assertEqual(parts, got["total"])
        self.assertEqual(got["total"], s.total_supply())
        self.assertEqual(got["total"], s.expected_supply())
        self.assertEqual(got["total"], got["genesis"] + got["minted"] - got["burned"])
        self.assertLessEqual(got["bootstrap"], got["staked"])
        for k, v in got["vibe"].items():  # each VIBE string is the syn amount exactly
            self.assertRegex(v, r"^\d+(\.\d{1,8})?$")
            self.assertEqual(Decimal(v) * SYN_PER_VIBE, got[k], k)
        return got

    def test_the_numbers_reconcile_on_a_busy_chain(self):
        """Transfers (one to a recipient no key has), stakes, unstakes, a job and its cancelling, slashes of cheaters,
        bootstrap stake: after every block, where the coins are adds up to what was created less what was burned."""
        proto, workers, job = network([("honest",)] * 3 + [("shortcut",), ("honest", "false_accuser")],
                                      replication_rate=0.5)
        chain = proto.chain
        req = KeyPair.generate(b"req")
        friend = KeyPair.generate(b"friend")
        self.check(chain.state, 0)

        def send(keys, kind, recipient, amount, memo=""):
            tx = Tx.signed(keys, kind, recipient, amount, chain.state.nonces.get(keys.address, 0)
                           + sum(t.sender == keys.address for t in proto.coord.mempool), memo, chain.chain_id)
            with mock.patch.object(ledger, "_relayable", lambda tx: None):  # (a non-canonical recipient: below)
                proto.submit(tx)

        send(req, "transfer", friend.address, 25 * VIBE)
        send(req, "transfer", "vibe1" + "AB" * 18, 2 * VIBE)  # a spelling no key's address_of makes
        send(req, "stake", req.address, 7 * VIBE)
        send(workers[0].keys, "unstake", workers[0].address, 3 * VIBE)
        seen = []
        for _ in range(4):
            proto.run_round()
            seen.append(self.check(chain.state, chain.tip.height))
        send(req, "cancel_job", "system", 0, job.job_id)
        proto.run_round()
        last = self.check(chain.state, chain.tip.height)
        self.assertGreater(last["burned"], 0, "cheaters were slashed and unclaimed pools burned")
        self.assertGreater(last["minted"], 0)
        self.assertEqual(last["unspendable"], 2 * VIBE)
        self.assertTrue(any(g["unbonding"] for g in seen))
        self.assertTrue(any(g["escrow"] for g in seen))
        self.assertEqual(last["escrow"], 0, "the job was cancelled")
        self.assertTrue(any(tx.kind == "slash" for b in chain.blocks for tx in b.txs))

    def test_bootstrap_stake_counts_as_staked_until_burned(self):
        v = KeyPair.generate(b"bootstrap validator")
        chain = ledger.Chain({REQ.address: 10 * VIBE}, bootstrap={v.address: 20 * VIBE},
                             validators={v.address: "http://127.0.0.1:1"})
        got = self.check(chain.state, 0)
        self.assertEqual((got["staked"], got["bootstrap"], got["circulating"]), (20 * VIBE, 20 * VIBE, 10 * VIBE))

    def test_halvings_on_each_clock(self):
        chain = ledger.Chain({REQ.address: VIBE}, economics=econ.NEW_SCHEDULES["testnet"])
        s = chain.state
        s.paid_rounds = 419
        h = supply.halving(s, 5000)
        self.assertEqual((h["counts"], h["halvings"], h["next_at"], h["to_go"]), ("paid_rounds", 1, 420, 1))
        self.assertEqual((supply.supply(s, 5000)["subsidy"], h["subsidy_after"]), (25 * VIBE, 25 * VIBE // 2))
        old = ledger.Chain({REQ.address: VIBE}).state  # counts heights: the next block is 210, a halving
        h = supply.halving(old, 209)
        self.assertEqual((h["counts"], h["halvings"], h["next_at"], h["to_go"]), ("blocks", 1, 420, 210))
        h = supply.halving(old, 208)
        self.assertEqual((h["halvings"], h["next_at"], h["to_go"]), (0, 210, 1))
        s.paid_rounds = 64 * 210
        self.assertIsNone(supply.halving(s, 0)["next_at"], "nothing left to halve")
        capped = ledger.Chain({REQ.address: VIBE}, economics=econ.MAINNET).state
        self.assertEqual(supply.supply(capped, 0)["max_supply"], 21_000_000 * VIBE)

    def test_vibe_strings(self):
        self.assertEqual([supply.vibe(n) for n in (0, 1, 150_000_000, 10**8, 21 * 10**14, 123_456_789_012)],
                         ["0", "0.00000001", "1.5", "1", "21000000", "1234.56789012"])

    def test_the_endpoints(self):
        node = start_node(self)
        api = NodeClient(node.url)
        friend = KeyPair.generate(b"friend of the endpoints")
        api.send_tx(REQ, "transfer", friend.address, 12 * VIBE + 34)
        wait_for(lambda: node.chain.state.balance(friend.address))  # (send_tx reads the nonce from the chain)
        api.send_tx(REQ, "stake", REQ.address, 5 * VIBE)
        wait_for(lambda: node.chain.state.stake(REQ.address))
        got = api.get("/supply")
        self.assertEqual(got, json.loads(json.dumps(supply.supply(node.chain.state, node.chain.tip.height))))
        self.check(node.chain.state, node.chain.tip.height)
        self.assertEqual(got["total"], 510 * VIBE)
        self.assertEqual(got["staked"], 15 * VIBE)
        for path, key in (("/supply/total", "total"), ("/supply/circulating", "circulating")):
            r = urllib.request.urlopen(node.url + path, timeout=10)
            self.assertEqual(r.headers["Content-Type"], "text/plain; charset=utf-8")
            text = r.read().decode()
            self.assertRegex(text, r"^\d+(\.\d{1,8})?$")  # a number alone, as CoinGecko and CoinMarketCap read it
            self.assertEqual(Decimal(text) * SYN_PER_VIBE, got[key])
        self.assertEqual(urllib.request.urlopen(node.url + "/supply/total", timeout=10).read(), b"510")

    def test_blocks_carry_their_hash_and_txids(self):
        node = start_node(self)
        api = NodeClient(node.url)
        txid = api.send_tx(REQ, "transfer", KeyPair.generate(b"ids").address, VIBE)
        wait_for(lambda: node.chain.tip.height >= 1)
        page = api.get(f"/blocks?from=0&to={node.chain.tip.height}")
        for d, b in zip(page, node.chain.blocks):
            self.assertEqual(d["hash"], b.hash.hex())
            self.assertEqual(d["proposer"], ledger.address_of(b.proposer_pubkey) if b.height else None)
            self.assertEqual(ledger.Block.from_dict(d).hash.hex(), d["hash"], "readers ignore the extra fields")
            self.assertEqual([t["txid"] for t in d["txs"]], [t.txid.hex() for t in b.txs])
        self.assertIn(txid, [t["txid"] for d in page for t in d["txs"]])

    def test_accounts_read_a_checksummed_address(self):
        node = start_node(self)
        api = NodeClient(node.url)
        shown = checksummed(REQ.address)
        self.assertNotEqual(shown, REQ.address)
        got = api.get(f"/account/{shown}")
        self.assertEqual((got["address"], got["balance"]), (REQ.address, 500 * VIBE))


class ExplorerTests(unittest.TestCase):
    def test_served_with_a_strict_policy(self):
        node = start_node(self)
        r = urllib.request.urlopen(node.url + "/explorer", timeout=10)
        html = r.read().decode()
        self.assertEqual(r.headers["Content-Type"], "text/html; charset=utf-8")
        csp = r.headers["Content-Security-Policy"]
        for part in ("default-src 'none'", "connect-src 'self'", "base-uri 'none'", "frame-ancestors 'none'"):
            self.assertIn(part, csp)
        self.assertNotIn("unsafe", csp, "inline code is allowed by its hash, not wholesale")
        self.assertRegex(csp, r"script-src 'sha256-[A-Za-z0-9+/=]+'")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        # Self-contained: nothing from elsewhere, and node text is never put in as markup.
        self.assertNotRegex(html, r"""(src|href)=["']?(https?:)?//""")
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("insertAdjacentHTML", html)
        self.assertNotIn("document.write", html)
        self.assertNotIn(" style=", html, "a style attribute would need 'unsafe-inline'")
        self.assertLess(len(html.splitlines()), 800)
        for path in ("/status", "/blocks", "/supply", "/validators", "/account/", "/job/", "/network"):
            self.assertIn(path, html)

    def test_its_requests_are_reads(self):
        for path in ("/explorer", "/supply", "/supply/total", "/supply/circulating"):
            self.assertEqual(bucket_of("GET", path), "read")
        node = start_node(self, limits=Limits.parse("read=1/2", local=True))
        codes = [raw(node.url, "GET", "/explorer")[0] for _ in range(4)]
        self.assertEqual(codes[:2], [200, 200])
        self.assertIn(429, codes[2:], "the page is limited like any other read")


if __name__ == "__main__":
    unittest.main()
