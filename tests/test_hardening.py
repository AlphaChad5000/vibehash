"""Production hardening: recovery words, transaction fees, the issuance schedule, TLS and rate limits.

Vectors come from the official sources, copied here so the tests never touch the network: BIP39 from
trezor/python-mnemonic's vectors.json (passphrase "TREZOR"), SLIP-0010 from satoshilabs/slips.
"""
import contextlib
import hashlib
import io
import socket
import threading
import urllib.error
import urllib.request
import json
import logging
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from vibehash import cli, crypto, fees, mnemonic, ratelimit, settings, tls
from vibehash import economics as econ
from vibehash import wallet as wallets
from vibehash.client import NodeClient, NodeError
from vibehash.coordinator import Coordinator, CoordinatorConfig, ProtocolError
from vibehash.crypto import KeyPair, public_key
from vibehash.ledger import SYSTEM, Chain, InvalidBlock, Tx
from vibehash.node import Node, PhaseTimes
from vibehash.protocol import Protocol
from vibehash.ratelimit import Limits, RateLimiter, bucket_of
from vibehash.storage import Store
from vibehash.tasks import SYN_PER_VIBE, TrainingJob
from vibehash.worker import OFFICE_DESKTOP, HardwareProfile, Worker

VIBE = SYN_PER_VIBE
ABANDON_12 = "abandon " * 11 + "about"
ALICE, BOB = KeyPair.generate(b"alice"), KeyPair.generate(b"bob")
# A genesis, and the hashes the code before fees and schedules gave it: blocks already on disk must keep them.
COMPAT_GENESIS = {"chain_id": "compat-net", "timestamp": 1700000000, "allocations": {ALICE.address: 100 * VIBE},
                  "stakes": {ALICE.address: 10 * VIBE}}
COMPAT_GENESIS_HASH = "b0f27cd812f3814d236a8f0da58309661029eac01b939001943ccce454250d35"
COMPAT_BLOCK1_HASH = "415d8928b332aff32754ace7fa6161a7ff58bf93cc3fd29fec82f4a128db2c6e"


def network(n_workers=3, economics=None, config=None, budget=100 * VIBE):
    """An in-process network (protocol.py) with honest, reliable workers and one job posted."""
    hw = HardwareProfile(OFFICE_DESKTOP.name, OFFICE_DESKTOP.flops_per_sec, OFFICE_DESKTOP.memory_mb, reliability=1.0)
    workers = [Worker(f"w{i}", KeyPair.generate(f"w{i}".encode()), hw, "honest") for i in range(n_workers)]
    req = KeyPair.generate(b"req")
    chain = Chain({req.address: 1000 * VIBE}, stakes={w.address: 10 * VIBE for w in workers}, economics=economics)
    proto = Protocol(chain, workers, config or CoordinatorConfig(), seed=3)
    job = TrainingJob("job", req.address, budget=budget, fee_per_round=10 * VIBE, units_per_round=12)
    proto.post_job(req, job)
    return proto, workers, req, job


class Cli:
    """Runs `vibehash ...` in a scratch home, as test_network's command-line tests do."""

    def setUp(self):
        home = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(home.cleanup)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": home.name, "VIBEHASH_PASSWORD": "test-only"})
        patch.start()
        self.addCleanup(patch.stop)
        for var in ("VIBEHASH_NODE", "VIBEHASH_WALLET", "VIBEHASH_MNEMONIC", "VIBEHASH_MNEMONIC_PASSPHRASE",
                    "VIBEHASH_TLS_FINGERPRINT", "VIBEHASH_TLS_CA"):
            os.environ.pop(var, None)
        logging.disable(logging.INFO)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.home = Path(home.name)

    def run_cli(self, *argv) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main([str(a) for a in argv])
            except SystemExit as e:
                if e.code not in (None, 0):
                    self.fail(f"vibehash {' '.join(map(str, argv))} exited with {e.code!r}\n{out.getvalue()}")
        return out.getvalue()


# --- recovery words ----------------------------------------------------------------------------

TREZOR = [  # entropy, words, seed with the passphrase "TREZOR"
    ("00000000000000000000000000000000", ABANDON_12,
     "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04"),
    ("9e885d952ad362caeb4efe34a8e91bd2", "ozone drill grab fiber curtain grace pudding thank cruise elder eight picnic",
     "274ddc525802f7c828d8ef7ddbcdc5304e87ac3535913611fbbfa986d0c9e5476c91689f9c8a54fd55bd38606aa6a8595ad213d4c9c9f9aca3fb217069a41028"),
    ("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", "zoo " * 23 + "vote",
     "dd48c104698c30cfe2b6142103248622fb7bb0ff692eebb00089b32d22484e1613912f0a5b694407be899ffd31ed3992c456cdf60f5d4564b8ba3f05a69890ad"),
    ("9f6a2878b2520799a44ef18bc7df394e7061a224d2c33cd015b157d746869863",
     "panda eyebrow bullet gorilla call smoke muffin taste mesh discover soft ostrich alcohol speed nation flash devote level "
     "hobby quick inner drive ghost inside",
     "72be8e052fc4919d2adf28d5306b5474b0069df35b02303de8c1729c9538dbb6fc2d731d5f832193cd9fb6aeecbc469594a70e3dd50811b5067f3b88b28c3e8d"),
    ("f585c11aec520db57dd353c69554b21a89b20fb0650966fa0a9d6f74fd989d8f",
     "void come effort suffer camp survey warrior heavy shoot primary clutch crush open amazing screen patrol group space "
     "point ten exist slush involve unfold",
     "01f5bced59dec48e362f2c45b5de68b9fd6c92c6634f44d6d40aab69056506f0e35524a518034ddc1192e1dacd32c1ed3eaa3c3b131c88ed8e7e54c49a5d0998"),
]

SLIP10 = [  # seed, path, chain code, private key, public key (SLIP-0010's 0x00-prefixed form)
    ("000102030405060708090a0b0c0d0e0f", "m",
     "90046a93de5380a72b5e45010748567d5ea02bbf6522f979e05c0d8d8ca9fffb",
     "2b4be7f19ee27bbf30c667b642d5f4aa69fd169872f8fc3059c08ebae2eb19e7",
     "00a4b2856bfec510abab89753fac1ac0e1112364e7d250545963f135f2a33188ed"),
    ("000102030405060708090a0b0c0d0e0f", "m/0'",
     "8b59aa11380b624e81507a27fedda59fea6d0b779a778918a2fd3590e16e9c69",
     "68e0fe46dfb67e368c75379acec591dad19df3cde26e63b93a8e704f1dade7a3",
     "008c8a13df77a28f3445213a0f432fde644acaa215fc72dcdf300d5efaa85d350c"),
    ("000102030405060708090a0b0c0d0e0f", "m/0'/1'/2'/2'/1000000000'",
     "68789923a0cac2cd5a29172a475fe9e0fb14cd6adb5ad98a3fa70333e7afa230",
     "8f94d394a8e8fd6b1bc2f3f49f5c47e385281d5c17e65324b0f62483e37e8793",
     "003c24da049451555d51a7014a37337aa4e12d41e485abccfa46b47dfb2af54b7a"),
    ("fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542",
     "m/0'/2147483647'/1'/2147483646'/2'",
     "5d70af781f3a37b829f0d060924d5e960bdc02e85423494afc0b1a41bbe196d4",
     "551d333177df541ad876a60ea71f00447931c0a9da16f227c11ea080d7391b8d",
     "0047150c75db263559a70d5778bf36abbab30fb061ad69f69ece61a72b0cfa4fc0"),
]

# A wallet file written by the code before recovery words existed (scrypt + AES-GCM, password "old-password").
LEGACY_ENCRYPTED = {
    "version": 1, "address": "vibe1dca89531da5777dd6908243974b9e836155c",
    "public": "312d9be22c05e0e79c4cd28dcd369721845878b4f8ad4a669f7cad7ef5ac86ce",
    "crypto": {"kdf": "scrypt", "n": 32768, "r": 8, "p": 1, "salt": "24180fbf689426d19db46cd1f7fe59ac", "cipher": "aes-256-gcm",
               "nonce": "7ae53c9f8ed61b864d08ffa2",
               "ciphertext": "f53f38c7c235cc1445ee7bde6eb41091c00d6f22de70b0377ee5578879f28d76995e6182b0fe7d3e0fe739718f3b4051"},
}


class RecoveryWordsTests(unittest.TestCase):
    def test_official_word_list(self):
        words = mnemonic.wordlist()
        self.assertEqual(len(words), 2048)
        self.assertEqual(len(set(words)), 2048)
        self.assertEqual(list(words), sorted(words))
        self.assertEqual(len({w[:4] for w in words}), 2048)  # 4 letters identify a word
        self.assertEqual((words[0], words[-1]), ("abandon", "zoo"))

    def test_bip39_vectors(self):
        for entropy, words, seed in TREZOR:
            self.assertEqual(mnemonic.from_entropy(bytes.fromhex(entropy)), words)
            self.assertEqual(mnemonic.to_entropy(words).hex(), entropy)
            self.assertEqual(mnemonic.to_seed(words, "TREZOR").hex(), seed)

    def test_slip10_ed25519_vectors(self):
        for seed, path, chain, private, public in SLIP10:
            key, code = mnemonic.slip10_ed25519(bytes.fromhex(seed), path)
            self.assertEqual((key.hex(), code.hex()), (private, chain), path)
            self.assertEqual("00" + public_key(key).hex(), public, path)

    def test_derivation_path_is_pinned(self):
        """Changing the path or scheme would silently move everyone's coins to a different address."""
        self.assertEqual(mnemonic.PATH, "m/44'/1'/0'/0'/0'")
        self.assertEqual(mnemonic.keypair(ABANDON_12).address, "vibe12243ac3a24188b27e7fda3e382f960adf193")
        self.assertEqual(mnemonic.keypair(ABANDON_12, "TREZOR").address, "vibe1ce7a8faa0c917eef848f09c08e53e589c0a2")
        self.assertEqual(mnemonic.keypair(ABANDON_12, path="m/44'/1'/1'/0'/0'").address, "vibe1d8885978876512c8dce0d4ff333588492404")

    def test_typos_and_abbreviations(self):
        words = mnemonic.generate()
        self.assertEqual(len(words.split()), 24)
        self.assertEqual(mnemonic.check("  " + words.upper().replace(" ", "\n  ") + " "), words)
        self.assertEqual(mnemonic.check(" ".join(w[:4] for w in words.split())), words)  # what metal backups keep
        with self.assertRaisesRegex(mnemonic.MnemonicError, "word 3 isn't"):
            mnemonic.check("abandon abandon abandonn " + "abandon " * 8 + "about")
        with self.assertRaisesRegex(mnemonic.MnemonicError, "checksum"):
            mnemonic.check("abandon " * 12)
        swapped = words.split()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        if swapped[0] != swapped[1]:
            with self.assertRaises(mnemonic.MnemonicError):
                mnemonic.check(" ".join(swapped))
        with self.assertRaisesRegex(mnemonic.MnemonicError, "12, 15, 18, 21 or 24"):
            mnemonic.check("abandon " * 10 + "about")
        with self.assertRaises(mnemonic.MnemonicError):
            mnemonic.parse_path("m/44/1'")  # Ed25519 has no unhardened steps

    def test_old_wallet_files_load_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            plain = KeyPair.generate(b"legacy-plain")
            (Path(d) / "plain.json").write_text(json.dumps(
                {"version": 1, "address": plain.address, "public": plain.public.hex(), "secret": plain.secret.hex()}))
            self.assertEqual(wallets.load(Path(d) / "plain.json", None), plain)
            self.assertIsNone(wallets.hd_path(Path(d) / "plain.json"))
            if crypto.BACKEND == "openssl":
                (Path(d) / "enc.json").write_text(json.dumps(LEGACY_ENCRYPTED))
                self.assertEqual(wallets.load(Path(d) / "enc.json", "old-password"), KeyPair.generate(b"legacy-wallet"))

    def test_wallet_file_records_the_path_not_the_words(self):
        with tempfile.TemporaryDirectory() as d:
            words = mnemonic.generate()
            keys = mnemonic.keypair(words)
            wallets.save(keys, Path(d) / "w.json", None, mnemonic.PATH)
            text = (Path(d) / "w.json").read_text()
            self.assertNotIn(words.split()[0] + " " + words.split()[1], text)
            self.assertEqual(wallets.hd_path(Path(d) / "w.json"), mnemonic.PATH)
            self.assertEqual(wallets.load(Path(d) / "w.json", None), keys)


# --- transaction fees ------------------------------------------------------------------------------

CAROL, DAVE = KeyPair.generate(b"carol"), KeyPair.generate(b"dave")


class FeeTests(unittest.TestCase):
    def setUp(self):
        # Bob proposes; Alice, Carol and Dave spend.
        self.chain = Chain({ALICE.address: 100 * VIBE, CAROL.address: 5 * VIBE, DAVE.address: 5 * VIBE},
                           stakes={BOB.address: 10 * VIBE})

    def tx(self, keys, recipient, amount, fee=0, nonce=None, kind="transfer"):
        nonce = self.chain.next_nonce(keys.address) if nonce is None else nonce
        return Tx.signed(keys, kind, recipient, amount, nonce, "", self.chain.chain_id, fee)

    def seal(self, txs, signer=BOB):
        block, dropped = self.chain.build_block(None, txs, self.chain.tip.height + 1)
        block.sign(signer)
        self.chain.append(block)
        return block, dropped

    def test_fee_free_txs_and_blocks_keep_their_hashes(self):
        tx = Tx.signed(ALICE, "transfer", BOB.address, 40 * VIBE, 0, "rent", "vibehash-testnet-1")
        self.assertEqual(tx.payload().decode(), f'["vibehash-testnet-1","transfer","{ALICE.address}","{BOB.address}",4000000000,0,"rent"]')
        self.assertEqual(tx.txid.hex(), "f94b312a800eb9069de244f4f0f79a518213c71906054770d3ef9decc8820c92")
        self.assertNotIn("fee", tx.to_dict())
        self.assertEqual(Tx.from_dict(tx.to_dict()), tx)
        chain = Chain.from_genesis(COMPAT_GENESIS)
        block, _ = chain.build_block(None, [Tx.signed(ALICE, "transfer", BOB.address, 40 * VIBE, 0, "", "compat-net")], 1700000100)
        block.sign(ALICE)
        chain.append(block)
        self.assertEqual(block.hash.hex(), COMPAT_BLOCK1_HASH)

    def test_the_fee_is_signed(self):
        tx = self.tx(ALICE, CAROL.address, VIBE, fee=1234)
        self.assertTrue(tx.payload().endswith(b',"",1234]'))
        self.assertEqual(Tx.from_dict(tx.to_dict()).txid, tx.txid)
        for fee in (0, 1, 999_999):  # a relay can't strip, lower or raise it
            forged = Tx(tx.kind, tx.sender, tx.recipient, tx.amount, tx.nonce, tx.memo, tx.pubkey, tx.signature, tx.chain_id, fee)
            with self.assertRaisesRegex(InvalidBlock, "bad signature"):
                self.chain.check_tx(forged, [])

    def test_sender_pays_and_the_proposer_is_paid(self):
        self.seal([self.tx(ALICE, CAROL.address, 40 * VIBE, fee=700), self.tx(DAVE, CAROL.address, VIBE, fee=300)])
        st = self.chain.state
        self.assertEqual(st.balance(ALICE.address), 60 * VIBE - 700)
        self.assertEqual(st.balance(DAVE.address), 4 * VIBE - 300)
        self.assertEqual(st.balance(CAROL.address), 46 * VIBE)
        self.assertEqual(st.balance(BOB.address), 1000)
        self.assertEqual(st.total_supply(), st.expected_supply())
        self.assertTrue(self.chain.validate())

    def test_fee_must_be_affordable_and_not_negative(self):
        with self.assertRaisesRegex(InvalidBlock, "cannot spend"):
            self.chain.check_tx(self.tx(CAROL, ALICE.address, 5 * VIBE, fee=1), [])
        with self.assertRaisesRegex(InvalidBlock, "negative fee"):
            self.chain.check_tx(self.tx(CAROL, ALICE.address, VIBE, fee=-1), [])
        self.chain.check_tx(self.tx(CAROL, ALICE.address, 5 * VIBE - 1, fee=1), [])
        stake = self.tx(CAROL, SYSTEM, 5 * VIBE, fee=1, kind="stake")  # fees come out of the spendable balance
        with self.assertRaises(InvalidBlock):
            self.chain.check_tx(stake, [])

    def test_fees_arrive_after_the_block(self):
        """A proposer can't spend a fee in the block that pays it: every node credits it at the same point."""
        pay, spend = self.tx(ALICE, CAROL.address, VIBE, fee=1000), self.tx(BOB, CAROL.address, 500)
        block, _ = self.chain.build_block(None, [pay], self.chain.tip.height + 1)
        block.txs.append(spend)  # forced in: every node rejects the block
        block.sign(BOB)
        with self.assertRaisesRegex(InvalidBlock, "cannot spend"):
            self.chain.append(block)
        _, dropped = self.seal([pay, spend])  # and build_block leaves it out
        self.assertEqual(dropped, [spend])
        self.seal([spend])  # the next block can spend it
        self.assertEqual(self.chain.state.balance(BOB.address), 500)

    def test_round_blocks_pay_fees_to_their_proposer(self):
        proto, workers, req, job = network()
        st = proto.chain.state
        tx = Tx.signed(req, "transfer", CAROL.address, VIBE, proto.chain.next_nonce(req.address), "", proto.chain.chain_id, 5000)
        proto.submit(tx)
        before = {w.address: st.balance(w.address) for w in workers}
        report = proto.run_round()
        self.assertIn(tx.txid, [t.txid for t in report.block.txs])
        st = proto.chain.state
        rewards = {t.recipient: 0 for t in report.block.txs}
        for t in report.block.txs:
            if t.kind == "reward":
                rewards[t.recipient] += t.amount
        for w in workers:
            fee = 5000 if w.address == report.proposer else 0
            self.assertEqual(st.balance(w.address), before[w.address] + rewards.get(w.address, 0) + fee)
        self.assertEqual(st.total_supply(), st.expected_supply())
        self.assertTrue(proto.chain.validate())


class MempoolTests(unittest.TestCase):
    def setUp(self):
        people = [ALICE, BOB, CAROL, DAVE]
        self.chain = Chain({k.address: 10 * VIBE for k in people}, stakes={ALICE.address: 10 * VIBE})
        self.coord = Coordinator(self.chain, ALICE, CoordinatorConfig(min_fee=10, max_mempool=3))

    def tx(self, keys, fee, nonce=0, to=None, amount=1):
        return Tx.signed(keys, "transfer", (to or ALICE).address, amount, nonce, "", self.chain.chain_id, fee)

    def fees_in_pool(self):
        return [t.fee for t in self.coord.mempool]

    def test_minimum_relay_fee(self):
        with self.assertRaisesRegex(ProtocolError, "fee too low"):
            self.coord.submit_tx(self.tx(BOB, 9))
        self.coord.submit_tx(self.tx(BOB, 10))

    def test_ordered_by_fee_but_never_ahead_of_what_it_depends_on(self):
        self.coord.submit_tx(self.tx(BOB, 10))  # Bob's first...
        self.coord.submit_tx(self.tx(BOB, 90, nonce=1))  # ...and second, which can't jump the first
        self.coord.submit_tx(self.tx(CAROL, 50))
        self.assertEqual(self.fees_in_pool(), [50, 10, 90])
        self.coord.config.max_mempool = 10
        self.coord.submit_tx(self.tx(DAVE, 20, to=CAROL, amount=10 * VIBE - 20 - 60))  # Dave pays Carol...
        self.coord.submit_tx(self.tx(CAROL, 60, nonce=1, amount=10 * VIBE))  # ...who spends it, at a higher fee
        order = [(t.sender, t.fee) for t in self.coord.mempool]
        self.assertLess(order.index((DAVE.address, 20)), order.index((CAROL.address, 60)))
        block = self.coord.seal_pending(timestamp=1)
        self.assertEqual(len(block.txs), 5)  # the whole fee-ordered pool applies in order
        self.assertEqual(self.chain.state.balance(ALICE.address), 20 * VIBE + 3 + (10 + 90 + 50 + 20 + 60))  # payments + fees

    def test_full_pool_evicts_the_cheapest(self):
        for keys, fee in ((BOB, 20), (CAROL, 30), (DAVE, 40)):
            self.coord.submit_tx(self.tx(keys, fee))
        with self.assertRaisesRegex(ProtocolError, "mempool full: only transactions paying more than 20 syn"):
            self.coord.submit_tx(self.tx(BOB, 20, nonce=1))
        self.coord.submit_tx(self.tx(ALICE, 25))
        self.assertEqual(self.fees_in_pool(), [40, 30, 25])

    def test_eviction_keeps_what_others_depend_on(self):
        self.coord.submit_tx(self.tx(BOB, 10))
        self.coord.submit_tx(self.tx(BOB, 90, nonce=1))  # depends on the 10: that one isn't evictable
        self.coord.submit_tx(self.tx(CAROL, 30))
        self.assertEqual(fees.evictable(self.coord.mempool).fee, 30)
        self.coord.submit_tx(self.tx(DAVE, 31))
        self.assertEqual(sorted(self.fees_in_pool()), [10, 31, 90])
        with self.assertRaisesRegex(ProtocolError, "rejected"):  # pending txs can't be replaced...
            self.coord.submit_tx(self.tx(BOB, 95, nonce=1))
        self.assertEqual(sorted(self.fees_in_pool()), [10, 31, 90])  # ...and a refused tx evicts nothing


class FeeNetworkTests(Cli, unittest.TestCase):
    def test_node_minimum_fee_and_cli_defaults(self):
        genesis = {"chain_id": "fee-net", "timestamp": 0, "allocations": {CAROL.address: 50 * VIBE}, "stakes": {ALICE.address: 10 * VIBE}}
        node = start_node(self, genesis, config=CoordinatorConfig(min_fee=fees.DEFAULT_MIN_FEE))
        api = NodeClient(node.url)
        self.assertEqual(api.status()["min_fee"], fees.DEFAULT_MIN_FEE)
        with self.assertRaisesRegex(NodeError, "fee too low") as e:
            api.send_tx(CAROL, "transfer", DAVE.address, VIBE, fee=0)
        self.assertFalse(e.exception.retryable)
        wallets.save(CAROL, self.home / "carol.json", None)
        flags = ["--node", node.url, "--wallet", self.home / "carol.json"]
        self.assertIn("with a 0.0001 VIBE fee", self.run_cli("send", *flags, "--to", DAVE.address, "--amount", "2"))
        wait_for(lambda: node.chain.state.balance(DAVE.address) == 2 * VIBE)
        self.assertIn("with a 0.01 VIBE fee", self.run_cli("stake", *flags, "--amount", "5", "--fee", "0.01"))
        wait_for(lambda: node.chain.state.stake(CAROL.address) == 5 * VIBE)
        st = node.chain.state
        self.assertEqual(st.balance(CAROL.address), 43 * VIBE - fees.DEFAULT_MIN_FEE - VIBE // 100)
        self.assertEqual(st.balance(ALICE.address), fees.DEFAULT_MIN_FEE + VIBE // 100)  # the node proposed both blocks
        out = self.run_cli("balance", *flags)
        self.assertIn("balance:    42.9899 VIBE", out)
        self.assertIn("tx fee:     0.0001 VIBE per transaction", out)


# --- rate limits -----------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def raw_get(url: str, path: str = "/status") -> tuple[int, dict, dict]:
    """One request with no retries: (status, headers, JSON body)."""
    try:
        with urllib.request.urlopen(url + path, timeout=10) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, dict(e.headers), json.loads(e.read())


class RateLimitTests(unittest.TestCase):
    def test_every_route_has_a_bucket(self):
        cases = {("GET", "/status"): "read", ("GET", "/blob/ab?offset=8"): "read", ("POST", "/tx"): "tx",
                 ("POST", "/commit"): "worker", ("POST", "/verdict"): "worker", ("PUT", "/dataset/ab"): "upload",
                 ("POST", "/consensus/vote"): "other", ("POST", "/p2p/tx?from=x"): "other", ("GET", "/p2p/blocks"): "read"}
        for (method, path), bucket in cases.items():
            self.assertEqual(bucket_of(method, path), bucket, path)

    def test_token_buckets(self):
        clock = Clock()
        limiter = RateLimiter(Limits.parse("tx=2/3"), clock)
        self.assertEqual([limiter.check("10.0.0.1", "tx") for _ in range(3)], [0, 0, 0])  # the burst
        self.assertAlmostEqual(limiter.check("10.0.0.1", "tx"), 0.5)  # then 2 a second
        self.assertEqual(limiter.check("10.0.0.2", "tx"), 0)  # per address...
        self.assertEqual(limiter.check("10.0.0.1", "read"), 0)  # ...and per bucket
        clock.t += 0.25
        self.assertAlmostEqual(limiter.check("10.0.0.1", "tx"), 0.25)
        clock.t += 0.25
        self.assertEqual(limiter.check("10.0.0.1", "tx"), 0)
        clock.t += 100
        self.assertGreater([limiter.check("10.0.0.1", "tx") for _ in range(4)][-1], 0)  # refills to the burst, no more

    def test_this_computer_is_never_limited_unless_asked(self):
        strict = Limits.parse("read=0.001/1,worker=0.001/1,tx=0.001/1,upload=0.001/1,other=0.001/1")
        limiter = RateLimiter(strict)
        for ip in ("127.0.0.1", "127.0.0.9", "::1", "::ffff:127.0.0.1"):
            self.assertEqual(max(limiter.check(ip, b) for b in ratelimit.BUCKETS for _ in range(100)), 0, ip)
        self.assertEqual(limiter.check("192.168.1.20", "read"), 0)
        self.assertGreater(limiter.check("192.168.1.20", "read"), 0)
        local = RateLimiter(Limits.parse("read=0.001/1", local=True))
        local.check("127.0.0.1", "read")
        self.assertGreater(local.check("127.0.0.1", "read"), 0)
        off = RateLimiter(Limits.parse("off", local=True))
        self.assertEqual(max(off.check("192.168.1.20", "tx") for _ in range(1000)), 0)

    def test_parse(self):
        limits = Limits.parse("tx=1/10, read=100")
        self.assertEqual((limits.buckets["tx"], limits.buckets["read"]), ((1, 10), (100, 1000)))
        self.assertEqual(limits.buckets["upload"], ratelimit.BUCKETS["upload"])
        self.assertFalse(Limits.parse("off").enabled)
        for bad in ("nope=1", "tx=abc", "tx=0", "tx=1/0.5"):
            with self.assertRaises(ValueError, msg=bad):
                Limits.parse(bad)

    def test_idle_buckets_are_forgotten(self):
        clock = Clock()
        limiter = RateLimiter(Limits(), clock)
        with mock.patch.object(ratelimit, "MAX_TRACKED", 10):
            for i in range(10):
                limiter.check(f"10.0.0.{i}", "read")
            clock.t += 60
            limiter.check("10.0.1.1", "read")
        self.assertEqual(len(limiter._buckets), 1)


class RateLimitNetworkTests(unittest.TestCase):
    GENESIS = {"chain_id": "limits-net", "timestamp": 0, "allocations": {}, "stakes": {ALICE.address: 10 * VIBE}}

    def test_defaults_never_throttle_a_local_network(self):
        node = start_node(self, self.GENESIS)
        codes = []
        threads = [threading.Thread(target=lambda: codes.extend(raw_get(node.url)[0] for _ in range(150))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(codes, [200] * 1200)  # 8 clients on this computer, in a burst: none turned away

    def test_429_then_the_client_backs_off_and_retries(self):
        node = start_node(self, self.GENESIS, limits=Limits.parse("read=2/2", local=True))
        codes = [raw_get(node.url) for _ in range(3)]
        self.assertEqual([c for c, _, _ in codes], [200, 200, 429])
        _, headers, body = codes[2]
        self.assertIn("rate limited: too many read requests", body["error"])
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        start = time.time()
        for _ in range(4):
            NodeClient(node.url).status()  # each 429 is waited out and retried
        self.assertGreater(time.time() - start, 0.4)
        self.assertEqual(raw_get(node.url, "/account/" + ALICE.address)[0], 429)  # still over budget...
        with self.assertRaises(NodeError) as e:  # ...which a POST to another bucket doesn't share
            NodeClient(node.url).post("/tx", {})
        self.assertNotIn(e.exception.status, (429, 503))
        self.assertTrue(NodeError("x", 429).retryable)

    def test_slow_loris_is_cut_off(self):
        node = start_node(self, self.GENESIS, limits=Limits(header_timeout=0.5))
        host, port = node.server.server_address[:2]
        with socket.create_connection((host, port), timeout=10) as s:
            s.sendall(b"GET /status HTTP/1.0\r\nX-Slow: 1")  # and never finish the headers
            start = time.time()
            try:
                self.assertEqual(s.recv(100), b"")
            except ConnectionResetError:
                pass
            self.assertLess(time.time() - start, 5)

    def test_connection_caps_answer_503_and_the_client_waits(self):
        node = start_node(self, self.GENESIS, limits=Limits(local=True, max_connections_per_ip=2, header_timeout=1.5))
        host, port = node.server.server_address[:2]
        idle = [socket.create_connection((host, port), timeout=10) for _ in range(2)]  # hold both slots
        self.addCleanup(lambda: [s.close() for s in idle])
        wait_for(lambda: node.server._open_total == 2)
        code, headers, body = raw_get(node.url)
        self.assertEqual(code, 503)
        self.assertIn("too many connections from your address", body["error"])
        self.assertEqual(NodeClient(node.url).status()["chain_id"], "limits-net")  # once the idle ones time out
        node.server.limits.max_connections_per_ip = 100
        node.server.limits.max_connections = 1
        hold = socket.create_connection((host, port), timeout=10)
        self.addCleanup(hold.close)
        wait_for(lambda: node.server._open_total == 1)
        self.assertIn("connection limit", raw_get(node.url)[2]["error"])  # the global cap holds for everyone


# --- TLS -------------------------------------------------------------------------------------------

@unittest.skipUnless(tls.can_make_certs(), "making certificates needs the cryptography package")
class TlsTests(Cli, unittest.TestCase):
    GENESIS = {"chain_id": "tls-net", "timestamp": 0, "allocations": {CAROL.address: 50 * VIBE}, "stakes": {ALICE.address: 10 * VIBE}}

    def setUp(self):
        super().setUp()
        self.cert, self.key = self.home / "node-cert.pem", self.home / "node-key.pem"
        self.fp = tls.make_self_signed(self.cert, self.key, ["localhost", "127.0.0.1"])

    def https_node(self, **kw):
        return start_node(self, self.GENESIS, tls=tls.server_context(self.cert, self.key), **kw)

    def test_fingerprints(self):
        self.assertEqual(tls.cert_fingerprint(self.cert), self.fp)
        # A CA's fullchain.pem (Let's Encrypt's, say) holds the site's certificate and then its issuer's: the
        # fingerprint is the site's, and a node serves the whole chain.
        other_cert, other_key = self.home / "issuer.pem", self.home / "issuer-key.pem"
        tls.make_self_signed(other_cert, other_key, ["issuer.example"])
        chain = self.home / "fullchain.pem"
        chain.write_text(self.cert.read_text() + other_cert.read_text())
        self.assertEqual(tls.cert_fingerprint(chain), self.fp)
        tls.server_context(chain, self.key)
        with self.assertRaisesRegex(tls.TLSError, "isn't a PEM certificate"):
            tls.cert_fingerprint(self.key)
        colons = ":".join(self.fp[i:i + 2] for i in range(0, 64, 2)).upper()
        self.assertEqual(tls.normalize_fingerprint(colons), self.fp)
        self.assertEqual(tls.normalize_fingerprint(f"SHA256 Fingerprint={colons}"), self.fp)  # as openssl prints it
        with self.assertRaises(tls.TLSError):
            tls.normalize_fingerprint("abc123")
        with self.assertRaisesRegex(tls.TLSError, "already exists"):
            tls.make_self_signed(self.cert, self.key, ["localhost"])
        self.assertIn("subjectAltName=DNS:localhost,IP:127.0.0.1", tls.openssl_command("c.pem", "k.pem", ["localhost", "127.0.0.1"]))

    def test_https_node_trusted_by_pin_or_ca_and_nothing_else(self):
        node = self.https_node()
        self.assertTrue(node.url.startswith("https://127.0.0.1:"))
        self.assertEqual(NodeClient(node.url, fingerprint=self.fp).status()["chain_id"], "tls-net")
        self.assertEqual(NodeClient(node.url, ca_file=str(self.cert)).status()["chain_id"], "tls-net")
        self.assertEqual(NodeClient(node.url, ca_file=str(self.cert), fingerprint=self.fp).status()["chain_id"], "tls-net")
        with self.assertRaises(NodeError):  # self-signed: not trusted by the system's CAs
            NodeClient(node.url).status()
        with self.assertRaisesRegex(NodeError, "isn't the pinned one"):
            NodeClient(node.url, fingerprint="ab" * 32).status()
        with self.assertRaises(NodeError):  # plain http to an https node
            NodeClient(node.url.replace("https", "http"), timeout=5).status()
        # Transactions and chunked uploads work over it too.
        api = NodeClient(node.url, fingerprint=self.fp)
        api.send_tx(CAROL, "transfer", DAVE.address, VIBE)
        wait_for(lambda: node.chain.state.balance(DAVE.address) == VIBE)
        import numpy as np
        from vibehash.model import Dataset, ModelSpec
        data = Dataset.synthetic(ModelSpec(), 1, n_train=64, n_val=16)
        np.savez(self.home / "d.npz", x_train=data.x_train, y_train=data.y_train, x_val=data.x_val, y_val=data.y_val)
        sha = api.upload_dataset(CAROL, self.home / "d.npz")
        self.assertTrue(node.store.datasets.has(sha))

    def test_a_stalled_handshake_doesnt_block_others(self):
        node = start_node(self, self.GENESIS, tls=tls.server_context(self.cert, self.key), limits=Limits(header_timeout=30))
        host, port = node.server.server_address[:2]
        stalled = socket.create_connection((host, port), timeout=10)  # connects, never says hello
        self.addCleanup(stalled.close)
        start = time.time()
        self.assertEqual(NodeClient(node.url, fingerprint=self.fp).status()["chain_id"], "tls-net")
        self.assertLess(time.time() - start, 5)

    def test_rate_limits_apply_over_https(self):
        node = self.https_node(limits=Limits.parse("read=1/1", local=True))
        opener = tls.opener(pin=self.fp)
        opener.open(node.url + "/status", timeout=10).read()
        with self.assertRaises(urllib.error.HTTPError) as e:
            opener.open(node.url + "/status", timeout=10)
        self.assertEqual(e.exception.code, 429)
        e.exception.close()
        self.assertEqual(NodeClient(node.url, fingerprint=self.fp).status()["chain_id"], "tls-net")  # waits, retries

    def test_setup_saves_the_pin_for_every_later_command(self):
        node = self.https_node()
        address = node.url.split("://")[1]  # typed without a scheme: a fingerprint means https
        out = self.run_cli("setup", "--node", address, "--tls-fingerprint", self.fp.upper(), "--gpu", "skip")
        self.assertIn(f"connected to {node.url}", out)
        self.assertEqual(settings.tls_trust(node.url), (None, self.fp))
        self.assertIn("balance:    0 VIBE", self.run_cli("balance"))  # no flags: the saved node, trusted by its pin
        with mock.patch.dict(os.environ, {"VIBEHASH_TLS_FINGERPRINT": "cd" * 32}):  # the environment overrides it
            with self.assertRaises(AssertionError):
                self.run_cli("balance")

    def test_tls_commands(self):
        out = self.run_cli("tls", "cert", "--cert", self.home / "c2.pem", "--key", self.home / "k2.pem", "--host", "127.0.0.1",
                           "--host", "node.example")
        fp = re.search(r"sha256 fingerprint ([0-9a-f]{64})", out)[1]
        self.assertIn("valid 365 days for 127.0.0.1, node.example", out)
        self.assertEqual(self.run_cli("tls", "fingerprint", self.home / "c2.pem").strip(), fp)
        with mock.patch.object(tls, "can_make_certs", return_value=False):
            with self.assertRaises(SystemExit) as e:
                cli.main(["tls", "cert", "--cert", str(self.home / "c3.pem"), "--key", str(self.home / "k3.pem"), "--host", "127.0.0.1"])
        self.assertIn("openssl req -x509", str(e.exception.code))


# --- issuance schedule -----------------------------------------------------------------------------

def start_node(test, genesis, keys=ALICE, **kw):
    """A node on a free localhost port, stopped when the test ends."""
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(tmp.cleanup)
    node = Node(Path(tmp.name) / "node", genesis, keys, port=0,
                times=PhaseTimes(register_s=0.5, compute_s=1.0, reveal_s=0.5, verify_s=0.5, idle_s=0.2), **kw).start()
    test.addCleanup(node.stop)
    return node


def wait_for(condition, seconds=30):
    deadline = time.time() + seconds
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.05)


def released(report) -> int:
    """The job fee a round's block released from escrow into its pool."""
    return sum(t.amount for t in report.block.txs if t.kind == "fee_release")


class IssuanceTests(unittest.TestCase):
    def test_testnet_schedule_is_unchanged(self):
        for h in (0, 1, 209, 210, 211, 420, 210 * 33, 210 * 63, 210 * 64, 10**9):
            old = 0 if h // 210 >= 64 else (50 * VIBE) >> (h // 210)
            self.assertEqual(econ.TESTNET.subsidy(h), old)
            self.assertEqual(econ.block_subsidy(h), old)
        self.assertEqual((econ.TESTNET.max_supply, econ.TESTNET.name), (0, "testnet"))

    def test_mainnet_schedule(self):
        m = econ.MAINNET
        years = m.halving_interval * econ.MAINNET_BLOCK_S / (365.25 * 86400)
        self.assertAlmostEqual(years, 4, delta=0.05)
        self.assertEqual(m.subsidy(1), 3.125 * VIBE)
        self.assertEqual(m.subsidy(m.halving_interval), m.subsidy(1) // 2)
        self.assertEqual(m.max_supply, 21_000_000 * VIBE)
        self.assertLess(m.total_issuance(), m.max_supply)
        self.assertGreater(m.total_issuance(), 20_700_000 * VIBE)
        self.assertEqual(m.subsidy(1, created=m.max_supply - 7), 7)  # the cap binds exactly
        self.assertEqual(m.subsidy(1, created=m.max_supply), 0)

    def test_genesis_hash_commits_to_the_schedule(self):
        old = Chain.from_genesis(COMPAT_GENESIS)
        self.assertEqual(old.genesis_hash, COMPAT_GENESIS_HASH)  # genesis files without the field keep their hash
        self.assertEqual(old.state.economics, econ.TESTNET)
        for same in ("testnet", econ.TESTNET.to_dict()):
            self.assertEqual(Chain.from_genesis({**COMPAT_GENESIS, "economics": same}).genesis_hash, COMPAT_GENESIS_HASH)
        main = Chain.from_genesis({**COMPAT_GENESIS, "economics": "mainnet"})
        self.assertNotEqual(main.genesis_hash, COMPAT_GENESIS_HASH)
        self.assertEqual(main.state.economics, econ.MAINNET)
        self.assertTrue(main.validate())
        custom = dict(econ.MAINNET.to_dict(), max_supply=econ.MAINNET.max_supply + 1)
        self.assertEqual(Chain.from_genesis({**COMPAT_GENESIS, "economics": custom}).state.economics.name, "custom")
        self.assertNotEqual(Chain.from_genesis({**COMPAT_GENESIS, "economics": custom}).genesis_hash, main.genesis_hash)

    def test_a_node_refuses_a_data_dir_from_another_schedule(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = Store(d)
            store.open_chain(COMPAT_GENESIS)
            with self.assertRaisesRegex(ValueError, "different network"):
                store.open_chain({**COMPAT_GENESIS, "economics": "mainnet"})
            store.close()

    def test_rounds_pay_the_genesis_schedule(self):
        proto, workers, req, job = network(economics=econ.MAINNET)
        report = proto.run_round()
        self.assertEqual(report.pool, econ.MAINNET.subsidy(report.block.height) + released(report))
        st = proto.chain.state
        self.assertEqual(st.minted, 3.125 * VIBE)
        self.assertEqual(st.total_supply(), st.expected_supply())
        self.assertTrue(proto.chain.validate())

    def test_supply_cap(self):
        genesis_total = 1000 * VIBE + 3 * 10 * VIBE
        capped = econ.Economics(50 * VIBE, 210, max_supply=genesis_total + 120 * VIBE)
        proto, workers, req, job = network(economics=capped, budget=1000 * VIBE)
        pools = [r.pool - released(r) for r in (proto.run_round() for _ in range(4))]
        self.assertEqual(pools, [50 * VIBE, 50 * VIBE, 20 * VIBE, 0])
        st = proto.chain.state
        self.assertEqual(st.genesis + st.minted, capped.max_supply)
        self.assertTrue(proto.chain.validate())
        with self.assertRaisesRegex(InvalidBlock, "maximum supply"):
            Chain({ALICE.address: 10 * VIBE}, economics=econ.Economics(50 * VIBE, 210, max_supply=9 * VIBE))

    def test_node_reports_its_schedule(self):
        node = start_node(self, {**COMPAT_GENESIS, "economics": "mainnet"})
        st = NodeClient(node.url).status()
        self.assertEqual(st["economics"]["name"], "mainnet")
        self.assertEqual(st["subsidy"], 3.125 * VIBE)

    def test_bad_schedules_are_refused(self):
        for bad in ({"initial_subsidy": 1, "halving_interval": 0}, {"initial_subsidy": -1, "halving_interval": 5},
                    {"initial_subsidy": 1.5, "halving_interval": 5}, {"halving_interval": 5}, "moonshot", 7):
            with self.assertRaises(ValueError, msg=bad):
                Chain.from_genesis({**COMPAT_GENESIS, "economics": bad})


class IssuanceCommandLineTests(Cli, unittest.TestCase):
    def test_genesis_create_economics(self):
        out = self.run_cli("genesis", "create", "--chain-id", "net", "--out", self.home / "main.json", "--economics", "mainnet",
                           "--alloc", f"{ALICE.address}=100")
        self.assertIn("issuance (mainnet): 3.125 VIBE per round, halving every 3,320,000 blocks (~4 years", out)
        main = json.loads((self.home / "main.json").read_text())
        self.assertEqual(econ.Economics.from_genesis(main), econ.MAINNET)
        self.run_cli("genesis", "create", "--chain-id", "net", "--out", self.home / "test.json", "--alloc", f"{ALICE.address}=100")
        test = json.loads((self.home / "test.json").read_text())
        self.assertEqual(econ.Economics.from_genesis(test), econ.TESTNET)
        test["timestamp"] = main["timestamp"]
        self.assertNotEqual(Chain.from_genesis(main).genesis_hash, Chain.from_genesis(test).genesis_hash)
        del test["economics"]  # the testnet schedule is the one genesis files had before the field existed
        self.assertEqual(Chain.from_genesis(test).genesis_hash, Chain.from_genesis({**test, "economics": "testnet"}).genesis_hash)


class RecoveryWordsCommandLineTests(Cli, unittest.TestCase):
    def test_create_shows_words_once_and_restore_rebuilds_the_wallet(self):
        out = self.run_cli("wallet", "create", "--out", self.home / "a.json")
        address = re.search(r"address: (vibe1\w+)", out)[1]
        words = " ".join(re.findall(r"\d+\. ([a-z]+)", out))
        self.assertEqual(len(words.split()), 24)
        self.assertIn("won't be shown again", out)
        # Numbered down the columns: rebuild the order from the numbers.
        ordered = " ".join(w for _, w in sorted((int(n), w) for n, w in re.findall(r"(\d+)\. ([a-z]+)", out)))
        self.assertEqual(mnemonic.keypair(ordered).address, address)
        self.assertNotIn(ordered, (self.home / "a.json").read_text())
        self.assertIn(f"recovery words: yes (path {mnemonic.PATH})", self.run_cli("wallet", "show", "--wallet", self.home / "a.json"))

        with mock.patch.dict(os.environ, {"VIBEHASH_MNEMONIC": ordered}):
            out = self.run_cli("wallet", "restore", "--out", self.home / "b.json")
        self.assertIn(f"address: {address}", out)
        self.assertTrue(wallets.is_encrypted(self.home / "b.json"))
        self.assertEqual(wallets.load(self.home / "b.json", "test-only").address, address)
        with mock.patch.dict(os.environ, {"VIBEHASH_MNEMONIC": ordered, "VIBEHASH_MNEMONIC_PASSPHRASE": "extra"}):
            out = self.run_cli("wallet", "restore", "--passphrase", "--out", self.home / "c.json")
        self.assertNotIn(address, out)  # a passphrase is a different wallet
        self.assertEqual(re.search(r"address: (vibe1\w+)", out)[1], mnemonic.keypair(ordered, "extra").address)

    def test_old_wallet_shows_it_has_no_words(self):
        k = KeyPair.generate(b"legacy-plain")
        wallets.save(k, self.home / "old.json", None)
        self.assertIn("recovery words: no", self.run_cli("wallet", "show", "--wallet", self.home / "old.json"))


if __name__ == "__main__":
    unittest.main()
