"""`vibehash genesis check`: a genesis file held against the mainnet checklist (vibehash/genesis_check.py)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vibehash import economics as econ
from vibehash import genesis_check as gc
from vibehash.crypto import KeyPair
from vibehash.tasks import SYN_PER_VIBE, TrainingJob

V = SYN_PER_VIBE


def addr(n: int) -> str:
    return KeyPair.generate(bytes([n]) * 32).address


def starter(job_id: str = "starter-mlp", **over) -> dict:
    return {**TrainingJob(job_id, "system", budget=0, fee_per_round=0).to_spec(), **over}


def mainnet(**over) -> dict:
    """A genesis that passes: four validators on four public HTTPS hosts with equal bootstrap stake, no premine,
    earn-in, Evo on, the mainnet schedule counting paid rounds, and a starter job."""
    vals = {addr(i): f"https://v{i}.example.org:8545" for i in range(1, 5)}
    g = {"chain_id": "vibehash-1", "timestamp": 1_800_000_000, "allocations": {}, "stakes": {},
         "validators": vals, "bootstrap": {a: 10 * V for a in vals}, "earn_in": True, "es": True,
         "economics": econ.NEW_SCHEDULES["mainnet"].to_dict(), "starter_jobs": [starter()]}
    g.update(over)
    return g


def by_title(findings) -> dict[str, list[gc.Finding]]:
    out: dict[str, list[gc.Finding]] = {}
    for f in findings:
        out.setdefault(f.title, []).append(f)
    return out


class GenesisCheckTests(unittest.TestCase):
    def status(self, g: dict, title: str) -> set[str]:
        return {f.status for f in by_title(gc.check(g)).get(title, [])}

    def test_a_good_mainnet_genesis_has_nothing_to_fix(self):
        findings = gc.check(mainnet())
        self.assertFalse(gc.failed(findings), [f for f in findings if f.status == "fail"])
        self.assertEqual(self.status(mainnet(), "validators"), {"ok"})
        self.assertEqual(self.status(mainnet(), "premine"), {"ok"})
        # Only the wallet path is left to decide while mnemonic.PATH has the testnet coin type.
        self.assertEqual({f.title for f in findings if f.status == "warn"}, {"recovery words"})

    def test_a_testnet_chain_id_fails(self):
        self.assertEqual(self.status(mainnet(chain_id="vibehash-public-1"), "chain id"), {"fail"})
        self.assertEqual(self.status(mainnet(chain_id="vibehash-local-abc123"), "chain id"), {"fail"})
        self.assertEqual(self.status(mainnet(chain_id="vibe-testing"), "chain id"), {"warn"})

    def test_the_old_issuance_rule_and_no_cap_fail(self):
        self.assertEqual(self.status(mainnet(economics="mainnet"), "issuance"), {"fail"})  # a preset: the old rule
        uncapped = econ.NEW_SCHEDULES["testnet"].to_dict()
        self.assertEqual(self.status(mainnet(economics=uncapped), "supply cap"), {"fail"})

    def test_a_premine_is_flagged_with_its_share_of_the_cap(self):
        g = mainnet(allocations={addr(9): 21_000 * V}, stakes={addr(8): 10 * V})
        [f] = by_title(gc.check(g))["premine"]
        self.assertEqual(f.status, "warn")
        self.assertIn("21,010 VIBE", f.detail)
        self.assertIn("0.1%", f.detail)
        self.assertIn("2 addresses", f.detail)

    def test_bootstrap_stake_is_not_a_premine(self):
        self.assertEqual(self.status(mainnet(), "premine"), {"ok"})

    def test_one_validator_or_one_with_a_third_of_the_vote_fails(self):
        one = {addr(1): "https://v1.example.org:8545"}
        self.assertEqual(self.status(mainnet(validators=one, bootstrap={addr(1): 10 * V}), "validators"), {"fail"})
        three = {addr(i): f"https://v{i}.example.org:8545" for i in range(1, 4)}
        g = mainnet(validators=three, bootstrap={a: 10 * V for a in three})
        self.assertEqual(self.status(g, "validators"), {"fail"})  # three equal: one offline halts it
        g = mainnet(bootstrap={addr(1): 40 * V, addr(2): 10 * V, addr(3): 10 * V, addr(4): 10 * V})
        self.assertEqual(self.status(g, "validators"), {"fail"})  # 4 of 7 parts: it alone can stop the chain
        g = mainnet(validators={}, bootstrap={})
        self.assertEqual(self.status(g, "validators"), {"fail"})

    def test_the_first_job(self):
        # No coins and no starter job: nobody can post a job, so nothing is ever mined.
        self.assertEqual(self.status(mainnet(starter_jobs=[]), "first job"), {"fail"})
        self.assertEqual(self.status(mainnet(starter_jobs=[], allocations={addr(9): V}), "first job"), {"info"})
        self.assertEqual(self.status(mainnet(), "first job"), {"ok"})
        on_a_file = starter(data={"kind": "npz", "sha256": "ab" * 32})
        self.assertEqual(self.status(mainnet(starter_jobs=[on_a_file]), "first job"), {"ok", "warn"})
        # The starter job's 1 syn of budget isn't a premine.
        self.assertEqual(self.status(mainnet(), "premine"), {"ok"})

    def test_tolerates(self):
        self.assertEqual(gc.tolerates([1, 1, 1, 1]), 1)
        self.assertEqual(gc.tolerates([1, 1, 1]), 0)
        self.assertEqual(gc.tolerates([1] * 7), 2)
        self.assertEqual(gc.tolerates([5, 1, 1, 1, 1]), 0)

    def test_validator_hosts(self):
        vals = {addr(1): "https://same.example.org:8545", addr(2): "https://same.example.org:9545",
                addr(3): "http://v3.example.org:8545", addr(4): "https://192.168.1.20:8545"}
        found = by_title(gc.check(mainnet(validators=vals, bootstrap={a: 10 * V for a in vals})))
        self.assertEqual([f.status for f in found["validator hosts"]], ["fail"])
        self.assertIn("same.example.org", found["validator hosts"][0].detail)
        statuses = sorted(f.status for f in found["validator address"])
        self.assertEqual(statuses, ["fail", "warn"])  # the private address fails, plain HTTP is a warning

    def test_evo_off_and_earn_in_off_are_warnings(self):
        g = mainnet()
        del g["es"]
        self.assertEqual(self.status(g, "Evo Hash"), {"warn"})
        self.assertEqual(self.status(mainnet(earn_in=False), "earn-in"), {"warn"})

    def test_a_genesis_the_ledger_refuses_is_one_failure(self):
        g = mainnet(allocations={addr(9): 21_000_000 * V})  # over the cap with the issuance on top
        findings = gc.check(g)
        self.assertEqual([(f.status, f.title) for f in findings], [("fail", "genesis")])


class CommandTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "vibehash", *args], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=120)

    def test_check_exits_1_on_a_failure_and_0_otherwise(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d, "good.json"), Path(d, "bad.json")
            good.write_text(json.dumps(mainnet()))
            bad.write_text(json.dumps(mainnet(chain_id="vibehash-public-1")))
            r = self.run_cli("genesis", "check", str(good))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("0 to fix, 1 to decide on purpose", r.stdout)
            r = self.run_cli("genesis", "check", str(bad), "--json")
            self.assertEqual(r.returncode, 1)
            self.assertIn({"status": "fail", "title": "chain id"},
                          [{k: f[k] for k in ("status", "title")} for f in json.loads(r.stdout)])

    def test_create_with_es_and_mainnet(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d, "g.json")
            vals = [f"--validator={addr(i)}=https://v{i}.example.org:8545" for i in range(1, 5)]
            boot = [f"--bootstrap={addr(i)}=10" for i in range(1, 5)]
            spec = Path(d, "starter.json")
            spec.write_text(json.dumps(starter(fee_per_round=5 * V)))  # a fee is set to 0, and said so
            r = self.run_cli("genesis", "create", "--chain-id", "vibehash-1", "--economics", "mainnet", "--es",
                             *vals, *boot, "--starter-job", str(spec), "--out", str(out))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("vibehash genesis check", r.stdout)
            self.assertIn("fee_per_round is set to 0", r.stdout)
            self.assertIn("starter job starter-mlp", r.stdout)
            g = json.loads(out.read_text())
            self.assertIs(g["es"], True)
            self.assertEqual([j["fee_per_round"] for j in g["starter_jobs"]], [0])
            self.assertFalse(gc.failed(gc.check(g)))


if __name__ == "__main__":
    unittest.main()
