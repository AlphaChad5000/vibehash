"""What workers and job posters keep on disk (records.py): the event log and its cap, a round's JSON line from what
the worker did and what its block paid, a job's history from /job and /blocks, and the page beside a downloaded
model. Never in the real ~/.vibehash: every test has its own $VIBEHASH_HOME."""
import json
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from vibehash import records
from vibehash.client import NodeError
from vibehash.tasks import SYN_PER_VIBE as VIBE

ME, OTHER = "vibe1me", "vibe1other"
JOB = "fruit-3fa2"


class Home(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patch = mock.patch.dict(os.environ, {"VIBEHASH_HOME": str(self.home)})
        patch.start()
        self.addCleanup(patch.stop)


class CleanTests(unittest.TestCase):
    def test_control_and_format_characters_are_written_out(self):
        self.assertEqual(records.clean("plain text, ünïcode and all"), "plain text, ünïcode and all")
        self.assertEqual(records.clean("\x1b[2Jgone\x07"), "\\x1b[2Jgone\\x07")
        self.assertEqual(records.clean("two\nlines\r"), "two\\x0alines\\x0d")
        self.assertEqual(records.clean("evil\u202etxt.exe"), "evil\\u202etxt.exe")  # a bidi override
        self.assertEqual(records.clean("\x9b31m"), "\\x9b31m")  # the one-byte CSI


class LogFileTests(Home):
    def logger(self, handler):
        lg = logging.getLogger("vibehash.test-records")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg.addHandler(handler)
        self.addCleanup(lg.removeHandler, handler)
        self.addCleanup(handler.close)
        return lg

    def test_lines_are_timestamped_and_safe(self):
        handler = records.LogFile(self.home / "logs" / "worker-2026-11-02.log")
        lg = self.logger(handler)
        lg.warning("round %d: unit %s rejected: %s", 7, "ab12", "\x1b]0;pwned\x07bad\nsecond line")
        try:
            raise ValueError("a node said \x1b[31mno")
        except ValueError:
            lg.exception("unexpected error")
        handler.close()
        text = (self.home / "logs" / "worker-2026-11-02.log").read_text(encoding="utf-8")
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\x07", text)
        first, second, *rest = text.splitlines()
        self.assertRegex(first, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d  WARNING  test-records: round 7: unit ab12 rejected: "
                                r"\\x1b\]0;pwned\\x07bad$")
        self.assertEqual(second, "    second line")  # a message's later lines, indented under it
        self.assertIn("ERROR    test-records: unexpected error", rest[0])
        self.assertTrue(all(line.startswith("    ") for line in rest[1:]), rest)
        self.assertIn("ValueError: a node said \\x1b[31mno", text)

    def test_a_testnet_worker_is_named_by_its_thread(self):
        handler = records.LogFile(self.home / "w.log")
        lg = self.logger(handler)
        t = threading.Thread(target=lambda: lg.info("round 3: registered"), name="worker-2")
        t.start()
        t.join()
        handler.close()
        self.assertIn("INFO     worker-2: round 3: registered", (self.home / "w.log").read_text(encoding="utf-8"))

    def test_capped_and_turned(self):
        path = self.home / "worker-2026-11-02.log"
        handler = records.LogFile(path, cap=4000)
        lg = self.logger(handler)
        for i in range(400):
            lg.info("line %d of a long night %s", i, "x" * 40)
        handler.close()
        old = self.home / "worker-2026-11-02.1.log"
        self.assertTrue(old.exists())
        self.assertLessEqual(path.stat().st_size, 4000)
        self.assertLessEqual(old.stat().st_size, 4000)
        self.assertIn("line 399 of", path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), [old.name, path.name])  # no more than the two

    def test_a_file_that_cant_turn_stops_at_twice_the_cap(self):
        path = self.home / "w.log"
        handler = records.LogFile(path, cap=2000)
        lg = self.logger(handler)
        with mock.patch.object(records.os, "replace", side_effect=PermissionError("in use by another process")):
            for i in range(400):
                lg.info("line %d %s", i, "y" * 40)
        self.assertLessEqual(path.stat().st_size, 4000)
        self.assertFalse((self.home / "w.1.log").exists())

    def test_writing_never_raises(self):
        handler = records.LogFile(self.home / "no" / "such" / "w.log")
        lg = self.logger(handler)
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            lg.info("lost, quietly")
        self.assertEqual(handler.failed, "disk full")

    def test_old_days_are_pruned(self):
        logs = self.home / "logs"
        logs.mkdir()
        for day in range(1, 20):
            (logs / f"worker-2026-10-{day:02d}.log").write_text("x")
        (logs / "worker-2026-10-01.1.log").write_text("x")
        (logs / "rounds.jsonl").write_text("{}\n")
        records.prune(logs, keep=5)
        self.assertEqual(sorted(p.name for p in logs.iterdir()),
                         ["rounds.jsonl", *(f"worker-2026-10-{d}.log" for d in range(15, 20))])


def block(height=7, job=JOB, worker_flops=None, rewards=(), slashes=(), slash_txs=(), missed=(), fee=0):
    """A sealed block as /blocks serves it, with only the fields the records read and a few others."""
    txs = [{"kind": "slash", "sender": o, "recipient": r, "amount": a, "memo": why} for o, r, a, why in slash_txs]
    if fee:
        txs.append({"kind": "fee_release", "sender": "system", "recipient": "system", "amount": fee, "memo": job})
    txs += [{"kind": "transfer", "sender": OTHER, "recipient": ME, "amount": 5 * VIBE, "memo": ""}]  # not pay
    txs += [{"kind": "reward", "sender": "system", "recipient": who, "amount": a, "memo": memo} for who, a, memo in rewards]
    return {"height": height, "timestamp": 1_793_650_000 + height, "prev_hash": "00" * 32,
            "summary": {"job_id": job, "accepted_units": 3, "worker_flops": worker_flops or {}, "verifier_flops": {},
                        "slashes": [list(s) for s in slashes], "accepted": sorted(worker_flops or {}),
                        "missed": list(missed), "params_hash": "ab" * 32, "proposer": OTHER},
            "txs": txs, "chain_id": "vibehash-public-1", "version": 3}


class RoundLogTests(Home):
    def log(self):
        return records.RoundLog(self.home / "logs" / "rounds.jsonl", ME, "https://node.vibehash.org/", "NVIDIA RTX (CUDA)",
                                host="dorm-pc")

    def lines(self):
        return [json.loads(line) for line in (self.home / "logs" / "rounds.jsonl").read_text().splitlines()]

    def play(self, log, rnd=7):
        """A round as the worker loop notes it (client.WorkerClient.step): registered, given 4 units, 3 trained,
        uploaded and committed, one refused at the reveal, two checks sent (one rejecting)."""
        log.begin(rnd, attempt="a1", chain_id="vibehash-public-1", job_id=JOB, kind="mlp", speed=2.5e9,
                  leader="https://node.vibehash.org")
        log.note(rnd, assigned=4)
        for _ in range(3):
            log.add(rnd, trained=1, train_s=1.25, upload_s=0.5)
            log.add(rnd, committed=1)
            log.append(rnd, "_unit_flops", 1_000_000)
        log.add(rnd, revealed=3)
        log.append(rnd, "rejected", {"unit": "u3", "why": "sanity check failed: \x1b[31m"})
        log.add(rnd, checks=1)
        log.tally(rnd, "check_kinds", "check")
        log.add(rnd, checks=1, found_bad=1)
        log.tally(rnd, "check_kinds", "check")
        log.add(rnd, checks_given=2, check_s=0.75)

    def test_a_round_becomes_one_line(self):
        log = self.log()
        self.play(log)
        self.assertEqual(log.due(6), [])
        self.assertEqual(log.due(7), [7])
        b = block(worker_flops={ME: 2_000_000, OTHER: 3_000_000},
                  rewards=[(ME, 150_000_000, "work"), (ME, 25_000_000, "verify"), (OTHER, 10 ** 9, "work")],
                  slashes=[[OTHER, ME, "false accusation"]], slash_txs=[(OTHER, ME, 40_000_000, "false accusation")])
        log.sealed(7, b, {"attempt": "a1", "prep": False}, {"balance": 1_234_500_000, "stake": 500_000_000,
                                                            "reputation": 0.987654})
        [line] = self.lines()
        self.assertRegex(line.pop("time"), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}$")
        self.assertRegex(line.pop("registered"), r"^\d{4}-\d\d-\d\dT")
        self.assertEqual(line.pop("pid"), os.getpid())
        self.assertEqual(line, {
            "host": "dorm-pc", "worker": ME, "node": "https://node.vibehash.org", "chain_id": "vibehash-public-1",
            "round": 7, "attempt": "a1", "job": JOB, "kind": "mlp", "device": "NVIDIA RTX (CUDA)",
            "speed_flops_per_s": 2_500_000_000, "assigned": 4, "trained": 3, "committed": 3, "revealed": 3, "checks": 2,
            "train_s": 3.75, "upload_s": 1.5, "check_s": 0.75,
            "rejected": [{"unit": "u3", "why": "sanity check failed: \x1b[31m"}],  # (JSON escapes it: \u001b)
            "check_kinds": {"check": 2}, "found_bad": 1, "checks_given": 2,
            "sealed": True, "block_time": "2026-11-02T20:06:47Z", "accepted": 2, "accepted_flops": 2_000_000,
            "missed": False, "caught": ["false accusation"], "earned_vibe": 2.15,
            "paid": {"work": 1.5, "verify": 0.25}, "bounty_vibe": 0.4,
            "outcome": "2 of 3 units accepted, 1 refused at the reveal, 2 checks sent, earned 2.1500 VIBE",
            "balance_vibe": 12.345, "stake_vibe": 5.0, "reputation": 0.9877})
        self.assertNotIn(b"\x1b", (self.home / "logs" / "rounds.jsonl").read_bytes())
        self.assertEqual((log.rounds, log.units, log.earned), (1, 2, 215_000_000))

    def test_missed_and_slashed(self):
        log = self.log()
        self.play(log)
        b = block(worker_flops={OTHER: 3_000_000}, missed=[ME], slashes=[[ME, OTHER, "segment 2 mismatch"]],
                  slash_txs=[(ME, OTHER, 30_000_000, "segment 2 mismatch"), (ME, "burn", 20_000_000, "segment 2 mismatch")])
        e = log.sealed(7, b)
        self.assertEqual((e["accepted"], e["missed"], e["slashed"], e["lost_vibe"], e["earned_vibe"]),
                         (0, True, ["segment 2 mismatch"], 0.5, 0.0))
        self.assertIn("missed a deadline", e["outcome"])
        self.assertIn("slashed 0.50 VIBE: segment 2 mismatch", e["outcome"])
        self.assertEqual(log.earned, -50_000_000)

    def test_a_round_with_no_units(self):
        log = self.log()
        log.begin(9, job_id=JOB, chain_id="c")
        log.note(9, assigned=0, note="this fine-tune waits for 2 staked workers")
        e = log.sealed(9, block(9))
        self.assertEqual((e["accepted"], e["earned_vibe"], e["paid"]), (0, 0.0, {}))
        self.assertEqual(e["outcome"], "no units for this worker")
        self.assertEqual(e["note"], "this fine-tune waits for 2 staked workers")

    def test_a_deepseek_round_kind_comes_from_its_record(self):
        log = self.log()
        log.begin(11, job_id="think-shorter-pilot", kind="deepseek")
        self.assertEqual(log.sealed(11, block(11, job="think-shorter-pilot"), {"attempt": None, "prep": True})["kind"],
                         "deepseek-prep")
        log.begin(12, job_id="think-shorter-pilot", kind="deepseek")
        log.note(12, prep=[{"unit": "layers 0-4 chunk 3", "seconds": 412.5, "tokens_per_s": 318.0, "peak_gpu_gb": 9.8}])
        line = log.sealed(12, block(12, job="think-shorter-pilot"), {"attempt": None, "prep": False})
        self.assertEqual(line["kind"], "deepseek-training")
        self.assertEqual(line["prep"][0]["peak_gpu_gb"], 9.8)

    def test_a_block_that_cant_be_fetched_and_a_stop(self):
        log = self.log()
        self.play(log, 7)
        self.play(log, 8)
        for _ in range(records.SEAL_TRIES - 1):
            log.missed_block(7, "can't reach node")
            self.assertEqual(log.due(8), [8])  # asked again only after a wait
            log.tries[7] = (log.tries[7][0], 0.0)
        log.missed_block(7, "can't reach node")
        log.flush()
        first, second = self.lines()
        self.assertEqual((first["round"], first["sealed"], first["note"]), (7, False, "its block couldn't be fetched: can't reach node"))
        self.assertEqual((second["round"], second["sealed"], second["note"]), (8, False, "not sealed when the worker stopped"))
        self.assertIn("what counted isn't known", second["outcome"])
        self.assertEqual((log.rounds, log.unsealed), (0, 2))
        log.begin(9, job_id=JOB)  # a worker thread still finishing its step
        self.assertEqual(log.open, {})

    def test_odd_blocks_are_read_safely(self):
        for odd in ({}, {"summary": "x", "txs": "y"}, {"summary": {"worker_flops": {ME: "lots"}, "slashes": [1, [ME]]},
                                                       "txs": [None, {"kind": "reward", "recipient": ME, "amount": -5}]},
                    {"summary": {"missed": "everyone", "job_id": 5}, "timestamp": "noon"}):
            with self.subTest(block=odd):
                log = self.log()
                self.play(log)
                e = log.sealed(7, odd)
                self.assertTrue(e["sealed"])
                self.assertEqual((e["accepted"], e["earned_vibe"]), (0, 0.0))

    def test_rounds_file_is_capped(self):
        path = self.home / "r.jsonl"
        for i in range(200):
            records.append_line(path, {"round": i, "pad": "z" * 50}, cap=3000)
        self.assertLessEqual(path.stat().st_size, 3000)
        self.assertTrue((self.home / "r.1.jsonl").exists())
        self.assertEqual(json.loads(path.read_text().splitlines()[-1])["round"], 199)


class SessionTests(Home):
    def test_logs_and_summary(self):
        class Client:  # what Session needs of a WorkerClient
            def __init__(self, addr):
                self.keys = mock.Mock(address=addr)
                self.home = mock.Mock(url="http://127.0.0.1:8545")
                self.backend = mock.Mock(describe=lambda: "CPU (NumPy, 8 threads)")

        clients = [Client(ME), Client(OTHER)]
        session = records.Session(clients)
        self.assertEqual(session.dir, self.home / "logs")
        self.assertIn("rounds.jsonl (a line per round)", session.where())
        with mock.patch.object(logging.getLogger("vibehash.worker"), "propagate", False):  # (not on the test's screen)
            logging.getLogger("vibehash.worker").warning("round 5: unit u1 rejected: no")
        clients[0].record.begin(5, job_id=JOB)
        clients[0].record.note(5, assigned=2, committed=2, _unit_flops=[10])
        clients[0].record.sealed(5, block(5, worker_flops={ME: 20}, rewards=[(ME, 3 * VIBE, "work")]))
        clients[1].record.begin(5, job_id=JOB)
        text = session.close()
        self.assertIn("Recorded 1 sealed round of 2 workers: 2 units accepted, +3.0000 VIBE earned, as the blocks paid it "
                      "(1 more not sealed yet).", text)
        self.assertIn(str(self.home / "logs" / "rounds.jsonl"), text)
        self.assertEqual(len((self.home / "logs" / "rounds.jsonl").read_text().splitlines()), 2)
        log = next((self.home / "logs").glob("worker-*.log")).read_text(encoding="utf-8")
        self.assertIn("worker: round 5: unit u1 rejected: no", log)
        self.assertNotIn(session.log, logging.getLogger("vibehash.worker").handlers)  # detached


def job_reply(rounds, escrow, fee=10 * VIBE, units=12, base=None):
    spec = {"id": JOB, "fee_per_round": fee, "units_per_round": units, "precision": "float32",
            "model": {"n_in": 5, "n_hidden": [128, 128], "n_out": 4}, "data": {"kind": "npz", "sha256": "ab" * 32}}
    if base:
        spec = {"id": JOB, "fee_per_round": fee, "units_per_round": units, "base": base, "model": {"hidden": 1024},
                "lora": {"rank": 16, "alpha": 32, "targets": ["q_proj", "v_proj", "\x1b[2J"]},
                "data": {"kind": "tokens", "sha256": "cd" * 32}}
    return {"spec": spec, "owner": "vibe1poster", "escrow": escrow, "rounds": rounds,
            "model": {"height": rounds[-1]["height"] if rounds else 0, "bytes": 1, "sha256": "ef" * 32}}


ROUNDS = [{"height": 40, "accepted_units": 0, "val_loss": 1.3863, "val_acc": 0.25},
          {"height": 42, "accepted_units": 12, "val_loss": 0.9, "val_acc": 0.625},
          {"height": 45, "accepted_units": 6, "val_loss": 0.5, "val_acc": 0.875},
          {"height": 47, "accepted_units": 12, "val_loss": 0.25, "val_acc": 0.9375}]


class FakeNode:
    """/blocks from a list of blocks, at most `page` a reply (a node stops a long reply early)."""

    def __init__(self, blocks, page=100, job=None):
        self.blocks, self.page, self.job, self.calls = {b["height"]: b for b in blocks}, page, job, []

    def get(self, path):
        self.calls.append(path)
        if path.startswith("/job/"):
            if self.job is None:
                raise NodeError(f"GET {path}: no such job", 404)
            return self.job
        lo, hi = (int(x.split("=")[1]) for x in path.split("?")[1].split("&"))
        return [self.blocks[h] for h in range(lo, hi + 1) if h in self.blocks][:self.page]


def chain():
    """Blocks 40-47: the job's rounds at 40, 42, 45 and 47 paid 0, 10, 5 and 10 VIBE; the rest trained other jobs."""
    fees = {42: 10 * VIBE, 45: 5 * VIBE, 47: 10 * VIBE}
    return [block(h, job=JOB if h in (40, 42, 45, 47) else "other", fee=fees.get(h, 0)) for h in range(40, 48)]


EXPECTED_CSV = """round,time,units_accepted,loss,accuracy,paid_vibe,budget_left_vibe
40,2026-11-02T20:07:20Z,0,1.3863,0.25,0,100
42,2026-11-02T20:07:22Z,12,0.9,0.625,10,90
45,2026-11-02T20:07:25Z,6,0.5,0.875,5,85
47,2026-11-02T20:07:27Z,12,0.25,0.9375,10,75
"""


class HistoryTests(Home):
    def test_a_job_reply_and_its_blocks_make_the_csv(self):
        node = FakeNode(chain(), job=job_reply(ROUNDS, 75 * VIBE))
        info, rows = records.fetch_history(node, JOB)
        self.assertEqual(records.csv_text(rows), EXPECTED_CSV)
        self.assertEqual(node.calls, [f"/job/{JOB}", "/blocks?from=40&to=47"])  # one request covers them all
        self.assertEqual(records.json_rows(rows)[1], {"round": 42, "time": "2026-11-02T20:07:22Z", "units_accepted": 12,
                                                      "loss": 0.9, "accuracy": 0.625, "paid_vibe": 10.0,
                                                      "budget_left_vibe": 90.0})

    def test_short_replies_and_missing_blocks(self):
        node = FakeNode(chain()[:6], page=2)  # pages of two, and no blocks past 45
        got = records.fetch_blocks(node, JOB, [40, 42, 45, 47])
        self.assertLessEqual({40, 42, 45, 47}, set(got))  # (and the blocks between that came with them)
        self.assertIsNone(got[47])  # asked once, not again
        self.assertEqual(got[45], {"time": 1_793_650_045, "paid": 5 * VIBE})
        self.assertEqual(node.calls, ["/blocks?from=40&to=47", "/blocks?from=42&to=47", "/blocks?from=45&to=47",
                                      "/blocks?from=47&to=47"])

    def test_without_blocks_the_ledgers_fee_is_used(self):
        rows = records.history_rows(job_reply(ROUNDS, 75 * VIBE), {})
        self.assertEqual(records.csv_text(rows).splitlines()[1:],
                         ["40,,0,1.3863,0.25,0,100", "42,,12,0.9,0.625,10,90", "45,,6,0.5,0.875,5,85",
                          "47,,12,0.25,0.9375,10,75"])

    def test_a_long_lived_job_fetches_only_its_working_rounds(self):
        rounds = [{"height": h, "accepted_units": 4 if h in (2501, 2601) else 0, "val_loss": 1.0, "val_acc": 0.5}
                  for h in range(3, 7000, 2)]  # like fruit-demo on the public network: 3,500 rounds, a few that trained
        from vibehash.trainui import parse_job
        self.assertEqual(records.wanted(parse_job(job_reply(rounds, 0)).rounds, {}), [3, 2501, 2601, 6999])
        short = rounds[:50]
        self.assertEqual(len(records.wanted(parse_job(job_reply(short, 0)).rounds, {})), 50)  # a short one: every round

    def test_kept_up_to_date_while_watching(self):
        node = FakeNode(chain())
        folder = self.home / "jobs" / JOB
        history = records.JobHistory(JOB, folder, node)
        self.assertIsNone(history.update(job_reply(ROUNDS[:2], 90 * VIBE)))
        self.assertEqual(node.calls, ["/blocks?from=40&to=42"])
        self.assertEqual(history.update(job_reply(ROUNDS, 75 * VIBE)), None)
        self.assertEqual(node.calls[1:], ["/blocks?from=45&to=47"])  # only the new rounds' blocks
        self.assertEqual((folder / "history.csv").read_text(), EXPECTED_CSV)
        lines = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        self.assertEqual([(r["job"], r["round"]) for r in lines], [(JOB, 40), (JOB, 42), (JOB, 45), (JOB, 47)])
        before = (folder / "history.csv").stat().st_mtime_ns
        os.utime(folder / "history.csv", ns=(before - 10**9, before - 10**9))
        history.update(job_reply(ROUNDS, 75 * VIBE))
        self.assertEqual((folder / "history.csv").stat().st_mtime_ns, before - 10**9)  # nothing new: not rewritten
        self.assertEqual(len(node.calls), 2)

    def test_watching_with_a_node_that_wont_serve_blocks(self):
        class Down(FakeNode):
            def get(self, path):
                self.calls.append(path)
                raise NodeError("GET /blocks: busy", 429)

        node = Down([])
        history = records.JobHistory(JOB, self.home / "jobs" / JOB, node)
        self.assertIsNone(history.update(job_reply(ROUNDS, 75 * VIBE)))
        self.assertIsNone(history.update(job_reply(ROUNDS, 75 * VIBE)))
        self.assertEqual(len(node.calls), 1, "it waits before asking again")
        self.assertIn("47,,12,0.25,0.9375,10,75", (self.home / "jobs" / JOB / "history.csv").read_text())

    def test_a_file_open_elsewhere_is_said_not_raised(self):
        history = records.JobHistory(JOB, self.home / "jobs" / JOB, FakeNode(chain()))
        with mock.patch.object(records.os, "replace", side_effect=PermissionError("locked")):
            why = history.update(job_reply(ROUNDS, 75 * VIBE))
        self.assertIn("is open in another program", why)

    def test_a_job_id_that_isnt_one_gets_no_folder(self):
        self.assertEqual(records.job_folder(JOB), self.home / "jobs" / JOB)
        for bad in ("../../etc", "a/b", "", "Caps", "x" * 65, None):
            self.assertIsNone(records.job_folder(bad))


class ReportTests(Home):
    def test_a_model_trained_from_scratch(self):
        node = FakeNode(chain(), job=job_reply(ROUNDS, 75 * VIBE))
        info, rows = records.fetch_history(node, JOB)
        receipt = {"source": "C:\\Users\\me\\fruit.csv", "train_rows": 256, "val_rows": 64, "budget": 100 * VIBE}
        model = self.home / "fruit-model.npz"
        model.write_bytes(b"model")
        paths = records.write_report(model, JOB, info, rows, receipt, "http://127.0.0.1:8545", "vibehash-local-abc")
        self.assertEqual([p.name for p in paths], ["fruit-model-training-history.csv", "fruit-model-TRAINING.md"])
        self.assertEqual(paths[0].read_text(), EXPECTED_CSV)
        md = paths[1].read_text(encoding="utf-8")
        for fact in (f"# How {JOB} was trained", "on http://127.0.0.1:8545, chain `vibehash-local-abc`",
                     "a network trained from scratch: 5 → 128 → 128 → 4, float32",
                     "fruit.csv: 256 rows to train on, 64 held back", "4 rounds of this job were scored, from block 40 "
                     "(2026-11-02) to block 47 (2026-11-02); 3 of them trained it, with 30 units of work accepted",
                     "25.0% → 93.8% (best 93.8%); loss 1.3863 → 0.2500", "posted with a budget of 100 VIBE, 25 VIBE paid "
                     "for the work in these rounds, 75 VIBE left", "the model as of block 47",
                     "`fruit-model-training-history.csv`", "and the receipt `train` kept on this computer"):
            self.assertIn(fact, md)

    def test_a_fine_tune_from_another_computer(self):
        base = "hf:Qwen/Qwen3-0.6B@" + "c1899de289a0" * 3 + "abcd"
        info = job_reply(ROUNDS, 75 * VIBE, base=base)
        rows = records.history_rows(info, {})
        folder = self.home / "chats-lora"
        folder.mkdir()
        paths = records.write_report(folder, JOB, info, rows, None, "https://node.vibehash.org", "vibehash-public-1\x1b[2J")
        self.assertEqual([p.name for p in paths], ["training-history.csv", "TRAINING.md"])
        md = paths[1].read_text(encoding="utf-8")
        self.assertNotIn("\x1b", md)
        self.assertIn(f"a LoRA adapter (fine-tune; rank 16, alpha 32, on q_proj, v_proj) for `{base}`", md)
        self.assertIn("posted from another computer", md)
        self.assertIn("Held-back loss**: 1.3863 → 0.2500; reply tokens predicted right: 25.0% → 93.8%", md)
        self.assertIn("the model each round starts from", md)
        self.assertNotIn("receipt", md)

    def test_a_base_with_escape_sequences_isnt_shown(self):
        info = job_reply(ROUNDS, 0, base="hf:evil/\x1b]0;x\x07@main")
        md = records.report(JOB, info, records.history_rows(info, {}), None, "n", "c", "h.csv")
        self.assertNotIn("\x1b", md)
        self.assertIn("a base model whose name can't be shown", md)


if __name__ == "__main__":
    unittest.main()
