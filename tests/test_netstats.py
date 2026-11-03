import json
import unittest

from vibehash import netstats


def block(height, t, **summary):
    return {"height": height, "timestamp": t, "summary": summary or None}


class NetstatsTests(unittest.TestCase):
    def test_miners_online_today_and_top(self):
        now = 100_000.0
        blocks = [
            block(1, now - 2 * netstats.DAY_S, worker_flops={"old": 99}),  # more than a day ago: not counted
            block(2, now - 3600, worker_flops={"a": 10}, verifier_flops={"b": 5}, accepted=["a"], accepted_units=2),
            block(3, now - 300, worker_flops={"a": 30}, missed=["c"], accepted_units=1, job_id="fruit"),
            block(4, now - 30),  # a round nobody worked
        ]
        m = netstats.summarize(blocks, now)
        self.assertEqual(m["online"], 2)  # a worked and c missed a deadline in the last 10 minutes
        self.assertEqual(m["online_by_job"], {"fruit": 2})
        self.assertEqual(m["day"], 3)
        self.assertEqual(m["units_day"], 3)
        self.assertEqual(m["flops_day"], 45)
        self.assertEqual([t["address"] for t in m["top_miners"]], ["a", "b"])
        self.assertEqual(m["top_miners"][0], {"address": "a", "flops": 40, "rounds": 2})
        self.assertAlmostEqual(m["block_time"], (now - 30 - (now - 2 * netstats.DAY_S)) / 3)

    def test_job_accuracy_from_its_start_through_the_rounds_that_worked(self):
        rounds = [{"height": 3, "accepted_units": 0, "val_loss": 1.2, "val_acc": 0.375}]
        rounds += [{"height": h, "accepted_units": 0, "val_loss": 1.2, "val_acc": 0.375} for h in range(4, 100)]
        rounds += [{"height": h, "accepted_units": 4, "val_loss": 1.2 - h / 1000, "val_acc": 0.375 + h / 1000} for h in range(100, 300)]
        spec = {"model": {"n_in": 5, "n_hidden": [128], "n_out": 4}, "fee_per_round": 10, "batch_size": 8, "steps_per_unit": 5}
        j = netstats.job_summary("fruit", spec, 500, rounds, "fruit", lambda h: 1000 + h)
        self.assertEqual((j["status"], j["model"], j["metric"]), ("training", "5 inputs to 4 classes", "accuracy"))
        self.assertEqual((j["rounds_trained"], j["units"], j["examples"]), (200, 800, 800 * 40))
        self.assertEqual(j["start"], 0.375)
        self.assertAlmostEqual(j["now"], 0.375 + 299 / 1000)
        self.assertEqual(j["last_trained"], {"height": 299, "time": 1299})
        self.assertEqual(len(j["history"]), netstats.HISTORY_POINTS)
        self.assertEqual(j["history"][0], [3, 0.375])  # the untrained start, then the thinned curve
        self.assertEqual(j["history"][-1][0], 299)

    def test_fine_tunes_report_loss_and_jobs_without_work(self):
        spec = {"base": "Qwen/Qwen3-4B", "lora": {"rank": 16}}
        nan = float("nan")
        # A fine-tune's round scores the model it started from, so a round's training shows in the next scored round:
        # test night's job trained one unit at 7277 (scored 5.13 before it) and 7280 scored 3.65. It said 5.13.
        rounds = [{"height": 7277, "accepted_units": 1, "val_loss": 5.13, "val_acc": nan},
                  {"height": 7280, "accepted_units": 0, "val_loss": 3.65, "val_acc": nan},
                  {"height": 7286, "accepted_units": 0, "val_loss": 3.65, "val_acc": nan}]
        j = netstats.job_summary("ft", spec, 0, rounds, None, lambda h: None)
        self.assertEqual((j["model"], j["metric"], j["status"]), ("fine-tune of Qwen/Qwen3-4B", "loss", "finished"))
        self.assertEqual((j["start"], j["now"], j["history"], j["rounds_trained"]),
                         (5.13, 3.65, [[7277, 5.13], [7280, 3.65]], 1))
        json.dumps(j, allow_nan=False)  # no NaN reaches the page
        # Rounds that worked one after another, a round nobody scored, and a last round whose result isn't scored yet.
        rounds = [{"height": 10, "accepted_units": 2, "val_loss": 3.0}, {"height": 11, "accepted_units": 2, "val_loss": 2.5},
                  {"height": 12, "accepted_units": 0, "val_loss": nan}, {"height": 13, "accepted_units": 4, "val_loss": 2.1},
                  {"height": 14, "accepted_units": 3, "val_loss": 1.9}]
        j = netstats.job_summary("ft", spec, 0, rounds, None, lambda h: None)
        self.assertEqual(j["history"], [[10, 3.0], [11, 2.5], [13, 2.1], [14, 1.9]])
        self.assertEqual((j["start"], j["now"], j["rounds_trained"]), (3.0, 1.9, 4))
        empty = netstats.job_summary("new", {"model": {}}, 10, [], None, lambda h: None)
        self.assertEqual((empty["status"], empty["history"], empty["now"], empty["last_trained"], empty["examples"]),
                         ("open", [], None, None, None))


if __name__ == "__main__":
    unittest.main()
