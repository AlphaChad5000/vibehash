"""A DeepSeek job's whole life on one PC, over HTTP (the tiny config, on the CPU): a node holding a bucket's keys
(FakeS3, every link signed), and workers that said yes to DeepSeek work.

- LocalRunTests: the job is posted; prep rounds hand out the work graph's ready units, workers run them through the
  bucket and commit to their commitments, and two checkers replay each. One worker bends one op of its first unit
  (layer 1's MoE output, its state roots its own work's): a checker rejects it, and the unit ends unchecked and goes
  out again, nobody paid or slashed for it (the pilot's rule: prep units aren't refereed). Once every unit is in, the
  node pins the records' root, one machine's to the bit, and the job's rounds train from the records, op by op
  (deepseek_program.py): claims checked, paid, and scored on the held-back sequences. The same worker trains with one
  op's output scaled; a checker rejects its claim and the node referees the dispute down to that op
  (referee_deepseek.py) and slashes it. Another worker rejects an honest claim, finds no op the referee would refuse,
  and withdraws: unpaid, not slashed. An outside auditor replays every round from its record and the token file, and
  referees every game again from the node's blobs and the records in the bucket.
"""
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    import numpy as np
    from vibehash import deepseek, deepseek_job as dj, deepseek_prep as prep, deepseek_split as split
    from vibehash import deepseek_unit, deepseek_worker as dw
    from vibehash.client import NodeClient, NodeError, WorkerClient
    from vibehash.coordinator import CoordinatorConfig
    from vibehash.crypto import KeyPair
    from vibehash.ledger import Block, Chain
    from vibehash.node import LeaderBlobs, Node, PhaseTimes
    from vibehash.replay import replay_block
    from vibehash.tasks import SYN_PER_VIBE as VIBE, TrainingJob
    from tests.test_deepseek_split import FakeS3
except ImportError:
    torch = None

JOB = "think-shorter-tiny"


@unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
class LocalRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows keeps memory-mapped files locked
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        logging.disable(logging.INFO if not os.environ.get("VIBEHASH_TEST_LOG") else logging.NOTSET)
        self.addCleanup(logging.disable, logging.NOTSET)
        deepseek_unit.unload()
        self.addCleanup(deepseek_unit.unload)
        self.s3 = FakeS3()
        self.addCleanup(self.s3.close)

    def text(self, n=130):
        """n short sequences, about 2,500 tokens: three chunks of the prep, so its units hand off to each other, and
        each of three workers is handed one of the first pieces (the assignment balances their load)."""
        spec = dj.spec("tiny", seq_len=64, pieces=2, chunk_tokens=1024, row_ranges=2)
        g = np.random.default_rng(0)
        seqs = []
        for _ in range(n):
            k = int(g.integers(10, 30))
            seqs.append((g.integers(0, spec.cfg.vocab_size, k).tolist(), (g.random(k) < 0.5).astype(int).tolist()))
        path = self.tmp / "data.npz"
        prep.save_dataset(path, seqs)
        return spec, path

    def test_prep_then_training(self):
        spec, path = self.text()
        node_keys, req = KeyPair.generate(b"node"), KeyPair.generate(b"requester")
        workers = [KeyPair.generate(f"worker{i}".encode()) for i in range(3)]  # each claim's checkers: the other two
        genesis = {"chain_id": "vibehash-deepseek-test", "timestamp": 0, "allocations": {req.address: 500 * VIBE},
                   "stakes": {k.address: 10 * VIBE for k in [node_keys, *workers]}}
        node = Node(self.tmp / "node", genesis, node_keys, port=0,
                    times=PhaseTimes(register_s=1.0, compute_s=20.0, reveal_s=3.0, verify_s=20.0, idle_s=0.3),
                    config=CoordinatorConfig(canaries_per_round=1, newcomer_units=6), bucket=self.s3.signer()).start()
        self.addCleanup(node.stop)
        behaviors = [("honest", "honest"), ("honest", "false_accuser"), ("tamper", "honest")]
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=self.tmp / f"cache{i}", deepseek=True, behavior=b, verifier_behavior=v)
                   for i, (k, (b, v)) in enumerate(zip(workers, behaviors))]
        cheat, bent = clients[2], {}  # clients[1] rejects honest claims

        def prep_cheat(name):
            """The cheat's first prep unit with layer 1: that layer's MoE output scaled, in its run and so in its
            commitment (its state roots are its own work's), and in its dispute's re-runs."""
            if not bent and name.startswith("layers-00-"):
                bent.update(unit=name, round=cheat.ds.round)
            if bent.get("unit") != name or cheat.ds.round != bent["round"]:  # once: its redo is honest
                return None
            return lambda where, op: 1.25 if where[:1] == ("layer 1 after attention",) and op.kind == "ds_moe_sum"                 else None
        cheat.ds.prep_cheat = prep_cheat

        api = NodeClient(node.url)
        sha = api.upload_dataset(req, path)
        job = TrainingJob(job_id=JOB, requester=req.address, budget=200 * VIBE, fee_per_round=VIBE, spec=spec,
                          data_source={"kind": "deepseek", "sha256": sha, "train": [0, 104], "heldback": [104, 130]},
                          init_seed=spec.seed, lr=1e-3, batch_size=4, steps_per_unit=4, checkpoint_every=4,
                          units_per_round=8, memory_mb=8000, tolerance=1e-4, max_update_ratio=1.0)
        api.send_tx(req, "post_job", "system", 200 * VIBE, job.spec_json())

        stop = threading.Event()
        self.addCleanup(stop.set)
        threads = [threading.Thread(target=c.run, args=(stop,), daemon=True) for c in clients]
        for t in threads:
            t.start()

        def trained() -> bool:
            try:
                info = api.get(f"/job/{JOB}")
            except NodeError:  # not in a block yet
                return False
            return bool((info.get("prep") or {}).get("root")) and any(
                np.isfinite(r.get("val_loss", np.nan)) for r in info["rounds"]) and any(
                api.get(f"/round/{b.height}/record").get("games") for b in node.chain.blocks if b.summary)
        deadline = time.time() + 900
        while not trained() and time.time() < deadline:
            time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=60)
        info = api.get(f"/job/{JOB}")
        records = [api.get(f"/round/{b.height}/record") for b in node.chain.blocks if b.summary]
        preps = [r for r in records if r.get("prep")]
        self.assertTrue(trained(), (info.get("prep"), info["rounds"], [r.get("prep") for r in preps]))

        # The prep: the one-PC prep's records and root, to the bit.
        weights = dw.Weights(spec, JOB)
        tokens = dj.TokenFile.from_npz(path)
        chunks = tokens.chunks(spec.chunk_tokens)
        self.assertEqual(len(chunks), 3)
        one = split.run_local(prep.LocalSource(deepseek.random_weights(spec.cfg, 0)),
                              split.LocalStore(self.tmp / "one-pc"), prep.load_dataset(path), chunks, spec.top,
                              split.pieces(spec.cut, spec.pieces), weights.token_map(self.tmp), self.tmp / "one-pc-work",
                              ranges=spec.row_ranges)
        self.assertEqual(info["prep"], {"done": True, "root": one["root"]})
        manifest = json.loads(self.s3.objects[f"{JOB}/manifest.json"])
        self.assertEqual(manifest["root"], one["root"])
        graph = spec.graph(tokens)
        accepted = {}
        for r in preps:
            for name in r["prep"].get("accepted", {}):
                self.assertNotIn(name, accepted, "a unit is accepted once")
                accepted[name] = r["round"]
        self.assertEqual(sorted(accepted), sorted(graph.units))
        ours = next(j for j in node._network(node.chain.tip, time.time())["jobs"] if j["id"] == JOB)
        self.assertEqual(ours["prep"]["pieces_done"], ours["prep"]["pieces_total"])

        # The cheat: its bent prep unit was rejected and went out again, unrefereed (coordinator.PREP_GAMES).
        self.assertTrue(bent, "the cheat ran a layers unit")
        first = next(r for r in preps if r["round"] == bent["round"])
        self.assertIn(bent["unit"], first["prep"]["units"].values())
        self.assertNotIn(bent["unit"], first["prep"].get("accepted", {}))
        self.assertGreater(accepted[bent["unit"]], bent["round"])
        rejections = [v for v in first["verdicts"] if not v["body"]["ok"]]
        self.assertTrue(rejections, first["verdicts"])
        by_height = {b.height: api.get(f"/round/{b.height}/record") for b in node.chain.blocks if b.summary}
        self.assertEqual([g for r in preps for g in r.get("games", [])], [])
        prep_slashes = [s for b in node.chain.blocks if b.summary and by_height[b.height].get("prep")
                        for s in b.summary.slashes]
        self.assertEqual(prep_slashes, [])
        sealed = next(b.summary for b in node.chain.blocks if b.summary and b.summary.params_hash and
                      api.get(f"/round/{b.height}/record")["round"] == bent["round"])
        self.assertTrue(sealed.verifier_flops, "the checkers were paid")

        # Training: after the prep, rounds of claims checked, paid and scored on the held-back sequences.
        training = [r for r in records if not r.get("prep") and r.get("job_id") == JOB]
        self.assertTrue(training)
        self.assertTrue(all(r["version"] == 4 for r in records))
        self.assertTrue(any(np.isfinite(r["val_loss"]) for r in info["rounds"]))
        self.assertTrue(node.chain.state.balance(workers[0].address), "the honest worker was paid")
        self.assertLess(node.chain.state.escrow[JOB], 200 * VIBE)

        # The disputes: refereed op by op, the cheat's down to the op it scaled, and slashed. The false accuser wins its
        # games against the cheat and, finding no op the referee would refuse in the honest worker's, withdraws those
        # (unpaid, not slashed). Its four games a round may all go on the cheat's claims (coordinator.MAX_GAMES).
        games = [g for r in training for g in r.get("games", [])]
        lost = [g for g in games if g["defender"] == workers[2].address and g["outcome"] == "challenger"]
        self.assertTrue(lost, [(g["defender"][:12], g["outcome"], g["detail"]) for g in games])
        self.assertTrue(any(g["detail"].startswith("ds_attn_dkv") for g in lost), [g["detail"] for g in lost])
        accused = [g for g in games if g["challenger"] == workers[1].address]
        self.assertTrue(accused and not [g for g in accused if g["outcome"] == "defender" or
                                         g["outcome"] == "challenger" and g["defender"] != workers[2].address],
                        [(g["defender"][:12], g["outcome"], g["detail"]) for g in accused])
        slashes = [s for b in node.chain.blocks if b.summary and not by_height[b.height].get("prep")
                   for s in b.summary.slashes]
        self.assertTrue(any(s[0] == workers[2].address and s[2] == "failed_replay" and s[1] in
                            (workers[0].address, workers[1].address) for s in slashes), slashes)  # its catcher's bounty
        self.assertFalse([s for s in slashes if s[0] in (workers[0].address, workers[1].address)], slashes)

        # An outside auditor replays every round from its record, the prep's included, and referees every game again:
        # with the job's token file (a game's tree is built from it), the node's blobs (the games' states and
        # evidence) and the bucket (the records the referee reads), as a validator does.
        stats = {}
        audit = Chain.from_genesis(api.get("/genesis"))
        for d in api.get(f"/blocks?from=1&to={node.chain.tip.height}"):
            block = Block.from_dict(d)
            if block.summary is not None:
                self.assertEqual(replay_block(audit, block, api.get(f"/round/{block.height}/record"), stats=stats,
                                              datasets=node.store, roots=LeaderBlobs(node, block.summary.proposer),
                                              records=node.prep_store), [], f"block {block.height}")
            audit.append(block)
        self.assertEqual(audit.tip.hash, node.chain.tip.hash)
        self.assertFalse(stats.get("unchecked"), stats)


if __name__ == "__main__":
    unittest.main()
