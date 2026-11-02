"""A DeepSeek job's whole life on one PC, over HTTP (the tiny config, on the CPU): a node holding a bucket's keys
(FakeS3, every link signed), and workers that said yes to DeepSeek work.

- LocalRunTests: the job is posted; prep rounds hand out the work graph's ready units, workers run them through the
  bucket and commit to their commitments, and two checkers replay each. One worker bends its first unit's output (and
  its commitment to match): a checker rejects it, it ends unchecked (unpaid, nobody slashed: deepseek_rounds' interim
  rule, with no DeepSeek program in the referee yet), and the unit goes out again. Once every unit is in, the node
  pins the records' root, the one-PC prep's to the bit, and the job's rounds train from the records: claims checked,
  paid, and scored on the held-back sequences. An outside auditor replays every round from its record alone.
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
    from vibehash.node import Node, PhaseTimes
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
        clients = [WorkerClient(node.url, k, memory_mb=8192, flops_override=1e12, poll_s=0.1, device="cpu",
                                cache_dir=self.tmp / f"cache{i}", deepseek=True) for i, k in enumerate(workers)]
        cheat, bent = clients[-1], {}

        def bend(store, unit, commit):
            """The cheat's first layers unit: an output it hands on, scaled up, and its commitment made to match."""
            if bent or unit.kind != "layers" or unit.b == spec.cut:
                return commit
            name = split.stream_name(unit.chunk, unit.b)
            got = split.decode(store.get(name))
            other = {k: (v.float() * 1.2).to(v.dtype) if k.endswith(".x") else v for k, v in got.items()}
            store.put(name, split.encode(other))
            commit = json.loads(json.dumps(commit))
            commit["outputs"][name] = split.digest(other)
            store.put(split.unit_commit(unit), json.dumps(commit).encode())
            bent.update(unit=unit.name, round=cheat.ds.round)
            return commit
        cheat.ds.bend = bend

        api = NodeClient(node.url)
        sha = api.upload_dataset(req, path)
        job = TrainingJob(job_id=JOB, requester=req.address, budget=200 * VIBE, fee_per_round=1, spec=spec,
                          data_source={"kind": "deepseek", "sha256": sha, "train": [0, 104], "heldback": [104, 130]},
                          init_seed=spec.seed, lr=1e-3, batch_size=4, steps_per_unit=4, checkpoint_every=2,
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
                np.isfinite(r.get("val_loss", np.nan)) for r in info["rounds"])
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

        # The cheat: its bent unit was rejected, unpaid and unslashed, and went out again.
        self.assertTrue(bent, "the cheat ran a layers unit")
        first = next(r for r in preps if r["round"] == bent["round"])
        self.assertIn(bent["unit"], first["prep"]["units"].values())
        self.assertNotIn(bent["unit"], first["prep"].get("accepted", {}))
        self.assertGreater(accepted[bent["unit"]], bent["round"])
        rejections = [v for v in first["verdicts"] if not v["body"]["ok"]]
        self.assertTrue(rejections, first["verdicts"])
        slashes = [s for b in node.chain.blocks if b.summary for s in b.summary.slashes]
        self.assertEqual(slashes, [])
        sealed = next(b.summary for b in node.chain.blocks if b.summary and b.summary.params_hash and
                      api.get(f"/round/{b.height}/record")["round"] == bent["round"])
        self.assertTrue(sealed.verifier_flops, "the checkers were paid")

        # Training: after the prep, rounds of claims checked, paid and scored on the held-back sequences.
        training = [r for r in records if not r.get("prep") and r.get("job_id") == JOB]
        self.assertTrue(training)
        self.assertTrue(all(r["version"] == 4 for r in records))
        self.assertTrue(any(np.isfinite(r["val_loss"]) for r in info["rounds"]))
        self.assertTrue(any(node.chain.state.balance(w.address) for w in workers), "workers were paid")
        self.assertLess(node.chain.state.escrow[JOB], 200 * VIBE)

        # An outside auditor replays every round from its record, the prep's included, without the text or the bucket.
        audit = Chain.from_genesis(api.get("/genesis"))
        for d in api.get(f"/blocks?from=1&to={node.chain.tip.height}"):
            block = Block.from_dict(d)
            if block.summary is not None:
                self.assertEqual(replay_block(audit, block, api.get(f"/round/{block.height}/record")), [],
                                 f"block {block.height}")
            audit.append(block)
        self.assertEqual(audit.tip.hash, node.chain.tip.hash)


if __name__ == "__main__":
    unittest.main()
