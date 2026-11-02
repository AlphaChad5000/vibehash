"""Watch the training dashboard forge a model's emblem, on a made-up job: no node, nothing posted or spent.

Fine-tuning isn't live yet, so no real job names a base model for the forge to show. This plays one: a round lands
every six seconds, accuracy climbing, until the budget runs out, and the forge works the emblem of the model named
(vibehash/emblems.py). A model from a CSV file, with no base, keeps the blade.

    python scripts/preview_forge.py deepseek-v4.1-flash
    python scripts/preview_forge.py Qwen/Qwen3-8B

Ctrl+C stops it. It needs a terminal; `--plain` doesn't apply.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibehash import trainui  # noqa: E402
from vibehash.tasks import SYN_PER_VIBE  # noqa: E402

JOB, ROUND_S, ROUNDS, FEE, UNITS = "preview", 6.0, 12, 10 * SYN_PER_VIBE, 12
PHASES = (("register", 1.0), ("compute", 3.0), ("reveal", 1.0), ("verify", 1.0))  # one round, ROUND_S long


class PretendNode:
    """The two replies the dashboard reads, made up from the clock."""

    def __init__(self, base: str):
        self.base, self.start = base, time.time()

    def _rounds(self) -> int:
        return min(ROUNDS, int((time.time() - self.start) // ROUND_S))

    def get(self, path: str) -> dict:
        n = self._rounds()
        if path == "/status":
            into = (time.time() - self.start) % ROUND_S
            for phase, length in PHASES:
                if into < length or phase == "verify":
                    break
                into -= length
            return {"chain_id": "preview", "height": 100 + n, "phase": phase if n < ROUNDS else "idle",
                    "phase_ends": time.time() + max(0.0, length - into), "round": 101 + n,
                    "job_id": JOB if n < ROUNDS else None}
        rounds = [{"height": 101 + i, "accepted_units": UNITS - (i % 5 == 4), "val_loss": round(2.1 * 0.8 ** i + 0.4, 4),
                   "val_acc": round(0.3 + 0.6 * (1 - 0.75 ** (i + 1)), 4)} for i in range(n)]
        spent = sum(FEE * r["accepted_units"] // UNITS for r in rounds)
        return {"spec": {"id": JOB, "fee_per_round": FEE, "units_per_round": UNITS, "base": self.base,
                         "model": {"n_out": 0}}, "escrow": max(0, ROUNDS * FEE - spent) if n < ROUNDS else 0,
                "rounds": rounds}


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4.1-flash"
    if not trainui.supported():
        sys.exit("this draws the live dashboard, so it needs a terminal: run it in one, not through a pipe")
    watch = trainui.JobWatch(JOB, "http://preview", budget=ROUNDS * FEE, posted=True, api=PretendNode(base), base=base)
    trainui.TrainDashboard(watch).run()


if __name__ == "__main__":
    main()
