"""Jobs, work units, and the messages workers exchange with the protocol."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .crypto import h256
from .deepseek_job import DeepSeekData, DeepSeekSpec, TokenFile, is_deepseek_spec
from .finetune import LoraSpec, TokenDataset
from .model import PRECISIONS, Dataset, ModelSpec, evaluate

SYN_PER_VIBE = 100_000_000
MAX_SPEC_BYTES = 4096
_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MAX_CHECKPOINT_BYTES = 2**30  # 1 GiB of weights (~268M float32); sent as chunked binary blobs
# What a job's rounds are sized for, so a modest computer on a home connection can always finish a unit in time:
# an old laptop's CPU running the reference model (about 2 GFLOP/s), and 20 Mbit/s up, 100 Mbit/s down.
REF_FLOPS = 2e9
# Fine-tunes run on GPUs (finetune.py), so their reference is a gaming card: what this code sustains on a mid-range one,
# in float32 with TF32 off (an RTX 4070 SUPER sustains 9-15 TFLOP/s of a job's nominal FLOPs; an RTX 3060 about a third).
REF_GPU_FLOPS = 5e12
REF_UPLOAD = 2.5e6  # bytes/s
REF_DOWNLOAD = 12.5e6  # bytes/s
CHECKED_SEGMENTS = 3  # segments of a unit the beacon challenges (CoordinatorConfig.challenge_segments)
MAX_COMPUTE_S = 3600.0


class InvalidJob(ValueError):
    pass


@dataclass
class TrainingJob:
    job_id: str
    requester: str
    budget: int  # syn escrowed by the requester
    fee_per_round: int  # syn released to the reward pool per fully-verified round
    spec: ModelSpec | LoraSpec = field(default_factory=ModelSpec)  # a LoraSpec makes it a fine-tune (finetune.py)
    data_source: dict = field(default_factory=lambda: {"kind": "synthetic", "seed": 7})
    init_seed: int = 11
    lr: float = 0.1
    batch_size: int = 32
    steps_per_unit: int = 40
    checkpoint_every: int = 5
    units_per_round: int = 16
    memory_mb: int = 1024  # minimum RAM/VRAM a rig needs to take or verify a unit
    tolerance: float = 1e-6  # relative distance allowed between replay and claim
    max_update_ratio: float = 0.5  # |final - start| / |start| must stay below this (a fine-tune's: LoraSpec.sane)
    compute_s: float = 0.0  # a compute phase the job asks for, in seconds; 0: sized from the job (phase_lengths)
    shards: int = 1  # blocks of layers a unit is split into, one worker each (shards.py, SHARDING.md); 1: whole units
    _data: Dataset | TokenDataset | None = field(default=None, repr=False)

    def __post_init__(self):
        self.validate()

    @property
    def finetune(self) -> bool:
        """Whether it fine-tunes a pretrained model's LoRA adapters (finetune.py) rather than training an MLP. A
        DeepSeek top-layers job (deepseek_job.py) is one too: its training rounds are a fine-tune's."""
        return isinstance(self.spec, LoraSpec)

    @property
    def deepseek(self) -> bool:
        """Whether it's a DeepSeek-V4.1 top-layers job: a prep of the frozen layers, then training rounds."""
        return isinstance(self.spec, DeepSeekSpec)

    # --- on-chain spec -------------------------------------------------------

    def to_spec(self) -> dict:
        if self.finetune:  # a shape of its own, so MLP specs stay exactly as they were
            spec = {"id": self.job_id, "fee_per_round": self.fee_per_round, **self.spec.to_json(),
                    "data": self.data_source, "init_seed": self.init_seed, "lr": self.lr, "batch_size": self.batch_size,
                    "steps_per_unit": self.steps_per_unit, "checkpoint_every": self.checkpoint_every,
                    "units_per_round": self.units_per_round, "memory_mb": self.memory_mb, "tolerance": self.tolerance,
                    "max_update_ratio": self.max_update_ratio}
            if self.compute_s:
                spec["compute_s"] = self.compute_s
            return spec
        spec = {
            "id": self.job_id, "fee_per_round": self.fee_per_round,
            "model": {"n_in": self.spec.n_in, "n_out": self.spec.n_out,
                      "n_hidden": self.spec.n_hidden if isinstance(self.spec.n_hidden, int) else list(self.spec.n_hidden)},
            "precision": self.spec.precision,
            "data": self.data_source, "init_seed": self.init_seed, "lr": self.lr,
            "batch_size": self.batch_size, "steps_per_unit": self.steps_per_unit,
            "checkpoint_every": self.checkpoint_every, "units_per_round": self.units_per_round,
            "memory_mb": self.memory_mb, "tolerance": self.tolerance, "max_update_ratio": self.max_update_ratio,
        }
        if self.compute_s:  # left out otherwise, so specs posted before it existed are unchanged
            spec["compute_s"] = self.compute_s
        if self.shards > 1:  # likewise
            spec["shards"] = self.shards
        return spec

    def spec_json(self) -> str:
        return json.dumps(self.to_spec(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_spec_json(cls, text: str, requester: str, budget: int) -> "TrainingJob":
        if len(text.encode()) > MAX_SPEC_BYTES:
            raise InvalidJob("job spec too large")
        try:
            d = json.loads(text)
            if is_deepseek_spec(d):
                return cls._from_finetune(d, requester, budget, deepseek=True)
            if is_finetune_spec(d):
                return cls._from_finetune(d, requester, budget)
            m = d["model"]
            hidden = m["n_hidden"]
            hidden = int(hidden) if not isinstance(hidden, list) else tuple(int(h) for h in hidden)
            return cls(
                job_id=d["id"], requester=requester, budget=budget, fee_per_round=int(d["fee_per_round"]),
                spec=ModelSpec(int(m["n_in"]), hidden, int(m["n_out"]), str(d.get("precision", "float64"))),
                data_source=dict(d["data"]), init_seed=int(d["init_seed"]), lr=float(d["lr"]),
                batch_size=int(d["batch_size"]), steps_per_unit=int(d["steps_per_unit"]),
                checkpoint_every=int(d["checkpoint_every"]), units_per_round=int(d["units_per_round"]),
                memory_mb=int(d["memory_mb"]), tolerance=float(d["tolerance"]),
                max_update_ratio=float(d["max_update_ratio"]), compute_s=float(d.get("compute_s", 0)),
                shards=int(d.get("shards", 1)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            raise InvalidJob(f"malformed job spec: {e}") from e

    @classmethod
    def _from_finetune(cls, d: dict, requester: str, budget: int, deepseek: bool = False) -> "TrainingJob":
        keys = FINETUNE_KEYS | ({"deepseek"} if deepseek else set())
        if set(d) - keys:
            raise InvalidJob(f"a fine-tune's spec has no field {sorted(set(d) - keys)[0]!r}")
        return cls(
            job_id=d["id"], requester=requester, budget=budget, fee_per_round=int(d["fee_per_round"]),
            spec=(DeepSeekSpec if deepseek else LoraSpec).from_json(d, seed=int(d["init_seed"])),
            data_source=dict(d["data"]),
            init_seed=int(d["init_seed"]), lr=float(d["lr"]), batch_size=int(d["batch_size"]),
            steps_per_unit=int(d["steps_per_unit"]), checkpoint_every=int(d["checkpoint_every"]),
            units_per_round=int(d["units_per_round"]), memory_mb=int(d["memory_mb"]), tolerance=float(d["tolerance"]),
            max_update_ratio=float(d["max_update_ratio"]), compute_s=float(d.get("compute_s", 0)),
        )

    def validate(self) -> None:
        def need(cond, msg):
            if not cond:
                raise InvalidJob(msg)

        need(isinstance(self.job_id, str) and _JOB_ID.match(self.job_id), "job id must be 1-64 chars of a-z 0-9 . _ -")
        if self.finetune:
            return self._validate_finetune(need)
        need(self.spec.precision in PRECISIONS, "precision must be 'float64' or 'float32'")
        hidden = self.spec.hidden
        need(1 <= len(hidden) <= 8 and all(isinstance(h, int) and 1 <= h <= 8192 for h in hidden), "1-8 hidden layers of width 1-8192")
        need(1 <= self.spec.n_in <= 4096 and 2 <= self.spec.n_out <= 1000, "model size out of range")
        need(self.spec.n_params * np.dtype(self.spec.dtype).itemsize <= MAX_CHECKPOINT_BYTES,
             f"model too large: at most {MAX_CHECKPOINT_BYTES // 2**30} GB of weights per checkpoint")
        need(0 < self.lr <= 10 and 1 <= self.batch_size <= 4096, "lr / batch size out of range")
        need(1 <= self.checkpoint_every <= self.steps_per_unit <= 100_000, "steps out of range")
        need(self.steps_per_unit % self.checkpoint_every == 0, "checkpoint_every must divide steps_per_unit")
        need(1 <= self.units_per_round <= 10_000 and 1 <= self.memory_mb <= 1_000_000, "units / memory out of range")
        need(0 < self.tolerance <= 1e-2 and 0 < self.max_update_ratio <= 10, "tolerance out of range")
        need(self.fee_per_round >= 0 and self.budget >= 0, "negative fee or budget")
        need(self.compute_s == 0 or 5 <= self.compute_s <= MAX_COMPUTE_S,
             f"compute_s is 0 (sized from the job) or 5-{MAX_COMPUTE_S:.0f} seconds")
        n_layers = len(self.spec.widths) - 1
        need(isinstance(self.shards, int) and 1 <= self.shards <= n_layers,
             f"shards is 1 (whole units) to the model's {n_layers} layers: a shard is a block of whole layers")
        kind = self.data_source.get("kind")
        if kind == "synthetic":
            need(isinstance(self.data_source.get("seed"), int), "synthetic data needs an integer seed")
        elif kind == "npz":
            need(re.fullmatch(r"[0-9a-f]{64}", str(self.data_source.get("sha256", ""))), "npz data needs a sha256")
        else:
            need(False, "data kind must be 'synthetic' or 'npz'")

    def _validate_finetune(self, need) -> None:
        problems = self.spec.problems()
        need(not problems, "; ".join(problems))
        need(self.spec.seed == self.init_seed, "a fine-tune's LoRA seed is its init_seed")
        need(self.spec.n_params * 4 <= MAX_CHECKPOINT_BYTES,
             f"adapters too large: at most {MAX_CHECKPOINT_BYTES // 2**30} GB of them and their optimiser state")
        need(0 < self.lr <= 1 and 1 <= self.batch_size <= 1024, "lr / batch size out of range")
        need(1 <= self.checkpoint_every <= self.steps_per_unit <= 10_000, "steps out of range")
        need(self.steps_per_unit % self.checkpoint_every == 0, "checkpoint_every must divide steps_per_unit")
        need(1 <= self.units_per_round <= 10_000 and 1 <= self.memory_mb <= 1_000_000, "units / memory out of range")
        need(0 < self.tolerance <= 1e-1 and 0 < self.max_update_ratio <= 10, "tolerance out of range")
        need(self.fee_per_round >= 0 and self.budget >= 0, "negative fee or budget")
        need(self.compute_s == 0 or 5 <= self.compute_s <= MAX_COMPUTE_S,
             f"compute_s is 0 (sized from the job) or 5-{MAX_COMPUTE_S:.0f} seconds")
        need(self.shards == 1, "fine-tunes train whole models: shards is 1")
        if self.deepseek:
            return self._validate_deepseek_data(need)
        need(self.data_source.get("kind") == "tokens" and set(self.data_source) == {"kind", "sha256"}
             and re.fullmatch(r"[0-9a-f]{64}", str(self.data_source.get("sha256", ""))),
             'a fine-tune\'s data is {"kind": "tokens", "sha256": ...}')

    def _validate_deepseek_data(self, need) -> None:
        """A DeepSeek job's data: its text as token ids (deepseek_prep's .npz), and which sequences train and which
        are held back for the scorers: {"kind": "deepseek", "sha256", "train": [first, end], "heldback": [...]}."""
        d = self.data_source

        def span(x):
            return isinstance(x, list) and len(x) == 2 and all(type(v) is int for v in x) and 0 <= x[0] < x[1]
        need(d.get("kind") == "deepseek" and set(d) == {"kind", "sha256", "train", "heldback"}
             and re.fullmatch(r"[0-9a-f]{64}", str(d.get("sha256", ""))) and span(d.get("train"))
             and span(d.get("heldback")) and (d["train"][1] <= d["heldback"][0] or d["heldback"][1] <= d["train"][0]),
             "a DeepSeek job's data is {\"kind\": \"deepseek\", \"sha256\": ..., \"train\": [first, end], "
             "\"heldback\": [first, end]}, the two apart")

    # --- data ------------------------------------------------------------------

    @property
    def data(self) -> Dataset | TokenDataset:
        if self._data is None:
            if self.data_source["kind"] != "synthetic":
                raise InvalidJob(f"dataset for {self.job_id} not loaded; call attach_data()")
            self._data = Dataset.synthetic(self.spec, self.data_source["seed"])
        return self._data

    def attach_data(self, data: Dataset | TokenDataset | TokenFile, file_sha256: str) -> None:
        """Attach a downloaded dataset after checking it's the one the requester committed to."""
        if file_sha256 != self.data_source.get("sha256"):
            raise InvalidJob("dataset hash does not match the job's commitment")
        if self.deepseek or isinstance(data, TokenFile):
            if not (self.deepseek and isinstance(data, TokenFile)):
                raise InvalidJob("a DeepSeek job's data is its token file (deepseek_job.TokenFile), and no other job's")
            try:
                data.check(self.spec, self.data_source)
            except ValueError as e:
                raise InvalidJob(f"the dataset doesn't suit the job: {e}") from None
            self._data = DeepSeekData(data, dict(self.data_source))
            return
        if self.finetune != isinstance(data, TokenDataset):
            raise InvalidJob("a fine-tune trains on token sequences, and other jobs on tables")
        if self.finetune:
            try:
                data.check(self.spec)
            except Exception as e:  # noqa: BLE001 - whatever is wrong with a poster's file, it's this job's problem
                raise InvalidJob(f"the dataset doesn't suit the job: {e}") from None
        else:
            data.check_shape(self.spec)
        self._data = data

    def evaluate(self, params: np.ndarray, backend=None) -> tuple[float, float]:
        """(loss, accuracy) of these weights on the job's held-back data: for an MLP, cross-entropy and the share of
        rows classified right; for a fine-tune, cross-entropy and the share of reply tokens predicted right."""
        if self.deepseek:
            from . import deepseek_unit
            return deepseek_unit.evaluate(self.spec, self.data, params, backend)
        if self.finetune:
            from . import finetune
            return finetune.evaluate(self.spec, self.data, params, backend)
        return evaluate(self.spec, params, self.data.x_val, self.data.y_val)

    @property
    def sharded(self) -> bool:
        """Whether its units are trained by teams, one worker per shard (coordinator.py, shards.py)."""
        return self.shards > 1

    @property
    def wire_dtype(self) -> str:
        """Weights travel at the job's precision: float32 jobs move half the bytes."""
        return "<f4" if self.spec.precision == "float32" else "<f8"

    @property
    def unit_flops(self) -> int:
        return self.spec.flops_per_step(self.batch_size) * self.steps_per_unit

    @property
    def n_segments(self) -> int:
        return self.steps_per_unit // self.checkpoint_every

    @property
    def checkpoint_bytes(self) -> int:
        return self.spec.n_params * np.dtype(self.wire_dtype).itemsize

    def phase_lengths(self) -> dict[str, float]:
        """The shortest phases this job's rounds can have, in seconds, so the reference computer (REF_*) can do its
        part: in compute, fetch the weights, train one unit and upload its checkpoints; in reveal, resend a couple
        of checkpoints whose upload failed; in verify, fetch and replay the challenged segments of two units. A job
        can ask for a longer compute phase (compute_s), for bigger units. Nodes use these or their own phase
        lengths, whichever are longer."""
        b, ref = self.checkpoint_bytes, REF_GPU_FLOPS if self.finetune else REF_FLOPS
        unit = b / REF_DOWNLOAD + self.unit_flops / ref + self.n_segments * b / REF_UPLOAD
        check = 2 * CHECKED_SEGMENTS * b / REF_DOWNLOAD + CHECKED_SEGMENTS * self.unit_flops / self.n_segments / ref
        return {"compute": max(self.compute_s, 1.25 * unit), "reveal": 2 * b / REF_UPLOAD, "verify": 2 * check}


def is_finetune_spec(d) -> bool:
    """Whether a parsed spec is a fine-tune's: it has `lora`, and a `model` without `n_in`. That's a shape every node
    before fine-tunes refused (an MLP's model has n_in), so no spec they took, whatever extra fields it carried (a
    stray "base", say: they ignored unknown fields), changes meaning now."""
    return isinstance(d, dict) and "lora" in d and isinstance(d.get("model"), dict) and "n_in" not in d["model"]


# Every field a fine-tune's spec may have (TrainingJob.to_spec): anything else is refused rather than ignored.
FINETUNE_KEYS = {"id", "fee_per_round", "base", "manifest", "model", "lora", "seq_len", "optimizer", "precision", "data",
                 "init_seed", "lr", "batch_size", "steps_per_unit", "checkpoint_every", "units_per_round", "memory_mb",
                 "tolerance", "max_update_ratio", "compute_s"}


def load_dataset(path, max_bytes: int | None = None) -> Dataset | TokenDataset | TokenFile:
    """A dataset file of any kind: a table for an MLP (model.Dataset), token sequences for a fine-tune
    (finetune.TokenDataset), or a DeepSeek job's token file (deepseek_job.TokenFile), told apart by the arrays inside.
    Never unpickles; refuses anything else."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            names = {n.removesuffix(".npy") for n in z.namelist()}
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a dataset (.npz): {e}") from e
    if names == {"tokens", "lengths", "learn"}:
        return TokenFile.from_npz(path, max_bytes)
    return TokenDataset.from_npz(path, max_bytes) if "train_ids" in names else Dataset.from_npz(path, max_bytes)


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    job_id: str
    round: int
    index: int
    seed: int
    start_hash: bytes
    flops: int
    memory_mb: int

    @classmethod
    def make(cls, job: TrainingJob, round_no: int, index: int, start_hash: bytes) -> "WorkUnit":
        tag = h256(job.job_id.encode(), round_no.to_bytes(8, "big"), index.to_bytes(8, "big"))
        return cls(
            unit_id=tag.hex()[:16],
            job_id=job.job_id,
            round=round_no,
            index=index,
            seed=int.from_bytes(h256(b"seed", tag)[:8], "big"),
            start_hash=start_hash,
            flops=job.unit_flops,
            memory_mb=job.memory_mb,
        )

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "job_id": self.job_id, "round": self.round, "index": self.index,
                "seed": self.seed, "start_hash": self.start_hash.hex(), "flops": self.flops, "memory_mb": self.memory_mb}

    @classmethod
    def from_dict(cls, d: dict) -> "WorkUnit":
        return cls(str(d["unit_id"]), str(d["job_id"]), int(d["round"]), int(d["index"]), int(d["seed"]),
                   bytes.fromhex(d["start_hash"]), int(d["flops"]), int(d["memory_mb"]))


def commit_digest(unit_id: str, pubkey: bytes, root: bytes, salt: bytes) -> bytes:
    return h256(b"commit", unit_id.encode(), pubkey, root, salt)


@dataclass(frozen=True)
class Commitment:
    unit_id: str
    worker: str
    pubkey: bytes
    digest: bytes
    signature: bytes

    def message(self) -> bytes:
        return b"commit|" + self.unit_id.encode() + b"|" + self.digest


@dataclass(frozen=True)
class Reveal:
    """Everything a worker publishes after the beacon: the result plus the
    checkpoints the beacon says it must open (with Merkle proofs)."""
    unit_id: str
    worker: str
    merkle_root: bytes
    salt: bytes
    final_params: np.ndarray
    openings: dict[int, tuple[np.ndarray, list]] = field(default_factory=dict)


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    MISSED_DEADLINE = "missed_deadline"
    WITHHELD = "withheld"  # committed, then didn't reveal: forfeits the round, not slashed
    BAD_REVEAL = "bad_reveal"  # reveal doesn't match commitment, or openings don't match root
    FAILED_SANITY = "failed_sanity"  # NaN, wrong start/final, oversized update
    FAILED_REPLAY = "failed_replay"  # spot-check replay mismatch
    FAILED_REDUNDANCY = "failed_redundancy"  # lost a full-replay arbitration
    UNCHECKED = "unchecked"  # a fine-tune's claim nobody answered for, or the referee couldn't decide: unpaid, unslashed

    @property
    def is_fraud(self) -> bool:
        return self not in (Verdict.ACCEPTED, Verdict.MISSED_DEADLINE, Verdict.WITHHELD, Verdict.UNCHECKED)
