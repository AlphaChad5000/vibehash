"""`vibehash selftest`: can this computer mine, and what can it take? A minute or two, no wallet needed, nothing spent.
One line per check, then what the computer can take tonight and the command to run next. Exit status 0 only if it
can mine at all: some device trains work verifiers accept, and (unless --offline) the node answers with a clock this
computer agrees with. `--json` is for scripts.

  Software  Python, NumPy, cryptography, and this release against the newest on PyPI (one short request).
  Devices   what `vibehash devices` finds and each GPU's self-test (compute.self_test), PyTorch's CUDA build against
            what the driver supports (gpusetup), each GPU's memory and how much of it is free.
  Training  a real unit of the public network's fruit-demo job: its spec field for field, and its data (the packaged
            fruit.csv makes its dataset byte for byte), trained on the device the worker would use and checked as a
            verifier checks it: commitment, reveal, openings, sanity, every segment replayed on the CPU's NumPy
            reference within the job's tolerance (verification.py). A broken driver or a non-deterministic setup fails
            here. Then, with PyTorch and a GPU, a step of the LoRA code path (finetune.py, llm.py) on a tiny random
            Qwen3 built in memory, replayed on the CPU the way a fine-tune's checkers replay it. `--model NAME`
            adds one real unit of a ready model (it downloads the weights, so it asks first), timed, with its peak GPU
            memory. Measured FLOP/s is what the worker would tell the node.
  Network   the node's /status (round trip, network, height, round and phase, jobs waiting and why), this clock
            against the node's (its HTTP Date header; signed messages more than wire.MAX_CLOCK_SKEW_S off are
            refused), download speed from it, and whether huggingface.co answers (fine-tunes fetch weights there).
  Wallet    whether setup has run, the saved node and wallet, its address (read from the file, no password), balance
            and stake, and how long earning the stake should take at the network's recent pace.
  Disk      free space for the cache (fine-tunes' weights) and the model store (DeepSeek's ~30 GB, modelstore.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import advice, hardware, settings, termui, unattended
from .tasks import SYN_PER_VIBE

PYPI = "https://pypi.org/pypi/vibehash/json"
# The public fruit-demo job's dataset, by what's in it (arrays_sha): its file (sha256 6c20387f...) came from Windows, and
# the same arrays saved on Linux, or by another Python or NumPy, are other bytes (an .npz is a zip, which notes the system
# that wrote it), so the file's hash only matched where it was made. CI on Linux found it.
FRUIT_ARRAYS = "60107810e50f4a90efaf0d9e845ac457756be0887c23652ae6bf252992591ccf"
SKEW_WARN_S = 30  # a clock this far off still works, but drifts: worth fixing before it doesn't
CACHE_FOR_FINETUNES = 10e9  # Qwen3-4B's 8 GB of weights, plus room
STORE_FOR_DEEPSEEK = 40e9  # DeepSeek's ~30 GB of model files, hand-offs and modelstore's 2 GB margin (the menu says 40)
# The ready models' shapes, from their config.json (Qwen/Qwen3-0.6B and Qwen/Qwen3-4B), so the summary needs no
# download: what finetune.memory_mb sizes a fine-tune's workers by, at the longest sequences `train` makes (512).
SHAPES = {
    "qwen3-0.6b": {"family": "qwen3", "vocab": 151936, "hidden": 1024, "layers": 28, "heads": 16, "kv_heads": 8,
                   "head_dim": 128, "ffn": 3072, "tied": True},
    "qwen3-4b": {"family": "qwen3", "vocab": 151936, "hidden": 2560, "layers": 36, "heads": 32, "kv_heads": 8,
                 "head_dim": 128, "ffn": 9728, "tied": True},
}
# A tiny Qwen3 for the LoRA step (tests/finetune_fixtures.py makes the same shape), random, built in memory.
TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 96, "hidden_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 48, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
        "hidden_act": "silu", "max_position_embeddings": 512, "attention_bias": False}
PREFIX, SUFFIX = "This PC can't take DeepSeek work: ", ". It keeps earning on other jobs."  # deepseek_worker.can_take's
UNICODE = {"ok": "✓", "warn": "!", "fail": "✗", "skip": "·"}
ASCII = {"ok": "ok", "warn": "!!", "fail": "XX", "skip": "--"}


# --- the report -------------------------------------------------------------------------------------------------------

@dataclass
class Check:
    section: str
    name: str
    status: str  # ok, warn, fail or skip
    text: str
    data: dict = field(default_factory=dict)


class Report:
    """The checks so far, printed as they come (unless `quiet`, for --json)."""

    def __init__(self, quiet: bool = False, out=None):
        self.checks: list[Check] = []
        self.quiet, self.out = quiet, out or sys.stdout
        try:  # Unicode marks where this output can show them, else ASCII (as the dashboards fall back)
            "".join(UNICODE.values()).encode(getattr(self.out, "encoding", None) or "ascii")
            self.marks = UNICODE
        except (UnicodeEncodeError, LookupError):
            self.marks = ASCII
        self.colour = self.out is sys.stdout and termui.supported() and termui.COLOR
        self.width = max(60, min(110, shutil.get_terminal_size((100, 24)).columns - 1))
        self.section = ""

    def title(self, text: str) -> None:
        self.section = text
        self.say(f"\n{text}")

    def say(self, text: str = "") -> None:
        if not self.quiet:
            print(text, file=self.out, flush=True)

    def add(self, name: str, status: str, text: str, **data) -> Check:
        c = Check(self.section, name, status, text, data)
        self.checks.append(c)
        mark = self.marks[status]
        if self.colour:
            tint = {"ok": termui.GREEN, "warn": termui.YELLOW, "fail": termui.RED, "skip": termui.DIM}[status]
            shown = f"{tint}{mark}{termui.RESET}"
        else:
            shown = mark
        pad = " " * (len(mark) + 3)
        lines = textwrap.wrap(text, self.width - len(pad), break_on_hyphens=False, break_long_words=False) or [""]
        self.say(f" {shown}  {lines[0]}" + "".join(f"\n{pad}{line}" for line in lines[1:]))
        return c

    def get(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def failed(self, name: str) -> bool:
        c = self.get(name)
        return c is not None and c.status == "fail"


def _mb(n: float) -> str:
    return f"{n / 1024:.1f} GB" if n >= 1024 else f"{n:.0f} MB"


def _bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def _flops(f: float) -> str:
    return f"{f / 1e12:.2f} TFLOP/s" if f >= 1e12 else f"{f / 1e9:.2f} GFLOP/s"


def _vibe(syn: int) -> str:
    return f"{syn / SYN_PER_VIBE:,.8g} VIBE"


def _duration(s: float) -> str:
    return f"{s / 3600:.1f} hours" if s >= 5400 else f"{s / 60:.0f} minutes" if s >= 90 else f"{s:.0f} seconds"


# --- software ---------------------------------------------------------------------------------------------------------

def version() -> str:
    """This release (vibehash.__version__: a source checkout's pyproject.toml, else the installed package's)."""
    from . import __version__
    return __version__


def _parse(v: str) -> tuple:
    return tuple(int(p) for p in v.split(".") if p.isdigit())


def newest_release(timeout: float = 5.0) -> str:
    """The newest vibehash on PyPI (one request)."""
    with urllib.request.urlopen(urllib.request.Request(PYPI, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return str(json.loads(r.read(2**20))["info"]["version"])


def software(rep: Report, offline: bool) -> None:
    rep.title("Software")
    v = sys.version_info
    py = f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.release()} ({platform.machine()})"
    rep.add("python", "ok" if v >= (3, 10) else "fail", py if v >= (3, 10) else f"{py}: Vibe Hash needs 3.10 or newer",
            version=f"{v.major}.{v.minor}.{v.micro}")
    from . import crypto
    fast = crypto.BACKEND == "openssl"
    rep.add("packages", "ok" if fast else "warn",
            f"NumPy {np.__version__}" + (", cryptography (fast signatures)" if fast else
                                         "; cryptography is missing, so signatures are slow pure Python: "
                                         "`python -m pip install cryptography`"), numpy=np.__version__)
    mine = version()
    if offline:
        rep.add("release", "skip", f"Vibe Hash {mine} (not compared with PyPI: --offline)", version=mine)
        return
    try:
        newest = newest_release()
    except Exception as e:  # noqa: BLE001 - offline, PyPI down, a proxy: the rest of the test goes on
        rep.add("release", "skip", f"Vibe Hash {mine}; couldn't ask PyPI for the newest ({type(e).__name__})", version=mine)
        return
    if _parse(mine) < _parse(newest):
        rep.add("release", "warn", f"Vibe Hash {mine}, and {newest} is out: `python -m pip install --upgrade vibehash` "
                "(or run the one-line installer again)", version=mine, newest=newest)
    else:
        rep.add("release", "ok", f"Vibe Hash {mine}, the newest release", version=mine, newest=newest)


# --- devices ----------------------------------------------------------------------------------------------------------

@dataclass
class Device:
    id: str
    label: str
    backend: object = None
    problem: str | None = None
    total_mb: int | None = None
    free_mb: int | None = None


def devices(rep: Report) -> list[Device]:
    """Every device `vibehash devices` lists, self-tested, with each GPU's memory. The CPU is last."""
    from . import compute

    rep.title("Devices")
    found = []
    for d in compute.list_devices():
        if d.id == "cpu":
            found.append(Device("cpu", d.label, compute.NumpyBackend(), None, hardware.total_memory_mb()))
            continue
        try:
            backend = compute.TorchBackend(d.id)
            problem = compute.self_test(backend)
        except Exception as e:  # noqa: BLE001
            backend, problem = None, f"{type(e).__name__}: {e}"
        dev = Device(d.id, d.label, backend, problem, backend.memory_mb() if backend and not problem else None)
        if dev.total_mb and d.id.startswith("cuda"):
            try:
                import torch
                free, _ = torch.cuda.mem_get_info(int(d.id.split(":")[1]))
                dev.free_mb = int(free // 2**20)
            except Exception:  # noqa: BLE001
                pass
        found.append(dev)
    gpus = [d for d in found if d.id != "cpu"]
    for d in gpus:
        if d.problem:
            rep.add(f"device {d.id}", "fail", f"{d.id}  {d.label}: NOT USABLE ({d.problem[:160]}). Update the graphics "
                    "driver; until then this computer mines on the CPU", problem=d.problem)
            continue
        mem = f"{_mb(d.total_mb)}" if d.total_mb else "memory shared with the system"
        busy = ""
        if d.total_mb and d.free_mb is not None and d.free_mb < 0.7 * d.total_mb:
            busy = (f", but only {_mb(d.free_mb)} free: other programs hold {_mb(d.total_mb - d.free_mb)} of it (close "
                    "games, browsers or other workers before fine-tunes)")
        rep.add(f"device {d.id}", "warn" if busy else "ok",
                f"{d.id}  {d.label}: {mem}" + (f", {_mb(d.free_mb)} free" if d.free_mb is not None and not busy else "")
                + busy + "; self-test passed", total_mb=d.total_mb, free_mb=d.free_mb)
    cpu = found[-1]
    rep.add("device cpu", "ok", f"cpu     {cpu.label}, {_mb(cpu.total_mb)} of memory", total_mb=cpu.total_mb)
    _torch_line(rep, gpus)
    return found


def _torch_line(rep: Report, gpus: list[Device]) -> None:
    """PyTorch: present, and built for what the driver supports?"""
    from . import compute, gpusetup

    try:
        hardware_gpus = gpusetup.detect()
    except Exception:  # noqa: BLE001
        hardware_gpus = []
    nvidia = [g for g in hardware_gpus if g.vendor == "nvidia"]
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None or not compute.torch_available():
        if hardware_gpus:
            names = ", ".join(g.name for g in hardware_gpus)
            rep.add("pytorch", "warn", f"No GPU support installed, so {names} sits idle: `vibehash install-gpu` installs "
                    "the PyTorch build for it. Without it this computer mines CSV jobs on the CPU, and can't take "
                    "fine-tunes", installed=False)
        else:
            rep.add("pytorch", "skip", "PyTorch isn't installed, and no graphics card it supports was found: this "
                    "computer mines CSV jobs on the CPU", installed=False)
        return
    cuda, hip = getattr(torch.version, "cuda", None), getattr(torch.version, "hip", None)
    driver = next((g.cuda for g in nvidia if g.cuda), None)
    built = f"CUDA {cuda}" if cuda else f"ROCm {hip}" if hip else "the CPU only"
    head = f"PyTorch {torch.__version__}, built for {built}"
    data = {"installed": True, "version": torch.__version__, "cuda": cuda, "driver_cuda": ".".join(map(str, driver)) if driver else None}
    if nvidia and not cuda:
        rep.add("pytorch", "warn", f"{head}, so the NVIDIA card can't be used: `vibehash install-gpu` replaces it with "
                "the build for the card", **data)
    elif cuda and driver and tuple(int(x) for x in cuda.split(".")[:2]) > driver:
        rep.add("pytorch", "fail", f"{head}, newer than the {driver[0]}.{driver[1]} the graphics driver supports: update "
                "the driver from nvidia.com (or `vibehash install-gpu` picks a build this driver runs)", **data)
    elif nvidia and not any(d.id.startswith("cuda") and not d.problem for d in gpus):
        rep.add("pytorch", "fail", f"{head}, but it can't use the NVIDIA card (torch.cuda.is_available() is false): "
                "update the graphics driver, then `vibehash devices`", **data)
    else:
        rep.add("pytorch", "ok", head + (f"; the driver supports CUDA up to {driver[0]}.{driver[1]}" if driver else ""),
                **data)


def choose(found: list[Device], want: str = "auto") -> Device:
    """The device a worker started with `--device want` trains on (compute.get_backend's choice), or the CPU."""
    usable = [d for d in found if not d.problem]
    if want == "auto":
        return usable[0]
    full = want if ":" in want or want in ("cpu", "mps") else f"{want}:0"
    return next((d for d in usable if d.id == full), found[-1])


# --- training ---------------------------------------------------------------------------------------------------------

def arrays_sha(data) -> str:
    """A dataset's sha256 by its contents: each array's name, type, shape and little-endian bytes, in a fixed order. The
    same wherever it's worked out, unlike its .npz file's."""
    h = hashlib.sha256()
    for name in ("x_train", "y_train", "x_val", "y_val"):
        a = np.ascontiguousarray(getattr(data, name))
        h.update(f"{name} {a.dtype.str} {a.shape}\n".encode())
        h.update(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


def fruit_job():
    """The public network's fruit-demo job, as `vibehash train fruit.csv` posted it, with its dataset attached: the
    packaged fruit.csv makes the same arrays (FRUIT_ARRAYS), though not the same file bytes everywhere."""
    from importlib import resources

    from . import tabular
    from .blobs import file_sha256
    from .model import ModelSpec
    from .tasks import TrainingJob

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = Path(tmp) / "fruit.csv"
        src.write_bytes(resources.files("vibehash").joinpath("data", "fruit.csv").read_bytes())
        data, _, _, _ = tabular.to_dataset(tabular.read_csv(src))
        sha = file_sha256(tabular.save_dataset(data, Path(tmp) / "fruit.npz"))
    job = TrainingJob("fruit-demo", "", budget=0, fee_per_round=SYN_PER_VIBE, units_per_round=12,
                      spec=ModelSpec(5, (128, 128), 4, "float32"), data_source={"kind": "npz", "sha256": sha},
                      init_seed=11, lr=0.05, batch_size=128, steps_per_unit=40, checkpoint_every=5, memory_mb=1024,
                      tolerance=1e-4, max_update_ratio=0.5)
    job.attach_data(data, sha)
    return job


def train_unit(job, backend, cpu=None) -> dict:
    """Train one unit of `job` on `backend` as a worker does, then check it as a verifier would, on the CPU's NumPy
    reference: the commitment and reveal, the beacon's openings, the sanity checks, and a replay of every segment
    (not only the three a beacon challenges). Also: whether the same device repeats it bit for bit, and how far the
    CPU's whole replay of the unit lands from it."""
    from . import compute, verification as v
    from .crypto import KeyPair
    from .model import params_hash, relative_distance
    from .tasks import CHECKED_SEGMENTS, WorkUnit
    from .worker import HardwareProfile, Worker

    cpu = cpu or compute.NumpyBackend()
    start = job.spec.init_params(job.init_seed)
    unit = WorkUnit.make(job, 1, 0, params_hash(start))
    worker = Worker("selftest", KeyPair.generate(), HardwareProfile("this computer", 0, 0), fp_noise=0, backend=backend)
    t = time.perf_counter()
    c = worker.compute(unit, job, start)
    took = time.perf_counter() - t
    beacon = os.urandom(32)
    r = worker.reveal(unit, job, beacon, CHECKED_SEGMENTS)
    need = v.required_openings(job, beacon, unit.unit_id, worker.address, CHECKED_SEGMENTS)
    segments = v.challenge_segments(beacon, unit.unit_id, worker.address, job.n_segments, CHECKED_SEGMENTS)
    problems = []
    if not (v.check_commitment_signature(c) and v.check_reveal(c, r)):
        problems.append("its commitment doesn't match its reveal")
    if not v.check_openings(job, r, need):
        problems.append("its openings aren't in its Merkle tree")
    if not v.check_sanity(job, unit, start, r):
        problems.append("it fails the sanity checks (not finite, or not where it started)")
    ok, bad = v.replay_openings(job, unit, r.openings, segments, backend=cpu)
    if not ok:
        problems.append(f"a verifier's replay of segment {bad} on the CPU doesn't match")
    checkpoints = worker._work[unit.unit_id].checkpoints
    every = {i: (cp, None) for i, cp in enumerate(checkpoints)}
    ok, bad = v.replay_openings(job, unit, every, list(range(job.n_segments)), backend=cpu)
    if not ok and not problems:
        problems.append(f"segment {bad} doesn't replay on the CPU within the job's tolerance")
    replay = v.full_replay(job, unit, start, cpu)
    update = float(np.linalg.norm(replay - start))
    distance = relative_distance(r.final_params, replay)
    share = float(np.linalg.norm(r.final_params - replay)) / update if update else 0.0
    again = Worker("selftest", worker.keys, worker.hardware, fp_noise=0, backend=backend)
    again.compute(unit, job, start)
    repeats = again._work[unit.unit_id].root == worker._work[unit.unit_id].root
    return {"ok": not problems, "problems": problems, "seconds": took, "distance": distance, "share": share,
            "tolerance": job.tolerance, "repeats": repeats, "flops": job.unit_flops}


def training(rep: Report, found: list[Device], want: str) -> Device | None:
    """The fruit-demo unit on the worker's device (falling back to the CPU, as the worker would, if that fails)."""
    rep.title("Training")
    from . import compute

    job = fruit_job()
    device = choose(found, want)
    if want not in ("auto", "cpu") and device.id == "cpu":
        rep.add("train", "fail", f"--device {want} isn't usable here (see Devices); trying the CPU instead")
    tried = [device] + ([found[-1]] if device.id != "cpu" else [])
    for d in tried:
        try:
            got = train_unit(job, d.backend)
        except Exception as e:  # noqa: BLE001 - whatever the driver throws, it's this device's problem
            got = {"ok": False, "problems": [f"training failed: {type(e).__name__}: {str(e)[:160]}"]}
        where = f"on {d.id} ({d.label})"
        if got["ok"]:
            flops = hardware.benchmark(job, 0.5, d.backend)
            same = "repeats bit for bit" if got["repeats"] else "doesn't repeat bit for bit (still within tolerance)"
            rep.add("train", "ok" if got["repeats"] or d.id == "cpu" else "warn",
                    f"fruit-demo unit {where}: {job.steps_per_unit} steps in {got['seconds']:.2f} s, {_flops(flops)}. "
                    f"Checked as a verifier checks it: every segment replays on the CPU, the whole unit "
                    f"{got['distance']:.1e} from the CPU's (the job allows {job.tolerance:g}; "
                    f"{got['share']:.1e} of the update, 1e-2 allowed); it {same}",
                    device=d.id, seconds=got["seconds"], flops_per_sec=flops, distance=got["distance"],
                    share=got["share"], repeats=got["repeats"])
            if d.id != "cpu":
                cpu_flops = hardware.benchmark(job, 0.5, compute.NumpyBackend())
                rep.add("cpu speed", "ok", f"the same job on the CPU: {_flops(cpu_flops)}", flops_per_sec=cpu_flops)
            return d
        rep.add(f"train {d.id}", "fail", f"fruit-demo unit {where}: verifiers would reject it: "
                f"{'; '.join(got['problems'])}. " + ("Update the graphics driver, or mine on the CPU with --device cpu"
                                                     if d.id != "cpu" else "This Python's NumPy is broken: reinstall it"),
                device=d.id, problems=got["problems"])
    return None


def _tiny_model():
    """The tiny Qwen3's tensors {name: (dtype, shape, bytes)} in bf16, and its manifest and config as bytes."""
    from . import hub
    from .arch import MODULES, TARGETS, Arch

    arch = Arch.from_hf_config(TINY)
    rng = np.random.default_rng(0)
    tensors = {}

    def put(name, shape, scale=None, around=0.0):
        a = rng.normal(around, scale if scale is not None else 1 / np.sqrt(shape[-1]), shape).astype(np.float32)
        bits = a.view(np.uint32).astype(np.uint64)
        tensors[name] = ("BF16", list(shape), ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes())
    put("model.embed_tokens.weight", (arch.vocab, arch.hidden))
    for i in range(arch.layers):
        p = f"model.layers.{i}."
        for t in TARGETS:
            put(p + MODULES[t] + ".weight", arch.shape(t))
        put(p + "input_layernorm.weight", (arch.hidden,), 0.1, 1.0)
        put(p + "post_attention_layernorm.weight", (arch.hidden,), 0.1, 1.0)
        put(p + "self_attn.q_norm.weight", (arch.head_dim,), 0.1, 1.0)
        put(p + "self_attn.k_norm.weight", (arch.head_dim,), 0.1, 1.0)
    put("model.norm.weight", (arch.hidden,), 0.1, 1.0)
    raw = b"".join(t[2] for t in tensors.values())
    config = json.dumps(TINY, indent=2).encode()
    entries, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        entries[name] = hub.TensorEntry("model.safetensors", dtype, tuple(shape), offset, len(data),
                                        hashlib.sha256(data).hexdigest())
        offset += len(data)
    files = {"model.safetensors": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
             "config.json": {"size": len(config), "sha256": hashlib.sha256(config).hexdigest()}}
    manifest = hub.Manifest("selftest/tiny-qwen3", hashlib.sha1(raw).hexdigest(), files, entries)
    return arch, tensors, manifest.to_bytes(), config


def _token_data(manifest: bytes, config: bytes, vocab: int, seq_len: int, n_train: int, n_val: int, seed: int = 1):
    """Learnable sequences: a random prompt, a separator, the prompt again (the reply, the only part learned)."""
    from .finetune import TokenDataset

    rng = np.random.default_rng(seed)

    def make(n):
        ids, mask, lens = (np.zeros((n, seq_len), dtype=np.int32), np.zeros((n, seq_len), dtype=np.uint8),
                           np.zeros(n, dtype=np.int32))
        for i in range(n):
            k = int(rng.integers(3, (seq_len - 1) // 2 + 1))
            prompt = rng.integers(2, vocab, k)
            row = np.concatenate([prompt, [1], prompt])
            ids[i, :len(row)], lens[i] = row, len(row)
            mask[i, k + 1:len(row)] = 1
        return ids, lens, mask
    return TokenDataset(*make(n_train), *make(n_val), manifest=manifest, config=config)


def lora_step(device: str, replay: str = "cpu", steps: int = 4) -> dict:
    """Adam steps of a fine-tune unit on the tiny model, through finetune.train_segment (program.py's ops, as workers
    train), on `device`; then the same on `replay`, compared as a checker compares them (verification.adam_reproduces,
    part by part, from finetune.TOLERANCE). The model is put into finetune's runtimes directly, so nothing is fetched,
    and taken out again after."""
    from . import compute, finetune, llm, verification
    from .model import relative_distance

    arch, tensors, manifest, config = _tiny_model()
    data = _token_data(manifest, config, TINY["vocab_size"], 32, 32, 4)
    spec = finetune.LoraSpec(base=f"hf:selftest/tiny-qwen3@{hashlib.sha1(b''.join(t[2] for t in tensors.values())).hexdigest()}",
                             manifest=hashlib.sha256(manifest).hexdigest(), dims=finetune.arch_dims(TINY), seq_len=32)

    def read(name):
        return tensors[name]
    keys = []
    try:
        backends = {}
        for dev in dict.fromkeys((device, replay)):
            backends[dev] = compute.TorchBackend(dev)  # (TF32 off, deterministic kernels: what replays rely on)
            with finetune._lock:
                finetune._runtimes[(spec.manifest, dev)] = finetune._Runtime(llm.load(arch, read, dev), dev)
            keys.append((spec.manifest, dev))
        start = spec.init_params()
        t = time.perf_counter()
        trained = finetune.train_segment(spec, data, start, 7, 0, steps, 1e-3, 4, backends[device])
        took = time.perf_counter() - t
        again = finetune.train_segment(spec, data, start, 7, 0, steps, 1e-3, 4, backends[replay])
    finally:
        with finetune._lock:
            for k in keys:
                finetune._runtimes.pop(k, None)
    ok = verification.adam_reproduces(trained, again, start, finetune.TOLERANCE, spec, 1e-3, steps)
    worst = max(relative_distance(trained[p], again[p]) for p in spec.parts)
    moved = float(np.max(np.abs(trained[spec.parts[0]])))
    return {"ok": bool(ok and moved > 0), "seconds": took, "worst": worst, "tolerance": finetune.TOLERANCE,
            "moved": moved, "steps": steps}


def finetune_check(rep: Report, device: Device | None, found: list[Device]) -> None:
    from . import compute

    gpu = device if device is not None and device.id != "cpu" else next((d for d in found if d.id != "cpu" and not d.problem), None)
    if not compute.torch_available():
        rep.add("lora", "skip", "fine-tune step: skipped (needs PyTorch; `vibehash install-gpu` installs it for a GPU)")
        return
    if gpu is None:
        rep.add("lora", "skip", "fine-tune step: skipped (fine-tunes train on a GPU, and none is usable here)")
        return
    try:
        got = lora_step(gpu.id)
    except Exception as e:  # noqa: BLE001
        rep.add("lora", "fail", f"fine-tune step on {gpu.id}: failed ({type(e).__name__}: {str(e)[:200]}). This computer "
                "can still mine CSV jobs", device=gpu.id)
        return
    if got["ok"]:
        rep.add("lora", "ok", f"fine-tune step on {gpu.id} (LoRA on a tiny random Qwen3, {got['steps']} Adam steps, "
                f"{got['seconds']:.1f} s): the CPU's replay agrees to {got['worst']:.1e} (checkers allow "
                f"{got['tolerance']:g})", device=gpu.id, worst=got["worst"])
    else:
        rep.add("lora", "fail", f"fine-tune step on {gpu.id}: the CPU's replay is {got['worst']:.1e} away (checkers allow "
                f"{got['tolerance']:g}), so fine-tune work from this GPU would be rejected. Update the graphics driver",
                device=gpu.id, worst=got["worst"])


def model_unit(name: str, device: str, yes: bool = False, ask=None, say=print) -> dict:
    """One real unit of a ready model on `device`: its weights fetched as a worker fetches them (asked first), 16 Adam
    steps at the batch a job would use, on synthetic chats of 512 tokens (the longest `train` makes), timed, with
    its peak GPU memory; then its first segment again, which the same GPU must repeat bit for bit. The model's
    manifest and weights stay cached, so `train --base` and the worker reuse them."""
    import torch

    from . import catalog, compute, finetune, hub

    ref, _ = catalog.resolve(name)
    h = hub.Hub()
    repo, commit = h.resolve(ref)
    config = h.get_file(repo, commit, "config.json")
    dims = finetune.arch_dims(json.loads(config))
    listing = h.list_files(repo, commit)
    weights = sum(f.size for f in listing.values() if f.path.endswith(".safetensors"))
    pinned = finetune.manifest_path(repo, commit).exists()
    question = (f"One real unit of {repo} downloads its weights: {_bytes(weights)} "
                + ("(its hashes are already checked here) " if pinned else "streamed once to check their hashes, then ")
                + f"into {hub.default_cache_dir()}, kept there for its fine-tune jobs. Go ahead?")
    if not yes and not (ask and ask(question)):
        return {"ok": None, "why": "not downloaded" + ("" if ask else ": add --yes to run it with no terminal")}
    mbytes = finetune.model_manifest(repo, commit, h, progress=_progress(say))
    data = _token_data(mbytes, config, dims["vocab"], 512, 64, 4)
    spec = finetune.LoraSpec(base=f"hf:{repo}@{commit}", manifest=hashlib.sha256(mbytes).hexdigest(), dims=dims,
                             seq_len=512)
    s = finetune.settings_for(dims)
    manifest = hub.Manifest.from_bytes(mbytes)
    need = finetune.memory_mb(spec, s["batch_size"], manifest.bytes_for(n for n in manifest.tensors
                                                                       if n in finetune._model_tensors(dims, json.loads(config))))
    backend = compute.TorchBackend(device)
    index = int(device.split(":")[1]) if ":" in device else 0
    finetune.runtime(spec, data, device, progress=_progress(say))
    torch.cuda.synchronize(index)
    torch.cuda.reset_peak_memory_stats(index)
    theta, first = spec.init_params(), None
    t = time.perf_counter()
    for seg in range(finetune.UNIT_STEPS // finetune.SEGMENT_STEPS):
        theta = finetune.train_segment(spec, data, theta, 5, seg * finetune.SEGMENT_STEPS, finetune.SEGMENT_STEPS,
                                       s["lr"], s["batch_size"], backend)
        first = theta if first is None else first
    torch.cuda.synchronize(index)
    took = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated(index) / 2**20
    again = finetune.train_segment(spec, data, spec.init_params(), 5, 0, finetune.SEGMENT_STEPS, s["lr"],
                                   s["batch_size"], backend)
    finetune.unload()
    flops = spec.flops_per_step(s["batch_size"]) * finetune.UNIT_STEPS
    return {"ok": True, "repo": repo, "seconds": took, "peak_mb": peak, "need_mb": need, "flops_per_sec": flops / took,
            "repeats": bool(np.array_equal(first, again)), "batch": s["batch_size"]}


def _progress(say):
    last = [0.0]

    def show(done: int, total: int, what: str) -> None:
        now = time.monotonic()
        if done < total and now - last[0] < 2:
            return
        last[0] = now
        say(f"      {what[:50]}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
    return show


def model_check(rep: Report, name: str, device: Device | None, yes: bool) -> None:
    from . import advice as adv

    if device is None or not device.id.startswith("cuda"):
        rep.add("model", "skip", f"{name} unit: skipped (needs an NVIDIA GPU with PyTorch)")
        return
    ask = None
    if unattended.interactive():
        def ask(question):
            try:
                return input(f"      {question} [Y/n] ").strip().lower() in ("", "y", "yes")
            except EOFError:
                return False
    try:
        got = model_unit(name, device.id, yes, ask, rep.say)
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "OutOfMemoryError":
            rep.add("model", "fail", f"{name} unit on {device.id}: {adv.out_of_memory(device.id, None, device.total_mb, device.free_mb)}")
        else:
            rep.add("model", "fail", f"{name} unit on {device.id}: failed ({type(e).__name__}: {str(e)[:200]})")
        return
    if got["ok"] is None:
        rep.add("model", "skip", f"{name} unit: {got['why']}")
        return
    room = device.total_mb or 0
    rep.add("model", "ok" if got["repeats"] else "warn",
            f"{got['repo']} unit on {device.id}: 16 Adam steps of {got['batch']} sequences of 512 tokens in "
            f"{got['seconds']:.0f} s ({_flops(got['flops_per_sec'])}), peak GPU memory {_mb(got['peak_mb'])} plus "
            f"CUDA's own (jobs of it ask workers for {_mb(got['need_mb'])}; this card has {_mb(room)}); its first "
            f"segment {'repeats bit for bit' if got['repeats'] else 'does NOT repeat bit for bit on this GPU'}",
            **{k: v for k, v in got.items() if k != "ok"})


# --- network ----------------------------------------------------------------------------------------------------------

def _get(node, path: str, limit: int = 2**20):
    """GET `path` from a NodeClient's node: (body, the reply's Date header, seconds it took). Errors as NodeClient's."""
    import urllib.error

    from .client import NodeError

    req = urllib.request.Request(node.url + path)
    t = time.perf_counter()
    try:
        with node._open(req) as resp:
            body = resp.read(limit)
            date = resp.headers.get("Date")
    except urllib.error.HTTPError as e:
        e.close()
        raise NodeError(f"GET {path}: HTTP {e.code} {e.reason}", e.code) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise NodeError(f"can't reach node at {node.url}: {getattr(e, 'reason', e)}") from None
    return body, date, time.perf_counter() - t


def clock_skew(node, clock=time.time) -> float | None:
    """This computer's clock minus the node's (a NodeClient's), or None if the node can't say: the worker checks it
    before it starts, since every message it sends is signed with this clock."""
    from .client import NodeError
    try:
        sent = clock()
        _, date, _ = _get(node, "/status")
        return skew_from(date, sent, clock())
    except NodeError:
        return None


def skew_from(date: str | None, sent: float, got: float) -> float | None:
    """This clock minus the node's, from its Date header (whole seconds, so ±0.5 s plus half the round trip)."""
    from email.utils import parsedate_to_datetime
    if not date:
        return None
    try:
        theirs = parsedate_to_datetime(date).timestamp() + 0.5
    except (TypeError, ValueError):
        return None
    return (sent + got) / 2 - theirs


@dataclass
class Net:
    url: str
    status: dict | None = None
    account: dict | None = None
    jobs: dict = field(default_factory=dict)  # job id -> its spec, for the current and waiting jobs
    funded: list = field(default_factory=list)  # every job with budget left (/network, on nodes from 0.5.5)
    pace: tuple | None = None  # (VIBE a worker earned a paid round, seconds between paid rounds, rounds seen)


def network(rep: Report, url: str, address: str | None, offline: bool, clock=time.time) -> Net:
    from .client import NodeClient, NodeError
    from .wire import MAX_CLOCK_SKEW_S

    rep.title("Network")
    net = Net(url)
    if offline:
        rep.add("node", "skip", f"node {url}: not asked (--offline)")
        return net
    node = NodeClient(url, timeout=10)
    try:
        sent = clock()
        body, date, rtt = _get(node, "/status")
        got = clock()
        net.status = json.loads(body)
    except NodeError as e:
        rep.add("node", "fail", advice.explain(e, url) or str(e), url=url)
        return net
    except (ValueError, KeyError) as e:
        rep.add("node", "fail", f"{url} answered, but not as a Vibe Hash node does ({type(e).__name__}): check the address",
                url=url)
        return net
    s = net.status
    public = s.get("chain_id") == "vibehash-public-1"
    rnd = (f"round {s['round']} ({s['phase']}" + (f", job {s['job_id']})" if s.get("job_id") else ")")
           if s.get("round") is not None else "no round open (no job with budget)")
    rep.add("node", "ok", f"{url} answers in {rtt * 1000:.0f} ms: network {s['chain_id']}"
            + (" (the public network)" if public else "") + f", block {s['height']}, {rnd}",
            url=url, rtt_ms=rtt * 1000, chain_id=s["chain_id"], height=s["height"], round=s.get("round"),
            phase=s.get("phase"), job_id=s.get("job_id"))
    skew = skew_from(date, sent, got)
    if skew is None:
        rep.add("clock", "skip", "clock: the node didn't say its time")
    elif abs(skew) > MAX_CLOCK_SKEW_S - 2:
        rep.add("clock", "fail", advice.clock(skew), skew_s=skew, limit_s=MAX_CLOCK_SKEW_S)
    else:
        rep.add("clock", "warn" if abs(skew) > SKEW_WARN_S else "ok",
                f"clock: {abs(skew):.1f} s {'ahead of' if skew > 0 else 'behind'} the node's (it allows "
                f"{MAX_CLOCK_SKEW_S} s)" + (": set it automatically before it drifts further" if abs(skew) > SKEW_WARN_S
                                           else ""), skew_s=skew, limit_s=MAX_CLOCK_SKEW_S)
    _speed(rep, node, s)
    for job_id in dict.fromkeys([j for j in [s.get("job_id"), *s.get("waiting", {})] if j]):
        try:
            net.jobs[job_id] = node.get(f"/job/{job_id}")["spec"]
        except (NodeError, KeyError, TypeError):
            pass
    try:
        net.funded = [j["id"] for j in node.get("/network")["jobs"] if j.get("escrow")]
    except (NodeError, KeyError, TypeError):
        net.funded = list(net.jobs)
    if address:
        try:
            net.account = node.account(address)
        except NodeError:
            pass
    if s.get("earn_in") and (net.account or {}).get("stake", 0) < s.get("min_stake", 0):
        net.pace = earn_pace(node, s["height"])  # how fast a newcomer's stake fills, for the wait it has
    return net


def _speed(rep: Report, node, status: dict) -> None:
    """A real download from the node: the first chunk of this round's starting weights, or of a job's model."""
    from .blobs import CHUNK
    from .client import NodeError

    paths = []
    try:
        paths.append(f"/blob/{node.get('/round/params')['blob']}?offset=0")
    except (NodeError, KeyError, TypeError):
        pass
    if status.get("job_id"):
        paths.append(f"/job/{status['job_id']}/model?offset=0")
    for path in paths:
        try:
            body, _, took = _get(node, path, CHUNK)
        except NodeError:
            continue
        if not body:
            continue
        small = " (a small file, so mostly the round trip)" if len(body) < 2**20 else ""
        rate = len(body) / max(took, 1e-6)
        rep.add("download", "ok", f"download from the node: {_bytes(len(body))} in {took * 1000:.0f} ms, "
                f"{rate / 1e6:.1f} MB/s{small}", bytes=len(body), seconds=took, rate=rate)
        return
    rep.add("download", "skip", "download from the node: nothing to fetch right now (no round open)")


def hugging_face(rep: Report, offline: bool) -> bool | None:
    from . import hub

    if offline:
        rep.add("huggingface", "skip", "huggingface.co: not asked (--offline)")
        return None
    try:
        t = time.perf_counter()
        hub.Hub(timeout=8, retries=0).model_info("Qwen/Qwen3-0.6B")
        rep.add("huggingface", "ok", f"huggingface.co answers in {(time.perf_counter() - t) * 1000:.0f} ms (fine-tunes "
                "fetch their base model's weights there)")
        return True
    except Exception as e:  # noqa: BLE001
        rep.add("huggingface", "warn", f"huggingface.co didn't answer ({str(e)[:120]}): fine-tunes fetch their weights "
                "from it, so this computer can't take them until it can (a campus filter or firewall?). CSV jobs are "
                "fine")
        return False


def earn_pace(node, height: int, blocks: int = 40) -> tuple | None:
    """(VIBE each worker was paid a paid round, seconds between paid rounds, paid rounds seen) over the last
    `blocks` blocks: how fast a newcomer's stake fills, roughly (newcomers get smaller batches at first)."""
    from .client import NodeError

    try:
        got = node.get(f"/blocks?from={max(1, height - blocks + 1)}&to={height}")
    except NodeError:
        return None
    paid = []
    for b in got:
        pay = [t["amount"] for t in b.get("txs", []) if t.get("kind") == "reward" and t.get("memo") == "work"]
        if pay:
            paid.append((b["timestamp"], sum(pay) / len(pay)))
    if len(paid) < 2:
        return None
    span = paid[-1][0] - paid[0][0]
    return sum(p for _, p in paid) / len(paid), span / (len(paid) - 1), len(paid)


def waiting_lines(rep: Report, net: Net, gpu_mb: int | None) -> None:
    """Each job the node says is waiting, and what that means for this computer."""
    s = net.status or {}
    for job_id, why in (s.get("waiting") or {}).items():
        spec = net.jobs.get(job_id) or {}
        others = [j for j in net.funded if j != job_id and j not in s.get("waiting", {})]
        stake = net.account["stake"] if net.account else None
        rep.add(f"waiting {job_id}", "warn",
                advice.waiting(job_id, why, stake if stake is not None else 0, s.get("min_stake", 0), gpu_mb,
                               spec.get("memory_mb"), others, _eta(net), bool(s.get("earn_in"))), job_id=job_id, why=why)


def _eta(net: Net) -> str:
    if not net.pace or not net.status:
        return ""
    per_round, interval, seen = net.pace
    short = max(0, net.status["min_stake"] - (net.account or {}).get("stake", 0))
    if not per_round or not short:
        return ""
    rounds = math.ceil(short / per_round)
    return (f" (at the last {seen} paid rounds' pace, {_vibe(int(per_round))} a worker a round and one every "
            f"{_duration(interval)}, that's about {rounds} round{'s' * (rounds != 1)}, roughly "
            f"{_duration(rounds * interval)}; more miners share each round)")


# --- wallet and disk --------------------------------------------------------------------------------------------------

def wallet_address() -> tuple[Path | None, str | None, bool]:
    """(wallet file, its address, encrypted?) from the saved wallet, without its password."""
    path = settings.wallet_path()
    if not path or not path.exists():
        return path, None, False
    try:
        doc = json.loads(path.read_text())
        return path, doc["address"], "crypto" in doc
    except (OSError, ValueError, KeyError):
        return path, None, False


def wallet(rep: Report, net: Net, path: Path | None, address: str | None, encrypted: bool) -> None:
    rep.title("Wallet")
    cfg = settings.load()
    home = settings.home()
    pod = settings.pod_volume()
    where = f" (on the pod's {pod} volume, so it survives a restart)" if pod and home.is_relative_to(pod) else ""
    if pod and not home.is_relative_to(pod):
        rep.add("home", "warn", f"settings in {home}, which a pod restart wipes: set VIBEHASH_HOME=/workspace/.vibehash "
                "(see the README's RunPod notes) so the wallet survives", home=str(home))
    if not cfg:
        rep.add("setup", "warn", f"setup hasn't been run here ({home} has no settings): `vibehash setup` makes a wallet "
                "and picks the node", home=str(home))
    else:
        rep.add("setup", "ok", f"setup saved its settings in {home}{where}: node {cfg.get('node') or settings.node_url()}"
                + (f"; DeepSeek work: {cfg['deepseek']}" if cfg.get("deepseek") else ""), home=str(home),
                node=cfg.get("node"))
    if address is None:
        rep.add("wallet", "warn", "no wallet yet: `vibehash setup` makes one" if not path or not path.exists()
                else f"{path} isn't a wallet file Vibe Hash can read", path=str(path) if path else None)
        return
    unlock = ("encrypted; the worker asks for its password, or reads $VIBEHASH_PASSWORD when there's no terminal "
              + ("(set now)" if os.environ.get("VIBEHASH_PASSWORD") else "(not set)")) if encrypted else "not encrypted"
    rep.add("wallet", "ok", f"wallet {path}: {address} ({unlock})", path=str(path), address=address, encrypted=encrypted)
    if net.account is None:
        if net.status is not None:
            rep.add("balance", "skip", "balance: the node didn't say")
        return
    a, need = net.account, (net.status or {}).get("min_stake", 0)
    if a["stake"] >= need:
        state = f"staked {_vibe(a['stake'])}: it takes full work and checks others'"
    elif (net.status or {}).get("earn_in"):
        state = (f"stake {_vibe(a['stake'])} of {_vibe(need)}: its next {_vibe(need - a['stake'])} of pay becomes stake; "
                 "until then it gets small batches and checks nobody's work" + _eta(net))
    else:
        state = f"stake {_vibe(a['stake'])}: this network needs {_vibe(need)} staked before it gives work"
    rep.add("balance", "ok" if a["stake"] >= need or (net.status or {}).get("earn_in") else "warn",
            f"balance {_vibe(a['balance'])}, {state}", balance=a["balance"], stake=a["stake"], min_stake=need)


def disk(rep: Report) -> dict:
    from . import modelstore

    rep.title("Disk")
    cache = settings.home() / "cache"
    try:
        free = modelstore.free_space(cache)
    except OSError:
        free = 0
    rep.add("cache", "ok" if free >= CACHE_FOR_FINETUNES else "warn" if free >= 2e9 else "fail",
            f"cache {cache}: {_bytes(free)} free" + (": room for fine-tunes' weights (Qwen3-0.6B 1.2 GB, Qwen3-4B 8 GB)"
                                                    if free >= CACHE_FOR_FINETUNES else
                                                    ": enough for Qwen3-0.6B fine-tunes, not Qwen3-4B's 8 GB"
                                                    if free >= 2e9 else
                                                    ": too little for fine-tunes; free some, or set VIBEHASH_HOME to a "
                                                    "bigger drive"), path=str(cache), free=free)
    try:
        store = modelstore.Store.from_settings()
        sfree = modelstore.free_space(store.folder)
    except OSError:
        return {"cache": free, "store": 0, "cap": 0}
    roomy = sfree >= STORE_FOR_DEEPSEEK and store.cap >= 32e9
    rep.add("models", "ok" if roomy else "warn",
            f"model store {store.folder}: {_bytes(sfree)} free, up to {_bytes(store.cap)} allowed" +
            (": room for DeepSeek's ~30 GB" if roomy else
             f": DeepSeek work needs about {_bytes(STORE_FOR_DEEPSEEK)} free and allowed (`vibehash setup --models-dir "
             "FOLDER --models-gb 40`)"), path=str(store.folder), free=sfree, cap=store.cap)
    return {"cache": free, "store": sfree, "cap": store.cap}


# --- what it can take -------------------------------------------------------------------------------------------------

def finetune_need(name: str) -> int:
    """GPU memory (MB) a job fine-tuning `name` asks of each worker, at 512-token sequences and its usual batch."""
    from . import finetune

    dims = SHAPES[name]
    spec = finetune.LoraSpec(base="hf:x/y@" + "0" * 40, manifest="0" * 64, dims=dims, seq_len=512)
    return finetune.memory_mb(spec, finetune.settings_for(dims)["batch_size"], 2 * spec.base_params)


def summary(rep: Report, found: list[Device], trained: Device | None, net: Net, room: dict, offline: bool,
            hf: bool | None, address: str | None, want: str) -> dict:
    """What this computer can take tonight, and the next command."""
    from . import compute

    torch = compute.torch_available()
    gpu = trained if trained is not None and trained.id != "cpu" else None
    lora_ok = getattr(rep.get("lora"), "status", None) == "ok"
    take, lines = {}, []

    def line(key, ok, text):
        take[key] = ok
        lines.append((ok, text))
    node_ok = offline or (net.status is not None and not rep.failed("clock"))
    mine_ok = trained is not None and node_ok
    where = f"on its {gpu.label}" if gpu else "on the CPU"
    line("csv", trained is not None, f"fruit-demo and other CSV jobs, {where}" if trained else
         "no CSV jobs: training failed on every device (see Training)")
    gpu_mb = gpu.total_mb if gpu else None
    for name in SHAPES:
        need = finetune_need(name)
        if not torch or gpu is None:
            line(name, False, f"{name} fine-tunes: no ({'needs PyTorch and a GPU' if not torch else 'needs a usable GPU'};"
                 f" {_mb(need)} of GPU memory)")
        elif not lora_ok:
            line(name, False, f"{name} fine-tunes: no (the fine-tune step failed; see Training)")
        elif gpu_mb is not None and gpu_mb < need:
            line(name, False, f"{name} fine-tunes: no (each worker needs {_mb(need)} of GPU memory; {gpu.id} has "
                 f"{_mb(gpu_mb)})")
        elif room["cache"] < (1.5e9 if name == "qwen3-0.6b" else 9e9):
            line(name, False, f"{name} fine-tunes: no (not enough disk for its weights; see Disk)")
        else:
            snug = gpu_mb is not None and gpu_mb - need < 1024
            line(name, True, f"{name} fine-tunes: yes (needs {_mb(need)} of GPU memory; {gpu.id} has {_mb(gpu_mb)}"
                 + (": only just, so close other programs using the GPU" if snug else "") + ")"
                 + ("" if hf is not False else ", once huggingface.co answers"))
    for job_id, spec in net.jobs.items():
        if "base" in spec and "deepseek" not in spec and gpu_mb is not None:
            need = int(spec.get("memory_mb", 0))
            fits = torch and lora_ok and gpu_mb >= need
            line(f"job {job_id}", fits, f"job {job_id} on this node (fine-tunes {spec['base'].split('@')[0][3:]}): "
                 f"{'yes' if fits else 'no'} (needs {_mb(need)} of GPU memory)")
    deepseek = False
    if torch and gpu is not None and gpu.id.startswith("cuda"):
        from .deepseek_worker import can_take
        mach = {"gpu": {"name": gpu.label, "gb": (gpu.total_mb or 0) / 1024}, "ram_gb": hardware.total_memory_mb() / 1024}
        for stage in ("prep", "train"):
            ok, why = can_take(stage, mach)
            if ok and room["store"] < STORE_FOR_DEEPSEEK:
                ok, why = False, f"needs about {_bytes(STORE_FOR_DEEPSEEK)} free for its model files (see Disk)"
            take[f"deepseek {stage}"] = ok
            deepseek = deepseek or ok
        reason = "" if deepseek else f" ({why.removeprefix(PREFIX).removesuffix(SUFFIX)})"
        line("deepseek", deepseek, f"DeepSeek pilot (prep and training): {'yes' if deepseek else 'no'}{reason}"
             + ("; about 30 GB of model files, asked first" if deepseek else ""))
    else:
        take["deepseek prep"] = take["deepseek train"] = False
        line("deepseek", False, "DeepSeek pilot: no (needs an NVIDIA GPU with 8 GB, PyTorch, 16 GB of memory)")
    gpus = [d for d in found if d.id != "cpu" and not d.problem]
    cfg = settings.load()
    has_deepseek_job = any("deepseek" in spec for spec in net.jobs.values())
    if rep.failed("clock"):
        nxt = "fix the clock (see Network), then `vibehash selftest` again"
    elif not offline and net.status is None:
        nxt = "fix the connection to the node (see Network), then `vibehash selftest` again"
    elif trained is None:
        nxt = "fix the device problems above, then `vibehash selftest` again"
    elif address is None or not cfg:
        nxt = "vibehash setup" + (f" --node {net.url}" if net.url != settings.PUBLIC_NODE else "")
    elif deepseek and has_deepseek_job and cfg.get("deepseek") != "yes":
        nxt = "vibehash worker --deepseek yes   (or plain `vibehash worker` to leave DeepSeek out)"
    else:
        nxt = "vibehash worker" + (f" --device {want}" if want != "auto" else "")
    rep.say("\nTonight this computer can take:")
    for ok, text in lines:
        rep.say(f" {rep.marks['ok' if ok else 'skip']}  {text}")
    if len(gpus) > 1:
        rep.say(f"\nIt has {len(gpus)} GPUs: one worker uses one. Run a worker per GPU, each with its own wallet: "
                f"`vibehash worker --device cuda:1 --wallet gpu1.json` (README, \"Several GPUs\").")
    verdict = "can mine" if mine_ok else "can't mine yet"
    rep.say(f"\nThis computer {verdict}. Next: {nxt}")
    return {"can_mine": mine_ok, "can_take": take, "next": nxt}


# --- the command ------------------------------------------------------------------------------------------------------

def run(node: str | None = None, device: str = "auto", offline: bool = False, model: str | None = None,
        yes: bool = False, quiet: bool = False, out=None) -> tuple[int, dict]:
    """Every check, printed as it goes (unless `quiet`). -> (exit status, the report as JSON)."""
    from .cli import normalise_url

    rep = Report(quiet, out)
    url = normalise_url(settings.node_url(node))
    rep.say(f"Vibe Hash self-test: can this computer mine, and what can it take? (Nothing is spent, and no wallet is "
            f"needed.)")
    software(rep, offline)
    found = devices(rep)
    trained = training(rep, found, device)
    finetune_check(rep, trained, found)
    if model:
        model_check(rep, model, trained if trained is not None and trained.id != "cpu" else None, yes)
    path, address, encrypted = wallet_address()
    net = network(rep, url, address, offline)
    gpu_mb = trained.total_mb if trained is not None and trained.id != "cpu" else None
    waiting_lines(rep, net, gpu_mb)
    hf = hugging_face(rep, offline)
    wallet(rep, net, path, address, encrypted)
    room = disk(rep)
    result = summary(rep, found, trained, net, room, offline, hf, address, device)
    report = {"ok": result["can_mine"], "version": version(), "node": url, "checks": [c.__dict__ for c in rep.checks],
              "can_take": result["can_take"], "next": result["next"]}
    return (0 if result["can_mine"] else 1), report


def command(a) -> None:
    code, report = run(a.node, a.device, a.offline, a.model, a.yes, quiet=a.json)
    if a.json:
        print(json.dumps(report, indent=1, default=str))
    sys.exit(code)


def add_parser(sub) -> None:
    p = sub.add_parser("selftest", help="check this computer is ready to mine: a real training test, its GPUs, the "
                       "node, the clock, disk space, and what it can take")
    p.add_argument("--node", help="node to check against (default: the one `setup` saved, else the public network's)")
    p.add_argument("--device", default="auto", help="the device to train on, as for `worker` (default auto)")
    p.add_argument("--model", metavar="NAME", help=f"also train one real unit of a ready model, e.g. qwen3-0.6b "
                   "(downloads its weights once, asked first)")
    p.add_argument("--yes", "-y", action="store_true", help="don't ask before --model's download")
    p.add_argument("--offline", action="store_true", help="skip everything that goes online (node, PyPI, Hugging Face)")
    p.add_argument("--json", action="store_true", help="machine-readable output: every check, what it can take, next")
    p.set_defaults(fn=command)
