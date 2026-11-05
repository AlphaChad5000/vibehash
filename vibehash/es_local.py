"""Evo mode on a local test network: evolution-strategies jobs (es.py), set up the way scripts/es_local_run.py sets up
the measured run, for `vibehash testnet --evo` and `vibehash train --evo`. People see them as "Evo" jobs; the code's
names stay `es`.

* **The base model** comes from this computer's Hugging Face cache (`find`: `snapshot_download(local_files_only=True)`
  first, else a download), or from a local folder in Hugging Face's layout. A local network's job pins it by its
  commit, `hf:org/name@<commit>`, and by a stand-in for hub's manifest (`manifest`, the same as es_local_run's): fetching
  an ES job's base from the Hub by its manifest isn't built yet, so the base is wherever this computer keeps it.
  `find_base` is the worker's side: a job whose manifest is that stand-in finds its base in the cache by itself, so a
  `vibehash worker` started on its own takes an Evo job too, with nothing to switch on. It never downloads.
* **The spec** (`spec_for`) is the measured run's (HANDOFF.md, "Branch es-chain"): the last layer's MLP perturbed
  (layer 23 of Qwen2.5-0.5B-Instruct), sigma 1e-3, sgd with the paper's step (alpha = sigma**2 / 2), 8 antithetic
  pairs, 32 Countdown problems a round, 64 tokens, binary reward, level 2, retarget every 2 rounds toward half solved.
  Its tensors come from the model's config on PyTorch's meta device, as named_parameters names them, so nothing is
  loaded to post a job.
* **Only a chain whose genesis says `"es": true` takes one** (`ledger.ES_GATE`); `accepts` reads the genesis a node
  serves, which is how `train --evo` refuses a chain like today's public one before anything is spent.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from . import es
from .ledger import canonical
from .tasks import SYN_PER_VIBE, TrainingJob

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # the measured run's base
TAGLINE = "Evo mode: training by evolution, the chain is the model"
LOCAL_ONLY = ("Evo mode is for local test networks for now: this network doesn't take Evo jobs (start one with "
              "`vibehash testnet --evo`)")
# The measured run's settings (es_local_run.py's defaults), all but the layer, which is the model's last.
SETTINGS = dict(preset="mlp", sigma=1e-3, optimizer="sgd", pairs=8, pairs_per_unit=2, problems=32, max_tokens=64,
                reward="binary", level=2, retarget=(2, 0.5, 0.1))
FEE = 1 * SYN_PER_VIBE  # a round's: the measured run paid 1 VIBE a round
COMPUTE_S = 300  # the longest a round's compute phase may take; it ends as soon as every unit is in
STAKE = 10 * SYN_PER_VIBE  # each worker's in an Evo testnet's genesis: an ES job's work is only given to staked workers
FILES = ["config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer*",
         "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template*"]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$", re.ASCII)


@dataclass(frozen=True)
class Base:
    """A base model on this computer: its folder (Hugging Face's layout) and the reference a job pins it by."""
    folder: Path
    base: str  # hf:org/name@<40-hex commit>

    @property
    def repo(self) -> str:
        return self.base[3:].split("@")[0]


def manifest(base: str) -> str:
    """The stand-in for hub's manifest of a local network's base, as scripts/es_local_run.py makes it: the base
    model is registered under it (es.register) on each computer that has it."""
    return hashlib.sha256(f"es local run {base}".encode()).hexdigest()


def parse_ref(text: str) -> tuple[str, str | None]:
    """org/name, hf:org/name[@rev] or a huggingface.co address, as (repo, revision or None)."""
    ref, url = text.strip(), False
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf:"):
        if ref.lower().startswith(prefix):
            ref, url = ref[len(prefix):], prefix != "hf:"
    repo, _, rev = ref.partition("@")
    if url:  # an address's page: huggingface.co/org/name/tree/main and the like
        repo = "/".join(repo.strip("/").split("/")[:2])
    if not _REPO.match(repo):
        raise ValueError(f"{text!r} isn't a Hugging Face model (org/name) or a folder on this computer")
    return repo, rev or None


def find(model: str = MODEL, download: bool = True, config_only: bool = False) -> Base:
    """The base model `model` names: a folder on this computer, else a Hugging Face model from the local cache, or
    downloaded into it when it isn't there and `download` allows (`config_only`: just its config.json, which is
    all a job's spec needs). A folder has no commit, so it's pinned by a hash of where it is."""
    folder = Path(model).expanduser()
    if folder.is_dir():
        if not (folder / "config.json").is_file():
            raise ValueError(f"{folder} has no config.json: give a Hugging Face model's folder")
        name = re.sub(r"[^\w.-]+", "-", folder.resolve().name).strip("-.") or "model"
        return Base(folder.resolve(), f"hf:local/{name}@{hashlib.sha1(str(folder.resolve()).encode()).hexdigest()}")
    if "/" not in model:  # a short name from the model picker (catalog.py), as `train --base` takes them
        from .catalog import find as catalog_find
        known = catalog_find(model)
        model = known.repo if known is not None else model
    repo, rev = parse_ref(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("Evo mode needs huggingface_hub to find the model: pip install huggingface_hub") from None
    patterns = ["config.json"] if config_only else FILES
    try:
        path = snapshot_download(repo, revision=rev, allow_patterns=patterns, local_files_only=True)
        if not (Path(path) / "config.json").is_file() or not (config_only or list(Path(path).glob("*.safetensors"))):
            raise FileNotFoundError(path)
    except Exception:  # noqa: BLE001 - not in the cache (huggingface_hub's LocalEntryNotFoundError and the like)
        if not download:
            raise FileNotFoundError(f"{repo} isn't in this computer's Hugging Face cache") from None
        path = snapshot_download(repo, revision=rev, allow_patterns=patterns)
    folder = Path(path)
    commit = folder.name  # a snapshot's folder is named by its commit
    if not _COMMIT.match(commit):
        raise ValueError(f"{repo}: couldn't tell which commit {folder} is")
    return Base(folder, f"hf:{repo}@{commit}")


def register(found: Base) -> str:
    """Make `found` the base of every job that pins it, in this process; returns its manifest."""
    m = manifest(found.base)
    es.register(m, found.folder)
    return m


def find_base(spec: es.EsSpec) -> bool:
    """A worker's side: whether this process knows where `spec`'s base is, after looking for it in the local
    Hugging Face cache if the job is a local network's (its manifest is `manifest`'s stand-in for its base). Looks,
    never downloads; any trouble is just "no"."""
    if spec.manifest in es.LOCAL:
        return True
    if spec.manifest != manifest(spec.base) or spec.base.startswith("hf:local/"):
        return False
    try:
        found = find(f"{spec.repo}@{spec.commit}", download=False)
    except Exception:  # noqa: BLE001
        return False
    if found.base != spec.base:
        return False
    register(found)
    return True


def spec_for(found: Base, **settings) -> es.EsSpec:
    """An Evo job's spec for `found`: the measured run's settings (SETTINGS, which `settings` override), the last
    layer's MLP perturbed unless `layer` or `tensors` say otherwise, and the tensors' names and shapes from the model's
    config on PyTorch's meta device (nothing loaded)."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        raise RuntimeError("Evo mode needs PyTorch and transformers: `vibehash install-gpu`, then pip install "
                           "transformers") from None
    s = {**SETTINGS, **settings}
    c = AutoConfig.from_pretrained(found.folder)
    layer = s.pop("layer", None)
    preset = s.pop("preset")
    pats = tuple(s.pop("tensors", None) or es.preset(preset, c.num_hidden_layers - 1 if layer is None else layer))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(c)
    n, sha = es.tensors_digest(es.select(((k, p.shape) for k, p in model.named_parameters()), pats))
    if not n:
        raise ValueError(f"{found.repo} has no tensors named like {', '.join(pats)}")
    dims = {"family": c.model_type, "vocab": c.vocab_size, "hidden": c.hidden_size, "layers": c.num_hidden_layers,
            "heads": c.num_attention_heads, "kv_heads": getattr(c, "num_key_value_heads", None) or c.num_attention_heads,
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
            "ffn": c.intermediate_size, "tied": bool(getattr(c, "tie_word_embeddings", False))}
    sigma = s.pop("sigma")
    alpha = s.pop("alpha", None)
    spec = es.EsSpec(base=found.base, manifest=manifest(found.base), dims=dims, tensors=pats, n=n, tensors_sha=sha,
                     sigma=sigma, alpha=sigma * sigma / 2 if alpha is None else alpha, **s)
    bad = spec.problems_list()
    if bad:
        raise ValueError(f"an Evo job for {found.repo} can't run: {'; '.join(bad)}")
    return spec


def memory_mb(spec: es.EsSpec) -> int:
    """What a worker needs to take the job's units: the base in float32, twice over for the members' batch."""
    return max(1024, math.ceil(spec.base_params * 4 * 2 / 2**20))


def job(spec: es.EsSpec, job_id: str, requester: str, budget: int, fee: int = FEE) -> TrainingJob:
    """The Evo job to post: `spec`, a round's fee, a unit per `pairs_per_unit` pairs, checked as a node checks it."""
    return TrainingJob.from_spec_json(json.dumps({"id": job_id, "fee_per_round": fee, **spec.to_json(), "init_seed": 0,
                                                  "units_per_round": spec.units, "memory_mb": memory_mb(spec),
                                                  "compute_s": COMPUTE_S}), requester, budget)


def noise_mb(spec: es.EsSpec) -> int:
    """es.cache_noise's room for a process that plays every part (a testnet): every pair's noise for a round."""
    return min(2000, math.ceil(spec.pairs * spec.n * 4 / 2**20) + 16)


def accepts(genesis) -> bool:
    """Whether a chain whose genesis is `genesis` (as a node serves it at /genesis) takes Evo jobs."""
    return isinstance(genesis, dict) and genesis.get("es") is True


def chain_bytes(entry) -> int:
    """What a round adds to the chain to train the model: its record's ES entry (the round's seed, its level and every
    accepted pair's two scores) as canonical JSON, the bytes a rebuild works the model out from."""
    return len(canonical(entry)) if isinstance(entry, dict) else 0


# --- Evo v2 (es2.py): the fast recipe, on every layer, in cycles averaged into the next cycle's start ---------------

# The fast recipe's settings (the speed and multi-task tests): 64 antithetic pairs, 16 problems a round, 64 tokens,
# sigma 1e-3, a cosine from 1e-3 to a quarter of it over the job, replay of 16 earlier correct answers, the anchor.
V2_SETTINGS = dict(pairs=64, pairs_per_unit=8, problems=16, max_tokens=64, sigma=1e-3, lr=1e-3, lr_floor=0.25,
                   rounds=12, replay=16, margin=0.1, lp_tol=0.05, decay=1 / 64, lenient=True)
# lr 1e-3 and decay 1/64, the anchored step (es2's docstring, HANDOFF "The Evo collapse: cause and fix"): lr 4e-3 with
# no anchor drifted the model 1.6% from its base a round at the cosine's top, broke it within about 100 rounds, and
# made the testnet's ~400-round line worse on every task. Anchored, random drift levels off near 1-2%. lenient: a right
# answer pays with or without its <answer> tags (es_tasks.lenient), so rounds reward being right, not the format.
# margin 0.1: a checker teacher-forces answers through the engine that wrote them, in its passes of es_fast.ROWS rows,
# so it rounds exactly as the worker did. Measured on Qwen2.5-0.5B-Instruct (RTX 4070 SUPER, a worker's 64 members x 16
# answers on countdown, trivia_qa, commonsense_qa and sst2, each member checked alone, in the same process and in
# another): every honest member's worst gap was exactly 0 (with transformers' path, as before, up to 0.5625, and 1.125
# on the testnet). Answers that pay and that the model didn't write (the answer key's, or the model's with its answer
# swapped for the key's) sat 0.125 or more below at their worst token, but for 14 of 1,439 on commonsense_qa that tie
# exactly in bf16 (no margin tells those apart). So 0.1, under the smallest of the rest: for workers on one kind of GPU;
# another kind rounds differently and needs measuring first.
# lp_tol 0.05: workers and checkers score the replay answers on the same path (es_fast.score_items).
V2_COMPUTE_S = 300


def tensors2_for(config) -> tuple:
    """The linears a rule-2 job twists in a model with this config: every attention and MLP projection (es2.TENSORS),
    and on Qwen3.5's hybrid layers (Gated DeltaNet + full attention) es_hybrid.TENSORS."""
    from . import es2, es_hybrid
    tc = config.get_text_config() if hasattr(config, "get_text_config") else config
    return es_hybrid.TENSORS if "linear_attention" in (getattr(tc, "layer_types", None) or ()) else es2.TENSORS


def spec2_for(found: Base, task: str, start=(), **settings):
    """A rule-2 Evo job's spec for `found` on `task` (es_tasks.NAMES), starting from `start` ([[job id, weight,
    params_hash], ...]: the parents averaged into its start): every attention and MLP projection twisted
    (tensors2_for), their names and shapes from the model's config on PyTorch's meta device (nothing loaded). A
    Qwen3.5 model's config holds its text model's dims in text_config; its vision tower isn't loaded."""
    from . import es2
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        raise RuntimeError("Evo mode needs PyTorch and transformers: `vibehash install-gpu`, then pip install "
                           "transformers") from None
    s = {**V2_SETTINGS, **settings}
    full = AutoConfig.from_pretrained(found.folder)
    c = full.get_text_config()
    tensors = tensors2_for(full)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(full)
    n, sha = es.tensors_digest(es.select(((k, p.shape) for k, p in model.named_parameters()), tensors))
    dims = {"family": c.model_type, "vocab": c.vocab_size, "hidden": c.hidden_size, "layers": c.num_hidden_layers,
            "heads": c.num_attention_heads, "kv_heads": getattr(c, "num_key_value_heads", None) or c.num_attention_heads,
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
            "ffn": c.intermediate_size, "tied": bool(getattr(c, "tie_word_embeddings", False))}
    spec = es2.Spec2(base=found.base, manifest=manifest(found.base), dims=dims, tensors=tensors, n=n,
                     tensors_sha=sha, task=task, start=tuple(tuple(x) for x in start), **s)
    bad = spec.problems_list()
    if bad:
        raise ValueError(f"an Evo v2 job for {found.repo} can't run: {'; '.join(bad)}")
    return spec


def memory_mb2(spec) -> int:
    """What a v2 worker needs: the base in bf16, its float32 change, and the fast engine's caches for every member."""
    return max(4096, math.ceil(spec.base_params * 2 * 6 / 2**20))


def job2(spec, job_id: str, requester: str, budget: int | None = None, fee: int = FEE) -> TrainingJob:
    """The v2 Evo job to post: a round's fee, a unit per `pairs_per_unit` pairs, and a budget for exactly the planned
    rounds unless one is given (more would run at the step size's floor)."""
    budget = budget if budget is not None else spec.rounds * fee
    return TrainingJob.from_spec_json(json.dumps({"id": job_id, "fee_per_round": fee, **spec.to_json(), "init_seed": 0,
                                                  "units_per_round": spec.units, "memory_mb": memory_mb2(spec),
                                                  "compute_s": V2_COMPUTE_S}), requester, budget)
