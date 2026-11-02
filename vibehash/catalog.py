"""Popular open-weight models to fine-tune, and what training each one would take.

`vibehash train DATA --base NAME` fine-tunes a model one GPU holds, of a family finetune.py trains (`ready`), on a
network whose node runs this version (it needs no GPU: workers check and score the work): a local `testnet` today, the
public network once its node is updated (FINETUNING.md). Models that need a
team of GPUs are coming (DEEPSEEK.md). `vibehash models` lists them all, `vibehash train --base NAME` alone describes
one and posts nothing, and the guide's command builder shows the same list, baked into the page
(scripts/bake_models.py).

The choice of models is ours: the most downloaded text models of the big open-weight makers on Hugging Face in
November 2026, across sizes, with the small ones first. A model that one gaming GPU holds is what the network will
fine-tune first (DEEPSEEK.md §5). The facts come live from Hugging Face (hub.Hub.describe): parameters, weight size,
licence, gated or not, experts and layers. They're cached for a day in ~/.vibehash/cache/models.json, so the list comes
back at once after the first time, and from the cache, however old, when the Hub can't be reached.

"What it takes" is DEEPSEEK.md §2's maths: a team holds the weights as stored, whole layers per GPU, with a quarter of
each card's memory left for activations, adapters and the CUDA context. For DeepSeek-V4.1-Flash that gives the 40, 20
and 14 GPUs DEEPSEEK.md measured.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from . import settings
from .hub import DTYPE_BITS, AccessError, Hub, HubError, ModelSummary, parse_ref

GPU_GB = (12, 24, 32)  # gaming cards: an RTX 4070 SUPER has 12 GB, a 4090 24, a 5090 32
USABLE = 0.75  # of a card's memory, for weights; the rest is for activations, adapters and the CUDA context
CACHE_TTL = 24 * 3600  # seconds a model's facts are reused before Hugging Face is asked again
TIMEOUT_S = 8.0  # for each request, with no retries: a Hub that's out of reach should cost seconds, not minutes
TODAY = "small models sized to a CSV file"  # what the public network trains today
# Architectures finetune.py trains (its FAMILIES, by the name config.json gives them).
ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "SmolLM3ForCausalLM"}
FINETUNING = "https://github.com/AlphaChad5000/vibehash/blob/main/FINETUNING.md"
# The owner's words, shown wherever model names and emblems are (here, the guide, the README, emblems.py).
DISCLAIMER = ("Disclaimer: All product names, logos, and brands are property of their respective owners and are "
              "used purely for identification. Vibe Hash isn't affiliated with or endorsed by them.")


@dataclass(frozen=True)
class Model:
    name: str  # what `train --base` takes
    title: str  # what people call it
    repo: str  # its Hugging Face repository
    left_out: tuple[tuple[str, str, int], ...] = ()  # parts of the weights a team never holds on its GPUs: (what, why, bytes)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


# Smallest first. Instruct (chat-tuned) versions where a family has both: that's what fine-tunes usually start from.
CATALOG = (
    Model("qwen3-0.6b", "Qwen3 0.6B", "Qwen/Qwen3-0.6B"),
    Model("llama-3.2-1b", "Llama 3.2 1B Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    Model("smollm3-3b", "SmolLM3 3B", "HuggingFaceTB/SmolLM3-3B"),
    Model("phi-4-mini", "Phi-4 mini", "microsoft/Phi-4-mini-instruct"),
    Model("qwen3-4b", "Qwen3 4B", "Qwen/Qwen3-4B"),
    Model("gemma-4-e2b", "Gemma 4 E2B", "google/gemma-4-E2B-it"),
    Model("mistral-7b", "Mistral 7B Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
    Model("llama-3.1-8b", "Llama 3.1 8B Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    Model("qwen3-8b", "Qwen3 8B", "Qwen/Qwen3-8B"),
    Model("gpt-oss-20b", "gpt-oss 20B", "openai/gpt-oss-20b"),
    Model("qwen3-14b", "Qwen3 14B", "Qwen/Qwen3-14B"),
    Model("qwen3.8-27b", "Qwen3.8 27B", "Qwen/Qwen3.8-27B"),
    Model("gemma-4-31b", "Gemma 4 31B", "google/gemma-4-31B-it"),
    Model("gpt-oss-120b", "gpt-oss 120B", "openai/gpt-oss-120b"),
    Model("glm-5.3-flash", "GLM-5.3-Flash", "zai-org/GLM-5.3-Flash"),
    Model("glm-5.3", "GLM-5.3", "zai-org/GLM-5.3"),
    # DEEPSEEK.md §1-2, measured from its files: the two engram tables become per-job slices of the rows a job's text
    # reads, and a text fine-tune never loads the draft (MTP and DSpark) layers or the vision tower.
    Model("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash",
          left_out=(("the engram tables", "each job gets a slice of the rows its data reads", 203_100_000_000),
                    ("the draft layers and vision tower", "a text fine-tune doesn't use them", 8_900_000_000))),
)


def find(name: str) -> Model | None:
    """A catalog model by its short name or its repository, in any case."""
    name = name.strip().lower()
    return next((m for m in CATALOG if name in (m.name, m.repo.lower())), None)


def resolve(text: str) -> tuple[str, Model | None]:
    """What `--base` was given, as (a reference Hub.describe takes, the catalog model if it's one): a short name, an
    `org/name[@rev]`, `hf:org/name` or a huggingface.co address. Anything else is a ValueError saying what to give."""
    model = find(text)
    if model:
        return model.repo, model
    if "/" not in text and ":" not in text:
        close = difflib.get_close_matches(text.strip().lower(), [m.name for m in CATALOG], n=3, cutoff=0.6)
        hint = f"did you mean {', '.join(close[:-1]) + ' or ' if len(close) > 1 else ''}{close[-1]}? " if close else ""
        raise ValueError(f"no model called {text!r} in the catalog: {hint}`vibehash models` lists them, or give a "
                         "Hugging Face name like Qwen/Qwen3-0.6B")
    repo, rev = parse_ref(text)
    return (repo if rev is None else f"{repo}@{rev}"), (find(repo) if rev is None else None)


# --- facts from the Hub --------------------------------------------------------------------

@dataclass(frozen=True)
class Facts:
    """What Hugging Face says about a model: ModelSummary's numbers, kept as plain JSON for the cache."""
    repo: str
    commit: str
    parameters: int | None
    weight_bytes: int | None
    weight_files: int | None  # None when the files couldn't be listed (estimated)
    estimated: bool  # weight_bytes is worked out from the Hub's parameter counts, not summed from the files
    licence: str | None
    gated: bool
    architecture: str | None
    dtype: str | None
    layers: int | None
    experts: int | None
    experts_per_token: int | None
    has_tokenizer: bool | None
    pickle_only: bool
    checked: float  # when Hugging Face was asked (Unix time)

    @classmethod
    def from_summary(cls, s: ModelSummary) -> "Facts":
        # Models that also read images (Gemma 4, Qwen 3.5, DeepSeek-V4.1) keep their language model's shape in
        # config.json's text_config, which describe() doesn't look in yet (DEEPSEEK.md §5 step 3).
        text = s.config.get("text_config") if isinstance(s.config.get("text_config"), dict) else {}
        experts = s.experts or next((text[k] for k in ("n_routed_experts", "num_local_experts", "num_experts")
                                     if type(text.get(k)) is int), None)
        dtype = s.dtype or text.get("torch_dtype") or text.get("dtype")
        return cls(repo=s.repo, commit=s.commit, parameters=s.parameters, weight_bytes=s.weight_bytes,
                   weight_files=s.weight_files, estimated=False, licence=s.licence, gated=s.gated,
                   architecture=s.architecture, dtype=str(dtype) if dtype else None,
                   layers=s.layers or _count(text.get("num_hidden_layers")), experts=experts,
                   experts_per_token=s.experts_per_token or _count(text.get("num_experts_per_tok")),
                   has_tokenizer=s.has_tokenizer, pickle_only=s.pickle_only, checked=time.time())

    @classmethod
    def from_info(cls, repo: str, info: dict) -> "Facts":
        """From the Hub's description alone, for a model whose files it won't list. The weight size is worked out from
        the Hub's parameter count per dtype, which is what the files hold, give or take their headers."""
        st = info.get("safetensors") if isinstance(info.get("safetensors"), dict) else {}
        counts = st.get("parameters") if isinstance(st.get("parameters"), dict) else {}
        size = None
        if counts and all(d in DTYPE_BITS and _count(n) is not None for d, n in counts.items()):
            size = sum(n * DTYPE_BITS[d] // 8 for d, n in counts.items())
        card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
        licence = card.get("license_name") if card.get("license") == "other" else card.get("license")
        licence = licence or next((t[8:] for t in info.get("tags") or [] if isinstance(t, str) and t.startswith("license:")), None)
        config = info.get("config") if isinstance(info.get("config"), dict) else {}
        archs = config.get("architectures")
        return cls(repo=repo, commit=str(info.get("sha")), parameters=_count(st.get("total")), weight_bytes=size,
                   weight_files=None, estimated=True, licence=str(licence) if licence else None,
                   gated=bool(info.get("gated")),
                   architecture=str(archs[0]) if isinstance(archs, list) and archs else config.get("model_type"),
                   dtype="/".join(sorted(counts)).lower() or None, layers=None, experts=None, experts_per_token=None,
                   has_tokenizer=None, pickle_only=False, checked=time.time())

    @classmethod
    def from_dict(cls, d: dict) -> "Facts":
        if not isinstance(d.get("checked"), (int, float)):
            raise TypeError("not a cache entry")
        return cls(**{f.name: d[f.name] for f in fields(cls)})


def _count(x) -> int | None:
    return x if type(x) is int and x >= 0 else None


def describe(ref: str, hub: Hub | None = None) -> Facts:
    """A model's facts, asked of Hugging Face now. For a gated model whose licence you haven't accepted, the Hub lists
    its files with their hashes hidden, which Hub.describe refuses; its description still gives the rest."""
    hub = hub or Hub(timeout=TIMEOUT_S, retries=0)
    try:
        return Facts.from_summary(hub.describe(ref))
    except AccessError:
        raise
    except HubError as e:
        if e.retryable:  # couldn't reach it, or it's busy: asking again differently won't help
            raise
        try:
            repo, rev = parse_ref(ref, hub.endpoint)
            info = hub.model_info(repo, rev)
        except HubError:
            raise e from None
        if not info.get("gated"):
            raise e
        return Facts.from_info(repo, info)


def cache_path() -> Path:
    return settings.home() / "cache" / "models.json"


def _load_cache() -> dict[str, dict]:
    try:
        cached = json.loads(cache_path().read_text())
        return cached["models"] if isinstance(cached.get("models"), dict) else {}
    except (OSError, ValueError, AttributeError, KeyError):
        return {}


def _save_cache(models: dict[str, dict]) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"models": models}, indent=1))
        os.replace(tmp, path)
    except OSError:
        pass  # a cache that can't be written only makes the next look slower


def cached(refs) -> dict[str, Facts]:
    """What the cache holds for each reference, however old, without asking anyone: for the `train` menu, which has
    to open at once wherever Hugging Face is out of reach."""
    saved = _load_cache()
    facts = {}
    for ref in dict.fromkeys(refs):
        try:
            facts[ref] = Facts.from_dict(saved[ref])
        except (KeyError, TypeError):
            pass
    return facts


def lookup(refs, *, refresh: bool = False) -> tuple[dict[str, Facts], dict[str, str]]:
    """Facts for each reference: from the cache when it's less than a day old (unless `refresh`), else from Hugging
    Face, several at once. Returns (facts, errors). When the Hub can't answer for a model, its error is in `errors`,
    and its facts are the cached ones however old, or missing if it was never looked up. Once one request finds
    the Hub out of reach, the ones not yet sent aren't: they'd only wait out the same timeout."""
    refs = list(dict.fromkeys(refs))
    saved = _load_cache()
    facts = cached(refs)
    ask = [r for r in refs if refresh or r not in facts or time.time() - facts[r].checked > CACHE_TTL]
    errors: dict[str, str] = {}
    if ask:
        unreachable = None
        with ThreadPoolExecutor(min(8, len(ask))) as pool:
            futures = {pool.submit(describe, ref): ref for ref in ask}
            for done in as_completed(futures):
                ref = futures[done]
                try:
                    got = done.result()
                except CancelledError:
                    errors[ref] = unreachable
                    continue
                except (HubError, ValueError) as e:
                    errors[ref] = str(e)
                    if isinstance(e, HubError) and e.status is None and e.retryable and unreachable is None:
                        unreachable = str(e)  # no connection at all: the rest would fail the same way
                        for f in futures:
                            f.cancel()
                    continue
                facts[ref] = got
                saved[ref] = asdict(got)
        _save_cache(saved)
    return facts, errors


# --- what training would take --------------------------------------------------------------

def held_bytes(facts: Facts, model: Model | None = None) -> int | None:
    """The bytes a team holds on its GPUs: the weights as stored, less what DEEPSEEK.md cuts out."""
    if facts.weight_bytes is None:
        return None
    return max(0, facts.weight_bytes - sum(b for _, _, b in (model.left_out if model else ())))


def gpus(weight_bytes: int, layers: int | None, gb: int) -> int | None:
    """GPUs with `gb` of memory that hold `weight_bytes` between them, whole layers each, a quarter of each card kept
    free. None if one layer doesn't fit. Without a layer count, the bytes are shared out evenly."""
    usable = gb * 2**30 * USABLE  # cards are sold in GiB: a "12 GB" card holds 12 × 2^30 bytes
    if weight_bytes <= usable:
        return 1
    if not layers:
        return math.ceil(weight_bytes / usable)
    per_gpu = int(usable // (weight_bytes / layers))
    return math.ceil(layers / per_gpu) if per_gpu else None


def team(facts: Facts, model: Model | None = None) -> dict[int, int | None]:
    """{GPU memory in GB: GPUs a team needs}, for each of GPU_GB. Empty if the weight size isn't known."""
    held = held_bytes(facts, model)
    return {} if held is None else {gb: gpus(held, facts.layers, gb) for gb in GPU_GB}


# --- words ---------------------------------------------------------------------------------

def size_text(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} kB"


def params_text(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n:,}"


def ready(f: Facts | None, model: Model | None = None) -> bool:
    """Whether finetune.py can fine-tune it today: an architecture it trains, held by one gaming GPU."""
    return f is not None and not f.pickle_only and f.architecture in ARCHITECTURES and team(f, model).get(GPU_GB[0]) == 1


def status_of(f: Facts | None, model: Model | None = None) -> str:
    return "ready" if ready(f, model) else "coming"


def team_text(t: dict[int, int | None]) -> str:
    return " / ".join("-" if t.get(gb) is None else str(t[gb]) for gb in GPU_GB) if t else "?"


def experts_text(f: Facts | None) -> str:
    if f is None or not f.experts:
        return "-" if f is not None and not f.estimated else "?"
    return f"{f.experts_per_token} of {f.experts}" if f.experts_per_token else str(f.experts)


def checked_text(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def status_line(prog: str = "vibehash") -> str:
    return (f"Status: \"ready\" models fine-tune with `{prog} train chats.jsonl --base NAME` on a network whose node runs "
            f"this version (a local `{prog} testnet` today; the public network once its node is updated). Today the "
            f"public network trains {TODAY} (`{prog} train data.csv`). Fine-tuning models that need a team of GPUs is "
            "coming.")


def to_json(facts: dict[str, Facts], errors: dict[str, str]) -> dict:
    """What `vibehash models --json` prints."""
    out = []
    for m in CATALOG:
        f = facts.get(m.repo)
        row = {"name": m.name, "title": m.title, "repo": m.repo, "url": m.url, "status": status_of(f, m)}
        row.update({k: v for k, v in asdict(f).items() if k != "repo"} if f else {})
        if f:
            row["checked"] = datetime.fromtimestamp(f.checked, timezone.utc).isoformat(timespec="seconds")
            row["team_bytes"] = held_bytes(f, m)
            row["team"] = {str(gb): n for gb, n in team(f, m).items()}
        row["error"] = errors.get(m.repo)
        out.append(row)
    return {"today": f"{TODAY} (vibehash train data.csv)",
            "fine_tuning": "ready models: vibehash train DATA --base NAME, on a node that runs fine-tunes (a local "
                           "testnet; not the public network yet); the others are coming",
            "gpu_gb": list(GPU_GB), "usable_fraction": USABLE, "models": out, "disclaimer": DISCLAIMER}


def table(facts: dict[str, Facts], errors: dict[str, str], width: int = 120, prog: str = "vibehash") -> str:
    """The catalog as `vibehash models` prints it. The repository column is left out when it wouldn't fit."""
    head = ["name", "parameters", "experts", "weights", "licence", "gated", "GPUs at " + " / ".join(map(str, GPU_GB)) + " GB",
            "status"]
    rows = []
    for m in CATALOG:
        f = facts.get(m.repo)
        rows.append([m.name, params_text(f.parameters) if f else "?", experts_text(f),
                     (size_text(f.weight_bytes) + ("*" if f.estimated else "")) if f else "?",
                     (f.licence or "not stated") if f else "?", ("yes" if f.gated else "no") if f else "?",
                     team_text(team(f, m)) if f else "?", status_of(f, m), m.repo])
    widths = [max(len(r[i]) for r in rows + [head + ["Hugging Face"]]) for i in range(len(head) + 1)]
    with_repo = sum(widths) + 2 * len(widths) <= width
    if with_repo:
        head, rows = head[:1] + ["Hugging Face"] + head[1:], [r[:1] + r[-1:] + r[1:-1] for r in rows]
        widths = widths[:1] + widths[-1:] + widths[1:-1]
    else:
        widths = widths[:-1]
        rows = [r[:-1] for r in rows]
    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    out = [status_line(prog), "", line(head)] + [line(r) for r in rows] + [""]
    out.append(f"GPUs: how many cards of each size hold the weights between them, a quarter of each card's memory left "
               "for training (DEEPSEEK.md, section 2). \"-\": one layer doesn't fit.")
    for m in CATALOG:
        f = facts.get(m.repo)
        if f and m.left_out:
            parts = " and ".join(f"{what} ({size_text(b)}: {why})" for what, why, b in m.left_out)
            out.append(f"{m.name}: a team holds {size_text(held_bytes(f, m))}, leaving out {parts}.")
    if any(f.estimated for f in facts.values()):
        out.append("*: worked out from the Hub's parameter count. A gated model lists its files only to accounts that "
                   "have accepted its licence.")
    for m in CATALOG:
        if m.repo in errors:
            had = facts.get(m.repo)
            out.append(f"{m.name}: couldn't ask Hugging Face ({errors[m.repo]})"
                       + (f"; showing what it said on {checked_text(had.checked)}" if had else ""))
    when = [f.checked for m in CATALOG if (f := facts.get(m.repo)) and m.repo not in errors]
    if when:
        out.append(f"Facts from Hugging Face, checked {checked_text(min(when))}"
                   + ("" if max(when) - min(when) < 60 else f" to {checked_text(max(when))}") + ".")
    if not with_repo:
        out.append(f"`{prog} models --json` gives each model's Hugging Face page.")
    out.append(f"`{prog} train --base NAME` shows what training one would take, and posts nothing; "
               f"`{prog} train chats.jsonl --base NAME` fine-tunes a ready one.")
    out.append(DISCLAIMER)
    return "\n".join(out)


def preflight(ref: str, model: Model | None, f: Facts, data: Path | None = None, prog: str = "vibehash") -> str:
    """What `train --base` shows before anything could be posted: the model, its size, what a team needs, its licence."""
    name = f"{model.name}: {f.repo}" if model else f.repo
    shape = [p for p in (f.architecture,
                         f"{params_text(f.parameters)} parameters" if f.parameters is not None else None,
                         f"a mixture of {f.experts} experts" + (f" ({f.experts_per_token} per token)" if f.experts_per_token else "")
                         if f.experts else None,
                         f"{f.layers} layers" if f.layers else None) if p]
    lines = [f"Model     {name}",
             f"          https://huggingface.co/{f.repo} at commit {f.commit[:12]}" + (f" ({ref.partition('@')[2]})" if "@" in ref else ""),
             f"          {', '.join(shape) or 'shape not stated'}"]
    if f.pickle_only:
        lines.append("Weights   no safetensors, only PyTorch pickles. Loading a pickle can run any code its author put "
                     "in it, so Vibe Hash never will: this model can't be fine-tuned here.")
    elif f.estimated:
        lines.append(f"Weights   about {size_text(f.weight_bytes)} ({f.dtype or 'dtype not stated'}), from the Hub's "
                     "parameter count: its files are listed only to accounts that accepted its licence")
    else:
        lines.append(f"Weights   {size_text(f.weight_bytes)} in {f.weight_files} safetensors file{'s' * (f.weight_files != 1)}"
                     + (f" ({f.dtype})" if f.dtype else ""))
    if f.gated:
        lines.append(f"Licence   {f.licence or 'not stated'}, gated: accept it at https://huggingface.co/{f.repo}, then "
                     "set HF_TOKEN to a token of that account")
    else:
        lines.append(f"Licence   {f.licence or 'not stated'}, not gated")
    t = team(f, model)
    if t and not f.pickle_only:
        fits = [f"{n} with {gb} GB" for gb, n in t.items() if n is not None]
        held = size_text(held_bytes(f, model))
        if t[GPU_GB[0]] == 1:
            lines.append(f"GPUs      one {GPU_GB[0]} GB card holds its {held} of weights, with a quarter of it left for training")
        else:
            say = ", ".join(fits[:-1]) + (" or " if len(fits) > 1 else "") + fits[-1] if fits else                 "none of these sizes: one layer is bigger than the largest card"
            say += "".join(f"; one layer is too big for {gb} GB" for gb, n in t.items() if n is None and fits)
            lines += [f"GPUs      {say}",
                      f"          holding {held} of weights between them, a quarter of each card left for training"]
        for what, why, b in (model.left_out if model else ()):
            lines.append(f"          leaving out {what}, {size_text(b)}: {why} (DEEPSEEK.md)")
    if f.has_tokenizer is False:
        lines.append("Tokenizer no tokenizer.json, which turning text into training data needs")
    if data is not None:
        lines.append(f"Data      {data.name}: {size_text(data.stat().st_size)}" + (f" of {kind}" if (kind := data_kind(data)) else ""))
    if ready(f, model) and data is None:
        lines += ["", f"Ready: `{prog} train chats.jsonl --base {model.name if model else f.repo}` fine-tunes it on chats "
                  "(JSON lines of {\"messages\": [...]}) or text ({\"text\": ...}): a pre-flight, then one question "
                  "before anything is posted.",
                  f"It runs on a network whose node runs this version (no GPU needed there): a local `{prog} testnet --job "
                  f"none` today, the public network once its node is updated. Nothing was posted and nothing was spent. More: {FINETUNING}"]
    else:
        lines += ["", "Not yet: fine-tuning this model isn't live on the network (it needs a team of GPUs, or an "
                  "architecture that isn't supported yet), so nothing was posted and nothing was spent.",
                  f"What works today: a small model sized to a CSV file, `{prog} train data.csv`, and fine-tunes of the "
                  f"models `{prog} models` marks ready. What's coming: {FINETUNING}"]
    return "\n".join(lines)


TEXT_KINDS = {".jsonl": "JSON lines", ".json": "JSON", ".txt": "plain text", ".md": "Markdown text"}


def data_kind(path: Path) -> str | None:
    """Text a fine-tune would read, by its file name; None for anything else (a CSV is today's kind of training)."""
    return TEXT_KINDS.get(path.suffix.lower())
