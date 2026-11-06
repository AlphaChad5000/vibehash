"""Vibe Bench: score Hugging Face models side by side on the benchmarks Qwen publishes for its small models, with the
same items, prompts and settings for every model, and ~95% intervals so a real gain can be told from noise.

  python scripts/vibe_bench.py Qwen/Qwen3.5-0.8B [FOLDER_OR_REPO ...] [--bench mmlu_pro,ifeval] [--limit 200] [--full]
      [--out ~/.vibehash/bench] [--engine auto|fast|hf] [--fresh] [--list]

The first model is the reference: every other one gets a delta against it on the items both answered, with a paired
95% interval (the per-item differences' standard error), marked `+` when the whole interval is above zero and `-`
when it is below. A model's own score has a binomial interval, 1.96 * sqrt(p (1 - p) / n).

How it scores (the same for every model):
* Each benchmark is a fixed subset, pinned at a dataset commit: its items ordered by sha256("vibe-bench|BENCH|ID")
  and the first --limit of them taken (so a bigger limit keeps the smaller one's items), or all of them with --full.
* One chat turn, the model's own chat template with enable_thinking=False (Qwen3.5's non-thinking mode: an empty
  <think></think>), greedy decoding, stopping at the end-of-turn token. Multiple choice asks for brief reasoning and a
  last line "The answer is X"; math asks Qwen's own way (reason step by step, the answer in \\boxed{}); IFEval's
  prompts go in as they are. Every reply may have 1,024 tokens (--max-new): at 512, Qwen3.5-0.8B's MMLU-Pro replies
  were cut off mid-reasoning 42% of the time, at 1,024 a third, and nearly all of those never finish (still going at
  2,048: re-checking options that don't match its arithmetic, or repeating itself), so more mostly costs time.
* Answers are read leniently (the letter after "answer is" in several languages, \\boxed{}, "答案是", a lone letter on
  the last line; for math math-verify's equivalence check), and every item records how its answer was read, whether
  the reply ran out of tokens and whether it was in the expected language, so format failures show up as numbers.
* IFEval is scored by lm-evaluation-harness's port of Google's checker (strict and loose, prompt and instruction
  level): `pip install lm-eval langdetect nltk immutabledict`. MATH-500 uses `pip install math-verify`.
* Engine: a Qwen3.5-style hybrid model (Gated DeltaNet layers) on CUDA decodes with vibehash's own es_hybrid engine
  (continuous batching, CUDA graphs, ~3x transformers' generate here) with a zero twist; anything else, or
  --engine hf, uses transformers' generate. Both are greedy; scores are only compared between runs of one engine.

Per-item results go to OUT/MODEL/BENCH.jsonl (the prompt, the reply, the answer read, right or wrong) beside a
.meta.json; a later run reuses a model's saved results when its weights, the items, the prompts and the engine are
unchanged, so scoring a new export against the official model only runs the new one. --fresh runs everything again.

GPQA Diamond is gated on Hugging Face: it's scored only when this computer's token has access (accept the terms at
huggingface.co/datasets/Idavidrein/gpqa, then `hf auth login`), and skipped with a note otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS_V = 1  # bump when a prompt or an answer reader changes: saved results from another version are run again
SALT = "vibe-bench"

# Qwen's published numbers for Qwen3.5-0.8B in non-thinking mode (model card), as percentages.
PUBLISHED = {"mmlu_pro": 29.7, "mmlu_redux": 48.5, "ceval": 46.4, "supergpqa": 16.9, "ifeval/prompt_strict": 52.1,
             "mmmlu": 34.1}

LETTERS = "ABCDEFGHIJ"


@dataclass
class Item:
    id: str
    prompt: str
    gold: object
    meta: dict = field(default_factory=dict)


@dataclass
class Bench:
    name: str
    kind: str  # mc, mc_zh, gsm8k, math, ifeval
    limit: int  # default subset size
    max_new: int
    source: str  # repo@commit
    load: Callable[[], list]
    lang: str = "en"  # the language a reply should be in: en, zh, or "" (not checked)


# --- the datasets, each pinned at a commit -------------------------------------------------------------------------

def _download(repo: str, filename: str, rev: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=rev)


def _files(repo: str, rev: str, suffix: str) -> list[str]:
    """The repo's files at `rev` ending in `suffix`: from the Hub, or from the local cache when offline."""
    try:
        from huggingface_hub import HfApi
        names = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
    except Exception:  # noqa: BLE001 - offline: what an earlier run cached
        from huggingface_hub.constants import HF_HUB_CACHE
        snap = Path(HF_HUB_CACHE) / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev
        names = [p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file()]
    return sorted(n for n in names if n.endswith(suffix))


def _mc_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options))
    return ("Answer the following multiple choice question. Think step by step, briefly, then end your reply with "
            "\"The answer is X\", where X is the letter of the correct option.\n\n"
            f"Question: {question.strip()}\n\nOptions:\n{opts}")


def _mc_prompt_zh(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options))
    return ("以下是一道单项选择题。请逐步简要思考，最后以“答案是 X”结束回答，其中 X 是正确选项的字母。\n\n"
            f"题目：{question.strip()}\n\n选项：\n{opts}")


def _math_prompt(problem: str) -> str:
    return f"{problem.strip()}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


MMLU_PRO = ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be")
MMLU_REDUX = ("edinburgh-dawg/mmlu-redux-2.0", "372ea425445d51e1ba1188c56e5e893f8138621f")
GPQA = ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6")
SUPERGPQA = ("m-a-p/SuperGPQA", "4430d4458112c7d4497fdcf94d7cc223313d6acf")
IFEVAL = ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84")
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
CEVAL = ("ceval/ceval-exam", "617524a00b307ff6f9933702f724131fe12ca7ce")
MMMLU = ("openai/MMMLU", "325a01dc3e173cac1578df94120499aaca2e2504")


def load_mmlu_pro():
    import pandas as pd
    df = pd.read_parquet(_download(*MMLU_PRO[:1], "data/test-00000-of-00001.parquet", MMLU_PRO[1]))
    out = []
    for r in df.itertuples(index=False):
        opts = [str(o) for o in r.options]
        out.append(Item(str(r.question_id), _mc_prompt(r.question, opts), r.answer,
                        {"category": r.category, "options": len(opts)}))
    return out


def load_mmlu_redux():
    """MMLU-Redux 2.0's 57 subjects (100 re-annotated MMLU questions each), only the questions its annotators found
    sound (error_type "ok"): about 5,300."""
    from datasets import Dataset
    out = []
    for f in _files(*MMLU_REDUX, ".arrow"):
        subject = f.split("/")[0]
        d = Dataset.from_file(_download(MMLU_REDUX[0], f, MMLU_REDUX[1]))
        for i, r in enumerate(d):
            if r["error_type"] != "ok":
                continue
            out.append(Item(f"{subject}:{i}", _mc_prompt(r["question"], r["choices"]), LETTERS[int(r["answer"])],
                            {"subject": subject}))
    return out


def load_gpqa_diamond():
    """GPQA Diamond's 198 questions, the right answer and three wrong ones in an order seeded by the question's id."""
    import pandas as pd
    df = pd.read_csv(_download(GPQA[0], "gpqa_diamond.csv", GPQA[1]))
    out = []
    for r in df.to_dict("records"):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"{SALT}|gpqa|{r['Record ID']}").shuffle(order)
        shown = [str(opts[i]).strip() for i in order]
        out.append(Item(str(r["Record ID"]), _mc_prompt(r["Question"], shown), LETTERS[order.index(0)],
                        {"domain": r.get("High-level domain")}))
    return out


def load_supergpqa():
    rows = [json.loads(line) for line in open(_download(SUPERGPQA[0], "SuperGPQA-all.jsonl", SUPERGPQA[1]),
                                              encoding="utf-8")]
    return [Item(r["uuid"], _mc_prompt(r["question"], r["options"]), r["answer_letter"],
                 {"discipline": r["discipline"], "difficulty": r["difficulty"], "options": len(r["options"])})
            for r in rows]


def load_ifeval():
    rows = [json.loads(line) for line in open(_download(IFEVAL[0], "ifeval_input_data.jsonl", IFEVAL[1]),
                                              encoding="utf-8")]
    return [Item(str(r["key"]), r["prompt"], {"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]})
            for r in rows]


def load_gsm8k():
    import pandas as pd
    df = pd.read_parquet(_download(GSM8K[0], "main/test-00000-of-00001.parquet", GSM8K[1]))
    return [Item(str(i), _math_prompt(r.question), float(r.answer.split("####")[-1].replace(",", "").strip()))
            for i, r in enumerate(df.itertuples(index=False))]


def load_math500():
    rows = [json.loads(line) for line in open(_download(MATH500[0], "test.jsonl", MATH500[1]), encoding="utf-8")]
    return [Item(r["unique_id"], _math_prompt(r["problem"]), r["answer"],
                 {"level": r["level"], "subject": r["subject"]}) for r in rows]


def load_ceval():
    """C-Eval's validation questions (52 subjects, 1,346): its test answers aren't public."""
    import pandas as pd
    out = []
    for f in _files(*CEVAL, ".parquet"):
        if "/val-" not in f:
            continue
        subject = f.split("/")[0]
        df = pd.read_parquet(_download(CEVAL[0], f, CEVAL[1]))
        for r in df.itertuples(index=False):
            out.append(Item(f"{subject}:{r.id}", _mc_prompt_zh(r.question, [r.A, r.B, r.C, r.D]), r.answer,
                            {"subject": subject}))
    return out


def load_mmmlu():
    """MMMLU: MMLU's test set professionally translated into 14 languages (14,042 questions each); a subset draws
    from all of them at once, so each language gets about 1/14 of it."""
    import pandas as pd
    out = []
    for f in _files(*MMMLU, ".csv"):
        loc = f.rsplit("_", 1)[-1].removesuffix(".csv")
        df = pd.read_csv(_download(MMMLU[0], f, MMMLU[1]), keep_default_na=False)
        for i, r in enumerate(df.to_dict("records")):
            out.append(Item(f"{loc}:{i}", _mc_prompt(str(r["Question"]), [r["A"], r["B"], r["C"], r["D"]]),
                            str(r["Answer"]).strip(), {"lang": loc, "subject": r.get("Subject")}))
    return out


def _src(pair):
    return f"{pair[0]}@{pair[1][:12]}"


BENCHES = {b.name: b for b in (
    Bench("mmlu_pro", "mc", 400, 1024, _src(MMLU_PRO), load_mmlu_pro),
    Bench("mmlu_redux", "mc", 400, 1024, _src(MMLU_REDUX), load_mmlu_redux),
    Bench("gpqa_diamond", "mc", 198, 1024, _src(GPQA), load_gpqa_diamond),
    Bench("supergpqa", "mc", 400, 1024, _src(SUPERGPQA), load_supergpqa),
    Bench("ifeval", "ifeval", 541, 1024, _src(IFEVAL), load_ifeval),
    Bench("gsm8k", "gsm8k", 400, 1024, _src(GSM8K), load_gsm8k),
    Bench("math500", "math", 300, 2048, _src(MATH500), load_math500),
    Bench("ceval", "mc_zh", 400, 1024, _src(CEVAL), load_ceval, lang="zh"),
    Bench("mmmlu", "mc", 400, 1024, _src(MMMLU), load_mmmlu, lang=""),
)}


def subset(bench: str, items: list[Item], limit: int | None) -> list[Item]:
    """The bench's fixed subset: items in sha256 order of their ids, the first `limit` (all of them for None)."""
    key = lambda it: hashlib.sha256(f"{SALT}|{bench}|{it.id}".encode()).hexdigest()  # noqa: E731
    ordered = sorted(items, key=key)
    return ordered if limit is None else ordered[:limit]


# --- reading answers --------------------------------------------------------------------------------------------------

_LET = r"[\(\[（]?\s*([A-J])(?![A-Za-z0-9])"
_SKIP = r"(?:\*\*|__|\$|\\\(|\\\[|\\textbf\{|\\text\{|\s)*"
_MC_SAID = [re.compile(p) for p in (
    r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?" + _LET,
    # "the answer is B", "Answer: (B)", "**Final answer:** B", "La respuesta correcta es la B", "Die Antwort ist B"
    r"(?i:answer|respuesta|r[ée]ponse|antwort|risposta|resposta|jawaban|jibu|ìdáhùn)\**(?:\s+[^\W\d_]{1,12}){0,4}"
    r"\s*[:：]?" + _SKIP + _LET,
    r"(?:答案|正确答案|正確答案|答え|正解|정답|답)\s*(?:应该是|应为|是|为|為|は|：|:)?\s*[:：]?" + _SKIP + _LET,
    r"(?:故|所以|因此|则|則|应|應)\s*选(?:择)?\s*[:：]?" + _SKIP + _LET,
)]
_LONE = re.compile(r"(?<![A-Za-z0-9])[\(\[]?([A-J])[\)\]]?(?![A-Za-z0-9])")


def read_letter(text: str, finished: bool = True) -> tuple[str | None, str]:
    """(the letter a reply chose, how it was read): the last place it says so ("said"), else, for a reply that
    finished (not cut off mid-sentence, where a last line is any formula), the one letter alone on its last line
    ("last line"), else (None, "none")."""
    best = None
    for p in _MC_SAID:
        for m in p.finditer(text):
            if best is None or m.start(1) > best[0]:
                best = (m.start(1), m.group(1))
    if best:
        return best[1], "said"
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and finished:
        found = set(_LONE.findall(lines[-1].replace("*", "")))
        if len(found) == 1:
            return found.pop(), "last line"
    return None, "none"


def last_boxed(text: str) -> str | None:
    """What the last \\boxed{...} (or \\fbox{...}) holds, braces balanced."""
    at = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if at < 0:
        return None
    i = text.find("{", at)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _number(s: str) -> float | None:
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\{", "", s)  # \text{18 dollars} -> 18 dollars
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 1,000 -> 1000
    s = re.sub(r"\\[a-zA-Z]+|[${}\\]", " ", s)
    nums = _NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[0].replace(",", ""))
    except ValueError:
        return None


def read_number(text: str, finished: bool = True) -> tuple[float | None, str]:
    """(a math reply's final number, how it was read): the last \boxed{}, else after "answer", else (a finished
    reply only) its last number."""
    b = last_boxed(text)
    if b is not None:
        n = _number(b)
        if n is not None:
            return n, "boxed"
    m = list(re.finditer(r"(?i:answer)(?:\s+[^\W\d_]{1,12}){0,3}\s*[:：]?\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", text))
    if m:
        return float(m[-1].group(1).replace(",", "")), "said"
    nums = _NUM.findall(text) if finished else []
    if nums:
        try:
            return float(nums[-1].replace(",", "")), "last number"
        except ValueError:
            pass
    return None, "none"


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-z]")


def lang_ok(text: str, lang: str) -> bool | None:
    """Is a reply in the language its benchmark is in? en: little CJK; zh: mostly CJK (letters A-J aside)."""
    if not lang:
        return None
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk + latin == 0:
        return True
    frac = cjk / (cjk + latin)
    return frac < 0.2 if lang == "en" else frac >= 0.5


def _ifeval_scorer():
    try:
        from lm_eval.tasks.ifeval import utils
    except ImportError:
        sys.exit("IFEval needs lm-evaluation-harness's checker: pip install lm-eval langdetect nltk immutabledict")
    return utils.process_results


def _math_checker():
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError:
        sys.exit("MATH-500 needs math-verify: pip install math-verify")

    def check(gold: str, reply: str) -> tuple[bool, str | None]:
        g = parse(f"${gold}$", parsing_timeout=None)
        p = parse(reply, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()], parsing_timeout=None)
        ok = bool(g) and bool(p) and verify(g, p, timeout_seconds=None)
        return ok, (str(p[-1]) if p else None)
    return check


def score_item(bench: Bench, item: Item, reply: str, checkers: dict, finished: bool = True) -> dict:
    """An item's score, and how its answer was read. A reply cut off at the token limit is read only where it states
    an answer (its last line or number is then mid-thought)."""
    out = {"lang_ok": lang_ok(reply, bench.lang)}
    if bench.kind in ("mc", "mc_zh"):
        pred, how = read_letter(reply, finished)
        out.update(pred=pred, how=how, correct=int(pred == item.gold))
    elif bench.kind == "gsm8k":
        pred, how = read_number(reply, finished)
        ok = pred is not None and abs(pred - item.gold) <= 1e-6 * max(1.0, abs(item.gold))
        out.update(pred=pred, how=how, correct=int(ok))
    elif bench.kind == "math":
        boxed = last_boxed(reply) is not None
        if boxed or finished:
            ok, pred = checkers["math"](item.gold, reply)
        else:
            ok, pred = False, None
        out.update(pred=pred, how="boxed" if boxed else ("other" if pred else "none"), correct=int(ok))
    elif bench.kind == "ifeval":
        doc = {"key": item.id, "prompt": item.prompt, **item.gold}
        r = checkers["ifeval"](doc, [reply])
        out.update(correct=int(r["prompt_level_strict_acc"]), prompt_strict=int(r["prompt_level_strict_acc"]),
                   prompt_loose=int(r["prompt_level_loose_acc"]),
                   inst_strict=[int(x) for x in r["inst_level_strict_acc"]],
                   inst_loose=[int(x) for x in r["inst_level_loose_acc"]], how="-")
    return out


# --- the models ---------------------------------------------------------------------------------------------------

def resolve(model: str) -> Path:
    """A model folder: the path itself, or the Hub repo's snapshot (downloaded once)."""
    p = Path(model).expanduser()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(model, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        return Path(snapshot_download(model))


def fingerprint(folder: Path, cache_file: Path) -> str:
    """sha256 of the folder's weight files (and config), remembered by (path, size, mtime) so it's hashed once."""
    try:
        known = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        known = {}
    h = hashlib.sha256()
    files = sorted(list(folder.glob("*.safetensors")) + list(folder.glob("*.bin")) + [folder / "config.json"])
    for f in files:
        if not f.exists():
            continue
        st = f.stat()
        k = f"{f.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        if k not in known:
            fh = hashlib.sha256()
            with open(f, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 23), b""):
                    fh.update(chunk)
            known[k] = fh.hexdigest()
        h.update(f"{f.name}:{known[k]}\n".encode())
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(known, indent=1))
    return h.hexdigest()


def slug(model: str, fp: str) -> str:
    name = model.replace("\\", "/").rstrip("/")
    name = name.replace("/", "__") if not Path(model).expanduser().is_dir() else Path(name).name
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}-{fp[:8]}"


def _shared_in_parts(rt, rows, part_tokens: int = 16_000):
    """es_hybrid._shared (the prompts' reading before decoding), the prompts read a few at a time into one set of
    states, so a chunk of many long prompts fits the GPU: the engine then packs its 64 decoding slots from all of
    them, and a chunk costs about as many steps as its longest answer rather than one long answer per few prompts."""
    import torch
    from vibehash import es

    dev = next(rt.model.parameters()).device
    ids, mask = es._left_pad(rows, rt.pad, dev)
    W = ids.shape[1]
    if W < 2:
        raise ValueError("a prompt of one token has nothing to share")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    K = rt.model.config.linear_conv_kernel_dim
    step = max(1, part_tokens // W)
    states = None
    for lo in range(0, len(rows), step):
        hi = min(len(rows), lo + step)
        out = rt.model(input_ids=ids[lo:hi, :W - 1], attention_mask=mask[lo:hi, :W - 1],
                       position_ids=pos[lo:hi, :W - 1], use_cache=True, logits_to_keep=1)
        part = []
        for kind, layer in zip(rt.model.config.layer_types, out.past_key_values.layers):
            if kind == "linear_attention":
                part.append((layer.conv_states[0][..., -(K - 1):], layer.recurrent_states[0]))
            else:
                part.append((layer.keys, layer.values))
        if states is None:
            states = [tuple(torch.empty((len(rows),) + t.shape[1:], dtype=t.dtype, device=t.device) for t in pair)
                      for pair in part]
        for full, pair in zip(states, part):
            for f, t in zip(full, pair):
                f[lo:hi].copy_(t)
        del out, part
    return ids, mask, pos, states


class Generator:
    """A model on the GPU answering prompts greedily, in non-thinking mode, up to T tokens each."""

    def __init__(self, folder: Path, engine: str, batch: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vibehash import es_hybrid

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.model = self.model.cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(folder)
        eos = []
        gen = getattr(self.model, "generation_config", None)
        g_eos = getattr(gen, "eos_token_id", None)
        eos += g_eos if isinstance(g_eos, list) else ([g_eos] if g_eos is not None else [])
        if self.tok.eos_token_id is not None:
            eos.append(self.tok.eos_token_id)
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id and i >= 0:
                eos.append(i)
        self.eos = tuple(dict.fromkeys(int(e) for e in eos))
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]
        hybrid = es_hybrid.is_hybrid(self.model)
        if engine == "fast" and not hybrid:
            sys.exit("--engine fast is es_hybrid's engine, for Qwen3.5-style hybrid models only")
        self.engine = "fast" if engine in ("auto", "fast") and hybrid else "hf"
        self.batch = batch
        if self.engine == "fast":
            from vibehash import es
            es_hybrid._ENGINE.clear()  # another model's engine (its buffers and graphs) must not be reused
            self._was_shared = es_hybrid._shared
            es_hybrid._shared = _shared_in_parts
            self.rt = es.Runtime(self.model, self.tok, "cuda", self.eos, self.pad)
            name, mod = next((n, m) for n, m in self.model.named_modules() if n.endswith("mlp.down_proj"))
            self.linears = [(name, mod)]  # one twisted linear, its twist all zeros: the engine wants at least one
            z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device="cuda")  # noqa: E731
            self.fac = {name: (z(1, mod.out_features, 1), z(1, mod.in_features, 1))}

    def close(self):
        if self.engine == "fast":
            from vibehash import es_hybrid
            es_hybrid._ENGINE.clear()
            es_hybrid._shared = self._was_shared
        del self.model
        self.torch.cuda.empty_cache()

    def ids(self, prompt: str) -> list[int]:
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def answer(self, rows: list[list[int]], T: int, progress: Callable[[int], None]) -> list[list[int]]:
        """Each row's answer tokens (ending at an end-of-turn token, kept, or at T tokens), longest prompts first in
        chunks that fit the GPU."""
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
        out: list = [None] * len(rows)
        cap = 192 if self.engine == "fast" else self.batch  # fast: 192 prompts' states, ~3.6 GB on Qwen3.5-0.8B
        budget = 10**9 if self.engine == "fast" else 16_000  # prompt tokens a chunk reads at once
        at, done = 0, 0
        while at < len(order):
            n = max(1, min(cap, budget // max(1, len(rows[order[at]]))))
            chunk = order[at:at + n]
            at += n
            got = self._fast([rows[i] for i in chunk], T) if self.engine == "fast" else \
                self._hf([rows[i] for i in chunk], T)
            for i, a in zip(chunk, got):
                out[i] = a
            done += len(chunk)
            progress(done)
        return out

    def _fast(self, rows, T):
        from vibehash import es_hybrid
        with self.torch.no_grad():
            return es_hybrid.answer(self.rt, self.linears, self.fac, [(0, 1)], rows, T, 0.0, 1)[0]

    def _hf(self, rows, T):
        torch = self.torch
        W = max(len(r) for r in rows)
        ids = torch.full((len(rows), W), self.pad, dtype=torch.long)
        mask = torch.zeros((len(rows), W), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, W - len(r):] = torch.tensor(r)
            mask[i, W - len(r):] = 1
        with torch.no_grad():
            g = self.model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=T,
                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                    eos_token_id=list(self.eos), pad_token_id=self.pad)
        out = []
        for row in g[:, W:].tolist():
            cut = next((i + 1 for i, t in enumerate(row) if t in self.eos), len(row))
            out.append(row[:cut])
        return out

    def text(self, toks: list[int]) -> str:
        return self.tok.decode([t for t in toks if t not in self.eos], skip_special_tokens=True)


# --- a run --------------------------------------------------------------------------------------------------------

def run_bench(gen: Generator, bench: Bench, items: list[Item], checkers: dict, path: Path) -> list[dict]:
    t0 = time.time()
    rows = [gen.ids(it.prompt) for it in items]
    last = [0.0]

    def progress(done):
        if time.time() - last[0] > 20 or done == len(items):
            last[0] = time.time()
            print(f"    {bench.name}: {done}/{len(items)} answered ({time.time() - t0:.0f}s)", flush=True)

    answers = gen.answer(rows, bench.max_new, progress)
    recs = []
    for it, row, ans in zip(items, rows, answers):
        reply = gen.text(ans)
        rec = {"id": it.id, "prompt_tokens": len(row), "tokens": len(ans),
               "truncated": int(len(ans) >= bench.max_new and (not ans or ans[-1] not in gen.eos)),
               "gold": it.gold if bench.kind != "ifeval" else None, **it.meta}
        rec.update(score_item(bench, it, reply, checkers, not rec["truncated"]))
        rec["prompt"], rec["reply"] = it.prompt, reply
        recs.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {bench.name}: {sum(r['correct'] for r in recs) / len(recs):.1%} of {len(recs)} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return recs


def _ids_hash(items: list[Item]) -> str:
    return hashlib.sha256("\n".join(it.id for it in items).encode()).hexdigest()[:16]


def metrics(bench: str, recs: list[dict]) -> dict:
    """{metric: (per-unit 0/1 list keyed for pairing)}: one unit per item, or per instruction for IFEval's
    instruction-level scores."""
    if bench == "ifeval":
        return {
            "ifeval/prompt_strict": {r["id"]: r["prompt_strict"] for r in recs},
            "ifeval/prompt_loose": {r["id"]: r["prompt_loose"] for r in recs},
            "ifeval/inst_strict": {f"{r['id']}#{k}": v for r in recs for k, v in enumerate(r["inst_strict"])},
            "ifeval/inst_loose": {f"{r['id']}#{k}": v for r in recs for k, v in enumerate(r["inst_loose"])},
        }
    return {bench: {r["id"]: r["correct"] for r in recs}}


def ci95(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    p = sum(xs) / n
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def paired_ci95(d: list[float]) -> float:
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def health(bench: Bench, recs: list[dict]) -> dict:
    n = len(recs)
    lang = [r["lang_ok"] for r in recs if r.get("lang_ok") is not None]
    return {
        "n": n,
        "no_answer": sum(r.get("how") == "none" for r in recs) / n if bench.kind != "ifeval" else None,
        "fallback": sum(r.get("how") in ("last line", "last number", "other") for r in recs) / n
        if bench.kind != "ifeval" else None,
        "truncated": sum(r["truncated"] for r in recs) / n,
        "wrong_lang": (sum(not x for x in lang) / len(lang)) if lang else None,
        "mean_tokens": sum(r["tokens"] for r in recs) / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help="Hugging Face repos or model folders; the first is the reference")
    p.add_argument("--bench", help=f"comma-separated, from: {', '.join(BENCHES)} (default: all)")
    p.add_argument("--limit", type=int, help="items per benchmark (default: each benchmark's own, 400 for most)")
    p.add_argument("--full", action="store_true", help="every item of every benchmark (hours)")
    p.add_argument("--out", default=str(Path.home() / ".vibehash" / "bench"), help="where results go")
    p.add_argument("--engine", choices=("auto", "fast", "hf"), default="auto",
                   help="fast: es_hybrid's engine (Qwen3.5-style models); hf: transformers' generate")
    p.add_argument("--batch", type=int, default=64, help="rows a batch for --engine hf")
    p.add_argument("--max-new", type=int, help="tokens a reply may have, for every benchmark (default: each one's own)")
    p.add_argument("--fresh", action="store_true", help="run again even when saved results match")
    p.add_argument("--list", action="store_true", help="list the benchmarks and exit")
    a = p.parse_args()

    if a.list or not a.models:
        for b in BENCHES.values():
            print(f"{b.name:14s} {b.kind:7s} default {b.limit:4d} items, {b.max_new:4d} tokens  {b.source}")
        return
    names = [s.strip() for s in a.bench.split(",")] if a.bench else list(BENCHES)
    bad = [n for n in names if n not in BENCHES]
    if bad:
        sys.exit(f"unknown benchmark(s): {', '.join(bad)} (have: {', '.join(BENCHES)})")
    out = Path(a.out).expanduser()

    # the items: the same for every model
    sets: dict[str, list[Item]] = {}
    for n in names:
        b = BENCHES[n]
        if a.max_new:
            b.max_new = a.max_new
        try:
            items = b.load()
        except Exception as e:  # noqa: BLE001 - gated, offline or missing: skip it, say why
            msg = str(e).splitlines()[0][:160]
            print(f"note: skipping {n} ({b.source}): {type(e).__name__}: {msg}", file=sys.stderr)
            continue
        sets[n] = subset(n, items, None if a.full else (a.limit or b.limit))
        print(f"{n}: {len(sets[n])} of {len(items)} items ({b.source})", flush=True)
    checkers = {}
    if any(BENCHES[n].kind == "ifeval" for n in sets):
        checkers["ifeval"] = _ifeval_scorer()
    if any(BENCHES[n].kind == "math" for n in sets):
        checkers["math"] = _math_checker()

    results: dict[str, dict[str, list[dict]]] = {}
    engines: dict[str, str] = {}
    labels = []
    for model in a.models:
        folder = resolve(model)
        fp = fingerprint(folder, out / "fingerprints.json")
        s = slug(model, fp)
        label = model if not Path(model).expanduser().is_dir() else Path(model).name
        labels.append(label)
        print(f"\n{label}  ({folder}, weights {fp[:12]})", flush=True)
        gen = None
        results[label] = {}
        for n, items in sets.items():
            b = BENCHES[n]
            path = out / s / f"{n}.jsonl"
            meta_path = out / s / f"{n}.meta.json"
            want = {"prompts_v": PROMPTS_V, "ids": _ids_hash(items), "max_new": b.max_new, "weights": fp,
                    "source": b.source}
            if not a.fresh and path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if all(meta.get(k) == v for k, v in want.items()) and (a.engine == "auto" or meta["engine"] == a.engine):
                    results[label][n] = [json.loads(line) for line in open(path, encoding="utf-8")]
                    engines[f"{label}|{n}"] = meta["engine"]
                    print(f"  {n}: saved results ({meta['engine']} engine, {meta['date']})", flush=True)
                    continue
            if gen is None:
                t = time.time()
                gen = Generator(folder, a.engine, a.batch)
                print(f"  loaded in {time.time() - t:.0f}s; engine: {gen.engine}; stop tokens {gen.eos}", flush=True)
            t = time.time()
            results[label][n] = run_bench(gen, b, items, checkers, path)
            engines[f"{label}|{n}"] = gen.engine
            meta_path.write_text(json.dumps({**want, "model": model, "folder": str(folder), "engine": gen.engine,
                                             "n": len(items), "seconds": round(time.time() - t, 1),
                                             "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        if gen is not None:
            gen.close()

    report(labels, results, engines, sets, out)


def report(labels, results, engines, sets, out: Path) -> None:
    ref = labels[0]
    rows = []
    for n in sets:
        per = {lab: metrics(n, results[lab][n]) for lab in labels}
        for metric in per[ref]:
            row = {"metric": metric, "published": PUBLISHED.get(metric)}
            base = per[ref][metric]
            row["n"] = len(base)
            for lab in labels:
                xs = list(per[lab][metric].values())
                row[lab] = (100 * sum(xs) / len(xs), 100 * ci95(xs), len(xs))
                if lab != ref:
                    common = [k for k in base if k in per[lab][metric]]
                    d = [per[lab][metric][k] - base[k] for k in common]
                    row[f"d:{lab}"] = (100 * sum(d) / len(d), 100 * paired_ci95(d), len(d)) if d else None
            rows.append(row)

    w = max(len(r["metric"]) for r in rows) + 1
    head = f"{'benchmark':{w}s} {'n':>5s}  " + "  ".join(f"{lab[-22:]:>22s}" for lab in labels)
    for lab in labels[1:]:
        head += f"  {'delta ' + lab[-14:]:>22s}"
    head += f"  {'Qwen published':>14s}"
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['metric']:{w}s} {r['n']:5d}  "
        line += "  ".join(f"{r[lab][0]:13.1f} ± {r[lab][1]:4.1f}  " for lab in labels)
        for lab in labels[1:]:
            d = r.get(f"d:{lab}")
            if d is None:
                line += f"  {'-':>22s}"
            else:
                mark = "+" if d[0] - d[1] > 0 else ("-" if d[0] + d[1] < 0 else " ")
                line += f"  {d[0]:+14.1f} ± {d[1]:4.1f} {mark}"
        pub = r["published"]
        line += f"  {pub:14.1f}" if pub is not None else f"  {'':14s}"
        print(line)
    print("(percent right ± a ~95% interval; deltas are paired over the same items: + / - mark ones whose interval "
          "clears zero)")
    mixed = {n for n in sets if len({engines.get(f'{lab}|{n}') for lab in labels}) > 1}
    if mixed:
        print(f"warning: models were scored with different engines on {', '.join(sorted(mixed))}: deltas there mix "
              "engine differences in")

    print(f"\n{'health':{w}s} " + "  ".join(f"{lab[-40:]:>40s}" for lab in labels))
    print(f"{'':{w}s} " + "  ".join(f"{'no-ans  fallbk  trunc  wrong-lang  tokens':>40s}" for _ in labels))
    pct = lambda x: f"{100 * x:5.1f}%" if x is not None else "    - "  # noqa: E731
    for n in sets:
        line = f"{n:{w}s} "
        for lab in labels:
            h = health(BENCHES[n], results[lab][n])
            line += f"  {pct(h['no_answer']):>7s} {pct(h['fallback']):>7s} {pct(h['truncated']):>6s} " \
                    f"{pct(h['wrong_lang']):>11s} {h['mean_tokens']:7.0f}"
        print(line)
    print("(no-ans: no answer could be read; fallbk: read only from a lone letter / the last number; trunc: ran out of "
          "tokens; wrong-lang: English benchmarks answered with CJK, C-Eval not in Chinese)")

    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    f = runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps({"models": labels, "rows": rows,
                             "health": {lab: {n: health(BENCHES[n], results[lab][n]) for n in sets} for lab in labels},
                             "engines": engines}, indent=1))
    print(f"\nsaved: {f}")


if __name__ == "__main__":
    main()
