"""The dataset of the first DeepSeek-V4.1-Flash fine-tune, "think shorter" (docs/datasets/think-shorter.md): GSM8K's
human-written solutions as the model's thinking, in DeepSeek-V4.1-Flash's own chat template, as token ids in the
format `python -m vibehash.deepseek_prep --data` takes.

    python scripts/make_think_shorter.py [--out DIR] [--show 3]

Needs numpy, pyarrow, jinja2 and tokenizers; the job's sizing at the end also needs PyTorch (it reads the split
prep's numbers from scripts/deepseek_split_plan.py) and is skipped without it. Downloads about 9 MB, every file by
pinned revision and checked against its sha256 before it's used: GSM8K's two parquet files (2.7 MB), and the model's
chat_template.jinja and tokenizer.json (6.4 MB; the copy deepseek_prep caches is used if it's there).

Writes to DIR (default ~/.vibehash/datasets/think-shorter, or under $VIBEHASH_HOME), the same bytes on every run:
- think-shorter.npz: GSM8K's 7,473 train problems, then its 1,319 test problems (held back: the scorers' loss), one
  sequence each, in deepseek_prep's format: `tokens` (int32, every sequence end to end), `lengths` (int32, one per
  sequence), `learn` (uint8, 1 where a token is a target). One file, so one prep and one Merkle root serve both.
- think-shorter-pilot.npz: a smaller first job in the same format, PILOT_TRAIN train problems then PILOT_HELDBACK
  test ones, each picked by the sha256 of its question (so the pick depends on nothing but the text).
- manifest.json: the pinned sources, the transformation, which sequences are held back, each file's sha256, and the
  numbers.

Each example, rendered by the template's own Jinja in thinking mode at its default reasoning effort ("high", 75):

    <｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, ...)\\n\\n<｜User｜>{question}
    <｜Assistant｜><think>{solution lines}</think>The answer is **N**.<｜end▁of▁sentence｜>

(no line break before <｜Assistant｜>). The solution lines are GSM8K's, without the calculator annotations (<<48/2=24>>)
and blank lines; N is its "#### N" without thousands commas (718 of the train set's 797 answers over 999 have none).
The rendering
is also checked against encoding/encoding.py's format, which the template was ported from. Learned: everything the
model writes after the prompt's <think>: the thinking, </think>, the answer line and the end-of-sentence token.
"""
import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vibehash.settings import home  # noqa: E402

MODEL, MODEL_COMMIT = "deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6"  # deepseek.py's
GSM8K, GSM8K_REVISION = "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"
HUB = "https://huggingface.co"
# file: (URL, sha256, git blob id or LFS oid as the Hub lists it)
SOURCES = {
    "gsm8k-train.parquet": (f"{HUB}/datasets/{GSM8K}/resolve/{GSM8K_REVISION}/main/train-00000-of-00001.parquet",
                            "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
    "gsm8k-test.parquet": (f"{HUB}/datasets/{GSM8K}/resolve/{GSM8K_REVISION}/main/test-00000-of-00001.parquet",
                           "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "chat_template.jinja": (f"{HUB}/{MODEL}/resolve/{MODEL_COMMIT}/chat_template.jinja",
                            "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3"),
    "tokenizer.json": (f"{HUB}/{MODEL}/resolve/{MODEL_COMMIT}/tokenizer.json",
                       "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"),
}
SIZES = {"gsm8k-train": 7473, "gsm8k-test": 1319}
PILOT_TRAIN, PILOT_HELDBACK = 1500, 500

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
SYSTEM, USER, ASSISTANT, THINK, END_THINK = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"
# encoding.py's REASONING_EFFORT_TEMPLATE at its default, "high": the template puts it first in thinking mode.
EFFORT = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
ANNOTATION = re.compile(r"<<[^<>]*>>")  # GSM8K's calculator calls: none nest, and every one is closed
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # the earliest a zip can say


# --- sources -----------------------------------------------------------------------------------------------------------

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str, folder: Path, also=()) -> bytes:
    """A pinned source's bytes, checked against its sha256: from `also` or `folder` if a copy there checks out,
    otherwise downloaded into `folder`."""
    url, want = SOURCES[name]
    for path in (*also, folder / name):
        if path.is_file() and sha256(data := path.read_bytes()) == want:
            return data
    print(f"fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "vibehash-make-think-shorter"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if sha256(data) != want:
        raise SystemExit(f"{name} from {url} isn't the pinned file (sha256 {sha256(data)}, expected {want})")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)
    return data


def gsm8k(data: bytes, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(io.BytesIO(data)).to_pylist()
    if len(rows) != SIZES[split]:
        raise SystemExit(f"{split} has {len(rows)} rows, not {SIZES[split]}")
    return rows


def compile_template(source: str):
    """The chat template as transformers compiles one (sandboxed Jinja, trim_blocks and lstrip_blocks)."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = tojson
    return env.from_string(source)


# --- one example -------------------------------------------------------------------------------------------------------

def solution(answer: str) -> tuple[str, str]:
    """(thinking, answer line) from a GSM8K answer: its reasoning lines without calculator annotations or blank lines,
    and "The answer is **N**." from its closing "#### N"."""
    if answer.count("####") != 1:
        raise ValueError(f"not one '####' in {answer!r}")
    body, final = answer.split("####")
    lines = (line.strip() for line in ANNOTATION.sub("", body).split("\n"))
    thinking = "\n".join(line for line in lines if line)
    n = final.strip().replace(",", "")
    if not thinking or not re.fullmatch(r"-?\d+", n):
        raise ValueError(f"no reasoning, or a final answer that isn't an integer, in {answer!r}")
    return thinking, f"The answer is **{n}**."


def render(template, question: str, thinking: str, answer: str) -> tuple[str, int]:
    """(the example's text, where the model's part starts): the template's rendering, which must be encoding.py's."""
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "reasoning_content": thinking, "content": answer}]
    text = template.render(messages=messages, add_generation_prompt=False)
    prompt = BOS + SYSTEM + EFFORT + USER + question + ASSISTANT + THINK
    if text != prompt + thinking + END_THINK + answer + EOS:
        raise SystemExit(f"the template renders this example differently from encoding.py's format: {text!r}")
    return text, len(prompt)


class Encoder:
    """Examples to token ids and learn flags with the model's tokenizer."""

    def __init__(self, tokenizer_json: bytes, template_source: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        self.template = compile_template(template_source)
        self.added = sorted(t.content for t in self.tok.get_added_tokens_decoder().values())
        self.think, self.end_think = self.tok.token_to_id(THINK), self.tok.token_to_id(END_THINK)

    def texts(self, rows: list[dict]) -> list[tuple[str, int]]:
        out = []
        for row in rows:
            q = row["question"]
            thinking, answer = solution(row["answer"])
            for s in (q, thinking):
                if any(t in s for t in self.added):  # it would tokenize as a control token
                    raise SystemExit(f"a special token's text is in {s!r}")
            out.append(render(self.template, q, thinking, answer))
        return out

    def encode(self, rows: list[dict]) -> list[dict]:
        """Per example: ids, learn flags (1 from the first token after the prompt's <think> to the end), and counts."""
        texts = self.texts(rows)
        encs = self.tok.encode_batch([t for t, _ in texts], add_special_tokens=False)
        prompts = self.tok.encode_batch([t[:start] for t, start in texts], add_special_tokens=False)
        out = []
        for (text, start), enc, prompt in zip(texts, encs, prompts):
            n = len(prompt.ids)
            # The prompt alone tokenizes as the start of the whole (what a server feeds the model is what it trained
            # on), and the model's part starts on a token boundary.
            if enc.ids[:n] != prompt.ids or enc.offsets[n][0] != start or enc.ids[n - 1] != self.think:
                raise SystemExit(f"the prompt doesn't end on a token boundary at <think> in {text!r}")
            ids = np.asarray(enc.ids, np.int32)
            learn = np.zeros(len(ids), np.uint8)
            learn[n:] = 1
            close = n + enc.ids[n:].index(self.end_think)
            out.append(dict(ids=ids, learn=learn, prompt=n, thinking=close - n, answer=len(ids) - close - 1,
                            lines=text[start:].split(END_THINK)[0].count("\n") + 1))
        return out


# --- output ------------------------------------------------------------------------------------------------------------

def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """An .npz that np.load reads, as np.savez writes it (uncompressed), but the same bytes on every run and every
    machine: np.savez dates each member with the time it's written, and zip records the system it was made on."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, a in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(a), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.create_system, info.external_attr = 3, 0o644 << 16
            z.writestr(info, member.getvalue())
    return buf.getvalue()


def prep_input(examples: list[dict]) -> bytes:
    """deepseek_prep's --data file (its load_dataset and save_dataset)."""
    return npz_bytes({"tokens": np.concatenate([e["ids"] for e in examples]).astype("<i4"),
                      "lengths": np.asarray([len(e["ids"]) for e in examples], "<i4"),
                      "learn": np.concatenate([e["learn"] for e in examples]).astype("u1")})


def stats(examples: list[dict]) -> dict:
    lengths = np.asarray([len(e["ids"]) for e in examples])
    mean = lambda k: round(float(np.mean([e[k] for e in examples])), 1)  # noqa: E731
    return {"examples": len(examples), "tokens": int(lengths.sum()), "learned": int(sum(e["learn"].sum() for e in examples)),
            "tokens_mean": round(float(lengths.mean()), 1), "tokens_median": int(np.median(lengths)),
            "tokens_p95": int(np.percentile(lengths, 95, method="higher")), "tokens_max": int(lengths.max()),
            "prompt_mean": mean("prompt"), "thinking_mean": mean("thinking"),
            "thinking_median": int(np.median([e["thinking"] for e in examples])),
            "thinking_lines_mean": mean("lines"), "answer_mean": mean("answer")}


def pick(rows: list[dict], n: int) -> list[int]:
    """n row indices, the rows whose questions' sha256 come first, in their original order."""
    order = sorted(range(len(rows)), key=lambda i: sha256(rows[i]["question"].encode("utf-8")))
    return sorted(order[:n])


def show(encoder: Encoder, e: dict) -> str:
    """An example decoded, its learned tokens in [[ ]]."""
    ids, n = e["ids"].tolist(), e["prompt"]
    return (encoder.tok.decode(ids[:n], skip_special_tokens=False) + "[["
            + encoder.tok.decode(ids[n:], skip_special_tokens=False) + "]]")


# --- the job's size ----------------------------------------------------------------------------------------------------

def sizing(name: str, s: dict, train: dict, longest: int) -> str | None:
    """The prep and training traffic for one prep input, from scripts/deepseek_split_plan.py's model (DEEPSEEK.md §8):
    pieces of 4 layers, cut at 36. None without PyTorch."""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("deepseek_split_plan", ROOT / "scripts" / "deepseek_split_plan.py")
        plan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan)
    except ImportError:
        return None
    from vibehash import deepseek_split as split
    from vibehash import deepseek_top as top
    from vibehash.deepseek import Config

    cfg = Config.flash()
    cut, g, tokens, seq = top.Cut(cfg, 36), 4, s["tokens"], longest
    record = cut.bytes_per_token(seq_len=seq)["total"]
    record_fp8 = cut.bytes_per_token(fp8_stream=True, seq_len=seq)["total"]
    stored, packed = plan.traffic(cut, g, tokens, seq, packed=False), plan.traffic(cut, g, tokens, seq, packed=True)
    size = plan.sizes(cfg, seq)
    graph = split.plan(cut, [[0]], split.pieces(cut.first, g))
    cuts = len(split.pieces(cut.first, g)) - 1
    gb = tokens / 1e9
    lines = [f"### {name}: {tokens:,} tokens ({s['examples']:,} sequences)", "",
             f"- Records (the trainers' data, cut at 36): {record:,.0f} B a token, **{record * gb:,.1f} GB** "
             f"({record_fp8 * gb:,.1f} GB with an FP8 stream)",
             f"- Hand-offs at G = 4 ({cuts} cuts): {stored['egress']:,.1f} GB of bucket egress as stored "
             f"(42.3 KB a token a cut), {packed['egress']:,.1f} GB compressed losslessly (the design's 28 KB), "
             f"{22.4e3 * cuts * gb:,.1f} GB at FP8's 22.4 KB (FP8 hand-offs' quality unmeasured)",
             f"- Miners' upload, records included: {packed['upload']:,.1f} GB",
             "", "| piece | weights (once, cached across jobs) | download for this job | upload |", "|---|---|---|---|"]
    for a, b in split.pieces(cut.first, g):
        unit = graph.units[f"layers-{a:02d}-{b:02d}-c00000"]
        down = sum(plan.object_bytes(i, cfg, size, True) for i in unit.inputs if not i.endswith(".json"))
        down += 4 if a == 0 else 0  # the token ids
        up = record if b == cut.first else sum(plan.object_bytes(o, cfg, size, True) for o in unit.outputs
                                               if not o.endswith(".json") and not o.startswith("block-"))
        weights = (b - a) * plan.LAYER_GB + (plan.EMBED_GB if a == 0 else 0)
        lines.append(f"| layers {a}-{b - 1} | {weights:.1f} GB | {down * gb:,.1f} GB | {up * gb:,.1f} GB |")
    rows_up = plan.ROWS_PER_TOKEN * tokens / 15 / 1e9
    lines += ["", f"- Rows units: 30 (15 a table), each streaming {plan.TABLES_GB / 30:.1f} GB of table from the Hub "
              f"(the whole dataset's rows in one pass, so the same for any size) and uploading {rows_up:.2f} GB",
              f"- A trainer: the top 4 layers and the head once (30.9 GB), then its shard of the records: "
              f"{record * gb / 6:,.1f} GB with 6 trainers ({record_fp8 * gb / 6:,.1f} with FP8)"]
    cable = plan.LINKS["cable, 100/20 Mbit/s"]
    for n in (9, 18):
        at_once = plan.simulate(cut, g, tokens, 65536, n, cable, seq)
        cached = plan.simulate(cut, g, tokens, 65536, n, cable, seq, cached=True)
        lines.append(f"- Prep end to end with {n} miners on 100/20 Mbit/s cable: {at_once:.1f} h "
                     f"({cached:.1f} h with layers already cached)")
    if train is not None:
        lines.append(f"- Training, an epoch of the {train['examples']:,} train sequences ({train['tokens']:,} tokens): "
                     f"{train['tokens'] * 4 * 0.6 / 3.6e6:.2f}-{train['tokens'] * 4 * 0.75 / 3.6e6:.2f} GPU-hours on "
                     "4070 SUPERs (estimate: §7's 0.6-0.75 ms a token a layer, in bf16; float32 unmeasured)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=home() / "datasets" / "think-shorter")
    ap.add_argument("--show", type=int, default=0, help="print this many decoded examples, learned tokens in [[ ]]")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the special tokens' ｜ and ▁, on a Windows console too
    out, src = args.out, args.out / "src"
    out.mkdir(parents=True, exist_ok=True)

    cached_tokenizer = home() / "cache" / "deepseek" / MODEL_COMMIT / "reference" / "tokenizer.json"
    encoder = Encoder(fetch("tokenizer.json", src, also=[cached_tokenizer]),
                      fetch("chat_template.jinja", src).decode("utf-8"))
    rows = {s: gsm8k(fetch(f"{s}.parquet", src), s) for s in SIZES}
    train, test = encoder.encode(rows["gsm8k-train"]), encoder.encode(rows["gsm8k-test"])
    pilot_train, pilot_test = pick(rows["gsm8k-train"], PILOT_TRAIN), pick(rows["gsm8k-test"], PILOT_HELDBACK)

    files = {"think-shorter.npz": (train + test, len(train), None),
             "think-shorter-pilot.npz": ([train[i] for i in pilot_train] + [test[i] for i in pilot_test],
                                         len(pilot_train), {"train": pilot_train, "heldback": pilot_test})}
    manifest = {
        "format": "vibehash-think-shorter/1",
        "input_for": "python -m vibehash.deepseek_prep --data FILE: tokens (int32, every sequence end to end), "
                     "lengths (int32, one per sequence), learn (uint8, 1 where a token is a target)",
        "model": {"repo": MODEL, "commit": MODEL_COMMIT, "thinking_mode": "thinking", "reasoning_effort": "high (75)",
                  "files": {k: SOURCES[k][1] for k in ("chat_template.jinja", "tokenizer.json")}},
        "source": {"repo": GSM8K, "config": "main", "revision": GSM8K_REVISION, "license": "MIT",
                   "citation": "Cobbe et al., Training Verifiers to Solve Math Word Problems, arXiv:2110.14168 (2021)",
                   "files": {k: SOURCES[k][1] for k in ("gsm8k-train.parquet", "gsm8k-test.parquet")}},
        "transformation": "user: the question as is. assistant: <think> + the answer's lines without <<...>> "
                          "calculator annotations or blank lines, each stripped + </think> + 'The answer is **N**.', "
                          "N the '#### N' without commas. Learned: every token after the prompt's <think>, up to and "
                          "including <｜end▁of▁sentence｜>. One example a sequence, no packing.",
        "files": {},
    }
    for name, (examples, n_train, picked) in files.items():
        data = prep_input(examples)
        (out / name).write_bytes(data)
        entry = {"sha256": sha256(data), "bytes": len(data), "sequences": len(examples),
                 "train": [0, n_train], "heldback": [n_train, len(examples)],
                 "stats": {"train": stats(examples[:n_train]), "heldback": stats(examples[n_train:]),
                           "all": stats(examples)}}
        if picked is not None:
            entry["gsm8k_rows"] = picked
        manifest["files"][name] = entry
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    (out / "manifest.json").write_bytes(text.encode("utf-8"))

    print(f"wrote {out}")
    for name, entry in manifest["files"].items():
        print(f"\n{name}: sha256 {entry['sha256']}, {entry['bytes']:,} bytes")
        print("| | examples | tokens | learned | mean a sequence | median / p95 / max | prompt | thinking (median) "
              "| thinking lines | answer + EOS |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for part in ("train", "heldback", "all"):
            s = entry["stats"][part]
            print(f"| {part} | {s['examples']:,} | {s['tokens']:,} | {s['learned']:,} | {s['tokens_mean']} | "
                  f"{s['tokens_median']} / {s['tokens_p95']} / {s['tokens_max']} | {s['prompt_mean']} | "
                  f"{s['thinking_mean']} ({s['thinking_median']}) | {s['thinking_lines_mean']} | {s['answer_mean']} |")
    print(f"manifest.json: sha256 {sha256(text.encode('utf-8'))}")
    for name, entry in manifest["files"].items():
        s = entry["stats"]
        report = sizing(name, s["all"], s["train"], s["all"]["tokens_max"])
        if report is None:
            print("\n(no PyTorch: the job's sizing is skipped)")
            break
        print("\n" + report)
    for e in train[:args.show]:
        print("\n" + show(encoder, e))


if __name__ == "__main__":
    main()
