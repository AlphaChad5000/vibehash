"""scripts/make_think_shorter.py: the "think shorter" dataset's builder (docs/datasets/think-shorter.md).

- SolutionTests: a GSM8K answer becomes its reasoning lines, without calculator annotations or blank lines, and
  "The answer is **N**." from its "#### N" (commas dropped); malformed answers are refused.
- FileTests: the prep input is the same bytes every time, and deepseek_prep reads it (needs PyTorch for that part).
- TemplateTests (needs jinja2, tokenizers and the pinned template and tokenizer already fetched, as the builder
  leaves them): an example renders as encoding.py's format, and only the model's part is learned.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_think_shorter", ROOT / "scripts" / "make_think_shorter.py")
mts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mts)

try:
    import torch
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise ImportError
    from vibehash import deepseek_prep as prep
except ImportError:
    torch = None

ANSWER = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n\nShe sold 1,000+24 = <<1000+24=1024>>1,024 clips "
          "because 24 < 48.\n#### 1,024")


class SolutionTests(unittest.TestCase):
    def test_a_gsm8k_answer(self):
        thinking, answer = mts.solution(ANSWER)
        self.assertEqual(thinking, "Natalia sold 48/2 = 24 clips in May.\nShe sold 1,000+24 = 1,024 clips because 24 < 48.")
        self.assertEqual(answer, "The answer is **1024**.")
        self.assertEqual(mts.solution("Down 3.\n#### -3")[1], "The answer is **-3**.")

    def test_malformed_answers_are_refused(self):
        for bad in ("no final line", "a\n#### 1\n#### 2", "a\n#### 2.5", "<<1+1=2>>\n#### 2", "a\n#### "):
            with self.assertRaises(ValueError, msg=bad):
                mts.solution(bad)


class FileTests(unittest.TestCase):
    def examples(self):
        return [dict(ids=np.array([0, 5, 6, 7, 1], np.int32), learn=np.array([0, 0, 1, 1, 1], np.uint8)),
                dict(ids=np.array([0, 9, 1], np.int32), learn=np.array([0, 0, 1], np.uint8))]

    def test_the_same_bytes_every_time(self):
        a = mts.prep_input(self.examples())
        self.assertEqual(a, mts.prep_input(self.examples()))
        import io
        with np.load(io.BytesIO(a)) as z:
            self.assertEqual(z["tokens"].tolist(), [0, 5, 6, 7, 1, 0, 9, 1])
            self.assertEqual(z["lengths"].tolist(), [5, 3])
            self.assertEqual(z["learn"].tolist(), [0, 0, 1, 1, 1, 0, 0, 1])
            self.assertEqual((z["tokens"].dtype, z["lengths"].dtype, z["learn"].dtype),
                             (np.dtype("<i4"), np.dtype("<i4"), np.dtype("u1")))

    @unittest.skipIf(torch is None, "needs PyTorch 2.8 or later")
    def test_deepseek_prep_reads_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.npz"
            path.write_bytes(mts.prep_input(self.examples()))
            seqs = prep.load_dataset(path)
        self.assertEqual([ids.tolist() for ids, _ in seqs], [[0, 5, 6, 7, 1], [0, 9, 1]])
        # position t's next token is learned: the model's part is predicted from the token before it on
        self.assertEqual([m.tolist() for _, m in seqs], [[False, True, True, True, False], [False, True, False]])


def _local_sources():
    if importlib.util.find_spec("jinja2") is None or importlib.util.find_spec("tokenizers") is None:
        return None
    src = mts.home() / "datasets" / "think-shorter" / "src"
    cached = mts.home() / "cache" / "deepseek" / mts.MODEL_COMMIT / "reference" / "tokenizer.json"
    found = {}
    for name, candidates in (("tokenizer.json", (cached, src / "tokenizer.json")),
                             ("chat_template.jinja", (src / "chat_template.jinja",))):
        for path in candidates:
            if path.is_file() and mts.sha256(path.read_bytes()) == mts.SOURCES[name][1]:
                found[name] = path.read_bytes()
                break
        else:
            return None
    return found


@unittest.skipIf(_local_sources() is None, "needs jinja2, tokenizers and the pinned template and tokenizer, fetched")
class TemplateTests(unittest.TestCase):
    def test_an_example(self):
        found = _local_sources()
        enc = mts.Encoder(found["tokenizer.json"], found["chat_template.jinja"].decode("utf-8"))
        e = enc.encode([{"question": "What is 2 + 2?", "answer": "2 + 2 = <<2+2=4>>4\n#### 4"}])[0]
        text = enc.tok.decode(e["ids"].tolist(), skip_special_tokens=False)
        self.assertEqual(text, "<｜begin▁of▁sentence｜><｜System｜>" + mts.EFFORT + "<｜User｜>What is 2 + 2?"
                         "<｜Assistant｜><think>2 + 2 = 4</think>The answer is **4**.<｜end▁of▁sentence｜>")
        n = e["prompt"]
        self.assertEqual(e["ids"][n - 1], enc.tok.token_to_id("<think>"))
        self.assertEqual(e["learn"].tolist(), [0] * n + [1] * (len(e["ids"]) - n))
        self.assertEqual(enc.tok.decode(e["ids"][n:].tolist(), skip_special_tokens=False),
                         "2 + 2 = 4</think>The answer is **4**.<｜end▁of▁sentence｜>")
        self.assertEqual(e["thinking"], len(enc.tok.encode("2 + 2 = 4", add_special_tokens=False).ids))

    def test_special_token_text_is_refused(self):
        found = _local_sources()
        enc = mts.Encoder(found["tokenizer.json"], found["chat_template.jinja"].decode("utf-8"))
        with self.assertRaises(SystemExit):
            enc.encode([{"question": "Say </think> please", "answer": "ok\n#### 1"}])


if __name__ == "__main__":
    unittest.main()
