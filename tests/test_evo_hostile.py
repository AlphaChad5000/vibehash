"""What one worker's answers can do to the node that re-scores them: nothing slow and nothing that escapes.

A node re-scores every Evo claim from its answers blob (es.rescore, es2_rounds.rescore) with the coordinator's lock held,
so a check that takes long on some text, or an error that isn't the "not a unit's answers" kind, is a way for one worker
to stop the network. The answer readers must also give exactly what they gave before: a worker scores its own answers
with them, and a node that scores them differently slashes it.
"""
import json
import random
import re
import time
import unittest

from vibehash import es, es2_rounds, es_tasks

OLD_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)  # the readers as they were, to compare against
OLD_LETTER_SAID = re.compile(r"\banswer(?:\s+is)?\s*[:\-]?\s*\(?([A-H])\b", re.I)
PIECES = ["<answer>", "</answer>", "<answer", "answer>", "answer", "Answer", "ANSWER", " is", "is", " ", "  ", "\n",
          "\t", ":", "-", "(", ")", "A", "B", "H", "I", "a", "b", "x", "1", "2.5", "the ", "<", ">", "/", "é", "\\boxed{"]


def texts(n: int, seed: int = 7):
    rng = random.Random(seed)
    for _ in range(n):
        yield "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 40)))


class SameAnswersTests(unittest.TestCase):
    def test_tagged_is_findall(self):
        for t in texts(30_000):
            self.assertEqual(es.tagged(t), OLD_ANSWER.findall(t), repr(t))
        self.assertEqual(es.tagged("<answer>a</answer><answer>b"), ["a"])
        self.assertEqual(es.tagged("<answer><answer>x</answer>"), ["<answer>x"])
        self.assertEqual(es.tagged("<answer></answer>"), [""])
        self.assertEqual(es.tagged("<answer>\n1\n</answer> and <answer>2</answer>"), ["\n1\n", "2"])

    def test_the_letter_reader_reads_what_it_did(self):
        for t in texts(30_000, seed=8):
            self.assertEqual([m.group(1) for m in es_tasks._LETTER_SAID.finditer(t)],
                             [m.group(1) for m in OLD_LETTER_SAID.finditer(t)], repr(t))
        for t in ("the answer is B", "Answer: (C)", "answer - d", "answer  is  :  (h)", "answer:B", "answers B"):
            self.assertEqual([m.group(1) for m in es_tasks._LETTER_SAID.finditer(t)],
                             [m.group(1) for m in OLD_LETTER_SAID.finditer(t)], t)


class HostileTextTests(unittest.TestCase):
    """Texts that took the old readers seconds at 32-64 KB (and hours at a blob's 8 MB): now each is quick at 1 MB."""
    N = 1_000_000

    def quick(self, f, text, limit=1.0):
        t0 = time.perf_counter()
        f(text)
        self.assertLess(time.perf_counter() - t0, limit, f"{f.__name__} on {text[:30]!r}...")

    def test_answer_readers(self):
        for text in ("<answer>" * (self.N // 8), "<answer>1" * (self.N // 9), "answer" + " " * self.N + "B",
                     "answer " * (self.N // 7) + "B", ("answer:" + " " * 60) * (self.N // 67) + "(B"):
            self.quick(es_tasks.last_answer, text)
            self.quick(lambda t: es_tasks.salvage(t, "B"), text)
            self.quick(lambda t: es_tasks.salvage(t, "12"), text)
            self.quick(lambda t: es.reward(es.Problem((1, 2, 3), 6, "1+2+3"), t), text)


class NestedBlobTests(unittest.TestCase):
    """JSON nested deeper than the parser goes raises RecursionError, which is a RuntimeError: rescore let it out, and
    the node's reveal handler had already put the reveal in the round's record, with no result for it."""
    DEEP = b"[" * 200_000

    def test_the_parser_gives_up_with_a_recursion_error(self):
        with self.assertRaises(RecursionError):
            json.loads(self.DEEP)

    def test_rescore_says_not_a_units_answers(self):
        from tests.test_es import spec_for
        self.assertIsNone(es.rescore(spec_for(), 1, 1, self.DEEP, "u1"))
        self.assertIsNone(es.rescore(spec_for(), 1, 1, b'{"unit": "u1", "answers": ' + self.DEEP + b"}", "u1"))
        spec2 = type("Spec", (), {"members": 2, "problems": 2, "max_tokens": 8})()
        self.assertIsNone(es2_rounds.rescore(spec2, 1, self.DEEP, "u1"))


if __name__ == "__main__":
    unittest.main()
