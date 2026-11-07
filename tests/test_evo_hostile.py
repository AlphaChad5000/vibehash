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

from vibehash import es, es2_rounds, es_ifeval, es_tasks

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

    def test_ifeval_checks(self):
        """`\\[.*?\\]`, `^\\s*\\*...` and `<<[^\\n]+>>` took 4 s at 40 KB of "[", "\\n" or "<" (hours at 8 MB)."""
        for text in ("[" * self.N, "<" * self.N, "<<" * (self.N // 2), "\n" * self.N, " \n" * (self.N // 2),
                     "\t" * self.N, "[]" * (self.N // 2), "<<x" * (self.N // 3)):
            self.quick(lambda t: es_ifeval.placeholders(t, {"N": 3}), text)
            self.quick(lambda t: es_ifeval.bullet_points(t, {"N": 3}), text)
            self.quick(lambda t: es_ifeval.title(t, {}), text)


OLD_PLACEHOLDER = re.compile(r"\[.*?\]")  # es_ifeval's checks as they were, to compare against
OLD_BULLETS = (re.compile(r"^\s*\*[^\*].*$", re.M), re.compile(r"^\s*-.*$", re.M))
OLD_TITLE = re.compile(r"<<[^\n]+>>")
IFEVAL_PIECES = ["[", "]", "[x]", "<", ">", "<<", ">>", "<<a>>", "*", "**", "* ", "-", "- ", " ", "\t", "\r", "\n",
                 "\n\n", "\x0b", " ", "\x85", "a", "word ", "é"]


class SameIfevalTests(unittest.TestCase):
    """The ifeval checks count what the benchmark's regexes count: a worker scores its own answers with them, and a node
    that scores them differently slashes it."""

    def test_on_random_texts(self):
        rng = random.Random(11)
        for _ in range(60_000):
            t = "".join(rng.choice(IFEVAL_PIECES) for _ in range(rng.randint(0, 30)))
            n = len(OLD_PLACEHOLDER.findall(t))
            self.assertTrue(es_ifeval.placeholders(t, {"N": n}), repr(t))
            self.assertFalse(es_ifeval.placeholders(t, {"N": n + 1}), repr(t))
            bullets = sum(len(r.findall(t)) for r in OLD_BULLETS)
            self.assertTrue(es_ifeval.bullet_points(t, {"N": bullets}), repr(t))
            self.assertFalse(es_ifeval.bullet_points(t, {"N": bullets + 1}), repr(t))
            old_title = any(m.lstrip("<").rstrip(">").strip() for m in OLD_TITLE.findall(t))
            self.assertEqual(es_ifeval.title(t, {}), old_title, repr(t))

    def test_known_texts(self):
        self.assertTrue(es_ifeval.title("<<My Title>>\nbody", {}))
        self.assertFalse(es_ifeval.title("<<>>\n<<<>>", {}))
        self.assertTrue(es_ifeval.title("<<>> and <<<>>", {}))  # ">> and <<<" is something
        self.assertTrue(es_ifeval.title("<<>> then <<x>>", {}))  # one match, first "<<" to last ">>": ">> then <<x"
        self.assertFalse(es_ifeval.title("<<\nx>>", {}))
        self.assertTrue(es_ifeval.placeholders("[a] [b]\n[c", {"N": 2}))
        self.assertFalse(es_ifeval.placeholders("[a] [b]\n[c", {"N": 3}))
        self.assertTrue(es_ifeval.bullet_points("* one\n  * two\n- three\n**bold**", {"N": 3}))
        self.assertTrue(es_ifeval.bullet_points("\n\n  \n* one", {"N": 1}))


class BadTokenIdTests(unittest.TestCase):
    """An accepted member's right answer can become a replay target, which every later round scores, and checkers sample
    only some of a claim's members: ids the model doesn't have (-1, 10**12) or none at all in an unsampled member made a
    target that ended every worker's and checker's scoring, every round after (IndexError, or a device-side assert)."""

    def spec(self):
        from tests.test_es import spec_for
        from vibehash import es2
        rule1_only = ("rule", "pairs", "pairs_per_unit", "problems", "max_tokens", "alpha", "reward", "level",
                      "retarget", "optimizer", "betas", "eps")
        base = {k: v for k, v in spec_for().__dict__.items() if k not in rule1_only}
        return es2.Spec2(**(base | dict(pairs=4, pairs_per_unit=2, problems=4, max_tokens=8, rounds=3)))

    def blob(self, spec, bad=None):
        answers = [[{"ids": [5, 6], "text": "<answer>1</answer>"} for _ in range(spec.problems)]
                   for _ in range(spec.members)]
        if bad is not None:
            answers[3][1] = {"ids": bad, "text": "<answer>1</answer>"}
        return es.claim_bytes("u1", answers)

    def test_rescore_refuses_ids_no_model_has(self):
        spec = self.spec()
        self.assertIsNotNone(es2_rounds.rescore(spec, 1, self.blob(spec), "u1"))
        self.assertIsNotNone(es2_rounds.rescore(spec, 1, self.blob(spec, [0, 259]), "u1"))
        # Past the vocabulary is not a fraud by itself (a spec might say too few): it just never becomes a target.
        self.assertIsNotNone(es2_rounds.rescore(spec, 1, self.blob(spec, [260]), "u1"))
        for bad in ([-1], [], [True], [1.0]):
            self.assertIsNone(es2_rounds.rescore(spec, 1, self.blob(spec, bad), "u1"), bad)

    def test_targets_hold_only_ids_the_model_has(self):
        from vibehash import es2
        right = [[True, True], [True, True], [True, True]]
        for bad in ([-1], [260], [10**12], []):
            answers = [[bad, [5, 6, 7]], [bad, [5, 6, 7]], [[9, 9], [5, 6, 7]]]  # two votes for the bad one, and shorter
            self.assertEqual(es2.replay_targets(2, answers, right, vocab=260), [(0, [9, 9]), (1, [5, 6, 7])], bad)
            alone = [[bad, [5]], [bad, [5]], [bad, [5]]]
            self.assertEqual(es2.replay_targets(2, alone, right, vocab=260), [(1, [5])], bad)
        answers = [[[4], [5]], [[4], [5]], [[3, 3], [5]]]
        self.assertEqual(es2.replay_targets(2, answers, right), [(0, [4]), (1, [5])], "as it was, otherwise")


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
