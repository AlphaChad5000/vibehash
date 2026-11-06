"""The tasks added for the Qwen3.5-0.8B Evo job (es_tasks.VIBE): ifeval's constraints (es_ifeval) one by one, gsm8k's
and math's reasoned answers (math_norm, last_boxed), medmcqa's multiple choice, and their items drawn the same way every
time. No network, no GPU: rows are made up, and HubTask is given them instead of a download."""
import json
import unittest
from unittest import mock

from vibehash import es2, es2_rounds, es_ifeval as ife, es_tasks as T


def gt(func_name, **args):
    """A ground truth as RLVR-IFeval writes it: every argument named, the unused ones null."""
    keys = ("N", "quantifier", "end_phrase", "keyword_list", "word", "forbidden_words", "letter", "i", "first_word",
            "postscript_marker", "options", "section_splitter", "original_prompt")
    return json.dumps({"func_name": func_name, **{k: args.get(k) for k in keys}})


class IfevalConstraintTests(unittest.TestCase):
    """Each constraint on a reply that keeps it and one that doesn't."""

    def check(self, func_name, right, wrong, **args):
        ok = ife.constraint(gt(func_name, **args))
        self.assertIsNotNone(ok, func_name)
        for r in ([right] if isinstance(right, str) else right):
            self.assertTrue(ok(r), f"{func_name} should take {r!r}")
        for w in ([wrong] if isinstance(wrong, str) else wrong):
            self.assertFalse(ok(w), f"{func_name} should refuse {w!r}")

    def test_keywords(self):
        self.check("verify_keywords", "The KEVEL was reunitable.", "Only the kevel.",
                   keyword_list=["kevel", "reunitable"])

    def test_keyword_frequency(self):
        self.check("verify_keyword_frequency", "Cat, cat and a dog. CAT!", ["cat cat", "cat cat cat cat", "category"],
                   word="cat", N=3)

    def test_forbidden_words(self):
        self.check("validate_forbidden_words", "A category of things.", ["The cat sat.", "SECRETLY so"],
                   forbidden_words=["cat", "secretly"])

    def test_letter_frequency(self):
        self.check("verify_letter_frequency", "Dad did", ["dad", "Dad did d"], letter="d", N=4)

    def test_paragraph_count(self):
        self.check("verify_paragraph_count", ["One.\n* * *\nTwo.\n* * *\nThree.", "One.\n***\nTwo.\n***\nThree.\n"],
                   ["One.\n* * *\nTwo.", "One.\n* * *\n\n* * *\nThree.\n* * *\nFour."], N=3)

    def test_word_constraint(self):
        self.check("validate_word_constraint", "one two three", "one two three four", N=3, quantifier="at most")
        self.check("validate_word_constraint", "one two three four", "one two", N=3, quantifier="at least")
        self.check("validate_word_constraint", ["one two three", "one two"], "one", N=3, quantifier="around")

    def test_sentence_constraint(self):
        self.check("verify_sentence_constraint", "It rains. We stay! Do you? ", "It rains. We stay.", N=3,
                   quantifier="at least")
        self.check("verify_sentence_constraint", "Dr. Who arrived. He left.", "A. B. C. D.", N=2, quantifier="at most")

    def test_paragraphs_with_a_first_word(self):
        self.check("validate_paragraphs", ["Hi there.\n\nSynodalist views, mostly.\n\nEnd.",
                                           'Hi.\n\n"synodalist, first.\n\nEnd.'],
                   ["Hi.\n\nOther start.\n\nEnd.", "Hi.\n\nSynodalist.", "Hi.\n\nSynodalists.\n\nEnd."],
                   N=3, i=2, first_word="synodalist")

    def test_postscript(self):
        self.check("verify_postscript", "Body text.\n\nPS: see you.", ["Body text.", "Body. PS:   "],
                   postscript_marker="PS:")

    def test_placeholders(self):
        self.check("validate_placeholders", "Hi [name], at [address].", "Hi [name].", N=2)

    def test_bullet_points(self):
        self.check("verify_bullet_points", "* one\n* two\n- three",
                   ["* one\n* two", "* one\n**bold**\n- two\n- x\n- y"], N=3)

    def test_title(self):
        self.check("validate_title", "<<Poem of joy>>\nLines.", ["<< >> no", "No title", "<<two\nlines>>"])

    def test_choice(self):
        self.check("validate_choice", ["Yes.", "maybe, since", "No"], ["Yesterday it rained", "I'd say yes"],
                   options="yes/no/maybe")
        self.check("validate_choice", ["I don't know.", "I know"], "I think I know", options="I know or I don't know")
        self.check("validate_choice", "b) because", ["e) none", "The answer is a)"], options="a), b), c), d)")

    def test_highlighted_sections(self):
        self.check("validate_highlighted_sections", "This *is* one and **this** two.", "Only *one* here.", N=2)

    def test_sections(self):
        self.check("validate_sections", "Section: 1\nA\nSection: 2\nB", ["Section: 1\nA", "Section 1\nA\nSection 2\nB"],
                   N=2, section_splitter="Section:")

    def test_json_format(self):
        self.check("validate_json_format", ['{"a": 1}', '```json\n{"a": [1, 2]}\n```'], ["{'a': 1}", "Here: {}"])

    def test_repeat_prompt(self):
        prompt = "Find the median of 4, 8, 10.\n\n" + ife.REPEAT
        self.check("validate_repeat_prompt", ["find the median of 4, 8, 10. The median is 8.",
                                              "Find the median of 4,\n8, 10.\nIt is 8."],
                   ["The median is 8.", "Find the median."], original_prompt=prompt)
        self.check("validate_repeat_prompt", "Find the median of 4, 8, 10. 8", "8",
                   original_prompt=ife.REPEAT + " Find the median of 4, 8, 10.")

    def test_two_responses(self):
        self.check("validate_two_responses", "First one.\n******\nSecond one.",
                   ["Same.\n******\nSame.", "One\n******\nTwo\n******\nThree", "Only one"])

    def test_cases(self):
        self.check("validate_uppercase", "HELLO, WORLD 42.", ["Hello", "1234"])
        self.check("validate_lowercase", "hello, world 42.", ["Hello", "1234"])

    def test_capital_words(self):
        self.check("validate_frequency_capital_words", "THIS IS big", "THIS is big", N=2, quantifier="at least")
        self.check("validate_frequency_capital_words", "THIS is big", "THIS IS BIG", N=2, quantifier="at most")

    def test_end(self):
        self.check("validate_end", ["We're done. That's all.", "done. that's all.\n"], "That's all. More.",
                   end_phrase="That's all.")

    def test_quotation(self):
        self.check("validate_quotation", '"Wrapped reply."', ['Not "wrapped"', '"'])

    def test_no_commas(self):
        self.check("validate_no_commas", "No commas here.", "One, two.")

    def test_rows_left_out(self):
        self.assertIsNone(ife.constraint(gt("validate_response_language", language="kn")), "needs langdetect")
        self.assertIsNone(ife.constraint(gt("verify_paragraph_count", N=39)), "39 paragraphs won't fit")
        self.assertIsNotNone(ife.constraint(gt("validate_word_constraint", N=39, quantifier="at least")))
        self.assertIsNotNone(ife.constraint(gt("verify_sentence_constraint", N=39, quantifier="at most")))
        self.assertIsNone(ife.constraint(gt("verify_sentence_constraint", N=39, quantifier="around")))
        self.assertIsNone(ife.constraint(gt("validate_paragraphs", N=3, i=4, first_word="x")))
        self.assertIsNone(ife.constraint(gt("validate_word_constraint", N=3, quantifier="exactly")))
        self.assertIsNone(ife.constraint(gt("verify_keywords", keyword_list=[])))
        self.assertIsNone(ife.constraint(gt("validate_choice", options="yes")))
        self.assertIsNone(ife.constraint("not json"))
        self.assertIsNone(ife.constraint(gt("validate_repeat_prompt", original_prompt=ife.REPEAT)))
        self.assertEqual(len(ife.CHECKS), 24, "RLVR-IFeval's 25 constraints but the language one")

    def test_checks_never_raise(self):
        for name in ife.CHECKS:
            args = dict(N=2, quantifier="at least", end_phrase="x", keyword_list=["x"], word="x",
                        forbidden_words=["x"], letter="x", i=1, first_word="x", postscript_marker="x",
                        options="a/b", section_splitter="x", original_prompt="x")
            ok = ife.constraint(gt(name, **args))
            for reply in ("", "\n\n", "[" * 5000, "{" * 100_000, None):
                self.assertIn(ok(reply), (True, False), name)


class IfevalItemTests(unittest.TestCase):
    def test_a_row_makes_an_item_checked_on_the_whole_reply(self):
        row = {"messages": [{"role": "user", "content": "Describe IPv6. Your entire response should be in all "
                                                        "lowercase letters."}],
               "ground_truth": gt("validate_lowercase")}
        it = T._ifeval(row)
        self.assertEqual(it.messages, row["messages"])
        self.assertEqual(it.gold, "")
        self.assertTrue(it.check("ipv6 is the newest internet protocol."))
        self.assertFalse(it.check("IPv6 is the newest internet protocol."))
        loose = T.lenient(it)  # the same check: there are no tags to read
        self.assertTrue(loose.check("ipv6 is the newest internet protocol."))
        self.assertFalse(loose.check("<answer>IPv6</answer>"))

    def test_rows_left_out(self):
        long = {"messages": [{"role": "user", "content": "x" * (T.MAX_REASON + 1)}],
                "ground_truth": gt("validate_title")}
        self.assertIsNone(T._ifeval(long))
        lang = {"messages": [{"role": "user", "content": "Hi"}], "ground_truth": gt("validate_response_language")}
        self.assertIsNone(T._ifeval(lang))
        two = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey"}],
               "ground_truth": gt("validate_title")}
        self.assertIsNone(T._ifeval(two))


class Gsm8kTests(unittest.TestCase):
    row = {"question": "Natalia sold 48 clips in April and half as many in May. How many in all?",
           "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nIn all 48+24 = <<48+24=72>>72.\n#### 72"}

    def test_the_item(self):
        it = T._gsm8k(self.row)
        self.assertEqual(it.gold, "72")
        self.assertTrue(it.messages[0]["content"].endswith(T._ASK_REASON_NUMBER))
        self.assertIsNone(T._gsm8k({"question": "Q?", "answer": "no marker"}))
        self.assertEqual(T._gsm8k({"question": "Q?", "answer": "x\n#### 1,200"}).gold, "1200")

    def test_the_check(self):
        it = T._gsm8k(self.row)
        loose = T.lenient(it)
        for reply in ("48 + 24 = 72. <answer>72</answer>", "<answer>$72.00</answer>", "<answer>72 clips</answer>",
                      "<answer>1</answer> no wait <answer>72</answer>"):
            self.assertEqual((it.check(reply), loose.check(reply)), (True, True), reply)
        for reply in ("So she sold 72 clips.", "The total is \\boxed{72}. Check: 48 / 2 = 24."):
            self.assertEqual((it.check(reply), loose.check(reply)), (False, True), reply)
        for reply in ("<answer>71</answer>", "48 + 24 = 72, so <answer>70</answer>", "72 or 73? 73.",
                      "\\boxed{70}, since 72"):
            self.assertFalse(loose.check(reply), reply)


class MathTests(unittest.TestCase):
    def test_last_boxed(self):
        self.assertEqual(T.last_boxed("so \\boxed{1} then \\boxed{\\frac{1}{2}}."), "\\frac{1}{2}")
        self.assertEqual(T.last_boxed("$\\boxed 5$"), "5")
        self.assertEqual(T.last_boxed("\\fbox{x+1}"), "x+1")
        self.assertIsNone(T.last_boxed("no box"))
        self.assertIsNone(T.last_boxed("\\boxed{unclosed"))

    def test_normalisation(self):
        same = [("\\dfrac{1}{2}", "\\frac{1}{2}"), ("\\tfrac12", "\\frac{1}{2}"), ("0.5", "\\frac{1}{2}"),
                ("1/2", "\\frac{1}{2}"), ("x = 5", "5"), ("5.0", "5"), ("5.00", "5"), ("5\\text{ cm}", "5"),
                ("$12$", "12"), ("\\$1.05", "1.05"), ("90^\\circ", "90"), ("90^{\\circ}", "90"), ("50\\%", "50"),
                ("\\left( 1, 2 \\right)", "(1,2)"), ("\\sqrt3", "\\sqrt{3}"), ("1,000", "1000"), ("1\\,000", "1000"),
                ("\\text{(C)}", "(C)"), (".5", "\\frac{1}{2}"), ("10.", "10"),
                ("\\frac{\\sqrt{3}}{2}", "\\frac{\\sqrt{3}}{2}"), ("2\\!\\cdot\\!3", "2\\cdot3")]
        for raw, want in same:
            self.assertEqual(T.math_norm(raw), want, raw)
        self.assertNotEqual(T.math_norm("\\frac{1}{3}"), T.math_norm("\\frac{1}{2}"))
        self.assertNotEqual(T.math_norm("(1,2)"), T.math_norm("(2,1)"))

    def test_clean_answers_only(self):
        for gold in ("5", "\\frac{1}{2}", "(1,2)", "Monday", "\\sqrt{3}"):
            self.assertTrue(T._math_clean(T.math_norm(gold)), gold)
        for gold in ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "", "x" * 41, "\\frac{1}{2"):
            self.assertFalse(T._math_clean(T.math_norm(gold)), gold)

    def test_the_item_and_check(self):
        row = {"problem": "What is $\\frac{1}{4} + \\frac{1}{4}$?", "level": "Level 1", "type": "Prealgebra",
               "solution": "Adding, $\\frac{1}{4}+\\frac{1}{4} = \\boxed{\\dfrac{1}{2}}$."}
        it = T._math(row)
        loose = T.lenient(it)
        self.assertEqual(it.gold, "\\dfrac{1}{2}")
        self.assertTrue(it.messages[0]["content"].endswith(T._ASK_REASON_MATH))
        for reply in ("<answer>\\frac{1}{2}</answer>", "<answer>$\\frac12$</answer>", "<answer>0.5</answer>",
                      "<answer>1/2</answer>", "<answer>\\boxed{\\frac{1}{2}}</answer>", f"<answer>{it.gold}</answer>"):
            self.assertEqual((it.check(reply), loose.check(reply)), (True, True), reply)
        self.assertEqual((it.check("So it's \\boxed{\\frac{1}{2}}."), loose.check("So it's \\boxed{\\frac{1}{2}}.")),
                         (False, True))
        for reply in ("<answer>\\frac{1}{4}</answer>", "\\boxed{2}", "The answer is one half.", "<answer></answer>"):
            self.assertFalse(loose.check(reply), reply)
        five = T._math({"problem": "2+3?", "solution": "\\boxed{5}"})
        self.assertTrue(T.lenient(five).check("2 + 3 = 5"), "a numeric gold: the reply's last number, loosely")
        self.assertFalse(five.check("2 + 3 = 5"))
        self.assertIsNone(T._math({"problem": "Find M.", "solution": "\\boxed{\\begin{pmatrix} 1 \\end{pmatrix}}"}))
        self.assertIsNone(T._math({"problem": "No box.", "solution": "It is 5."}))


class MedmcqaTests(unittest.TestCase):
    def test_the_item(self):
        row = {"question": "Which vitamin comes only from animals?", "opa": "Vitamin C", "opb": "Vitamin B7",
               "opc": "Vitamin B12", "opd": "Vitamin D", "cop": 2}
        it = T._medmcqa(row)
        self.assertEqual(it.gold, "C")
        self.assertEqual(it.messages[0]["content"].splitlines()[3], "C. Vitamin B12")
        self.assertTrue(it.check("<answer>C</answer>"))
        self.assertTrue(T.lenient(it).check("The answer is C."))
        self.assertFalse(T.lenient(it).check("B"))
        self.assertIsNone(T._medmcqa({**row, "cop": -1}), "the test split's hidden answers")
        self.assertIsNone(T._medmcqa({**row, "opd": None}))


def fake_task(name, rows_by_split):
    """The task called `name` as es_tasks.get makes it, reading `rows_by_split` instead of its download."""
    task = T.HubTask(name, *T.HUB[name])
    task._data = lambda: rows_by_split
    return task


def fake_rows():
    gsm = [{"question": f"What is {i} + {i}?", "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}"} for i in range(400)]
    math = [{"problem": f"What is ${i}^2$?", "solution": f"It is $\\boxed{{{i * i}}}$."} for i in range(400)]
    ifeval = [{"messages": [{"role": "user", "content": f"Tell me about {i}. No commas."}],
               "ground_truth": gt("validate_no_commas")} for i in range(400)]
    med = [{"question": f"Q{i}?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": i % 4} for i in range(500)]
    return {"gsm8k": {"train": gsm}, "math": {"train": math}, "ifeval": {"train": ifeval},
            "medmcqa": {"train": med[:400], "validation": med[400:]}}


class DrawingTests(unittest.TestCase):
    """Items are drawn from a round's seed the same way every time, and training never sees the held-out rows."""

    def test_the_tasks_and_the_mix(self):
        self.assertEqual(T.NAMES[:22], ("countdown", "arithmetic", "arc_easy", "sciq", "svamp", "trivia_qa",
                                        "commonsense_qa", "sst2", "openbookqa", "asdiv", "nq_open", "winogrande",
                                        "ag_news", "qasc", "aqua_rat", "web_questions", "hellaswag", "emotion",
                                        "boolq", "arc_challenge", "piqa", "social_iqa"), "earlier tasks keep places")
        self.assertEqual(T.NAMES[22:], ("ifeval", "gsm8k", "math", "medmcqa"))
        self.assertEqual(len(T.CURATED), 17)
        self.assertTrue(set(T.VIBE) <= set(T.NAMES))
        self.assertEqual(len(set(T.VIBE)), len(T.VIBE))
        self.assertEqual(len(T.VIBE), 16, "16 problems a round: one of each")
        self.assertFalse({"sst2", "ag_news", "emotion", "trivia_qa", "nq_open", "web_questions"} & set(T.VIBE))
        self.assertTrue(set(T.HUB) >= set(T.NAMES) - {"countdown", "arithmetic", "arc_easy"})

    def test_items_are_the_same_every_time(self):
        rows = fake_rows()
        for name in ("gsm8k", "math", "ifeval", "medmcqa"):
            a, b = fake_task(name, rows[name]), fake_task(name, rows[name])
            first = [it.messages for it in a.items(12345, 8)]
            self.assertEqual(first, [it.messages for it in b.items(12345, 8)], name)
            self.assertNotEqual(first, [it.messages for it in a.items(12346, 8)], name)
            self.assertEqual([it.messages for it in a.eval_items()], [it.messages for it in b.eval_items()], name)
            # the held-out parts are apart from the training rows, and from each other
            _, train = a._pool(a.train)
            _, test = a._pool(a.test)
            _, val = a._pool(a.val)
            if a.train[0] == a.test[0]:
                self.assertFalse(set(train) & set(test) or set(train) & set(val) or set(test) & set(val), name)
            held = {json.dumps(it.messages) for it in a.eval_items() + a.val_items(10)}
            self.assertFalse(held & {json.dumps(it.messages) for it in a.items(7, 200)}, name)

    def test_a_mixed_lenient_round(self):
        from tests.test_es2 import AnchorTests
        rows = fake_rows()
        tasks = ("ifeval", "gsm8k", "math", "medmcqa")
        spec = AnchorTests().spec(task="ifeval", tasks=tasks, mix=True, lenient=True, problems=4, max_tokens=512)
        self.assertEqual(spec.problems_list(), [])
        self.assertEqual(es2.Spec2.from_json(spec.to_json()), spec)
        with mock.patch.dict(T._TASKS, {n: fake_task(n, rows[n]) for n in tasks}):
            a = es2_rounds.items(spec, 99, 0)
            self.assertEqual([it.messages for it in a], [it.messages for it in es2_rounds.items(spec, 99, 0)])
            self.assertTrue(a[0].check("no commas at all"))  # ifeval: the whole reply
            self.assertFalse(a[0].check("one, two"))
            self.assertTrue(a[1].check(f"So the answer is \\boxed{{{a[1].gold}}}."))  # gsm8k, lenient
            self.assertTrue(a[2].check(f"Thus \\boxed{{{a[2].gold}}}."))  # math, lenient
            self.assertTrue(a[3].check(f"The answer is {a[3].gold}."))  # medmcqa, lenient


if __name__ == "__main__":
    unittest.main()
