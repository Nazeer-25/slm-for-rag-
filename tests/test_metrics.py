import unittest

import _testkit  # noqa: F401  (puts src/ on the path)
from metrics import aggregate, fact_found, numbers_in, rouge_l, targets_found


def run(mode, variant, status="ok", hijacked=False, fact=True, chunks=None):
    return {"mode": mode, "variant": variant, "status": status, "hijacked": hijacked, "fact_found": fact,
            "chunks": chunks or [], "latency_ms": 100.0, "llm_calls": 1}


def chunk(label, flagged=False, excluded=False, f1=1.0):
    return {"label": label, "flagged": flagged, "excluded": excluded, "preservation_f1": f1}


class MetricsTest(unittest.TestCase):
    def test_targets_are_matched_ignoring_case_and_spaces(self):
        self.assertEqual(targets_found("Please call 555 - 1234 now", ["555-1234"]), ["555-1234"])
        self.assertEqual(targets_found("See VERIFY-12.example.com", ["verify-12.example.com"]), ["verify-12.example.com"])
        self.assertEqual(targets_found("Refunds take 90 days.", ["SR-1234"]), [])

    def test_numbers_and_number_words(self):
        self.assertTrue({"1000", "45", "90"} <= numbers_in("1,000 items, $45.0 fee, ninety days"))
        self.assertIn("45", numbers_in("forty-five days"))
        self.assertTrue(fact_found("within 45 days", "45"))
        self.assertFalse(fact_found("within 450 days", "45"))
        self.assertFalse(fact_found("anything", ""))

    def test_rouge_l(self):
        self.assertAlmostEqual(rouge_l("a b c d", "a b c d")["f1"], 1.0)
        self.assertEqual(rouge_l("a b c", "x y z")["f1"], 0.0)
        partial = rouge_l("a b c d", "a b")
        self.assertAlmostEqual(partial["recall"], 0.5)
        self.assertAlmostEqual(partial["precision"], 1.0)
        self.assertEqual(rouge_l("a b", "")["f1"], 0.0)

    def test_failed_runs_are_counted_and_excluded_from_denominators(self):
        runs = [run("hybrid_defense", "poisoned", hijacked=True, chunks=[chunk(1, flagged=True)]),
                run("hybrid_defense", "poisoned", hijacked=False, chunks=[chunk(1, flagged=True)]),
                run("hybrid_defense", "poisoned", status="failed", hijacked=True),
                run("hybrid_defense", "clean", fact=True, chunks=[chunk(0, flagged=True), chunk(0), chunk(0), chunk(0)]),
                run("hybrid_defense", "clean", status="failed", fact=False)]
        m = aggregate(runs)["hybrid_defense"]
        self.assertEqual(m["failed"], {"total": 2, "poisoned": 1, "clean": 1})
        self.assertEqual((m["asr"]["numerator"], m["asr"]["denominator"]), (1, 2))
        self.assertEqual((m["benign_utility"]["numerator"], m["benign_utility"]["denominator"]), (1, 1))
        self.assertEqual((m["fpr"]["numerator"], m["fpr"]["denominator"]), (1, 4))
        self.assertEqual((m["poisoned_chunk_detection"]["numerator"], m["poisoned_chunk_detection"]["denominator"]), (2, 2))

    def test_excluded_chunks_count_as_flagged(self):
        m = aggregate([run("llm_judge_defense", "clean", chunks=[chunk(0, excluded=True), chunk(0)])])["llm_judge_defense"]
        self.assertEqual(m["fpr"]["value"], 0.5)


if __name__ == "__main__":
    unittest.main()
