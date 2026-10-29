import contextlib
import io
import json
import unittest

import _testkit as tk
import evaluate_all
from config import MODES


class EvaluateAllTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = tk.trained_workspace()

    @classmethod
    def tearDownClass(cls):
        tk.remove_workspace(cls.paths)

    def test_smoke_benchmark_writes_all_outputs(self):
        ws = tk.offline_workspace(self.paths)
        with contextlib.redirect_stdout(io.StringIO()):
            code = evaluate_all.main(["--limit", "2", "--top-k", "1"], paths=self.paths, workspace=ws)
        self.assertEqual(code, 0)
        runs = json.loads(self.paths.benchmark_runs_json.read_text(encoding="utf-8"))["runs"]
        self.assertEqual(len(runs), 2 * len(MODES) * 2)
        for case in (0, 1):
            order = [r["mode"] for r in runs if r["case_index"] == case and r["variant"] == "poisoned"]
            self.assertEqual(sorted(order), sorted(MODES))
        results = json.loads(self.paths.benchmark_results_json.read_text(encoding="utf-8"))
        self.assertEqual(set(results["modes"]), set(MODES))
        self.assertIn("warm_up_ms_excluded", results["meta"])
        self.assertEqual(results["meta"]["top_k"], 1)
        self.assertTrue(self.paths.benchmark_chart_png.stat().st_size > 10_000)

    def test_mode_order_is_shuffled_per_case(self):
        ws = tk.offline_workspace(self.paths)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_all.main(["--limit", "6", "--top-k", "1", "--modes", *MODES], paths=self.paths, workspace=ws)
        runs = json.loads(self.paths.benchmark_runs_json.read_text(encoding="utf-8"))["runs"]
        orders = {tuple(r["mode"] for r in runs if r["case_index"] == c and r["variant"] == "poisoned") for c in range(6)}
        self.assertGreater(len(orders), 1)

    def test_unreachable_llm_stops_before_timing(self):
        ws = tk.offline_workspace(self.paths, llm=tk.ScriptedLLM(fail=True))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evaluate_all.main(["--limit", "1"], paths=self.paths, workspace=ws), 1)


if __name__ == "__main__":
    unittest.main()
