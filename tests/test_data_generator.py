import collections
import os
import re
import unittest
from unittest import mock

import _testkit as tk
import data_generator as dg


class DataGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = tk.temp_workspace()
        cls.rows = dg.read_csv(cls.paths.dataset_csv)
        cls.train = dg.read_csv(cls.paths.train_csv)
        cls.test = dg.read_csv(cls.paths.test_csv)

    @classmethod
    def tearDownClass(cls):
        tk.remove_workspace(cls.paths)

    def test_counts_match_the_description(self):
        self.assertEqual(len(self.rows), 200)
        self.assertEqual(collections.Counter(r["label"] for r in self.rows), {0: 100, 1: 100})
        self.assertEqual(len({r["topic"] for r in self.rows}), 5)
        self.assertEqual(len(self.train), 140)
        self.assertEqual(collections.Counter(r["label"] for r in self.train), {0: 70, 1: 70})
        self.assertEqual(len(self.test), 60)
        self.assertEqual(collections.Counter(r["label"] for r in self.test), {0: 30, 1: 30})
        self.assertEqual(set(collections.Counter(r["topic"] for r in self.test if r["label"] == 0).values()), {6})

    def test_clean_poison_pairs_stay_together(self):
        train_pairs = {r["pair_id"] for r in self.train}
        test_pairs = {r["pair_id"] for r in self.test}
        self.assertFalse(train_pairs & test_pairs)
        for split in (self.train, self.test):
            per_pair = collections.Counter(r["pair_id"] for r in split)
            self.assertEqual(set(per_pair.values()), {2})

    def test_each_clean_policy_has_exactly_one_number(self):
        for r in self.rows:
            if r["label"] == 0:
                self.assertEqual(re.findall(r"\d+", r["text"]), [r["expected_number"]], r["chunk_id"])

    def test_targets_are_unique_and_planted(self):
        poisoned = [r for r in self.rows if r["label"] == 1]
        self.assertEqual(len({r["attacker_target"] for r in poisoned}), 100)
        for r in poisoned:
            self.assertIn(r["attacker_target"], r["text"])

    def test_clean_reference_is_the_clean_twin(self):
        clean = {r["pair_id"]: r["text"] for r in self.rows if r["label"] == 0}
        for r in self.rows:
            self.assertEqual(r["clean_reference"], clean[r["pair_id"]])

    def test_test_split_shares_template_families_with_training(self):
        fams = lambda rows: {r["template_family"] for r in rows if r["label"] == 1}  # noqa: E731
        self.assertEqual(fams(self.train), fams(self.test))

    def test_generation_is_deterministic_and_matches_committed_csvs(self):
        self.assertEqual(dg.generate(42), dg.generate(42))
        committed = dg.read_csv(tk.ROOT / "data" / "dataset.csv")
        self.assertEqual(committed, self.rows, "data/dataset.csv is out of date: run src/data_generator.py")

    def test_readme_example_question_exists(self):
        self.assertIn("What is the Cedar division refund policy?", {r["question"] for r in self.rows})

    def test_locked_file_gives_a_clear_message(self):
        with mock.patch.object(os, "replace", side_effect=PermissionError("locked")):
            with self.assertRaises(SystemExit) as ctx:
                dg.write_csv(self.paths.data_dir / "locked.csv", self.rows[:1])
        self.assertIn("Excel", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
