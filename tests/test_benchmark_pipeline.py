import os
import unittest
import numpy as np

from train_distance_benchmarks import load_dataset, compute_metrics


class TestBenchmarkPipeline(unittest.TestCase):
    def setUp(self):
        self.data_path = "data/interim/eligible.csv"
        self.split_path = "data/manifests/split_v1.csv"

    def test_data_and_split_integrity(self):
        records = load_dataset(self.data_path, self.split_path)
        self.assertGreaterEqual(len(records), 60)

        # Check partitions exist
        splits = {r["split"] for r in records}
        self.assertEqual(splits, {"train", "validation", "test"})

        # Group disjointness: ensure no experiment_id belongs to multiple splits
        group_to_splits = {}
        for r in records:
            group_to_splits.setdefault(r["experiment_id"], set()).add(r["split"])

        for gid, sp_set in group_to_splits.items():
            self.assertEqual(len(sp_set), 1, f"Group leakage detected in {gid}: {sp_set}")

    def test_metrics_computation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 4.0])
        m = compute_metrics(y_true, y_pred)
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["mae_m"], 0.375)
        self.assertAlmostEqual(m["bias_m"], 0.125)


if __name__ == "__main__":
    unittest.main()
