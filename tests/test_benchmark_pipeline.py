import os
import tempfile
import unittest
import csv
import numpy as np

from train_distance_benchmarks import load_dataset, compute_metrics


class TestBenchmarkPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p_data = os.path.join(self.temp_dir.name, "eligible.csv")
        self.p_split = os.path.join(self.temp_dir.name, "split_v1.csv")

        # Create self-contained fixture data
        data_rows = []
        split_rows = []
        groups = [
            ("g1", "train", "veg", "chamise"),
            ("g2", "train", "struct", "fence"),
            ("g3", "validation", "veg", "pine"),
            ("g4", "test", "struct", "roof")
        ]
        for gid, part, cat, fuel in groups:
            for w in [5.36, 11.17, 17.88]:
                rid = f"rec_{gid}_{fuel}_{w}"
                data_rows.append({
                    "record_id": rid,
                    "source_id": f"usda_{cat}",
                    "experiment_id": gid,
                    "fuel_type": fuel,
                    "wind_speed_m_s": str(w),
                    "mass_kg": "0.0001",
                    "projected_area_m2": "0.00005",
                    "landing_distance_m": str(1.5 + w * 0.1)
                })
                split_rows.append({
                    "record_id": rid,
                    "task_family": "particle_distance",
                    "source_id": f"usda_{cat}",
                    "group_id": gid,
                    "split": part
                })

        with open(self.p_data, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0].keys()))
            w.writeheader()
            w.writerows(data_rows)

        with open(self.p_split, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
            w.writeheader()
            w.writerows(split_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_data_and_split_integrity(self):
        records = load_dataset(self.p_data, self.p_split)
        self.assertEqual(len(records), 12)

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
