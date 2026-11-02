"""Run with python -m unittest discover -s tests."""
import csv
import tempfile
import unittest
from pathlib import Path

from train_ember import REQUIRED, grouped_split, load_rows


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.csv"
        self.rows = []
        for group in "ABCDEF":
            for i in range(2):
                self.rows.append({
                    "experiment_id": group, "source_id": "lab-notebook",
                    "data_kind": "measured", "release_height_m": "20",
                    "wind_x_m_s": "5", "wind_y_m_s": "0",
                    "mass_kg": "0.001", "area_m2": "0.0004",
                    "drag_coefficient": "1", "air_density_kg_m3": "1.225",
                    "landing_x_m": str(i + 3), "landing_y_m": "0",
                    "flight_time_s": "4"
                })
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_group_holdout_has_no_leakage(self):
        rows = load_rows(self.path)
        train, test, _ = grouped_split(rows, 0.25, seed=7)
        self.assertFalse({rows[i]["experiment_id"] for i in range(len(rows)) if train[i]}
                         & {rows[i]["experiment_id"] for i in range(len(rows)) if test[i]})

    def test_reject_synthetic(self):
        self.rows[0]["data_kind"] = "synthetic"
        self.write()
        with self.assertRaisesRegex(ValueError, "Synthetic data refused"):
            load_rows(self.path)

    def test_invalid_mass(self):
        self.rows[0]["mass_kg"] = "-1"
        self.write()
        with self.assertRaisesRegex(ValueError, "positive"):
            load_rows(self.path)


if __name__ == "__main__":
    unittest.main()
