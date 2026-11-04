import os
import io
import contextlib
import unittest
from unittest.mock import patch
import numpy as np

import run_pipeline


class TestPipelineRunner(unittest.TestCase):
    def test_predict_functionality(self):
        # Ensure operational predict mode runs without error
        try:
            run_pipeline.run_predict(wind=11.17, moisture=12.0, category="vegetative")
        except Exception as e:
            self.fail(f"run_predict raised an exception: {e}")

    def test_predict_structural(self):
        try:
            run_pipeline.run_predict(wind=17.88, moisture=8.0, category="structural")
        except Exception as e:
            self.fail(f"run_predict structural raised an exception: {e}")


    def test_rejects_invalid_wind(self):
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            run_pipeline.run_predict(wind=float("nan"), category="vegetative")
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            run_pipeline.run_predict(wind=-1.0, category="vegetative")

    def test_prospective_mode_never_fabricates_model_e_features(self):
        # The real input data may be generated locally; use a minimal deterministic
        # fixture to exercise inference without needing raw USDA archives.
        rows = []
        for i in range(16):
            w = [5.36, 11.17, 17.88, 8.2][i % 4]
            rows.append({
                "split": "train" if i < 12 else "validation",
                "landing_distance_m": 0.5 + 0.25 * w + 0.1 * (i % 3),
                "fuel_category": "vegetative" if i % 2 else "structural",
                "wind_speed_m_s": w,
            })
        output = io.StringIO()
        original_exists = os.path.exists
        def exists_stub(path):
            if path in ("data/interim/eligible.csv", "data/manifests/split_v1.csv"):
                return True
            return original_exists(path)

        with patch("run_pipeline.os.path.exists", side_effect=exists_stub), \\
             patch("train_distance_benchmarks.load_dataset", return_value=rows), \\
             contextlib.redirect_stdout(output):
            run_pipeline.run_predict(wind=11.17, category="vegetative")
        printed = output.getvalue()
        self.assertIn("Model E omitted", printed)
        self.assertIn("Individual-ember dispersion omitted", printed)
        self.assertNotIn("Deep Experimental Surrogate:", printed)
        self.assertNotIn("RECOMMENDED OPERATIONAL", printed)


if __name__ == "__main__":
    unittest.main()

