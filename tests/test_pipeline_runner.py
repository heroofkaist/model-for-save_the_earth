import os
import unittest
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


if __name__ == "__main__":
    unittest.main()
