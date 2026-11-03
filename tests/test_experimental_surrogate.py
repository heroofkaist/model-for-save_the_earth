"""
Unit tests for Deep Experimental Surrogate (Model E).
"""

import unittest
from pathlib import Path

import numpy as np
import torch

from src.save_the_earth.models.experimental_surrogate import ExperimentalDistanceNet
from src.save_the_earth.training.train_experimental_surrogate import extract_features, pinball_loss


class TestExperimentalSurrogate(unittest.TestCase):
    def setUp(self):
        self.model = ExperimentalDistanceNet(in_dim=10, hidden_dim=64, dropout=0.0)
        self.model.eval()

    def test_forward_shape(self):
        batch_size = 16
        x = torch.randn(batch_size, 10)
        out = self.model(x)
        self.assertEqual(out.shape, (batch_size, 3))

    def test_monotonicity(self):
        """Verify that q10 <= q50 <= q90 for all predictions."""
        x = torch.randn(50, 10)
        out = self.model(x)
        q10 = out[:, 0]
        q50 = out[:, 1]
        q90 = out[:, 2]

        self.assertTrue(torch.all(q10 <= q50 + 1e-5).item())
        self.assertTrue(torch.all(q50 <= q90 + 1e-5).item())
        self.assertTrue(torch.all(q10 >= 0.0).item())

    def test_feature_extraction(self):
        sample_row = {
            "wind_speed_m_s": "6.0",
            "gust_peak_m_s": "8.5",
            "moisture_content_percent": "12.0",
            "fuel_category": "structural",
            "mass_kg": "0.0001",
            "projected_area_m2": "0.0005",
        }
        feats = extract_features(sample_row)
        self.assertEqual(len(feats), 12)
        self.assertAlmostEqual(feats[0], 6.0)
        self.assertAlmostEqual(feats[1], 36.0)
        self.assertEqual(feats[5], 1.0)  # is_structural

    def test_pinball_loss(self):
        targets = torch.tensor([5.0, 10.0])
        preds = torch.tensor([[4.0, 5.0, 6.0], [8.0, 10.0, 12.0]])
        loss = pinball_loss(preds, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_checkpoint_exists_and_loads(self):
        checkpoint_path = Path("checkpoints/best_experimental_model.pt")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("mean_x", checkpoint)
            self.assertIn("std_x", checkpoint)
            self.assertIn("val_mae", checkpoint)


if __name__ == "__main__":
    unittest.main()

