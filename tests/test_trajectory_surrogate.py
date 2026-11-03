"""
Unit tests for 3D Firebrand Trajectory Models and Aerodynamics Engine.
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from src.save_the_earth.physics.aerodynamics import (
    simulate_advanced_trajectory,
    atmospheric_wind_vector,
    compute_tohidi_drag_lift,
    compute_combustion_buoyancy
)
from src.save_the_earth.models.trajectory_models import (
    NeuralTrajectorySurrogate,
    PhysicsInformedHybridSurrogate,
)
from src.save_the_earth.training.train_surrogate import compute_ballistic_baseline


class TestTrajectorySurrogate(unittest.TestCase):
    def test_tohidi_and_combustion_physics(self):
        # Flat plate/bark: high slenderness
        cd_flat, cl_flat = compute_tohidi_drag_lift(mass=1e-4, area=1e-3, thickness=0.001)
        self.assertGreater(cd_flat, 1.0)
        self.assertGreater(cl_flat, 0.2)

        # Compact chunk: low slenderness
        cd_chunk, cl_chunk = compute_tohidi_drag_lift(mass=1e-4, area=1e-5, thickness=0.01)
        self.assertLess(cd_chunk, cd_flat)
        self.assertLess(cl_chunk, cl_flat)

        # Combustion buoyancy
        g_burn = compute_combustion_buoyancy(is_burning=True, eta_buoy=1.20)
        g_inert = compute_combustion_buoyancy(is_burning=False)
        self.assertLess(g_burn, g_inert)

        # Simulation with combustion buoyancy travels farther than inert
        _, _, info_burn = simulate_advanced_trajectory(
            release_height=3.0, u_ref=10.0, mass=1e-4, area=2e-4, cd=cd_flat,
            is_burning=True, eta_buoy=1.20, cl_lift=cl_flat
        )
        _, _, info_inert = simulate_advanced_trajectory(
            release_height=3.0, u_ref=10.0, mass=1e-4, area=2e-4, cd=cd_flat,
            is_burning=False, eta_buoy=1.0, cl_lift=0.0
        )
        self.assertGreater(info_burn["landing_distance_m"], info_inert["landing_distance_m"])
    def test_aerodynamic_simulation(self):
        t_pts, states, info = simulate_advanced_trajectory(
            release_height=5.0,
            u_ref=10.0,
            mass=5e-5,
            area=2e-4,
            cd=1.0,
            resample_points=50,
        )
        self.assertEqual(len(t_pts), 50)
        self.assertEqual(states.shape, (50, 6))
        self.assertAlmostEqual(states[-1, 2], 0.0, places=3)
        self.assertGreater(info["landing_distance_m"], 0.0)
        self.assertGreater(info["flight_time_s"], 0.0)

    def test_wind_vector_plume_decay(self):
        # Center of plume
        w_center = atmospheric_wind_vector(0.0, 0.0, 5.0, u_ref=10.0, w0_updraft=3.0)
        # 50m downwind (away from plume)
        w_far = atmospheric_wind_vector(50.0, 0.0, 5.0, u_ref=10.0, w0_updraft=3.0)
        self.assertGreater(w_center[2], w_far[2])
        self.assertAlmostEqual(w_far[2], 0.0, places=2)

    def test_neural_surrogate_shapes(self):
        model = NeuralTrajectorySurrogate(input_dim=9, hidden_dim=64, num_blocks=2, num_trajectory_points=50)
        x = torch.randn(8, 9)
        traj, land = model(x)
        self.assertEqual(traj.shape, (8, 50, 3))
        self.assertEqual(land.shape, (8, 3))

    def test_hybrid_surrogate_shapes(self):
        model = PhysicsInformedHybridSurrogate(input_dim=9, hidden_dim=64, num_pts=50)
        x = torch.randn(8, 9)
        base_traj = torch.zeros(8, 50, 3)
        base_land = torch.zeros(8, 3)
        pred_traj, pred_land = model(x, base_traj, base_land)
        self.assertEqual(pred_traj.shape, (8, 50, 3))
        self.assertEqual(pred_land.shape, (8, 3))

    def test_ballistic_baseline_computation(self):
        inputs = np.array([[10.0, 10.0, 0.0, 1e-4, 1e-4, 1.0, 0.01, 0.0, 0.0]], dtype=np.float32)
        times = np.linspace(0, 1.4278, 50, dtype=np.float32).reshape(1, 50)
        b_traj, b_land = compute_ballistic_baseline(inputs, times)
        self.assertEqual(b_traj.shape, (1, 50, 3))
        self.assertEqual(b_land.shape, (1, 3))
        self.assertGreater(b_land[0, 0], 0.0)
        self.assertAlmostEqual(b_land[0, 1], 0.0, places=4)

    def test_trained_checkpoint_exists(self):
        ckpt_path = Path("checkpoints/best_trajectory_model.pt")
        self.assertTrue(ckpt_path.exists(), "Trained surrogate checkpoint must exist on disk")


if __name__ == "__main__":
    unittest.main()
