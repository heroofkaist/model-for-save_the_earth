"""
Save the Earth — 10,000 Particle Trajectory Dataset Generator.

Generates at least 10,000 distinct, physically simulated 3D firebrand trajectories:
- Rigorous parameter sampling based on experimental wildfire literature
- Correlated mass and area matching measured USDA / IBHS firebrands
- Scenario-based grouping to prevent cross-partition leakage
- Output saved in compressed NPZ format with full metadata and summary CSV
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.save_the_earth.physics.aerodynamics import simulate_advanced_trajectory


def _sim_worker(args):
    """Worker function for parallel trajectory generation."""
    traj_id, sc_id, split_name, height, sc_wind, sc_wind_angle, mass, area, cd, sc_z0, sc_updraft, v0, dt_sim, max_t, resample_pts = args
    t_pts, states, info = simulate_advanced_trajectory(
        release_height=height,
        u_ref=sc_wind,
        mass=mass,
        area=area,
        cd=cd,
        v0=v0,
        wind_angle_rad=sc_wind_angle,
        z0=sc_z0,
        w0_updraft=sc_updraft,
        dt=dt_sim,
        max_t=max_t,
        resample_points=resample_pts
    )
    feat = np.array([height, sc_wind, sc_wind_angle, mass, area, cd, sc_z0, sc_updraft, v0[2]], dtype=np.float32)
    summary_item = {
        "trajectory_id": traj_id,
        "scenario_id": f"scenario_{sc_id:03d}",
        "split": split_name,
        "release_height_m": f"{height:.3f}",
        "wind_ref_m_s": f"{sc_wind:.2f}",
        "mass_kg": f"{mass:.6e}",
        "area_m2": f"{area:.6e}",
        "cd": f"{cd:.3f}",
        "updraft_w0_m_s": f"{sc_updraft:.2f}",
        "flight_time_s": f"{info['flight_time_s']:.3f}",
        "landing_x_m": f"{info['landing_x_m']:.3f}",
        "landing_y_m": f"{info['landing_y_m']:.3f}",
        "landing_dist_m": f"{info['landing_dist_m']:.3f}"
    }
    landing_target = np.array([info["landing_x_m"], info["landing_y_m"], info["flight_time_s"]], dtype=np.float32)
    return traj_id, split_name, feat, t_pts.astype(np.float32), states[:, :3].astype(np.float32), states[:, 3:].astype(np.float32), landing_target, summary_item


def generate_dataset(config_path: str = "configs/simulation_10k.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    n_total = cfg["num_trajectories"]
    n_ood = cfg.get("num_ood_trajectories", 500)
    p = cfg["parameters"]
    num = cfg["numerical"]
    resample_pts = num["trajectory_points"]
    dt_sim = num["dt_sim"]
    max_t = num["max_flight_time"]

    print(f"Generating {n_total} standard + {n_ood} OOD firebrand trajectories...")
    start_time = time.time()

    num_scenarios = 100
    particles_per_scenario = n_total // num_scenarios

    train_scenarios = set(range(0, 70))
    val_scenarios = set(range(70, 85))
    test_scenarios = set(range(85, 100))

    sim_tasks = []

    for sc_id in range(num_scenarios):
        sc_wind = float(np.random.uniform(p["wind_ref_min"], p["wind_ref_max"]))
        sc_wind_angle = float(np.random.uniform(-0.15, 0.15))
        sc_z0 = float(np.random.uniform(p["z0_min"], p["z0_max"]))
        sc_updraft = float(np.random.uniform(p["updraft_w0_min"], p["updraft_w0_max"]))
        sc_base_height = float(np.random.uniform(p["release_height_min"], p["release_height_max"]))

        if sc_id in train_scenarios:
            split_name = "train"
        elif sc_id in val_scenarios:
            split_name = "val"
        else:
            split_name = "test"

        for p_idx in range(particles_per_scenario):
            traj_id = f"traj_{sc_id:03d}_{p_idx:03d}"
            height = max(0.25, float(sc_base_height + np.random.uniform(-0.3, 0.3)))

            log_m = np.random.uniform(np.log(p["mass_kg_min"]), np.log(p["mass_kg_max"]))
            mass = float(np.exp(log_m))

            # Empirical area-to-mass ratio sampling: A/m in [0.35, 6.0] m^2/kg
            log_am = np.random.uniform(np.log(0.35), np.log(6.0))
            area = float(mass * np.exp(log_am))

            cd = float(np.random.uniform(p["cd_min"], p["cd_max"]))

            v0 = np.array([
                float(np.random.normal(0.0, 0.3)),
                float(np.random.normal(0.0, 0.2)),
                float(np.random.uniform(0.0, 0.6))
            ], dtype=np.float64)

            sim_tasks.append((
                traj_id, sc_id, split_name, height, sc_wind, sc_wind_angle,
                mass, area, cd, sc_z0, sc_updraft, v0, dt_sim, max_t, resample_pts
            ))

    print(f"Executing parallel simulations across CPU cores...")
    inputs_list, times_list, trajectories_list, velocities_list, landing_targets_list = [], [], [], [], []
    summary_rows = []
    split_assignments = {"train": [], "val": [], "test": []}

    workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, res in enumerate(executor.map(_sim_worker, sim_tasks, chunksize=100)):
            traj_id, split_name, feat, t_pts, pos3d, vel3d, landing_target, summary_item = res
            split_assignments[split_name].append(traj_id)
            inputs_list.append(feat)
            times_list.append(t_pts)
            trajectories_list.append(pos3d)
            velocities_list.append(vel3d)
            landing_targets_list.append(landing_target)
            summary_rows.append(summary_item)

            if (idx + 1) % 2500 == 0:
                print(f"  Completed {idx + 1}/{n_total} trajectories ({time.time() - start_time:.1f}s)...")

    # Generate Out-Of-Distribution (OOD) Scenarios (extreme winds & heavy chunks)
    print("Simulating Out-of-Distribution (OOD) test trajectories...")
    ood = cfg["ood_parameters"]
    ood_inputs, ood_trajectories, ood_targets, ood_times = [], [], [], []
    for ood_idx in range(n_ood):
        h_ood = float(np.random.uniform(ood["release_height_min"], ood["release_height_max"]))
        w_ood = float(np.random.uniform(ood["wind_ref_min"], ood["wind_ref_max"]))
        m_ood = float(np.exp(np.random.uniform(np.log(ood["mass_kg_min"]), np.log(ood["mass_kg_max"]))))
        a_ood = float(m_ood * np.exp(np.random.uniform(np.log(0.3), np.log(4.0))))
        cd_ood = float(np.random.uniform(0.6, 1.4))

        t_pts, states, info = simulate_advanced_trajectory(
            release_height=h_ood,
            u_ref=w_ood,
            mass=m_ood,
            area=a_ood,
            cd=cd_ood,
            dt=dt_sim,
            max_t=60.0,
            resample_points=resample_pts
        )
        ood_inputs.append(np.array([h_ood, w_ood, 0.0, m_ood, a_ood, cd_ood, 0.01, 0.0, 0.0], dtype=np.float32))
        ood_times.append(t_pts.astype(np.float32))
        ood_trajectories.append(states[:, :3].astype(np.float32))
        ood_targets.append(np.array([info["landing_x_m"], info["landing_y_m"], info["flight_time_s"]], dtype=np.float32))

    # Save to data/synthetic/ and data/manifests/
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "trajectories_10k.npz"
    np.savez_compressed(
        npz_path,
        inputs=np.array(inputs_list),
        times=np.array(times_list),
        trajectories=np.array(trajectories_list),
        velocities=np.array(velocities_list),
        landing_targets=np.array(landing_targets_list)
    )
    print(f"Saved compressed 10k dataset to: {npz_path} ({os.path.getsize(npz_path) / (1024*1024):.1f} MB)")

    ood_npz_path = out_dir / "trajectories_ood.npz"
    np.savez_compressed(
        ood_npz_path,
        inputs=np.array(ood_inputs),
        times=np.array(ood_times),
        trajectories=np.array(ood_trajectories),
        landing_targets=np.array(ood_targets)
    )

    csv_path = out_dir / "trajectories_10k_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary metadata to: {csv_path}")

    split_json_path = manifest_dir / "trajectory_splits.json"
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_assignments, f, indent=2)
    print(f"Saved split assignments to: {split_json_path}")

    elapsed = time.time() - start_time
    print(f"\nSuccessfully generated {n_total} standard + {n_ood} OOD trajectories in {elapsed:.2f}s ({n_total/elapsed:.1f} traj/s)!")
    print(f"Allocations: Train = {len(split_assignments['train'])}, Val = {len(split_assignments['val'])}, Test = {len(split_assignments['test'])}")


if __name__ == "__main__":
    generate_dataset()
