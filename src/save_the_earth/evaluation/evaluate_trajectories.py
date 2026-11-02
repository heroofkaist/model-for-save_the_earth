"""
Save the Earth — Trajectory Evaluation, Benchmarking & Diagnostic Figure Generation.

Evaluates:
- Baseline: Analytical Ballistic
- Model A: Mechanistic RK4 numerical integration
- Model B: Neural Trajectory Surrogate
- Model C: Physics-Informed Hybrid Surrogate

Tests against:
- 1,500 Held-Out Synthetic Test Trajectories
- 500 Out-of-Distribution (OOD) Extreme Trajectories
- 69 USDA JFSP Experimental Wind-Tunnel Runs

Measures throughput (trajectories/sec) and generates publication-grade figures.
"""

import os
import sys
import json
import time
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.save_the_earth.physics.aerodynamics import simulate_advanced_trajectory
from src.save_the_earth.models.trajectory_models import (
    NeuralTrajectorySurrogate,
    PhysicsInformedHybridSurrogate,
)
from src.save_the_earth.training.train_surrogate import compute_ballistic_baseline, get_device


def load_model_and_stats(checkpoint_path: str, model_type: str = "neural_surrogate", device: torch.device = torch.device("cpu")):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    
    if model_type == "neural_surrogate":
        model = NeuralTrajectorySurrogate(
            input_dim=9,
            hidden_dim=cfg["architecture"]["hidden_dims"][0],
            num_blocks=len(cfg["architecture"]["hidden_dims"]) - 1,
            num_trajectory_points=50,
            dropout=cfg["architecture"]["dropout"],
        )
    else:
        model = PhysicsInformedHybridSurrogate(
            input_dim=9,
            hidden_dim=cfg["architecture"]["hidden_dims"][0],
            num_pts=50,
        )
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    stats = {
        "x_mean": checkpoint["x_mean"],
        "x_std": checkpoint["x_std"],
        "traj_mean": checkpoint["traj_mean"],
        "traj_std": checkpoint["traj_std"],
        "land_mean": checkpoint["land_mean"],
        "land_std": checkpoint["land_std"],
    }
    return model, stats


def compute_metrics(y_true_traj, y_pred_traj, y_true_land, y_pred_land):
    """
    Computes rigorous MAE, RMSE, and maximum deviations across trajectory and landing.
    """
    # 3D position errors across all points
    diff_traj = y_pred_traj - y_true_traj # [N, 50, 3]
    dist_err_per_pt = np.linalg.norm(diff_traj, axis=-1) # [N, 50]
    traj_3d_mae = float(np.mean(dist_err_per_pt))
    traj_3d_rmse = float(np.sqrt(np.mean(dist_err_per_pt**2)))
    max_traj_dev = float(np.max(dist_err_per_pt))

    # Landing distance errors
    dist_true = np.linalg.norm(y_true_land[:, :2], axis=-1)
    dist_pred = np.linalg.norm(y_pred_land[:, :2], axis=-1)
    land_dist_mae = float(np.mean(np.abs(dist_pred - dist_true)))
    land_dist_rmse = float(np.sqrt(np.mean((dist_pred - dist_true)**2)))

    # Landing coordinates errors
    land_x_mae = float(np.mean(np.abs(y_pred_land[:, 0] - y_true_land[:, 0])))
    land_y_mae = float(np.mean(np.abs(y_pred_land[:, 1] - y_true_land[:, 1])))

    # Flight time errors
    flight_time_mae = float(np.mean(np.abs(y_pred_land[:, 2] - y_true_land[:, 2])))
    flight_time_rmse = float(np.sqrt(np.mean((y_pred_land[:, 2] - y_true_land[:, 2])**2)))

    return {
        "traj_3d_mae_m": traj_3d_mae,
        "traj_3d_rmse_m": traj_3d_rmse,
        "max_trajectory_deviation_m": max_traj_dev,
        "landing_distance_mae_m": land_dist_mae,
        "landing_distance_rmse_m": land_dist_rmse,
        "landing_x_mae_m": land_x_mae,
        "landing_y_mae_m": land_y_mae,
        "flight_time_mae_s": flight_time_mae,
        "flight_time_rmse_s": flight_time_rmse,
    }


def benchmark_throughput(model_b, stats_b, test_inputs, num_warmup: int = 10, num_runs: int = 50, device: torch.device = torch.device("cpu")):
    """
    Measures inference throughput: Numerical RK4 vs PyTorch Neural Surrogate.
    """
    print("\n--- Benchmarking Inference Throughput ---")
    
    # 1. RK4 Numerical Solver (single-core)
    n_sample_rk4 = 100
    sub_inputs = test_inputs[:n_sample_rk4]
    
    t0 = time.perf_counter()
    for row in sub_inputs:
        h, u, angle, m, a, cd, z0, w_up, v0z = row
        simulate_advanced_trajectory(
            release_height=float(h),
            u_ref=float(u),
            mass=float(m),
            area=float(a),
            cd=float(cd),
            wind_angle_rad=float(angle),
            z0=float(z0),
            w0_updraft=float(w_up),
            v0=np.array([0.0, 0.0, float(v0z)]),
        )
    rk4_time = time.perf_counter() - t0
    rk4_throughput = n_sample_rk4 / rk4_time
    print(f"Model A (RK4 Numerical): {rk4_throughput:.1f} trajectories/sec ({1000/rk4_throughput:.2f} ms/traj)")

    # 2. Neural Surrogate Batch Inference
    x_mean = stats_b["x_mean"]
    x_std = stats_b["x_std"]
    norm_x = (test_inputs - x_mean) / x_std
    tensor_x = torch.from_numpy(norm_x).float().to(device)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model_b(tensor_x)
    if device.type == "mps":
        torch.mps.synchronize()

    # Timed batched runs
    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model_b(tensor_x)
    if device.type == "mps":
        torch.mps.synchronize()
    ml_time = time.perf_counter() - t0
    ml_total_trajs = len(test_inputs) * num_runs
    ml_throughput = ml_total_trajs / ml_time
    speedup = ml_throughput / rk4_throughput
    print(f"Model B (Neural Surrogate Batch {len(test_inputs)}): {ml_throughput:.1f} trajectories/sec ({1000000/ml_throughput:.2f} µs/traj)")
    print(f"Speedup Ratio: {speedup:.1f}x acceleration over RK4 numerical integration!")

    return {
        "rk4_throughput_traj_per_sec": rk4_throughput,
        "neural_throughput_traj_per_sec": ml_throughput,
        "speedup_ratio": speedup,
    }


def evaluate_on_real_usda(model_b, stats_b, device: torch.device):
    """
    Evaluates model landing distance predictions on genuine USDA JFSP experimental runs.
    """
    print("\n--- Evaluating on Real USDA JFSP Wind-Tunnel Data ---")
    data_path = Path("data/processed/canonical_observations.csv")
    if not data_path.exists():
        print("USDA canonical observations not found. Skipping real evaluation.")
        return {}

    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    print(f"Loaded {len(records)} records from USDA JFSP canonical observations.")

    obs_distances = []
    pred_distances = []
    
    # Feature parameters for wind tunnel geometry:
    # Release height = 2.0m (typical nozzle/source center height in IBHS / USDA wind tunnel test section)
    # Wind angle = 0.0 (aligned with tunnel flow)
    # z0 = 0.005 (smooth tunnel floor)
    # updraft = 0.0 (horizontal wind tunnel flow)
    # v0z = 0.0
    for r in records:
        try:
            obs_dist_str = r.get("landing_distance_m", "")
            if not obs_dist_str:
                continue
            obs_dist = float(obs_dist_str)
            if obs_dist <= 0:
                continue
            
            wind_speed = float(r.get("wind_speed_m_s", 6.7))
            mass_kg = float(r.get("mass_kg", 5e-5))
            area_m2 = float(r.get("projected_area_m2", 2e-4))
            cd = 1.05
            h = 2.0

            obs_distances.append(obs_dist)

            # Build feature vector [h, u, angle, mass, area, cd, z0, w_up, v0z]
            feat = np.array([[h, wind_speed, 0.0, mass_kg, area_m2, cd, 0.005, 0.0, 0.0]], dtype=np.float32)
            norm_feat = (feat - stats_b["x_mean"]) / stats_b["x_std"]
            
            with torch.no_grad():
                tensor_feat = torch.from_numpy(norm_feat).to(device)
                norm_pred_traj, norm_pred_land = model_b(tensor_feat)
                pred_land = norm_pred_land.cpu().numpy() * stats_b["land_std"] + stats_b["land_mean"]
                pred_d = float(np.linalg.norm(pred_land[0, :2]))
                pred_distances.append(pred_d)
        except Exception as e:
            continue

    obs_d = np.array(obs_distances)
    pred_d = np.array(pred_distances)

    real_mae = float(np.mean(np.abs(pred_d - obs_d)))
    real_rmse = float(np.sqrt(np.mean((pred_d - obs_d)**2)))
    print(f"USDA Real Data Performance: MAE = {real_mae:.3f} m, RMSE = {real_rmse:.3f} m on {len(obs_d)} experimental runs.")

    return {
        "num_runs": len(obs_d),
        "real_mae_m": real_mae,
        "real_rmse_m": real_rmse,
        "obs_distances": obs_d.tolist(),
        "pred_distances": pred_d.tolist(),
    }


def generate_diagnostic_figures(
    fig_dir: Path,
    test_inputs,
    y_true_traj,
    y_pred_traj_b,
    y_pred_traj_c,
    base_traj,
    y_true_land,
    y_pred_land_b,
    history_b,
    test_metrics_b,
    ood_metrics_b
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Generating Publication Diagnostic Figures ---")

    # 1. 3D Trajectory Comparison (Representative flight paths)
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    
    sample_indices = [5, 42, 128]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for idx, col in zip(sample_indices, colors):
        ax.plot(y_true_traj[idx, :, 0], y_true_traj[idx, :, 1], y_true_traj[idx, :, 2],
                label=f"Truth (RK4) #{idx}", color=col, linewidth=2.5)
        ax.plot(y_pred_traj_b[idx, :, 0], y_pred_traj_b[idx, :, 1], y_pred_traj_b[idx, :, 2],
                "--", label=f"Neural Surrogate #{idx}", color=col, alpha=0.85, linewidth=2.0)
        ax.scatter([y_true_land[idx, 0]], [y_true_land[idx, 1]], [0.0],
                   marker="x", s=60, color=col)
        ax.scatter([y_pred_land_b[idx, 0]], [y_pred_land_b[idx, 1]], [0.0],
                   marker="o", s=40, facecolors="none", edgecolors=col)

    ax.set_title("3D Firebrand Trajectory: RK4 Truth vs Neural Surrogate", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Downwind Distance X (m)", labelpad=8)
    ax.set_ylabel("Crosswind Distance Y (m)", labelpad=8)
    ax.set_zlabel("Altitude Z (m)", labelpad=8)
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(fig_dir / "trajectory_3d_comparison.png")
    plt.close()
    print("  -> Saved reports/figures/trajectory_3d_comparison.png")

    # 2. Side-elevation profiles (x-z) across different wind regimes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)
    wind_speeds = test_inputs[:, 1]
    
    low_idx = np.where(wind_speeds < 7.0)[0][0]
    med_idx = np.where((wind_speeds >= 10.0) & (wind_speeds <= 14.0))[0][0]
    high_idx = np.where(wind_speeds > 18.0)[0][0]
    
    subsets = [
        ("Low Wind (< 7 m/s)", low_idx, axes[0]),
        ("Moderate Wind (10-14 m/s)", med_idx, axes[1]),
        ("High Wind (> 18 m/s)", high_idx, axes[2]),
    ]
    
    for title, s_idx, ax in subsets:
        ax.plot(y_true_traj[s_idx, :, 0], y_true_traj[s_idx, :, 2], "k-", linewidth=2.2, label="Truth (RK4)")
        ax.plot(y_pred_traj_b[s_idx, :, 0], y_pred_traj_b[s_idx, :, 2], "b--", linewidth=2.0, label="Neural Surrogate")
        ax.plot(y_pred_traj_c[s_idx, :, 0], y_pred_traj_c[s_idx, :, 2], "g-.", linewidth=1.8, label="Hybrid Surrogate")
        ax.plot(base_traj[s_idx, :, 0], base_traj[s_idx, :, 2], "r:", alpha=0.6, label="Vacuum Ballistic")
        ax.set_title(f"{title}\n$U={wind_speeds[s_idx]:.1f}$ m/s", fontsize=11, fontweight="bold")
        ax.set_xlabel("Downwind Distance X (m)")
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Height Z (m)")
            ax.legend(fontsize=8, loc="upper right")
            
    plt.suptitle("Side-View Elevation Profiles (X-Z) Across Wind Regimes", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(fig_dir / "trajectory_side_views.png")
    plt.close()
    print("  -> Saved reports/figures/trajectory_side_views.png")

    # 3. Training & Validation Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    epochs = range(1, len(history_b["train_loss"]) + 1)
    
    ax1.plot(epochs, history_b["train_loss"], "b-", linewidth=2, label="Train Multi-Task Loss")
    ax1.plot(epochs, history_b["val_loss"], "r--", linewidth=2, label="Val Multi-Task Loss")
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multi-Task Loss (MSE)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, history_b["val_traj_mae"], "g-", linewidth=2, label="Val Trajectory MAE (m)")
    ax2.plot(epochs, history_b["val_landing_mae"], "m--", linewidth=2, label="Val Landing Distance MAE (m)")
    ax2.set_title("Physical Error Progression", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error (m)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "trajectory_loss_curves.png")
    plt.close()
    print("  -> Saved reports/figures/trajectory_loss_curves.png")

    # 4. Landing Scatter Plot (Truth vs Predicted)
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=300)
    true_land_dist = np.linalg.norm(y_true_land[:, :2], axis=-1)
    pred_land_dist = np.linalg.norm(y_pred_land_b[:, :2], axis=-1)
    
    max_val = max(np.max(true_land_dist), np.max(pred_land_dist)) * 1.05
    ax.scatter(true_land_dist, pred_land_dist, alpha=0.4, edgecolors="none", s=25, c="#1f77b4", label="Test Predictions (N=1500)")
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="Ideal 1:1 Identity")
    ax.fill_between([0, max_val], [0, max_val * 0.9], [0, max_val * 1.1], color="gray", alpha=0.15, label="±10% Relative Band")
    
    ax.set_title("Landing Distance Prediction: Neural Surrogate vs RK4 Truth", fontsize=12, fontweight="bold")
    ax.set_xlabel("True RK4 Landing Distance (m)", fontsize=11)
    ax.set_ylabel("Predicted Landing Distance (m)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    
    ax.text(0.65, 0.1, f"Test MAE: {test_metrics_b['landing_distance_mae_m']:.2f} m\nRMSE: {test_metrics_b['landing_distance_rmse_m']:.2f} m",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9), fontsize=10)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "landing_scatter_ml_vs_rk4.png")
    plt.close()
    print("  -> Saved reports/figures/landing_scatter_ml_vs_rk4.png")

    # 5. Trajectory Error Distributions over Normalized Flight Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    diff_pt = np.linalg.norm(y_pred_traj_b - y_true_traj, axis=-1) # [N, 50]
    time_fractions = np.linspace(0.0, 1.0, 50)
    
    mean_err_t = np.mean(diff_pt, axis=0)
    p25_err_t = np.percentile(diff_pt, 25, axis=0)
    p75_err_t = np.percentile(diff_pt, 75, axis=0)
    p95_err_t = np.percentile(diff_pt, 95, axis=0)
    
    ax1.plot(time_fractions, mean_err_t, "b-", linewidth=2, label="Mean Error")
    ax1.fill_between(time_fractions, p25_err_t, p75_err_t, color="blue", alpha=0.25, label="Interquartile Range (25-75%)")
    ax1.fill_between(time_fractions, p75_err_t, p95_err_t, color="blue", alpha=0.1, label="95th Percentile")
    ax1.set_title("Trajectory Error Accumulation along Flight Path", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Normalized Flight Step ($t / t_{land}$)")
    ax1.set_ylabel("Position Error (m)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.hist(diff_pt.flatten(), bins=50, color="#2ca02c", alpha=0.75, edgecolor="black", density=True)
    ax2.axvline(test_metrics_b["traj_3d_mae_m"], color="red", linestyle="--", linewidth=2, label=f"Mean: {test_metrics_b['traj_3d_mae_m']:.2f} m")
    ax2.set_title("Overall 3D Point Error Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Point-to-Point Euclidean Error (m)")
    ax2.set_ylabel("Probability Density")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "trajectory_error_distributions.png")
    plt.close()
    print("  -> Saved reports/figures/trajectory_error_distributions.png")

    # 6. In-Distribution vs Out-of-Distribution (OOD) Performance
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    categories = ["3D Trajectory MAE (m)", "Landing Distance MAE (m)", "Flight Time MAE (s)"]
    in_dist = [
        test_metrics_b["traj_3d_mae_m"],
        test_metrics_b["landing_distance_mae_m"],
        test_metrics_b["flight_time_mae_s"],
    ]
    ood = [
        ood_metrics_b["traj_3d_mae_m"],
        ood_metrics_b["landing_distance_mae_m"],
        ood_metrics_b["flight_time_mae_s"],
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, in_dist, width, label="In-Distribution Test (N=1500)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, ood, width, label="Out-of-Distribution OOD (N=500)", color="#d62728", alpha=0.85)
    
    for i, v in enumerate(in_dist):
        ax.text(i - width/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    for i, v in enumerate(ood):
        ax.text(i + width/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        
    ax.set_title("Generalization: In-Distribution vs Out-of-Distribution Extremes", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "ood_performance.png")
    plt.close()
    print("  -> Saved reports/figures/ood_performance.png")


def run_full_evaluation():
    device = get_device("auto")
    print(f"[Evaluation] Using device: {device}")

    # 1. Load data
    npz_path = "data/synthetic/trajectories_10k.npz"
    split_path = "data/manifests/trajectory_splits.json"
    summary_path = "data/synthetic/trajectories_10k_summary.csv"
    ood_path = "data/synthetic/trajectories_ood.npz"

    print(f"[Evaluation] Loading datasets...")
    raw = np.load(npz_path)
    inputs = raw["inputs"]
    times = raw["times"]
    trajectories = raw["trajectories"]
    landing_targets = raw["landing_targets"]

    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        id_to_idx = {row["trajectory_id"]: idx for idx, row in enumerate(reader)}

    with open(split_path, "r") as f:
        splits = json.load(f)

    test_idx = np.array([id_to_idx[tid] for tid in splits["test"]], dtype=int)
    test_inputs = inputs[test_idx]
    test_times = times[test_idx]
    y_true_traj = trajectories[test_idx]
    y_true_land = landing_targets[test_idx]

    # Load trained models
    model_b, stats_b = load_model_and_stats("checkpoints/best_trajectory_model.pt", "neural_surrogate", device)
    model_c, stats_c = load_model_and_stats("checkpoints/best_hybrid_surrogate.pt", "physics_hybrid", device)

    # 2. Compute Ballistic Baseline
    base_traj, base_landing = compute_ballistic_baseline(test_inputs, test_times)
    metrics_ballistic = compute_metrics(y_true_traj, base_traj, y_true_land, base_landing)

    # 3. Predict Model B (Neural Trajectory Surrogate)
    norm_test_x_b = (test_inputs - stats_b["x_mean"]) / stats_b["x_std"]
    with torch.no_grad():
        tensor_x_b = torch.from_numpy(norm_test_x_b).float().to(device)
        norm_pred_traj_b, norm_pred_land_b = model_b(tensor_x_b)
        y_pred_traj_b = norm_pred_traj_b.cpu().numpy() * stats_b["traj_std"] + stats_b["traj_mean"]
        y_pred_land_b = norm_pred_land_b.cpu().numpy() * stats_b["land_std"] + stats_b["land_mean"]
    metrics_b = compute_metrics(y_true_traj, y_pred_traj_b, y_true_land, y_pred_land_b)

    # 4. Predict Model C (Physics-Informed Hybrid Surrogate)
    norm_test_x_c = (test_inputs - stats_c["x_mean"]) / stats_c["x_std"]
    with torch.no_grad():
        tensor_x_c = torch.from_numpy(norm_test_x_c).float().to(device)
        t_base_traj = torch.from_numpy(base_traj).float().to(device)
        t_base_land = torch.from_numpy(base_landing).float().to(device)
        pred_traj_c, pred_land_c = model_c(tensor_x_c, t_base_traj, t_base_land)
        y_pred_traj_c = pred_traj_c.cpu().numpy()
        y_pred_land_c = pred_land_c.cpu().numpy()
    metrics_c = compute_metrics(y_true_traj, y_pred_traj_c, y_true_land, y_pred_land_c)

    # 5. Out-of-Distribution (OOD) Evaluation
    raw_ood = np.load(ood_path)
    ood_inputs = raw_ood["inputs"]
    ood_times = raw_ood["times"]
    ood_true_traj = raw_ood["trajectories"]
    ood_true_land = raw_ood["landing_targets"]

    norm_ood_x = (ood_inputs - stats_b["x_mean"]) / stats_b["x_std"]
    with torch.no_grad():
        tensor_ood_x = torch.from_numpy(norm_ood_x).float().to(device)
        norm_ood_traj_b, norm_ood_land_b = model_b(tensor_ood_x)
        ood_pred_traj_b = norm_ood_traj_b.cpu().numpy() * stats_b["traj_std"] + stats_b["traj_mean"]
        ood_pred_land_b = norm_ood_land_b.cpu().numpy() * stats_b["land_std"] + stats_b["land_mean"]
    ood_metrics_b = compute_metrics(ood_true_traj, ood_pred_traj_b, ood_true_land, ood_pred_land_b)

    # 6. Throughput Benchmark
    throughput_results = benchmark_throughput(model_b, stats_b, test_inputs, device=device)

    # 7. Real USDA Evaluation
    real_results = evaluate_on_real_usda(model_b, stats_b, device)

    # 8. Load training history for plotting
    with open("checkpoints/training_history.json", "r") as f:
        history_b = json.load(f)

    # 9. Generate figures
    fig_dir = Path("reports/figures")
    generate_diagnostic_figures(
        fig_dir=fig_dir,
        test_inputs=test_inputs,
        y_true_traj=y_true_traj,
        y_pred_traj_b=y_pred_traj_b,
        y_pred_traj_c=y_pred_traj_c,
        base_traj=base_traj,
        y_true_land=y_true_land,
        y_pred_land_b=y_pred_land_b,
        history_b=history_b,
        test_metrics_b=metrics_b,
        ood_metrics_b=ood_metrics_b,
    )

    # 10. Compile comprehensive evaluation report JSON
    final_report = {
        "dataset_summary": {
            "total_trajectories": len(inputs),
            "train_trajectories": len(splits["train"]),
            "val_trajectories": len(splits["val"]),
            "test_trajectories": len(splits["test"]),
            "ood_trajectories": len(ood_inputs),
        },
        "test_metrics": {
            "analytical_ballistic": metrics_ballistic,
            "model_b_neural_surrogate": metrics_b,
            "model_c_physics_hybrid": metrics_c,
        },
        "ood_metrics": {
            "model_b_neural_surrogate": ood_metrics_b,
        },
        "throughput_benchmark": throughput_results,
        "real_usda_evaluation": {
            "num_experimental_runs": real_results.get("num_runs", 0),
            "mae_m": real_results.get("real_mae_m", 0.0),
            "rmse_m": real_results.get("real_rmse_m", 0.0),
        }
    }

    report_json_path = "reports/trajectory_evaluation_metrics.json"
    with open(report_json_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Evaluation] Saved complete evaluation metrics to {report_json_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Model Architecture':<30} | {'3D Traj MAE':<12} | {'Land Dist MAE':<14} | {'Flight Time MAE':<15}")
    print("="*80)
    print(f"{'Analytical Ballistic':<30} | {metrics_ballistic['traj_3d_mae_m']:<12.3f} | {metrics_ballistic['landing_distance_mae_m']:<14.3f} | {metrics_ballistic['flight_time_mae_s']:<15.3f}")
    print(f"{'Model B (Neural Surrogate)':<30} | {metrics_b['traj_3d_mae_m']:<12.3f} | {metrics_b['landing_distance_mae_m']:<14.3f} | {metrics_b['flight_time_mae_s']:<15.3f}")
    print(f"{'Model C (Physics Hybrid)':<30} | {metrics_c['traj_3d_mae_m']:<12.3f} | {metrics_c['landing_distance_mae_m']:<14.3f} | {metrics_c['flight_time_mae_s']:<15.3f}")
    print(f"{'Model B on OOD Extremes':<30} | {ood_metrics_b['traj_3d_mae_m']:<12.3f} | {ood_metrics_b['landing_distance_mae_m']:<14.3f} | {ood_metrics_b['flight_time_mae_s']:<15.3f}")
    print("="*80)
    print(f"Inference Acceleration: {throughput_results['speedup_ratio']:.1f}x speedup ({throughput_results['neural_throughput_traj_per_sec']:.1f} traj/s vs {throughput_results['rk4_throughput_traj_per_sec']:.1f} traj/s)")
    print("="*80)

    return final_report


if __name__ == "__main__":
    run_full_evaluation()
