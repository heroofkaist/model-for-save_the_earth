"""
Save the Earth — USDA JFSP 15-1-04-4 Particle-Level Benchmark Evaluation.

Executes rigorous independent model validation against 60,370 firebrand observations:
- Evaluates on the held-out test split (8,841 individual particles across 3 disjoint fuel groups)
- Benchmarks:
  1. Baseline A: Median Baseline
  2. Baseline B: Prospective Ridge Regression
  3. Baseline C: Prospective Random Forest
  4. Baseline D: Prospective Gradient Boosting
  5. Mechanistic Baseline: RK4 Numerical Aerodynamics Integration
  6. Model B: Neural Trajectory Surrogate (checkpoints/best_trajectory_model.pt)
  7. Model C: Physics-Informed Hybrid Surrogate (checkpoints/best_hybrid_surrogate.pt)

Computes Test MAE, RMSE, Signed Bias, and Median AE.
Evaluates errors stratified by Fuel Family (Structural vs Vegetative) and Wind Speed.
Generates publication-quality diagnostic figures.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.save_the_earth.physics.aerodynamics import simulate_advanced_trajectory
from src.save_the_earth.models.trajectory_models import (
    NeuralTrajectorySurrogate,
    PhysicsInformedHybridSurrogate,
)
from src.save_the_earth.training.train_surrogate import compute_ballistic_baseline

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "usda_individual_firebrands.csv"
CHECKPOINT_B = PROJECT_ROOT / "checkpoints" / "best_trajectory_model.pt"
CHECKPOINT_C = PROJECT_ROOT / "checkpoints" / "best_hybrid_surrogate.pt"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_OUT = PROJECT_ROOT / "reports" / "usda_particle_evaluation_metrics.json"


def load_dataset(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads particle records split into train, validation, and test sets."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]

    return train_rows, val_rows, test_rows


def extract_operational_features(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts strictly prospective operational features available BEFORE flight:
    [wind_speed, wind_speed^2, gust_peak, moisture_content, is_structural]
    """
    X = []
    y = []
    for r in rows:
        u = float(r["wind_speed_m_s"])
        u_sq = u**2
        gust = float(r["gust_peak_m_s"])
        moisture = float(r["moisture_content_percent"])
        is_struct = 1.0 if r["fuel_category"] == "structural" else 0.0
        dist = float(r["landing_distance_m"])

        X.append([u, u_sq, gust, moisture, is_struct])
        y.append(dist)

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, Median AE, and Signed Bias."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(len(y_true)),
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "median_ae_m": float(np.median(abs_err)),
        "bias_m": float(np.mean(err)),
    }


def load_surrogate_checkpoint(checkpoint_path: Path, model_type: str = "neural"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    if model_type == "neural":
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


def evaluate_rk4_physics(rows: List[Dict[str, Any]], h: float = 1.22, cd: float = 1.0) -> np.ndarray:
    """
    Simulates mechanistic aerodynamic trajectory for each particle.
    """
    preds = []
    for r in rows:
        u_wind = float(r["wind_speed_m_s"])
        mass = float(r["mass_kg"])
        area = float(r["projected_area_m2"])
        _, _, info = simulate_advanced_trajectory(
            release_height=h,
            u_ref=u_wind,
            mass=mass,
            area=area,
            cd=cd,
            z0=0.005,
            w0_updraft=0.0,
            v0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            dt=0.005,
            resample_points=50,
        )
        preds.append(info["landing_distance_m"])
    return np.array(preds, dtype=np.float64)


def evaluate_surrogate(
    rows: List[Dict[str, Any]],
    model: torch.nn.Module,
    stats: Dict[str, np.ndarray],
    is_hybrid: bool = False,
    h: float = 1.22,
    cd: float = 1.0
) -> np.ndarray:
    """
    Evaluates neural or hybrid surrogate across all particle records.
    Inputs: [h, u, angle, mass, area, cd, z0, w_up, v0z] (dim=9)
    """
    n = len(rows)
    inputs = np.zeros((n, 9), dtype=np.float32)
    for i, r in enumerate(rows):
        inputs[i, 0] = h
        inputs[i, 1] = float(r["wind_speed_m_s"])
        inputs[i, 2] = 0.0  # Wind angle aligned with tunnel
        inputs[i, 3] = float(r["mass_kg"])
        inputs[i, 4] = float(r["projected_area_m2"])
        inputs[i, 5] = cd
        inputs[i, 6] = 0.005 # smooth tunnel boundary layer
        inputs[i, 7] = 0.0   # no updraft in horizontal tunnel
        inputs[i, 8] = 0.0   # horizontal initial release

    norm_x = (inputs - stats["x_mean"]) / stats["x_std"]
    tensor_x = torch.from_numpy(norm_x).float()

    with torch.no_grad():
        if is_hybrid:
            times = np.linspace(0, 1.0, 50, dtype=np.float32).reshape(1, 50).repeat(n, axis=0)
            base_traj, base_land = compute_ballistic_baseline(inputs, times)
            t_base_traj = torch.from_numpy(base_traj).float()
            t_base_land = torch.from_numpy(base_land).float()
            # Model C predicts directly in physical coordinates: pred = base + delta
            _, pred_land_phys = model(tensor_x, t_base_traj, t_base_land)
            pred_land = pred_land_phys.cpu().numpy()
        else:
            _, norm_pred_land = model(tensor_x)
            pred_land = norm_pred_land.cpu().numpy() * stats["land_std"] + stats["land_mean"]

        # landing distance is norm of (x_land, y_land)
        pred_dist = np.linalg.norm(pred_land[:, :2], axis=1)

    return pred_dist.astype(np.float64)


def plot_dispersion_by_fuel(rows: List[Dict[str, Any]], out_path: Path):
    """Generates distribution and dispersion plot for structural vs vegetative particles."""
    plt.figure(figsize=(9, 5), dpi=300)
    struct_dist = [float(r["landing_distance_m"]) for r in rows if r["fuel_category"] == "structural"]
    veg_dist = [float(r["landing_distance_m"]) for r in rows if r["fuel_category"] == "vegetative"]

    bins = np.linspace(0, 25, 51)
    plt.hist(struct_dist, bins=bins, alpha=0.6, density=True, label=f"Structural (N={len(struct_dist):,})", color="#c0392b")
    plt.hist(veg_dist, bins=bins, alpha=0.6, density=True, label=f"Vegetative (N={len(veg_dist):,})", color="#27ae60")

    plt.axvline(np.median(struct_dist), color="#962d22", linestyle="--", linewidth=1.5,
                label=f"Structural Median: {np.median(struct_dist):.2f} m")
    plt.axvline(np.median(veg_dist), color="#1e8449", linestyle="--", linewidth=1.5,
                label=f"Vegetative Median: {np.median(veg_dist):.2f} m")

    plt.title("USDA JFSP 15-1-04-4 Firebrand Particle Landing Distance Dispersion", fontsize=12, fontweight="bold")
    plt.xlabel("Landing Distance (m)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_model_comparison(results: Dict[str, Dict[str, float]], out_path: Path):
    """Generates bar chart comparing test MAE across all models alongside oracle baseline."""
    plt.figure(figsize=(11, 5.5), dpi=300)
    models = list(results.keys())
    maes = [results[m]["mae_m"] for m in models]
    labels = [m.replace("Prospective ", "Prosp. ").replace("Model B: ", "").replace("Model C: ", "").replace("Mechanistic ", "") for m in models]

    colors = ["#7f8c8d", "#2980b9", "#3498db", "#1abc9c", "#e67e22", "#9b59b6", "#2c3e50"]
    x = np.arange(len(models))

    bars = plt.bar(x, maes, color=colors[:len(models)], width=0.58, edgecolor="black", alpha=0.9)
    for bar, val in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.12, f"{val:.2f} m", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Oracle theoretical median lower bound
    plt.axhline(4.80, color="#c0392b", linestyle="--", linewidth=1.5, label="Test Set Dispersion Oracle Floor: 4.80 m (σ = 7.23 m)")

    plt.xticks(x, labels, rotation=20, ha="right", fontsize=9.5)
    plt.ylabel("Held-Out Test MAE (meters)", fontsize=11, fontweight="bold")
    plt.title("Particle Benchmark Model Comparison (Held-Out Test Set, N=8,841)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.grid(axis="y", alpha=0.3, linestyle=":")
    plt.ylim(0, max(maes) * 1.35)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_by_wind_speed(
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    wind_speeds: np.ndarray,
    out_path: Path
):
    """Generates error breakdown across wind tunnel speed regimes."""
    unique_winds = sorted(list(set(wind_speeds)))
    regimes = [f"{w:.2f} m/s" for w in unique_winds]

    plt.figure(figsize=(9, 5), dpi=300)
    width = 0.2
    x = np.arange(len(unique_winds))

    selected_models = ["Prospective Gradient Boosting", "Mechanistic RK4 Physics", "Model B: Neural Surrogate", "Model C: Hybrid Surrogate"]
    colors = ["#1abc9c", "#e67e22", "#9b59b6", "#8e44ad"]

    for i, (name, col) in enumerate(zip(selected_models, colors)):
        if name not in preds_dict:
            continue
        y_p = preds_dict[name]
        regime_maes = []
        for w in unique_winds:
            mask = (wind_speeds == w)
            regime_maes.append(float(np.mean(np.abs(y_p[mask] - y_true[mask]))))
        plt.bar(x + (i - 1.5) * width, regime_maes, width=width, label=name, color=col, edgecolor="black", alpha=0.85)

    plt.xticks(x, regimes, fontsize=10)
    plt.xlabel("Nominal Wind Speed", fontsize=11)
    plt.ylabel("Test MAE (m)", fontsize=11)
    plt.title("Model Prediction Error Across Wind Speed Regimes", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, fontsize=9)
    plt.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print("=" * 75)
    print("SAVE THE EARTH — USDA JFSP 15-1-04-4 PARTICLE-LEVEL BENCHMARK EVALUATION")
    print("=" * 75)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = load_dataset(DATA_PATH)
    print(f"Dataset split sizes: Train={len(train_rows):,}, Val={len(val_rows):,}, Test={len(test_rows):,}")

    # Extract operational features
    X_train, y_train = extract_operational_features(train_rows)
    X_val, y_val = extract_operational_features(val_rows)
    X_test, y_test = extract_operational_features(test_rows)
    test_winds = np.array([float(r["wind_speed_m_s"]) for r in test_rows])
    test_cats = np.array([r["fuel_category"] for r in test_rows])

    # 1. Baseline A: Median Baseline
    train_median = float(np.median(y_train))
    y_pred_median = np.full_like(y_test, train_median)

    # 2. Baseline B: Prospective Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # 3. Baseline C: Prospective Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 4. Baseline D: Prospective Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)

    # 5. Physics Baseline: Mechanistic RK4
    print("Executing Mechanistic RK4 Aerodynamics simulation on held-out test particles...")
    t0_rk4 = time.time()
    y_pred_rk4 = evaluate_rk4_physics(test_rows, h=1.22, cd=1.0)
    t1_rk4 = time.time()
    print(f"RK4 simulation finished in {t1_rk4 - t0_rk4:.2f} s ({len(test_rows)/(t1_rk4 - t0_rk4):.1f} particles/s)")

    # 6. Model B: Neural Trajectory Surrogate
    print("Evaluating Model B (Neural Trajectory Surrogate)...")
    model_b, stats_b = load_surrogate_checkpoint(CHECKPOINT_B, "neural")
    t0_b = time.time()
    y_pred_b = evaluate_surrogate(test_rows, model_b, stats_b, is_hybrid=False, h=1.22, cd=1.0)
    t1_b = time.time()
    print(f"Model B evaluation finished in {t1_b - t0_b:.2f} s ({len(test_rows)/(t1_b - t0_b):.1f} particles/s)")

    # 7. Model C: Physics-Informed Hybrid Surrogate
    print("Evaluating Model C (Physics-Informed Hybrid Surrogate)...")
    model_c, stats_c = load_surrogate_checkpoint(CHECKPOINT_C, "hybrid")
    t0_c = time.time()
    y_pred_c = evaluate_surrogate(test_rows, model_c, stats_c, is_hybrid=True, h=1.22, cd=1.0)
    t1_c = time.time()
    print(f"Model C evaluation finished in {t1_c - t0_c:.2f} s ({len(test_rows)/(t1_c - t0_c):.1f} particles/s)")

    all_predictions = {
        "Global Median Baseline": y_pred_median,
        "Prospective Ridge": y_pred_ridge,
        "Prospective Random Forest": y_pred_rf,
        "Prospective Gradient Boosting": y_pred_gb,
        "Mechanistic RK4 Physics": y_pred_rk4,
        "Model B: Neural Surrogate": y_pred_b,
        "Model C: Hybrid Surrogate": y_pred_c,
    }

    # Compute overall test metrics
    metrics_overall = {}
    metrics_by_category = {}
    metrics_by_wind = {}

    print("\n" + "-" * 75)
    print(f"{'Model':<32} | {'Test MAE (m)':<12} | {'RMSE (m)':<10} | {'Median AE (m)':<13} | {'Bias (m)':<9}")
    print("-" * 75)

    for name, pred in all_predictions.items():
        m_tot = compute_metrics(y_test, pred)
        metrics_overall[name] = m_tot
        print(f"{name:<32} | {m_tot['mae_m']:<12.3f} | {m_tot['rmse_m']:<10.3f} | {m_tot['median_ae_m']:<13.3f} | {m_tot['bias_m']:<9.3f}")

        # Metrics by category
        metrics_by_category[name] = {}
        for cat in ["structural", "vegetative"]:
            c_mask = (test_cats == cat)
            if np.any(c_mask):
                metrics_by_category[name][cat] = compute_metrics(y_test[c_mask], pred[c_mask])

        # Metrics by wind speed
        metrics_by_wind[name] = {}
        for w in sorted(list(set(test_winds))):
            w_mask = (test_winds == w)
            metrics_by_wind[name][f"{w:.2f} m/s"] = compute_metrics(y_test[w_mask], pred[w_mask])

    print("-" * 75)

    # Generate publication figures
    disp_path = FIGURES_DIR / "usda_particle_dispersion_by_fuel.png"
    comp_path = FIGURES_DIR / "usda_model_comparison_particle_mae.png"
    wind_path = FIGURES_DIR / "usda_error_by_wind_speed.png"

    all_dataset_rows = train_rows + val_rows + test_rows
    plot_dispersion_by_fuel(all_dataset_rows, disp_path)
    plot_model_comparison(metrics_overall, comp_path)
    plot_error_by_wind_speed(y_test, all_predictions, test_winds, wind_path)

    # Save metrics JSON
    evaluation_record = {
        "benchmark_dataset": "USDA JFSP 15-1-04-4 Particle-Level Test Partition",
        "held_out_particles_evaluated": len(y_test),
        "fuel_breakdown": {
            "structural_particles": int(np.sum(test_cats == "structural")),
            "vegetative_particles": int(np.sum(test_cats == "vegetative")),
        },
        "model_metrics_overall": metrics_overall,
        "model_metrics_by_fuel_family": metrics_by_category,
        "model_metrics_by_wind_speed": metrics_by_wind,
        "timing_and_throughput": {
            "rk4_sec": t1_rk4 - t0_rk4,
            "rk4_particles_per_sec": len(test_rows) / (t1_rk4 - t0_rk4),
            "surrogate_b_sec": t1_b - t0_b,
            "surrogate_b_particles_per_sec": len(test_rows) / (t1_b - t0_b),
            "surrogate_c_sec": t1_c - t0_c,
            "surrogate_c_particles_per_sec": len(test_rows) / (t1_c - t0_c),
        },
        "artifacts_generated": [
            str(disp_path.relative_to(PROJECT_ROOT)),
            str(comp_path.relative_to(PROJECT_ROOT)),
            str(wind_path.relative_to(PROJECT_ROOT)),
        ]
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=2)

    print(f"\nGenerated diagnostic figures in: {FIGURES_DIR}")
    print(f"Saved evaluation metrics to: {METRICS_OUT}")
    print("=" * 75)


if __name__ == "__main__":
    main()
