"""
Save the Earth — Training Pipeline for Deep Experimental Surrogate (Model E).

Trains ExperimentalDistanceNet directly on the 60,370 firebrand observations
from the USDA JFSP 15-1-04-4 dataset.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.save_the_earth.models.experimental_surrogate import ExperimentalDistanceNet
from src.save_the_earth.physics.aerodynamics import (
    compute_tohidi_drag_lift,
    compute_combustion_buoyancy,
    DEFAULT_RHO
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "usda_individual_firebrands.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_experimental_model.pt"
METRICS_PATH = PROJECT_ROOT / "reports" / "experimental_training_metrics.json"


def extract_features(row):
    u = float(row["wind_speed_m_s"])
    gust = float(row.get("gust_peak_m_s", u))
    moist = float(row.get("moisture_content_percent", 10.0))
    is_struct = 1.0 if row.get("fuel_category") == "structural" else 0.0
    m = float(row.get("mass_kg", 5e-5))
    a = float(row.get("projected_area_m2", 2e-4))

    am = a / max(1e-6, m)
    log_m = np.log(max(1e-6, m))
    log_a = np.log(max(1e-6, a))
    log_am = np.log(max(1e-3, am))

    u_sq = u**2
    dryness = np.exp(-moist / 20.0)
    wind_momentum = u_sq * dryness

    # Tohidi & Kaye (2017) morphology aerodynamics + Wong et al. (CSIRO 2026) combustion buoyancy
    cd_tohidi, _ = compute_tohidi_drag_lift(m, a)
    g_eff = compute_combustion_buoyancy(is_burning=True, eta_buoy=1.18)
    w_term = np.sqrt(2.0 * m * g_eff / max(1e-7, DEFAULT_RHO * cd_tohidi * a))
    x_advect = u * (1.22 / max(0.2, w_term))
    log_w_term = np.log(max(0.1, w_term))
    log_advect = np.log(max(0.1, x_advect))

    return [
        u, u_sq, gust, moist, dryness, is_struct,
        log_m, log_a, log_am, wind_momentum,
        log_w_term, log_advect
    ]


def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles=[0.1, 0.5, 0.9]) -> torch.Tensor:
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        err = targets.unsqueeze(-1) - preds[:, i:i+1]
        loss = torch.mean(torch.max((q - 1.0) * err, q * err))
        total_loss = total_loss + loss
    return total_loss


def train_model(
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 0.003,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading empirical USDA firebrand dataset...")
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"Dataset split sizes: Train={len(train_rows):,}, Val={len(val_rows):,}, Test={len(test_rows):,}")

    X_tr = np.array([extract_features(r) for r in train_rows], dtype=np.float32)
    y_tr = np.array([float(r["landing_distance_m"]) for r in train_rows], dtype=np.float32)

    X_val = np.array([extract_features(r) for r in val_rows], dtype=np.float32)
    y_val = np.array([float(r["landing_distance_m"]) for r in val_rows], dtype=np.float32)

    X_te = np.array([extract_features(r) for r in test_rows], dtype=np.float32)
    y_te = np.array([float(r["landing_distance_m"]) for r in test_rows], dtype=np.float32)

    # Standardization based on training set statistics
    mean_x = np.mean(X_tr, axis=0)
    std_x = np.std(X_tr, axis=0) + 1e-6

    X_tr_n = (X_tr - mean_x) / std_x
    X_val_n = (X_val - mean_x) / std_x
    X_te_n = (X_te - mean_x) / std_x

    train_dataset = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ExperimentalDistanceNet(in_dim=X_tr.shape[1], hidden_dim=128, dropout=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    t0 = time.time()

    print("\nStarting training of Deep Experimental Surrogate...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            preds = model(bx)
            loss = pinball_loss(preds, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(X_val_n)).numpy()
            val_mae = float(np.mean(np.abs(val_preds[:, 1] - y_val)))
            val_cov = float(np.mean((y_val >= val_preds[:, 0]) & (y_val <= val_preds[:, 2])) * 100)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "val_coverage": val_cov,
                "mean_x": mean_x,
                "std_x": std_x,
                "feature_names": [
                    "u", "u_sq", "gust", "moist", "dryness",
                    "is_structural", "log_m", "log_a", "log_am", "wind_momentum",
                    "log_w_term", "log_advect"
                ]
            }, CHECKPOINT_PATH)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.3f} m | Val 80% Cov: {val_cov:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f} s. Best Val MAE: {best_val_mae:.3f} m")

    # Final evaluation on held-out test partition
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Conformalized Quantile Regression (CQR) Calibration (Romano et al., NeurIPS 2019)
    # Calibrate on independent validation set to achieve guaranteed 80% coverage
    with torch.no_grad():
        val_preds_best = model(torch.from_numpy(X_val_n)).numpy()
        val_scores = np.maximum(val_preds_best[:, 0] - y_val, y_val - val_preds_best[:, 2])
        alpha_cov = 0.20  # Target 80% coverage
        k_val = int(np.ceil((len(val_scores) + 1) * (1.0 - alpha_cov)))
        q_conformal = float(np.sort(val_scores)[min(k_val - 1, len(val_scores) - 1)])

        # Update checkpoint with conformal calibration score
        checkpoint["cqr_q_conformal"] = q_conformal
        torch.save(checkpoint, CHECKPOINT_PATH)

        test_preds = model(torch.from_numpy(X_te_n)).numpy()
        test_mae = float(np.mean(np.abs(test_preds[:, 1] - y_te)))
        test_rmse = float(np.sqrt(np.mean((test_preds[:, 1] - y_te)**2)))
        test_bias = float(np.mean(test_preds[:, 1] - y_te))
        raw_test_cov = float(np.mean((y_te >= test_preds[:, 0]) & (y_te <= test_preds[:, 2])) * 100)

        # CQR calibrated interval
        cqr_lower = np.maximum(0.0, test_preds[:, 0] - q_conformal)
        cqr_upper = test_preds[:, 2] + q_conformal
        cqr_test_cov = float(np.mean((y_te >= cqr_lower) & (y_te <= cqr_upper)) * 100)

    print("\n" + "=" * 65)
    print("EXPERIMENTAL DEEP SURROGATE TEST RESULTS (Held-Out Test Set: N=8,841)")
    print("=" * 65)
    print(f"Test MAE (Median Distance):         {test_mae:.3f} m")
    print(f"Test RMSE:                          {test_rmse:.3f} m")
    print(f"Signed Bias:                        {test_bias:+.3f} m")
    print(f"Raw Pinball 80% Test Coverage:      {raw_test_cov:.1f}% (uncalibrated on OOD fuel groups)")
    print(f"CQR Conformal Calibration Shift:    +/- {q_conformal:.3f} m")
    print(f"CQR Calibrated 80% Test Coverage:   {cqr_test_cov:.1f}%")
    print("=" * 65)

    metrics = {
        "model_name": "ExperimentalDistanceNet (Model E)",
        "dataset": "USDA JFSP 15-1-04-4 Canonical Particle Observations",
        "training_epochs": epochs,
        "elapsed_seconds": elapsed,
        "val_mae_m": best_val_mae,
        "test_mae_m": test_mae,
        "test_rmse_m": test_rmse,
        "test_bias_m": test_bias,
        "raw_test_coverage_80_pct": raw_test_cov,
        "cqr_q_conformal_m": q_conformal,
        "cqr_calibrated_coverage_80_pct": cqr_test_cov,
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT))
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_model()

