"""
Save the Earth — Trajectory Surrogate Training Pipeline.

Trains Model B (NeuralTrajectorySurrogate) and Model C (PhysicsInformedHybridSurrogate)
on the 10,000 firebrand trajectory dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.save_the_earth.models.trajectory_models import (
    NeuralTrajectorySurrogate,
    PhysicsInformedHybridSurrogate,
)


class TrajectoryDataset(Dataset):
    def __init__(self, inputs, times, trajectories, landing_targets, base_traj=None, base_landing=None):
        self.inputs = torch.from_numpy(inputs).float()
        self.times = torch.from_numpy(times).float()
        self.trajectories = torch.from_numpy(trajectories).float()
        self.landing_targets = torch.from_numpy(landing_targets).float()
        self.base_traj = torch.from_numpy(base_traj).float() if base_traj is not None else None
        self.base_landing = torch.from_numpy(base_landing).float() if base_landing is not None else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {
            "inputs": self.inputs[idx],
            "times": self.times[idx],
            "trajectories": self.trajectories[idx],
            "landing_targets": self.landing_targets[idx],
        }
        if self.base_traj is not None:
            item["base_traj"] = self.base_traj[idx]
        if self.base_landing is not None:
            item["base_landing"] = self.base_landing[idx]
        return item


def compute_ballistic_baseline(inputs: np.ndarray, times: np.ndarray, g: float = 9.80665):
    """
    Computes analytical vacuum ballistic baseline trajectory for each sample.
    inputs: [h, u_wind, wind_angle, mass, area, cd, z0, w_updraft, v0z]
    """
    N, num_pts = times.shape
    h = inputs[:, 0:1] # [N, 1]
    u = inputs[:, 1:2]
    angle = inputs[:, 2:3]
    v0z = inputs[:, 8:9]

    # Ballistic landing time
    # z(t) = h + v0z*t - 0.5*g*t^2 = 0 => t_land = (v0z + sqrt(v0z^2 + 2gh)) / g
    disc = np.maximum(0.0, v0z**2 + 2 * g * h)
    t_land = (v0z + np.sqrt(disc)) / g

    u_x = u * np.cos(angle)
    u_y = u * np.sin(angle)

    x_land = u_x * t_land
    y_land = u_y * t_land
    base_landing = np.concatenate([x_land, y_land, t_land], axis=1) # [N, 3]

    # Trajectories across time steps
    t_expanded = times[:, :, np.newaxis] # [N, num_pts, 1]
    u_x_exp = u_x[:, np.newaxis, :] # [N, 1, 1]
    u_y_exp = u_y[:, np.newaxis, :]
    h_exp = h[:, np.newaxis, :]
    v0z_exp = v0z[:, np.newaxis, :]

    x_t = u_x_exp * t_expanded
    y_t = u_y_exp * t_expanded
    z_t = np.maximum(0.0, h_exp + v0z_exp * t_expanded - 0.5 * g * (t_expanded**2))

    base_traj = np.concatenate([x_t, y_t, z_t], axis=2) # [N, num_pts, 3]
    return base_traj.astype(np.float32), base_landing.astype(np.float32)


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested_device)


def compute_smoothness_loss(traj_pred: torch.Tensor) -> torch.Tensor:
    """Second-order finite difference smoothness penalty along trajectory."""
    # traj_pred: [B, N, 3]
    diff2 = traj_pred[:, 2:, :] - 2 * traj_pred[:, 1:-1, :] + traj_pred[:, :-2, :]
    return torch.mean(diff2**2)


def train_model(config_path: str = "configs/training.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    print(f"[Training] Using compute device: {device}")

    # 1. Load data
    npz_path = cfg["data"]["dataset_npz"]
    split_path = cfg["data"]["split_json"]
    summary_path = cfg["data"].get("summary_csv", "data/synthetic/trajectories_10k_summary.csv")
    print(f"[Training] Loading dataset from {npz_path}...")
    raw_data = np.load(npz_path)

    inputs = raw_data["inputs"]
    times = raw_data["times"]
    trajectories = raw_data["trajectories"]
    landing_targets = raw_data["landing_targets"]

    # Build mapping from trajectory_id to row index
    import csv
    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        id_to_idx = {row["trajectory_id"]: idx for idx, row in enumerate(reader)}

    with open(split_path, "r") as f:
        splits = json.load(f)

    train_idx = np.array([id_to_idx[tid] for tid in splits["train"]], dtype=int)
    val_idx = np.array([id_to_idx[tid] for tid in splits["val"]], dtype=int)
    test_idx = np.array([id_to_idx[tid] for tid in splits["test"]], dtype=int)

    print(f"[Training] Partitions: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 2. Strict Normalization based on TRAIN ONLY
    x_mean = np.mean(inputs[train_idx], axis=0, keepdims=True)
    x_std = np.std(inputs[train_idx], axis=0, keepdims=True) + 1e-7

    traj_mean = np.mean(trajectories[train_idx], axis=(0, 1), keepdims=True) # [1, 1, 3]
    traj_std = np.std(trajectories[train_idx], axis=(0, 1), keepdims=True) + 1e-7

    land_mean = np.mean(landing_targets[train_idx], axis=0, keepdims=True) # [1, 3]
    land_std = np.std(landing_targets[train_idx], axis=0, keepdims=True) + 1e-7

    # Normalize inputs
    norm_inputs = (inputs - x_mean) / x_std

    # Analytical ballistic baselines for Hybrid model
    base_traj, base_landing = compute_ballistic_baseline(inputs, times)

    # Datasets
    train_dataset = TrajectoryDataset(
        norm_inputs[train_idx], times[train_idx], trajectories[train_idx],
        landing_targets[train_idx], base_traj[train_idx], base_landing[train_idx]
    )
    val_dataset = TrajectoryDataset(
        norm_inputs[val_idx], times[val_idx], trajectories[val_idx],
        landing_targets[val_idx], base_traj[val_idx], base_landing[val_idx]
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model_type = cfg["architecture"].get("model_type", "neural_surrogate")
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

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = cfg["training"]["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(cfg["training"]["min_lr"])
    )

    save_dir = cfg["checkpointing"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, cfg["checkpointing"]["model_filename"])
    history_path = os.path.join(save_dir, cfg["checkpointing"]["log_filename"])

    traj_w = cfg["loss"]["trajectory_pos_weight"]
    land_w = cfg["loss"]["landing_coord_weight"]
    time_w = cfg["loss"]["flight_time_weight"]
    smooth_w = cfg["loss"]["smoothness_weight"]

    # Target normalizers as tensors on device
    t_mean_t = torch.tensor(traj_mean, device=device, dtype=torch.float32)
    t_std_t = torch.tensor(traj_std, device=device, dtype=torch.float32)
    l_mean_t = torch.tensor(land_mean, device=device, dtype=torch.float32)
    l_std_t = torch.tensor(land_std, device=device, dtype=torch.float32)

    history = {
        "train_loss": [], "val_loss": [],
        "train_traj_mae": [], "val_traj_mae": [],
        "train_landing_mae": [], "val_landing_mae": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    print(f"[Training] Commencing training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_traj_mae = 0.0
        total_land_mae = 0.0
        count = 0

        for batch in train_loader:
            bx = batch["inputs"].to(device)
            by_traj = batch["trajectories"].to(device)
            by_land = batch["landing_targets"].to(device)

            optimizer.zero_grad()
            if model_type == "neural_surrogate":
                norm_pred_traj, norm_pred_land = model(bx)
                pred_traj = norm_pred_traj * t_std_t + t_mean_t
                pred_land = norm_pred_land * l_std_t + l_mean_t
            else:
                b_base_traj = batch["base_traj"].to(device)
                b_base_land = batch["base_landing"].to(device)
                pred_traj, pred_land = model(bx, b_base_traj, b_base_land)
                norm_pred_traj = (pred_traj - t_mean_t) / t_std_t
                norm_pred_land = (pred_land - l_mean_t) / l_std_t

            norm_true_traj = (by_traj - t_mean_t) / t_std_t
            norm_true_land = (by_land - l_mean_t) / l_std_t

            loss_traj = F.mse_loss(norm_pred_traj, norm_true_traj)
            loss_land_xy = F.mse_loss(norm_pred_land[:, :2], norm_true_land[:, :2])
            loss_time = F.mse_loss(norm_pred_land[:, 2], norm_true_land[:, 2])
            loss_smooth = compute_smoothness_loss(pred_traj)

            loss = (
                traj_w * loss_traj
                + land_w * loss_land_xy
                + time_w * loss_time
                + smooth_w * loss_smooth
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()

            # Physical MAE tracking (unnormalized meters)
            with torch.no_grad():
                traj_mae = torch.mean(torch.abs(pred_traj - by_traj)).item()
                land_dist_true = torch.sqrt(torch.sum(by_land[:, :2]**2, dim=-1))
                land_dist_pred = torch.sqrt(torch.sum(pred_land[:, :2]**2, dim=-1))
                land_mae = torch.mean(torch.abs(land_dist_pred - land_dist_true)).item()

            total_loss += loss.item() * len(bx)
            total_traj_mae += traj_mae * len(bx)
            total_land_mae += land_mae * len(bx)
            count += len(bx)

        scheduler.step()
        train_loss = total_loss / count
        train_traj_mae = total_traj_mae / count
        train_land_mae = total_land_mae / count

        # Validation phase
        model.eval()
        v_loss = 0.0
        v_traj_mae = 0.0
        v_land_mae = 0.0
        v_count = 0

        with torch.no_grad():
            for batch in val_loader:
                bx = batch["inputs"].to(device)
                by_traj = batch["trajectories"].to(device)
                by_land = batch["landing_targets"].to(device)

                if model_type == "neural_surrogate":
                    norm_pred_traj, norm_pred_land = model(bx)
                    pred_traj = norm_pred_traj * t_std_t + t_mean_t
                    pred_land = norm_pred_land * l_std_t + l_mean_t
                else:
                    b_base_traj = batch["base_traj"].to(device)
                    b_base_land = batch["base_landing"].to(device)
                    pred_traj, pred_land = model(bx, b_base_traj, b_base_land)
                    norm_pred_traj = (pred_traj - t_mean_t) / t_std_t
                    norm_pred_land = (pred_land - l_mean_t) / l_std_t

                norm_true_traj = (by_traj - t_mean_t) / t_std_t
                norm_true_land = (by_land - l_mean_t) / l_std_t

                loss_traj = F.mse_loss(norm_pred_traj, norm_true_traj)
                loss_land_xy = F.mse_loss(norm_pred_land[:, :2], norm_true_land[:, :2])
                loss_time = F.mse_loss(norm_pred_land[:, 2], norm_true_land[:, 2])

                val_loss_batch = traj_w * loss_traj + land_w * loss_land_xy + time_w * loss_time

                traj_mae = torch.mean(torch.abs(pred_traj - by_traj)).item()
                land_dist_true = torch.sqrt(torch.sum(by_land[:, :2]**2, dim=-1))
                land_dist_pred = torch.sqrt(torch.sum(pred_land[:, :2]**2, dim=-1))
                land_mae = torch.mean(torch.abs(land_dist_pred - land_dist_true)).item()

                v_loss += val_loss_batch.item() * len(bx)
                v_traj_mae += traj_mae * len(bx)
                v_land_mae += land_mae * len(bx)
                v_count += len(bx)

        val_loss = v_loss / v_count
        val_traj_mae = v_traj_mae / v_count
        val_land_mae = v_land_mae / v_count
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_traj_mae"].append(train_traj_mae)
        history["val_traj_mae"].append(val_traj_mae)
        history["train_landing_mae"].append(train_land_mae)
        history["val_landing_mae"].append(val_land_mae)
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Traj MAE: {val_traj_mae:.3f}m | Val Land MAE: {val_land_mae:.3f}m | "
                f"LR: {current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_traj_mae": val_traj_mae,
                "val_land_mae": val_land_mae,
                "config": cfg,
                "x_mean": x_mean,
                "x_std": x_std,
                "traj_mean": traj_mean,
                "traj_std": traj_std,
                "land_mean": land_mean,
                "land_std": land_std,
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Training] Early stopping triggered at epoch {epoch}.")
                break

    duration = time.time() - start_time
    print(f"[Training] Finished in {duration:.2f}s. Best Val Loss: {best_val_loss:.4f}")
    print(f"[Training] Saved best checkpoint to {best_model_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Training] Saved history to {history_path}")

    return best_model_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train trajectory surrogate model")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()
    train_model(args.config)
