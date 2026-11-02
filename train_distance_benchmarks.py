"""
Group-Validated Firebrand Distance Benchmarks.

Evaluates multiple model classes on the frozen, group-disjoint partition (data/manifests/split_v1.csv):
1. Baselines:
   - Global Training Median
   - Fuel-Family Training Median
2. Prospective Release Models (observable BEFORE flight: wind speed, fuel category, fuel type):
   - Prospective Ridge Regression
   - Prospective Random Forest Regressor
   - Prospective Gradient Boosting Regressor
3. Retrospective Particle Models (uses recovered post-landing mass and projected area):
   - Retrospective Ridge Regression
   - Retrospective Random Forest Regressor
   - Retrospective Gradient Boosting Regressor
4. Physical Mechanistic Baseline (ballistic RK4 trajectory simulation under wind).

Strictly isolates test partition until final evaluation. All preprocessors fitted on train only.
"""

import argparse
import csv
import json
import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ember_simulation import simulate_single_trajectory

DATA_PATH = "data/interim/eligible.csv"
SPLIT_PATH = "data/manifests/split_v1.csv"
OUT_DIR = "distance_results"


def load_dataset(data_path, split_path):
    with open(data_path, newline="", encoding="utf-8") as f:
        data_rows = list(csv.DictReader(f))
    with open(split_path, newline="", encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))

    split_map = {r["record_id"]: r["split"] for r in split_rows}

    records = []
    for r in data_rows:
        rid = r["record_id"]
        if rid not in split_map:
            continue
        part = split_map[rid]
        records.append({
            "record_id": rid,
            "split": part,
            "experiment_id": r["experiment_id"],
            "source_id": r["source_id"],
            "fuel_type": r["fuel_type"],
            "fuel_category": "vegetative" if "veg" in r["source_id"] else "structural",
            "wind_speed_m_s": float(r["wind_speed_m_s"]),
            "mass_kg": float(r["mass_kg"]),
            "projected_area_m2": float(r["projected_area_m2"]),
            "landing_distance_m": float(r["landing_distance_m"])
        })
    return records


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "n": int(len(y_true)),
        "mae_m": float(np.mean(abs_err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "median_ae_m": float(np.median(abs_err)),
        "bias_m": float(np.mean(err))
    }


def run_physics_baseline(rows, release_height=2.0):
    """
    Simulates each ember with measured mass, area, and wind under standard RK4 aerodynamics.
    """
    preds = []
    for r in rows:
        w_vec = np.array([r["wind_speed_m_s"], 0.0, 0.0])
        _, _, info = simulate_single_trajectory(
            release_height=release_height,
            wind_vec=w_vec,
            mass=r["mass_kg"],
            area=r["projected_area_m2"],
            dt=0.005,
            cd=1.0
        )
        preds.append(info["landing_x"])
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--split", default=SPLIT_PATH)
    parser.add_argument("--outdir", default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    records = load_dataset(args.data, args.split)

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "validation"]
    test_recs = [r for r in records if r["split"] == "test"]

    print(f"Loaded {len(records)} records: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    y_train = np.array([r["landing_distance_m"] for r in train_recs])
    y_val = np.array([r["landing_distance_m"] for r in val_recs])
    y_test = np.array([r["landing_distance_m"] for r in test_recs])

    results = {}
    test_predictions_dict = {"record_id": [r["record_id"] for r in test_recs], "y_true": y_test.tolist()}

    # --- 1. Global Training Median Baseline ---
    med_global = float(np.median(y_train))
    results["Baseline_Global_Median"] = {
        "description": "Constant predictor using training set median distance.",
        "prospective": True,
        "val": compute_metrics(y_val, np.full_like(y_val, med_global)),
        "test": compute_metrics(y_test, np.full_like(y_test, med_global))
    }
    test_predictions_dict["pred_global_median"] = np.full_like(y_test, med_global).tolist()

    # --- 2. Fuel-Category Training Median Baseline ---
    cat_medians = {}
    for cat in ("vegetative", "structural"):
        sub = [r["landing_distance_m"] for r in train_recs if r["fuel_category"] == cat]
        cat_medians[cat] = float(np.median(sub)) if sub else med_global

    val_cat_pred = np.array([cat_medians.get(r["fuel_category"], med_global) for r in val_recs])
    test_cat_pred = np.array([cat_medians.get(r["fuel_category"], med_global) for r in test_recs])
    results["Baseline_Category_Median"] = {
        "description": "Category-specific training median distance.",
        "prospective": True,
        "val": compute_metrics(y_val, val_cat_pred),
        "test": compute_metrics(y_test, test_cat_pred)
    }
    test_predictions_dict["pred_category_median"] = test_cat_pred.tolist()

    # --- Feature Extraction ---
    # Prospective features: wind_speed, wind_speed^2, fuel_category
    def get_prospective_matrix(recs, enc=None, scaler=None, is_train=False):
        num_feat = np.column_stack([
            [r["wind_speed_m_s"] for r in recs],
            [r["wind_speed_m_s"]**2 for r in recs]
        ])
        cat_raw = np.array([[r["fuel_category"]] for r in recs])
        if is_train:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_feat = enc.fit_transform(cat_raw)
            scaler = StandardScaler()
            num_feat = scaler.fit_transform(num_feat)
        else:
            cat_feat = enc.transform(cat_raw)
            num_feat = scaler.transform(num_feat)
        X = np.column_stack([num_feat, cat_feat])
        return X, enc, scaler

    X_pro_train, enc_pro, scale_pro = get_prospective_matrix(train_recs, is_train=True)
    X_pro_val, _, _ = get_prospective_matrix(val_recs, enc=enc_pro, scaler=scale_pro, is_train=False)
    X_pro_test, _, _ = get_prospective_matrix(test_recs, enc=enc_pro, scaler=scale_pro, is_train=False)

    # Retrospective features: log(mass), log(area), log(area/mass), wind, wind^2, category
    def get_retrospective_matrix(recs, enc=None, scaler=None, is_train=False):
        mass = np.array([r["mass_kg"] for r in recs])
        area = np.array([r["projected_area_m2"] for r in recs])
        wind = np.array([r["wind_speed_m_s"] for r in recs])
        num_feat = np.column_stack([
            np.log(mass),
            np.log(area),
            np.log(area / mass),
            wind,
            wind**2
        ])
        cat_raw = np.array([[r["fuel_category"]] for r in recs])
        if is_train:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_feat = enc.fit_transform(cat_raw)
            scaler = StandardScaler()
            num_feat = scaler.fit_transform(num_feat)
        else:
            cat_feat = enc.transform(cat_raw)
            num_feat = scaler.transform(num_feat)
        X = np.column_stack([num_feat, cat_feat])
        return X, enc, scaler

    X_retro_train, enc_retro, scale_retro = get_retrospective_matrix(train_recs, is_train=True)
    X_retro_val, _, _ = get_retrospective_matrix(val_recs, enc=enc_retro, scaler=scale_retro, is_train=False)
    X_retro_test, _, _ = get_retrospective_matrix(test_recs, enc=enc_retro, scaler=scale_retro, is_train=False)

    # --- 3. Prospective Ridge Regression ---
    model_pro_ridge = Ridge(alpha=1.0)
    model_pro_ridge.fit(X_pro_train, y_train)
    p_val_pro_ridge = np.maximum(0, model_pro_ridge.predict(X_pro_val))
    p_test_pro_ridge = np.maximum(0, model_pro_ridge.predict(X_pro_test))
    results["Prospective_Ridge"] = {
        "description": "Linear Ridge Regression using only wind and fuel category.",
        "prospective": True,
        "val": compute_metrics(y_val, p_val_pro_ridge),
        "test": compute_metrics(y_test, p_test_pro_ridge)
    }
    test_predictions_dict["pred_prospective_ridge"] = p_test_pro_ridge.tolist()

    # --- 4. Prospective Random Forest ---
    model_pro_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=args.seed)
    model_pro_rf.fit(X_pro_train, y_train)
    p_val_pro_rf = np.maximum(0, model_pro_rf.predict(X_pro_val))
    p_test_pro_rf = np.maximum(0, model_pro_rf.predict(X_pro_test))
    results["Prospective_Random_Forest"] = {
        "description": "Random Forest using only wind and fuel category.",
        "prospective": True,
        "val": compute_metrics(y_val, p_val_pro_rf),
        "test": compute_metrics(y_test, p_test_pro_rf)
    }
    test_predictions_dict["pred_prospective_rf"] = p_test_pro_rf.tolist()

    # --- 5. Retrospective Ridge Regression ---
    model_retro_ridge = Ridge(alpha=1.0)
    model_retro_ridge.fit(X_retro_train, y_train)
    p_val_retro_ridge = np.maximum(0, model_retro_ridge.predict(X_retro_val))
    p_test_retro_ridge = np.maximum(0, model_retro_ridge.predict(X_retro_test))
    results["Retrospective_Ridge"] = {
        "description": "Ridge Regression using post-landing mass, area, area/mass ratio, wind.",
        "prospective": False,
        "val": compute_metrics(y_val, p_val_retro_ridge),
        "test": compute_metrics(y_test, p_test_retro_ridge)
    }
    test_predictions_dict["pred_retrospective_ridge"] = p_test_retro_ridge.tolist()

    # --- 6. Retrospective Gradient Boosting ---
    model_retro_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=args.seed)
    model_retro_gb.fit(X_retro_train, y_train)
    p_val_retro_gb = np.maximum(0, model_retro_gb.predict(X_retro_val))
    p_test_retro_gb = np.maximum(0, model_retro_gb.predict(X_retro_test))
    results["Retrospective_Gradient_Boosting"] = {
        "description": "Gradient Boosting on post-landing particle mass, area, wind.",
        "prospective": False,
        "val": compute_metrics(y_val, p_val_retro_gb),
        "test": compute_metrics(y_test, p_test_retro_gb)
    }
    test_predictions_dict["pred_retrospective_gb"] = p_test_retro_gb.tolist()

    # --- 7. Physical Mechanistic Baseline ---
    p_val_phys = run_physics_baseline(val_recs, release_height=2.0)
    p_test_phys = run_physics_baseline(test_recs, release_height=2.0)
    results["Physical_RK4_Ballistic"] = {
        "description": "Mechanistic ballistic particle flight under quadratic drag at release height = 2.0 m.",
        "prospective": False,
        "val": compute_metrics(y_val, p_val_phys),
        "test": compute_metrics(y_test, p_test_phys)
    }
    test_predictions_dict["pred_physical_rk4"] = p_test_phys.tolist()

    # --- Save Reports and Predictions ---
    eval_json_path = os.path.join(args.outdir, "benchmark_comparison.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark summary to: {eval_json_path}")

    preds_csv_path = os.path.join(args.outdir, "test_predictions.csv")
    with open(preds_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_predictions_dict.keys()))
        writer.writeheader()
        keys = list(test_predictions_dict.keys())
        n_rows = len(test_predictions_dict["record_id"])
        for i in range(n_rows):
            row = {k: test_predictions_dict[k][i] for k in keys}
            writer.writerow(row)
    print(f"Saved test predictions to: {preds_csv_path}")

    # Generate summary table CSV
    summary_rows = []
    for model_name, data in results.items():
        summary_rows.append({
            "model": model_name,
            "type": "Prospective" if data["prospective"] else "Retrospective/Physical",
            "val_mae_m": f"{data['val']['mae_m']:.3f}",
            "val_rmse_m": f"{data['val']['rmse_m']:.3f}",
            "test_mae_m": f"{data['test']['mae_m']:.3f}",
            "test_rmse_m": f"{data['test']['rmse_m']:.3f}",
            "test_med_ae_m": f"{data['test']['median_ae_m']:.3f}",
            "test_bias_m": f"{data['test']['bias_m']:.3f}"
        })
    metrics_csv_path = os.path.join(args.outdir, "metrics_table.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved metrics table to: {metrics_csv_path}")

    # Print summary
    print("\n================== BENCHMARK EVALUATION RESULTS (HELD-OUT TEST SET) ==================")
    print(f"{'Model':<35} | {'Type':<15} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Bias':<8}")
    print("-" * 88)
    for r in summary_rows:
        print(f"{r['model']:<35} | {r['type']:<15} | {r['test_mae_m'] + ' m':<10} | {r['test_rmse_m'] + ' m':<10} | {r['test_bias_m'] + ' m':<8}")
    print("=" * 88)


if __name__ == "__main__":
    main()

