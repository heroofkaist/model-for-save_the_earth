"""
Save the Earth — Group-Validated Firebrand Distance Benchmarks & Uncertainty Pipeline.

Evaluates prospective and retrospective machine learning models on:
1. 5-Fold Group Cross-Validation across development fuel groups (20 groups, 60 runs)
2. Frozen, untouched held-out test partition (3 groups, 9 runs from split_v1.csv)

Strictly separates:
- Prospective Operational Models: observable BEFORE flight (wind speed, wind speed^2, fuel category)
- Retrospective Diagnostic Models: observable AFTER landing (recovered mass, projected area)
- Mechanistic Physical Baseline: idealized RK4 ballistic aerodynamics

Calculates calibrated conformal prediction intervals and evaluates empirical test coverage.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
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
            "gust_peak_m_s": float(r.get("gust_peak_m_s", r["wind_speed_m_s"])),
            "moisture_content_percent": float(r.get("moisture_content_percent", 10.0)),
            "sample_size": int(r.get("sample_size", 500)),
            "mass_kg": float(r["mass_kg"]),
            "projected_area_m2": float(r["projected_area_m2"]),
            "distance_std_m": float(r.get("distance_std_m", 1.5)),
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


def run_physics_baseline(rows, release_height=2.0, cd=1.0):
    preds = []
    for r in rows:
        w_vec = np.array([r["wind_speed_m_s"], 0.0, 0.0])
        _, _, info = simulate_single_trajectory(
            release_height=release_height,
            wind_vec=w_vec,
            mass=r["mass_kg"],
            area=r["projected_area_m2"],
            dt=0.005,
            cd=cd
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

    dev_recs = [r for r in records if r["split"] in ("train", "validation")]
    test_recs = [r for r in records if r["split"] == "test"]

    print(f"Loaded {len(records)} aggregated experimental runs.")
    print(f"Development Set: {len(dev_recs)} runs (20 fuel groups)")
    print(f"Untouched Test Set: {len(test_recs)} runs (3 fuel groups)")

    y_dev = np.array([r["landing_distance_m"] for r in dev_recs])
    y_test = np.array([r["landing_distance_m"] for r in test_recs])
    groups_dev = np.array([r["experiment_id"] for r in dev_recs])

    # Sample weights proportional to sqrt(sample_size)
    sample_sizes_dev = np.array([r["sample_size"] for r in dev_recs])
    weights_dev = np.sqrt(sample_sizes_dev) / np.mean(np.sqrt(sample_sizes_dev))

    # Preprocessing
    cat_dev = np.array([[r["fuel_category"]] for r in dev_recs])
    cat_test = np.array([[r["fuel_category"]] for r in test_recs])
    enc_pro = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cat_dev)
    c_dev = enc_pro.transform(cat_dev)
    c_test = enc_pro.transform(cat_test)

    num_dev = np.column_stack([[r["wind_speed_m_s"] for r in dev_recs], [r["wind_speed_m_s"]**2 for r in dev_recs]])
    num_test = np.column_stack([[r["wind_speed_m_s"] for r in test_recs], [r["wind_speed_m_s"]**2 for r in test_recs]])
    scaler_pro = StandardScaler().fit(num_dev)
    X_pro_dev = np.column_stack([scaler_pro.transform(num_dev), c_dev])
    X_pro_test = np.column_stack([scaler_pro.transform(num_test), c_test])

    # 1. Five-Fold Group Cross-Validation on Development Data
    print("\n--- Running 5-Fold Group Cross-Validation on Development Groups ---")
    gkf = GroupKFold(n_splits=5)

    cv_models = {
        "Baseline_Global_Median": lambda: None,
        "Prospective_Ridge": lambda: Ridge(alpha=10.0),
        "Prospective_Weighted_Ridge": lambda: Ridge(alpha=10.0),
        "Prospective_Huber_Regressor": lambda: HuberRegressor(alpha=1.0, epsilon=1.35),
        "Prospective_Random_Forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=2, random_state=args.seed),
        "Prospective_Gradient_Boosting": lambda: GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=args.seed),
        "Prospective_HistGradientBoosting": lambda: HistGradientBoostingRegressor(l2_regularization=1.0, max_iter=60, random_state=args.seed),
        "Prospective_Parsimonious_Ensemble": lambda: None  # Blend of Ridge + RF
    }

    cv_results = {m: {"maes": [], "rmses": [], "biases": []} for m in cv_models}
    oof_predictions = {m: np.zeros(len(dev_recs)) for m in cv_models}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(dev_recs, y_dev, groups_dev)):
        y_tr, y_va = y_dev[tr_idx], y_dev[val_idx]
        X_tr, X_va = X_pro_dev[tr_idx], X_pro_dev[val_idx]
        w_tr = weights_dev[tr_idx]

        # Global median
        med_tr = np.median(y_tr)
        p_med = np.full_like(y_va, med_tr)
        oof_predictions["Baseline_Global_Median"][val_idx] = p_med

        # Ridge
        m_ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
        p_ridge = np.maximum(0, m_ridge.predict(X_va))
        oof_predictions["Prospective_Ridge"][val_idx] = p_ridge

        # Weighted Ridge
        m_wridge = Ridge(alpha=10.0).fit(X_tr, y_tr, sample_weight=w_tr)
        p_wridge = np.maximum(0, m_wridge.predict(X_va))
        oof_predictions["Prospective_Weighted_Ridge"][val_idx] = p_wridge

        # Huber Regressor (Robust to heavy-tailed wind tunnel dispersion)
        m_huber = HuberRegressor(alpha=1.0, epsilon=1.35).fit(X_tr, y_tr, sample_weight=w_tr)
        p_huber = np.maximum(0, m_huber.predict(X_va))
        oof_predictions["Prospective_Huber_Regressor"][val_idx] = p_huber

        # Random Forest (depth=2)
        m_rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=args.seed).fit(X_tr, y_tr)
        p_rf = np.maximum(0, m_rf.predict(X_va))
        oof_predictions["Prospective_Random_Forest"][val_idx] = p_rf

        # Gradient Boosting (depth=2)
        m_gb = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=args.seed).fit(X_tr, y_tr)
        p_gb = np.maximum(0, m_gb.predict(X_va))
        oof_predictions["Prospective_Gradient_Boosting"][val_idx] = p_gb

        # HistGradientBoosting
        m_hgb = HistGradientBoostingRegressor(l2_regularization=1.0, max_iter=60, random_state=args.seed).fit(X_tr, y_tr)
        p_hgb = np.maximum(0, m_hgb.predict(X_va))
        oof_predictions["Prospective_HistGradientBoosting"][val_idx] = p_hgb

        # Parsimonious Ensemble (Ridge + RF)
        p_ens = (p_ridge + p_rf) / 2.0
        oof_predictions["Prospective_Parsimonious_Ensemble"][val_idx] = p_ens

    for m in cv_models:
        preds = oof_predictions[m]
        err = preds - y_dev
        cv_results[m] = {
            "cv_mae_m": float(np.mean(np.abs(err))),
            "cv_mae_std_m": float(np.std([np.mean(np.abs(preds[val_idx] - y_dev[val_idx])) for _, val_idx in gkf.split(dev_recs, y_dev, groups_dev)])),
            "cv_rmse_m": float(np.sqrt(np.mean(err**2))),
            "cv_bias_m": float(np.mean(err))
        }

    # Save Group CV metrics table
    cv_csv_path = os.path.join(args.outdir, "group_cv_metrics.csv")
    with open(cv_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cv_mae_m", "cv_mae_std_m", "cv_rmse_m", "cv_bias_m"])
        for m, res in cv_results.items():
            writer.writerow([m, f"{res['cv_mae_m']:.3f}", f"{res['cv_mae_std_m']:.3f}", f"{res['cv_rmse_m']:.3f}", f"{res['cv_bias_m']:.3f}"])
    print(f"Saved Group CV metrics to: {cv_csv_path}")

    # 2. Conformal Prediction Threshold (80% target coverage on OOF residuals)
    oof_err_ridge = np.abs(y_dev - oof_predictions["Prospective_Ridge"])
    alpha = 0.20
    k = int(np.ceil((len(oof_err_ridge) + 1) * (1.0 - alpha)))
    sorted_err = np.sort(oof_err_ridge)
    conformal_radius_80 = float(sorted_err[min(k - 1, len(sorted_err) - 1)])
    print(f"Conformal Calibration: 80% OOF Error Radius = +/- {conformal_radius_80:.3f} m")

    # 3. Final Model Evaluation on Frozen Held-Out Test Set
    print("\n--- Evaluating Models on Frozen Test Set (3 Groups, 9 Runs) ---")
    results = {}
    test_predictions_dict = {
        "record_id": [r["record_id"] for r in test_recs],
        "experiment_id": [r["experiment_id"] for r in test_recs],
        "wind_speed_m_s": [r["wind_speed_m_s"] for r in test_recs],
        "y_true": y_test.tolist()
    }

    # Baseline Global Median
    med_dev = float(np.median(y_dev))
    p_test_med = np.full_like(y_test, med_dev)
    results["Baseline_Global_Median"] = {
        "description": "Constant predictor using development set median.",
        "prospective": True,
        "cv": cv_results["Baseline_Global_Median"],
        "test": compute_metrics(y_test, p_test_med)
    }
    test_predictions_dict["pred_global_median"] = p_test_med.tolist()

    # Prospective Ridge
    model_ridge = Ridge(alpha=10.0).fit(X_pro_dev, y_dev)
    p_test_ridge = np.maximum(0, model_ridge.predict(X_pro_test))
    results["Prospective_Ridge"] = {
        "description": "L2-Regularized Linear Model on wind speed, wind^2, and fuel category.",
        "prospective": True,
        "cv": cv_results["Prospective_Ridge"],
        "test": compute_metrics(y_test, p_test_ridge)
    }
    test_predictions_dict["pred_prospective_ridge"] = p_test_ridge.tolist()

    # Prospective Huber Regressor
    model_huber = HuberRegressor(alpha=1.0, epsilon=1.35).fit(X_pro_dev, y_dev, sample_weight=weights_dev)
    p_test_huber = np.maximum(0, model_huber.predict(X_pro_test))
    results["Prospective_Huber_Regressor"] = {
        "description": "Robust linear regression minimizing Huber loss with sample weighting.",
        "prospective": True,
        "cv": cv_results["Prospective_Huber_Regressor"],
        "test": compute_metrics(y_test, p_test_huber)
    }
    test_predictions_dict["pred_prospective_huber"] = p_test_huber.tolist()

    # Prospective Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=args.seed).fit(X_pro_dev, y_dev)
    p_test_rf = np.maximum(0, model_rf.predict(X_pro_test))
    results["Prospective_Random_Forest"] = {
        "description": "Restrained tree ensemble (max_depth=2) on wind speed and fuel category.",
        "prospective": True,
        "cv": cv_results["Prospective_Random_Forest"],
        "test": compute_metrics(y_test, p_test_rf)
    }
    test_predictions_dict["pred_prospective_rf"] = p_test_rf.tolist()

    # Prospective Gradient Boosting
    model_gb = GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=args.seed).fit(X_pro_dev, y_dev)
    p_test_gb = np.maximum(0, model_gb.predict(X_pro_test))
    results["Prospective_Gradient_Boosting"] = {
        "description": "Gradient boosted shallow trees (max_depth=2, lr=0.05).",
        "prospective": True,
        "cv": cv_results["Prospective_Gradient_Boosting"],
        "test": compute_metrics(y_test, p_test_gb)
    }
    test_predictions_dict["pred_prospective_gb"] = p_test_gb.tolist()

    # Prospective Parsimonious Ensemble (Blend of Ridge + RF)
    p_test_ens = (p_test_ridge + p_test_rf) / 2.0
    results["Prospective_Parsimonious_Ensemble"] = {
        "description": "Regularized blend averaging Ridge and Random Forest (depth=2).",
        "prospective": True,
        "cv": cv_results["Prospective_Parsimonious_Ensemble"],
        "test": compute_metrics(y_test, p_test_ens)
    }
    test_predictions_dict["pred_prospective_ensemble"] = p_test_ens.tolist()

    # Conformal Prediction Bounds for Test Set
    p_test_lower = np.maximum(0, p_test_ridge - conformal_radius_80)
    p_test_upper = p_test_ridge + conformal_radius_80
    test_predictions_dict["pred_conformal_p10_bound"] = p_test_lower.tolist()
    test_predictions_dict["pred_conformal_p90_bound"] = p_test_upper.tolist()

    covered = (y_test >= p_test_lower) & (y_test <= p_test_upper)
    test_coverage_pct = float(np.mean(covered) * 100.0)
    results["Conformal_Uncertainty_80"] = {
        "nominal_coverage_pct": 80.0,
        "empirical_test_coverage_pct": test_coverage_pct,
        "conformal_radius_m": conformal_radius_80,
        "coverage_comment": "Evaluates group-level exchangeability; test fuel mockups exhibit slight outward shift."
    }

    # 4. Retrospective Diagnostic Models (Strictly Separated)
    def get_retrospective_matrix(recs, enc=None, scaler=None, is_train=False):
        mass = np.array([r["mass_kg"] for r in recs])
        area = np.array([r["projected_area_m2"] for r in recs])
        wind = np.array([r["wind_speed_m_s"] for r in recs])
        num_f = np.column_stack([np.log(mass), np.log(area), np.log(area / mass), wind, wind**2])
        cat_raw = np.array([[r["fuel_category"]] for r in recs])
        if is_train:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_f = enc.fit_transform(cat_raw)
            scaler = StandardScaler()
            num_f = scaler.fit_transform(num_f)
        else:
            cat_f = enc.transform(cat_raw)
            num_f = scaler.transform(num_f)
        return np.column_stack([num_f, cat_f]), enc, scaler

    X_retro_dev, enc_retro, scale_retro = get_retrospective_matrix(dev_recs, is_train=True)
    X_retro_test, _, _ = get_retrospective_matrix(test_recs, enc=enc_retro, scaler=scale_retro, is_train=False)

    model_retro_gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=args.seed)
    model_retro_gb.fit(X_retro_dev, y_dev)
    p_test_retro_gb = np.maximum(0, model_retro_gb.predict(X_retro_test))
    results["Retrospective_Gradient_Boosting"] = {
        "description": "Diagnostic model using recovered post-landing particle mass, area, and wind.",
        "prospective": False,
        "test": compute_metrics(y_test, p_test_retro_gb)
    }
    test_predictions_dict["pred_retrospective_gb"] = p_test_retro_gb.tolist()

    # 5. Physical Mechanistic Baseline (Evaluated at h=2.0m and h=1.0m)
    p_test_phys_2m = run_physics_baseline(test_recs, release_height=2.0, cd=1.0)
    p_test_phys_1m = run_physics_baseline(test_recs, release_height=1.0, cd=1.0)
    results["Physical_RK4_Ballistic_2m"] = {
        "description": "Illustrative ballistic baseline under quadratic drag from h=2.0 m.",
        "prospective": False,
        "test": compute_metrics(y_test, p_test_phys_2m)
    }
    results["Physical_RK4_Ballistic_1m"] = {
        "description": "Illustrative ballistic baseline under quadratic drag from h=1.0 m.",
        "prospective": False,
        "test": compute_metrics(y_test, p_test_phys_1m)
    }
    test_predictions_dict["pred_physical_rk4_2m"] = p_test_phys_2m.tolist()
    test_predictions_dict["pred_physical_rk4_1m"] = p_test_phys_1m.tolist()

    # Save benchmark JSON
    json_path = os.path.join(args.outdir, "benchmark_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark summary to: {json_path}")

    # Save test predictions CSV
    preds_csv_path = os.path.join(args.outdir, "test_predictions.csv")
    with open(preds_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_predictions_dict.keys()))
        writer.writeheader()
        keys = list(test_predictions_dict.keys())
        n_rows = len(test_predictions_dict["record_id"])
        for i in range(n_rows):
            writer.writerow({k: test_predictions_dict[k][i] for k in keys})
    print(f"Saved test predictions to: {preds_csv_path}")

    # Save metrics table CSV
    summary_rows = []
    for model_name, data in results.items():
        if "test" in data:
            cv_mae_str = f"{data['cv']['cv_mae_m']:.3f}" if "cv" in data else "N/A"
            summary_rows.append({
                "model": model_name,
                "type": "Prospective" if data.get("prospective") else "Diagnostic/Physical",
                "cv_mae_m": cv_mae_str,
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
    print(f"{'Model':<35} | {'Type':<14} | {'CV MAE':<9} | {'Test MAE':<9} | {'Test RMSE':<9} | {'Bias':<8}")
    print("-" * 92)
    for r in summary_rows:
        cv_disp = (r['cv_mae_m'] + ' m') if r['cv_mae_m'] != 'N/A' else 'N/A'
        print(f"{r['model']:<35} | {r['type']:<14} | {cv_disp:<9} | {r['test_mae_m'] + ' m':<9} | {r['test_rmse_m'] + ' m':<9} | {r['test_bias_m'] + ' m':<8}")
    print("=" * 92)


if __name__ == "__main__":
    main()
