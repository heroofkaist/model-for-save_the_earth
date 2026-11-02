"""
Save the Earth — Scientific Diagnostic Visualizations & Reporting.

Generates the complete suite of diagnostic figures and summary tables:
- mass_histogram.png, area_histogram.png, distance_histogram.png
- area_vs_mass.png, distance_vs_wind.png, fuel_breakdown.png, missingness_matrix.png
- prediction_vs_observed.png (with conformal uncertainty bounds)
- residual_vs_wind.png, residual_by_fuel.png
- cv_model_comparison.png (Group CV performance & stability across development folds)
"""

import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/interim/eligible.csv"
SPLIT_PATH = "data/manifests/split_v1.csv"
RUNS_PATH = "data/processed/jfsp_experiments_summary.csv"
PREDS_PATH = "distance_results/test_predictions.csv"
CV_PATH = "distance_results/group_cv_metrics.csv"
FIG_DIR = "reports/figures"
REP_DIR = "reports"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

# 1. Load Data
with open(DATA_PATH, newline="", encoding="utf-8") as f:
    eligible_rows = list(csv.DictReader(f))

with open(SPLIT_PATH, newline="", encoding="utf-8") as f:
    split_rows = list(csv.DictReader(f))

split_map = {r["record_id"]: r["split"] for r in split_rows}
for r in eligible_rows:
    r["split"] = split_map.get(r["record_id"], "unassigned")

with open(PREDS_PATH, newline="", encoding="utf-8") as f:
    preds_rows = list(csv.DictReader(f))

masses = np.array([float(r["mass_kg"]) for r in eligible_rows])
areas = np.array([float(r["projected_area_m2"]) for r in eligible_rows])
distances = np.array([float(r["landing_distance_m"]) for r in eligible_rows])
winds = np.array([float(r["wind_speed_m_s"]) for r in eligible_rows])
sources = [r["source_id"] for r in eligible_rows]
fuels = [r["fuel_type"] for r in eligible_rows]
splits = [r["split"] for r in eligible_rows]

# Set matplotlib style
plt.rcParams.update({"font.sans-serif": "Arial", "axes.grid": True, "grid.alpha": 0.4})

# Figure 1: mass_histogram.png
plt.figure(figsize=(7, 5))
plt.hist(masses * 1e3, bins=15, color="#2b5c8f", edgecolor="black", alpha=0.8)
plt.xlabel("Particle Mass (g)", fontsize=11)
plt.ylabel("Experimental Run Count", fontsize=11)
plt.title("Measured Firebrand Mass Distribution (All 69 Runs)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mass_histogram.png", dpi=200)
plt.close()

# Figure 2: area_histogram.png
plt.figure(figsize=(7, 5))
plt.hist(areas * 1e4, bins=15, color="#e67e22", edgecolor="black", alpha=0.8)
plt.xlabel("Projected Area (cm²)", fontsize=11)
plt.ylabel("Experimental Run Count", fontsize=11)
plt.title("Measured Projected Area Distribution (All 69 Runs)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/area_histogram.png", dpi=200)
plt.close()

# Figure 3: distance_histogram.png
plt.figure(figsize=(7, 5))
plt.hist(distances, bins=15, color="#27ae60", edgecolor="black", alpha=0.8)
plt.xlabel("Median Landing Distance (m)", fontsize=11)
plt.ylabel("Experimental Run Count", fontsize=11)
plt.title("Reported Experimental Run Median Distances (All 69 Runs)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/distance_histogram.png", dpi=200)
plt.close()

# Figure 4: area_vs_mass.png
plt.figure(figsize=(7, 5))
is_veg = np.array(["veg" in s for s in sources])
plt.scatter((masses[is_veg] * 1e3), (areas[is_veg] * 1e4), color="#27ae60", s=60, alpha=0.8, label="Vegetative (15 runs)")
plt.scatter((masses[~is_veg] * 1e3), (areas[~is_veg] * 1e4), color="#d35400", s=60, alpha=0.8, label="Structural (54 runs)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mass (g, log scale)", fontsize=11)
plt.ylabel("Projected Area (cm², log scale)", fontsize=11)
plt.title("Projected Area vs. Mass by Fuel Category", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/area_vs_mass.png", dpi=200)
plt.close()

# Figure 5: distance_vs_wind.png
plt.figure(figsize=(7, 5))
plt.scatter(winds[is_veg] - 0.15, distances[is_veg], color="#27ae60", s=70, alpha=0.8, label="Vegetative Fuels")
plt.scatter(winds[~is_veg] + 0.15, distances[~is_veg], color="#2980b9", s=70, alpha=0.8, label="Structural Assemblies")
plt.xlabel("Wind Speed (m/s)", fontsize=11)
plt.ylabel("Median Landing Distance (m)", fontsize=11)
plt.title("Experimental Landing Distance vs. Wind Speed", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/distance_vs_wind.png", dpi=200)
plt.close()

# Figure 6: fuel_breakdown.png
fuel_counts = {}
for f_name in fuels:
    fuel_counts[f_name] = fuel_counts.get(f_name, 0) + 1

plt.figure(figsize=(10, 6))
sorted_fuels = sorted(fuel_counts.items(), key=lambda x: x[1], reverse=True)
names = [x[0].replace("_", " ") for x in sorted_fuels]
counts = [x[1] for x in sorted_fuels]
plt.barh(names, counts, color="#34495e", edgecolor="black", alpha=0.8)
plt.xlabel("Number of Experimental Wind Runs (3 wind levels each)", fontsize=11)
plt.title("Run Counts by Fuel Category (23 Unique Groups)", fontsize=12, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fuel_breakdown.png", dpi=200)
plt.close()

# Figure 7: missingness_matrix.png
tasks = [
    "particle_distance",
    "particle_trajectory",
    "thermal_survival",
    "firebrand_flux",
    "wildfire_spotting"
]
features = ["wind_speed", "fuel_type", "mass_area", "3d_coords", "flight_time", "heat_flux", "ignition", "spot_coords"]
availability = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1]
])

plt.figure(figsize=(9, 5))
plt.imshow(availability, cmap="Blues", aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(features)), features, rotation=35, ha="right", fontsize=10)
plt.yticks(range(len(tasks)), tasks, fontsize=10)
for i in range(len(tasks)):
    for j in range(len(features)):
        text = "YES" if availability[i, j] == 1 else "NO"
        color = "white" if availability[i, j] == 1 else "#7f8c8d"
        plt.text(j, i, text, ha="center", va="center", color=color, fontweight="bold")
plt.title("Evidence Missingness Matrix Across Research Task Families", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/missingness_matrix.png", dpi=200)
plt.close()

# Figure 8: prediction_vs_observed.png (With Conformal Uncertainty Bounds)
y_true = np.array([float(r["y_true"]) for r in preds_rows])
p_median = np.array([float(r["pred_global_median"]) for r in preds_rows])
p_pro_rf = np.array([float(r["pred_prospective_rf"]) for r in preds_rows])
p_pro_ridge = np.array([float(r["pred_prospective_ridge"]) for r in preds_rows])
p_lower = np.array([float(r["pred_conformal_p10_bound"]) for r in preds_rows])
p_upper = np.array([float(r["pred_conformal_p90_bound"]) for r in preds_rows])
p_retro_gb = np.array([float(r["pred_retrospective_gb"]) for r in preds_rows])

plt.figure(figsize=(7, 6))
lim = max(np.max(y_true), np.max(p_upper), np.max(p_retro_gb)) * 1.15
plt.plot([0, lim], [0, lim], "k--", label="Ideal 1:1 Calibration")
plt.scatter(y_true, p_median, color="#95a5a6", s=60, label="Global Median Baseline", alpha=0.7)
plt.errorbar(y_true, p_pro_ridge, yerr=[p_pro_ridge - p_lower, p_upper - p_pro_ridge],
             fmt="^", color="#2980b9", ecolor="#7fb3d5", elinewidth=1.5, capsize=3,
             label="Prospective Ridge (80% Conformal Interval)")
plt.scatter(y_true, p_retro_gb, color="#c0392b", s=70, marker="o", label="Retrospective GB (Diagnostic)")
plt.xlabel("Observed Run Median Distance (m)", fontsize=11)
plt.ylabel("Predicted Distance (m)", fontsize=11)
plt.title("Prediction vs. Observed with 80% Conformal Interval (Test Set)", fontsize=11, fontweight="bold")
plt.legend(fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/prediction_vs_observed.png", dpi=200)
plt.close()

# Figure 9: residual_vs_wind.png (Test Set)
test_records = [r for r in eligible_rows if r["split"] == "test"]
test_winds = np.array([float(r["wind_speed_m_s"]) for r in test_records])
residuals_pro = p_pro_ridge - y_true
residuals_retro = p_retro_gb - y_true

plt.figure(figsize=(7, 5))
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.scatter(test_winds - 0.2, residuals_pro, color="#2980b9", s=70, marker="^", label="Prospective Ridge Residuals")
plt.scatter(test_winds + 0.2, residuals_retro, color="#c0392b", s=70, marker="o", label="Retrospective GB Residuals")
plt.xlabel("Wind Speed (m/s)", fontsize=11)
plt.ylabel("Residual (Predicted - Observed, m)", fontsize=11)
plt.title("Prediction Residuals vs. Wind Speed (Test Set)", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/residual_vs_wind.png", dpi=200)
plt.close()

# Figure 10: residual_by_fuel.png
test_fuels = [r["fuel_type"] for r in test_records]
plt.figure(figsize=(8, 5))
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
x_pos = np.arange(len(test_fuels))
plt.bar(x_pos - 0.2, residuals_pro, width=0.4, color="#2980b9", label="Prospective Ridge")
plt.bar(x_pos + 0.2, residuals_retro, width=0.4, color="#c0392b", label="Retrospective GB")
plt.xticks(x_pos, [f.replace("_", "\n") for f in test_fuels], fontsize=9)
plt.ylabel("Residual (m)", fontsize=11)
plt.title("Prediction Residuals by Held-Out Test Fuel Group", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/residual_by_fuel.png", dpi=200)
plt.close()

# Figure 11: cv_model_comparison.png (Group Cross-Validation)
if os.path.exists(CV_PATH):
    with open(CV_PATH, newline="", encoding="utf-8") as f:
        cv_rows = list(csv.DictReader(f))

    cv_names = [r["model"].replace("Prospective_", "").replace("_", " ") for r in cv_rows]
    cv_maes = [float(r["cv_mae_m"]) for r in cv_rows]
    cv_stds = [float(r["cv_mae_std_m"]) for r in cv_rows]

    plt.figure(figsize=(10, 5))
    x_idx = np.arange(len(cv_names))
    plt.bar(x_idx, cv_maes, yerr=cv_stds, color="#2c3e50", ecolor="#e74c3c", capsize=4, alpha=0.85)
    plt.xticks(x_idx, cv_names, rotation=25, ha="right", fontsize=9)
    plt.ylabel("5-Fold Group CV MAE (m)", fontsize=11)
    plt.title("Candidate Model Cross-Validation Error & Stability (20 Dev Groups)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cv_model_comparison.png", dpi=200)
    plt.close()

# Generate Summary CSV Tables
with open(f"{REP_DIR}/dataset_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "task_family", "total_runs", "independent_groups", "wind_levels", "total_firebrands_reported", "provenance_source"])
    writer.writerow(["usda_vegetative_2020_0035", "particle_distance", 15, 5, "Idle (5.36m/s), Med (11.17m/s), High (17.88m/s)", 9249, "JFSP 15-1-04-4 Report Tables 22-26"])
    writer.writerow(["usda_structural_2020_0034", "particle_distance", 54, 18, "Idle (5.36m/s), Med (11.17m/s), High (17.88m/s)", 50571, "JFSP 15-1-04-4 Report Tables 4-21"])
    writer.writerow(["zenodo_4837617", "firebrand_flux", 1, 1, "Field ambient 4.2m/s", 142, "Zenodo 4837617 Audio Classifier"])

with open(f"{REP_DIR}/qc_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "count", "pct", "action"])
    writer.writerow(["PASS", 69, "100%", "Eligible for group-validated distance training"])
    writer.writerow(["FLAG", 0, "0%", "None"])
    writer.writerow(["EXCLUDE", 0, "0%", "Quarantined in data/interim/quarantine.csv"])

split_counts = {"train": 0, "validation": 0, "test": 0}
for s in splits:
    split_counts[s] = split_counts.get(s, 0) + 1

with open(f"{REP_DIR}/split_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["partition", "runs_count", "pct_of_data", "independent_groups_count", "isolation_status"])
    writer.writerow(["train", split_counts["train"], f"{split_counts['train']/len(splits)*100:.1f}%", 17, "Fitted preprocessors and model weights"])
    writer.writerow(["validation", split_counts["validation"], f"{split_counts['validation']/len(splits)*100:.1f}%", 3, "Hyperparameter selection only"])
    writer.writerow(["test", split_counts["test"], f"{split_counts['test']/len(splits)*100:.1f}%", 3, "Frozen untouched evaluation"])

print(f"Generated diagnostic figures in {FIG_DIR} and summary CSVs in {REP_DIR} successfully!")
