"""
Generates the complete suite of diagnostic figures and summary tables
required by the Save the Earth scientific evaluation protocol (PDF Page 18).
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
plt.title("Measured Firebrand Mass Distribution (All Runs)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/mass_histogram.png", dpi=200)
plt.close()

# Figure 2: area_histogram.png
plt.figure(figsize=(7, 5))
plt.hist(areas * 1e4, bins=15, color="#e67e22", edgecolor="black", alpha=0.8)
plt.xlabel("Projected Area (cm²)", fontsize=11)
plt.ylabel("Experimental Run Count", fontsize=11)
plt.title("Measured Projected Area Distribution (All Runs)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/area_histogram.png", dpi=200)
plt.close()

# Figure 3: distance_histogram.png
plt.figure(figsize=(7, 5))
plt.hist(distances, bins=15, color="#27ae60", edgecolor="black", alpha=0.8)
plt.xlabel("Observed Landing Distance (m)", fontsize=11)
plt.ylabel("Experimental Run Count", fontsize=11)
plt.title("Observed Firebrand Travel Distance Distribution", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/distance_histogram.png", dpi=200)
plt.close()

# Figure 4: distance_vs_mass.png
plt.figure(figsize=(7, 5))
veg_mask = np.array(["veg" in s for s in sources])
plt.scatter(masses[veg_mask] * 1e3, distances[veg_mask], color="#27ae60", s=60, edgecolors="black", label="Vegetative Fuels")
plt.scatter(masses[~veg_mask] * 1e3, distances[~veg_mask], color="#9b59b6", s=60, edgecolors="black", marker="s", label="Structural Assemblies")
plt.xscale("log")
plt.xlabel("Particle Mass (g) [Log Scale]", fontsize=11)
plt.ylabel("Landing Distance (m)", fontsize=11)
plt.title("Observed Travel Distance vs. Mass", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/distance_vs_mass.png", dpi=200)
plt.close()

# Figure 5: distance_vs_area.png
plt.figure(figsize=(7, 5))
plt.scatter(areas[veg_mask] * 1e4, distances[veg_mask], color="#27ae60", s=60, edgecolors="black", label="Vegetative Fuels")
plt.scatter(areas[~veg_mask] * 1e4, distances[~veg_mask], color="#9b59b6", s=60, edgecolors="black", marker="s", label="Structural Assemblies")
plt.xscale("log")
plt.xlabel("Projected Area (cm²) [Log Scale]", fontsize=11)
plt.ylabel("Landing Distance (m)", fontsize=11)
plt.title("Observed Travel Distance vs. Projected Area", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/distance_vs_area.png", dpi=200)
plt.close()

# Figure 6: wind_distribution.png
plt.figure(figsize=(7, 5))
unique_winds, wind_counts = np.unique(winds, return_counts=True)
plt.bar([f"{w:.2f} m/s" for w in unique_winds], wind_counts, color="#34495e", edgecolor="black", width=0.5)
plt.xlabel("Nominal Wind Speed Level", fontsize=11)
plt.ylabel("Number of Experimental Runs", fontsize=11)
plt.title("Wind Condition Coverage across Experiments", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/wind_distribution.png", dpi=200)
plt.close()

# Figure 7: missingness_matrix.png
# Shows feature availability across tasks (why tasks cannot be naively concatenated)
tasks = ["Particle Distance (USDA)", "Trajectory 3D (CSIRO)", "Thermal Pile (Zenodo)", "Acoustic Flux (Zenodo)", "Wildfire Spotting (MDPI)"]
features = ["Mass (g)", "Area (cm²)", "Wind (m/s)", "Flight Time (s)", "3D Trajectory", "Heat Flux", "Spotting Dist", "Ignition State"]
# 1 = available, 0 = unavailable
availability = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],  # USDA distance
    [1, 1, 1, 1, 1, 0, 0, 0],  # CSIRO 3D
    [1, 0, 1, 0, 0, 1, 0, 0],  # Zenodo thermal
    [0, 0, 1, 0, 0, 0, 0, 0],  # Zenodo acoustic
    [0, 0, 1, 0, 0, 0, 1, 1]   # MDPI spotting
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

# Figure 8: prediction_vs_observed.png (Test Set Evaluation)
y_true = np.array([float(r["y_true"]) for r in preds_rows])
p_median = np.array([float(r["pred_global_median"]) for r in preds_rows])
p_pro_rf = np.array([float(r["pred_prospective_rf"]) for r in preds_rows])
p_retro_gb = np.array([float(r["pred_retrospective_gb"]) for r in preds_rows])

plt.figure(figsize=(7, 6))
lim = max(np.max(y_true), np.max(p_retro_gb)) * 1.2
plt.plot([0, lim], [0, lim], "k--", label="Ideal 1:1 Calibration")
plt.scatter(y_true, p_median, color="#95a5a6", s=70, label="Global Median Baseline", alpha=0.7)
plt.scatter(y_true, p_pro_rf, color="#2980b9", s=80, marker="^", label="Prospective RF (Wind/Fuel only)")
plt.scatter(y_true, p_retro_gb, color="#c0392b", s=80, marker="o", label="Retrospective GB (Post-flight)")
plt.xlabel("Observed Measured Distance (m)", fontsize=11)
plt.ylabel("Predicted Distance (m)", fontsize=11)
plt.title("Prediction vs. Observed (Held-Out Test Set)", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/prediction_vs_observed.png", dpi=200)
plt.close()

# Figure 9: residual_vs_wind.png (Test Set)
test_records = [r for r in eligible_rows if r["split"] == "test"]
test_winds = np.array([float(r["wind_speed_m_s"]) for r in test_records])
residuals_pro = p_pro_rf - y_true
residuals_retro = p_retro_gb - y_true

plt.figure(figsize=(7, 5))
plt.axhline(0, color="black", linestyle="--", alpha=0.7)
plt.scatter(test_winds - 0.2, residuals_pro, color="#2980b9", s=70, marker="^", label="Prospective RF Residuals")
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
plt.bar(x_pos - 0.2, residuals_pro, width=0.4, color="#2980b9", label="Prospective RF")
plt.bar(x_pos + 0.2, residuals_retro, width=0.4, color="#c0392b", label="Retrospective GB")
plt.xticks(x_pos, [f.replace("_", "\n") for f in test_fuels], fontsize=9)
plt.ylabel("Residual (m)", fontsize=11)
plt.title("Prediction Residuals by Unseen Fuel Category (Test Set)", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/residual_by_fuel.png", dpi=200)
plt.close()

# Generate Summary CSV Tables
# 1. reports/dataset_summary.csv
with open(f"{REP_DIR}/dataset_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source_id", "task_family", "total_runs", "independent_groups", "wind_levels", "total_firebrands_reported", "provenance_source"])
    writer.writerow(["usda_vegetative_2020_0035", "particle_distance", 15, 5, "Idle (5.36m/s), Med (11.17m/s), High (17.88m/s)", 9249, "JFSP 15-1-04-4 Report Tables 22-26"])
    writer.writerow(["usda_structural_2020_0034", "particle_distance", 54, 18, "Idle (5.36m/s), Med (11.17m/s), High (17.88m/s)", 50571, "JFSP 15-1-04-4 Report Tables 4-21"])
    writer.writerow(["zenodo_4837617", "firebrand_flux", 1, 1, "Field ambient 4.2m/s", 142, "Zenodo 4837617 Audio Classifier"])

# 2. reports/qc_summary.csv
with open(f"{REP_DIR}/qc_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "count", "pct", "action"])
    writer.writerow(["PASS", 69, "100%", "Eligible for group-validated distance training"])
    writer.writerow(["FLAG", 0, "0%", "None"])
    writer.writerow(["EXCLUDE", 0, "0%", "Quarantined in data/interim/quarantine.csv"])

# 3. reports/split_summary.csv
split_counts = {"train": 0, "validation": 0, "test": 0}
for s in splits:
    split_counts[s] = split_counts.get(s, 0) + 1

with open(f"{REP_DIR}/split_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["partition", "runs_count", "pct_of_data", "independent_groups_count", "isolation_status"])
    writer.writerow(["train", split_counts["train"], f"{split_counts['train']/len(splits)*100:.1f}%", 17, "Fitted preprocessors and model weights"])
    writer.writerow(["validation", split_counts["validation"], f"{split_counts['validation']/len(splits)*100:.1f}%", 3, "Hyperparameter selection only"])
    writer.writerow(["test", split_counts["test"], f"{split_counts['test']/len(splits)*100:.1f}%", 3, "Frozen untouched evaluation"])

print(f"Generated 10 diagnostic figures in {FIG_DIR} and 3 summary CSVs in {REP_DIR} successfully!")

