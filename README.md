# Save the Earth — Wildfire Firebrand Transport & Spotting Prediction Framework

An experimentally grounded, group-validated computational framework for predicting wildfire ember transport, landing distributions, and spotting risk (**KAISTian Amurs Research Project**).

Repository: [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
Primary Branch: `main`

---

## ⚡ Quickstart: Single-Command Execution

To execute the entire scientific pipeline end-to-end (verification, data extraction, quality control, multi-model ensemble training, diagnostic plotting, and test suite execution):

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the complete pipeline and ensemble with ONE command
python3 run_pipeline.py
```

### 🎯 Operational Firebrand Landing & Spotting Hazard Prediction
Once trained, query the multi-model ensemble for operational forecasts under any wind, fuel moisture, and vegetation scenario:

```bash
# Predict landing distance and 90th percentile spotting danger envelope:
python3 run_pipeline.py --predict --wind 15.0 --moisture 9.0 --category vegetative
```

Example Output:
```text
================ OPERATIONAL ENSEMBLE PREDICTION ================
Inputs: Wind = 15.00 m/s | Moisture Content = 9.0% | Fuel Category = vegetative
-----------------------------------------------------------------
Model Predictions for Landing Distance:
  Prospective Random Forest:          2.00 m
  Prospective Gradient Boosting:      2.14 m
  Prospective Ridge Regression:       1.61 m
  Prospective MLP Neural Network:     3.46 m
  -------------------------------------------------------------
  ★ ENSEMBLE MEAN PREDICTION:         2.31 m
  ★ 80% CONFIDENCE INTERVAL (P10-P90): [0.98 m  —  2.40 m]
  ★ SPOTTING HAZARD ENVELOPE (90th%):  2.40 m downwind
=================================================================
```

---

## 🔬 Scientific Architecture & Benchmarking Results

### 1. Six Distinct Research Targets
The framework strictly separates six distinct physical processes to avoid confounding:
1. **Firebrand Generation** (source fuels, initial particle size/mass)
2. **Atmospheric Transport** (aerodynamic drag, plume buoyancy, 3D trajectories)
3. **Landing & Deposition** (travel distance, landing coordinates, spatial dispersion)
4. **Thermal Survival** (in-flight charring, mass loss, cooling)
5. **Receptive Fuel Ignition** (fuel moisture, heat flux, ignition probability)
6. **Wildfire Spotting** (mesoscale secondary fire occurrences from line-scans)

> ⚠️ **Core Rule:** Never confuse an ember landing with a spot fire ignition. Spot fires represent successful ignitions conditional on fuel receptivity; wind-tunnel water pans measure mechanical transport.

### 2. Evaluated Performance on Held-Out Test Split (`data/manifests/split_v1.csv`)
All models are evaluated on untouched, independent experimental fuel groups (`mockup_fibercement_combustible`, `lattice_fence`, `non_frt_cedar_shake_roof`):

| Model Name | Configuration | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Median Baseline** | Prospective (Reference) | 1.785 m | 2.577 m | 0.832 m | -1.385 m |
| **Category Median Baseline** | Prospective (Reference) | 1.772 m | 2.556 m | 0.830 m | -1.347 m |
| **Prospective Ridge** | Prospective (Operational) | 1.731 m | 2.286 m | 0.892 m | -0.908 m |
| **Prospective Random Forest** | Prospective (Operational) | 1.935 m | 2.525 m | 0.912 m | -1.138 m |
| **Prospective Gradient Boosting** | Prospective (Operational) | 1.861 m | 2.488 m | 0.931 m | -1.157 m |
| **Prospective MLP Neural Net** | Prospective (Operational) | 1.847 m | 2.536 m | 0.938 m | -1.172 m |
| **Prospective Multi-Model Ensemble** | Prospective (Operational Blend) | 1.844 m | 2.450 m | 0.918 m | -1.094 m |
| **Retrospective Ridge** | Retrospective (Diagnostic) | 1.673 m | 2.272 m | 0.831 m | -0.877 m |
| **Retrospective Random Forest** | Retrospective (Diagnostic) | 1.635 m | 2.152 m | 0.825 m | -0.983 m |
| **Retrospective Gradient Boosting** | Retrospective (Diagnostic) | **1.571 m** | **2.130 m** | **0.812 m** | **-0.786 m** |
| **Retrospective Multi-Model Ensemble** | Retrospective (Diagnostic Blend) | 1.787 m | 2.439 m | 0.854 m | -0.977 m |
| **Physical RK4 Ballistic** | Physical (Mechanistic) | 7.921 m | 8.922 m | 7.921 m | +7.921 m |

* **Prospective Models:** Rely solely on pre-flight observable quantities (wind speed, gust peak, fuel moisture content %, fuel category).
* **Retrospective Models:** Include post-landing recovered particle mass and projected area. **Must not be advertised as forward-looking operational predictors.**
* **Physical Ballistic Overshoot:** Naive ballistic simulation overpredicts distance by ~7.9 m due to unmodeled structural wake recirculation in wind-tunnel water pans.

---

## 📁 Repository Structure

```text
├── run_pipeline.py                  # Single-command end-to-end master orchestration pipeline
├── ember_simulation.py              # Mechanistic 4th-order Runge-Kutta ballistic simulator
├── parse_usda_report_tables.py      # Automated table extractor for USDA JFSP 15-1-04-4
├── build_canonical_datasets.py      # Standardized SI-unit dataset builder
├── research_data.py                 # Cryptographic QC auditor & group-disjoint partitioner
├── train_distance_benchmarks.py     # Multi-model machine learning benchmark suite & ensemble
├── generate_diagnostics.py          # Diagnostic figure and publication table generator
├── inspect_raw.py                   # Raw archive and checksum auditor
├── data/
│   ├── manifests/
│   │   ├── source_registry.json     # Provenance ledger of all research datasets
│   │   ├── checksums.sha256         # SHA-256 integrity checksums for all raw files
│   │   └── split_v1.csv             # Frozen, group-disjoint train/val/test split
│   ├── processed/
│   │   ├── canonical_particle_distance.csv # Verified SI-unit experimental runs (69 runs)
│   │   ├── canonical_firebrand_flux.csv    # Zenodo 4837617 acoustic arrival events
│   │   └── jfsp_experiments_summary.csv   # Extracted 69 runs with full moments
│   ├── raw/                         # Immutable raw downloads (gitignored)
│   │   ├── 15-1-04-4/               # Official final project report (3.1 MB)
│   │   └── zenodo_4837617/          # Ingested acoustic field detection data
│   └── interim/                     # Quality-control audited and eligible views
├── distance_results/
│   ├── benchmark_comparison.json    # Exact metrics for all evaluated models
│   ├── metrics_table.csv            # Summary metrics table
│   └── test_predictions.csv        # Per-record predictions on held-out test set
├── reports/
│   ├── TECHNICAL_REPORT.md          # 14-section comprehensive scientific technical report
│   ├── dataset_card.md              # Dataset card detailing provenance & measurements
│   ├── model_card.md                # Model card detailing feature scope & limitations
│   ├── dataset_summary.csv          # Summary of rows, groups, and fuel coverage
│   ├── qc_summary.csv               # Pass / flag / exclude audit summary
│   ├── split_summary.csv            # Split partition sizes and group counts
│   └── figures/                     # 10 diagnostic figures (histograms, calibration, residuals)
└── tests/                           # 22 automated unit and integrity tests
```

---

## 🧪 Automated Testing
Run the complete unit test suite with:
```bash
python3 -m unittest discover -s tests -v
```
All 22 unit tests verify data leakage prevention, physical conservation laws, format integrity, and model inference without error.
