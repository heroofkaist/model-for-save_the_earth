# Save the Earth — Wildfire Firebrand Transport & Deposition Prediction Framework

An experimentally grounded, group-validated computational framework for predicting wildfire ember transport, landing distributions, and spotting risk (**KAISTian Amurs Research Project**).

Repository: [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
Primary Branch: `main`  
Development Branch: `feature/independent-experimental-validation`

---

## ⚡ Quickstart: Single-Command Execution

To execute the entire scientific pipeline end-to-end (physics verification, sensitivity analysis, data extraction, quality control, 5-fold group cross-validation, multi-model ensemble training, diagnostic plotting, and test suite execution):

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the complete pipeline with ONE command
python3 run_pipeline.py
```

### 🎯 Operational Firebrand Distance Prediction
Query the calibrated prospective model for predicted median travel distance and conformal uncertainty bounds under any wind and vegetation scenario:

```bash
python3 run_pipeline.py --predict --wind 15.0 --category vegetative
```

Example Output:
```text
================ EXPERIMENTAL FIREBRAND DISTANCE PREDICTION ================
Model Version: v2.0-group-cv-calibrated | Dataset: USDA JFSP 15-1-04-4 (Aggregated Runs)
Input Parameters: Nominal Wind = 15.00 m/s | Fuel Category = vegetative
---------------------------------------------------------------------------
Model Predictions for Experimental Landing Distance:
  Prospective Ridge Regression:       2.34 m
  Prospective Random Forest (depth=2): 2.12 m
  Prospective Parsimonious Ensemble:  2.23 m
  -------------------------------------------------------------
  ★ PREDICTED RUN MEDIAN DISTANCE:    2.23 m
  ★ 80% CONFORMAL INTERVAL (Median):  [0.77 m  —  3.92 m] (Radius: +/- 1.58 m)
  ★ ESTIMATED PARTICLE DISPERSION:    [0.00 m  —  5.30 m] (Spread of individual embers around median)
===========================================================================
⚠️  SCIENTIFIC DISCLAIMER:
   This prediction estimates physical deposition distance in wind-tunnel water pans.
   It does NOT predict particle thermal survival, fuel bed ignition, or macroscopic spot fires.
===========================================================================
```

---

## 🔬 Scientific Architecture & Benchmarking Results

### 1. Six Distinct Research Targets
The framework strictly separates six distinct physical processes to avoid confounding:
1. **Firebrand Generation** (`src/save_the_earth/interfaces/generation.py`)
2. **Atmospheric Transport** (`src/save_the_earth/interfaces/transport.py`)
3. **Landing & Deposition** (`src/save_the_earth/interfaces/deposition.py`) — *Currently active*
4. **Thermal Survival** (`src/save_the_earth/interfaces/thermal.py`)
5. **Receptive Fuel Ignition** (`src/save_the_earth/interfaces/ignition.py`)
6. **Wildfire Spotting** (`src/save_the_earth/interfaces/spotting.py`)

> ⚠️ **Core Rule:** Never confuse an ember landing with a spot fire ignition. Spot fires represent successful ignitions conditional on fuel receptivity; wind-tunnel water pans measure mechanical transport.

### 2. Evaluated Performance on 5-Fold Group CV & Frozen Test Partition (`data/manifests/split_v1.csv`)
Models evaluated across 20 development groups and tested on 3 untouched, independent fuel groups (`mockup_fibercement_combustible`, `mockup_wood_noncombustible_roof`, `corner_osb_osb`):

| Model Name | Configuration | 5-Fold Group CV MAE | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Global Median** | Prospective Reference | 1.027 ± 0.427 m | 1.838 m | 2.660 m | 0.832 m | -1.535 m |
| **Prospective Ridge ($\alpha=10.0$)** | Prospective Operational | **1.062 ± 0.347 m** | **1.721 m** | **2.315 m** | **0.892 m** | -1.012 m |
| **Prospective Random Forest (depth=2)** | Prospective Operational | 1.101 ± 0.320 m | 1.730 m | 2.306 m | 0.912 m | -0.981 m |
| **Prospective Parsimonious Ensemble** | Prospective Blend | 1.079 ± 0.331 m | 1.726 m | 2.310 m | 0.902 m | -0.997 m |
| **Retrospective Gradient Boosting** | Retrospective Diagnostic | N/A | 1.779 m | 2.279 m | 0.825 m | -0.716 m |
| **Physical RK4 Ballistic ($h=2.0$ m)** | Mechanistic Physical | N/A | 7.921 m | 8.922 m | 7.921 m | +7.921 m |
| **Physical RK4 Ballistic ($h=1.0$ m)** | Mechanistic Physical | N/A | 3.208 m | 4.176 m | 3.006 m | +3.006 m |

* **Prospective Models:** Rely solely on pre-flight observable quantities (wind speed, wind speed², fuel category).
* **Retrospective Models:** Include post-landing recovered particle mass and projected area. **Must not be advertised as forward-looking operational predictors.**
* **Detailed Scientific Comparison:** See [`reports/MODEL_IMPROVEMENT_COMPARISON.md`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/reports/MODEL_IMPROVEMENT_COMPARISON.md).

---

## 🔬 Statistically Reconstructed Distribution Benchmark (JFSP 15-1-04-4 Moments)

In addition to run-level median benchmarks, models are evaluated against an **8,841-particle held-out test distribution** reconstructed from the empirical summary moments published in the USDA JFSP 15-1-04-4 experiments across strictly unseen fuel groups (`corner_cedar_osb`, `recycled_shake_roof`, `mockup_fibercement_combustible`):

| Model Category | Model Name | Held-Out Test MAE (m) | RMSE (m) | Median AE (m) | Signed Bias (m) | Throughput (particles/s) |
|---|---|---:|---:|---:|---:|---:|
| **Baseline** | Global Median Baseline | 5.161 | 8.512 | 2.132 | -4.492 | N/A |
| **Statistical (Operational)** | **Prospective Ridge** | **4.689** | **7.556** | 2.683 | -2.558 | > 1,000,000 |
| **Statistical (Operational)** | Prospective Gradient Boosting | 4.875 | 7.859 | **2.657** | -2.941 | ~450,000 |
| **Statistical (Operational)** | Prospective Random Forest | 4.890 | 7.866 | 2.664 | -2.930 | ~350,000 |
| **Deep Learning Surrogate** | Model B: Neural Surrogate (Prototype) | 5.178 | 7.704 | 3.420 | -1.752 | **733,756** |
| **Mechanistic Aerodynamics** | Mechanistic RK4 Integration | 5.577 | 8.760 | 3.335 | -1.507 | 449 |
| **Physics-Informed Hybrid** | Model C: Hybrid Surrogate (Fixed) | 6.302 | 8.262 | 5.055 | +1.453 | > 950,000 |
| *Theoretical Reference* | *Oracle Constant Predictor (Median)* | *4.800* | *7.230* | *2.300* | *0.000* | *N/A* |

*Note: In the test set, the empirical landing distance standard deviation is $\sigma = 7.23$ m, meaning single-point particle predictions hit an irreducible physical scatter floor around ~4.8 m MAE. In contrast, for operational wildfire spotting, the relevant metric is the **expected deposition median distance**, where our models achieve **1.57 m to 1.73 m MAE**.*

* **Comprehensive Acquisition Audit:** See [`reports/REAL_DATA_ACQUISITION.md`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/reports/REAL_DATA_ACQUISITION.md).
* **Validation Methodology & Physical Analysis:** See [`reports/INDEPENDENT_MODEL_VALIDATION.md`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/reports/INDEPENDENT_MODEL_VALIDATION.md).
* **Machine-Readable Provenance Registry:** See [`data/manifests/external_validation_sources.json`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/data/manifests/external_validation_sources.json).

---

## 📁 Repository Structure

```text
├── run_pipeline.py                  # Single-command end-to-end master orchestration pipeline
├── ember_simulation.py              # Mechanistic 4th-order Runge-Kutta ballistic simulator
├── physics_sensitivity.py           # Aerodynamic sensitivity analysis (release height & ground shear)
├── parse_usda_report_tables.py      # Automated table extractor for USDA JFSP 15-1-04-4
├── build_canonical_datasets.py      # Standardized SI-unit dataset builder
├── research_data.py                 # Cryptographic QC auditor & group-disjoint partitioner
├── train_distance_benchmarks.py     # 5-fold Group CV and benchmark training suite
├── generate_diagnostics.py          # Diagnostic figure and publication table generator
├── inspect_raw.py                   # Raw archive and checksum auditor
├── src/save_the_earth/
│   ├── data/
│   │   ├── openptv.py               # OpenPTV 3D trajectory loader & track reconstructor
│   │   └── usda_particles.py        # USDA particle-level benchmark dataset constructor
│   ├── evaluation/
│   │   ├── evaluate_trajectories.py # Trajectory benchmark on synthetic & run data
│   │   ├── evaluate_usda_particles.py # 60,370 particle-level experimental benchmark
│   │   └── openptv_validation.py    # OpenPTV experimental velocity validation
│   ├── models/                      # Neural and Hybrid trajectory surrogates (Model B & C)
│   ├── physics/                     # Advanced 3D aerodynamics ODE engine (RK4)
│   ├── simulation/                  # 10k physical trajectory generator
│   └── interfaces/                  # Modular scientific interfaces for 6 research domains
├── data/
│   ├── manifests/
│   │   ├── source_registry.json     # Provenance ledger of all research datasets
│   │   ├── external_validation_sources.json # Audit of external validation sources
│   │   ├── usda_particles_manifest.json # Manifest of 60,370 firebrand particles
│   │   ├── checksums.sha256         # SHA-256 integrity checksums for all raw files
│   │   └── split_v1.csv             # Frozen, group-disjoint train/val/test split
│   ├── processed/
│   │   ├── usda_individual_firebrands.csv  # 60,370 particle-level experimental records
│   │   ├── openptv_reconstructed_tracks.csv # Reconstructed 3D experimental tracks
│   │   ├── canonical_particle_distance.csv # Verified SI-unit experimental runs (69 runs)
│   │   ├── canonical_firebrand_flux.csv    # Zenodo 4837617 acoustic arrival events
│   │   └── jfsp_experiments_summary.csv   # Extracted 69 runs with full moments
│   └── interim/                     # Quality-control audited and eligible views
├── distance_results/
│   ├── benchmark_comparison.json    # Exact metrics for all evaluated models
│   ├── group_cv_metrics.csv         # 5-fold Group Cross-Validation leaderboard
│   ├── metrics_table.csv            # Summary metrics table
│   ├── physics_sensitivity.json     # Aerodynamic sensitivity analysis results
│   └── test_predictions.csv        # Per-record predictions on held-out test set
├── reports/
│   ├── REAL_DATA_ACQUISITION.md     # Audit of external repositories (USDA, EIDC, OpenPTV, CSIRO)
│   ├── INDEPENDENT_MODEL_VALIDATION.md # Independent particle-level validation report
│   ├── OPENPTV_EXPERIMENTAL_VALIDATION.md # OpenPTV 3D tracking validation report
│   ├── TRAJECTORY_TRAINING_10K.md   # 10k physical trajectory training report
│   ├── TECHNICAL_REPORT.md          # Comprehensive scientific technical report (v2.0)
│   ├── MODEL_IMPROVEMENT_COMPARISON.md # Side-by-side v1.0 vs v1.1 vs v2.0 comparison
│   ├── dataset_card.md              # Dataset card detailing provenance & measurements
│   ├── model_card.md                # Model card detailing feature scope & limitations
│   └── figures/                     # Diagnostic figures (dispersion, model comparisons, wind trends)
└── tests/                           # 39 automated unit and integrity tests
```

---

## 🧪 Automated Testing
Run the complete unit test suite with:
```bash
python3 -m unittest discover -s tests -v
```
All 39 unit tests verify group leakage prevention, physical conservation laws, domain interface schemas, frozen checkpoint integrity, OpenPTV track reconstruction, and particle benchmark consistency without error.
