# Save the Earth — Wildfire Firebrand Transport & Spotting Prediction Framework

An experimentally grounded, group-validated computational framework for predicting wildfire ember transport, landing distributions, and spotting risk (**KAISTian Amurs Research Project**).

Repository: [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
Development Branch: `feature/ember-training-foundation`  
Draft Pull Request: [#1](https://github.com/heroofkaist/model-for-save_the_earth/pull/1)

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the full unit test suite (20 tests)
python3 -m unittest discover -s tests -v

# 3. Verify RK4 aerodynamic simulation physics
python3 ember_simulation.py --verify

# 4. Extract authentic experimental tables from USDA JFSP 15-1-04-4 Report
python3 parse_usda_report_tables.py

# 5. Build canonical observation datasets
python3 build_canonical_datasets.py

# 6. Audit quality control and verify frozen split
python3 research_data.py --input data/processed/canonical_particle_distance.csv --outdir data/interim

# 7. Execute benchmark model training on the frozen test partition
python3 train_distance_benchmarks.py

# 8. Generate diagnostic plots and summary tables
python3 generate_diagnostics.py
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

### 2. Evaluated Measured Performance on Frozen Test Split (`data/manifests/split_v1.csv`)
All models are evaluated on untouched, independent experimental fuel groups (`mockup_fibercement_combustible`, `lattice_fence`, `non_frt_cedar_shake_roof`):

| Model Name | Configuration | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Median Baseline** | Prospective (Reference) | 1.785 m | 2.577 m | 0.832 m | -1.385 m |
| **Category Median Baseline** | Prospective (Reference) | 1.772 m | 2.556 m | 0.830 m | -1.347 m |
| **Prospective Ridge** | Prospective (Operational) | 1.704 m | 2.265 m | 0.892 m | -0.911 m |
| **Prospective Random Forest** | Prospective (Operational) | **1.700 m** | **2.261 m** | 0.887 m | -0.907 m |
| **Retrospective Ridge** | Retrospective (Diagnostic) | 1.669 m | 2.268 m | 0.831 m | -0.878 m |
| **Retrospective Gradient Boosting** | Retrospective (Diagnostic) | **1.684 m** | **2.059 m** | 0.812 m | **-0.402 m** |
| **Physical RK4 Ballistic** | Physical (Mechanistic) | 7.921 m | 8.922 m | 7.921 m | +7.921 m |

* **Prospective Models:** Rely solely on pre-flight observable quantities (wind speed, wind speed², fuel category).
* **Retrospective Models:** Include post-landing recovered particle mass and projected area. **Must not be advertised as forward-looking operational predictors.**
* **Physical Ballistic Overshoot:** Naive ballistic simulation overpredicts distance by ~7.9 m due to unmodeled structural wake recirculation in wind-tunnel water pans.

---

## 📁 Repository Structure

```text
├── data/
│   ├── manifests/
│   │   ├── source_registry.json     # Provenance ledger of all research datasets
│   │   ├── checksums.sha256         # SHA-256 integrity checksums for all raw files
│   │   └── split_v1.csv             # Frozen, group-disjoint train/val/test split
│   ├── processed/
│   │   ├── canonical_particle_distance.csv # Verified SI-unit experimental runs
│   │   ├── canonical_firebrand_flux.csv    # Zenodo 4837617 acoustic arrival events
│   │   └── jfsp_experiments_summary.csv   # Extracted 69 runs from USDA report
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
├── tests/                           # 20 automated unit tests (leakage, units, physics)
├── .github/workflows/ci.yml         # Automated GitHub Actions validation pipeline
├── ember_simulation.py              # Verified 4th-order Runge-Kutta aerodynamic simulator
├── parse_usda_report_tables.py      # Automated table extractor for USDA report
├── build_canonical_datasets.py      # Canonical SI dataset builder
├── research_data.py                 # Non-destructive QC and group partitioning
├── train_distance_benchmarks.py     # Prospective vs. retrospective benchmark trainer
├── generate_diagnostics.py          # Diagnostic figure and table generator
└── requirements.txt                 # Reproducible environment specifications
```

---

## 🇦🇺 International Research Collaboration (CSIRO & KFS)
Detailed in [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md) and [docs/AUSTRALIA_RESEARCH_PLAN.md](docs/AUSTRALIA_RESEARCH_PLAN.md):
* **Target Partner:** CSIRO Bushfire Behaviour Research Laboratory (Black Mountain, ACT) & UNSW Canberra.
* **Core Experimental Request:** Access to vertical wind-tunnel facilities with synchronized multi-view high-speed tracking to reconstruct in-flight $(x, y, z, t)$ trajectories, instantaneous combustion mass loss $\dot{m}(t)$, and thermal extinction thresholds.
* **Domestic Field Program (Korea):** Integration with Korea Forest Service (KFS) incident tracking and KMA Automated Weather Station (AWS) data for regional spotting validation.
