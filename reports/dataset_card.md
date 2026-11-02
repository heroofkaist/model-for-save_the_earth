# Dataset Card — Save the Earth Observational Datasets

## 1. Summary
This dataset collection provides audited, quality-controlled, provenance-tracked experimental data for studying wildfire firebrand transport and landing distributions.

## 2. Dataset Sources & Verified Provenance

### A. USDA Forest Service / JFSP 15-1-04-4 Wind-Tunnel Experiments
* **Source DOI:** `10.2737/RDS-2020-0035` (Vegetative) & `10.2737/RDS-2020-0034` (Structural).
* **Official Report:** Joint Fire Science Program Project 15-1-04-4 Final Report (`data/raw/15-1-04-4/15-1-04-4_final_report.pdf`, SHA-256: `deb78532380f80332321df3553a5515e9634782e488e10f397f4a07d46af84bb`).
* **Facility:** Insurance Institute for Business & Home Safety (IBHS) full-scale wind-tunnel facility (Richburg, SC).
* **Sample Size:** 9,249 vegetative firebrands across 5 fuels; 50,571 structural firebrands across 18 building assemblies. Total: 59,820 firebrands in 69 experimental runs.
* **Measurement Methodology:** Water-pan collection downwind in 46 pans (capture area $0.65\,\text{m}^2$ each) arranged across 17 rows; oven drying to zero moisture; digital balance ($\pm 0.0001\,\text{g}$ resolution); high-resolution image analysis (MATLAB).
* **Reported Uncertainties:** Mass $\pm 0.0169\,\text{g}$, Projected Area $\pm 0.09\,\text{mm}^2$, Flying Distance $\pm 0.22\,\text{m}$ (relative to burning sample).
* **License:** CC0 1.0 Universal (Public Domain).

### B. Zenodo Acoustic Arrival Observations
* **Record ID:** Zenodo 4837617 (`10.5281/zenodo.4837617`).
* **Title:** Quantifying firebrand production and transport using acoustic analysis of in-fire cameras.
* **Files Ingested:** `Intensity_from_obs_flame_height.gpkg`, `train.wav`, `FireArrival_sec_after_ign.tif`, R scripts.
* **License:** CC BY 4.0.

### C. Zenodo Thermal Piles
* **Record ID:** Zenodo 1406207 (`10.5281/zenodo.1406207`).
* **Nature:** 2.8 GB pile thermal heat flux archive. Kept external; metadata and endpoint manifest audited.
* **License:** CC BY 4.0.

## 3. Preprocessing & Quality Control (QC)
All preprocessing follows strict rules in `research_data.py`:
* **Unit Standardization:** Converted to pure SI units: mass in $\text{kg}$, area in $\text{m}^2$, distance in $\text{m}$, wind speed in $\text{m/s}$.
* **Grouping:** Grouped by fuel species/assembly (`experiment_id`), ensuring all wind regimes of a fuel remain in the same split partition.
* **No Label Imputation:** Missing labels remain null and are never imputed.
* **Audited Records:** 69 experimental runs passed QC (`qc_status = pass`), 0 quarantined.

## 4. Frozen Group-Disjoint Split
Stored immutably in `data/manifests/split_v1.csv` (SHA-256: `4339bb6f...`):
* **Training Partition:** 51 runs (17 independent fuel groups, ~74%).
* **Validation Partition:** 9 runs (3 independent fuel groups, ~13%).
* **Test Partition:** 9 runs (3 independent fuel groups, ~13%).
* **Overlap:** Exactly zero group leakage between partitions.
