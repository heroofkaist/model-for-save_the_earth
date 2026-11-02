# Dataset Card — Save the Earth Observational Datasets (Version 2.0)

## 1. Summary
This dataset collection provides audited, quality-controlled, provenance-tracked experimental data for studying wildfire firebrand transport and landing distributions.

---

## 2. Dataset Sources & Verified Provenance

### A. USDA Forest Service / JFSP 15-1-04-4 Wind-Tunnel Experiments
* **Source DOIs:** `10.2737/RDS-2020-0035` (Vegetative) & `10.2737/RDS-2020-0034` (Structural).
* **Official Report:** Joint Fire Science Program Project 15-1-04-4 Final Report (`data/raw/15-1-04-4/15-1-04-4_final_report.pdf`, SHA-256: `deb78532380f80332321df3553a5515e9634782e488e10f397f4a07d46af84bb`).
* **Facility:** Insurance Institute for Business & Home Safety (IBHS) full-scale wind-tunnel facility (Richburg, SC).
* **Statistical Unit of Observation:** **Aggregated Experimental Run** ($N=69$ runs across 23 independent fuel groups).
* **Firebrand Population:** Represents 60,370 physical firebrands collected in water pans:
  * 9,249 vegetative firebrands across 5 fuels (Chamise, Saw palmetto, Loblolly pine, Leyland cypress, Little bluestem grass).
  * 51,121 structural firebrands across 18 building assemblies (Fences, Re-entrant corners, Shake roofs, Wall-roof mockups).
* **Crucial Distinction:** **The number of independent samples is 69 experimental runs, NOT 60,370 rows.** Models predict the run-level median landing distance ($\tilde{x}_i$).
* **Measurement Methodology:** Water-pan collection downwind in 46 pans (capture area $0.65\,\text{m}^2$ each) arranged across 17 rows; oven drying to zero moisture; digital balance ($\pm 0.0001\,\text{g}$ resolution); high-resolution image analysis (MATLAB).
* **Preserved Statistical Moments per Run:**
  * Sample size $N_i$ (ranging from 170 to 1,800+ particles).
  * Distance: mean, median, standard deviation ($\sigma_i$), skewness.
  * Projected Area: mean, median, standard deviation, skewness.
  * Mass: mean, median, standard deviation, skewness.
  * Correlations: $r(\text{mass}, \text{area})$, $r(\text{mass}, \text{dist})$, $r(\text{area}, \text{dist})$.
* **DOI Access Realities:** Direct programmatic downloads from `www.fs.usda.gov/rds/archive/` are blocked by Microsoft Azure Front Door WAF (HTTP 403 Forbidden with `x-azure-ref` challenge) for non-US/automated requests. The authentic official final project report was downloaded from the Northern Rockies Fire Science Network archive (`nrfirescience.org`), verified cryptographically, and parsed without data fabrication.
* **License:** CC0 1.0 Universal (Public Domain).

### B. Zenodo Acoustic Arrival Observations
* **Record ID:** Zenodo 4837617 (`10.5281/zenodo.4837617`).
* **Title:** Quantifying firebrand production and transport using acoustic analysis of in-fire cameras.
* **Files Ingested:** `Intensity_from_obs_flame_height.gpkg`, `train.wav`, `FireArrival_sec_after_ign.tif`, R scripts.
* **Task Family:** `firebrand_flux` (Acoustic impact rate, strictly separated from particle distance).
* **License:** CC BY 4.0.

### C. Zenodo Thermal Piles
* **Record ID:** Zenodo 1406207 (`10.5281/zenodo.1406207`).
* **Nature:** 2.8 GB pile thermal heat flux archive. Kept external; metadata and endpoint manifest audited.
* **Task Family:** `thermal_survival` (Kept strictly separated).
* **License:** CC BY 4.0.

---

## 3. Preprocessing & Quality Control (QC)
All preprocessing follows strict rules in `research_data.py`:
* **Unit Standardization:** Converted to pure SI units: mass in $\text{kg}$, area in $\text{m}^2$, distance in $\text{m}$, wind speed in $\text{m/s}$.
* **Grouping:** Grouped by fuel species/assembly (`experiment_id`), ensuring all wind regimes of a fuel remain in the same split partition.
* **No Label Imputation:** Missing labels remain null and are never imputed.
* **Audited Records:** 69 experimental runs passed QC (`qc_status = pass`), 0 quarantined.

---

## 4. Frozen Group-Disjoint Split
Stored immutably in `data/manifests/split_v1.csv` (SHA-256: `4339bb6f...`):
* **Development Partition:** 60 runs (20 independent fuel groups: 17 train + 3 validation). Used for 5-Fold Group Cross-Validation.
* **Test Partition:** 9 runs (3 independent fuel groups: `mockup_fibercement_combustible`, `mockup_wood_noncombustible_roof`, `corner_osb_osb`). Strictly frozen and untouched until final evaluation.
* **Group Leakage:** Exactly 0% group leakage between partitions.
