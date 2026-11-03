# Save the Earth — Real Experimental Data Acquisition Report

**Project:** Save the Earth — Computational Wildfire Firebrand Transport & Spotting  
**Date:** November 2026  
**Research Branch:** `feature/independent-experimental-validation`  
**Status:** Complete Experimental Audit & Benchmark Ingestion  
**Provenance Registry:** [`data/manifests/external_validation_sources.json`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/data/manifests/external_validation_sources.json)

---

## 1. Executive Summary & Audit Scope

A primary requirement of scientifically defensible wildfire research is distinguishing between:
1. **Measured experimental observations** (physical tests in wind tunnels, burns, and photogrammetric chambers).
2. **Numerical simulation outputs** (FDS LES field simulations, physics-based ODE/PDE integrators).
3. **Synthetic training distributions** (surrogate training datasets with parameter sweeps).

This report documents our exhaustive acquisition audit across public, academic, and government firebrand datasets to obtain real physical observations for validating our physics simulator and machine-learning surrogates.

### Summary of Evaluated Data Sources

| Source Identifier | Description | Access Status | Usable for Airborne Trajectory? | Usable for Landing Distance? |
|---|---|---|---|---|
| **USDA RDS-2020-0035** | Vegetative Firebrands (9,249 particles, 21 runs) | `blocked_by_host_waf` | No (Only water pans measured) | **Statistical Distribution Benchmark** (Copula reconstruction from moments) |
| **USDA RDS-2020-0034** | Structural Firebrands (51,121 particles, 48 runs) | `blocked_by_host_waf` | No (Only water pans measured) | **Statistical Distribution Benchmark** (Copula reconstruction from moments) |
| **UK EIDC FDS Oregon** | Sycan Marsh FDS simulation outputs (DOI: `10.5285/f9180989-6528-404e-8110-e3dcc205a6a1`) | `downloaded_and_inspected` | **No** (Stationary fuel elements) | **No** (Fuel element mass loss) |
| **OpenPTV 2-Cam Firebrands** | UMD/NIST 3D-PTV tracking (`github.com/OpenPTV/2_cam_firebrands`) | `downloaded_and_verified` | **Partial near-field geometry** (Unconfirmed FPS / Delta t) | No (Particles exit camera FOV) |
| **CSIRO 2026** | Untethered burning firebrands in vertical wind tunnel (DOI: `10.1016/j.jaecs.2026.100520`) | `author_request` | Pending data sharing agreement | N/A (Vertical tunnel) |
| **NIST Emberometer** | Dual-camera ember tracking at NFRL (`nist.gov/.../nist-emberometer`) | `author_request` | Pending collaborative request | N/A |

---

## 2. Priority 1 — USDA JFSP 15-1-04-4 Datasets

### 2.1 Background & Experimental Context
The Joint Fire Science Program (JFSP) Project 15-1-04-4 (*"Characterizing Firebrands Generated from WUI Fuels"*, Hedayati et al., 2018) conducted full-scale firebrand generation and transport experiments inside the Insurance Institute for Business & Home Safety (IBHS) Research Center wind tunnel.

- **Collection Methodology:** Firebrands were generated from full-scale burning vegetative fuels (conifers, shrubs) and structural building assemblies (privacy fences, lattice fences, corner wall mockups, cedar shake roofs). Embers were carried downwind in controlled wind tunnel flows (nominal speeds: 5.36 m/s, 11.17 m/s, and 17.88 m/s).
- **Physical Measurements:** Embers were collected in water-filled pans arranged in rows downwind to extinguish flaming/glowing combustion and preserve physical geometry. For every collected particle, projected surface area ($A$) was measured via high-resolution 2D flatbed optical scanning, and dry mass ($m$) was measured using analytical microbalances. Landing distance ($x$) was recorded by water pan row location.

### 2.2 Direct Download Audit & Access Status
We probed the canonical digital repositories for the two primary archives:
- **Vegetative Firebrands:** DOI `10.2737/RDS-2020-0035`, Figshare `27009073`, Mendeley Data `10.17632/3cmtb9n9ph.2`.
- **Structural Firebrands:** DOI `10.2737/RDS-2020-0034`, Figshare `27009070`, Mendeley Data `10.17632/x6yww4dnm6.2`.

**Audit Findings:**
1. **USDA Forest Service Research Data Archive (`www.fs.usda.gov`):**  
   Direct HTTP GET requests to `https://www.fs.usda.gov/rds/archive/products/RDS-2020-0035/RDS-2020-0035.zip` and `RDS-2020-0034.zip` return `HTTP Error 403: Forbidden`. The host infrastructure is protected by an Azure Front Door Web Application Firewall (WAF) that blocks non-browser automated client headers.
2. **Figshare API (`api.figshare.com/v2/articles/27009073` and `27009070`):**  
   Metadata stubs exist, but the hosted file list is empty (`size: 0, download_url: https://doi.org/10.2737/RDS-2020-0035`), pointing back to the WAF-protected USDA portal.
3. **Mendeley Data API (`data.mendeley.com/public-api/datasets/...`):**  
   Metadata records resolve, but return empty file arrays (`files: []`).

### 2.3 Scientific Nature of the Ingested Benchmark: Statistical Reconstructed Distribution
> [!IMPORTANT]
> **Scientific Integrity Clarification: Statistical Reconstructed Distribution vs. Raw Sensor Observations**  
> Because the direct particle catch tables at `www.fs.usda.gov` are protected by Azure WAF (403), the particle benchmark in [`src/save_the_earth/data/usda_particles.py`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/src/save_the_earth/data/usda_particles.py) is a **synthetic parametric distribution reconstruction (moment-matched via multivariate Gaussian copula)**. It is **NOT** a collection of raw physical particle sensor logs.
> 
> The evaluation metrics on this benchmark reflect **model consistency with the empirical joint statistical distributions** published in the USDA final report, **not** prediction accuracy on individual physically caught embers.

### 2.4 Reconciliation of Particle Counts (59,820 vs 60,370 — 550 Particle Difference)
Our repository records two numbers that must be rigorously reconciled:
- **Historical metadata entry in `source_registry.json`:** 9,249 vegetative + 50,571 structural = **59,820 particles**.
- **Table-by-table sum from Appendix E of the official report:** 9,249 vegetative + 51,121 structural = **60,370 particles**.

**Exact Mathematical Origin of the 550 Difference:**
1. **Structural Components (Tables 4 through 11, 24 test runs):**
   $$\sum N_{\text{comp}} = 351 + 776 + 300 + 777 + 742 + 252 + 1595 + 1435 + 1442 + 1495 + 1400 + 1478 + 940 + 1400 + 1744 + 743 + 1401 + 1401 + 623 + 1401 + 1455 + 1001 + 1430 + 1390 = \mathbf{26,972}$$
2. **Structural Wall/Roof Assemblies (Tables 12 through 21, 24 test runs):**
   $$\sum N_{\text{assembly}} = 537 + 891 + 915 + 497 + 814 + 1109 + 690 + 1104 + 1035 + 493 + 1070 + 994 + 516 + 860 + 1000 + 381 + 776 + 1131 + 559 + 1004 + 1002 + 540 + 584 + 1213 + 452 + 870 + 1015 + 364 + 867 + 866 = \mathbf{24,149}$$
3. **True Structural Total:** $26,972 + 24,149 = \mathbf{51,121}$ particles.
4. **Discrepancy Explanation:** In the preliminary `source_registry.json`, the sub-assemblies were inadvertently labeled as `structural_components: 24,149` (the assembly sum) and `structural_wall_roof_assemblies: 26,422` (a preliminary rounded figure). The difference between the actual component sum ($26,972$) and the preliminary figure ($26,422$) is exactly:
   $$26,972 - 26,422 = \mathbf{550} \text{ particles}$$
The table-by-table sum of **51,121 structural particles (60,370 total)** is the authoritative, verified count from Appendix E.

To enable individual-particle benchmarking without fabricating arbitrary trajectory coordinates, we implemented `src/save_the_earth/data/usda_particles.py`:
- Reconstructs $N=60,370$ individual firebrand records matching the exact measured joint multivariate distribution via a Gaussian copula with log-normal marginals.
- Enforces physical positivity ($m > 0$, $A > 0$, $d > 0$) and physical wind-tunnel boundaries ($d \le 35.0$ m).
- Integrates strictly with `data/manifests/split_v1.csv` for group-disjoint train (46,078 particles), validation (5,451 particles), and test (8,841 particles) partitions.

---

## 3. Priority 2 — UK EIDC Sycan Marsh Oregon FDS Fire Simulations

### 3.1 Dataset Identification & Acquisition
- **Citation:** UK Environmental Information Data Centre (EIDC), DOI: `10.5285/f9180989-6528-404e-8110-e3dcc205a6a1`.
- **Title:** *"Simulating fuel consumption and fire behaviour in a dry conifer forest: Advancing fuels FDS model outputs"*.
- **Location:** Sycan Marsh Preserve, Oregon, USA ($40 \times 40$ m plot).
- **Files Acquired:**
  - Download package: `data/raw/eidc_fds_oregon.zip` (52 KB)
  - Documentation: `advancing_fuels_fds_outputs.docx`, `ro-crate-metadata.json`
  - Output files: `whole_plot.out` (206 KB), `whole_plot.smv` (37 KB)
  - Particle binary: `whole_plot_1.prt5` (80.2 MB available on server)

### 3.2 Scientific Ineligibility Audit
The user request instructed investigating whether this dataset contains firebrand trajectory information for independent validation.

Upon inspecting `whole_plot.out` and the technical documentation:
```
FDS Version: 6.7.1
Simulation Title: Advancing Fuels - Whole Plot
Lagrangian Particle Classes:
  1: grass
  2: shrub
  3: wood
  4: leaf
```
**Scientific Findings:**
1. **Stationary Subgrid Vegetation:** In this FDS simulation, the Lagrangian particles are fixed boundary elements representing vegetation bulk density, surface area-to-volume ratio, and moisture derived from terrestrial LiDAR scans.
2. **Combustion & Mass Loss Role:** The particles undergo pyrolytic decomposition and convective heat transfer at stationary spatial locations $(x, y, z)$. They do not experience aerodynamic lofting, free-flight drag, or atmospheric advection.
3. **Formal Classification:** This dataset is classified as `unsuitable_stationary_vegetation_particles` for airborne firebrand trajectory and spotting validation. It is preserved in the provenance registry to prevent duplicate re-investigation.

---

## 4. Priority 3 — OpenPTV 2-Camera 3D Firebrand Tracking

### 4.1 Reconstructed Trajectory Characteristics
- **Repository:** `https://github.com/OpenPTV/2_cam_firebrands.git` (Commit: `2f45dcf`)
- **Institution:** University of Maryland / NIST Fire Research Division
- **Method:** 3D Particle Tracking Velocimetry (3D-PTV) using stereoscopic cameras viewing a small downstream observation window.
- **Pipeline Implementation:** Implemented in [`src/save_the_earth/data/openptv.py`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/src/save_the_earth/data/openptv.py) and inspected in [`src/save_the_earth/evaluation/openptv_validation.py`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/src/save_the_earth/evaluation/openptv_validation.py).
- **Observed Characteristics:**
  - 247 3D particle detections across frames 303 to 308.
  - 68 reciprocal-linked multi-frame tracks.
  - Reconstructed spatial coordinates are in **millimeters** in `res/rt_is.*`; converted strictly to SI meters in our ingestion pipeline.
  - Spatial displacement per frame: $\overline{\Delta x} \approx 4.2$ mm/frame, $\overline{\Delta y} \approx -0.4$ mm/frame, $\overline{\Delta z} \approx 0.6$ mm/frame.

### 4.2 Limitations & Retraction of Unconfirmed Temporal Assumptions
> [!CAUTION]
> **Retraction of Frame-Rate Assumption:**  
> The camera frame rate (FPS) and time-step ($\Delta t$) are **unconfirmed and not documented** in the public `OpenPTV/2_cam_firebrands` repository files or calibration parameters (`parameters_Run1.yaml`). Asserting an assumed 1,000 FPS is speculative.
> 
> Without verified timestamps, **temporal velocity vectors ($m/s$) and physical acceleration cannot be claimed as validated independent benchmarks.**

1. **Unconfirmed Temporal Base:** Physical velocities cannot be calculated without authoritative camera timing records from the experimental investigators.
2. **Short Spatial Window:** The observation volume is limited to a small near-field window ($10.9 \times 8.4 \times 5.7$ cm); tracked embers traverse only 1–3 cm before exiting the camera view.
3. **Missing Particle Mass & Area:** Particle dry masses ($m$) and aerodynamic areas ($A$) were not recorded for specific PTV track IDs.
4. **No Deposition:** Embers fly through the viewing window without landing. OpenPTV provides near-field geometric particle displacements, not landing or deposition measurements.

---

## 5. Priority 4 & 5 — University and National Laboratory Sources

### 5.1 CSIRO Untethered Firebrands (DOI: 10.1016/j.jaecs.2026.100520)
- **Title:** *"Reconstructing the trajectories and combustion behaviour of individual untethered burning firebrands in a vertical wind tunnel"*, Elsevier (2026).
- **Status:** `author_request`.
- **Audit:** The article describes full 3D stereoscopic trajectory reconstructions and thermal shrinkage of untethered barks and twigs. The raw time-series coordinate tables are not hosted on open repositories and require author data transfer.

### 5.2 NIST Emberometer
- **Apparatus:** Dual-camera high-speed telecentric system at the NIST National Fire Research Laboratory.
- **Status:** `author_request`.
- **Audit:** Public documentation covers camera calibration and particle segmentation. Raw numerical particle trajectories require inter-laboratory collaboration agreements.

---

## 6. Machine-Readable Provenance Registry

All investigated sources, DOIs, URLs, checksums, access statuses, and limitations are codified in:
👉 [`data/manifests/external_validation_sources.json`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/data/manifests/external_validation_sources.json)

```json
{
  "usda_rds_2020_0035_vegetative": "blocked_by_host_waf",
  "usda_rds_2020_0034_structural": "blocked_by_host_waf",
  "uk_eidc_fds_oregon": "unsuitable_stationary_vegetation_particles",
  "openptv_2_cam_firebrands": "downloaded_and_verified",
  "csiro_2026_vertical_wind_tunnel": "author_request",
  "nist_emberometer": "author_request"
}
```

---

## 7. Scientific Data Integrity Guarantees

In accordance with academic rigor:
1. **No Invented Coordinates:** We do not synthesize artificial 3D coordinates for the USDA experiments, where only landing pan locations were measured.
2. **No Data Snooping / Leakage:** Partitioning into train, validation, and test sets is strictly group-disjoint by fuel family. Test metrics are evaluated exclusively on unseen fuels.
3. **Transparent Reporting:** When models fail or exhibit large errors outside their training domain (e.g. Model C on wind tunnel geometry), the discrepancy is reported openly and explained physically.

