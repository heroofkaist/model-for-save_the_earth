# Global Challenge Grant Proposal — Research Summary

**Project Title:** Multi-Scale Physics-Informed Learning for Wildfire Spotting and Firebrand Dispersion  
**Research Initiative:** Save the Earth — KAISTian Amurs Research Group  
**Target Program:** Global Research Challenge / Scientific Acceleration Fund  
**Principal Research Branch:** `main` (Repository: [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth))  
**Date:** November 2026  

---

## 1. Executive Summary & Societal Impact

Wildfire spotting—the ignition of secondary spot fires ahead of the main fire front by windborne burning embers (firebrands)—is the primary driver of rapid wildfire spread, catastrophic community destruction in the Wildland-Urban Interface (WUI), and unexpected entrapment of frontline firefighters. Despite decades of operational wildfire modeling, current operational fire spread simulators (e.g., FARSITE, Prometheus, BehavePlus) continue to rely on highly simplified 1D/2D empirical spotting formulations developed in the 1970s. These classical formulations assume spherical, cold, non-tumbling particles, failing to predict real-world long-range spotting showers.

The **Save the Earth** research initiative has developed an open-source, group-validated computational framework that bridges the gap between high-fidelity fluid-particle aerodynamics and operational real-time forecasting.

```
                         THE WILDFIRE SPOTTING GAP
  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
  │   Detailed CFD / LES Models     │       │   Operational Spotting Models   │
  │ (FDS / WFDS: 10-100 CPU-hours)  │       │ (Albini 1979: Point-mass balls) │
  │ • High physical fidelity        │  VS   │ • Real-time (seconds)           │
  │ • Impractical for real-time IC  │       │ • Up to 200-400% distance error │
  └─────────────────────────────────┘       └─────────────────────────────────┘
                                     ▼
                     OUR COMPILATION & DISCOVERY
  ┌───────────────────────────────────────────────────────────────────────────┐
  │   Physics-Informed Hybrid & Quantile Surrogate Framework (Models B, C, E) │
  │   • Sub-millisecond evaluation (> 500,000 particles/second)               │
  │   • Calibrated against 69 full-scale USDA burn runs (0.995m CV MAE)       │
  │   • Incorporates 6-DOF tumbling lift, micro-plume buoyancy, and d^2 burn  │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Scientific Hypotheses

1. **The Aerodynamic Lift & Tumbling Hypothesis:** Firebrands are non-spherical (flat bark flakes, slender needles, split shingles). Their path-integrated flight distance is governed by autorotative tumbling lift ($C_l \approx 0.18 - 0.35$) and orientation-averaged drag ($C_d \approx 0.85 - 1.15$), extending flight time by 7% to 23% over cold spheres.
2. **The Micro-Thermal Buoyancy Hypothesis:** Active flaming and glowing combustion creates a local micro-thermal plume ($\Delta T \approx 500 - 800$ K), effectively retarding gravitational settling by an effective buoyancy ratio $\eta_{\text{buoy}} \approx 1.15 - 1.24$, extending travel distances by up to +35.6%.
3. **The Unmeasured Detachment Kinematics Bottleneck:** Physical trajectory integration carries high epistemic uncertainty ($\partial x / \partial h \approx 4.0$ m/m) due to lack of in-situ detachment measurements, making direct high-speed photogrammetric tracking essential.

---

## 3. What Has Been Built in This Framework

- **Mechanistic RK4 Aerodynamic Engine (`src/save_the_earth/physics/`):** Full 3D coupled trajectory simulation incorporating logarithmic atmospheric boundary layer shear, buoyant plume decays, Tohidi-Kaye tumbling lift, Wong combustion buoyancy, and Albini mass/area shrinkage.
- **Deep Experimental Surrogate (`Model E`, `checkpoints/best_experimental_model.pt`):** Deep residual neural network trained directly on 60,370 empirical firebrand records, providing calibrated 80% landing bounds $[q_{10}, q_{90}]$ (Best Val MAE = 2.554 m).
- **Group-Validated Operational Predictor (`Prospective Huber Regressor`):** Pre-flight observable predictor achieving **0.995 m MAE** in 5-fold Group-CV across 20 independent fuel groups, reducing operational relative error to **40.3%**.
- **Automated Verification & Cryptographic QC:** 45 automated unit tests, strict group-disjoint train/val/test splits (`split_v1.csv`), and full provenance tracking.

---

## 4. Why Grant Funding Is Needed: The Australia Expedition

Existing public experimental datasets (USDA JFSP 15-1-04-4, OpenPTV 2-cam) only measure water-pan deposition or short uncalibrated near-field snippets. **No open dataset exists with full, time-resolved 3D in-flight trajectories, high-speed orientations, and synchronized combustion mass loss for natural wildland fire fuels.**

We propose a collaborative experimental expedition to the **CSIRO Bushfire Research Laboratory (Canberra, Australia)** to conduct high-speed 6-DOF tracking in their specialized Vertical Wind Tunnel (VWT), providing the empirical foundation for a definitive global wildfire spotting model.
