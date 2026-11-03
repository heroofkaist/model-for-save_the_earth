# Scientific Evidence & Validation Dossier

**Project:** Save the Earth — Wildfire Firebrand Dispersion  
**Applying Team:** KAIST Undergraduate / Graduate Research Team (4 Members)  
**Target Program:** KAIST Save the Earth Global Challenge  
**Date:** November 2026  

---

> **Evidence dossier for context, not an independent replication certificate.** Use [AI_CONTEXT.md](AI_CONTEXT.md) and [CLAIM_REGISTER.md](CLAIM_REGISTER.md) to distinguish original experimental-run aggregates, synthetic particle reconstruction, numerical ablations and unverified host activities. Never call the recorded 3D surrogate errors experimental accuracy.

## 1. Hierarchy of Scientific Evidence

To maintain complete scientific honesty and prevent overclaiming, our research framework rigorously separates four distinct tiers of evidence:

```
  LEVEL 4: Kinematic Track Geometry (OpenPTV 2-Cam)
  └─ Direct near-field displacement (Δx ~ 4.2 mm/frame); unconfirmed camera FPS prevents velocity validation
     ▲
  LEVEL 3: Physical Landing Deposition (USDA 69 Full-Scale Experimental Runs)
  └─ Ground-truth water-pan landing distances (Group-CV MAE = 0.995 m, Test MAE = 1.76 m)
     ▲
  LEVEL 2: Reconstructed Empirical Distribution (USDA 60,370 Particles)
  └─ Moment-matched Gaussian copula distribution testing surrogate distributional fidelity (NOT raw catch logs)
     ▲
  LEVEL 1: Numerical Surrogate Learning Capacity (10,000 In-Silico RK4 Simulations)
  └─ Proof-of-concept demonstrating deep neural approximation of non-linear ODE aerodynamics
```

---

## 2. Quantitative Model Performance

### 2.1 Group-Validated Operational Spotting Distance (69 Full-Scale Tests)
Evaluated across 20 development fuel groups (60 burn tests) under 5-Fold Group Cross-Validation, and tested on 3 frozen, untouched fuel groups (`corner_cedar_osb`, `recycled_shake_roof`, `mockup_fibercement_combustible`, 9 burn tests):

| Model Architecture | Model Category | 5-Fold Group CV MAE (m) | Test MAE (m) | Test RMSE (m) | Test Signed Bias (m) | Test MAPE (%) |
|---|---|---:|---:|---:|---:|---:|
| **Prospective Huber Regressor** | **Operational (Pre-Flight)** | **0.995** | **1.763** | **2.494** | -1.353 | **40.3%** |
| Prospective Ridge Regression | Operational (Pre-Flight) | 1.062 | 1.721 | 2.315 | -1.012 | 42.1% |
| Prospective Parsimonious Ensemble | Operational (Pre-Flight) | 1.079 | 1.726 | 2.310 | -0.997 | 41.8% |
| Prospective Random Forest (depth=2) | Operational (Pre-Flight) | 1.101 | 1.730 | 2.306 | -0.981 | 42.4% |
| Baseline Global Median | Constant Reference | 1.027 | 1.838 | 2.660 | -1.535 | 51.9% |
| Mechanistic RK4 (h=1.0 m) | Mechanistic Ballistic | N/A | 3.208 | 4.176 | +3.006 | > 100% |
| Mechanistic RK4 (h=2.0 m) | Mechanistic Ballistic | N/A | 7.921 | 8.922 | +7.921 | > 200% |

*Key Finding:* By using a robust Huber loss and sample-weighting by group size, our operational model predicts median shower landing distance within **0.995 meters** on cross-validation and **1.76 meters** on held-out test runs.

### 2.2 Deep Quantile Surrogate (Model E) & Uncertainty Calibration
- **Training Foundation:** Trained on 60,370 moment-matched Gaussian copula particle records reconstructed from the empirical multivariate moments in the USDA JFSP 15-1-04-4 final report.
- **Validation Accuracy:** Achieves a Best Validation MAE of **2.554 meters** on median distance prediction.
- **Uncertainty Calibration & Domain Shift:**
  - On the validation set, the nominal 80% pinball interval $[q_{10}, q_{90}]$ achieves **75.4% coverage**.
  - When transferred to the held-out test set (unseen structural mockups: cedar corner walls, shake roofs, fiber cement assemblies), raw test coverage drops to **60.1%**, recovering to **63.4%** under Conformalized Quantile Regression (CQR, Romano et al. 2019).
  - *Honest Assessment:* This under-calibration confirms that structural assemblies generate heavier, irregularly shaped firebrands that shift the landing distribution outward relative to standard vegetative fuels. A single homogeneous quantile network cannot generalize across heterogeneous fuel types without fuel-specific empirical priors.

---

## 3. Physical Sensitivity Analysis: The Limits of Synthetic Simulation

Using our mechanistic aerodynamics engine (`physics_sensitivity.py`), we evaluated local sensitivities (partial derivatives) and global variations around nominal wind-tunnel conditions ($u = 11.17$ m/s, $h = 1.22$ m, $C_d = 1.0$, $z_0 = 0.005$ m, $w_0 = 0$, $v_{0z} = 0$):

$$\text{Nominal Landing Distance } x_{\text{base}} = 8.169 \text{ meters}$$

### Local Sensitivities (Partial Derivatives):
- **Detachment Height:** $\frac{\partial x}{\partial h} = \mathbf{+4.015 \text{ m/m}}$
- **Drag Coefficient:** $\frac{\partial x}{\partial C_d} = \mathbf{+2.033 \text{ m/unit}}$
- **Surface Roughness:** $\frac{\partial x}{\partial z_0} = \mathbf{-66.055 \text{ m/m}}$ (ground boundary layer deceleration)
- **Burner Updraft:** $\frac{\partial x}{\partial w_0} = \mathbf{+1.970 \text{ m/(m/s)}}$
- **Initial Ejection:** $\frac{\partial x}{\partial v_{0z}} = \mathbf{+0.465 \text{ m/(m/s)}}$

### Global Impact across Physical Bounds:
- Varying detachment height across the specimen panel ($0.2$ m to $2.4$ m) shifts predicted landing distance from **3.17 m to 10.77 m** ($\Delta = 7.60$ m, a **$\pm 70\%$ relative shift**).
- *Scientific Conclusion:* Synthetic numerical integration has reached its limit. Without direct optical measurement of in-situ detachment heights and near-field flow fields, deterministic models carry irreducible epistemic uncertainty.

---

## 4. Computational Ablation Study: Investigating Physical Mechanism Contributions

We conducted a multi-factorial computational ablation study (`src/save_the_earth/physics/ablation_study.py`) across 3 particle morphologies (flat bark flake, cylindrical rod, compact wood chunk), 3 wind speeds (5.36, 11.17, 17.88 m/s), and 2 release heights (1.22 m, 2.0 m):

| Configuration | Aerodynamic Drag | Tumbling Lift ($C_l$) | Combustion Buoyancy ($\eta_{\text{buoy}}$) | Dynamic Mass/Area Loss ($\beta$) | Mean Distance Shift (%) | Max Distance Shift (%) | Mean Flight Time Shift (%) |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|
| **Config 1 (Baseline)** | Yes ($C_d=1.0$) | No | No | No | 0.0% (Ref) | 0.0% (Ref) | 0.0% (Ref) |
| **Config 2 (+ Tohidi)** | Yes ($C_d(\lambda)$) | Yes ($0.18-0.35$) | No | No | **+7.4%** | **+22.6%** | **+9.3%** |
| **Config 3 (+ Wong)** | Yes | Yes | Yes ($1.18$) | No | **+19.5%** | **+35.6%** | **+19.9%** |
| **Config 4 (+ Albini Full)** | Yes | Yes | Yes | Yes ($\beta = 0.02 \text{ s}^{-1}$) | **+19.7%** | **+36.0%** | **+20.2%** |

*Findings:*
1. **Tumbling Lift:** Autorotation and lift (parameterized from Tohidi & Kaye 2017) significantly extend flight distance for slender bark flakes (+22.6%).
2. **Combustion Micro-Buoyancy:** Active combustion micro-thermal plumes (parameterized from Wong et al. 2026) produce a major +19.5% shift in downwind transport by retarding vertical gravitational settling.
3. **Coupled Shrinkage:** Coupling area shrinkage with mass loss ($A(t) \propto m(t)^{2/3}$) maintains physical aerodynamic equilibrium without unphysical drag blow-up.

*Important Note:* This ablation study represents a **numerical exploration of physical hypotheses**, not an experimental validation. Direct laboratory validation of these mechanisms is the primary objective of our proposed Australian expedition.
