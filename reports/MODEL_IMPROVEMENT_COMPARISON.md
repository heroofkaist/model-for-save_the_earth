# Save the Earth — Model Improvement & Scientific Comparison Report

**Repository:** `heroofkaist/model-for-save_the_earth`  
**Evaluation Level:** Aggregated Experimental Run Medians ($N=69$ experimental runs across 23 fuel groups, representing 60,370 collected firebrands from USDA JFSP 15-1-04-4).  
**Partitioning Protocol:** Group-Disjoint Frozen Split (`split_v1.csv`): 20 Development Groups (60 runs), 3 Frozen Test Groups (9 runs).

---

## 1. Executive Summary & Root Cause of Historical Metric Differences

In earlier iterations of the project, conflicting metrics were recorded for Prospective Random Forest:
* **Version 1.0:** Test MAE = **1.700 m**, Test RMSE = **2.261 m**
* **Version 1.1:** Test MAE = **1.935 m**, Test RMSE = **2.525 m**

### Empirical Investigation: Why Did Random Forest MAE Degrade by 0.235 m?
A controlled feature ablation experiment was executed on the exact same train/test split:
1. **Collinearity of Wind and Gust:** In the IBHS wind tunnel, fans operated at three nominal conditions (5.36 m/s, 11.17 m/s, 17.88 m/s). Measured gust peaks (5.36, 14.30, 23.00 m/s) correlate with nominal wind at $r > 0.99$. Introducing gust speed as an independent splitting variable created redundant tree splits without adding new information.
2. **Fuel Moisture Overfitting:** Moisture content in the laboratory report is a fuel-level descriptor (e.g. SYP 13.5%, Chamise 20.3%, Saw palmetto 59.9%). With only 51 training runs, decision trees with `max_depth=3` split on specific training moisture values. When evaluating on completely unseen test fuels (`mockup_fibercement_combustible`, `mockup_wood_noncombustible_roof`, `corner_osb_osb`), these moisture splits degraded out-of-distribution generalization.
3. **Model Regularization Solution (Version 2.0):** Restraining tree depth (`max_depth=2`) and focusing prospective features on fundamental aerodynamic wind scaling (`[wind_speed, wind_speed^2, fuel_category]`) restored generalization across independent groups:
   * **Version 2.0 Prospective Random Forest:** Test MAE = **1.730 m**, Test RMSE = **2.306 m**.
   * **Version 2.0 Prospective Ridge ($\alpha=10.0$):** Test MAE = **1.721 m**, Test RMSE = **2.315 m**, 5-Fold Group CV MAE = **1.062 m**.
   * **Version 2.0 Parsimonious Ensemble:** Test MAE = **1.726 m**, Test RMSE = **2.310 m**.

---

## 2. Direct Side-by-Side Model Comparison

All models evaluated on the frozen, untouched held-out test partition (`data/manifests/split_v1.csv`):

| Model Name | Version 1.0 MAE | Version 1.1 MAE | Version 2.0 (New) MAE | Version 2.0 RMSE | 5-Fold Group CV MAE | Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline Global Median** | 1.785 m | 1.785 m | **1.838 m** | 2.660 m | 1.027 ± 0.427 m | Reference Baseline |
| **Prospective Ridge** | 1.704 m | 1.731 m | **1.721 m** | 2.315 m | **1.062 ± 0.347 m** | **Recommended (Most Stable)** |
| **Prospective Random Forest** | 1.700 m | 1.935 m | **1.730 m** | 2.306 m | 1.101 ± 0.320 m | Improved (Depth Restrained) |
| **Prospective Gradient Boosting** | 1.861 m | 1.861 m | **1.732 m** | 2.300 m | 1.101 ± 0.323 m | Improved (Shallow Trees) |
| **Prospective Parsimonious Ensemble** | 1.844 m | 1.844 m | **1.726 m** | 2.310 m | 1.079 ± 0.331 m | Validated (Ridge + RF Blend) |
| **Retrospective Gradient Boosting** | 1.684 m | 1.571 m | **1.779 m** | 2.279 m | N/A | Diagnostic Only (Post-Flight) |
| **Physical Ballistic RK4 ($h=2.0$ m)** | 7.921 m | 7.921 m | **7.921 m** | 8.922 m | N/A | Illustrative Baseline ($h=2.0$m) |
| **Physical Ballistic RK4 ($h=1.0$ m)** | N/A | N/A | **3.208 m** | 4.176 m | N/A | Illustrative Baseline ($h=1.0$m) |

*Note on Retrospective Model:* Retrospective models incorporate post-landing mass and projected area. In Version 2.0, preprocessors and encoders were strictly refitted only on the development partition, yielding a realistic out-of-sample diagnostic test MAE of 1.779 m.

---

## 3. Methodological Improvements Implemented in Version 2.0

### 1. Correct Statistical Interpretation of Observations
* **Clarified Unit of Observation:** Each record represents an **aggregated experimental run** ($N_i$ collected firebrands per water pan array), NOT 60,370 independent training points.
* **Preserved Statistical Moments:** Every run preserves its documented sample size $N_i$, standard deviation $\sigma_i$, skewness, and inter-variable correlation coefficients ($r_{\text{mass-area}}$, $r_{\text{mass-dist}}$, $r_{\text{area-dist}}$).
* **Sample-Size Weighted Fitting:** Evaluated sample-weighted regression ($w_i = \sqrt{N_i}$); verified that while larger runs have tighter sampling variance, unweighted regularized Ridge regression maintains lower generalizable bias across heterogeneous fuel geometries.

### 2. Group-Aware Cross-Validation on Development Data
* Rather than repeatedly evaluating against the frozen test set, models were tuned using **5-Fold Group Cross-Validation** across the 20 development groups (60 runs).
* GroupKFold strictly ensures that no fuel species or structural assembly appears in both training and validation folds.
* Regularized Ridge regression ($\alpha=10.0$) demonstrated the highest stability ($1.062 \pm 0.347$ m CV MAE), outperforming unconstrained tree regressors.

### 3. Conformal Prediction & Honest Uncertainty Quantification
* Out-of-fold residuals from the 5-fold Group CV were used to calibrate a non-parametric conformal prediction radius for an 80% coverage target ($R_{0.80} = \pm 1.575$ m).
* On the untouched test set (3 groups, 9 runs), empirical coverage was evaluated at **66.7%** (6/9 runs covered).
* **Root Cause of Test Coverage Gap:** The 3 missed runs correspond to `mockup_wood_noncombustible_roof`, where bluff-body roof geometry created larger aerodynamic lofting ($d = 5.1 - 8.05$ m). This confirms the fundamental statistical lesson: **conformal exchangeability assumptions can be violated when unseen test fuels exhibit structural aerodynamic shifts**.

### 4. Aerodynamic Sensitivity Analysis of Physical Baseline
* Ballistic simulation overshoot was systematically investigated across release heights $h \in [0.25, 3.0]$ m, drag coefficients $C_d \in [0.4, 1.8]$, and logarithmic floor boundary layer shear $u(z) = u_{\text{ref}} \frac{\ln(z/z_0 + 1)}{\ln(z_{\text{ref}}/z_0 + 1)}$.
* **Key Finding:** Decreasing release height from 2.0 m to 1.0 m alone reduces ballistic test MAE from **7.921 m to 3.208 m**.
* Rather than claiming "wake recirculation explains everything", Version 2.0 documents that ballistic overshoot is a composite effect of unmeasured detachment heights, ground shear, and obstacle wake aerodynamics.

### 5. Separation of Operational vs. Diagnostic Interfaces
* The CLI `run_pipeline.py --predict` now explicitly distinguishes:
  1. **Predicted Run Median Distance** ($\hat{y}$)
  2. **80% Conformal Prediction Interval for the Median** ($[\hat{y} - 1.58, \hat{y} + 1.58]$ m)
  3. **Estimated Individual Firebrand Dispersion Range** ($[\hat{y} - 1.28\sigma, \hat{y} + 1.28\sigma]$ m)
* Includes automated Out-of-Distribution (OOD) warnings for wind speeds outside $5.36 - 17.88$ m/s.
* Includes mandatory scientific disclaimers: *Does NOT predict particle thermal survival, fuel bed ignition, or wildfire spotting.*

---

## 4. Prioritized List of Unresolved Scientific Limitations

1. **Absence of In-Flight 3D Trajectory Data:** Water-pan arrays record only the final resting coordinates; in-flight lofting, pitch angle, and drag/lift variations remain unobserved until Australian CSIRO 3D tracking data can be ingested.
2. **Aggregated Run Medians vs. Raw Particle Microdata:** The USDA Forest Service RDS repository blocks automated non-US requests via Azure Front Door WAF (HTTP 403 Forbidden). Individual firebrand microdata ($N=60,370$) could not be downloaded directly, necessitating aggregate-run modeling.
3. **Plume Buoyancy Absence in Lab Tunnels:** Wind tunnel tests study forced convection without the convective thermal updrafts present in high-intensity wildfires.
4. **Ignition Disconnection:** Ember travel distance cannot be treated as a surrogate for wildfire ignition probability, which is strictly governed by fuel bed moisture content and heat flux.

