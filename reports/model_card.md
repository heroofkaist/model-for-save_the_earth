# Model Card — Save the Earth Firebrand Distance Models (Version 2.0)

## 1. Model Overview
This suite of models evaluates firebrand travel distance predictions on unseen, group-disjoint experimental fuel categories from the USDA Forest Service / IBHS wind-tunnel experiments.
All models are trained on aggregated experimental runs ($N=69$ runs, 23 independent fuel groups, representing 60,370 collected firebrands).

---

## 2. Model Configurations & Feature Scope

### A. Prospective Release Models (Operational Candidate)
* **Intended Use:** Field-deployable pre-flight predictions.
* **Allowed Inputs:** Observable quantities available before release:
  * Ambient wind speed $u_{\text{wind}}$ ($\text{m/s}$) and $u_{\text{wind}}^2$.
  * Fuel category (`vegetative`, `structural_component`, `structural_assembly`).
* **Feature Selection Rational:** Group CV proved that redundant gust factors and noisy batch moisture descriptors induce tree overfitting on small sample sizes. A parsimonious aerodynamic feature set provides superior cross-group stability.
* **Models Evaluated:**
  * **Prospective Ridge Regression** ($\alpha = 10.0$): Regularized L2 linear model.
  * **Prospective Random Forest Regressor** ($n=100, \text{max\_depth}=2$): Constrained tree ensemble.
  * **Prospective Parsimonious Ensemble**: Arithmetic average of regularized Ridge and Random Forest.

### B. Retrospective Particle Models (Diagnostic / Scientific Only)
* **Intended Use:** Post-event scientific characterization of collected particles.
* **Inputs:** $\ln(\text{mass})$, $\ln(\text{area})$, $\ln(\text{area}/\text{mass})$, wind speed, wind speed², fuel category.
* **CRITICAL LIMITATION:** **Must NEVER be advertised as operational pre-flight predictors**, because ember mass and projected area were measured post-landing in water pans and dried in an oven.
* **Models Evaluated:**
  * Retrospective Gradient Boosting Regressor ($n=100, \text{lr}=0.05, \text{max\_depth}=3$).

### C. Physical Mechanistic Baseline (Illustrative Aerodynamic Baseline)
* **RK4 Ballistic Transport Simulator:** Aerodynamic drag relative to moving air column, $C_d = 1.0$, $\rho = 1.225\,\text{kg/m}^3$, release heights $h = 2.0\,\text{m}$ and $h = 1.0\,\text{m}$.
* **Scope Note:** Labelled as an *illustrative physical comparison*, not a calibrated experimental benchmark, due to unmeasured ember detachment heights, ground boundary layer shear, and bluff-body wake recirculation.

---

## 3. Evaluation Metrics on Held-Out Test Set (Unseen Fuel Groups)

| Model Name | Configuration | 5-Fold Group CV MAE | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Global Median** | Prospective | 1.027 ± 0.427 m | 1.838 m | 2.660 m | 0.832 m | -1.535 m |
| **Prospective Ridge** | Prospective | **1.062 ± 0.347 m** | **1.721 m** | **2.315 m** | **0.892 m** | -1.012 m |
| **Prospective Random Forest** | Prospective | 1.101 ± 0.320 m | 1.730 m | 2.306 m | 0.912 m | -0.981 m |
| **Prospective Parsimonious Ensemble** | Prospective | 1.079 ± 0.331 m | 1.726 m | 2.310 m | 0.902 m | -0.997 m |
| **Retrospective Gradient Boosting** | Retrospective | N/A | 1.779 m | 2.279 m | 0.825 m | -0.716 m |
| **Physical RK4 Ballistic ($h=2.0$ m)** | Physical | N/A | 7.921 m | 8.922 m | 7.921 m | +7.921 m |
| **Physical RK4 Ballistic ($h=1.0$ m)** | Physical | N/A | 3.208 m | 4.176 m | 3.006 m | +3.006 m |

---

## 4. Conformal Prediction & Uncertainty Quantification

* **Conformal Calibration:** Non-parametric out-of-fold residuals from 5-fold Group CV on 20 development groups.
* **Nominal Coverage:** 80% confidence level ($R_{0.80} = \pm 1.575$ m).
* **Empirical Test Coverage:** 66.7% (6 of 9 held-out test runs covered).
* **Exchangeability Limitation:** Missed test cases correspond to `mockup_wood_noncombustible_roof`, where bluff-body roof geometry created larger aerodynamic lofting than typical dev groups. Group-level shifts can violate exchangeability, requiring transparent uncertainty communication.

---

## 5. Domain Disclaimers
1. **Never Confuse Landing with Ignition:** This model estimates mechanical downwind deposition in water pans. It does not model fuel bed temperature, particle charring, or ignition probability.
2. **Never Confuse Pan Landing with Spotting Distance:** Wind-tunnel water pan distances (0.9 to 8.0 m) must not be scaled to mesoscale wildfire spotting distances (hundreds of meters to kilometers) without atmospheric boundary layer, convective plume, and ember generation modeling.
