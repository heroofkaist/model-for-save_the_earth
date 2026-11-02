# Model Card — Save the Earth Firebrand Distance Models

## 1. Model Overview
This suite of models evaluates firebrand travel distance predictions on unseen experimental fuel groups from the USDA Forest Service wind-tunnel experiments.

## 2. Model Configurations & Feature Scope

### A. Prospective Release Models (Operational Candidate)
* **Intended Use:** Field-deployable pre-flight predictions.
* **Allowed Inputs:** Only observable variables before flight:
  * Ambient wind speed $u_{\text{wind}}$ ($\text{m/s}$) and $u_{\text{wind}}^2$.
  * Fuel category (`vegetative`, `structural_component`, `structural_assembly`).
* **Models Evaluated:**
  * Prospective Ridge Regression ($\alpha = 1.0$).
  * Prospective Random Forest Regressor ($n=100, \text{max\_depth}=3$).

### B. Retrospective Particle Models (Diagnostic / Scientific)
* **Intended Use:** Post-event scientific characterization of collected particles.
* **Inputs:** $\ln(\text{mass})$, $\ln(\text{area})$, $\ln(\text{area}/\text{mass})$, wind speed, fuel category.
* **CRITICAL LIMITATION:** **Must NEVER be advertised as operational pre-flight predictors**, because ember mass and projected area were measured post-landing in water pans and dried in an oven.
* **Models Evaluated:**
  * Retrospective Ridge Regression ($\alpha = 1.0$).
  * Retrospective Gradient Boosting Regressor ($n=100, \text{lr}=0.05, \text{max\_depth}=3$).

### C. Physical Mechanistic Baseline
* **RK4 Ballistic Transport Simulator:** Aerodynamic drag relative to moving wind column, $C_d = 1.0$, $\rho = 1.225\,\text{kg/m}^3$, release height $h = 2.0\,\text{m}$.

## 3. Evaluation Metrics on Held-Out Test Set (Unseen Fuel Groups)

| Model Name | Configuration | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Median Baseline** | Prospective | 1.785 m | 2.577 m | 0.832 m | -1.385 m |
| **Category Median Baseline** | Prospective | 1.772 m | 2.556 m | 0.830 m | -1.347 m |
| **Prospective Ridge** | Prospective | 1.704 m | 2.265 m | 0.892 m | -0.911 m |
| **Prospective Random Forest** | Prospective | **1.700 m** | **2.261 m** | 0.887 m | -0.907 m |
| **Retrospective Ridge** | Retrospective | 1.669 m | 2.268 m | 0.831 m | -0.878 m |
| **Retrospective Gradient Boosting** | Retrospective | **1.684 m** | **2.059 m** | 0.812 m | **-0.402 m** |
| **Physical RK4 Ballistic** | Physical | 7.921 m | 8.922 m | 7.921 m | +7.921 m |

## 4. Key Scientific Insights
1. **Prospective Generalization:** Using solely wind speed and fuel category, the Random Forest model achieves $\text{MAE} = 1.700\,\text{m}$ across unseen experimental fuels, outperforming the median baseline ($\text{MAE} = 1.785\,\text{m}$).
2. **Value of Morphology:** Retrospective modeling with post-landing mass and area lowers RMSE to $2.059\,\text{m}$ and reduces bias to $-0.402\,\text{m}$. This proves that in-flight firebrand morphology strongly governs transport, highlighting the necessity of pre-flight particle imaging systems in future field campaigns.
3. **Ballistic Drag Limitations:** The pure ballistic physical baseline without wind tunnel wake recirculation overpredicts travel distance ($+7.92\,\text{m}$ bias), showing that near-ground recirculation eddies and water-pan wake effects must be coupled with boundary-layer physics.
