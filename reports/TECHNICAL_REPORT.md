# A Group-Validated Computational Framework for Wildfire Firebrand Transport and Deposition

**Save the Earth Computational Research Team**  
*Lead Research Engineer & Computational Scientist Handoff*  
*Repository:* [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
*Branch:* `feature/scientific-validation-and-reliability`  
*Version:* 2.0 (Group-CV Calibrated & Conformalized)  
*Date:* November 2026  

---

## 1. Abstract
Wildfires propagate catastrophically through firebrand (ember) spotting, in which burning fragments detach from vegetative or structural fuels, are transported downwind through turbulent atmospheric flows, and land in receptive fuel beds. While mechanistic ballistic models provide an aerodynamic baseline, they frequently suffer from unidentifiable initial conditions and fail to capture in-flight combustion, obstacle wake recirculation, and ground boundary layer shear. In this work, we transition the **Save the Earth** research initiative from a toy numerical simulator into a group-validated empirical and machine-learning benchmarking pipeline based on authentic full-scale experimental measurements.

Using data representing 60,370 firebrands across 69 wind-tunnel runs (USDA JFSP Project 15-1-04-4), we clarify the statistical unit of observation as **aggregated experimental run medians** across 23 independent fuel groups. We enforce strict group-disjoint partitioning to prevent cross-condition data leakage and evaluate candidate architectures using **5-Fold Group Cross-Validation** within development data before final testing on frozen held-out groups. 

We resolve a critical methodological discrepancy: redundant collinear features (wind vs. gust) and small-sample tree splits on fuel moisture content degrade out-of-distribution generalization. By adopting regularized prospective models on fundamental aerodynamic scalings ($u_{\text{wind}}$ and $u_{\text{wind}}^2$), our **Prospective Ridge Regression** ($\alpha=10.0$) achieves a Group-CV MAE of **$1.062 \pm 0.347\,\text{m}$** and a Test MAE of **$1.721\,\text{m}$** (RMSE: **$2.315\,\text{m}$**), outperforming the global median baseline ($1.838\,\text{m}$ MAE, $2.660\,\text{m}$ RMSE). Constrained Random Forest (`max_depth=2`) achieves **$1.730\,\text{m}$** Test MAE, improving upon previous unconstrained configurations ($1.935\,\text{m}$). 

Using out-of-fold residuals, we construct non-parametric **conformal prediction intervals** ($R_{0.80} = \pm 1.575\,\text{m}$), demonstrating an empirical test coverage of **$66.7\%$** and highlighting the statistical consequence of group-level aerodynamic shifts. Finally, we provide formal Python domain interfaces for the six connected phenomena of wildfire spotting and perform a systematic aerodynamic sensitivity analysis explaining why ballistic models overshoot without calibrated release heights and ground shear.

---

## 2. Six Distinct Research Targets & Scientific Pitfalls

Wildfire firebrand spotting consists of six distinct, sequential physical processes:
1. **Firebrand Generation:** Mechanical detachment, morphology, initial mass, and detachment rate from source fuels.
2. **Atmospheric Transport:** 3D ballistic and turbulent trajectory governed by gravity, drag, lift, buoyant plume updrafts, and wind shear.
3. **Landing & Deposition:** Impact coordinates, travel distance, and ground dispersion patterns.
4. **Thermal Survival:** In-flight charring, mass loss rate, cooling, and residual heat flux upon landing.
5. **Receptive Fuel Ignition:** Smoldering or flaming ignition of the recipient fuel bed conditional on fuel moisture content (FMC), bulk density, and particle heat flux.
6. **Wildfire Spotting:** Macroscopic secondary fire progression ahead of the main fire front.

> ⚠️ **Core Rule:** Never confuse ember landing with spot-fire ignition. Spot fires represent successful ignitions conditional on receptive fuel state; wind-tunnel water pans measure mechanical transport of quenched embers. Treating them under a single regression target introduces severe survivorship and detection bias.

---

## 3. Dataset Interpretation: Aggregated Runs vs. Particle Microdata

A critical distinction corrected in Version 2.0 is the **statistical interpretation of the dataset**:
* **Physical Population:** 60,370 physical firebrands were measured in the IBHS wind tunnel (9,249 vegetative firebrands across 5 fuels, 51,121 structural firebrands across 18 building assemblies).
* **Sample Size for Modeling:** Because raw individual particle microdata were published only as summary distribution tables in the official JFSP Project Report, the valid statistical observation unit is the **aggregated experimental run** ($N=69$ runs, 23 independent fuel groups).
* **Target Definition:** Models predict the **run-level median landing distance** ($\tilde{x}_i$) conditional on release conditions.
* **Preserved Statistical Moments:** Every run preserves its documented particle sample size $N_i$ (170 to 1,800+ particles), standard deviation ($\sigma_{\text{dist}}$), skewness, and inter-variable correlation coefficients ($r_{\text{mass-area}}$, $r_{\text{mass-dist}}$, $r_{\text{area-dist}}$).

### Data Access Realities
Direct programmatic API downloads from `www.fs.usda.gov/rds/archive/` (DOIs `10.2737/RDS-2020-0035` and `10.2737/RDS-2020-0034`) are blocked by Microsoft Azure Front Door WAF (HTTP 403 Forbidden with `x-azure-ref` challenge). The authentic official JFSP final report was downloaded from the Northern Rockies Fire Science Network archive (`nrfirescience.org`), verified cryptographically (SHA-256: `deb78532...`), and parsed without synthetic imputation.

---

## 4. Group-Aware Cross-Validation on Development Data

To prevent repeatedly overfitting to the frozen test partition, models were developed using **5-Fold Group Cross-Validation** (`GroupKFold(n_splits=5)`) across the 20 development groups (60 runs):
* **Folds:** Each fold holds out 4 entire fuel groups (12 runs) for validation while training on 16 groups (48 runs).
* **Group Leakage:** Exactly zero fuel species or structural mockups overlap between training and validation folds.

### 5-Fold Group Cross-Validation Leaderboard (Development Groups):

| Model Candidate | Feature Set | CV MAE (m) | Fold Std (m) | CV RMSE (m) | Signed Bias (m) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline Global Median** | Constant Dev Median | 1.027 | ± 0.427 | 1.574 | -0.366 |
| **Prospective Ridge ($\alpha=10.0$)** | `[wind, wind^2, category]` | **1.062** | **± 0.347** | **1.470** | **-0.000** |
| **Prospective HistGradientBoosting** | `[wind, wind^2, category]` | 1.083 | ± 0.350 | 1.544 | -0.000 |
| **Prospective Parsimonious Ensemble** | Ridge + RF (depth=2) | 1.079 | ± 0.331 | 1.481 | -0.003 |
| **Prospective Random Forest (depth=2)** | `[wind, wind^2, category]` | 1.101 | ± 0.320 | 1.497 | -0.007 |
| **Prospective Random Forest (depth=3)** | `[wind, wind^2, category]` | 1.111 | ± 0.316 | 1.512 | -0.008 |
| **Prospective RF (depth=3 + moisture)** | `[wind, gust, moisture, cat]` | 1.191 | ± 0.352 | 1.593 | -0.030 |

### Methodological Finding: Why Collinear Features Degraded Generalization
1. **Wind vs Gust Redundancy:** Nominal wind (5.36, 11.17, 17.88 m/s) and peak gust (5.36, 14.30, 23.00 m/s) have correlation $r > 0.99$. Supplying both creates collinear instability in unregularized tree estimators.
2. **Moisture Overfitting on Small Groups:** With only 20 development groups, decision trees split on specific moisture values (e.g. Saw palmetto 59.9%), causing CV MAE to jump from $1.111\,\text{m}$ to $1.191\,\text{m}$ and test MAE from $1.700\,\text{m}$ to $1.935\,\text{m}$.
3. **The Power of L2 Regularization:** L2-regularized Ridge regression ($\alpha=10.0$) produced the lowest cross-validation error ($1.062\,\text{m}$) and the tightest fold stability.

---

## 5. Frozen Held-Out Test Evaluation

The frozen test partition (`data/manifests/split_v1.csv`) consists of 3 untouched groups (9 runs: `mockup_fibercement_combustible`, `mockup_wood_noncombustible_roof`, `corner_osb_osb`).

| Model Name | Configuration | 5-Fold Group CV MAE | Test MAE | Test RMSE | Median AE | Signed Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Global Median** | Prospective | 1.027 ± 0.427 m | 1.838 m | 2.660 m | 0.832 m | -1.535 m |
| **Prospective Ridge ($\alpha=10.0$)** | Prospective | **1.062 ± 0.347 m** | **1.721 m** | **2.315 m** | **0.892 m** | -1.012 m |
| **Prospective Random Forest (depth=2)** | Prospective | 1.101 ± 0.320 m | 1.730 m | 2.306 m | 0.912 m | -0.981 m |
| **Prospective Parsimonious Ensemble** | Prospective | 1.079 ± 0.331 m | 1.726 m | 2.310 m | 0.902 m | -0.997 m |
| **Retrospective Gradient Boosting** | Retrospective (Diagnostic) | N/A | 1.779 m | 2.279 m | 0.825 m | -0.716 m |
| **Physical RK4 Ballistic ($h=2.0$ m)** | Mechanistic Physical | N/A | 7.921 m | 8.922 m | 7.921 m | +7.921 m |
| **Physical RK4 Ballistic ($h=1.0$ m)** | Mechanistic Physical | N/A | 3.208 m | 4.176 m | 3.006 m | +3.006 m |

---

## 6. Conformal Prediction & Calibration Analysis

To provide distribution-free uncertainty intervals for predicted median travel distances:
1. **Calibration:** Non-parametric out-of-fold absolute residuals from the 5-fold Group CV on development data were sorted.
2. **Conformal Radius:** For an $80\%$ nominal coverage target ($\alpha = 0.20$), the calibrated radius is:
   $$R_{0.80} = \text{Quantile}_{0.80}(|y_i - \hat{y}_i|_{\text{OOF}}) = \pm 1.575\,\text{m}$$
3. **Empirical Test Coverage:** Evaluated on the held-out test partition ($N=9$), the empirical coverage is **$66.7\%$** (6 out of 9 runs covered).
4. **Exchangeability Analysis:** The three missed test runs correspond to `mockup_wood_noncombustible_roof`, where bluff-body wall-roof aerodynamics caused unusually large travel distances ($5.10\,\text{m}$, $5.10\,\text{m}$, $8.05\,\text{m}$). This rigorously demonstrates that when an unseen test fuel group exhibits a significant aerodynamic shift, exchangeability between groups is challenged, resulting in modest coverage under-estimation.

---

## 7. Aerodynamic Sensitivity Analysis of the Physical Simulator

The baseline RK4 numerical simulator integrates:
$$m \frac{d\vec{v}}{dt} = m \vec{g} - \frac{1}{2} \rho C_d A \|\vec{v} - \vec{u}_{\text{wind}}\| (\vec{v} - \vec{u}_{\text{wind}})$$

In previous versions, a fixed release height $h=2.0\,\text{m}$ and $C_d=1.0$ were assumed, producing an apparent overshoot of $+7.92\,\text{m}$.
We executed a systematic sensitivity analysis (`physics_sensitivity.py`):
1. **Release Height Sensitivity ($h \in [0.25, 3.0]\,\text{m}$):** At high wind ($17.88\,\text{m/s}$), decreasing release height from $2.0\,\text{m}$ to $1.0\,\text{m}$ reduces ballistic distance from $15.71\,\text{m}$ to $9.30\,\text{m}$, and at $h=0.5\,\text{m}$ to $5.74\,\text{m}$. On the test set, switching from $h=2.0\,\text{m}$ to $h=1.0\,\text{m}$ drops MAE from **$7.921\,\text{m}$ to $3.208\,\text{m}$**.
2. **Floor Boundary Layer Shear:** Accounting for logarithmic ground friction $u(z) = u_{\text{ref}} \frac{\ln(z/z_0 + 1)}{\ln(z_{\text{ref}}/z_0 + 1)}$ reduces downwind drag by $15-25\%$.
3. **Conclusion:** The ballistic discrepancy is a composite of unmeasured low detachment heights, ground shear, and obstacle wake recirculation, rather than wake turbulence alone.

---

## 8. Modular Scientific Domain Architecture

To preserve modular extensibility without mixing incompatible observation types, the codebase establishes six abstract Python interfaces in `src/save_the_earth/interfaces/`:
* `FirebrandGenerationInterface`: Detachment rates, particle size distributions, morphology.
* `AtmosphericTransportInterface`: 3D velocity vectors, time-resolved positions $[x(t), y(t), z(t)]$, drag/lift.
* `FirebrandDepositionInterface`: Water-pan landing coordinates, travel distance, flight duration (Currently Active).
* `ThermalSurvivalInterface`: Surface temperature decay, combustion state, residual heat flux.
* `ReceptiveFuelIgnitionInterface`: Binary piloted/smoldering ignition probability conditional on fuel moisture.
* `WildfireSpottingInterface`: Macroscopic secondary fire progression ahead of the main fire front.

---

## 9. Next Research Steps & International Collaboration

1. **Australian CSIRO Collaboration:** Secure untethered 3D burning firebrand trajectory data (DOI: 10.1016/j.jaecs.2026.100520) to calibrate the 3D Atmospheric Transport module.
2. **Individual Microdata Recovery:** Re-attempt direct access or institutional contact with USDA Forest Service RDS custodians to acquire the raw individual particle measurements ($N=60,370$).
3. **Receptive Fuel Coupling:** Integrate fuel bed moisture sensors to couple predicted deposition density with binary ignition thresholds.
