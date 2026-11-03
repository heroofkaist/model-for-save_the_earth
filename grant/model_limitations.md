# Model Limitations & Honest Technical Boundaries

**Project:** Save the Earth — Wildfire Firebrand Dispersion  
**Document:** Grant Application Limitations & Open Challenges  
**Date:** November 2026  

---

## 1. Executive Statement of Humility & Scientific Integrity

A central requirement of high-impact scientific proposals is an honest, transparent appraisal of existing model limitations. We formally reject unsupported claims of "full operational readiness" or "experimentally validated 3D trajectories." Below are the four principal technical boundaries of our current models.

---

## 2. Boundary 1: The Post-Landing Feature Dependency

### Description:
Our diagnostic models (Retrospective Gradient Boosting) and neural surrogates (Models B and C) utilize post-deposition particle measurements:
- Dry particle mass: $m_{\text{dry}}$
- 2D projected surface area: $A_{\text{proj}}$
- Drag coefficient: $C_d$

### Operational Limitation:
In an active wildland fire, incident commanders and tactical fire behavior analysts **cannot** observe the mass or projected area of an ember prior to its detachment from the canopy. These features can only be measured *after* the firebrand has landed in a water pan or ground collector.

### Mitigation in Current Framework:
We strictly isolated **Prospective Operational Models** (Prospective Huber, Prospective Ridge, Parsimonious Ensemble) which rely exclusively on quantities observable before flight:
- Reference wind speed: $u_{\text{ref}}$ and $u_{\text{ref}}^2$
- Fuel category: vegetative vs. structural
- Moisture content: $M_c$

---

## 3. Boundary 2: Lack of Time-Resolved In-Flight 3D Experimental Benchmarks

### Description:
The primary physical datasets available in public literature (USDA JFSP 15-1-04-4, Hedayati et al. 2018) measure **landing position in water pans**, not in-flight 3D time-series coordinates.

While the OpenPTV repository (`OpenPTV/2_cam_firebrands`) contains reconstructed 3D stereoscopic tracks, an audit of the repository revealed that **neither camera frame rate nor time-step $\Delta t$ is documented in the calibration parameters.**

### Consequence:
We can validate spatial track geometry ($\overline{\Delta x} \approx 4.2$ mm/frame), but we **cannot** rigorously validate time-resolved velocity vectors ($\mathbf{v} = d\mathbf{x}/dt$) against OpenPTV without verified timestamps.

---

## 4. Boundary 3: Domain Shift and Quantile Calibration on Unseen Fuel Groups

### Description:
When evaluating our deep experimental surrogate (`Model E`, `ExperimentalDistanceNet`) on the held-out test split, the empirical coverage of the nominal 80% quantile interval $[q_{10}, q_{90}]$ drops from **75.4% on validation** to **60.1% on the held-out test set** (recovering to 63.4% with Conformalized Quantile Regression).

### Physical Explanation:
The test set consists strictly of unseen structural assemblies (`corner_cedar_osb`, `recycled_shake_roof`, `mockup_fibercement_combustible`). These assemblies release heavier, larger embers that shatter or catch higher winds, shifting the landing distribution outward relative to standard vegetative fuels.

### Consequence:
Static quantile networks suffer from severe domain shift when applied across heterogeneous fuel types. The uncertainty calibration cannot be described as successful: achieving only 63.4% coverage against a nominal 80% confidence level indicates substantial under-coverage. This proves that future models must incorporate **fuel-specific hierarchical priors and non-exchangeable conformal methods** rather than a single homogeneous network.

---

## 5. Boundary 4: Clarification on Error Baselines vs. Theoretical Limits

### Description:
In earlier exploratory analyses, a reference error of ~4.8 meters was occasionally cited. We explicitly clarify that **4.8 m is an unconditional sample median baseline, NOT a fundamental or irreducible MAE lower bound**.

### Operational Reality:
When conditioned on pre-flight wind speed and fuel classification, our prospective models achieve a 5-fold cross-validation MAE of **0.995 m** (and a held-out test MAE of **1.763 m**). The remaining error is governed by unmeasured initial conditions (detachment height $h$, ejection angle) and chaotic 6-DOF flutter, which require experimental tracking rather than purely statistical post-processing.

---

## 6. Boundary 5: Lumped Phenomenological Formulations

### Description:
In `src/save_the_earth/physics/aerodynamics.py`:
1. **Drag and Lift:** Tohidi & Kaye's aerodynamic curves are discretized into three slenderness bins (`compute_tohidi_drag_lift`). While literature-informed, this does not capture the full continuous spectrum of natural wood turbulence.
2. **Combustion Buoyancy:** The formulation $g_{\text{eff}} = g / \eta_{\text{buoy}}$ is a lumped parameterization that retards settling velocity, but does not resolve the Navier-Stokes boundary-layer temperature gradients or micro-convective vortices around the burning particle.
3. **Mass Loss:** The Albini $d^2$ mass loss assumes isotropic geometric shrinkage and constant particle density, which omits porous internal charring, ash spallation, and hollow tube formation in eucalyptus stringybark.

---

## 7. Summary: The Experimental Necessity

These limitations cannot be resolved by writing more code or generating more synthetic neural network weights. **They require empirical in-flight 3D measurements of burning embers under controlled vertical and horizontal flow fields.** This directly motivates our prospective research expedition to Australia (seeking consultation with bushfire researchers at CSIRO, UNSW Canberra, and the University of Melbourne) to bridge our computational framework with direct experimental methods and realistic fuel morphology data.

