# KAIST Save the Earth Global Challenge — Research Summary

**Project Title:** Multi-Scale Physics-Informed Learning for Wildfire Spotting and Firebrand Dispersion  
**Applying Team:** KAIST Undergraduate / Graduate Research Team (4 Members)  
**Research Initiative:** Save the Earth — KAISTian Amurs Project  
**Target Program:** KAIST Save the Earth Global Challenge (Expedition Program)  
**Proposed Expedition:** Australia (Canberra, Sydney, Melbourne) — 21 Days (3 Weeks)  
**Budget Request:** $14,950 USD (strictly within the $15,000 program limit)  
**Primary Repository:** [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
**Date:** November 2026  

---

## 1. Executive Summary & Societal Urgency

Wildfire spotting—the ignition of secondary fires ahead of the main fire front by windborne burning embers (firebrands)—is the single most catastrophic driver of uncontrollable wildfire escalation. Firebrands breach defensive containment lines, ignite residential structures, and cause sudden entrapment of wildland firefighters. Despite decades of operational wildfire modeling, current operational fire spread simulators (e.g., FARSITE, Prometheus, BehavePlus) rely on empirical formulas developed in the 1970s that assume spherical, cold, non-tumbling particles, underestimating spotting distances by up to 200–400% in extreme conditions.

Over the past research campaign, our 4-member KAIST student team has independently engineered an open-source computational framework that bridges fluid-particle aerodynamics and machine learning:
1. Built a **3D coupled aerodynamic RK4 integration engine** incorporating atmospheric boundary-layer shear, convective plume decay, aspect-ratio dependent drag, tumbling lift, and combustion buoyancy.
2. Formulated and trained **prospective operational distance estimators** achieving **0.995 m MAE** in 5-fold Group Cross-Validation across 69 full-scale USDA wind-tunnel burn series.
3. Implemented **deep quantile neural networks (Model E)** to estimate probabilistic dispersion envelopes.

```
                         THE COMPUTATIONAL FRONTIER & ITS LIMIT
  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
  │   What We Have Built (Software) │       │   The Missing Physical Ground   │
  │ • Coupled 3D RK4 ODE Simulator  │  VS   │ • In-situ detachment height (h) │
  │ • Deep Quantile Network (Model E)│      │ • 6-DOF tumbling lift & roll    │
  │ • 0.995m CV MAE on USDA runs    │       │ • Eucalypt bark micro-buoyancy  │
  └─────────────────────────────────┘       └─────────────────────────────────┘
                                     ▼
                SENSITIVITY ANALYSIS: dx/dh = +4.015 m/m (±70% SHIFT)
   Further synthetic in-silico simulation alone cannot resolve this boundary.
                                     ▼
                     OUR EXPEDITION PROPOSAL ($14,950)
  A focused 3-week research expedition to Australia (CSIRO, UNSW, Uni Melbourne)
  to inspect specialized vertical wind tunnels, learn 6-DOF optical tracking methods,
  sample wildland eucalypt bark morphologies, and design a validated experimental blueprint.
```

---

## 2. Core Scientific Findings & Evidence Hierarchy

To ensure scientific honesty and prevent overclaiming, our repository strictly classifies four tiers of evidence:

1. **Level 1 (Numerical Surrogate Learning Capacity):** 10,000 synthetic trajectories simulated via our numerical RK4 solver. Confirms neural network capacity to approximate non-linear aerodynamic ODEs, but does not validate against nature.
2. **Level 2 (Joint Distribution Consistency):** 60,370 moment-matched Gaussian-copula particles reconstructed from the published joint multivariate moments of the USDA JFSP 15-1-04-4 final report. Evaluates model fidelity against empirical statistical distributions, not physical individual sensor logs.
3. **Level 3 (Operational Deposition Validation):** 69 full-scale wind-tunnel burn series from IBHS/USDA. Validates run-level median deposition distance against physical water-pan measurements (**0.995 m CV MAE, 1.76 m Test MAE**).
4. **Level 4 (Photogrammetric Spatial Kinematics):** OpenPTV 2-camera stereoscopic tracking. Reconstructs near-field spatial displacement geometry ($\overline{\Delta x} \approx 4.2$ mm/frame), but unconfirmed camera frame rates prevent dynamic velocity validation.

---

## 3. The Central Grant Argument: Why Travel to Australia Is Essential

Our mathematical sensitivity analysis proves that **further progress through synthetic modeling alone is fundamentally blocked**:
- Detachment height sensitivity is extreme: $\frac{\partial x}{\partial h} = \mathbf{+4.015 \text{ m/m}}$. Across a 4-foot panel ($h \in [0.2, 2.4]$ m), predicted landing distance varies from **3.17 m to 10.77 m** ($\pm 70\%$).
- Literature parameterizations of tumbling lift ($C_l \approx 0.18 - 0.35$) and combustion micro-buoyancy ($\eta_{\text{buoy}} \approx 1.18$) shift simulated distances by **+7.4% to +35.6%**, yet exist only as simplified approximations in current code.
- **Why Australia:** Native Australian eucalyptus forests (stringybark, ribbon bark) generate the most extreme spotting in the world (5–30+ km). Australian research institutions (CSIRO, UNSW Canberra, University of Melbourne) lead the world in vertical wind tunnel testing and bushfire dynamics.

By traveling to Australia for 3 weeks, our team will:
- Consult directly with leading wildland fire researchers on vertical wind tunnel and optical tracking methodologies.
- Inspect experimental apparatus (air conditioning, illumination, high-speed camera placement) to understand how untethered embers are tracked in practice.
- Conduct physical morphological field sampling of naturally shed eucalyptus stringybark and candlebark to replace assumed aspect ratios with measured Australian fuel distributions.
- Interview operational Fire Behavior Analysts (RFS/CFA) to ensure our models address real-world tactical warning needs.

---

## 4. Expedition Overview & Achievable Deliverables

- **Team:** 4 KAIST students with complementary roles (Team Leader & Aerodynamics Modeler, Instrumentation & Vision Specialist, Machine Learning & Uncertainty Engineer, Fuel Ecology & Policy Analyst).
- **Duration & Budget:** 21 Days (3 Weeks across Canberra, Sydney, Melbourne); **$14,950 total request** (strictly within the $15,000 program ceiling).
- **Deliverables:**
  1. *Experimental Validation Blueprint:* A formal engineering whitepaper specifying camera placement, frame rates ($\ge 500$ FPS), and calibration standards for 6-DOF untethered firebrand tracking.
  2. *Australian Wildland Firebrand Morphology Atlas:* Measured physical metrics (area, mass, thickness, aspect ratio $\lambda$, dry density) for 200+ field-harvested bark samples.
  3. *Academic Data-Sharing Pathways:* Initiating formal institutional requests to access archival raw wind-tunnel datasets.
  4. *Model Retraining Roadmap:* Concrete specification to upgrade our open-source surrogate models with measured Australian fuel priors.

**Conclusion:** We are not seeking funding to start a project. We have built the computational models, identified their precise physical limitations, and designed an impactful, budget-compliant expedition to address those limitations at the global epicenter of wildfire spotting research.
