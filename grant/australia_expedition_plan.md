# International Research Expedition Plan: Australia (CSIRO)

**Project:** Save the Earth — Empirical 3D Firebrand In-Flight Dynamics  
**Host Institution:** CSIRO Environment / Bushfire Research Laboratory (Canberra, Australia)  
**Collaborative Leads:** Dr. Andrew L. Sullivan (CSIRO) & KAISTian Amurs Research Group  
**Target Window:** 2027 Field Season (Australian Bushfire Dynamics Window)  

---

## 1. Scientific Objectives & Purpose

The purpose of the Australia Expedition is to obtain the **world's first open-access, time-resolved 6-DOF trajectory and combustion database for natural wildland firebrands** to calibrate and validate our physics-informed neural network models.

Australia represents the premier global laboratory for firebrand research:
1. **Eucalyptus Forest spotting:** Australian stringybark and ribbon bark produce the longest documented spotting distances in the world (up to 30+ km ahead of catastrophic fire fronts).
2. **Specialized CSIRO Vertical Wind Tunnel (VWT):** CSIRO maintains the only facility capable of suspending untethered, actively burning natural firebrands in continuous upward airflow while maintaining stereo-photogrammetric observation.

---

## 2. Experimental Facility & Instrumentation Setup

```
                       CSIRO VERTICAL WIND TUNNEL (VWT)
                                  ┌─────────┐
                                  │ Exhaust │
                                  └───▲─▲───┘
                                      │ │
                             [Test Section: 1m x 1m]
               High-Speed Cam A ──►  ┌───────┐  ◄── High-Speed Cam B
               (1000 FPS, Photron)   │   ●   │      (1000 FPS, Photron)
                                     │ Ember │
               Synchronized IR  ──►  └───────┘  ◄── Particle Tracking PIV
               (FLIR 100 Hz Thermal)  ▲ ▲ ▲ ▲       (Dual-Cavity Nd:YAG)
                                      │ │ │ │
                                  [Flow Honeycomb]
                                  [Variable Blower]
```

### Instrumentation Specifications:
1. **Dual Synchronized High-Speed Photogrammetric Cameras:**
   - 2x Photron FASTCAM SA-Z (1000 FPS at 1024x1024 resolution) positioned orthogonally (90° baseline).
   - Calibrated via a precision 3D dot-grid calibration target with sub-millimeter reprojection error (< 0.15 mm).
   - Provides full 6-DOF tracking: position $\mathbf{x}(t)$, linear velocity $\mathbf{v}(t)$, orientation quaternions $\mathbf{q}(t)$, and angular velocity $\boldsymbol{\omega}(t)$.
2. **Synchronized High-Speed Radiometric Infrared Camera:**
   - FLIR SC7000 (100 Hz, 3–5 $\mu$m band, calibrated $300^\circ\text{C} - 1200^\circ\text{C}$).
   - Measures surface temperature distribution $T_p(x, y, t)$, flaming vs. smoldering transition, and combustion boundary layer expansion.
3. **Continuous Mass Loss Balance (Suspended Load Cell):**
   - Micro-load cell ($\pm 0.1$ mg accuracy at 50 Hz) for tethered baseline comparison, complemented by stereo-volumetric reconstruction for untethered trials.

---

## 3. Experimental Matrix (120 Factorial Trials)

We propose a full factorial test matrix: $3 \text{ Fuel Types} \times 2 \text{ Combustion States} \times 4 \text{ Wind Regimes} \times 5 \text{ Replicates} = \mathbf{120 \text{ Trials}}$.

| Factor | Levels | Specifications |
|---|---|---|
| **Fuel Species** | 3 Types | 1. Eucalyptus Stringybark (*Eucalyptus obliqua*)<br>2. Ribbon/Candlebark (*Eucalyptus viminalis*)<br>3. Monterey Pine (*Pinus radiata*) bark/needles |
| **Combustion State** | 2 States | 1. Cold Inert (Aerodynamic Baseline)<br>2. Actively Burning (Flaming/Glowing Smolder) |
| **Airflow Regime** | 4 Regimes | 1. Quiescent Laminar ($w_z = 2 - 10$ m/s)<br>2. Shear Flow ($du/dz \approx 1.5 \text{ s}^{-1}$)<br>3. Vortical / Swirl Flow (Swirl number $S \approx 0.3$)<br>4. Oscillatory Gusts ($\Delta w = \pm 20\%$, $f = 1$ Hz) |
| **Replication** | 5 Runs | 5 independent specimens per cell to quantify stochastic flutter variance |

---

## 4. Deliverables & Data Open-Science Commitments

1. **The CSIRO-SaveTheEarth Open Firebrand Trajectory Database (CS-FTD):**
   - Published under CC-BY 4.0 with DOI via Zenodo and CSIRO Data Access Portal.
   - 120 fully calibrated 3D time series: timestamps (ms), 3D positions, 6-DOF orientation angles, surface temperatures, and instantaneous drag/lift coefficients.
2. **Direct Model Integration into `model-for-save_the_earth`:**
   - Pretrained checkpoints updated with real 6-DOF dynamics, replacing synthetic RK4 surrogates.
   - Verification of the Wong combustion buoyancy factor ($\eta_{\text{buoy}}$) across natural fuels.
3. **Joint High-Impact Journal Publication:**
   - Target: *Combustion and Flame* or *Nature Communications* on *"Direct 6-DOF Photogrammetric Measurement of Burning Firebrand Trajectories"*.

---

## 5. Expedition Timeline & Budget Summary

### Timeline (6-Month Campaign):
- **Months 1–2:** Pre-expedition coordination, fuel specimen harvesting, sensor calibration protocols.
- **Month 3:** On-site wind-tunnel testing at CSIRO Bushfire Laboratory (Canberra).
- **Months 4–5:** Stereo-PTV reconstruction pipeline, volumetric calibration, mass-loss extraction.
- **Month 6:** Model retraining, open-data publishing, grant reporting.

### Budget Justification ($145,000 Total):
- **CSIRO Wind Tunnel Facility Access & Technician Support:** $65,000
- **High-Speed Camera & Optics Lease (Photron stereo rig):** $32,000
- **Travel & Field Subsistence (3 Researchers, 4 weeks in Canberra):** $28,000
- **Fuel Specimen Collection, Logistics & Hazardous Materials Clearance:** $8,000
- **Open-Access Data Hosting & Publication Charges:** $12,000
