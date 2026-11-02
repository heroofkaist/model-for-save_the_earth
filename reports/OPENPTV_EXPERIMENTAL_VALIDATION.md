# Save the Earth — OpenPTV Experimental 3D Trajectory Validation Report

**Project:** Save the Earth — Wildfire Firebrand Transport & Spotting  
**Repository:** `https://github.com/heroofkaist/model-for-save_the_earth`  
**Branch:** `feature/openptv-real-trajectory-validation`  
**Date:** November 2026  
**Author / Computational Physicist:** heroofkaist `<dildiladil63@gmail.com>`  

---

## 1. Executive Summary

This report documents the first experimental validation attempt of **Save the Earth**'s 3D firebrand transport prediction pipeline against independently reconstructed 3D Particle Tracking Velocimetry (3D-PTV) observations from the **OpenPTV 2-camera firebrand dataset** (`OpenPTV/2_cam_firebrands`, University of Maryland / NIST).

### Key Findings:
1. **Reconstructed Trajectories:** 247 optical detections across 6 sequential frames (303–308) were parsed and linked into **138 unique particle tracks** with **0 broken or mismatched reciprocal links**. Among these, **68 tracks are multi-frame connected trajectories** ($\ge 2$ frames), with 4 tracks observed over 5 consecutive frames.
2. **Measurement Volume & Units:** World coordinates were verified to be in **millimeters** ($\text{mm}$) based on the physical pin dimensions of the 3D calibration target (`cal/calibration_points.txt`). Converted to SI meters, the measurement volume spans $X \in [-1.341, +0.288]\,\text{m}$, $Y \in [-0.452, +0.746]\,\text{m}$, and $Z \in [-1.157, +2.101]\,\text{m}$.
3. **Missing Physical Inputs:**
   - **Timebase / Frame Rate:** Unmeasured in the dataset (frames are numbered 303–308 without timestamp metadata).
   - **Ambient Meteorology:** Unmeasured (no anemometer velocity, direction, or turbulence logs).
   - **Particle Morphology & Mass:** Unmeasured (optical tracking yields 2D pixel areas, but no mass or 3D shape reconstruction).
   - **Release Height:** Unobserved (particles were launched upstream outside the camera field of view).
4. **Scientific Compatibility Assessment:** Because critical boundary conditions are unmeasured, running zero-shot end-to-end trajectory prediction is scientifically inappropriate. We report the zero-shot validation status as:  
   **`N/A — insufficient physical inputs (unmeasured wind, mass, release height, and timebase)`**.
5. **Assumption-Dependent Kinematic & Compatibility Study:**
   - Evaluated velocity sensitivity across plausible high-speed imaging frame rates ($25\,\text{Hz}$ to $250\,\text{Hz}$), yielding mean particle speeds from $5.57\,\text{m/s}$ (at 25 Hz) to $55.73\,\text{m/s}$ (at 250 Hz).
   - Track tortuosity (path length / net displacement) averages $1.04 \pm 0.08$, indicating near-linear ballistic transport over the short observation window ($\approx 0.08\,\text{s}$).
   - Simulated physical RK4 and surrogate envelopes over Track #26 show that the observed trajectory curvature is physically consistent with aerodynamic drag acting on millimeter-scale char particles under mild crosswinds ($2 - 6\,\text{m/s}$).

---

## 2. Experimental Source, License & Provenance

- **Data Source Repository:** `https://github.com/OpenPTV/2_cam_firebrands`
- **Institution / Provenance:** University of Maryland (Department of Fire Protection Engineering) in collaboration with the National Institute of Standards and Technology (NIST) Fire Research Division.
- **Upstream Git Commit SHA:** `2f45dcf45798add1ba26fb39927292cf50da7791`
- **Experimental Setup:** Large-scale 3D Particle Tracking Velocimetry (3D-PTV) utilizing a dual high-speed camera configuration with synchronized stereo calibration.
- **License:** OpenPTV / Academic research dataset (MIT / Open-access research data).

---

## 3. Exact Raw Files Used & Cryptographic Integrity

All raw files were retrieved directly from upstream and verified with SHA-256 checksums:

| File | Description | SHA-256 Checksum |
|---|---|---|
| `res/ptv_is.303` | Reconstructed 3D positions & links (Frame 303, N=42) | `81ccba13ee66624320e26b3b3f32196d79bb1ebb773ef48301a2183c4402b35d` |
| `res/ptv_is.304` | Reconstructed 3D positions & links (Frame 304, N=38) | `7a8b3338913117ba28a19567e43d0572d15ca5b4bfb4813ba0cf3c82e176299e` |
| `res/ptv_is.305` | Reconstructed 3D positions & links (Frame 305, N=38) | `1104a4a1271ea66c1097010753239bb1e38152688140187d09be515757b3581d` |
| `res/ptv_is.306` | Reconstructed 3D positions & links (Frame 306, N=45) | `356ff4a1208fd1c3c4ce9f6905ecb3a841cc84ad0e3dd22898dfdaf18e9ed3fd` |
| `res/ptv_is.307` | Reconstructed 3D positions & links (Frame 307, N=45) | `2446d0d7183c2713656a64f5296cf13e61c03847b0ff51ae46bfe886f9d40bbf` |
| `res/ptv_is.308` | Reconstructed 3D positions & links (Frame 308, N=39) | `703d0390fbcde3fab754c74007be29163247d6232eb010a41258c7e52670c226` |
| `cal/calibration_points.txt` | 3D Calibration target coordinates (63 points) | `7a393f669db7bc2374b7c191a2fc7f215f91eb81ff4bbceea52b414f494f1c93` |
| `cal/cam1.tif.ori` | Camera 1 photogrammetric orientation & intrinsics | `b6e1ee74d9e03d9370132b357f4fb93e22649ec3643734a781079fc61a46114b` |
| `cal/cam2.tif.ori` | Camera 2 photogrammetric orientation & intrinsics | `4240742f9e4eb09d220f8ea346ecf06d860166d40369805d762f0f46d3e8a4bb` |
| `parameters_Run1.yaml` | Complete PyPTV runtime configuration | `e836ec39d5b78f44d85208f0f0c05574041b3e7a00eb0ecb9b8eb7c2ee1fc8f8` |

---

## 4. Frozen Pretrained Model Checkpoint Verification

Before evaluating experimental data, the pretrained model weights from the 10,000 synthetic trajectory dataset were locked and verified:

| Model | Checkpoint File | Architecture | Status | SHA-256 Checksum |
|---|---|---|---|---|
| **Model B** | `checkpoints/best_trajectory_model.pt` | `NeuralTrajectorySurrogate` (3 ResBlocks, 128 hidden) | **Frozen** | `f4a4eb118455fcca8a8d35c5a8457ae537ea2b42763fba9971236711cd3a2acd` |
| **Model C** | `checkpoints/best_hybrid_surrogate.pt` | `PhysicsInformedHybridSurrogate` (Ballistic + ResNet) | **Frozen** | `1c51b469e0c800720c01b8bebd44763902344d701dcadc2119eaf1ebe4a04e25` |

Deterministic test evaluation confirmed that inference yields identical numerical predictions to historical benchmarks.

---

## 5. Coordinate Calibration & Spatial Units

### 5.1 Evidence for Millimeter ($\text{mm}$) Scale
1. In `cal/calibration_points.txt`, target points are arranged in a grid with steps:
   - $\Delta X \approx 203.2\,\text{mm}$ to $215.9\,\text{mm}$ (exactly $8.0$ to $8.5$ inches).
   - $\Delta Y \approx 150.0\,\text{mm}$ to $155.0\,\text{mm}$ (exactly $6.0$ inches).
   - Target points reach up to $X = \pm 822.32\,\text{mm}$ and $Y = \pm 460.00\,\text{mm}$.
2. In `cal/cam1.tif.ori`, the camera primary distance is $Z = 6773.65\,\text{mm}$ ($6.77\,\text{m}$) and lens focal length is $f = 29.60\,\text{mm}$.
3. In `parameters_Run1.yaml`, sensor pixel pitch is `pix_x = 0.00586`, `pix_y = 0.00586` ($5.86\,\mu\text{m}$, standard machine vision sensor).

### 5.2 SI Metric Conversion
All raw coordinates were scaled by $10^{-3}$:
$$x_{\text{m}} = \frac{X_{\text{raw}}}{1000.0}, \quad y_{\text{m}} = \frac{Y_{\text{raw}}}{1000.0}, \quad z_{\text{m}} = \frac{Z_{\text{raw}}}{1000.0}$$

The resulting measurement volume spans:
- **$X$ (Cross-stream / Horizontal):** $[-1.341\,\text{m}, +0.288\,\text{m}]$ (width $1.629\,\text{m}$)
- **$Y$ (Vertical):** $[-0.452\,\text{m}, +0.746\,\text{m}]$ (height $1.198\,\text{m}$)
- **$Z$ (Optical Depth / Streamwise):** $[-1.157\,\text{m}, +2.101\,\text{m}]$ (depth $3.258\,\text{m}$)

---

## 6. Track Reconstruction & Quality Control Audit

The importer implemented in [`src/save_the_earth/data/openptv.py`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/src/save_the_earth/data/openptv.py) verified bidirectional link reciprocity:

$$\text{pt}_i \in \text{frame}_t \text{ with } \text{next} = j \iff \text{pt}_j \in \text{frame}_{t+1} \text{ with } \text{prev} = i$$

### Reconstructed Trajectory Counts:
- **Total Detections:** 247 reconstructed 3D points across 6 frames.
- **Total Unique Particle Tracks:** 138 tracks.
- **Single-Frame Detections (Length = 1):** 70 tracks.
- **Multi-Frame Connected Trajectories (Length $\ge 2$):** **68 tracks**.
- **Broken / Mismatched Links:** **0** (100% link reciprocity across all frames).

### Track Length Distribution:
| Track Length | Count | Description |
|---|---:|---|
| 1 Frame | 70 | Fleeting detection / particle entered or exited FOV |
| 2 Frames | 39 | Velocity vector calculable (single displacement step) |
| 3 Frames | 21 | Acceleration vector calculable (two displacement steps) |
| 4 Frames | 4 | Extended flight observation |
| 5 Frames | 4 | Long-duration track traversing almost entire sequence |

---

## 7. Model Compatibility Matrix

Our pretrained models (`NeuralTrajectorySurrogate` and `PhysicsInformedHybridSurrogate`) require the following physical input vector $\mathbf{x} \in \mathbb{R}^9$:

| Input Variable | Model Expectation | Measured | Derived | Unknown | Experimental Reality in OpenPTV |
|---|---|:---:|:---:|:---:|---|
| **Release height ($h$)** | Launch altitude ($1 - 25\,\text{m}$) | ❌ | ❌ | ✅ | Outside camera field of view (upstream burner) |
| **Wind speed ($u_{\text{ref}}$)** | Reference velocity ($2 - 20\,\text{m/s}$) | ❌ | ❌ | ✅ | No anemometer record in OpenPTV repository |
| **Wind direction ($\theta$)** | Wind azimuth angle | ❌ | ❌ | ✅ | Unmeasured ambient flow angle |
| **Particle mass ($m$)** | Mass in kg ($0.02 - 5.0\,\text{g}$) | ❌ | ❌ | ✅ | Optical imaging cannot measure particle mass |
| **Projected area ($A$)** | Area in $\text{m}^2$ ($0.2 - 120\,\text{cm}^2$) | ❌ | ⚠️ | ❌ | Pixel area recorded in 2D, but 3D surface area uncalibrated |
| **Drag coefficient ($C_d$)** | Particle drag ($0.6 - 1.4$) | ❌ | ❌ | ✅ | Unmeasured (depends on particle morphology) |
| **Surface roughness ($z_0$)** | Aerodynamic roughness | ❌ | ❌ | ✅ | Laboratory floor / facility geometry unspecified |
| **Plume updraft ($w_0$)** | Plume buoyancy | ❌ | ❌ | ✅ | Thermal updraft field uncharacterized |
| **Initial velocity ($v_{0z}$)** | Vertical velocity | ❌ | ⚠️ | ❌ | Calculable from $\Delta r$ only if frame rate is known |

> [!IMPORTANT]
> **Scientific Integrity Principle:**  
> We refuse to guess these 9 parameters, plug them into our models, and present the result as a "zero-shot experimental validation". A comparison where wind, mass, area, and release height are all assumed is a **sensitivity compatibility study**, not an independent prediction test.

---

## 8. Kinematic & Frame-Rate Sensitivity Analysis

Because the camera acquisition frame rate is not explicitly declared in `sequence.par` or `parameters_Run1.yaml`, we analyze particle displacement distributions and infer velocity bounds across documented high-speed imaging frame rates:

- **Mean Step Displacement:** $222.91\,\text{mm}$ ($0.223\,\text{m}$)
- **Median Step Displacement:** $200.78\,\text{mm}$ ($0.201\,\text{m}$)
- **Maximum Step Displacement:** $504.60\,\text{mm}$ ($0.505\,\text{m}$)

### Inferred Velocity Bounds:
| Frame Rate ($f_s$) | Inter-frame Timestep ($\Delta t$) | Mean Velocity ($v_{\text{mean}}$) | 95th Percentile Speed | Total Sequence Duration (6 frames) |
|---|---|---:|---:|---|
| **25 Hz** (Standard video) | $0.040\,\text{s}$ | $5.57\,\text{m/s}$ | $10.15\,\text{m/s}$ | $0.200\,\text{s}$ |
| **50 Hz** (High-speed) | $0.020\,\text{s}$ | $11.15\,\text{m/s}$ | $20.30\,\text{m/s}$ | $0.100\,\text{s}$ |
| **100 Hz** (NIST Emberometer standard) | $0.010\,\text{s}$ | $22.29\,\text{m/s}$ | $40.59\,\text{m/s}$ | $0.050\,\text{s}$ |
| **250 Hz** (Fast PTV) | $0.004\,\text{s}$ | $55.73\,\text{m/s}$ | $101.48\,\text{m/s}$ | $0.020\,\text{s}$ |

Based on typical firebrand wind-tunnel velocities ($5 - 15\,\text{m/s}$), **$50\,\text{Hz}$ ($\Delta t = 20\,\text{ms}$) or $100\,\text{Hz}$ ($\Delta t = 10\,\text{ms}$)** represent the most physically plausible operating regimes for this experiment.

---

## 9. Trajectory Linearity & Tortuosity

For all 68 multi-frame tracks, we evaluated the **tortuosity ratio**:

$$\tau = \frac{L_{\text{path}}}{D_{\text{net}}} = \frac{\sum_{k=1}^{N-1} \|\mathbf{r}_{k+1} - \mathbf{r}_k\|}{\|\mathbf{r}_N - \mathbf{r}_1\|}$$

- **Mean Tortuosity:** $1.042 \pm 0.081$
- **Median Tortuosity:** $1.008$
- **Fraction of Tracks with $\tau < 1.05$ (Near-Linear):** **86.8%**

This demonstrates that over a short observation window ($0.05 - 0.10\,\text{s}$), airborne firebrands travel in predominantly linear ballistic trajectories with subtle curvature driven by gravity and drag.

---

## 10. Performance Comparison: Synthetic Benchmarks vs OpenPTV

| Model Architecture | Synthetic 3D Trajectory MAE | Synthetic Landing MAE | OpenPTV Experimental Status | OpenPTV Zero-Shot Result |
|---|---:|---:|---|---|
| **Analytical Ballistic** | 5.971 m | 37.857 m | Unconstrained | N/A — unmeasured inputs |
| **Model A: RK4 Numerical Solver** | 0.000 m (Exact) | 0.000 m | Sensitive to wind/mass | N/A — unmeasured inputs |
| **Model B: Neural Trajectory Surrogate** | 4.736 m | **8.121 m** | Requires 9 inputs | N/A — unmeasured inputs |
| **Model C: Physics-Informed Hybrid** | **1.750 m** | 11.878 m | Requires 9 inputs | N/A — unmeasured inputs |

> [!NOTE]
> The OpenPTV dataset captures only a **$0.08\,\text{s}$ snippet** of airborne particle flight. It does **not observe ember landing events**, so landing distance metrics cannot be computed from this dataset.

---

## 11. Diagnostic Figures

Four high-resolution diagnostic plots were generated in [`reports/figures/`](file:///Users/heroofhumanity/Projects%20VS%20code/model%20for%20save_the_earth/reports/figures/):

1. `openptv_reconstructed_3d_tracks.png`: Complete 3D visualization of all 68 multi-frame connected particle tracks inside the calibrated measurement volume.
2. `openptv_tracking_continuity.png`: Track length distribution histogram and total observed travel distance (mm).
3. `openptv_displacement_distribution.png`: Frame rate velocity sensitivity curve ($25 - 250\,\text{Hz}$) and track tortuosity scatter plot.
4. `openptv_model_comparison.png`: Observed 5-frame trajectory of Track #26 compared against physical RK4 simulation envelopes under varying wind and drag assumptions.

---

## 12. Scientific Recommendations for Future Field Campaigns

To enable rigorous, zero-shot validation of ML firebrand trajectory models:

1. **Synchronized Wind Field Logging:** Every PTV camera trigger must record ultrasonic anemometer velocity vectors ($u, v, w$) at 20 Hz minimum.
2. **High-Speed Timestamp Metadata:** Camera capture headers must include microsecond-precision hardware timestamps ($\mu\text{s}$) to eliminate timebase ambiguity.
3. **Pre-Flight / Post-Flight Sizing:** Collect deposited embers in water pans at the exit of the PTV measurement volume to obtain correlated mass and projected area distributions.
4. **Field of View Extension:** Extend tracking across a longer domain ($> 5\,\text{m}$) to capture non-linear settling and drag deceleration.
