# Wildfire Ember Transport & Performance Modeling (save_the_earth)

Feasibility study and simulation framework for predicting wildfire ember transport, spot fire probability, and wildfire behavior (KAISTian Amurs Project).

## 🚀 Quickstart

### 1. Requirements & Installation
```bash
python3 -m pip install matplotlib numpy scipy
```

### 2. Run Baseline Simulation
```bash
python3 ember_simulation.py
```

### 3. Custom Scenarios
```bash
# Higher wind speed and release height with 1000 sampled particles
python3 ember_simulation.py --wind 10 --height 30 --particles 1000

# View all configuration flags
python3 ember_simulation.py --help
```

---

## 🔬 Model Physics & Verification

The baseline model simulates firebrands/embers as discrete particles under:
* **Gravity:** $g = 9.81\,\text{m/s}^2$
* **Quadratic aerodynamic drag:** $\vec{F}_{\text{drag}} = -\frac{1}{2} \rho C_d A \|\vec{v}_{\text{rel}}\| \vec{v}_{\text{rel}}$ relative to ambient wind $\vec{u}_{\text{wind}}$.
* **Numerical Integration:** 4th-order Runge-Kutta (RK4) with ground impact interpolation.

### Verification Suite:
* **Free fall ($C_d = 0$):** reproduces analytical projectile kinematics ($t = \sqrt{2h/g}$).
* **Terminal velocity:** matches $v_t = \sqrt{\frac{2mg}{\rho C_d A}}$ ($6.33\,\text{m/s}$ for baseline).
* **Timestep convergence:** verified between $\Delta t = 0.01\,\text{s}$ and $0.001\,\text{s}$.
* **Wind symmetry:** verified across $\pm u_{\text{wind}}$.

With standard defaults ($h = 20\,\text{m}$, $u_{\text{wind}} = 5\,\text{m/s}$, $m = 1\,\text{g}$, $A = 4\,\text{cm}^2$), the baseline ember lands at **$\approx 14.35\,\text{m}$** downwind after **$\approx 3.66\,\text{s}$**.

---

## 📊 Outputs & Visualizations
Results are saved to `ember_results/`:
* `trajectory_and_landing_distribution.png`: 2-panel plot (side-view trajectories + top-down 2D landing distribution).
* `baseline_trajectory.csv`: complete time-series of position and velocity.
* `landing_data.csv`: Monte Carlo landing coordinates, flight times, and sampled properties.

---

## Evidence-first research data pipeline (draft PR)

**Current status: no raw observational dataset has been verified or trained in this repository.** The 22-page Save the Earth research report identifies sources and methodology; it is not a table of measurements.

The research branch adds:

- [Source registry](data/manifests/source_registry.json) marking source accessibility, reported counts, ingestion and readiness honestly.
- [Data-readiness protocol](docs/DATA_READINESS.md) for separated particle-distance, 3D trajectory, thermal, ignition, flux and wildfire spotting datasets.
- `download_sources.py` for source metadata and optional USDA file acquisition (requires internet and compatible remote metadata API).
- `normalize_firebrands.py` for explicit verified-header/unit conversion.
- `research_data.py` for provenance-aware QC, quarantine, eligible-only views and deterministic, experiment-grouped, **write-once** train/validation/test split manifests.
- `train_locked_distance.py` for a measured-only frozen-partition **training-set median** reference benchmark; it does not use post-landing mass/area as if available at release.
- Separate exploratory `train_distance.py` and `train_ember.py` scripts for data-rich cases, NOT yet validated on real data.

```bash
python -m pip install numpy matplotlib
python -m unittest discover -s tests -v
python ember_simulation.py --verify
python download_sources.py --dataset both --metadata-only
# Download and audit real source bytes and their metadata first
python research_data.py --input data/processed/canonical.csv --freeze-split data/manifests/split_v1.csv
python train_locked_distance.py --data data/interim/eligible.csv --split data/manifests/split_v1.csv
```

The final two commands intentionally cannot succeed until measured and independently grouped observations are ingested. Incomplete provenance, source permission, uncertain units and unknown experiment/run identity are reasons to quarantine—not to invent data. Never mix simulated particles with measured evaluation or treat successful Australian spot fires as single ember landings. See [research documentation](docs/DATA_READINESS.md).
