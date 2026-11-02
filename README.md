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
