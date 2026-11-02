# Training and validation protocol (v0)

## Scientific target

**Current supervised targets**: ember landing x/y (metres) and flight time (seconds), conditional on known release height, constant 2D wind, mass, projected area, drag coefficient and air density. This is **not** wildfire spread, ignition probability, or true predictive skill in an uncontrolled fire.

The original simulator remains the interpretable physics baseline. `train_ember.py` fits a regularized linear correction to **physics residuals**; it is deliberately simple. The test set holds out entire experimental groups, not random trajectories. The evaluation includes both an unchanged physics baseline and the learned hybrid. A worse hybrid must be rejected.

## Data contract

One row = **one independently observed particle landing**, not a grid cell and not a simulator time-step. Put records in a CSV outside Git unless their reuse permission is clear.

| Column | Meaning |
| --- | --- |
| experiment_id | Independent release session/experimental campaign ID, used for grouped holdout |
| source_id | DOI, permanent repository identifier, or lab notebook record ID |
| data_kind | `measured` or `synthetic` |
| release_height_m | Height above local ground (m) |
| wind_x_m_s, wind_y_m_s | Two horizontal components (m/s), consistent axes |
| mass_kg | Particle mass (kg) |
| area_m2 | Projected area (m²) |
| drag_coefficient | Dimensionless Cd, with method recorded in dataset metadata |
| air_density_kg_m3 | Ambient air density (kg/m³) |
| landing_x_m, landing_y_m | Observed impact relative to release (m) |
| flight_time_s | Observed time from release to impact (s) |

Required variables must be **measured or clearly documented estimates**; no fabricated values. If mass, drag or wind is unknown, record the gap in the evidence register and **do not impute it silently**. Store source license, collection date, instrument accuracy, wind sensor position/averaging interval, particle type, coordinate orientation, and uncertainty separately in an accompanying metadata file. Cd can be strongly shape/orientation/Reynolds dependent; the fixed-Cd assumption is provisional.

## Usage

```bash
python -m pip install numpy matplotlib
python ember_simulation.py --verify
python train_ember.py --data path/to/measured_landings.csv
```

Writes `training_results/evaluation.json`, `heldout_predictions.csv`, and a NumPy `model.npz` artifact. The training code refuses synthetic data unless `--allow-synthetic` is passed, in which case the report explicitly labels itself as a **synthetic demo**, not external validation. Do not combine synthetic and measured rows in this first evaluation.

## Study design

1. Acquire primary experiment data with observed landing positions, flight times and release conditions. Confirm permission to train/publish.
2. Standardize frames/units and preserve the raw files unchanged. Run plausibility and missingness checks before training.
3. Assign group IDs at **experimental campaign** level; correlated repeated releases must share a group. Hold entire fires or sites out for later external validation.
4. Report sample sizes, range of conditions, errors and failure cases on untouched groups, including physics-only comparisons. Tune hyperparameters only using additional group-aware validation inside training data.
5. Evaluate distributional prediction separately when repeated measured releases under matching conditions exist (e.g., calibration, coverage, CRPS); one observed landing is not an empirical probability map.
6. Test genuinely new meteorological regimes and particle types before interpreting geographic risk. Modeling plume lifting, turbulence, terrain, combustion, and ignition requires additional measurements and distinct targets.

**Important**: Randomly varying inputs in a deterministic simulator creates a distribution of *assumed input uncertainty*, not measured spotting risk. Fitting to simulated outputs can test software but cannot demonstrate real-world accuracy.

## Immediate data work

Maintain `docs/evidence_register.csv` as a source-by-source ledger. Enter only verified facts. A paper describing trajectories does not mean a machine-readable dataset is available. Prioritize observed landing or path trajectories over fire-perimeter-only maps. For Korean sources, check the timestamp resolution and whether detached spot fires are distinguished from the main front.

### Next experiments

Record repeated cold/unheated inert-particle releases, synchronized release/impact video, wind readings at documented heights, mass and orientation. These can validate *mechanical transport* only; they do not validate burning embers, buoyant plumes or ignition. Obtain lab safety review before any heated-firebrand experiments.
