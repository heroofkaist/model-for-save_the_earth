"""Train a physics-residual surrogate on measured ember landing data.

This is a research scaffold, NOT a validated wildfire or ignition predictor.
No measured data are bundled. Use independent, provenance-tracked experiments.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ember_simulation import simulate_single_trajectory

FEATURES = ("release_height_m", "wind_x_m_s", "wind_y_m_s",
            "mass_kg", "area_m2", "drag_coefficient", "air_density_kg_m3")
TARGETS = ("landing_x_m", "landing_y_m", "flight_time_s")
REQUIRED = ("experiment_id", "source_id", "data_kind", *FEATURES, *TARGETS)


def load_rows(path, allow_synthetic=False):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if len(rows) < 6:
        raise ValueError("At least six labelled rows are required.")
    for i, row in enumerate(rows, 2):
        for name in ("experiment_id", "source_id", "data_kind"):
            if not row[name].strip():
                raise ValueError(f"Row {i}: {name} is required")
        kind = row["data_kind"].strip().lower()
        if kind not in {"measured", "synthetic"}:
            raise ValueError(f"Row {i}: data_kind must be measured or synthetic")
        if kind == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic data refused; pass --allow-synthetic for DEMO ONLY.")
        for col in (*FEATURES, *TARGETS):
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                raise ValueError(f"Row {i}: non-numeric {col}") from None
            if not np.isfinite(val):
                raise ValueError(f"Row {i}: non-finite {col}")
        if any(float(row[col]) <= 0 for col in
               ("release_height_m", "mass_kg", "area_m2", "air_density_kg_m3")):
            raise ValueError(f"Row {i}: physical inputs must be positive")
        if float(row["drag_coefficient"]) < 0 or float(row["flight_time_s"]) <= 0:
            raise ValueError(f"Row {i}: invalid drag or flight time")
    return rows


def physics_predictions(rows, dt):
    predictions = []
    for row in rows:
        _, _, info = simulate_single_trajectory(
            release_height=float(row["release_height_m"]),
            wind_vec=np.array([float(row["wind_x_m_s"]),
                               float(row["wind_y_m_s"]), 0.0]),
            mass=float(row["mass_kg"]),
            area=float(row["area_m2"]),
            cd=float(row["drag_coefficient"]),
            rho=float(row["air_density_kg_m3"]),
            dt=dt,
        )
        predictions.append([info["landing_x"], info["landing_y"], info["flight_time"]])
    return np.asarray(predictions, dtype=float)


def feature_matrix(rows):
    # Use log transforms for positive variables spanning orders of magnitude.
    a = np.array([[float(r[col]) for col in FEATURES] for r in rows], dtype=float)
    h, wx, wy, mass, area, cd, rho = a.T
    return np.column_stack([np.log(h), wx, wy, np.log(mass), np.log(area),
                            cd, rho, (area / mass), wx * np.sqrt(h),
                            wy * np.sqrt(h)])


def grouped_split(rows, fraction, seed):
    # One experiment must never appear in both train and test.
    groups = np.array([r["experiment_id"] for r in rows])
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Need at least three independent experiment_id groups.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(len(unique) * fraction)))
    n_test = min(n_test, len(unique) - 2)
    test_groups = set(shuffled[:n_test])
    test = np.array([g in test_groups for g in groups], dtype=bool)
    return ~test, test, sorted(test_groups)


def metrics(y, prediction):
    errors = prediction - y
    return {
        "mae": {name: float(np.mean(np.abs(errors[:, k])))
                for k, name in enumerate(TARGETS)},
        "rmse": {name: float(np.sqrt(np.mean(errors[:, k] ** 2)))
                 for k, name in enumerate(TARGETS)}
    }


def run(args):
    rows = load_rows(args.data, allow_synthetic=args.allow_synthetic)
    kinds = sorted({r["data_kind"].lower() for r in rows})
    if len(kinds) > 1:
        raise ValueError("Do not mix measured and synthetic rows in one evaluation.")
    train, test, test_groups = grouped_split(rows, args.test_fraction, args.seed)
    x = feature_matrix(rows)
    y = np.array([[float(r[c]) for c in TARGETS] for r in rows])
    physics = physics_predictions(rows, args.dt)

    # Standardize ONLY using training data. Learn residuals relative to physics.
    mu = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mu) / scale
    design = np.column_stack((np.ones(len(rows)), z))
    residual_train = y[train] - physics[train]

    # Ridge fit with unpenalized intercept; regularization is NOT tuned on test.
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    weights = np.linalg.solve(
        design[train].T @ design[train] + args.ridge * penalty,
        design[train].T @ residual_train
    )
    hybrid = physics + design @ weights
    report = {
        "status": "SYNTHETIC_DEMO_NOT_REAL_WORLD_VALIDATION" if kinds == ["synthetic"]
                  else "MEASURED_HELD_OUT_EVALUATION",
        "source_ids": sorted({r["source_id"] for r in rows}),
        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
        "held_out_experiments": test_groups,
        "physics_test": metrics(y[test], physics[test]),
        "hybrid_test": metrics(y[test], hybrid[test]),
        "units": {"landing_x_m": "m", "landing_y_m": "m", "flight_time_s": "s"},
        "limitations": ("Fixed wind, no plume, combustion or ignition; random "
                        "experiment holdout is not prospective or geographic validation.")
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.savez(outdir / "model.npz", weights=weights, mean=mu, scale=scale)
    with open(outdir / "heldout_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "source_id", "data_kind",
                         *["observed_" + c for c in TARGETS],
                         *["physics_" + c for c in TARGETS],
                         *["hybrid_" + c for c in TARGETS]])
        for i in np.flatnonzero(test):
            writer.writerow([rows[i]["experiment_id"], rows[i]["source_id"],
                             rows[i]["data_kind"], *y[i], *physics[i], *hybrid[i]])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with documented measured landing observations")
    parser.add_argument("--outdir", default="training_results")
    parser.add_argument("--allow-synthetic", action="store_true", help="Demo-only; NOT validation")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--ridge", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.dt <= 0 or args.ridge <= 0:
        parser.error("test-fraction in (0,1), dt > 0, ridge > 0 required")
    run(args)
