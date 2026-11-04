#!/usr/bin/env python3
"""
Save the Earth — Unified End-to-End Firebrand Research and Modeling Pipeline.

Runs the complete scientific pipeline with a single command:
  python3 run_pipeline.py

Pipeline Steps:
  1. Mechanistic Physics Verification (ember_simulation.py --verify)
  2. Physics Sensitivity Analysis (physics_sensitivity.py)
  3. Raw Source & Archive Audit (inspect_raw.py)
  4. Experimental Data & Moments Extraction (parse_usda_report_tables.py)
  5. Canonical SI Dataset Construction & QC (build_canonical_datasets.py & research_data.py)
  6. Multi-Model Group-CV & Test Benchmarking (train_distance_benchmarks.py)
  7. Diagnostic Figure & Scientific Summary Generation (generate_diagnostics.py)
  8. Automated Unit Test Suite Execution (tests/)

Operational prediction mode:
  python3 run_pipeline.py --predict --wind 12.5 --category vegetative
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def print_banner():
    banner = r"""
================================================================================
           SAVE THE EARTH — WILDFIRE FIREBRAND PREDICTION SYSTEM
    Unified Scientific Ingestion, Physics Simulation, and Ensemble Pipeline
================================================================================
"""
    print(banner)


def run_step(step_num, total_steps, title, command_args):
    print(f"\n[{step_num}/{total_steps}] >>> {title.upper()}")
    print(f"Executing: {' '.join(command_args)}")
    start_time = time.time()
    result = subprocess.run(command_args, capture_output=True, text=True)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"FAILED (Exit Code {result.returncode}) after {elapsed:.2f}s")
        print("\n--- STDOUT ---")
        print(result.stdout)
        print("\n--- STDERR ---")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        lines = result.stdout.strip().split("\n")
        display = lines[:15]
        for l in display:
            print(f"  {l}")
        if len(lines) > 15:
            print(f"  ... [+{len(lines) - 15} lines omitted]")
        print(f"COMPLETED in {elapsed:.2f}s")


def run_predict(wind, category, moisture=None):
    import numpy as np
    from sklearn.linear_model import Ridge, HuberRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if not math.isfinite(wind) or wind < 0:
        raise ValueError("wind speed must be a finite, nonnegative value")
    if category not in ("vegetative", "structural"):
        raise ValueError("fuel category must be vegetative or structural")

    print(f"\n================ EXPERIMENTAL FIREBRAND DISTANCE PREDICTION ================")
    print(f"Model Version: v2.0-group-cv-calibrated | Dataset: USDA JFSP 15-1-04-4 (Aggregated Runs)")
    print(f"Input Parameters: Nominal Wind = {wind:.2f} m/s | Fuel Category = {category}")
    if moisture is not None:
        print(f"Fuel Moisture: {moisture:.1f}% (context only; it is NOT a predictor input for these models)")

    # Out-of-distribution (OOD) checks
    if wind < 3.0 or wind > 25.0:
        print(f"⚠️  WARNING: Requested wind speed ({wind:.1f} m/s) is outside tested experimental range [5.36 - 17.88 m/s].")
        print("   Extrapolated aerodynamic predictions carry increased epistemic uncertainty.")
    elif wind < 5.36 or wind > 17.88:
        print(f"ℹ️  NOTE: Requested wind speed ({wind:.1f} m/s) is near boundary of experimental calibration range.")

    print("-" * 75)

    data_path = "data/interim/eligible.csv"
    split_path = "data/manifests/split_v1.csv"
    if not os.path.exists(data_path) or not os.path.exists(split_path):
        print("Missing training data. Please run `python3 run_pipeline.py` first.")
        return

    from train_distance_benchmarks import load_dataset
    records = load_dataset(data_path, split_path)
    dev_recs = [r for r in records if r["split"] in ("train", "validation")]
    y_dev = np.array([r["landing_distance_m"] for r in dev_recs])

    # Fit feature transformers on development data only
    cat_dev = np.array([[r["fuel_category"]] for r in dev_recs])
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(cat_dev)

    num_dev = np.column_stack([[r["wind_speed_m_s"] for r in dev_recs], [r["wind_speed_m_s"]**2 for r in dev_recs]])
    scaler = StandardScaler().fit(num_dev)

    X_dev = np.column_stack([scaler.transform(num_dev), enc.transform(cat_dev)])

    query_num = np.array([[wind, wind**2]])
    query_cat = np.array([[category]])
    query_X = np.column_stack([scaler.transform(query_num), enc.transform(query_cat)])

    # Models
    ridge = Ridge(alpha=10.0).fit(X_dev, y_dev)
    huber = HuberRegressor(alpha=1.0, epsilon=1.35).fit(X_dev, y_dev)
    rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42).fit(X_dev, y_dev)

    p_ridge = float(np.maximum(0, ridge.predict(query_X)[0]))
    p_huber = float(np.maximum(0, huber.predict(query_X)[0]))
    p_rf = float(np.maximum(0, rf.predict(query_X)[0]))
    p_ens = (p_ridge + p_rf) / 2.0

    # Model E is a retrospective research network. Its feature vector requires
    # post-collection mass and area, which are unknown at release time.
    # Never fabricate those values to produce a "prospective" inference.

    # Conformal interval belongs to the RIDGE predictor, not Huber.
    # The saved calibration is optional; no hard-coded fallback is justified.
    benchmark_path = Path("distance_results/benchmark_comparison.json")
    conformal_info = None
    if benchmark_path.is_file():
        with benchmark_path.open(encoding="utf-8") as f:
            benchmark = json.load(f)
        record = benchmark.get("Conformal_Uncertainty_80", {})
        radius = record.get("conformal_radius_m")
        if radius is not None and math.isfinite(float(radius)) and float(radius) >= 0:
            conformal_info = {
                "radius": float(radius),
                "test_coverage_pct": record.get("empirical_test_coverage_pct"),
            }

    print(f"Model Predictions for Experimental Landing Distance:")
    print(f"  Prospective Ridge Regression:       {p_ridge:.2f} m")
    print(f"  Prospective Huber Regressor (Robust): {p_huber:.2f} m")
    print(f"  Prospective Random Forest (depth=2): {p_rf:.2f} m")
    print(f"  Prospective Parsimonious Ensemble:  {p_ens:.2f} m")
    print(f"  -------------------------------------------------------------")
    print(f"  Laboratory run-median estimate (Huber): {p_huber:.2f} m")
    if conformal_info is not None:
        rad = conformal_info["radius"]
        print(
            f"  Nominal 80% group-CV interval around Ridge: "
            f"[{max(0.0, p_ridge-rad):.2f}, {p_ridge+rad:.2f}] m"
        )
        coverage = conformal_info["test_coverage_pct"]
        if coverage is not None:
            print(
                f"  Observed frozen-test coverage: {float(coverage):.1f}% "
                "(not a guarantee on unseen fuels)"
            )
    else:
        print("  No verified interval file available; interval not reported.")
    print("  Model E omitted: requires post-deposition mass/area and reconstructed-data training.")
    print("  Individual-ember dispersion omitted: no validated prospective distribution model.")
    print("=" * 75)
    print("⚠️  SCIENTIFIC DISCLAIMER:")
    print("   Research-only estimate of run-median wind-tunnel water-pan deposition.")
    print("   NOT a wildfire operational forecast. Does not predict burning survival or ignition.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Save the Earth Unified Pipeline Runner")
    parser.add_argument("--predict", action="store_true", help="Run operational model inference")
    parser.add_argument("--wind", type=float, default=11.17, help="Wind speed (m/s) for prediction")
    parser.add_argument("--moisture", type=float, default=None, help="Fuel moisture content (%%)")
    parser.add_argument("--category", choices=["vegetative", "structural"], default="vegetative", help="Fuel category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running unit tests at the end")
    args = parser.parse_args()

    print_banner()

    if args.predict:
        run_predict(args.wind, args.category, args.moisture)
        return

    total_steps = 8 if not args.skip_tests else 7
    py_exec = sys.executable

    # Step 1: Physics verification
    run_step(1, total_steps, "Mechanistic Physics Simulator Verification", [py_exec, "ember_simulation.py", "--verify"])

    # Step 2: Physics sensitivity analysis
    run_step(2, total_steps, "Mechanistic Aerodynamic Sensitivity Analysis", [py_exec, "physics_sensitivity.py"])

    # Step 3: Ingest & raw audit
    audit_path = Path("data/interim/raw_audit.json")
    if audit_path.exists():
        audit_path.unlink()
    run_step(3, total_steps, "Raw Data Sources & Cryptographic Audit", [py_exec, "inspect_raw.py", "data/raw"])

    # Step 4: Parse report tables
    run_step(4, total_steps, "USDA JFSP 15-1-04-4 Experimental Table Extraction", [py_exec, "parse_usda_report_tables.py"])

    # Step 5a: Canonical datasets
    run_step(5, total_steps, "Canonical Dataset Construction & Standardization", [py_exec, "build_canonical_datasets.py"])

    # Step 5b: QC audit & split check
    run_step(5, total_steps, "Observational Quality Control & Split Validation", [
        py_exec, "research_data.py", "--input", "data/processed/canonical_particle_distance.csv", "--outdir", "data/interim"
    ])

    # Step 6: Train ensemble benchmarks with Group CV
    run_step(6, total_steps, "Group-Validated Cross-Validation & Benchmark Training", [
        py_exec, "train_distance_benchmarks.py", "--seed", str(args.seed)
    ])

    # Step 7: Generate diagnostics
    run_step(7, total_steps, "Diagnostic Figures & Scientific Summary Generation", [py_exec, "generate_diagnostics.py"])

    # Step 8: Unit tests
    if not args.skip_tests:
        run_step(8, total_steps, "Automated Unit & Scientific Integrity Test Suite", [
            py_exec, "-m", "unittest", "discover", "-s", "tests", "-v"
        ])

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Generated Artifacts:")
    print("  - Group CV Performance Table: distance_results/group_cv_metrics.csv")
    print("  - Benchmark Comparison JSON:  distance_results/benchmark_comparison.json")
    print("  - Test Metrics Table:         distance_results/metrics_table.csv")
    print("  - Test Set Predictions:       distance_results/test_predictions.csv")
    print("  - Physics Sensitivity JSON:   distance_results/physics_sensitivity.json")
    print("  - Diagnostic Figures:         reports/figures/ (12 figures)")
    print("  - Dataset & QC Summaries:     reports/*.csv")
    print("  - Verified Canonical Data:    data/processed/canonical_particle_distance.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
