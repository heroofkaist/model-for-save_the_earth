#!/usr/bin/env python3
"""
Save the Earth — Unified End-to-End Firebrand Research and Modeling Pipeline.

Runs the complete scientific pipeline with a single command:
  python3 run_pipeline.py

Pipeline Steps:
  1. Mechanistic Physics Verification (ember_simulation.py --verify)
  2. Raw Source & Archive Audit (inspect_raw.py)
  3. Experimental Data & Moments Extraction (parse_usda_report_tables.py)
  4. Canonical SI Dataset Construction & QC (build_canonical_datasets.py & research_data.py)
  5. Multi-Model Ensemble Training & Benchmarking (train_distance_benchmarks.py)
  6. Diagnostic Figure & Summary Generation (generate_diagnostics.py)
  7. Automated Unit Test Suite Execution (tests/)

Also supports operational prediction mode:
  python3 run_pipeline.py --predict --wind 12.5 --moisture 10.0 --category vegetative
"""

import argparse
import csv
import json
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
        # Print concise output
        lines = result.stdout.strip().split("\n")
        display = lines[:15]
        for l in display:
            print(f"  {l}")
        if len(lines) > 15:
            print(f"  ... [+{len(lines) - 15} lines omitted]")
        print(f"COMPLETED in {elapsed:.2f}s")


def run_predict(wind, moisture, category):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    print(f"\n================ OPERATIONAL ENSEMBLE PREDICTION ================")
    print(f"Inputs: Wind = {wind:.2f} m/s | Moisture Content = {moisture:.1f}% | Fuel Category = {category}")
    print("-" * 65)

    data_path = "data/interim/eligible.csv"
    split_path = "data/manifests/split_v1.csv"
    if not os.path.exists(data_path) or not os.path.exists(split_path):
        print("Missing training data. Please run `python3 run_pipeline.py` first.")
        return

    from train_distance_benchmarks import load_dataset
    records = load_dataset(data_path, split_path)
    train_recs = [r for r in records if r["split"] == "train"]
    y_train = np.array([r["landing_distance_m"] for r in train_recs])

    # Fit feature transformers on train only
    num_train = np.column_stack([
        [r["wind_speed_m_s"] for r in train_recs],
        [r["gust_peak_m_s"] for r in train_recs],
        [r["wind_speed_m_s"]**2 for r in train_recs],
        [r["moisture_content_percent"] for r in train_recs]
    ])
    cat_train = np.array([[r["fuel_category"]] for r in train_recs])

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_feat_train = enc.fit_transform(cat_train)
    scaler = StandardScaler()
    num_feat_train = scaler.fit_transform(num_train)
    X_train = np.column_stack([num_feat_train, cat_feat_train])

    # Prepare query vector
    gust = wind * 1.28  # empirical gust factor from wind tunnel profiles
    query_num = np.array([[wind, gust, wind**2, moisture]])
    query_cat = np.array([[category]])
    query_X = np.column_stack([scaler.transform(query_num), enc.transform(query_cat)])

    # Models
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X_train, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000, solver="lbfgs", random_state=42).fit(X_train, y_train)

    # Quantiles
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=100, max_depth=3, random_state=42).fit(X_train, y_train)

    p_rf = float(np.maximum(0, rf.predict(query_X)[0]))
    p_gb = float(np.maximum(0, gb.predict(query_X)[0]))
    p_ridge = float(np.maximum(0, ridge.predict(query_X)[0]))
    p_mlp = float(np.maximum(0, mlp.predict(query_X)[0]))
    p_ens = (p_rf + p_gb + p_ridge + p_mlp) / 4.0
    p_lower = float(np.maximum(0, q10.predict(query_X)[0]))
    p_upper = float(np.maximum(0, q90.predict(query_X)[0]))

    print(f"Model Predictions for Landing Distance:")
    print(f"  Prospective Random Forest:          {p_rf:.2f} m")
    print(f"  Prospective Gradient Boosting:      {p_gb:.2f} m")
    print(f"  Prospective Ridge Regression:       {p_ridge:.2f} m")
    print(f"  Prospective MLP Neural Network:     {p_mlp:.2f} m")
    print(f"  -------------------------------------------------------------")
    print(f"  ★ ENSEMBLE MEAN PREDICTION:         {p_ens:.2f} m")
    print(f"  ★ 80% CONFIDENCE INTERVAL (P10-P90): [{p_lower:.2f} m  —  {p_upper:.2f} m]")
    print(f"  ★ SPOTTING HAZARD ENVELOPE (90th%):  {p_upper:.2f} m downwind")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Save the Earth Unified Pipeline Runner")
    parser.add_argument("--predict", action="store_true", help="Run operational model inference")
    parser.add_argument("--wind", type=float, default=11.17, help="Wind speed (m/s) for prediction")
    parser.add_argument("--moisture", type=float, default=12.0, help="Fuel moisture content (%%)")
    parser.add_argument("--category", choices=["vegetative", "structural"], default="vegetative", help="Fuel category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running unit tests at the end")
    args = parser.parse_args()

    print_banner()

    if args.predict:
        run_predict(args.wind, args.moisture, args.category)
        return

    total_steps = 7 if not args.skip_tests else 6
    py_exec = sys.executable

    # Step 1: Physics verification
    run_step(1, total_steps, "Mechanistic Physics Simulator Verification", [py_exec, "ember_simulation.py", "--verify"])

    # Step 2: Ingest & raw audit
    audit_path = Path("data/interim/raw_audit.json")
    if audit_path.exists():
        audit_path.unlink()
    run_step(2, total_steps, "Raw Data Sources & Cryptographic Audit", [py_exec, "inspect_raw.py", "data/raw"])

    # Step 3: Parse report tables
    run_step(3, total_steps, "USDA JFSP 15-1-04-4 Experimental Table Extraction", [py_exec, "parse_usda_report_tables.py"])

    # Step 4a: Canonical datasets
    run_step(4, total_steps, "Canonical Dataset Construction & Standardization", [py_exec, "build_canonical_datasets.py"])

    # Step 4b: QC audit & split check
    run_step(4, total_steps, "Observational Quality Control & Split Validation", [
        py_exec, "research_data.py", "--input", "data/processed/canonical_particle_distance.csv", "--outdir", "data/interim"
    ])

    # Step 5: Train ensemble benchmarks
    run_step(5, total_steps, "Multi-Model Ensemble Training & Held-Out Test Evaluation", [
        py_exec, "train_distance_benchmarks.py", "--seed", str(args.seed)
    ])

    # Step 6: Generate diagnostics
    run_step(6, total_steps, "Diagnostic Figures & Scientific Summary Generation", [py_exec, "generate_diagnostics.py"])

    # Step 7: Unit tests
    if not args.skip_tests:
        run_step(7, total_steps, "Automated Unit & Scientific Integrity Test Suite", [
            py_exec, "-m", "unittest", "discover", "-s", "tests", "-v"
        ])

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Generated Artifacts:")
    print("  - Models & Benchmark Results: distance_results/benchmark_comparison.json")
    print("  - Benchmark Metrics Table:    distance_results/metrics_table.csv")
    print("  - Test Set Predictions:       distance_results/test_predictions.csv")
    print("  - Diagnostic Figures:         reports/figures/ (10 figures)")
    print("  - Dataset & QC Summaries:     reports/*.csv")
    print("  - Verified Canonical Data:    data/processed/canonical_particle_distance.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
