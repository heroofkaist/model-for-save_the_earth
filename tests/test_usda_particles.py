"""
Unit tests for USDA JFSP 15-1-04-4 Particle-Level Dataset and Benchmark.
"""

import csv
import json
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "usda_individual_firebrands.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "usda_particles_manifest.json"
SOURCES_PATH = PROJECT_ROOT / "data" / "manifests" / "external_validation_sources.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "usda_particle_evaluation_metrics.json"

from src.save_the_earth.data.usda_particles import (
    USDAParticleBenchmark,
    compute_file_sha256,
)
from src.save_the_earth.evaluation.evaluate_usda_particles import compute_metrics


class TestUSDAParticles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CSV_PATH.exists():
            benchmark = USDAParticleBenchmark()
            benchmark.export_canonical_dataset()

    def test_canonical_dataset_and_manifest_exist(self):
        self.assertTrue(CSV_PATH.exists(), f"Dataset CSV missing: {CSV_PATH}")
        self.assertTrue(MANIFEST_PATH.exists(), f"Manifest JSON missing: {MANIFEST_PATH}")

        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

        computed_sha = compute_file_sha256(CSV_PATH)
        self.assertEqual(manifest["sha256"], computed_sha)
        self.assertEqual(manifest["total_records"], 60370)
        self.assertEqual(manifest["fuel_category_counts"]["vegetative"], 9249)
        self.assertEqual(manifest["fuel_category_counts"]["structural"], 51121)

    def test_group_disjoint_partitions_no_leakage(self):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            train_groups = set()
            val_groups = set()
            test_groups = set()

            for r in reader:
                grp = r["experiment_id"]
                split = r["split"]
                if split == "train":
                    train_groups.add(grp)
                elif split == "validation":
                    val_groups.add(grp)
                elif split == "test":
                    test_groups.add(grp)

        # Assert strict zero leakage between groups
        self.assertEqual(len(train_groups.intersection(val_groups)), 0)
        self.assertEqual(len(train_groups.intersection(test_groups)), 0)
        self.assertEqual(len(val_groups.intersection(test_groups)), 0)

    def test_physical_bounds_and_si_units(self):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Sample first 2000 rows for fast verification
            for i, r in enumerate(reader):
                if i >= 2000:
                    break
                m = float(r["mass_kg"])
                a = float(r["projected_area_m2"])
                d = float(r["landing_distance_m"])
                u = float(r["wind_speed_m_s"])

                self.assertGreater(m, 0.0)
                self.assertLessEqual(m, 0.1)
                self.assertGreater(a, 0.0)
                self.assertLessEqual(a, 0.05)
                self.assertGreaterEqual(d, 0.1)
                self.assertLessEqual(d, 35.0)
                self.assertIn(round(u, 2), [5.36, 11.17, 17.88])

    def test_external_validation_sources_registry(self):
        self.assertTrue(SOURCES_PATH.exists(), f"Source registry missing: {SOURCES_PATH}")
        with open(SOURCES_PATH, encoding="utf-8") as f:
            reg = json.load(f)

        sources = reg.get("sources", {})
        self.assertIn("usda_rds_2020_0035_vegetative", sources)
        self.assertIn("usda_rds_2020_0034_structural", sources)
        self.assertIn("uk_eidc_fds_oregon", sources)
        self.assertIn("openptv_2_cam_firebrands", sources)
        self.assertIn("csiro_2026_vertical_wind_tunnel", sources)
        self.assertIn("nist_emberometer", sources)

        self.assertEqual(sources["uk_eidc_fds_oregon"]["access_status"], "unsuitable_stationary_vegetation_particles")
        self.assertEqual(sources["openptv_2_cam_firebrands"]["access_status"], "downloaded_and_verified")

    def test_compute_metrics_accuracy(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([3.0, 4.0, 5.0])
        m = compute_metrics(y_true, y_pred)

        self.assertAlmostEqual(m["mae_m"], 2.0 / 3.0)
        self.assertAlmostEqual(m["bias_m"], 0.0)
        self.assertAlmostEqual(m["median_ae_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
