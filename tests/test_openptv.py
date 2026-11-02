"""
Unit tests for OpenPTV 3D Particle Tracking Importer and Experimental Validation Pipeline.
"""

import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch

from src.save_the_earth.data.openptv import OpenPTVImporter, compute_sha256
from src.save_the_earth.evaluation.openptv_validation import verify_frozen_checkpoints


class TestOpenPTV(unittest.TestCase):
    def setUp(self):
        # Create a self-contained synthetic fixture simulating OpenPTV res files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name)
        self.res_dir = self.raw_dir / "res"
        self.res_dir.mkdir(parents=True, exist_ok=True)

        # Frame 303 (2 particles: pt 0 links to pt 1 in frame 304; pt 1 is unlinked)
        frame_303_content = """2
-1   1  -500.0   200.0  1800.0
-1  -2  -100.0   150.0  1900.0
"""
        # Frame 304 (2 particles: pt 0 is unlinked; pt 1 links back to pt 0 in frame 303)
        frame_304_content = """2
-1  -2  -400.0   180.0  1850.0
 0  -2  -450.0   190.0  1820.0
"""
        with open(self.res_dir / "ptv_is.303", "w") as f:
            f.write(frame_303_content)
        with open(self.res_dir / "ptv_is.304", "w") as f:
            f.write(frame_304_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_frame_and_unit_conversion(self):
        importer = OpenPTVImporter(raw_data_dir=self.raw_dir)
        records = importer.parse_frame_ptv(self.res_dir / "ptv_is.303")
        self.assertEqual(len(records), 2)
        # Verify unit conversion (mm to m)
        self.assertAlmostEqual(records[0]["x_mm"], -500.0)
        self.assertAlmostEqual(records[0]["x_m"], -0.500)
        self.assertAlmostEqual(records[0]["y_m"], 0.200)
        self.assertAlmostEqual(records[0]["z_m"], 1.800)

    def test_reciprocal_track_linking(self):
        importer = OpenPTVImporter(raw_data_dir=self.raw_dir)
        rows, qc = importer.reconstruct_tracks()
        self.assertEqual(qc["total_frames"], 2)
        self.assertEqual(qc["total_reconstructed_detections"], 4)
        # Should have 1 connected track (length 2) and 2 single-frame detections (length 1)
        self.assertEqual(qc["multi_frame_tracks_ge2"], 1)
        self.assertEqual(qc["single_frame_detections"], 2)
        self.assertEqual(qc["broken_or_mismatched_links"], 0)

        # Check canonical format fields
        expected_fields = {
            "experiment_id", "track_id", "frame_id", "timestamp_s",
            "x_m", "y_m", "z_m", "tracking_confidence",
            "source_file", "source_file_sha256"
        }
        self.assertEqual(set(rows[0].keys()), expected_fields)
        # Timestamp must be empty string (unresolved)
        self.assertEqual(rows[0]["timestamp_s"], "")

    def test_broken_link_handling(self):
        # Create an inconsistent forward link in frame 303 pointing to an invalid index
        bad_frame_content = """1
-1   99  -500.0   200.0  1800.0
"""
        with open(self.res_dir / "ptv_is.303", "w") as f:
            f.write(bad_frame_content)

        importer = OpenPTVImporter(raw_data_dir=self.raw_dir)
        rows, qc = importer.reconstruct_tracks()
        # Out of bounds link must be flagged in audit anomalies and not crash
        self.assertGreater(qc["broken_or_mismatched_links"], 0)
        self.assertEqual(qc["audit_anomalies"][0]["type"], "out_of_bounds")

    def test_frozen_checkpoint_integrity(self):
        verification = verify_frozen_checkpoints()
        # Verify Model B hash
        self.assertEqual(
            verification["model_b"]["sha256"],
            "f4a4eb118455fcca8a8d35c5a8457ae537ea2b42763fba9971236711cd3a2acd"
        )
        # Verify Model C hash
        self.assertEqual(
            verification["model_c"]["sha256"],
            "1c51b469e0c800720c01b8bebd44763902344d701dcadc2119eaf1ebe4a04e25"
        )
        # Deterministic prediction check
        self.assertGreater(verification["model_b"]["test_sample_pred_land_dist_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
