"""
Save the Earth — OpenPTV Experimental Firebrand Trajectory Importer.

Parses 3D Particle Tracking Velocimetry (3D-PTV) reconstructions from the
OpenPTV 2-camera firebrand experimental dataset (University of Maryland / NIST).
Reconstructs multi-frame particle tracks, converts coordinates from millimeters to SI meters,
validates reciprocal track link consistency, records cryptographic provenance,
and outputs canonical tabular trajectories.
"""

import os
import sys
import glob
import json
import hashlib
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UPSTREAM_REPO_URL = "https://github.com/OpenPTV/2_cam_firebrands.git"
UPSTREAM_COMMIT_SHA = "2f45dcf45798add1ba26fb39927292cf50da7791"


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class OpenPTVImporter:
    """
    Importer and track reconstructor for OpenPTV 2-camera firebrand dataset.
    """
    def __init__(self, raw_data_dir: Optional[Path] = None):
        if raw_data_dir is None:
            raw_data_dir = PROJECT_ROOT / "data" / "raw" / "openptv_2_cam_firebrands"
        self.raw_dir = Path(raw_data_dir)
        self.res_dir = self.raw_dir / "res"
        self.experiment_id = "openptv_2_cam_firebrands_umd"

    def verify_raw_files(self) -> Dict[str, str]:
        """Verifies presence of raw reconstruction files and records checksums."""
        if not self.res_dir.exists():
            raise FileNotFoundError(f"OpenPTV results directory not found at: {self.res_dir}")

        ptv_files = sorted(glob.glob(str(self.res_dir / "ptv_is.*")))
        if not ptv_files:
            raise FileNotFoundError(f"No ptv_is.* files found in: {self.res_dir}")

        checksums = {}
        for f in ptv_files:
            p = Path(f)
            # Only include numeric frame extensions
            ext = p.suffix.lstrip(".")
            if ext.isdigit():
                checksums[p.name] = compute_sha256(p)
        return checksums

    def parse_frame_ptv(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses an OpenPTV ptv_is.<frame> file.
        Format:
        Line 0: N (number of particles)
        Lines 1..N: prev_idx next_idx x_mm y_mm z_mm
        """
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return []

        n_pts = int(lines[0])
        records = []
        for row_idx, line in enumerate(lines[1:]):
            parts = line.split()
            if len(parts) < 5:
                continue
            prev_idx = int(parts[0])
            next_idx = int(parts[1])
            x_mm = float(parts[2])
            y_mm = float(parts[3])
            z_mm = float(parts[4])

            # Convert to SI meters (1 mm = 0.001 m)
            records.append({
                "row_idx": row_idx,
                "prev_idx": prev_idx,
                "next_idx": next_idx,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
                "x_m": x_mm * 1e-3,
                "y_m": y_mm * 1e-3,
                "z_m": z_mm * 1e-3,
            })
        return records

    def reconstruct_tracks(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads all frame files, validates reciprocal links, and connects trajectories.
        Returns:
        - List of canonical row records
        - Quality Control (QC) summary dictionary
        """
        checksums = self.verify_raw_files()
        frame_files = {}
        for fname in checksums:
            frame_num = int(fname.split(".")[1])
            frame_files[frame_num] = self.res_dir / fname

        sorted_frames = sorted(frame_files.keys())
        frames_data = {}
        for f in sorted_frames:
            frames_data[f] = self.parse_frame_ptv(frame_files[f])

        # Step 1: Validate reciprocal tracking links
        audit_links = []
        for i, f in enumerate(sorted_frames[:-1]):
            f_next = sorted_frames[i + 1]
            curr_pts = frames_data[f]
            next_pts = frames_data[f_next]

            for idx, pt in enumerate(curr_pts):
                nxt = pt["next_idx"]
                if nxt >= 0:
                    if nxt >= len(next_pts):
                        audit_links.append({
                            "type": "out_of_bounds",
                            "frame": f,
                            "next_frame": f_next,
                            "pt_idx": idx,
                            "claimed_next": nxt,
                        })
                    else:
                        back = next_pts[nxt]["prev_idx"]
                        if back != idx:
                            audit_links.append({
                                "type": "reciprocal_mismatch",
                                "frame": f,
                                "next_frame": f_next,
                                "pt_idx": idx,
                                "claimed_next": nxt,
                                "target_prev": back,
                            })

        # Step 2: Track assembly
        tracks = []
        track_id_counter = 1
        visited = {f: set() for f in sorted_frames}

        for start_f_idx, f in enumerate(sorted_frames):
            for pt_idx, pt in enumerate(frames_data[f]):
                # A track starts if it has no valid reciprocal predecessor
                if pt["prev_idx"] < 0 or pt_idx in visited[f]:
                    if pt_idx in visited[f]:
                        continue

                    # Trace trajectory forward
                    curr_f_idx = start_f_idx
                    curr_pt_idx = pt_idx
                    track_nodes = []

                    while True:
                        curr_f = sorted_frames[curr_f_idx]
                        curr_node = frames_data[curr_f][curr_pt_idx]
                        visited[curr_f].add(curr_pt_idx)

                        track_nodes.append({
                            "frame_id": curr_f,
                            "pt_idx": curr_pt_idx,
                            "x_m": curr_node["x_m"],
                            "y_m": curr_node["y_m"],
                            "z_m": curr_node["z_m"],
                            "x_mm": curr_node["x_mm"],
                            "y_mm": curr_node["y_mm"],
                            "z_mm": curr_node["z_mm"],
                            "source_file": f"ptv_is.{curr_f}",
                            "source_file_sha256": checksums[f"ptv_is.{curr_f}"],
                        })

                        nxt = curr_node["next_idx"]
                        # Check forward link validity
                        if nxt >= 0 and (curr_f_idx + 1) < len(sorted_frames):
                            next_f = sorted_frames[curr_f_idx + 1]
                            next_pts = frames_data[next_f]
                            if nxt < len(next_pts) and next_pts[nxt]["prev_idx"] == curr_pt_idx:
                                curr_f_idx += 1
                                curr_pt_idx = nxt
                            else:
                                break
                        else:
                            break

                    tracks.append({
                        "track_id": track_id_counter,
                        "start_frame": f,
                        "length": len(track_nodes),
                        "nodes": track_nodes,
                    })
                    track_id_counter += 1

        # Step 3: Build canonical tabular rows
        canonical_rows = []
        for t in tracks:
            t_id = t["track_id"]
            t_len = t["length"]
            # Tracking confidence: high (1.0) for multi-frame connected tracks, moderate (0.5) for single detections
            conf = 1.0 if t_len >= 2 else 0.5

            for node in t["nodes"]:
                canonical_rows.append({
                    "experiment_id": self.experiment_id,
                    "track_id": t_id,
                    "frame_id": node["frame_id"],
                    "timestamp_s": "", # Unmeasured / unverified in raw dataset
                    "x_m": round(node["x_m"], 6),
                    "y_m": round(node["y_m"], 6),
                    "z_m": round(node["z_m"], 6),
                    "tracking_confidence": conf,
                    "source_file": node["source_file"],
                    "source_file_sha256": node["source_file_sha256"],
                })

        # Step 4: Quality Control & Spatial Summary
        all_x = [r["x_m"] for r in canonical_rows]
        all_y = [r["y_m"] for r in canonical_rows]
        all_z = [r["z_m"] for r in canonical_rows]

        lengths = [t["length"] for t in tracks]
        qc_summary = {
            "upstream_repo": UPSTREAM_REPO_URL,
            "upstream_commit": UPSTREAM_COMMIT_SHA,
            "total_frames": len(sorted_frames),
            "frame_ids": sorted_frames,
            "total_reconstructed_detections": len(canonical_rows),
            "total_unique_tracks": len(tracks),
            "multi_frame_tracks_ge2": sum(1 for l in lengths if l >= 2),
            "single_frame_detections": sum(1 for l in lengths if l == 1),
            "track_length_distribution": {
                str(k): sum(1 for l in lengths if l == k) for k in sorted(set(lengths))
            },
            "broken_or_mismatched_links": len(audit_links),
            "audit_anomalies": audit_links,
            "spatial_bounds_m": {
                "x_min": min(all_x), "x_max": max(all_x),
                "y_min": min(all_y), "y_max": max(all_y),
                "z_min": min(all_z), "z_max": max(all_z),
            },
            "spatial_bounds_mm": {
                "x_min": min(all_x) * 1000.0, "x_max": max(all_x) * 1000.0,
                "y_min": min(all_y) * 1000.0, "y_max": max(all_y) * 1000.0,
                "z_min": min(all_z) * 1000.0, "z_max": max(all_z) * 1000.0,
            },
            "timestamp_status": "unresolved_missing_in_raw_dataset",
        }

        return canonical_rows, qc_summary

    def export_canonical_dataset(
        self,
        output_csv: Path = PROJECT_ROOT / "data" / "processed" / "openptv_reconstructed_tracks.csv",
        output_qc: Path = PROJECT_ROOT / "data" / "manifests" / "openptv_qc_report.json"
    ) -> Tuple[Path, Path]:
        """Parses and saves canonical CSV and QC report."""
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_qc.parent.mkdir(parents=True, exist_ok=True)

        rows, qc = self.reconstruct_tracks()

        fieldnames = [
            "experiment_id",
            "track_id",
            "frame_id",
            "timestamp_s",
            "x_m",
            "y_m",
            "z_m",
            "tracking_confidence",
            "source_file",
            "source_file_sha256",
        ]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with open(output_qc, "w") as f:
            json.dump(qc, f, indent=2)

        print(f"[OpenPTV Importer] Exported {len(rows)} detections ({qc['total_unique_tracks']} tracks) to {output_csv}")
        print(f"[OpenPTV Importer] Saved QC report to {output_qc}")
        return output_csv, output_qc


if __name__ == "__main__":
    importer = OpenPTVImporter()
    importer.export_canonical_dataset()
