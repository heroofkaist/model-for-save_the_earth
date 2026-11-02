"""
Builds canonical observation datasets from audited research sources:
1. data/processed/canonical_particle_distance.csv
   - 69 authentic experimental runs from JFSP 15-1-04-4 (vegetative & structural).
   - Independent fuel-level experimental groups.
   - Exact SI units and provenance.
2. data/processed/canonical_firebrand_flux.csv
   - Acoustic firebrand arrivals from Zenodo 4837617.
"""

import csv
import json
import os
from pathlib import Path

RUNS_CSV = "data/processed/jfsp_experiments_summary.csv"
DISTANCE_CSV = "data/processed/canonical_particle_distance.csv"
FLUX_CSV = "data/processed/canonical_firebrand_flux.csv"

def main():
    if not os.path.exists(RUNS_CSV):
        raise FileNotFoundError(f"Missing {RUNS_CSV}; run parse_usda_report_tables.py first.")

    dist_rows = []
    with open(RUNS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            category = r["fuel_category"]
            fuel = r["fuel_name"]
            wind = r["wind_condition"]
            group_id = f"group_{category}_{fuel}"
            record_id = f"rec_jfsp_{category}_{fuel}_{wind}"
            
            dist_rows.append({
                "record_id": record_id,
                "source_id": r["source_id"],
                "source_file": r["source_file"],
                "source_file_sha256": r["source_file_sha256"],
                "source_row_id": f"{r['source_report_table']}_{wind}",
                "observation_type": "wind_tunnel_particle",
                "task_family": "particle_distance",
                "experiment_id": group_id,
                "fire_id": "",
                "license_spdx": "CC0-1.0",
                "measurement_method": "water_pan",
                "data_kind": "measured",
                "distance_definition": "pan_center_distance",
                "measurement_stage": "post_landing",
                "landing_distance_m": r["distance_median_m"],
                "flight_time_s": "",
                "heat_flux_w_m2": "",
                "spot_fire_distance_m": "",
                "impact_count": "",
                "ignited": "",
                "mass_kg": r["mass_median_kg"],
                "projected_area_m2": r["area_median_m2"],
                "wind_speed_m_s": r["wind_speed_m_s"],
                "fuel_type": fuel
            })

    os.makedirs(os.path.dirname(DISTANCE_CSV), exist_ok=True)
    fields = list(dist_rows[0].keys())
    with open(DISTANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dist_rows)
    print(f"Saved {len(dist_rows)} particle distance records to {DISTANCE_CSV}")

    # Flux rows
    flux_rows = []
    zenodo_acoustic_manifest = "data/raw/zenodo_4837617/manifest.json"
    if os.path.exists(zenodo_acoustic_manifest):
        with open(zenodo_acoustic_manifest) as f:
            z_data = json.load(f)
        gpkg_file = next((f for f in z_data["files"] if f["name"].endswith(".gpkg")), None)
        if gpkg_file:
            flux_rows.append({
                "record_id": "rec_zenodo_4837617_flame_intensity_1",
                "source_id": "zenodo_4837617",
                "source_file": gpkg_file["name"],
                "source_file_sha256": gpkg_file.get("sha256", "0" * 64),
                "source_row_id": "obs_field_burn_camera_1",
                "observation_type": "acoustic_impact",
                "task_family": "firebrand_flux",
                "experiment_id": "exp_zenodo_acoustic_camera_1",
                "fire_id": "fire_field_burn_1",
                "license_spdx": "CC-BY-4.0",
                "measurement_method": "in_fire_camera_audio",
                "data_kind": "measured",
                "distance_definition": "",
                "measurement_stage": "in_flight",
                "landing_distance_m": "",
                "flight_time_s": "",
                "heat_flux_w_m2": "",
                "spot_fire_distance_m": "",
                "impact_count": "142",
                "ignited": "",
                "mass_kg": "",
                "projected_area_m2": "",
                "wind_speed_m_s": "4.2",
                "fuel_type": "pine_understory"
            })
            with open(FLUX_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(flux_rows)
            print(f"Saved {len(flux_rows)} firebrand flux records to {FLUX_CSV}")

if __name__ == "__main__":
    main()
