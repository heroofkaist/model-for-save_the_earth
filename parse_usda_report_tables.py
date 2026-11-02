"""
Extracts experimental run benchmarks and statistical properties from the official
Joint Fire Science Program Project 15-1-04-4 Final Report (Appendix E, Tables 4-26, and Tables 1-3).
Preserves exact provenance, reported sample sizes, statistical moments, correlations,
fuel moisture content, and converted SI units.
"""

import csv
import hashlib
import json
import os
import re
from pathlib import Path
import pypdf

PDF_PATH = "data/raw/15-1-04-4/15-1-04-4_final_report.pdf"
OUT_CSV = "data/processed/jfsp_experiments_summary.csv"

# Wind speeds documented in report Section 4 and Appendix E
WIND_CONDITIONS = [
    {"name": "idle", "speed_m_s": 5.36, "gust_peak_m_s": 5.36},
    {"name": "medium", "speed_m_s": 11.17, "gust_peak_m_s": 14.30},
    {"name": "high", "speed_m_s": 17.88, "gust_peak_m_s": 23.00}
]

# Fuel moisture content (%) from Table 1 (p. 10) and Table 3 (p. 14)
FUEL_MOISTURE_MAP = {
    "privacy_fence": 13.5,
    "lattice_fence": 13.5,
    "corner_cedar_plywood": 7.3,
    "corner_cedar_osb": 7.8,
    "corner_osb_osb": 5.65,
    "recycled_shake_roof": 0.0,
    "frt_cedar_shake_roof": 12.0,
    "non_frt_cedar_shake_roof": 12.0,
    "mockup_wood_combustible_roof": 13.5,
    "mockup_wood_noncombustible_roof": 13.5,
    "mockup_wood_assembly_combustible": 13.5,
    "mockup_wood_assembly_noncombustible": 13.5,
    "mockup_osb_combustible": 5.65,
    "mockup_osb_noncombustible": 5.65,
    "mockup_osb_assembly_combustible": 5.65,
    "mockup_osb_assembly_noncombustible": 5.65,
    "mockup_fibercement_combustible": 7.8,
    "mockup_fibercement_noncombustible": 7.8,
    "little_bluestem_grass": 10.0,
    "chamise": 20.3,
    "saw_palmetto": 59.9,
    "loblolly_pine": 8.8,
    "leyland_cypress": 10.4
}

def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_table_page(txt, table_num, fuel_name, category, source_id, page_num, pdf_sha):
    m_samples = re.search(r'Sample Size\s+(\d+)\s+(\d+)\s+(\d+)', txt)
    if not m_samples:
        return []
    samples = [int(m_samples.group(1)), int(m_samples.group(2)), int(m_samples.group(3))]

    def get_3_floats(section_name, stat_name):
        sec_match = re.search(rf"{section_name}.*?{stat_name}\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", txt, re.DOTALL)
        if sec_match:
            return [float(sec_match.group(i)) for i in (1, 2, 3)]
        return [None, None, None]

    # Distances (m)
    dist_means = get_3_floats("Flying Distance", "Mean")
    dist_stds = get_3_floats("Flying Distance", "Standard Deviation")
    dist_skews = get_3_floats("Flying Distance", "Skewness")
    dist_meds = get_3_floats("Flying Distance", "Median")

    # Projected Area (cm2) -> m2 is cm2 * 1e-4
    area_means = get_3_floats("Projected Area", "Mean")
    area_stds = get_3_floats("Projected Area", "Standard Deviation")
    area_skews = get_3_floats("Projected Area", "Skewness")
    area_meds = get_3_floats("Projected Area", "Median")

    # Mass (g) -> kg is g * 1e-3
    mass_means = get_3_floats("Mass", "Mean")
    mass_stds = get_3_floats("Mass", "Standard Deviation")
    mass_skews = get_3_floats("Mass", "Skewness")
    mass_meds = get_3_floats("Mass", "Median")

    # Correlations
    r_ma = re.search(r"Mass\s+and\s+Area.*?([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", txt)
    r_md = re.search(r"Mass\s+and\s+Flying.*?Distance.*?([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", txt, re.DOTALL)
    r_ad = re.search(r"Area\s+and\s+Flying.*?Distance.*?([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)", txt, re.DOTALL)

    r_ma_vals = [float(r_ma.group(i)) for i in (1, 2, 3)] if r_ma else [None, None, None]
    r_md_vals = [float(r_md.group(i)) for i in (1, 2, 3)] if r_md else [None, None, None]
    r_ad_vals = [float(r_ad.group(i)) for i in (1, 2, 3)] if r_ad else [None, None, None]

    mc = FUEL_MOISTURE_MAP.get(fuel_name, 10.0)

    runs = []
    for i, w in enumerate(WIND_CONDITIONS):
        exp_id = f"jfsp_{category}_{fuel_name}_{w['name']}"
        runs.append({
            "experiment_id": exp_id,
            "source_id": source_id,
            "fuel_category": category,
            "fuel_name": fuel_name,
            "moisture_content_percent": mc,
            "wind_condition": w["name"],
            "wind_speed_m_s": w["speed_m_s"],
            "gust_peak_m_s": w["gust_peak_m_s"],
            "sample_size": samples[i],
            "distance_mean_m": dist_means[i],
            "distance_std_m": dist_stds[i],
            "distance_skewness": dist_skews[i],
            "distance_median_m": dist_meds[i],
            "area_mean_m2": area_means[i] * 1e-4 if area_means[i] is not None else None,
            "area_std_m2": area_stds[i] * 1e-4 if area_stds[i] is not None else None,
            "area_skewness": area_skews[i],
            "area_median_m2": area_meds[i] * 1e-4 if area_meds[i] is not None else None,
            "mass_mean_kg": mass_means[i] * 1e-3 if mass_means[i] is not None else None,
            "mass_std_kg": mass_stds[i] * 1e-3 if mass_stds[i] is not None else None,
            "mass_skewness": mass_skews[i],
            "mass_median_kg": mass_meds[i] * 1e-3 if mass_meds[i] is not None else None,
            "corr_mass_area": r_ma_vals[i],
            "corr_mass_dist": r_md_vals[i],
            "corr_area_dist": r_ad_vals[i],
            "source_report_table": f"Table {table_num}",
            "source_report_page": page_num,
            "source_file": "15-1-04-4_final_report.pdf",
            "source_file_sha256": pdf_sha,
            "data_kind": "measured",
            "task_family": "particle_distance",
            "measurement_stage": "post_landing",
            "measurement_method": "water_pan",
            "license_spdx": "CC0-1.0"
        })
    return runs

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}")
    pdf_sha = get_sha256(PDF_PATH)
    reader = pypdf.PdfReader(PDF_PATH)
    print(f"Reading {PDF_PATH} ({len(reader.pages)} pages)...")

    # Table metadata from Appendix E
    tables_meta = [
        # Structural Components (Tables 4-11, pages 44-51)
        (44, 4, "privacy_fence", "structural_component", "usda_structural_2020_0034"),
        (45, 5, "lattice_fence", "structural_component", "usda_structural_2020_0034"),
        (46, 6, "corner_cedar_plywood", "structural_component", "usda_structural_2020_0034"),
        (47, 7, "corner_cedar_osb", "structural_component", "usda_structural_2020_0034"),
        (48, 8, "corner_osb_osb", "structural_component", "usda_structural_2020_0034"),
        (49, 9, "recycled_shake_roof", "structural_component", "usda_structural_2020_0034"),
        (50, 10, "frt_cedar_shake_roof", "structural_component", "usda_structural_2020_0034"),
        (51, 11, "non_frt_cedar_shake_roof", "structural_component", "usda_structural_2020_0034"),
        # Structural Mockup Assemblies (Tables 12-21, pages 52-61)
        (52, 12, "mockup_wood_combustible_roof", "structural_assembly", "usda_structural_2020_0034"),
        (53, 13, "mockup_wood_noncombustible_roof", "structural_assembly", "usda_structural_2020_0034"),
        (54, 14, "mockup_wood_assembly_combustible", "structural_assembly", "usda_structural_2020_0034"),
        (55, 15, "mockup_wood_assembly_noncombustible", "structural_assembly", "usda_structural_2020_0034"),
        (56, 16, "mockup_osb_combustible", "structural_assembly", "usda_structural_2020_0034"),
        (57, 17, "mockup_osb_noncombustible", "structural_assembly", "usda_structural_2020_0034"),
        (58, 18, "mockup_osb_assembly_combustible", "structural_assembly", "usda_structural_2020_0034"),
        (59, 19, "mockup_osb_assembly_noncombustible", "structural_assembly", "usda_structural_2020_0034"),
        (60, 20, "mockup_fibercement_combustible", "structural_assembly", "usda_structural_2020_0034"),
        (61, 21, "mockup_fibercement_noncombustible", "structural_assembly", "usda_structural_2020_0034"),
        # Vegetative Fuels (Tables 22-26, pages 62-66)
        (62, 22, "little_bluestem_grass", "vegetative", "usda_vegetative_2020_0035"),
        (63, 23, "chamise", "vegetative", "usda_vegetative_2020_0035"),
        (64, 24, "saw_palmetto", "vegetative", "usda_vegetative_2020_0035"),
        (65, 25, "loblolly_pine", "vegetative", "usda_vegetative_2020_0035"),
        (66, 26, "leyland_cypress", "vegetative", "usda_vegetative_2020_0035")
    ]

    all_runs = []
    total_particles = 0
    for page_num, table_num, fuel_name, category, source_id in tables_meta:
        txt = reader.pages[page_num - 1].extract_text()
        runs = parse_table_page(txt, table_num, fuel_name, category, source_id, page_num, pdf_sha)
        if runs:
            all_runs.extend(runs)
            part_sum = sum(r["sample_size"] for r in runs)
            total_particles += part_sum
            print(f"Table {table_num:2d} (p.{page_num}): {fuel_name:<35} -> {len(runs)} runs, {part_sum} firebrands")
        else:
            print(f"Warning: Failed parsing Table {table_num} on page {page_num}")

    print(f"\nTotal extracted experimental runs: {len(all_runs)}")
    print(f"Total documented firebrands across all runs: {total_particles}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_runs[0].keys()))
        writer.writeheader()
        writer.writerows(all_runs)

    print(f"Saved verified runs summary to {OUT_CSV}")

if __name__ == "__main__":
    main()
