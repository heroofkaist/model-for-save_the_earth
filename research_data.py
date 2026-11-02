"""Evidence-first canonical observations, QA and immutable grouped partitions.

Standard-library only. The input must be a normalized CSV with explicit provenance.
Never imports synthetic simulation as observational truth; ambiguous records go to
quarantine, and split identities are written once, never silently overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

TASK_TARGETS = {
    "particle_distance": "landing_distance_m",
    "particle_trajectory": "flight_time_s",
    "thermal_survival": "heat_flux_w_m2",
    "wildfire_spotting": "spot_fire_distance_m",
}
BASE = ("record_id", "source_id", "source_file", "source_file_sha256",
        "source_row_id", "observation_type", "task_family", "experiment_id",
        "fire_id", "license_spdx", "measurement_method", "data_kind")
ALLOWED_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
POSITIVE = ("mass_kg", "projected_area_m2", "release_height_m", "flight_time_s")
NONNEGATIVE = ("landing_distance_m", "wind_speed_m_s", "heat_flux_w_m2",
               "spot_fire_distance_m")
OBSERVATIONS = {
    "particle_distance": {"wind_tunnel_particle", "field_particle"},
    "particle_trajectory": {"wind_tunnel_particle", "field_particle"},
    "thermal_survival": {"thermal"},
    "wildfire_spotting": {"spot_fire"},
}
OUTPUT_COLUMNS = ("record_id", "source_id", "task_family", "experiment_id",
                  "fire_id", "qc_status", "qc_reason", "measurement_stage",
                  "landing_distance_m", "flight_time_s", "heat_flux_w_m2",
                  "spot_fire_distance_m", "mass_kg", "projected_area_m2",
                  "wind_speed_m_s", "fuel_type", "source_file_sha256")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError("Duplicate header names")
        missing = set(BASE) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing canonical columns: " + ", ".join(sorted(missing)))
        return list(reader)


def number(value):
    if value is None or not str(value).strip():
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def review(row):
    reasons = []
    task = row.get("task_family", "")
    if task not in TASK_TARGETS:
        reasons.append("unknown_task")
    if row.get("data_kind", "").strip().lower() != "measured":
        reasons.append("not_measured")
    if task in OBSERVATIONS and row.get("observation_type") not in OBSERVATIONS[task]:
        reasons.append("wrong_observation_type")
    if row.get("license_spdx") not in ALLOWED_LICENSES:
        reasons.append("license_unverified")
    for col in ("record_id", "source_id", "source_file", "source_row_id",
                "measurement_method"):
        if not row.get(col, "").strip():
            reasons.append("missing_" + col)
    sha = row.get("source_file_sha256", "")
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
        reasons.append("invalid_source_hash")
    group = "fire_id" if task == "wildfire_spotting" else "experiment_id"
    if not row.get(group, "").strip():
        reasons.append("missing_independent_group")
    if task in TASK_TARGETS:
        target = number(row.get(TASK_TARGETS[task]))
        if target is None:
            reasons.append("missing_target")
        elif not math.isfinite(target) or target < 0 or (task == "particle_trajectory" and target == 0):
            reasons.append("invalid_target")
    for field in POSITIVE:
        value = number(row.get(field))
        if value is not None and (not math.isfinite(value) or value <= 0):
            reasons.append("invalid_" + field)
    for field in NONNEGATIVE:
        value = number(row.get(field))
        if value is not None and (not math.isfinite(value) or value < 0):
            reasons.append("invalid_" + field)
    if task == "particle_distance":
        if row.get("distance_definition") not in ("pan_center_distance", "measured_horizontal_distance"):
            reasons.append("distance_definition_unverified")
        if row.get("measurement_stage", "") not in ("post_landing", "pre_release", "unknown"):
            reasons.append("measurement_stage_unverified")
    # Missing trajectory points or thermal time-series are not silently invented.
    if task == "particle_trajectory" and row.get("trajectory_file_sha256", "") == "":
        reasons.append("missing_trajectory_points")
    # Include positive-finite targets only; keep extreme-but-valid observations.
    return ("pass" if not reasons else "exclude", ";".join(sorted(set(reasons))))


def audit(rows):
    seen_id = set()
    seen_source = set()
    result = []
    for row in rows:
        row = dict(row)
        status, reasons = review(row)
        issues = set(filter(None, reasons.split(";")))
        rid = row.get("record_id", "")
        origin = (row.get("source_id", ""), row.get("source_file_sha256", ""),
                  row.get("source_row_id", ""))
        if rid in seen_id:
            issues.add("duplicate_record_id")
        if origin in seen_source:
            issues.add("duplicate_source_row")
        seen_id.add(rid)
        seen_source.add(origin)
        # Exclusion is non-destructive: all rows remain in an audit table.
        row["qc_status"] = "exclude" if issues else "pass"
        row["qc_reason"] = ";".join(sorted(issues))
        result.append(row)
    return result


def partition(rows, seed="save-the-earth-splits-v1"):
    eligible = [r for r in rows if r["qc_status"] == "pass"]
    if not eligible:
        raise ValueError("No measured QC-eligible records. Do not fabricate data.")
    # Independent groups are scoped per task and source. A source experiment
    # never has rows in more than one split.
    groups = defaultdict(list)
    for row in eligible:
        key = (row["task_family"], row["source_id"],
               row["fire_id"] if row["task_family"] == "wildfire_spotting" else row["experiment_id"])
        groups[key].append(row)
    by_family = defaultdict(list)
    for key in groups:
        by_family[key[0]].append(key)
    assignment = {}
    for task, group_keys in by_family.items():
        if len(group_keys) < 3:
            raise ValueError(f"{task}: require >=3 independent groups to freeze train/validation/test")
        order = sorted(group_keys, key=lambda k: hashlib.sha256(
            (seed + "|" + "|".join(k)).encode()).hexdigest())
        n = len(order)
        n_test = max(1, round(n * .15))
        n_val = max(1, round(n * .15))
        if n_test + n_val >= n:
            n_test = n_val = 1
        for i, key in enumerate(order):
            assignment[key] = "test" if i < n_test else ("validation" if i < n_test + n_val else "train")
    split_rows = []
    for key, members in groups.items():
        for row in members:
            split_rows.append({"record_id": row["record_id"],
                               "task_family": row["task_family"],
                               "source_id": row["source_id"],
                               "group_id": key[2], "split": assignment[key]})
    return sorted(split_rows, key=lambda r: r["record_id"])


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="data/interim")
    p.add_argument("--freeze-split", default=None, help="Write split exactly once; refuses overwrite")
    args = p.parse_args()
    source = Path(args.input)
    rows = audit(load_csv(source))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys([*BASE, *OUTPUT_COLUMNS, *rows[0].keys()])) if rows else list(BASE)
    write_csv(out / "audited.csv", rows, columns)
    good = [r for r in rows if r["qc_status"] == "pass"]
    write_csv(out / "eligible.csv", good, columns)
    write_csv(out / "quarantine.csv", [r for r in rows if r["qc_status"] != "pass"], columns)
    summary = {
        "input_sha256": digest(source), "total": len(rows), "eligible": len(good),
        "quarantined": len(rows)-len(good),
        "by_task": dict(Counter(r.get("task_family", "") for r in good)),
        "reasons": dict(Counter(reason for r in rows for reason in filter(None,r["qc_reason"].split(";"))))
    }
    (out / "qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.freeze_split:
        target = Path(args.freeze_split)
        if target.exists():
            raise FileExistsError(f"Immutable split already exists: {target}")
        splits = partition(rows)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("record_id", "task_family", "source_id", "group_id", "split"))
            writer.writeheader()
            writer.writerows(splits)
        summary["frozen_split_sha256"] = digest(target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
