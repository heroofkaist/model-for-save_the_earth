"""Normalize observed firebrand CSV columns using a documented JSON mapping.

Example:
python normalize_firebrands.py --input raw.csv --mapping mapping.json --output data/processed/measured_firebrands.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path
from train_distance import REQUIRED, read_data

NUMERIC = ("mass_kg", "area_m2", "wind_speed_m_s", "flight_distance_m")

def normalize(raw_path, mapping_path, destination):
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    fields = mapping.get("fields", {})
    if set(fields) != set(REQUIRED):
        raise ValueError("Mapping must provide each required normalized field")
    for field,spec in fields.items():
        if not isinstance(spec,dict) or ("column" in spec) == ("value" in spec):
            raise ValueError(f"{field}: provide exactly one of column or value")
        if field in ("sample_id","experiment_id") and "value" in spec:
            raise ValueError(f"{field} must be a genuine raw ID, not an invented constant")
        if field=="wind_speed_m_s" and "value" in spec and not mapping.get("constant_wind_documentation"):
            raise ValueError("Constant wind requires a reference to the documented wind condition")
    with open(raw_path,newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        missing={s["column"] for s in fields.values() if "column" in s}-set(reader.fieldnames or [])
        if missing:
            raise ValueError("Unrecognized raw headers: "+", ".join(sorted(missing)))
        records=[]
        for line,row in enumerate(reader,2):
            clean={}
            for key,spec in fields.items():
                value=row.get(spec["column"]) if "column" in spec else spec["value"]
                if value is None or str(value).strip()=="":
                    raise ValueError(f"Line {line}: missing {key}")
                if key in NUMERIC:
                    scale=spec.get("scale",1)
                    if not isinstance(scale,(int,float)) or scale<=0:
                        raise ValueError(f"{key}: scale must be positive")
                    value=float(value)*scale
                    if not math.isfinite(value):
                        raise ValueError(f"Line {line}: invalid {key}")
                clean[key]=value
            records.append(clean)
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    with dest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(records)
    try:
        read_data(dest)
    except Exception:
        dest.unlink()
        raise
    return len(records)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--mapping",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(f"Validated {normalize(args.input,args.mapping,args.output)} observations")
