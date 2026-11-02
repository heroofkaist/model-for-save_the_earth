"""Measured firebrand flight-distance baseline. No synthetic labels.

Input CSV (SI): sample_id, experiment_id, source_id, data_kind, mass_kg,
area_m2, wind_speed_m_s, fuel_type, flight_distance_m.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

REQUIRED = ("sample_id","experiment_id","source_id","data_kind","mass_kg",
            "area_m2","wind_speed_m_s","fuel_type","flight_distance_m")

def read_data(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if len(rows) < 12:
        raise ValueError("At least 12 measured rows required")
    ids = set()
    for n,r in enumerate(rows,2):
        for key in ("sample_id","experiment_id","source_id","fuel_type"):
            if not (r[key] or "").strip():
                raise ValueError(f"Row {n}: missing {key}")
        if r["sample_id"] in ids:
            raise ValueError(f"Row {n}: duplicate sample_id")
        ids.add(r["sample_id"])
        if r["data_kind"].strip().lower() != "measured":
            raise ValueError(f"Row {n}: refusing synthetic labels")
        for key in ("mass_kg","area_m2","wind_speed_m_s","flight_distance_m"):
            try:
                value = float(r[key])
            except (ValueError,TypeError):
                raise ValueError(f"Row {n}: invalid {key}") from None
            if not np.isfinite(value) or (key in ("mass_kg","area_m2") and value <= 0) or (key not in ("mass_kg","area_m2") and value < 0):
                raise ValueError(f"Row {n}: invalid {key}")
    return rows

def grouped_split(rows, fraction, seed):
    groups = np.array([r["experiment_id"] for r in rows])
    unique = np.unique(groups)
    if len(unique) < 4:
        raise ValueError("Need at least four independent experimental groups")
    rng = np.random.default_rng(seed)
    selection = rng.permutation(unique)
    count = min(max(1,int(round(len(unique)*fraction))),len(unique)-2)
    held = set(selection[:count])
    test = np.array([g in held for g in groups])
    return ~test,test,sorted(held)

def features(rows, levels):
    mass = np.array([float(r["mass_kg"]) for r in rows])
    area = np.array([float(r["area_m2"]) for r in rows])
    wind = np.array([float(r["wind_speed_m_s"]) for r in rows])
    continuous = np.column_stack((np.log(mass),np.log(area),np.log(area/mass),
                                  wind,wind**2,wind*np.log(area/mass)))
    category = np.column_stack([
        np.array([float(r["fuel_type"] == fuel) for r in rows])
        for fuel in levels]) if levels else np.empty((len(rows),0))
    return np.column_stack((continuous,category))

def scores(y,p):
    e = p-y
    return {"n":len(y),"mae_m":float(np.mean(np.abs(e))),
            "rmse_m":float(np.sqrt(np.mean(e**2))),
            "bias_m":float(np.mean(e))}

def train(rows, fraction=0.25, seed=42, ridge=10):
    tr,te,held = grouped_split(rows,fraction,seed)
    levels = sorted({r["fuel_type"] for i,r in enumerate(rows) if tr[i]})
    x = features(rows,levels)
    mu,scale = x[tr].mean(axis=0),x[tr].std(axis=0)
    scale[scale < 1e-10] = 1
    x = np.column_stack((np.ones(len(rows)),(x-mu)/scale))
    y = np.array([float(r["flight_distance_m"]) for r in rows])
    median = float(np.median(y[tr]))
    reg = np.eye(x.shape[1])
    reg[0,0] = 0
    coef = np.linalg.solve(x[tr].T@x[tr]+ridge*reg,x[tr].T@y[tr])
    predictions = np.maximum(0,x@coef)
    report = {
        "status":"MEASURED_DISTANCE_ONLY_HELD_OUT",
        "train_size":int(tr.sum()),"test_size":int(te.sum()),
        "held_out_experiments":held,
        "sources":sorted({r["source_id"] for r in rows}),
        "train_only_median_baseline":scores(y[te],np.full(te.sum(),median)),
        "ridge_regression":scores(y[te],predictions[te]),
        "limitations":"Grouped split depends on authentic independent experiment IDs. No trajectory, flight time, ignition, or fire spread prediction."
    }
    model = {"coef":coef,"mean":mu,"scale":scale,"fuel_levels":levels}
    heldout = [(r,float(y[i]),float(predictions[i])) for i,r in enumerate(rows) if te[i]]
    return report,model,heldout

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--outdir",default="distance_results")
    parser.add_argument("--test-fraction",type=float,default=0.25)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--ridge",type=float,default=10)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.ridge <= 0:
        parser.error("test-fraction must be (0,1) and ridge must be positive")
    report,model,heldout = train(read_data(args.data),args.test_fraction,args.seed,args.ridge)
    out = Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez(out/"distance_model.npz",coef=model["coef"],mean=model["mean"],
             scale=model["scale"],fuel_levels=np.array(model["fuel_levels"]))
    with (out/"heldout_predictions.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["sample_id","experiment_id","source_id","observed_distance_m","predicted_distance_m"])
        for r,y,p in heldout:
            writer.writerow([r["sample_id"],r["experiment_id"],r["source_id"],y,p])
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
