"""Train descriptive distance baseline with a pre-frozen research_data.py split.

Does NOT use observed post-landing mass/area as prospective features. Only fuel
and documented wind are used. If source wind is unavailable, do not impute it:
a fuel-only analysis requires a separately specified valid design.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def read(path):
    with open(path,newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def evaluate(data,split):
    records=read(data)
    splits=read(split)
    by_id={}
    for s in splits:
        if s["record_id"] in by_id: raise ValueError("Duplicate split record ID")
        if s["split"] not in ("train","validation","test"): raise ValueError("Unknown split")
        by_id[s["record_id"]]=s
    rows=[]
    for r in records:
        if r.get("qc_status")!="pass" or r.get("task_family")!="particle_distance": continue
        s=by_id.get(r["record_id"])
        if s is None: raise ValueError("Missing frozen split for eligible record")
        if (s["task_family"],s["source_id"],s["group_id"]) != (
            r["task_family"],r["source_id"],r["experiment_id"]):
            raise ValueError("Split metadata does not match record")
        try:
            target=float(r["landing_distance_m"])
        except (ValueError,TypeError):
            raise ValueError("Missing measured distance label") from None
        if not np.isfinite(target) or target<0: raise ValueError("Invalid label")
        if r.get("data_kind")!="measured": raise ValueError("Synthetic data refused")
        rows.append((r,s["split"],target))
    if len(set(r["record_id"] for r,_,_ in rows))!=len(rows):
        raise ValueError("Duplicate canonical row IDs")
    if {k for _,k,_ in rows}!={"train","validation","test"}:
        raise ValueError("All 3 partitions required")
    groups={}
    for r,k,_ in rows:
        group=(r["source_id"],r["experiment_id"])
        groups.setdefault(group,set()).add(k)
    if any(len(v)>1 for v in groups.values()):
        raise ValueError("Experiment group leakage")
    train=[(r,y) for r,k,y in rows if k=="train"]
    # A training-only median baseline is valid without invented wind values.
    median=float(np.median([y for _,y in train]))
    def stats(part):
        arr=np.asarray([y for _,k,y in rows if k==part],dtype=float)
        err=np.full_like(arr,median)-arr
        return {"n":int(len(arr)),"mae_m":float(np.abs(err).mean()),
                "rmse_m":float(np.sqrt(np.mean(err**2))),
                "median_absolute_error_m":float(np.median(np.abs(err))),
                "bias_m":float(err.mean())}
    return {"status":"OBSERVED_DISTANCE_BASELINE_ONLY",
            "predictor":"training_partition_median",
            "train":stats("train"),"validation":stats("validation"),"test":stats("test"),
            "sources":sorted({r["source_id"] for r,_,_ in rows}),
            "warning":"Retrospective descriptive task, not fire spread or ignition prediction; test must remain untouched for future model selection."}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True)
    p.add_argument("--split",required=True)
    p.add_argument("--output",default="distance_results/locked_baseline.json")
    a=p.parse_args()
    report=evaluate(a.data,a.split)
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError("Refuse to overwrite immutable benchmark report")
    with path.open("x",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
