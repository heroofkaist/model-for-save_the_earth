"""Audit downloaded research files without guessing their schema or training labels.

Supports CSV, TSV, XLSX (openpyxl optional), ZIP directory listing.
Does not extract ZIP archives, infer units, or modify source bytes.
"""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

MAX_PREVIEW = 5
MAX_ZIP_FILES = 2000

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def csv_summary(path, delimiter=","):
    with path.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.reader(f,delimiter=delimiter)
        header=next(reader,[])
        preview=[]
        n=0
        for record in reader:
            n+=1
            if len(preview)<MAX_PREVIEW:
                preview.append(record[:len(header)])
    return {"columns":header,"records":n,"example_rows":preview}

def xlsx_summary(path):
    try:
        from openpyxl import load_workbook
    except ImportError:
        return {"error":"openpyxl not installed; run pip install openpyxl"}
    book=load_workbook(path,read_only=True,data_only=True)
    sheets=[]
    try:
        for sheet in book:
            iterator=sheet.iter_rows(values_only=True)
            head=next(iterator,())
            sheets.append({"sheet":sheet.title,
                           "columns":[str(v) if v is not None else "" for v in head],
                           "reported_row_count":sheet.max_row,
                           "warning":"Sheet dimension may be overstated; row counts require validation."})
    finally:
        book.close()
    return {"sheets":sheets}

def zip_summary(path):
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        if len(infos)>MAX_ZIP_FILES:
            raise ValueError("Too many ZIP entries; manual inspection required")
        return {"entries":[{"name":i.filename,"bytes_uncompressed":i.file_size,
                 "bytes_compressed":i.compress_size} for i in infos],
                "warning":"Listing only: ZIP not extracted or trusted."}

def inspect(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result={"filename":path.name,"size_bytes":path.stat().st_size,
            "sha256":sha256(path),"extension":path.suffix.lower()}
    suffix=path.suffix.lower()
    if suffix==".csv":
        result.update(csv_summary(path))
    elif suffix==".tsv":
        result.update(csv_summary(path,"\t"))
    elif suffix==".xlsx":
        result.update(xlsx_summary(path))
    elif suffix==".zip":
        result.update(zip_summary(path))
    else:
        result["warning"]="Binary or unsupported type: source preserved; no schema inferred."
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",help="Raw file or directory")
    parser.add_argument("--output",default="data/interim/raw_audit.json")
    args=parser.parse_args()
    target=Path(args.path)
    files=sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]
    report={"files":[inspect(p) for p in files if not p.name.endswith(".partial")]}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError("Audit exists; preserve version and choose a new output file")
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(json.dumps({"files_audited":len(report["files"]),"report":str(out)},indent=2))
