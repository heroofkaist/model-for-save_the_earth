"""Download open USDA firebrand research archives with provenance, no silent extraction.

Fetches public Figshare article file metadata, verifies hashes if reported, and
records all downloads. DOES NOT assert that records are training-ready.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ARTICLES = {"vegetative": 27009073, "structural": 27009070}
BASE = "https://api.figshare.com/v2/articles"


def get_json(url):
    req = Request(url, headers={"User-Agent": "save-the-earth-ember-research/0.1"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def fetch_file(url, target, expected_md5=None, max_bytes=2_000_000_000):
    temp = target.with_name(target.name + ".partial")
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    count = 0
    req = Request(url, headers={"User-Agent": "save-the-earth-ember-research/0.1"})
    try:
        with urlopen(req, timeout=120) as stream, temp.open("wb") as f:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("File exceeds configured safety limit")
                md5.update(chunk)
                sha256.update(chunk)
                f.write(chunk)
        if expected_md5 and md5.hexdigest().lower() != expected_md5.lower():
            raise ValueError(f"Checksum mismatch: {target.name}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return {"bytes": count, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download(article_id, dest, metadata_only=False):
    details = get_json(f"{BASE}/{article_id}")
    directory = Path(dest) / str(article_id)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "article_id": article_id,
        "title": details.get("title"),
        "doi": details.get("doi"),
        "source_url": details.get("url_public_html"),
        "license": details.get("license"),
        "description": details.get("description"),
        "files": []
    }
    for entry in details.get("files", []):
        filename = Path(entry["name"]).name
        if not filename or filename in (".", ".."):
            raise ValueError("Invalid filename")
        target = directory / filename
        record = {
            "name": filename,
            "source_file_id": entry.get("id"),
            "source_url": entry.get("download_url"),
            "published_bytes": entry.get("size"),
            "published_md5": entry.get("supplied_md5") or entry.get("computed_md5")
        }
        if not metadata_only:
            if target.exists():
                record["status"] = "exists_not_revalidated"
            elif record["source_url"]:
                record.update(fetch_file(record["source_url"], target,
                                         expected_md5=record["published_md5"]))
                record["status"] = "downloaded"
            else:
                record["status"] = "no_download_link"
        manifest["files"].append(record)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"Saved {path} ({len(manifest['files'])} listed files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*ARTICLES, "both"), default="vegetative")
    parser.add_argument("--outdir", default="data/raw")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    chosen = ARTICLES.values() if args.dataset == "both" else [ARTICLES[args.dataset]]
    for article_id in chosen:
        download(article_id, args.outdir, args.metadata_only)
