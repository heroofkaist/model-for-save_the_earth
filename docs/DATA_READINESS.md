# Save the Earth — observational data readiness gate

**Status as of 2026-11-02: data ingestion NOT completed.**
The attached 22-page Deep Research PDF is a *research report*, **not a raw observational dataset**. Do not derive a 59,820-row file from its abstract or claim measured model accuracy.

## Tasks are independent

| Task | Source | Label | Status |
|---|---|---|---|
| Particle distance | USDA vegetative + structural | water-pan flight distance | Raw files not fetched or verified |
| Particle trajectory | CSIRO/NIST when permission granted | time-resolved 3D coordinates | No public raw trajectories verified |
| Thermal | Zenodo 1406207 | experiment-level thermal response/heat flux | Archive not downloaded |
| Wildfire spotting | Storey et al. 2020 supplementary tables | source-fire -> secondary spot-fire observations | Raw supplement not downloaded |
| Flux | Zenodo 4837617 | acoustic firebrand arrival events | Raw archive not downloaded |
| Korean context | KFS/KMA | incident or weather fields | Not particle-level labels |

The official metadata registry is `data/manifests/source_registry.json`, containing declared published counts but **no fabricated measured records**.

## Required immutable files before training

1. Source metadata including DOI, official URI, licence at download time and original file list.
2. Raw source bytes kept out of ordinary Git with SHA-256 recorded (never overwrite).
3. Source-specific parsing map with **confirmed** column names, units and experiment group identity.
4. A canonical measured-only CSV with all `research_data.BASE` fields and a per-task target.
5. The QC audit, eligible rows and non-destructive quarantine from `research_data.py`.
6. Frozen `data/manifests/split_v1.csv` generated ONCE after confirming independent groups. No split is currently present because no legitimate rows exist.

## Local commands

```bash
python download_sources.py --dataset both --metadata-only
python download_sources.py --dataset both
# inspect actual source files, dictionaries and grouping BEFORE any mapping
python research_data.py --input data/processed/canonical.csv --outdir data/interim --freeze-split data/manifests/split_v1.csv
python -m unittest discover -s tests -v
```

The CLI creates audited, eligible and quarantine CSVs, reason-code summaries, and (only if >=3 independent groups per present task) a frozen group-disjoint split. **It does not silently train**; this is intentional.

### Scientific restrictions

- USDA mass and projected area were obtained after landing and drying. They are retrospective covariates, not release-time inputs.
- Collection-pan distances have spatial coarsening; source report gives approximate uncertainty values. Preserve detailed measurement definitions.
- Distances from wildfires to spot fires involve survival, ignition and detection, and are not individual-particle landings.
- Station meteorology cannot substitute without qualification for turbulent/plume winds near a fire.
- Extreme real distances should be flagged, not automatically deleted. Provenance should be preserved even for rejected records.
- Prospective models should use only pre-release or contemporaneous *observable* inputs; do not make operational claims from post-flight measurements.

## Minimum gate for technical report

Raw file hashes, published licences, verified column/unit dictionary, group identities, frozen split manifest, independent held-out results, source-specific error analysis and clear separation of particle / thermal / ignition / spotting results. Until then **measured test MAE/RMSE: not yet evaluated**.

Research baseline: [USFS 0035](https://doi.org/10.2737/RDS-2020-0035), [USFS 0034](https://doi.org/10.2737/RDS-2020-0034), [Australian spotting paper](https://doi.org/10.3390/fire3020010), [CSIRO 3D work](https://doi.org/10.1016/j.jaecs.2026.100520), [Zenodo thermal](https://doi.org/10.5281/zenodo.1406207).
