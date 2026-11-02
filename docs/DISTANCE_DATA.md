# Measured firebrand distance model

## Verified research sources

These are **identified research datasets**, not ingested rows. No experimental CSV has been committed, no trained performance claim has been made, and links do not imply that every required training column has been checked inside the raw files.

| Dataset | Published identifier | Published contents | Rights | Where to inspect |
| --- | --- | --- | --- | --- |
| Wildland/vegetative firebrands | [USFS RDS-2020-0035](https://doi.org/10.2737/RDS-2020-0035) | 9,249 firebrands from 5 vegetation fuels; mass, projected area and flight distance | USFS / Ag Data Commons record reports CC0 | [Ag Data Commons](https://agdatacommons.nal.usda.gov/articles/dataset/Data_for_firebrands_generated_from_selected_vegetative_fuels_Joint_Fire_Science_Program_project_15-1-04-4_/27009073), [Mendeley dataset](https://doi.org/10.17632/3cmtb9n9ph.2) (CC BY 4.0) |
| Structural firebrands | [USFS RDS-2020-0034](https://doi.org/10.2737/RDS-2020-0034) | 50,571 firebrands generated from building components and assemblies; mass, projected area and flight distance | USFS / Ag Data Commons record reports CC0 | [Ag Data Commons](https://agdatacommons.nal.usda.gov/articles/dataset/Data_for_firebrands_generated_from_selected_structural_fuels_Joint_Fire_Science_Program_project_15-1-04-4_/27009070), [Mendeley dataset](https://doi.org/10.17632/x6yww4dnm6.2) (CC BY 4.0) |

**Sources:** USFS Research Data Archive catalog (2020 datasets), Ag Data Commons dataset record, NLM dataset catalog; Hedayati et al. (2019), [A Framework to Facilitate Firebrand Characterization](https://doi.org/10.3389/fmech.2019.00043).

### Observational limitations

These experiments measured dry mass after collection and flight distance from a burning source to the center of a **water-collection pan**. Distance can therefore be binned/coarsened to pan centers, not a precise landing coordinate. They do not directly provide complete per-particle height/time/3D path. Wind may be described per experimental run, not per particle. Verify sampling, individual file column names, test/run identities, and units against the data dictionary before creating the normalized table. **Never guess a wind value or fabricate experiment IDs.**

Records from two repositories may duplicate the **same** underlying experiment. Do not concatenate them without deduplication or split duplicates between train and test.

## Required normalized CSV columns

```csv
sample_id,experiment_id,source_id,data_kind,mass_kg,area_m2,wind_speed_m_s,fuel_type,flight_distance_m
```

- One row per *measured* firebrand. Units strictly SI. Convert grams to kg (/1000), cm² to m² (/10000), when confirmed by the original data dictionary.
- `experiment_id` must correspond to a **genuinely independent test or session**, not an arbitrary row index; all particles in one run share the same group.
- `wind_speed_m_s` must be matched to the correct test/run from documented metadata. Missing values stay missing in the raw ledger and cannot enter this strict baseline.
- Preserve original source URLs, licensing, missing data, original file checksums and transformation scripts before publishing a derived dataset.
- Fuel-type labels must be recorded as observed; unknown fuel type is not an acceptable guessed category.
- For pan-center distance observations, record the pan's size and mapping as extra metadata: the current model does not yet account for interval-censored landing distance.
- Be alert to survivorship/collection bias, collection-pan coverage limits, combustion mass loss, varying wind/vertical flow and within-run correlation.

## Train

```bash
python -m pip install numpy matplotlib
python -m unittest discover -s tests
python train_distance.py --data data/processed/measured_firebrands.csv
```

The pipeline writes `distance_results/evaluation.json`, `heldout_predictions.csv`, `distance_model.npz`. Results compare a **training-only median** constant-distance baseline against ridge regression on unseen experiment groups. The train/test split is group-based, with feature standardization from training rows only; no hyperparameter tuning on test. For scientific claims, additionally hold out campaigns or test facilities, assess errors separately by wind/fuel regime and quantify pan discretization uncertainty. Do not report interpolation in a wind tunnel as a field-scale spotting prediction.

## Why a second training script?

`train_ember.py` requires measured x/y impact coordinates and flight times, which these sources may not supply. `train_distance.py` works with published *flight distance* as a single scalar target, not 3D trajectories and not ignition or wildfire spread.

If the data do not contain **documented per-test wind speeds or genuine experiment grouping**, the supervised distance model cannot run credibly as written. In that case, first fit descriptive distribution models using available measured mass/area/distance with honest limits, and request run-level metadata from the authors.

## Next actions

1. Download the vegetative archive through its source page. Inspect data dictionaries and raw headers. Document SHA-256, row counts, units, experiment/run IDs, missingness and license.
2. Map only verified raw fields into the normalized CSV, keeping a reproducible ETL script and source references.
3. Run group-held-out evaluation; compare against the median baseline and quantify pan-position error.
4. Repeat on structural firebrand experiments as a **different domain**, not just random mixed rows.
5. Request per-test wind time series, release/source geometry, individual particle observations and trajectory/flight-time benchmarks if unavailable.
