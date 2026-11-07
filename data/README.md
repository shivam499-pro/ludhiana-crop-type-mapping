# Data

This document describes data referenced by the P7 workflow. The training,
validation, and rebuilt prediction CSVs listed below are intentionally excluded
from public distribution and are not included in this repository. Do not treat
the inventory or metadata here as a claim that those datasets are distributed.

## Data referenced by the workflow

| File | Records | Description / public availability |
|---|---:|---|
| `P7_training_points.csv` | 608 | Rebuilt workflow's training records, eight predictors, and `class` label. Intentionally excluded from public distribution. |
| `P7_validation_points.csv` | 292 | Held-out records and reference labels used by the rebuilt GEE workflow. Intentionally excluded from public distribution. |
| `P7_validation_predictions_rebuilt.csv` | 292 | Rebuilt model's held-out `true_class` and `predicted_class`, plus predictors. Intentionally excluded from public distribution. |
| `P7_validation_results.csv` | 261 | Historical prediction/reference records from the earlier workflow. Intentionally excluded from public distribution; not the rebuilt evaluation input. |
| `P7_area_stats.csv` | 3 | Existing class-area estimates in km². Not recalculated from the rebuilt raster. |

## Rebuilt training and validation tables

The previously analyzed training and validation tables had these eight
predictors, plus the integer class property `class` (1, 2, or 3):

| Predictor | Description |
|---|---|
| `B11_kharif` | Kharif seasonal composite, Sentinel-2 SWIR B11 (native 20 m). |
| `B4_kharif` | Kharif seasonal composite, Sentinel-2 red B4 (native 10 m). |
| `B8_kharif` | Kharif seasonal composite, Sentinel-2 NIR B8 (native 10 m). |
| `ndvi_delta` | Kharif peak NDVI minus Rabi peak NDVI. |
| `ndvi_kharif_mean` | Mean monthly NDVI for Kharif. |
| `ndvi_kharif_peak` | Maximum monthly NDVI for Kharif. |
| `ndvi_rabi_mean` | Mean monthly NDVI for Rabi. |
| `ndvi_rabi_peak` | Maximum monthly NDVI for Rabi. |

| Class value | Class |
|---:|---|
| 1 | Double Cropping |
| 2 | Single Cropping |
| 3 | Fallow / Non-crop |

Previously reported table counts and class distributions:

| Dataset | Total | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| Training | 608 | 202 | 201 | 205 |
| Held-out validation | 292 | 98 | 99 | 95 |

Previously analyzed local copies of the training and validation tables were
reported to have unique `system:index` values, no shared IDs, no duplicate
records, and no missing or non-finite predictor values. The raw sample-level
records and dataset fingerprints are not published.

## Provenance and limitations

The original
[`GEE_Script03_Phenology_Labels.js`](../scripts/GEE_Script03_Phenology_Labels.js)
creates stratified samples, splits them into training and validation sets, and
extracts features with `sampleRegions()` at 30 m. It also defines exports of
the feature-enriched tables. Users need authorized training and validation
tables to run the rebuilt workflow and must upload or copy them to Earth Engine
assets they can access, then configure the project and asset paths in the
rebuilt script. The tables are not supplied in this public repository. Copies
associated with the workflow had empty `.geo` fields with no usable
coordinates, so exact pixel-level correspondence to the original 30 m sample
locations cannot be verified from those copies.

The previously analyzed 608-row training and 292-row validation tables are
distinct from the 261-row historical `P7_validation_results.csv`, which is
intentionally excluded from public distribution. The rebuilt evaluation used
`P7_validation_predictions_rebuilt.csv`, generated from held-out predictions;
that CSV is also intentionally excluded from public distribution. Do not use
the historical file as the rebuilt model's result.

Validation labels are derived from NDVI phenology rules, not independent field
observations. The previously analyzed tables have no usable geographic
coordinates. Separate IDs establish table-level separation only, not spatially
independent validation.

`P7_area_stats.csv` contains area values from the earlier workflow. Those
values have not been recalculated from the rebuilt classified raster.
