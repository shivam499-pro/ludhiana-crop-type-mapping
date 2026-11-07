# AI-Assisted Crop Type and Cropping Pattern Mapping

**Study area:** Ludhiana District, Punjab, India
**Study period:** Kharif 2024 and Rabi 2024–25 (11 monthly periods)
**Methods:** Sentinel-2 SR Harmonized, NDVI phenology, Random Forest, Google Earth Engine, Python

This project maps three cropping-pattern classes in Ludhiana District: double
cropping, single cropping, and fallow/non-crop. The classification is based on
seasonal NDVI phenology and Sentinel-2 reflectance features. The documented
workflow includes a separate rebuilt Random Forest script, held-out table
evaluation, existing area statistics, and Python-generated reports and figures.

> **Interpretation:** Reference labels are phenology-derived, not independent
> field observations. Reported validation metrics measure agreement with those
> labels. They are not independent field or real-world crop-map accuracy
> estimates, and the coordinate-free sample tables cannot establish spatial
> independence.

## Current validation results

The rebuilt model uses **608 training samples**, a **50-tree Random Forest**
with seed **42**, and **292 held-out validation records** from separate tables.
The evaluation is against phenology-derived labels.

| Metric | Result |
|---|---:|
| Held-out records | 292 |
| Overall agreement | 99.32% |
| Cohen's Kappa | 0.9897 |

Confusion matrix (rows = reference class; columns = predicted class; order
1, 2, 3):

| | Double Cropping (1) | Single Cropping (2) | Fallow / Non-crop (3) |
|---|---:|---:|---:|
| **Double Cropping (1)** | 98 | 0 | 0 |
| **Single Cropping (2)** | 0 | 99 | 0 |
| **Fallow / Non-crop (3)** | 0 | 2 | 93 |

The three workflow CSVs—`P7_training_points.csv`,
`P7_validation_points.csv`, and `P7_validation_predictions_rebuilt.csv`—are
intentionally excluded from public distribution and are not included in this
repository. The metrics above are retained from the existing rebuilt-model
evaluation; rerunning the Python report requires an authorized local copy of
the rebuilt prediction table. The metrics measure agreement with derived
reference labels, not independent field accuracy. The older
`data/P7_validation_results.csv` contains 261 historical records and is not
the rebuilt model's evaluation dataset.

## Classes

| Value | Class | Description |
|---:|---|---|
| 1 | Double Cropping | Two detected seasonal crop cycles, typically rice followed by wheat. |
| 2 | Single Cropping | One detected seasonal crop cycle. |
| 3 | Fallow / Non-crop | Fallow or other non-crop cover under the phenology rules. |

Zero is **not** a modeled class. The rebuilt classified GeoTIFF export is
configured for **10 m in EPSG:32643**. A separately inspected exported GeoTIFF
contained zero-valued pixels and had no explicit GDAL NoData tag; zero-valued
pixels must not be interpreted as a fourth class. That GeoTIFF is not included
in this repository, so its metadata cannot be independently verified from the
repository files.

## Study period and data

The monthly series covers June–October 2024 (five Kharif periods) and
November 2024–April 2025 (six Rabi periods), for **11 monthly periods**.
Script 03's feature definitions use monthly NDVI peaks and means, a Kharif
seasonal reflectance composite, and the difference between seasonal NDVI peaks.

Sentinel-2 B4 (red) and B8 (NIR) have native 10 m resolution; B11 (SWIR) has
native 20 m resolution. The feature stack mixes these native resolutions. Its
export grid or the classification export scale does not mean every predictor
has native 10 m detail. Script 03 samples training and validation features at
30 m and exports its label raster and feature stack at 10 m. The rebuilt
classification export is configured at 10 m; its projection/resampling
behavior should not be confused with the original 30 m sampling scale.

The training and validation records have no usable geographic coordinates.
Training/validation records are separated in their tables, but spatial
independence cannot be confirmed. Obtain authorized copies of the tables before
reproducing the workflow; they are not publicly distributed here.

## Area statistics

The reported area table remains sourced from the existing
[`data/P7_area_stats.csv`](data/P7_area_stats.csv), not recalculated from the
rebuilt classification raster.

| Class | Area (km²) | Share of listed classes |
|---|---:|---:|
| Double Cropping | 2,950.90 | 82.67% |
| Single Cropping | 220.69 | 6.18% |
| Fallow / Non-crop | 398.02 | 11.15% |
| **Total in area CSV** | **3,569.61** | **100.00%** |

Do not treat these values as area statistics derived from the rebuilt raster.
Reconcile them against that raster and its zero/NoData handling before making
such a claim.

## Workflow and scripts

The five original GEE scripts are preserved. Scripts 01–03 contain AOI/data,
NDVI, phenology, and sampling workflows. Original Script 04 and Script 05 are
byte-for-byte identical; their duplication is documented rather than silently
treated as two distinct implementations. The separate
[`GEE_Script04_RF_Classification_REBUILT.js`](scripts/GEE_Script04_RF_Classification_REBUILT.js)
uses authorized training and validation table assets configured by each user,
fits the rebuilt classifier, evaluates held-out predictions, and defines
exports. It does not calculate area statistics. Before running it, configure
the asset paths for your own accessible Earth Engine project; see
[`scripts/README.md`](scripts/README.md).

| Step | Artifact | Role |
|---|---|---|
| AOI and imagery | `scripts/GEE_Script01_AOI_Validation.js` | AOI and Sentinel-2 checks/visualization. |
| Seasonal NDVI | `scripts/GEE_Script02_NDVI_TimeSeries.js` | Seasonal NDVI processing and outputs. |
| Labels and samples | `scripts/GEE_Script03_Phenology_Labels.js` | Phenology labels, 30 m sample extraction, and feature-table exports. |
| Original classification | `scripts/GEE_Script04_RF_Classification.js` | Preserved original script; byte-identical to original Script 05. |
| Original validation/statistics | `scripts/GEE_Script05_Validation_Stats.js` | Preserved original script; byte-identical to original Script 04. |
| Rebuilt classification | `scripts/GEE_Script04_RF_Classification_REBUILT.js` | Separate 608/292 table-based RF workflow and validation exports. |
| Python reporting | `scripts/accuracy_assessment.py` | Reads rebuilt predictions and existing area CSV; generates report/figures. |

See [`scripts/README.md`](scripts/README.md) for execution guidance and
[`data/README.md`](data/README.md) for dataset provenance and limitations.

The training, validation, and rebuilt prediction CSVs are intentionally
excluded from public distribution. Users need authorized training and
validation tables and must configure their own accessible Earth Engine table
assets before running the rebuilt workflow.

## Outputs

The current `outputs/` directory contains the generated accuracy report,
confusion matrix, NDVI time-series chart, area chart, and area summary. The
classification map PNG and GeoTIFFs in the project are existing outputs; do not
assume they were generated by the rebuilt workflow. The rebuilt GeoTIFF export
is not currently present in the repository.

The NDVI chart uses 11 monthly values transcribed from GEE console output unless
an optional `data/monthly_ndvi.csv` is supplied. No monthly NDVI CSV is currently
listed among the local datasets. See [`outputs/README.md`](outputs/README.md).

## Run the Python report

From the project root, install the packages listed in `requirements.txt` and
run:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python scripts/accuracy_assessment.py
```

The script expects a locally available authorized copy of
`data/P7_validation_predictions_rebuilt.csv` (intentionally excluded from
public distribution) and reads `data/P7_area_stats.csv`. It writes
`accuracy_report.txt`,
`confusion_matrix.png`, `ndvi_timeseries.png`, `area_barchart.png`, and
`area_summary.csv` under `outputs/`. Running it regenerates those files; it
does not run Earth Engine or recalculate area from a raster.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── assets/
│   └── p7_workflow.png
├── data/
│   ├── P7_validation_results.csv          # historical, 261 records
│   ├── P7_area_stats.csv
│   └── README.md
├── outputs/
│   ├── maps/                              # existing map PNGs
│   ├── accuracy_report.txt
│   ├── accuracy and area figures
│   ├── area_summary.csv
│   └── README.md
├── report/
│   └── P7_TDD_v1.0.docx
└── scripts/
    ├── GEE_Script01_AOI_Validation.js
    ├── GEE_Script02_NDVI_TimeSeries.js
    ├── GEE_Script03_Phenology_Labels.js
    ├── GEE_Script04_RF_Classification.js  # original
    ├── GEE_Script05_Validation_Stats.js   # byte-identical to original 04
    ├── GEE_Script04_RF_Classification_REBUILT.js
    ├── accuracy_assessment.py
    └── README.md
```

The omitted training, validation, and rebuilt prediction CSVs are not part of
the public repository.

The original submitted report and historical outputs have not been rewritten to
match the rebuilt workflow. Their older results should be distinguished from
the current 292-record evaluation.

## Limitations

- Labels are derived from NDVI phenology rules; no independent field
  observations are available for this evaluation.
- The training and validation records have no usable sample coordinates.
  Disjoint sample IDs do not prove spatial independence. Their CSV tables are
  intentionally excluded from public distribution.
- The 10 m classified export grid does not make B11's native 20 m information
  10 m, nor establish the resampling used in the rebuilt feature stack.
- Area statistics are from the existing area CSV and have not been recalculated
  from the rebuilt raster.
- A separately inspected exported GeoTIFF contained zero-valued pixels without
  an explicit GDAL NoData tag; zero is not one of the three class IDs. That
  GeoTIFF is not included in this repository, so its metadata cannot be
  independently verified from the repository files.
- The monthly NDVI chart is based on console-transcribed values unless a
  monthly source CSV is provided.

## Attribution and licensing

The workflow uses Sentinel-2 Surface Reflectance Harmonized data through Google
Earth Engine. Review the applicable data-provider terms before reuse. No
project software license is specified in the files inspected.
