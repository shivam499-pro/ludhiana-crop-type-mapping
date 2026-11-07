# Scripts

This directory contains the original Google Earth Engine (GEE) workflow, a
separate rebuilt classification workflow, and the Python reporting script.

## GEE scripts

| File | Purpose | Current status |
|---|---|---|
| `GEE_Script01_AOI_Validation.js` | AOI and Sentinel-2 checks/visualization. | Source code present; execution not independently verified here. |
| `GEE_Script02_NDVI_TimeSeries.js` | Seasonal NDVI time-series and raster outputs. | Source code present; execution not independently verified here. |
| `GEE_Script03_Phenology_Labels.js` | Monthly NDVI features, phenology labels, sampling, and feature-table exports. | Source code present; sample extraction uses 30 m. |
| `GEE_Script04_RF_Classification.js` | Original RF, validation, area, and raster exports. | Preserved original; byte-for-byte identical to original Script 05. |
| `GEE_Script05_Validation_Stats.js` | Original validation/statistics workflow. | Preserved original; byte-for-byte identical to original Script 04. |
| `GEE_Script04_RF_Classification_REBUILT.js` | Separate RF workflow using authorized training/validation table assets. | Rebuilt workflow; configure your own assets; source CSVs are excluded from public distribution. |

The original scripts are retained. The rebuilt Script 04 is a separate file,
not a replacement for the originals. Do not infer successful Earth Engine
execution solely from source code being present.

The rebuilt model uses a 50-tree Random Forest with seed 42. Its evaluation is
against phenology-derived reference labels, not field observations. The rebuilt
script does not calculate area statistics.

### Rebuilt Earth Engine workflow prerequisites

The training, validation, and rebuilt prediction CSVs are intentionally
excluded from public distribution and are not included in this repository.
Before running the rebuilt GEE script, obtain authorized training and
validation tables, upload or copy them as table assets to an Earth Engine
project you can access, then replace the clearly marked project placeholder in
`GEE_Script04_RF_Classification_REBUILT.js`. Keep the asset names
`P7_training_points` and `P7_validation_points`, or update the configured
asset paths to match your own names. The project and assets must be accessible
to the account running the script.

## Python accuracy assessment

From the project root:

```bash
python scripts/accuracy_assessment.py
```

Install the packages listed in the root `requirements.txt` first. The script
uses paths relative to its own location and requires:

- An authorized local copy of `data/P7_validation_predictions_rebuilt.csv`
  for the rebuilt model's held-out predictions and reference labels. This CSV
  is intentionally excluded from public distribution.
- `data/P7_area_stats.csv` for the existing class-area table.

It generates `outputs/accuracy_report.txt`, `confusion_matrix.png`,
`ndvi_timeseries.png`, `area_barchart.png`, and `area_summary.csv`. Running it
overwrites those generated outputs.

The current rebuilt validation results are 292 records, 99.32% overall
agreement, and Cohen's Kappa 0.9897. The confusion matrix (rows = true,
columns = predicted; classes 1–3) is:

```text
[[98, 0, 0],
 [ 0,99, 0],
 [ 0, 2,93]]
```

These metrics measure agreement with phenology-derived labels and are not
independent field or real-world accuracy estimates. The CSVs have no usable
coordinates, so spatial independence cannot be established. The older
`data/P7_validation_results.csv` contains 261 historical records and is not the
input to the rebuilt evaluation.

Area chart values continue to come from `data/P7_area_stats.csv`; the Python
script does not recalculate them from the rebuilt classified raster. The NDVI
chart uses 11 console-transcribed monthly values unless the optional
`data/monthly_ndvi.csv` is supplied.

See the root [README](../README.md), [dataset notes](../data/README.md), and
[output notes](../outputs/README.md) for the full workflow context.
