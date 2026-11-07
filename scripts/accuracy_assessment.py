"""
P7 — Accuracy Assessment and Results Visualization
Ludhiana District Cropping Pattern Mapping

Expected project layout:
    project_root/
        data/
            P7_validation_predictions_rebuilt.csv
            P7_area_stats.csv
            monthly_ndvi.csv       (optional; recommended)
        outputs/
        scripts/
            accuracy_assessment.py

Run from any working directory:
    python scripts/accuracy_assessment.py

The validation metrics describe agreement with phenology-derived labels.
They are not independent field-ground-truth accuracy estimates.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)

# ---------------------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------------------

# This file is expected at <project_root>/scripts/accuracy_assessment.py.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = DATA_DIR / "P7_validation_predictions_rebuilt.csv"
AREA_FILE = DATA_DIR / "P7_area_stats.csv"
MONTHLY_NDVI_FILE = DATA_DIR / "monthly_ndvi.csv"  # optional

CLASS_IDS = [1, 2, 3]
CLASS_NAMES = [
    "Double Cropping",
    "Single Cropping",
    "Fallow / Non-crop",
]
CLASS_COLORS = ["#1a9850", "#fee08b", "#d73027"]
CLASS_NAME_BY_ID = dict(zip(CLASS_IDS, CLASS_NAMES))

# AOI area reported by the user's GEE AOI validation run.
# Keep this explicit and sourced; update only if the AOI changes.
AOI_AREA_KM2 = 3573.1078179393858

# Monthly mean NDVI values transcribed from the user's GEE console output.
# For a fully data-driven chart, place monthly_ndvi.csv in data/ with
# columns: month, mean_ndvi. Example month: 2024-06-15.
GEE_CONSOLE_NDVI = pd.DataFrame({
    "month": pd.to_datetime([
        "2024-06-15", "2024-07-15", "2024-08-15", "2024-09-15",
        "2024-10-15", "2024-11-15", "2024-12-15", "2025-01-15",
        "2025-02-15", "2025-03-15", "2025-04-15",
    ]),
    "mean_ndvi": [
        0.201, 0.313, 0.759, 0.669, 0.464,
        0.206, 0.367, 0.553, 0.678, 0.633, 0.280,
    ],
})

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            "Place the exported CSV in the project's data/ directory."
        )


def load_validation_data(path: Path) -> tuple[pd.DataFrame, int]:
    require_file(path)
    df = pd.read_csv(path)

    required = {"true_class", "predicted_class"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    original_rows = len(df)
    df = df[["true_class", "predicted_class"]].copy()

    # Convert invalid text to NaN so it is counted and reported, not hidden.
    for col in ["true_class", "predicted_class"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_rows = int(df.isna().any(axis=1).sum())
    df = df.dropna().copy()

    if df.empty:
        raise ValueError("No valid validation rows remain after checking missing values.")

    # Class IDs must be integer values and must belong to the expected scheme.
    for col in ["true_class", "predicted_class"]:
        if not np.all(np.isclose(df[col], np.round(df[col]))):
            raise ValueError(f"{col} contains non-integer class IDs.")
        df[col] = df[col].astype(int)

        unexpected = sorted(set(df[col]) - set(CLASS_IDS))
        if unexpected:
            raise ValueError(
                f"{col} contains unexpected class IDs {unexpected}; "
                f"expected only {CLASS_IDS}."
            )

    return df, invalid_rows


def load_area_data(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required = {"class_name", "area_km2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    df = df[["class_name", "area_km2"]].copy()
    df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")

    if df["area_km2"].isna().any():
        raise ValueError("Area CSV contains missing or non-numeric area values.")
    if (df["area_km2"] < 0).any():
        raise ValueError("Area CSV contains negative area values.")

    # Check that each expected class occurs exactly once.
    if df["class_name"].duplicated().any():
        raise ValueError("Area CSV contains duplicate class names.")

    missing_classes = set(CLASS_NAMES) - set(df["class_name"])
    extra_classes = set(df["class_name"]) - set(CLASS_NAMES)
    if missing_classes or extra_classes:
        raise ValueError(
            f"Area class mismatch. Missing: {sorted(missing_classes)}; "
            f"unexpected: {sorted(extra_classes)}."
        )

    return df.set_index("class_name").loc[CLASS_NAMES].reset_index()


# ---------------------------------------------------------------------
# 3. Accuracy assessment
# ---------------------------------------------------------------------

def create_accuracy_outputs(val_df: pd.DataFrame, invalid_rows: int) -> dict:
    y_true = val_df["true_class"].to_numpy()
    y_pred = val_df["predicted_class"].to_numpy()

    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    # Save a transparent text report.
    report_path = OUTPUT_DIR / "accuracy_report.txt"
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("P7 — Accuracy Assessment Report\n")
        f.write("Ludhiana District Cropping Pattern Mapping\n")
        f.write("=" * 62 + "\n\n")
        f.write("Evaluation basis: agreement with phenology-derived labels.\n")
        f.write(
            "This is not an independent field-ground-truth accuracy estimate.\n\n"
        )
        f.write(f"Valid evaluated samples : {len(y_true)}\n")
        f.write(f"Rows with missing/invalid class values excluded: {invalid_rows}\n")
        f.write(f"Overall Accuracy        : {oa:.4f} ({oa * 100:.2f}%)\n")
        f.write(f"Cohen's Kappa           : {kappa:.4f}\n\n")
        f.write("Confusion Matrix (rows=True, columns=Predicted):\n")
        f.write(cm_df.to_string())
        f.write("\n\nClass-wise Classification Report:\n")
        f.write(report_text)
        f.write("\n")
        f.write("Class support:\n")
        for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
            count = int((y_true == class_id).sum())
            f.write(f"  {class_name}: {count}\n")
        f.write("\nMethodological limitation:\n")
        f.write(
            "Reference labels were derived from NDVI phenology thresholds. "
            "The Random Forest feature set includes NDVI-derived variables "
            "that also contribute to label generation. Therefore, these "
            "metrics quantify model agreement with the derived labels and "
            "should not be presented as independent real-world crop-map "
            "accuracy. Independent field/reference data and spatially "
            "separated validation are recommended for that claim.\n"
        )

    # Normalized confusion matrix, robust to any absent class row.
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float) * 100,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Greens",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "Row percentage (%)"},
    )

    for i in range(len(CLASS_IDS)):
        for j in range(len(CLASS_IDS)):
            pct = cm_norm[i, j]
            count = cm[i, j]
            text_color = "white" if pct >= 60 else "#1a1a1a"
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=15, ha="right")
    ax.set_yticklabels(CLASS_NAMES, fontsize=9, rotation=0)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Reference Label Class", fontsize=11, labelpad=10)
    ax.set_title(
        "Confusion Matrix — Ludhiana Cropping Pattern Classification\n"
        f"Agreement = {oa * 100:.2f}% | Kappa = {kappa:.4f} | "
        f"n = {len(y_true)}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "oa": oa,
        "kappa": kappa,
        "cm": cm,
        "cm_df": cm_df,
        "report_dict": report_dict,
        "report_text": report_text,
        "sample_count": len(y_true),
        "invalid_rows": invalid_rows,
    }


# ---------------------------------------------------------------------
# 4. NDVI time-series visualization
# ---------------------------------------------------------------------

def load_monthly_ndvi() -> tuple[pd.DataFrame, str]:
    if MONTHLY_NDVI_FILE.is_file():
        df = pd.read_csv(MONTHLY_NDVI_FILE)
        required = {"month", "mean_ndvi"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"monthly_ndvi.csv is missing columns: {sorted(missing)}"
            )
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["mean_ndvi"] = pd.to_numeric(df["mean_ndvi"], errors="coerce")
        df = df.dropna(subset=["month", "mean_ndvi"]).sort_values("month")
        if df.empty:
            raise ValueError("monthly_ndvi.csv contains no valid monthly records.")
        return df, "Monthly mean NDVI from data/monthly_ndvi.csv"

    return GEE_CONSOLE_NDVI.copy(), (
        "Monthly mean NDVI transcribed from the GEE console; "
        "replace with monthly_ndvi.csv when exported"
    )


def create_ndvi_chart() -> str:
    df, source_note = load_monthly_ndvi()
    df = df.sort_values("month").reset_index(drop=True)

    dates = df["month"]
    values = df["mean_ndvi"].to_numpy()
    x = np.arange(len(df))

    # Seasonal groupings follow the project's stated observation windows.
    kharif_mask = dates.dt.month.isin([6, 7, 8, 9, 10]).to_numpy()
    rabi_mask = dates.dt.month.isin([11, 12, 1, 2, 3, 4]).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    if kharif_mask.any():
        kharif_x = x[kharif_mask]
        ax.axvspan(
            kharif_x.min() - 0.5,
            kharif_x.max() + 0.5,
            alpha=0.08,
            color="#1a9850",
        )
        peak_idx = kharif_x[np.argmax(values[kharif_mask])]
        ax.scatter(
            peak_idx, values[peak_idx], color="#1a9850", s=95, zorder=5
        )
        ax.annotate(
            f"Kharif seasonal maximum\nNDVI = {values[peak_idx]:.3f}",
            xy=(peak_idx, values[peak_idx]),
            xytext=(peak_idx - 0.8, min(0.96, values[peak_idx] + 0.16)),
            fontsize=8.5,
            color="#176b3a",
            arrowprops={"arrowstyle": "->", "color": "#176b3a", "lw": 1.2},
        )
        ax.text(
            kharif_x.mean(),
            0.98,
            "Kharif 2024",
            ha="center",
            va="top",
            fontsize=9,
            color="#176b3a",
            fontweight="bold",
        )

    if rabi_mask.any():
        rabi_x = x[rabi_mask]
        ax.axvspan(
            rabi_x.min() - 0.5,
            rabi_x.max() + 0.5,
            alpha=0.07,
            color="#fee08b",
        )
        peak_idx = rabi_x[np.argmax(values[rabi_mask])]
        ax.scatter(
            peak_idx, values[peak_idx], color="#b8860b", s=95, zorder=5
        )
        ax.annotate(
            f"Rabi seasonal maximum\nNDVI = {values[peak_idx]:.3f}",
            xy=(peak_idx, values[peak_idx]),
            xytext=(min(len(df) - 2, peak_idx + 0.35),
                    min(0.96, values[peak_idx] + 0.16)),
            fontsize=8.5,
            color="#8a6500",
            arrowprops={"arrowstyle": "->", "color": "#8a6500", "lw": 1.2},
        )
        ax.text(
            rabi_x.mean(),
            0.98,
            "Rabi 2024–25",
            ha="center",
            va="top",
            fontsize=9,
            color="#8a6500",
            fontweight="bold",
        )

    ax.plot(
        x,
        values,
        color="#1a6b3a",
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor="#1a6b3a",
        markeredgewidth=1.6,
        zorder=4,
    )

    labels = dates.dt.strftime("%b\n%Y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("District-wide Mean NDVI", fontsize=10)
    ax.set_xlabel("Monthly observation", fontsize=10)
    ax.set_ylim(0, max(1.0, float(np.nanmax(values)) + 0.2))
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_title(
        "Monthly NDVI Time Series — Ludhiana District, Punjab (2024–25)\n"
        "Sentinel-2 Surface Reflectance Harmonized",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ndvi_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return source_note


# ---------------------------------------------------------------------
# 5. Classified area statistics
# ---------------------------------------------------------------------

def create_area_outputs(area_df: pd.DataFrame) -> dict:
    area_df = area_df.copy()
    classified_total = float(area_df["area_km2"].sum())
    if classified_total <= 0:
        raise ValueError("Total classified area must be greater than zero.")

    area_df["share_pct"] = area_df["area_km2"] / classified_total * 100
    area_df["area_km2"] = area_df["area_km2"].astype(float)

    difference = AOI_AREA_KM2 - classified_total
    difference_pct = difference / AOI_AREA_KM2 * 100

    # Do not silently call the difference masked area; it is the
    # AOI-minus-classified-area balance and needs raster-level confirmation.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(
        area_df["class_name"],
        area_df["area_km2"],
        color=CLASS_COLORS,
        edgecolor="white",
        linewidth=1.1,
        width=0.55,
    )

    for bar, (_, row) in zip(bars, area_df.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + classified_total * 0.012,
            f'{row["area_km2"]:,.2f} km²\n({row["share_pct"]:.1f}%)',
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Classified Area (km²)", fontsize=10)
    ax.set_xlabel("Cropping Pattern Class", fontsize=10)
    ax.set_title(
        "Cropping Pattern Area Statistics — Ludhiana District, Punjab\n"
        f"Classified area: {classified_total:,.2f} km²",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, float(area_df["area_km2"].max()) * 1.25)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color=color, label=name)
        for color, name in zip(CLASS_COLORS, CLASS_NAMES)
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "area_barchart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save a machine-readable summary alongside the chart.
    area_summary_path = OUTPUT_DIR / "area_summary.csv"
    area_df.to_csv(area_summary_path, index=False, float_format="%.6f")

    return {
        "classified_total": classified_total,
        "aoi_area": AOI_AREA_KM2,
        "aoi_minus_classified": difference,
        "difference_pct_of_aoi": difference_pct,
        "area_df": area_df,
    }


# ---------------------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------------------

def main() -> int:
    print("=" * 65)
    print("P7 — ACCURACY ASSESSMENT AND RESULTS VISUALIZATION")
    print("=" * 65)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    val_df, invalid_rows = load_validation_data(VALIDATION_FILE)
    area_df = load_area_data(AREA_FILE)

    accuracy = create_accuracy_outputs(val_df, invalid_rows)
    ndvi_source = create_ndvi_chart()
    area = create_area_outputs(area_df)

    print("\nACCURACY RESULTS")
    print("-" * 45)
    print(f"Evaluated samples: {accuracy['sample_count']}")
    print(f"Excluded rows with invalid/missing class values: {invalid_rows}")
    print(f"Overall agreement: {accuracy['oa']:.4f} ({accuracy['oa'] * 100:.2f}%)")
    print(f"Cohen's Kappa: {accuracy['kappa']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(accuracy["cm_df"])
    print("\nClassification report:")
    print(accuracy["report_text"])

    print("\nAREA RESULTS")
    print("-" * 45)
    print(area["area_df"].to_string(index=False))
    print(f"\nTotal classified area: {area['classified_total']:,.3f} km²")
    print(f"AOI area reference: {area['aoi_area']:,.3f} km²")
    print(
        "AOI minus classified area: "
        f"{area['aoi_minus_classified']:,.3f} km² "
        f"({area['difference_pct_of_aoi']:.3f}% of AOI)"
    )
    print(
        "Note: this difference is not automatically equivalent to masked "
        "area; confirm it against the classified raster's NoData/unclassified pixels."
    )

    print("\nNDVI chart source:")
    print(ndvi_source)

    print("\nOUTPUTS GENERATED")
    print("-" * 45)
    for filename in [
        "accuracy_report.txt",
        "confusion_matrix.png",
        "ndvi_timeseries.png",
        "area_barchart.png",
        "area_summary.csv",
    ]:
        print(f"  {OUTPUT_DIR / filename}")

    print("\nCompleted successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
