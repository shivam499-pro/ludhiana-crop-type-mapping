import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    cohen_kappa_score, classification_report
)
import os

DATA_DIR   = r'C:\summer-project-P7\data'
OUTPUT_DIR = r'C:\summer-project-P7\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES  = ['Double Cropping', 'Single Cropping', 'Fallow / Non-crop']
CLASS_IDS    = [1, 2, 3]
CLASS_COLORS = ['#1a9850', '#fee08b', '#d73027']

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


val_df = pd.read_csv(os.path.join(DATA_DIR, 'P7_validation_results.csv'))
print("Validation CSV columns:", val_df.columns.tolist())
print(val_df.head())

val_df = val_df[['true_class', 'predicted_class']].dropna()
val_df = val_df.astype(int)

y_true = val_df['true_class'].values
y_pred = val_df['predicted_class'].values

print(f"\nTotal validation samples: {len(y_true)}")
print(f"True class distribution:      {pd.Series(y_true).value_counts().sort_index().to_dict()}")
print(f"Predicted class distribution: {pd.Series(y_pred).value_counts().sort_index().to_dict()}")

oa    = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
cm    = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
cr    = classification_report(
    y_true, y_pred,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    digits=4
)

print("\n" + "="*55)
print("ACCURACY ASSESSMENT RESULTS")
print("="*55)
print(f"Overall Accuracy (OA) : {oa:.4f} ({oa*100:.2f}%)")
print(f"Kappa Coefficient     : {kappa:.4f}")
print("\nConfusion Matrix (rows=True, cols=Predicted):")
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
print(cm_df)
print("\nClassification Report:")
print(cr)


report_path = os.path.join(OUTPUT_DIR, 'accuracy_report.txt')
with open(report_path, 'w') as f:
    f.write("P7 — Accuracy Assessment Report\n")
    f.write("Ludhiana District Cropping Pattern Mapping\n")
    f.write("="*55 + "\n\n")
    f.write(f"Overall Accuracy (OA) : {oa:.4f} ({oa*100:.2f}%)\n")
    f.write(f"Kappa Coefficient     : {kappa:.4f}\n\n")
    f.write("Confusion Matrix (rows=True, cols=Predicted):\n")
    f.write(cm_df.to_string())
    f.write("\n\nClassification Report:\n")
    f.write(cr)
print(f"\nAccuracy report saved → {report_path}")

fig, ax = plt.subplots(figsize=(7, 5.5))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

sns.heatmap(
    cm_norm,
    annot=False,
    cmap='Greens',
    vmin=0, vmax=100,
    linewidths=0.5,
    linecolor='white',
    ax=ax,
    cbar_kws={'label': 'Row percentage (%)'}
)


for i in range(3):
    for j in range(3):
        count = cm[i, j]
        pct   = cm_norm[i, j]
        color = 'white' if pct > 60 else '#1a1a1a'
        ax.text(
            j + 0.5, i + 0.5,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color=color
        )

ax.set_xticklabels(CLASS_NAMES, fontsize=10, rotation=15, ha='right')
ax.set_yticklabels(CLASS_NAMES, fontsize=10, rotation=0)
ax.set_xlabel('Predicted Class', fontsize=11, labelpad=10)
ax.set_ylabel('True Class', fontsize=11, labelpad=10)
ax.set_title(
    f'Confusion Matrix — Ludhiana Cropping Pattern Classification\n'
    f'OA = {oa*100:.2f}%  |  Kappa = {kappa:.4f}  |  n = {len(y_true)} validation samples',
    fontsize=11, fontweight='bold', pad=14
)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"OUT_PY01 saved → {cm_path}")

dates = [
    'Jun\n2024', 'Jul\n2024', 'Aug\n2024', 'Sep\n2024', 'Oct\n2024',
    'Nov\n2024', 'Dec\n2024', 'Jan\n2025', 'Feb\n2025', 'Mar\n2025', 'Apr\n2025'
]
ndvi_values = [0.201, 0.313, 0.759, 0.669, 0.464,
               0.206, 0.367, 0.553, 0.678, 0.633, 0.280]

fig, ax = plt.subplots(figsize=(11, 5))

x = np.arange(len(dates))

ax.axvspan(-0.5, 4.5, alpha=0.08, color='#1a9850', label='_nolegend_')
ax.axvspan(4.5, 10.5, alpha=0.08, color='#fee08b', label='_nolegend_')

ax.text(2.0, 0.93, 'Kharif 2024\n(Rice)', ha='center', va='top',
        fontsize=10, color='#1a9850', fontweight='bold')
ax.text(7.5, 0.93, 'Rabi 2024–25\n(Wheat)', ha='center', va='top',
        fontsize=10, color='#b8860b', fontweight='bold')


ax.plot(x, ndvi_values, color='#1a6b3a', linewidth=2.5,
        marker='o', markersize=6, markerfacecolor='white',
        markeredgecolor='#1a6b3a', markeredgewidth=2, zorder=5)

kharif_peak_idx = int(np.argmax(ndvi_values[:5]))        # Aug → index 2
rabi_peak_idx   = int(np.argmax(ndvi_values[5:])) + 5   # Feb → index 8


ax.scatter([kharif_peak_idx], [ndvi_values[kharif_peak_idx]],
           color='#1a9850', s=120, zorder=6)
ax.scatter([rabi_peak_idx], [ndvi_values[rabi_peak_idx]],
           color='#b8860b', s=120, zorder=6)

ax.annotate(f'Kharif Peak\nNDVI = {ndvi_values[kharif_peak_idx]:.3f}\n(Rice heading)',
            xy=(kharif_peak_idx, ndvi_values[kharif_peak_idx]),
            xytext=(kharif_peak_idx - 0.8, 0.85),
            fontsize=8.5, color='#1a6b3a',
            arrowprops=dict(arrowstyle='->', color='#1a6b3a', lw=1.4))

ax.annotate(f'Rabi Peak\nNDVI = {ndvi_values[rabi_peak_idx]:.3f}\n(Wheat heading)',
            xy=(rabi_peak_idx, ndvi_values[rabi_peak_idx]),
            xytext=(rabi_peak_idx + 0.3, 0.85),
            fontsize=8.5, color='#b8860b',
            arrowprops=dict(arrowstyle='->', color='#b8860b', lw=1.4))

ax.axhline(0.5, color='#d73027', linewidth=1.2, linestyle='--', alpha=0.7)
ax.axhline(0.4, color='#fc8d59', linewidth=1.0, linestyle=':', alpha=0.7)
ax.text(10.55, 0.50, '0.50\n(Peak\nthreshold)', fontsize=7.5,
        color='#d73027', va='center')
ax.text(10.55, 0.40, '0.40\n(Crop\npresence)', fontsize=7.5,
        color='#fc8d59', va='center')

ax.annotate('Inter-season\ntrough\n(NDVI = 0.21)',
            xy=(5, 0.206), xytext=(5.5, 0.12),
            fontsize=8, color='#555555',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(dates, fontsize=9)
ax.set_ylabel('Mean NDVI', fontsize=11)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylim(0, 1.02)
ax.set_xlim(-0.5, 10.5)
ax.set_title(
    'NDVI Time-Series — Ludhiana District, Punjab (2024–25)\n'
    'Sentinel-2 SR Harmonized | Monthly Median Composites',
    fontsize=11, fontweight='bold', pad=12
)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, 'ndvi_timeseries.png')
plt.savefig(ts_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"OUT_PY02 saved → {ts_path}")

area_df = pd.read_csv(os.path.join(DATA_DIR, 'P7_area_stats.csv'))
print("\nArea stats CSV:")
print(area_df)

# Keep only needed columns
area_df = area_df[['class_name', 'area_km2']].copy()
area_df['area_km2'] = area_df['area_km2'].astype(float).round(2)
total_area = area_df['area_km2'].sum()
area_df['pct'] = (area_df['area_km2'] / total_area * 100).round(1)

print("\nClass areas:")
for _, row in area_df.iterrows():
    print(f"  {row['class_name']}: {row['area_km2']:.2f} km² ({row['pct']:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    area_df['class_name'],
    area_df['area_km2'],
    color=CLASS_COLORS,
    edgecolor='white',
    linewidth=1.2,
    width=0.5
)

# Value labels on bars
for bar, (_, row) in zip(bars, area_df.iterrows()):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 15,
        f'{row["area_km2"]:,.1f} km²\n({row["pct"]:.1f}%)',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold', color='#1a1a1a'
    )

ax.set_ylabel('Area (km²)', fontsize=11)
ax.set_xlabel('Cropping Pattern Class', fontsize=11)
ax.set_title(
    'Cropping Pattern Area Statistics — Ludhiana District, Punjab\n'
    f'Total classified area: {total_area:,.1f} km²  |  Kharif 2024 + Rabi 2024–25',
    fontsize=11, fontweight='bold', pad=12
)
ax.set_ylim(0, area_df['area_km2'].max() * 1.22)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Legend patches
patches = [mpatches.Patch(color=c, label=n)
           for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
ax.legend(handles=patches, loc='upper right', fontsize=9,
          framealpha=0.8, edgecolor='#cccccc')

plt.tight_layout()
area_path = os.path.join(OUTPUT_DIR, 'area_barchart.png')
plt.savefig(area_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"OUT_PY03 saved → {area_path}")

print("\n" + "="*55)
print("ALL OUTPUTS GENERATED")
print("="*55)
print(f"  OUT_PY01 → outputs/confusion_matrix.png")
print(f"  OUT_PY02 → outputs/ndvi_timeseries.png")
print(f"  OUT_PY03 → outputs/area_barchart.png")
print(f"  TXT      → outputs/accuracy_report.txt")
print(f"\nOA = {oa*100:.2f}%  |  Kappa = {kappa:.4f}")
print("Python phase complete ✓")

