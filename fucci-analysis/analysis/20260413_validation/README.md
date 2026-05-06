# 20260413_validation

## Background characterization results (A1.4)

This directory stores outputs for step **A1.4** (background characterization) on Dataset 1 validation artifacts.

### Folder contents

- `background/`
  - Quadrant-level tables and robust (median-based) subtraction summaries for each channel.
- `plots/`
  - Heatmaps/figures used to visually verify whether background variation has structured (systematic) components.

### Generated outputs (today’s run)

Background tables:
- `background/quadrant_background_647.csv`
- `background/quadrant_background_561.csv`
- `background/timepoint_background_summary_647.csv`
- `background/timepoint_background_summary_561.csv`

Gradient visualization + metrics:
- `plots/background_gradient_heatmap_561_647.png`
- `background/background_gradient_metrics_561_647.csv`

### Key findings (today’s run)

- **Quantitative gradient metrics (A1.4)**
  - `561`
    - `row_gradient_range`: `5.591`
    - `col_gradient_range`: `44.953`
    - `well_nonuniform_std`: `13.838`
    - `tp_mediansub_std_mean`: `15.196`
  - `647`
    - `row_gradient_range`: `10.419`
    - `col_gradient_range`: `27.854`
    - `well_nonuniform_std`: `10.622`
    - `tp_mediansub_std_mean`: `10.709`
- `647` channel shows a **more structured systematic gradient**, especially along the **row axis**.
- `561` channel shows **stronger overall non-uniformity**, with larger **col-axis variation** (more axis-dependent “noise-like” behavior).

### Next step: pseudo-flat-field correction

There are no true blank wells in this acquisition, so a true flat-field is not directly available. The recommended next approach is a pseudo-flat-field:

1. Estimate background per `(timepoint, channel)` robustly using the set of wells.
2. Fit a smooth correction surface (row/col) or compute a pixelwise correction image.
3. Apply a correction (typically **subtraction**; consider division only if the shading is clearly multiplicative).
4. Re-run QC to confirm that quadrant residuals/heatmaps become more uniform across wells.

