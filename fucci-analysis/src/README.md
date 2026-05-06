# src/README

## A1.4 Background characterization

This folder contains scripts used in step **A1.4** to quantify spatial background non-uniformity in FUCCI timelapse frames before downstream segmentation/tracking.

### `quantify_background.py` (core computation)

**Purpose**
- For a single channel (`BF` / `561` / `647`), compute **quadrant mean intensity** for each `(timepoint, well)` frame.
- Apply a robust **per-timepoint median subtraction** across wells to estimate how much structured background can be removed without “blank wells”.

**Inputs**
- Raw dataset folder (default: `configs.paths.DATASET_1_RAW`):
  - `timepoint (0-66)/R{row}_{col}_{z}_{cellline}_FUCCI_{channel}.tiff`
- File naming convention is parsed by:
  - `R{row}_{col}_{z}_{cellline}_FUCCI_{channel}.tiff`

**Outputs**
- Writes CSVs under:
  - `analysis/20260413_validation/background/`
- Per channel:
  - `quadrant_background_{channel}.csv`
    - Columns include: `timepoint`, `well_id`, `row`, `col`, `z`, `cellline`, `channel`,
      `image_mean`, `top_left`, `top_right`, `bottom_left`, `bottom_right`,
      and median subtraction columns like `*_tp_median` and `*_median_subtracted`.
  - `timepoint_background_summary_{channel}.csv`
    - Per-timepoint summary statistics (global median/std + quadrant medians).

### `plot_background_gradient.py` (visualization + gradient metrics)

**Purpose**
- Read the `quadrant_background_{channel}.csv` outputs for **`561` and `647`**.
- Create a **row/col heatmap** showing mean background intensity patterns.
- Summarize gradient strength with simple metrics.

**Inputs**
- CSVs:
  - `analysis/20260413_validation/background/quadrant_background_561.csv`
  - `analysis/20260413_validation/background/quadrant_background_647.csv`

**Outputs**
- `analysis/20260413_validation/background/background_gradient_metrics_561_647.csv`
  - Includes metrics such as row/col range and residual non-uniformity proxies.
- `analysis/20260413_validation/plots/background_gradient_heatmap_561_647.png`

### Key finding (from today’s run)

- `647` channel: more pronounced **structured gradient along the `row` axis** (systematic component).
- `561` channel: larger overall **non-uniformity**, and stronger variation along the **`col` axis** (more “noise-like” + axis-dependent variation).

### Next step (pseudo-flat-field correction)

Because there are no true blank wells, the next stage is a **pseudo-flat-field** approach:
- Model background per `(timepoint, channel)` using robust statistics across wells (e.g., median across wells).
- Option A (quadrant-based): extrapolate a smooth row/col surface from quadrant residuals and remove it from the full frame.
- Option B (pixel-based, recommended later if needed): compute a per-timepoint/per-channel correction image (robust across wells), optionally smooth it, then apply **subtraction** (or optionally division if multiplicative shading dominates).

