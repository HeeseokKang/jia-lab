# fucci-analysis/src

Scripts for the FUCCI cell cycle timelapse pipeline. Two groups:

1. **Cell cycle pipeline** — current main path (single-well validation on `R1_1`).
2. **Background characterization** (precursor work) — earlier scripts that
   characterized 561/647 background and prototyped flat-field correction.

All scripts run from the repo root with the `fucci-analysis` conda environment:

```bash
conda activate fucci-analysis
python fucci-analysis/src/<script>.py
```

Outputs go to `fucci-analysis/analysis/20260413_validation/` (see that
folder's README for an inventory and findings).

---

## Cell cycle pipeline (in pipeline order)

### 1. `segment_test.py` — Cellpose-SAM smoke test (single frame)

Reference / sanity check only. Loads one BF frame
(`R1_1`, `t=30`), runs Cellpose-SAM (`CellposeModel(model_type='cpsam')`),
saves the integer mask as `.npy` and a BF + outline overlay PNG. Used once
to confirm Cellpose-SAM segments cells reasonably before scaling up.

### 2. `timeseries_one_well.py` — DEPRECATED (superseded by tracking pipeline)

Originally ran Cellpose-SAM on every BF frame of one well and computed a
population-level ratio time series (median + 10–90 percentile band). The
masks it produces under `segmentation_test/masks/` are still consumed
downstream by tracking and nuclear-intensity scripts, so the script remains
runnable for mask regeneration. Population summary plot is no longer used —
prefer per-track traces from `nuclear_intensity_one_well.py`.

### 3. `tracking_one_well.py` — Frame-to-frame cell tracking (trackpy)

Reads all 67 BF masks from `segmentation_test/masks/`, extracts centroids
with `regionprops`, then links detections across frames using
`trackpy.link(search_range=50, memory=1)`. Writes `R1_1_tracks.csv`
(`timepoint, cell_id, track_id, centroid_x, centroid_y, area`) and a track
length histogram PNG. Also reports QC categories: full-duration tracks,
edge-endpoint tracks, suspicious mid-FOV tracks.

### 4. `nuclear_intensity_one_well.py` — Per-track nuclear FUCCI intensity (raw 647)

For each filtered detection (`track length >= 5 AND area > 400 px`), erodes
the BF cell mask by `--erosion` iterations (default 3 px) to approximate the
nuclear region, then samples mean intensity in 561 (raw) and 647 (raw).
Writes `R1_1_nuclear_intensity{suffix}.csv` and three per-track trace plots
(561, 647, 647/561 ratio) plus a population snapshot at t=30. **Uses raw 647**:
the t0-reference flat-field correction was found to flatten FUCCI signal
(see the diagnostic script below).

CLI options for sensitivity testing without overwriting defaults:

- `--erosion N` — `binary_erosion` iterations (default 3)
- `--suffix STR` — appended to all output filenames, e.g. `_e15`
- `--with-grid` — also write `qc_segmentation_grid{suffix}.png`
  (auto-enabled when `--suffix` is non-empty)

Example: stronger erosion variant kept side-by-side with the default run:

```bash
python fucci-analysis/src/nuclear_intensity_one_well.py --erosion 15 --suffix _e15
```

### 5. `nuclear_intensity_raw_comparison.py` — Diagnostic: corrected vs raw 647

Same tracks, same BF masks, same erosion. Only the 647 source differs
(`corrected/` vs raw `DATA_ROOT`). Writes a side-by-side CSV and a 2×2
diagnostic figure (per-track traces and full-duration histograms for both
sources) and prints the standard-deviation ratio
`std_raw / std_corrected`. The current value (≈ 52.6×) was the basis for
switching the main pipeline to raw 647.

---

## Background characterization (precursor analyses)

Earlier scripts used to quantify spatial background and prototype flat-field
correction on this dataset. Outputs live under
`analysis/20260413_validation/background/` and
`analysis/20260413_validation/plots/`.

- `quantify_background.py` — Quadrant-mean background per `(timepoint, well)`,
  plus per-timepoint median subtraction. Writes
  `quadrant_background_{channel}.csv` and
  `timepoint_background_summary_{channel}.csv`.
- `plot_background_gradient.py` — Reads quadrant tables, generates
  `background_gradient_heatmap_561_647.png` and
  `background_gradient_metrics_561_647.csv`.
- `inspect_background.py` — Quick interactive inspection of one frame.
- `report_single_well_background.py` — Single-well/timepoint quadrant report.
- `flat_field_correction.py` — Pseudo flat-field correction
  (`--strategy-647 t0|min`, `--strategy-561 raw|t0|min`); produced
  `analysis/20260413_validation/corrected/` (gitignored).
- `check_data.py` — Verifies raw vs cleaned image counts per channel.
- `diagnose_mismatch.py` — Reports duplicate / mismatched channel indexes.
- `visual_qc.py` — Visual QC figure for a single frame across BF / 561 / 647.

## Other / intermediate

- `extract_intensity.py` — Earlier single-timepoint per-cell intensity test
  (one well, one timepoint). Superseded by
  `nuclear_intensity_one_well.py`.
