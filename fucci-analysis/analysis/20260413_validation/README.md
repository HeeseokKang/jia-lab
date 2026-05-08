# 20260413_validation

Outputs for FUCCI timelapse analysis on dataset `20260413_FUCCI_Timelapse`
(32 wells × 67 timepoints × 3 channels). Currently scoped to single-well
validation on **`R1_1`** before scaling.

The work has two phases:

1. **Background characterization (Phase 1)** — quadrant-level QC of 561/647 and
   prototype flat-field correction. (Earlier work.)
2. **Cell cycle pipeline (Phase 2)** — segmentation, tracking, and per-cell
   FUCCI intensity extraction. (Current main path.)

---

## Folder contents

- `background/` (Phase 1) — quadrant-level CSVs and gradient metrics.
- `plots/` (Phase 1) — background gradient heatmap and other QC PNGs.
- `corrected/` (Phase 1, **gitignored, ~18 GB**) — flat-field-corrected 647
  TIFFs, produced by `src/flat_field_correction.py` with `--strategy-647 t0`.
  Currently NOT used by the cell cycle pipeline (see Phase 2 finding below).
- `segmentation_test/` (Phase 2) — current pipeline outputs. Detailed below.

### Not in git

| Path | Size | How to recreate |
| --- | --- | --- |
| `corrected/` | ~18 GB | `python fucci-analysis/src/flat_field_correction.py` |
| `segmentation_test/masks/` | ~576 MB (67 files) | `python fucci-analysis/src/timeseries_one_well.py` |

On Dodo these directories already live at:

- `/home/heeseok/github/jia-lab/fucci-analysis/analysis/20260413_validation/corrected/`
- `/home/heeseok/github/jia-lab/fucci-analysis/analysis/20260413_validation/segmentation_test/masks/`

---

## `segmentation_test/` inventory (Phase 2)

All outputs are for well `R1_1`. The pipeline scripts that produced each file
are noted in parentheses.

### Cell tracking

- `R1_1_tracks.csv` — frame-to-frame linked tracks; columns
  `timepoint, cell_id, track_id, centroid_x, centroid_y, area`.
  246 tracks total, 2509 rows. (`tracking_one_well.py`)
- `R1_1_track_qc.png` — track length histogram with QC totals
  (full duration / edge endpoint / suspicious mid-FOV).

### Population time series (legacy, no tracking)

- `R1_1_all_timepoints.csv` — per-cell intensities at every timepoint, no
  tracking. Each timepoint reuses fresh Cellpose ids. (`timeseries_one_well.py`)
- `R1_1_ratio_timeseries.png` — population median + 10–90 percentile band.

### Per-track nuclear intensity (current main output)

- `R1_1_nuclear_intensity.csv` — per-track per-timepoint nuclear-mask intensity
  after filtering (`track length >= 5 AND area > 400 px`). 67 tracks,
  2142 rows. **Uses raw 647** (see finding below). Columns:
  `track_id, timepoint, cell_id, area, nuclear_area, mean_561, mean_647`.
  (`nuclear_intensity_one_well.py`, default erosion=3)
- `R1_1_trace_561.png`, `R1_1_trace_647.png`, `R1_1_trace_ratio.png`
  — per-track time traces; full-duration tracks highlighted in red.

Erosion sensitivity variant (kept side-by-side; same filter, raw 647):

- `R1_1_nuclear_intensity_e15.csv` — same columns, erosion=15.
- `R1_1_trace_561_e15.png`, `R1_1_trace_647_e15.png`, `R1_1_trace_ratio_e15.png`
- `qc_population_t30_e15.png`, `qc_segmentation_grid_e15.png`
  (`nuclear_intensity_one_well.py --erosion 15 --suffix _e15`)

### 647 source diagnostic

- `R1_1_nuclear_647_comparison.csv` — same tracks/erosion, both corrected and
  raw 647 sampled side-by-side. Columns:
  `track_id, timepoint, cell_id, nuclear_area, mean_561, mean_647_corrected, mean_647_raw`.
  (`nuclear_intensity_raw_comparison.py`)
- `R1_1_647_correction_diagnosis.png` — 2×2 figure: per-track 647 traces
  (corrected | raw) and per-track-mean histograms (corrected | raw),
  full-duration tracks only.

### Representative QC images

- `qc_segmentation_grid.png` — 4 timepoints (t = 0, 22, 44, 66) × 2 views
  (BF + all cell mask outlines | BF + filtered eroded nuclear masks).
- `qc_single_cell_trace.png` — best full-duration track (highest mean 647);
  100×100 BF and 647 crops at 4 timepoints.
- `qc_population_t30.png` — at t = 30, filtered nuclear regions colored by
  raw `mean_647`; useful for spatial heterogeneity check.

### Single-image legacy / reference

- `R1_1_t030_BF_mask.npy` — Cellpose-SAM mask from the very first
  `segment_test.py` run (R1_1, t=30). Reference / sanity-check only.
- `R1_1_t030_BF_overlay.png` — overlay PNG from the same run.
- `R1_1_t030_intensity.csv` — single-timepoint per-cell 561/647 from
  `extract_intensity.py` (intermediate test, superseded).

### `masks/` (gitignored)

67 BF masks `R1_1_t000_BF_mask.npy` … `R1_1_t066_BF_mask.npy`, int32,
1500 × 1500. Produced by `timeseries_one_well.py` and consumed by
`tracking_one_well.py`, `nuclear_intensity_one_well.py`, and
`nuclear_intensity_raw_comparison.py`.

---

## Key findings

### Phase 1 — background gradient (earlier)

Quantified from `background/background_gradient_metrics_561_647.csv`
(see `plots/background_gradient_heatmap_561_647.png`):

- 561: `row_gradient_range = 5.591`, `col_gradient_range = 44.953`,
  `well_nonuniform_std = 13.838` — stronger overall non-uniformity, larger
  variation along the col axis.
- 647: `row_gradient_range = 10.419`, `col_gradient_range = 27.854`,
  `well_nonuniform_std = 10.622` — more structured row-axis gradient.

This motivated trying a pseudo flat-field correction for 647.

### Phase 2 — flat-field correction killed FUCCI signal

Compared corrected vs raw 647 with everything else fixed (same tracks, same
BF mask, same 3-px erosion, full-duration tracks only):

- `std mean_647 corrected = 3.885`
- `std mean_647 raw       = 204.454`
- **`std_raw / std_corrected = 52.6×`**

The t0-reference correction (`corrected = raw / ref(t0) * median(ref(t0))`)
collapses cell-to-cell variation in 647 because t = 0 already contains cells
in various cycle phases at fixed positions, so the "reference" includes
biological signal that gets normalized away. Source: `R1_1_647_correction_diagnosis.png`,
script `src/nuclear_intensity_raw_comparison.py`.

**Decision**: the current cell cycle pipeline uses **raw 647**. The 647 row-axis
gradient is preserved, but biological FUCCI signal is not destroyed. A different
correction strategy (e.g. `--strategy-647 min` in `flat_field_correction.py`,
or a properly cell-free reference) should be evaluated before re-introducing
correction.

### Phase 2 — erosion sensitivity (default 3 vs 15)

`nuclear_intensity_one_well.py` exposes `--erosion` for sensitivity testing.
Comparing the default `erosion=3` run with `erosion=15` (`_e15` outputs):

| metric (full-duration tracks) | erosion=3 | erosion=15 |
| --- | --- | --- |
| mean `nuclear_area / area` | 0.760 | 0.261 |
| std `mean_561` (raw) | 91.9 | 178.3 |
| std `mean_647` (raw) | 204.5 | 204.2 |
| `ratio_647_561` range | 0.850–1.293 | 0.759–1.285 |

Stronger erosion strips most cytoplasm (≈26 % of cell area remains), which
roughly doubles the per-track 561 cell-to-cell std, while the 647 std is
already saturated at the raw value and barely moves. Visual check that the
yellow nuclear-mask overlay sits inside the nucleus:
`qc_segmentation_grid_e15.png`.

### Phase 2 — track quality on R1_1

From `tracking_one_well.py` and `nuclear_intensity_one_well.py`:

- Total tracks (raw, no filter): **246**
- Full-duration tracks (length = 67) before any filter: **20**
- After `track length >= 5 AND area > 400 px` filter: **67 tracks**, of which
  **12 are still full-duration**. The drop from 20 → 12 is mainly due to
  occasional small-area frames within otherwise long tracks; those rows are
  removed by the area filter, leaving the surviving track shorter than 67.
- Suspicious mid-FOV ghost tracks before filter: 153 (62% of all tracks),
  ~80% of which have length ≤ 3 frames. These are dominated by track
  fragmentation and short-lived false positives, both addressable by raising
  `MEMORY_FRAMES` and/or stricter post-tracking length cuts.
