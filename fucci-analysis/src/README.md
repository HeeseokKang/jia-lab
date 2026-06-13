# fucci-analysis/src

Scripts for two FUCCI experiments. All run from the repo root under the
`fucci-analysis` conda env; outputs go dataset-side under `<dataset>/analysis/`.

```bash
conda activate fucci-analysis
python fucci-analysis/src/<script>.py
```

1. **Variant characterization** (`variant_*.py`) — current main path,
   `20260610_fucci_constructs_test` snapshot.
2. **Single-well timelapse** (`*_one_well.py`) — earlier precursor,
   `20260413_FUCCI_Timelapse` validation on `R1_1`.

> Trial-and-error scripts, superseded prototypes, and the rejected flat-field
> background scripts were removed in repo cleanups (2026-06-01, 2026-06-10); the
> entries below are the maintained scripts. Removed code stays in git history.

---

## 1. Variant characterization pipeline (in stage order)

### `variant_config.py` — single source of truth
Well/construct/drug layout, channel→role mapping, exposure, gate thresholds
(`GATE_SNR`, `MIN_RED_GATE_FRAC`), erosion sweep + default. All other variant
scripts import this; change parameters here.

### Stage 0 — `variant_index.py` — snapshot manifest
Parses `2026-06-10_HH-MM-SS_<construct>_<well>_<channel>_<int>pct.tif` into one
row per image (construct, well, row/col, drug, replicate, channel, role, exposure,
time, path), applies the D900 benchmark-time filter (early titration excluded),
checks 36-well × 3-channel completeness. Writes `manifest.csv`.

### Stage 1 — `variant_segment.py` — BF segmentation (Cellpose-SAM)
Runs Cellpose-SAM (`cpsam`) on each BF snapshot, writes an integer label mask per
well to `<analysis>/masks/<construct>_<well>_BF_mask.npy` (heavy, dataset-side).
Resumable (`--overwrite` to force). Writes `seg_summary.csv` (n_cells/well).

### Stage 3 — `variant_measure.py` — per-cell reporter intensity
Per well: load BF mask + green (488) + red (2nd reporter) snapshots. Estimate
**per-image local background** from non-cell pixels (`bg_median`, robust
`bg_sigma = 1.4826·MAD`). For each cell and each erosion level in
`cfg.EROSION_SWEEP_PX`, erode to approximate the nucleus and record raw mean,
background-subtracted mean, and `SNR = (mean − bg_median)/bg_sigma` per channel.
**Background subtraction only — no flat-field.** Writes long-form
`per_cell_measurements.csv` (one row per cell × erosion) + `bg_summary.csv`.

### Stage 4-6 — `variant_analyze.py` — 2D gating, drug response, comparison
Reads `per_cell_measurements.csv` at the default erosion. Per-channel "on" =
`SNR ≥ GATE_SNR` → **2D quadrant gate** (green-only=G1, red-only=S,
double-positive=G2). PRIMARY endpoint = drug shifts the target-phase fraction
(Palbo→↑G1, RO→↑G2); a construct is `red_gateable` only if its DMSO red
bright-fraction ≥ `MIN_RED_GATE_FRAC`. Writes phase-fraction / drug-response /
brightness / separation / comparison CSVs, `RESULTS_SUMMARY.md`, and
phase-coloured scatter + stacked phase-fraction figures.

```bash
python fucci-analysis/src/variant_analyze.py [--erosion 7]
```

> The 1D `log2(red/green)` score is kept only as a demoted reference column: it
> mis-models G2 (double-positive → mid ratio) and under-detected the RO→G2 arrest.

---

## 2. Single-well timelapse pipeline (precursor, in order)

`20260413_FUCCI_Timelapse`, single-well validation on `R1_1`. Outputs under
`analysis/20260413_validation/` (see that folder's README for inventory + findings).

### `timeseries_one_well.py` — per-frame BF segmentation (Cellpose-SAM)
Runs Cellpose-SAM on every BF frame of one well, writes per-frame integer masks
under `segmentation_test/masks/` (gitignored). The **mask-regeneration** entry
point; its old population-ratio summary is deprecated in favour of per-track traces.

### `tracking_one_well.py` — frame-to-frame tracking (trackpy)
Reads the 67 BF masks, extracts centroids (`regionprops`), links with
`trackpy.link(search_range=50, memory=1)`. Writes `R1_1_tracks.csv`
(`timepoint, cell_id, track_id, centroid_x, centroid_y, area`) + a track-length
histogram, with QC categories (full-duration / edge-endpoint / suspicious mid-FOV).

### `nuclear_intensity_one_well.py` — per-track nuclear FUCCI intensity (raw 647)
For each filtered detection (`track length ≥ 5 AND area > 400 px`), erodes the BF
mask by `--erosion` iterations (default 3 px) to approximate the nucleus, samples
mean 561 (raw) and 647 (raw), writes `R1_1_nuclear_intensity{suffix}.csv` + per-track
trace plots. **Uses raw 647** — the t0-reference flat-field correction was found to
flatten FUCCI signal (`std_raw / std_corrected ≈ 52.6×`).

CLI: `--erosion N` (default 3), `--suffix STR` (e.g. `_e15`), `--with-grid`.

```bash
python fucci-analysis/src/nuclear_intensity_one_well.py --erosion 15 --suffix _e15
```
