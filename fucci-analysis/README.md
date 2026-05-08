# fucci-analysis

FUCCI cell cycle timelapse analysis subproject (Bill Jia Lab, UCSF).

Primary dataset: `20260413_FUCCI_Timelapse` — 32 wells × 67 timepoints × 3 channels (`BF`, `561`, `647`), 1500 × 1500 uint16. Currently scoped to single-well validation on `R1_1` before scaling.

## Layout

- `src/` — analysis scripts. See `src/README.md` for the pipeline order and per-script description.
- `analysis/20260413_validation/` — outputs (CSVs, QC PNGs, masks). See its `README.md` for inventory and findings.
- `environment.yml` — conda env (`fucci-analysis`).

## Two phases

1. **Background characterization** (precursor) — quadrant-level QC of 561/647 and pseudo flat-field correction (`quantify_background.py`, `plot_background_gradient.py`, `flat_field_correction.py`).
2. **Cell cycle pipeline** (current main path) — Cellpose-SAM segmentation → trackpy linking → per-track nuclear FUCCI intensity from a 3 px-eroded BF mask. Five scripts run in this order:
   - `segment_test.py` (one-frame smoke test)
   - `timeseries_one_well.py` (per-frame masks, deprecated population summary)
   - `tracking_one_well.py` (frame-to-frame linking)
   - `nuclear_intensity_one_well.py` (per-track 561/647 in nuclear region)
   - `nuclear_intensity_raw_comparison.py` (corrected vs raw 647 diagnostic)

## Headline result

The t0-reference flat-field correction was found to flatten 647 cell-to-cell variation (`std_raw / std_corrected ≈ 52.6×` on full-duration tracks). The cell cycle pipeline therefore uses **raw 647**; the row-axis 647 gradient is left in but biological FUCCI signal is preserved. See `analysis/20260413_validation/README.md` for the supporting figures and per-track stats.

## How to run

```bash
conda activate fucci-analysis
python fucci-analysis/src/<script>.py
```

Cellpose-SAM scripts auto-use the Dodo GPU. Large derived data (`corrected/` ~18 GB, `segmentation_test/masks/` ~576 MB) is gitignored — recreate with the relevant script.
