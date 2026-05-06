# fucci-analysis

## Purpose

This is the primary project workspace for FUCCI timelapse data quality validation and preprocessing checks prior to segmentation and tracking.

## Key Files

- `src/check_data.py`: Compares raw versus cleaned image counts by channel and verifies alignment targets for BF/561/647 data.
- `src/diagnose_mismatch.py`: Diagnoses channel-index mismatches by reporting channel-specific unique indexes and duplicated index groups.
- `src/visual_qc.py`: Generates visual quality-control figures from selected aligned frames to inspect image quality and channel consistency.
- `src/quantify_background.py`: Background characterization (A1.4) for a single channel; computes quadrant means per timepoint/well and applies per-timepoint median subtraction.
- `src/plot_background_gradient.py`: Builds row/col background gradient heatmaps from the A1.4 CSV outputs and summarizes gradient strength.

## A1.4 Background characterization (completed)

Purpose: quantify spatial non-uniformity and estimate how much structured background remains after robust (median-based) subtraction, prior to downstream segmentation/tracking.

Outputs (Dataset 1 / `20260413_validation`):

- Background tables:
  - `analysis/20260413_validation/background/quadrant_background_647.csv`
  - `analysis/20260413_validation/background/quadrant_background_561.csv`
  - `analysis/20260413_validation/background/timepoint_background_summary_647.csv`
  - `analysis/20260413_validation/background/timepoint_background_summary_561.csv`
- Gradient visualization:
  - `analysis/20260413_validation/plots/background_gradient_heatmap_561_647.png`
- Gradient metrics:
  - `analysis/20260413_validation/background/background_gradient_metrics_561_647.csv`

Key finding (high-level): `647` shows more pronounced structured gradient along the `row` axis, while `561` exhibits stronger overall non-uniformity and larger variation along the `col` axis.

Next step: move from quadrant-level summaries to a (pseudo-)flat-field correction map (robust per-timepoint/per-channel background model), then re-run QC on background-corrected frames.
