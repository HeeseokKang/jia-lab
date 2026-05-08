# Jia Lab Image Analysis Pipelines

Lead: Heeseok Kang
Advisor: Bill Jia (UCSF)

## Project

FUCCI cell cycle timelapse image analysis. The current end-to-end pipeline covers
brightfield (BF) cell segmentation, frame-to-frame tracking, and per-cell
fluorescence extraction in the 561 and 647 channels for downstream cell cycle
phase analysis.

Voltage-imaging workflows will be added later under `voltage-imaging/`.

## Primary dataset

- Experiment: `20260413_FUCCI_Timelapse`
- Layout: 32 wells × 67 timepoints × 3 channels (`BF`, `561`, `647`)
- Image size: 1500 × 1500, uint16 TIFF
- Acquisition subfolder: `timelapse_plus_bf_2026-04-16_12-11-45.768458`

## Repository structure

- `configs/` — Canonical data paths (`DATASET_1_RAW`, `RESULT_ROOT`, …) and runtime constants.
- `shared/` — Helper modules used across pipelines (e.g. `get_clean_file_list`).
- `fucci-analysis/`
  - `src/` — All analysis scripts. See `fucci-analysis/src/README.md` for the
    pipeline order and per-script description.
  - `analysis/20260413_validation/` — Outputs for the dataset above (CSV
    tables, QC PNGs, segmentation masks). See its `README.md` for inventory
    and key findings.
- `README.md` — This file.

## Data paths on Dodo

- Scratch (preferred for I/O): `/data/Project_Data/Voltage_CellCycle/Data/`
- NAS (canonical): `/mnt/nas1/Projects/Voltage_CellCycle/Data/`

The scripts use `configs/paths.py` (NAS) or hard-coded scratch paths where
explicitly faster. Update `configs/paths.py` when running on a different host.

## How to run

All scripts are designed to be runnable as standalone modules from the repo root
on Dodo:

```bash
conda activate fucci-analysis
python fucci-analysis/src/<script>.py
```

For example, the FUCCI cell cycle pipeline runs in this order:

```bash
conda activate fucci-analysis
python fucci-analysis/src/segment_test.py                      # one-frame Cellpose-SAM smoke test
python fucci-analysis/src/timeseries_one_well.py               # 67-frame segmentation, writes masks/
python fucci-analysis/src/tracking_one_well.py                 # trackpy linking
python fucci-analysis/src/nuclear_intensity_one_well.py        # nuclear-mask intensity (raw 647)
python fucci-analysis/src/nuclear_intensity_raw_comparison.py  # corrected vs raw 647 diagnostic
```

Scripts that need GPU (Cellpose-SAM) auto-detect and use the Dodo GPU.

## Current focus

- Cell cycle pipeline validation on a single well (`R1_1`) before scaling to all 32 wells.
- 647 flat-field correction is currently disabled in the analysis path: the
  t0-reference correction was found to suppress FUCCI signal on this dataset
  (raw vs corrected std ratio ≈ 52.6×). See
  `fucci-analysis/analysis/20260413_validation/README.md` for details.
