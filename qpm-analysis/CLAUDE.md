# qpm-analysis — Quantitative Phase Microscopy module

## Project goal

End-to-end DPC → WOTF → Tikhonov phase reconstruction pipeline for fixed HeLa
samples imaged on the lab's Cephla Squid microscope. The current dataset
(`20260513_Hela_p15`) is a 22-condition lid-ON / lid-OFF × media (5mM, 100mM,
serum, NO10) screen. Long-term goal: quantitative dry-mass-equivalent phase
maps; near-term goal: validate the DPC frontend, characterize LED asymmetry,
and lay down WOTF infrastructure.

## Data

- Raw root: `/data/Project_Data/QPM/`
- Active dataset: `/data/Project_Data/QPM/20260513_Hela_p15/`
- 22 condition folders: `off_{5mM,100mM,NO10,serum}{1..3}` and `on_{same}` (NO10
  has 2 reps per lid state, not 3).
- Each folder contains 5 TIFFs:
  - `*_bffull.tif` — full-aperture brightfield
  - `*_top.tif` / `*_bottom.tif` — y-axis half-aperture LEDs
  - `*_left.tif` / `*_right.tif` — x-axis half-aperture LEDs
- Format: uint16, 2400×2400.

## Hardware

- Cephla Squid microscope.
- SCI Dome programmable LED illumination (half-aperture top/bottom/left/right
  patterns for DPC).
- 10× / 0.3 NA objective.
- 10 ms exposure.

## Key scientific findings (drive code design)

1. **LED L/R asymmetry ≈ 11.2%** in this dataset. Each image must be
   mean-balanced (divided by its own global mean) **before** forming DPC, or
   the asymmetry leaks straight into the DPC signal.
2. **Raw DPC_x mean ≈ +0.11** is the LED-offset artifact, NOT a phase signal.
   After mean-balancing it collapses toward 0.
3. **DPC std is dominated by background (~92% of pixels by area).** Std must
   be decomposed into background (BF > 70th percentile) and cell-candidate
   (BF < 30th percentile) regions before it can be compared across conditions.
4. **DPC std is a contrast proxy only, not a quantitative phase metric.**
   Quantitative dry-mass / OPL values require WOTF-based Tikhonov-regularized
   phase reconstruction, which is the next pipeline stage.

## Repo layout

```
qpm-analysis/
├── CLAUDE.md                    # this file
├── src/
│   └── dpc_analyze.py           # mean-balanced DPC + bg/cell std + CLI
└── analysis/
    └── 20260513_Hela_p15/
        ├── figures/             # cross-condition figures (TBD)
        ├── dpc_master_table.csv # per-condition stats
        └── <cond>/              # per-condition outputs
            ├── dpc_y_bal.npy
            ├── dpc_x_bal.npy
            └── dpc_overview.png
```

Path registration lives in `configs/paths.py`:
`QPM_ROOT`, `QPM_20260513`, `QPM_RESULTS`.

## Environment

- Conda env: **`fucci-analysis`** (Python 3.10).
- Required: `numpy`, `tifffile`, `matplotlib`, `pandas` (already installed).

## Usage

```bash
conda activate fucci-analysis
python ~/github/jia-lab/qpm-analysis/src/dpc_analyze.py \
    /data/Project_Data/QPM/20260513_Hela_p15 \
    ~/github/jia-lab/qpm-analysis/analysis/20260513_Hela_p15
```

Per condition this writes `dpc_y_bal.npy`, `dpc_x_bal.npy`, and
`dpc_overview.png`. The aggregate stats land in `dpc_master_table.csv`.

## Current status — DPC validation phase

- [x] Read raw TIFFs, confirm geometry/dtype.
- [x] Mean-balanced DPC implementation (`compute_dpc_balanced`).
- [x] Per-condition bg/cell std decomposition.
- [x] Batch runner + master CSV for the 22 conditions.
- [ ] Cross-condition comparison plots (lid ON vs OFF × media).

## Next steps

1. **LED calibration.** Use the master table's `LR_asym_pct` / `TB_asym_pct` to
   characterize per-LED intensity drift across the session; build a flat-field
   correction for each LED direction.
2. **Empty-FOV control.** Acquire a no-sample reference set on the same stage
   to estimate the residual DPC pattern (illumination + camera + dust) that
   should be subtracted before reconstruction.
3. **WOTF / Tikhonov reconstruction.** Implement weak-object transfer function
   inversion with Tikhonov regularization to recover quantitative phase from
   the balanced DPC pairs. This is the step where outputs become a real phase
   metric, not a contrast proxy.
