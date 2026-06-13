# qpm-analysis — Quantitative Phase Microscopy (DPC/QPM)

## Project goal

Research-grade quantitative phase imaging from LED-array DPC, **following the
reference implementation of Tian & Waller, *Optics Express* 23(9):11394 (2015)
as faithfully as possible.** The active approach reconstructs quantitative phase
φ (radians → OPL nm) from half-aperture brightfield images using the Waller-Lab
reference `DPCSolver` engine unchanged, with a thin extension for variable
illumination NA.

> **Direction change (2026-06-02).** The earlier *staged* pipeline (stages
> 01–05: dataset inspection → empty-field calibration → DPC frontend → WOTF →
> per-cell quantification) was **removed from `src/`**. It used empty-field
> flat-fielding + mean-balancing, which diverged from the paper's
> self-normalization. The maintained code is now the **faithful reference
> pipeline** below. The removed code remains in git history; the 20260513
> staged artifacts under `analysis/` are **deprecated** (kept, not maintained).

## What to read / run

- **`src/dpc/dpc_tian2015.py`** — **the baseline pipeline** (Tian & Waller 2015).
  Imports the Waller-Lab reference engine `example/tian2015/dpc_algorithm.py::DPCSolver`
  **verbatim** for normalization (per-image self-normalization, **no blank/empty
  needed**), WOTF generation, and Tikhonov inversion. The only extension is
  `VariableNADPCSolver.sourceGen`, which decouples illumination (source) NA from
  objective (pupil) NA so the Fig-5 σ = NA_illum/NA_obj sweep can be modeled
  (at NA_illum = NA_obj it is byte-identical to the reference). Standard 4
  half-circle (T/B/L/R) capture, ideal pupil assumed. Produces a per-NA output
  tree (DPC both axes, phase WOTF, reconstruction) and a Tikhonov α (reg_p)
  **L-curve sweep** with knee detection.
- **`src/dpc/dpc_chen2018_aberration.py`** — **aberration-correcting upgrade**
  (Chen, Phillips & Waller, Opt. Express 26:32888, 2018). Jointly recovers the
  complex field **and** the pupil aberration (Zernike coeffs) by alternating a
  closed-form Tikhonov field solve with an **analytic-gradient** L-BFGS pupil
  update. Self-contained (numpy + scipy only — a port of the reference *MATLAB*,
  since the Waller repo's Python ships standard DPC only, no aberration code).
  **Capture = 3 half-circles (top/bottom/left) + 1 single on-axis LED** (the LED
  replaces the 4th half-circle and carries the coherent aberration contrast);
  full half-disk illumination at NA_illum = NA_obj, **no annulus**. Includes a
  turnkey CLI runner (`python -m src.dpc.dpc_chen2018_aberration --data … --out …`).
- **`src/dpc/white_vs_green.py`** — secondary. White(FOV3) vs green(FOV4)
  comparison, the full paper-Fig-5 diagnostic chain, and a "2-axis" explainer.
- **`example/`** — third-party Waller-Lab reference (unmodified). See
  `example/README.md` for provenance, citations, and the (no-)license status.
- **`example/runnable/`** — **lab-authored** generic, data-agnostic runners
  (`dpc_tian2015_run.py`, `dpc_chen2018_run.py`) so anyone (built for Bill Jia,
  June 2026) can reconstruct either paper from their **own** 4-image capture
  (Tian = 4 half-circles; Chen = 3 half-circles + 1 on-axis LED). Unlike the lab
  `src/dpc/*` baseline these are NOT hard-wired to a dataset — all optics are
  CLI flags, files auto-match by tag, runs from any CWD. See
  `example/runnable/README.md` (incl. the GPU-question answer for Bill).

## Key concepts (locked this dataset)

- **Blank/empty is NOT required.** The reference `normalization()` self-normalizes
  each image (÷ uniform_filter local mean, ÷ mean, −1). `(I_T−I_B)/(I_T+I_B)` is
  itself self-normalizing.
- **The final result is the phase reconstruction φ (2-axis), not the DPC.** DPC
  is an intermediate, qualitative, direction-dependent image. φ (rad) →
  OPL = φ·λ/2π (nm) → dry mass.
- **2-axis** = combine top-bottom DPC + left-right DPC in one inversion
  (`solve()` uses all 4 sources). Each single axis has a zero LINE along its own
  direction; combining fills all but the origin. (Paper Fig 3 = the multi-axis
  study; Fig 5 = the single-axis illumination-NA σ study.)
- **Green (530 nm) for quantitative work.** SCI Dome green is datasheet-confirmed
  530 nm, so λ (→ pupil radius NA/λ and OPL) is well-defined. White BF (RGB[1,1,1])
  is broadband → λ ill-defined → qualitative only. After normalization white and
  green give nearly identical contrast; green's advantage is λ definability.
- **Tikhonov α** trades data-misfit vs solution norm; pick the L-curve knee.
  NA0.4 half-circle knee ≈ 1e-2 (matches the old pipeline's L-curve, independent
  cross-validation).
- **Half-annulus = darkfield for ALL NA tags → not BF-DPC reconstructable**
  (corrected 2026-06-03). The ring is a **fixed firmware command** `ha.<h>.50.95`
  (inner NA 0.50 / outer 0.95, clipped to dome 0.80) — **not** scaled by the
  filename NA tag; the NA02/NA04/NA08 in annulus filenames is an operator
  `snap_tag` note only. Source: microscope-PC `lighting.py _DEFAULT_UNIFIED_MODES`
  (all 5 annulus modes hardcode `[0.5, 0.95]` since 2026-05-15), corroborated by
  the raw images (annulus background = 2–9 % of full BF = darkfield; 4 halves
  differ → `ha.` accepted not disabled; uncorrelated with full BF → not blank).
  Since inner 0.50 > NA_obj 0.40, every annulus sits **outside the pupil**
  (B0 = 0), so `dpc_tian2015.py` skips them all. The **earlier** model assumed
  `inner = 0.7×filename_NA`, which wrongly reconstructed NA02/NA04 annulus as
  brightfield — those bogus outputs are quarantined under
  `analysis/deprecated_annulus_model/`. Using these annulus shots would need a
  separate darkfield-phase method.

## Hard rules

- **Never modify raw data.** Raw TIFFs + acquisition YAMLs are read-only.
- **Heavy artifacts live dataset-side, not in git.** Reconstructed φ/OPL arrays,
  figures, large CSVs go under `/data/Project_Data/QPM/<dataset>/analysis/`.
  Only `.py` under `src/`, `example/`, and this `README.md` stay in git.

## Data

- Active dataset: `/data/Project_Data/QPM/20260602_DPC_test/`
  - **Raw under `raw/`** (moved there 2026-06-02): `*.tif` + `*_acquisition_metadata.yaml`.
    Code finders search `raw/` first, top-level as fallback.
  - Outputs under `analysis/` (per-NA tree from `dpc_tian2015.py`,
    `white_vs_green/` from the comparison script).
  - FOV4 = green (RGB[0,1,0], 530 nm) — the quantitative channel. FOV1–3 = white.
  - Per NA (filename `NA02/NA04/NA08` = **illumination** NA): full BF, half-circle
    (th/bh/lh/rh), half-annulus (hat/hab/hal/har). FOV3 white uses full-word tags
    (`tophalf` …, `NA0X_1`).
- Deprecated dataset: `20260513_Hela_p15` (old staged pipeline; artifacts remain
  under `analysis/` but are not maintained).

## Hardware

- **Cephla Squid microscope + SCI Dome programmable LED matrix.**
  - SCI Dome: RGB LEDs **R 625 / G 530 / B 485 nm**, 793 LEDs, **max illumination
    NA 0.8**, working distance 65 mm.
  - Default white BF = RGB[1,1,1] (broadband); per-channel on/off control
    (e.g. green = [0,1,0]).
- **Objective: 10× / 0.40 NA** (real), 180 mm tube lens, 6.5 µm sensor pitch →
  effective pixel ≈ 0.65 µm.
  - ⚠ **NA metadata bug:** acquisition YAMLs record `NA: 0.3` (a stale Squid
    software default). The real objective NA is **0.40**; `dpc_tian2015.py` uses
    `NA_OBJ = 0.40`. Illumination NA is encoded in the **filename**, not metadata.
- LED patterns: full BF, half-circle (top/bottom/left/right), half-annulus
  (ha_top/bottom/left/right), variable illumination NA, variable BF intensity.

## How to run

```bash
conda activate fucci-analysis        # numpy, scipy, tifffile, pyyaml, matplotlib

cd /home/heeseok/github/jia-lab/qpm-analysis
python -m src.dpc.dpc_tian2015        # baseline: per-NA DPC/WOTF/phase + α L-curve sweep
python -m src.dpc.white_vs_green      # white vs green + full-Fig5 + 2-axis explainer

# aberration-corrected DPC (Chen 2018): capture 3 half-circles + 1 single LED, then:
python -m src.dpc.dpc_chen2018_aberration \
    --data /data/Project_Data/QPM/<dataset>/raw \
    --out  /data/Project_Data/QPM/<dataset>/analysis/aberration
#   (files auto-matched by tag: top/bottom/left + led; or pass
#    --top/--bottom/--left/--led explicitly. Defaults are 10x/NA0.40/530nm.)
```

Outputs go to `/data/Project_Data/QPM/20260602_DPC_test/analysis/`.

### Capture spec for aberration-corrected DPC (Chen 2018)

4 images, σ = 1 (NA_illum = NA_obj = 0.40), full half-disk (no annulus):

| # | pattern | rotation | role |
|---|---------|----------|------|
| 1 | top half-circle    | 0°   | DPC |
| 2 | bottom half-circle | 180° | DPC |
| 3 | left half-circle   | 90°  | DPC |
| 4 | single on-axis LED | —    | coherent aberration contrast (replaces the 270° half) |

This is **not** the standard 4-half (T/B/L/R) set: the *right* half is replaced
by the single central LED. Annulus modes are unused here.

## Output layout (dataset-side)

```
analysis/
  NA02/ NA04/ NA08/
    half_circle/  (+ half_annulus/ where reconstructable)
      dpc/             dpc_TB.tif, dpc_LR.tif, dpc_pair.png
      wotf/            phase_wotf.png (TB, LR, combined)
      reconstruction/  phase_alpha_<a>.tif ×9, phase_alpha_montage.png,
                       opl_knee_nm.tif, lcurve.png, alpha_sweep.csv
  figures/         fig5_style_na_sweep.png (cross-NA, fixed reg_p)
  tables/          global_summary.csv, params.json
  white_vs_green/  green_full_fig5.png, two_axis_explainer_NA04.png,
                   white_vs_green_<NA>.png, phase_stats.csv
```

## Environment

- Conda env: **`fucci-analysis`** (Python 3.10).
- Required: `numpy`, `scipy`, `tifffile`, `pyyaml`, `matplotlib`.

## Open items / next acquisitions

- **Matched same-FOV white+green pair** (none exists yet — FOV3 white, FOV4 green
  are different fields) to isolate the wavelength effect.
- **Phantom (known-OPL object)** for absolute B0 / dry-mass calibration; current
  phase is relative/radians, absolute scale uncalibrated.
- ~~**Half-annulus ring NA** from the dome config (replace the 0.7×outer
  assumption).~~ **Resolved 2026-06-03:** fixed ring `[0.50, 0.80]` (firmware
  `ha.<h>.50.95`), darkfield for all NA → skipped. See Key concepts. A
  darkfield-phase method is the open follow-up if these shots are to be used.
- Source is still a continuous half-disk/annulus (not the 793 discrete LEDs);
  pupil assumed ideal — same idealizations as the paper.
