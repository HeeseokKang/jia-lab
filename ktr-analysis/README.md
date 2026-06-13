# ktr-analysis — Longitudinal ERK-KTR + H2B live-cell pipeline

## Project goal

Add a research-grade **longitudinal layer** (tracking, lineage, per-track
KTR ratio, identity-aware QC) on top of live-cell ERK-KTR + H2B-BFP2
imaging. The reference dataset is the ~47-hour, two-well baseline timelapse
acquired 2026-05-05. The first-priority output of this module is **nuclear
segmentation + tracking quality** — every downstream stage inherits whatever
identity errors enter at the seg/track step, so this is where that quality is
established.

The tracking **substrate** is a fresh **StarDist H2B** segmentation, validated
against a human **lineage ground truth** (`<dataset>/analysis/tracking/gt_lineage_prep/`).
The StarDist masks are a substrate, *not* ground truth: accuracy is judged
against the human lineage GT, never against another model's per-frame masks.

## Hard rules

- **Never modify raw data.** Raw root
  `/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse/`
  is read-only from this module's perspective.
- **All heavy artifacts go under `<dataset>/analysis/`**, not in git (see
  Outputs policy). The repo carries code, Quarto narrative, and small text
  summaries only.
- **Prefer `.py` scripts and Quarto (`.qmd`) reports** over notebooks.
- **The longitudinal schema is fixed; the tracker is swappable.** Schema
  changes bump `TRACKING_POLICY_VERSION` in `analysis/tracking/meta.json`;
  tracker swaps (`trackpy` → `btrack` → `Ultrack`) do not change the schema.
- **Reports are embed-only.** Quarto chapters consume pre-computed CSV / PNG /
  mask arrays; they never re-run segmentation, tracking, or measurement.

## Outputs policy (lab-wide convention)

Canonical artifact root, mirrored across the lab (qpm-analysis follows the same
convention):

```
/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse/
└── analysis/
    ├── inspection/      ← stage 01: dataset enumeration, channel/tp QC
    ├── segmentation/    ← stage 02: validation masks, comparison tables, meta.json
    ├── tracking/        ← stage 03: trajectory parquet/csv, lineage graph, meta.json
    ├── ratio/           ← stage 04: per-track KTR C/N csv
    ├── qc/              ← stage 05: TRA, mitosis F1, switch logs
    ├── figures/         ← PNG / SVG referenced by qmd reports
    ├── tables/          ← small text summaries (per-region, per-condition)
    ├── reports/         ← rendered Quarto HTML
    └── logs/            ← per-stage logs
```

**Stays in git** (`ktr-analysis/`): `src/`, `analysis/*.qmd`, `tests/`,
`scripts/`, `environment.yml`, this README. **Stays out of git:** heavy arrays
(`*.npy`, `*.npz`, zarr), mask stacks, HTML renders, Python cruft.

Resolve canonical paths in new code via `configs.paths` (single source of truth
for filesystem paths).

## Longitudinal schema

Lab-wide for any pipeline that emits per-cell time series. FUCCI and future QPM
longitudinal work should mirror these columns.

| Column | Type | Meaning |
| --- | --- | --- |
| `region` | str | well / acquisition region (`R0`, `R1`) |
| `fov` | int | field of view index within region |
| `timepoint` | int | frame index, contiguous from 0 |
| `cell_in_frame_id` | int | per-frame instance id from segmenter |
| `track_id` | int | identity across time; stable for the lifetime of one cell |
| `lineage_id` | int | root ancestor's `track_id` for this lineage |
| `parent_track_id` | int or null | `null` at lineage root; set on daughter tracks after a mitosis |
| `generation` | int | 0 at lineage root; +1 per mitosis from root to this track |
| `mitosis_event_frame` | int or null | frame at which this track was born from parent; `null` for lineage roots |
| `centroid_x` | float | x centroid (px) |
| `centroid_y` | float | y centroid (px) |
| `area` | float | mask area (px²) |
| `nuclear_mask_path` | str | per-frame mask reference (zarr URI or PNG/npy path) |

**Tracker swap policy.** The tracker impl is recorded in
`analysis/tracking/meta.json` (`IMPL`, `TRACKING_POLICY_VERSION`); the schema
does not change when the tracker changes.
- `trackpy` — minimum-cost linking, no mitosis logic. Mitosis appears as a track
  break; daughter tracks lose `parent_track_id` until a mitosis-detection pass.
- `btrack` — Bayesian, mitosis-aware. Sets `parent_track_id` / `generation`
  natively.
- `Ultrack` — promoted to **primary tracker for the H2B GT** (segmentation
  hypothesis space); BF-side comparison deferred so that, when built, the same
  tracker runs on both sides and a BF-vs-H2B comparison isolates the *modality*
  effect, not a *tracker* effect.

## Data

- **Active dataset folder:**
  `/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse/`
- **Active acquisition:** `timelapse_2026-05-05_18-12-11.466141/`
- **Filename pattern:** `R{region}_{fov}_{z}_20260505_Heeseok_{channel}_KTR.tiff`
- **Channels:** `BF`, `mKate2` (ERK-KTR sensor), `mTagBFP2` (H2B nuclear marker).
  `mTagBFP2` is the input to nuclear segmentation; `mKate2` is the KTR sensor;
  `BF` is the label-free channel reserved for downstream label-free experiments.
- **Shape per timepoint:** 2 regions × 16 fovs × 3 channels = 96 TIFFs.
- **Acquisition spec** (`0/metadata.json`): 10×, pixel_size_um = 1.3,
  1200 × 1200, 1 z-level, time_increment_s = 300 (5 min → ~48 hr total),
  `INDIVIDUAL_IMAGES`. Note a 576-vs-567 reported-timepoint discrepancy to
  reconcile in Stage 01.
- **Condition:** all rows in `sample_info.csv` are `baseline` — no perturbation.
  The dataset is a baseline characterisation of resting KTR C/N ratio +
  inter-replicate variability.

## Hardware

Cephla Squid microscope, 10× objective (pixel_size_um = 1.3, sensor-scaled),
wide-field epifluorescence + brightfield, 1200 × 1200 ROI, MONO16 TIFF. Two
wells as biological replicates (R0, R1), 16 FOVs each, 5-minute interval,
~48-hour duration.

## Repo layout

```
ktr-analysis/
├── README.md                          ← this file
├── environment.yml                    fucci-analysis conda env baseline
├── .gitignore
├── data/
│   └── raw                            symlink → <dataset>/timelapse_<...>/
├── src/
│   ├── io.py                          stage-01 helper: parse_dataset, FOV loaders
│   ├── segmentation/                  StarDist H2B substrate + QC metrics
│   │   ├── stardist_validate.py       fresh StarDist seg on R0/fov0 × all tp
│   │   ├── stardist_finetune.py       transfer-learn StarDist on corrected H2B nuclei
│   │   ├── seg_correction.py          HITL napari false-negative recovery tool
│   │   ├── io_runtime.py              configs, paths, JSONL logger
│   │   └── metrics.py                 QC metric kit
│   ├── tracking/                      trackpy / btrack / Ultrack + GT lineage tooling
│   │   ├── gt_annotator.py            click-driven napari GT-lineage annotator
│   │   └── chunked_track.py           memory-bounded chunked tracking
│   ├── ratio/                         per-track KTR C/N
│   └── qc/                            longitudinal metrics
├── analysis/                          quarto chapters (narrative only)
│   └── 01..05_*.qmd                   per-stage reports
├── scripts/                           runners + QC renderers (incl. _remeasure/)
├── tests/
│   └── test_io.py                     schema integrity of parse_dataset
└── notebooks/                         exploratory only; do not commit outputs
```

## Environment

Shared conda env `fucci-analysis` (declared in `environment.yml`). Stage 02 adds
`stardist` + `csbdeep`; Stage 03 adds `trackpy` (declared) and `btrack`. GPU is
needed for fresh-StarDist on the validation FOV (~20 min for ~576 frames at
1200×1200, single FOV).

## How to run

All stages take `<data_root>` and `<analysis_root>` as positional args (mirrors
the qpm-analysis convention).

```bash
DATA=/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
OUT=$DATA/analysis

# Stage 01 — dataset inspection
python -m src.io.dataset_inspection $DATA $OUT
quarto render analysis/01_dataset_inspection.qmd

# Stage 02 — segmentation validation (StarDist H2B substrate)
python -m src.segmentation.stardist_validate $DATA $OUT --region R0 --fov 0
quarto render analysis/02_segmentation_validation.qmd

# Stage 03 — tracking
python -m src.tracking.run $DATA $OUT --impl trackpy   # v1
python -m src.tracking.run $DATA $OUT --impl btrack    # v2 (mitosis-aware)
quarto render analysis/03_tracking.qmd

# Stage 04 — per-track KTR ratio
python -m src.ratio.per_track $DATA $OUT
quarto render analysis/04_ktr_ratio.qmd

# Stage 05 — longitudinal QC
python -m src.qc.longitudinal $DATA $OUT
quarto render analysis/05_longitudinal_qc.qmd
```

## Stage contracts

### Stage 01 — Dataset inspection
**Consumed:** raw TIFFs + `<dataset>/metadata.json` + `sample_info.csv` +
`locations.csv` + `config.yaml`.
**Produced:** `inspection/dataset_index.parquet` (one row per TIFF),
`inspection/scene_summary.csv` (per-region/fov coverage),
`figures/preview_R0_fov0.png` (BF + H2B + mKate2 triptych),
`inspection/meta.json` (counts, channel mapping, anomalies). Resolves the
576-vs-567 timepoint discrepancy; verifies no missing TIFFs.

### Stage 02 — Segmentation validation
**Subset:** 1 region × 1 FOV × all timepoints (default `R0`, `fov 0`).
Fresh StarDist on H2B (`2D_versatile_fluo`) → int label masks under
`segmentation/validation/R0_fov0/stardist/<tp>.npy` + an overlay GIF.
The H2B segmentation is scored on **per-frame nuclear recovery** (IoU / merge /
split vs the human GT) and on **downstream trackability** (centroid-displacement
consistency, cross-frame bijection rate, size CV).

A parallel **BF whole-cell** track is scored by **containment/coverage**
(DAPI/H2B recall, missed-nuclei, nuclei-per-cell, assignment uniqueness) — not
nucleus IoU, which is a category error on a whole-cell mask. BF is a label-free
identity benchmark, additive to (never a replacement for) the biological KTR
readout.

### Stage 03 — Tracking
**Consumed:** the StarDist H2B masks.
**Produced:** `tracking/trajectories.parquet` (full schema),
`tracking/lineage.graph.json` (directed graph), `tracking/meta.json` (`IMPL`,
`TRACKING_POLICY_VERSION`, params, runtime). v1 (trackpy) emits valid
`track_id` but `parent_track_id` may be `null` across mitoses; v2 (btrack) /
Ultrack fill lineage natively.

### Stage 04 — Per-track KTR ratio
**Consumed:** trajectories + masks + mKate2 arrays.
**Produced:** `ratio/per_track_ratio.parquet` (long format,
`(track_id, timepoint) → ktr_cn_ratio`), `ratio/meta.json` (cyto-ring expansion
parameter, background-subtraction policy), `figures/per_track_ratio_examples.png`.

### Stage 05 — Longitudinal QC
**Consumed:** trajectories + masks + (optionally) manual annotations.
**Produced:** `qc/tra.csv` (Cell Tracking Challenge TRA per region/fov),
`qc/mitosis_f1.csv`, `qc/identity_switches.csv`, `qc/lineage_purity.csv`,
`qc/drift.csv`, plus supporting `figures/qc_*.png`.

## Current status

| Stage | Status |
| --- | --- |
| 01 — Dataset inspection | `src/io.py` has `parse_dataset`; needs wrapping into the stage-01 entrypoint |
| 02 — Segmentation validation | StarDist H2B substrate adopted; smoke done, full sweep + fine-tune in progress |
| 03 — Tracking | trackpy declared; Ultrack the primary GT tracker; btrack not yet wired |
| 04 — Per-track ratio | `measure_ktr_ratio.py` implemented; ratio/background policy still exploratory |
| 05 — Longitudinal QC | not started (TRA / mitosis F1 / drift metrics) |

## Next steps (priority order)

1. **Stage 01** — port `src/io.py:parse_dataset` into `src.io.dataset_inspection`,
   write to `<dataset>/analysis/inspection/`, reconcile the 576-vs-567 timepoint
   discrepancy, draft `01_dataset_inspection.qmd`.
2. **Stage 02** — full-FOV StarDist H2B sweep + fine-tune for false-negative
   recovery; validate against the human lineage GT.
3. **Stage 03** — trackpy v1 baseline, then Ultrack / btrack for lineage-aware
   tracking; compare identity-switch and lineage metrics.
4. **Stage 04 + 05** — per-track KTR ratio + longitudinal QC.
</content>
