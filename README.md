# Jia Lab Image Analysis Pipelines

Lead: Heeseok Kang
Advisor: Bill Jia (UCSF)

## Project

Research-grade microscopy analysis for live-cell **cell-cycle** and **label-free**
imaging. The repo houses several independent pipelines that share one convention:
**code, small text/CSV summaries, and design docs live in git; heavy artifacts
(masks, reconstructed arrays, figures, large CSVs) live dataset-side under
`/data/Project_Data/.../analysis/`**, never in the repo.

## Pipelines

| Directory | What it does | Entry doc |
|---|---|---|
| `fucci-analysis/` | FUCCI / PIP-FUCCI cell-cycle reporters — variant characterization (snapshot, 2D quadrant gating) + an earlier single-well timelapse validation | `fucci-analysis/README.md` |
| `ktr-analysis/` | Longitudinal ERK-KTR + H2B live-cell pipeline — nuclear segmentation, tracking, lineage, per-track KTR C/N ratio | `ktr-analysis/CLAUDE.md` |
| `qpm-analysis/` | Quantitative phase microscopy (LED-array DPC/QPM) — faithful Tian-2015 reconstruction + Chen-2018 aberration-correcting variant | `qpm-analysis/CLAUDE.md` |
| `configs/` | Canonical data paths and runtime constants (single source of truth) | `configs/README.md` |
| `shared/` | Helper modules used across pipelines (channel detection, file-list cleanup) | `shared/README.md` |

Each pipeline is self-contained: read its own `README.md` / `CLAUDE.md` for the
dataset, stage order, and per-script description.

## Data paths on Dodo

- Datasets / heavy artifacts: `/data/Project_Data/<Project>/<dataset>/`
  (raw is read-only; outputs go under each dataset's `analysis/`).
- NAS (canonical raw): `/mnt/nas1/Projects/...`

Resolve filesystem paths through `configs/paths.py` rather than hard-coding.

## How to run

Pipelines run as standalone modules/scripts from the repo root on Dodo, under
their conda environment (see each pipeline's doc). GPU stages (Cellpose-SAM,
StarDist) auto-detect the Dodo GPU.

```bash
conda activate <env>            # e.g. fucci-analysis, ktr-segtrack, ultrack_env
python <pipeline>/src/<script>.py
```

## Branching

`main` is the integration branch (merged by Heeseok). Feature work lands on
topic branches; the `hermes-handoff` orphan branch is a code-free transport
channel for vault handoff notes and is **never** merged into `main`.
