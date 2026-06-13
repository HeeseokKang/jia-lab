# fucci-analysis

FUCCI / PIP-FUCCI cell-cycle reporter analysis (Bill Jia Lab, UCSF). Two distinct
experiments live here:

1. **Variant characterization** (current main path) — `20260610_fucci_constructs_test`.
2. **Single-well timelapse validation** (earlier precursor) — `20260413_FUCCI_Timelapse`.

See `src/README.md` for the per-script pipeline order. Heavy artifacts (masks,
figures, large CSVs) live dataset-side under `<dataset>/analysis/`, not in git.

---

## 1. Variant characterization (current) — `variant_*.py`

Snapshot (NOT timelapse) comparison of PIP-FUCCI variant constructs
(pBJ281 / 284 / 285) against the published PIP-FUCCI reference **D900**:
4 constructs × 3 drugs (DMSO / Palbociclib / RO-3306) × 3 replicates = 36 wells,
one BF + two reporter channels per well.

**Reporter biology (the interpretation framework).** These are PIP-FUCCI–type:
green/488 (PIP degron) is a **direct S-phase sensor** — high in G1 *and* G2, low
in S; red (2nd reporter) is **Geminin** (S/G2/M). Phase is therefore read from a
**2D quadrant gate** on the green-vs-red scatter, not a 1D ratio:

| quadrant | green (PIP) | red (Geminin) | phase |
|---|---|---|---|
| green-only | on | off | **G1** |
| red-only | off | on | **S** |
| double-positive | on | on | **G2** |

Drug expectation: Palbociclib (CDK4/6i) → ↑ G1 fraction; RO-3306 (CDK1i) → ↑ G2
(double-positive) fraction.

**Pipeline (Stage 0 → 4-6), in order:**

| Script | Stage | Role |
|---|---|---|
| `variant_config.py` | — | single source of truth (well layout, channels, gate thresholds) |
| `variant_index.py` | 0 | build/validate the 36-well snapshot manifest from filenames |
| `variant_segment.py` | 1 | Cellpose-SAM BF segmentation → per-well label masks |
| `variant_measure.py` | 3 | per-cell reporter intensity, per-image **local background** + erosion sweep, SNR |
| `variant_analyze.py` | 4-6 | 2D quadrant gating, drug-response (phase-fraction shift), cross-construct comparison |

**Headline result.** D900 is the only construct whose red (Geminin) channel is
bright enough to gate S vs G2; it behaves textbook (Palbo → G1, RO → G2). The
variants' 2nd reporters are too dim to resolve S/G2 (281 = emiRFP670 weak,
284 = mTagBFP2 buried in 405 autofluorescence, 285 = SiRhP dye not loaded).

> Background is handled by **per-cell local background subtraction**, not
> flat-field correction. (An earlier t0-reference flat-field prototype was found
> to suppress FUCCI signal — see the precursor section below — and was removed.)

---

## 2. Single-well timelapse validation (earlier precursor)

Cell-cycle pipeline on `20260413_FUCCI_Timelapse` (32 wells × 67 timepoints ×
`BF`/`561`/`647`), scoped to single-well validation on `R1_1`. Cellpose-SAM
segmentation → trackpy linking → per-track nuclear FUCCI intensity:

- `timeseries_one_well.py` — per-frame BF masks
- `tracking_one_well.py` — frame-to-frame linking
- `nuclear_intensity_one_well.py` — per-track 561/647 in a 3 px-eroded nuclear region (uses **raw 647**)

Findings and inventory: `analysis/20260413_validation/README.md`. Key result —
the t0-reference flat-field correction flattened 647 cell-to-cell variation
(`std_raw / std_corrected ≈ 52.6×`), so the pipeline uses raw 647; this is what
later motivated the local-background approach in the variant pipeline.

---

## How to run

```bash
conda activate fucci-analysis
python fucci-analysis/src/<script>.py        # see src/README.md for order + CLI
```

Cellpose-SAM scripts auto-use the Dodo GPU. Large derived data (masks, corrected
stacks) is dataset-side / gitignored — recreate with the relevant script.
