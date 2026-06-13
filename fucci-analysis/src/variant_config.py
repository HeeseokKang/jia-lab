"""Single source of truth for the 20260610 FUCCI variant-characterization snapshot run.

Snapshot dataset (NOT timelapse): 4 constructs x 3 drugs x 3 replicates = 36 wells,
one BF + two reporter snapshots per well. Reporter->phase mapping is a
PLASMID-SUPPORTED WORKING MODEL (architecture PIP-SuperNLS-mStayGold-P2A-
[2nd reporter]-Gem1(1-110)) -> 488/green = PIP/G1, 2nd reporter = Geminin/S-G2-M.
Not independently confirmed; label interpretations accordingly.

Design doc: ResearchRuntime/Voltage_CellCycle/fucci/reports/
20260610_variant_characterization_design.md
"""

from __future__ import annotations

from pathlib import Path

# --- paths (results are dataset-side, per Heeseok 2026-06-10) ---------------
DATASET_DIR = Path(
    "/data/Project_Data/Voltage_CellCycle/20260610_fucci_constructs_test"
)
ANALYSIS_DIR = DATASET_DIR / "analysis"          # light CSV/PNG/logs (small)
MASKS_DIR = ANALYSIS_DIR / "masks"               # heavy BF label masks (dataset-side)
FIG_DIR = ANALYSIS_DIR / "figures"
QC_DIR = ANALYSIS_DIR / "qc"

# --- optics -----------------------------------------------------------------
# Physical objective is 10x / NA 0.4. The acquisition sidecar yaml records
# objective_details.NA: 0.3 -- treat that as a STALE microscope-config field and
# surface it as a QC note; use 0.4 for any optics-dependent reasoning.
OBJECTIVE_NA = 0.4
SIDECAR_NA = 0.3  # what the yaml says (stale); kept for the QC discrepancy note
PIXEL_UM = 6.5 / 10.0  # sensor 6.5 um / 10x = 0.65 um/px

# --- channel exposures (ms), fixed per channel; all illumination at 5 % ------
EXPOSURE_MS = {"BF": 10.0, "488": 50.0, "561": 75.0, "635": 100.0, "405": 100.0}

# --- drug by plate column (rows are the 3 biological replicates) -------------
DRUG_BY_COL = {2: "DMSO", 3: "Palbo", 4: "RO", 5: "DMSO", 6: "Palbo", 7: "RO"}

# --- construct layout + reporter roles --------------------------------------
# green = 488 (StayGold / PIP-mVenus)  -> PIP/G1 reporter
# red   = 2nd reporter channel         -> Geminin / S-G2-M reporter
CONSTRUCTS = {
    "pBJ281": {"green": "488", "red": "635", "red_fluor": "emiRFP670",
               "rows": ["B", "D", "F"], "cols": [2, 3, 4]},
    "pBJ284": {"green": "488", "red": "405", "red_fluor": "mTagBFP2",
               "rows": ["B", "D", "F"], "cols": [5, 6, 7]},
    "pBJ285": {"green": "488", "red": "635", "red_fluor": "Rhobin2_SiRhP",
               "rows": ["C", "E", "G"], "cols": [2, 3, 4]},
    "D900":   {"green": "488", "red": "561", "red_fluor": "mCherry-Geminin",
               "rows": ["C", "E", "G"], "cols": [5, 6, 7]},
}

# D900 was imaged twice: an early exposure/intensity TITRATION sweep (skip) and a
# consistent BENCHMARK set at the end. Only D900 snapshots at/after this wall-clock
# time belong to the benchmark set used for analysis.
D900_BENCHMARK_MIN_HHMMSS = "12-25-00"

# Caveat to honor in any pBJ285 interpretation: SiRhP far-red looked weak at
# acquisition; do NOT assume successful labeling.
PBJ285_SIRHP_LABELING_UNCERTAIN = True

# --- measurement defaults ---------------------------------------------------
EROSION_SWEEP_PX = [1, 3, 5, 7, 9]   # extended sweep (Heeseok 2026-06-10): trade
EROSION_DEFAULT_PX = 7               # nuclear-dilution vs small-cell loss; revisit per diag

# --- 2D quadrant gating (PRIMARY readout, Heeseok 2026-06-10 eve) ------------
# Literature-grounded reframe (library/cell_cycle/reporters/notes.md): our
# constructs are PIP-FUCCI-type. Green/488 = PIP = a DIRECT S-PHASE sensor
# (HIGH in G1 *and* G2, LOW in S); red = Geminin (S/G2/M). The correct readout
# is therefore a 2D quadrant gate on the green/red scatter, NOT the 1D
# log2(red/green) ratio (which mis-models G2 as a middle value and under-detects
# the RO->G2 arrest). Per-channel "on" = SNR >= GATE_SNR (signal >= 2*bg_sigma
# above local background); the quadrant map is:
#   green-only  -> G1   |  red-only -> S   |  double-positive -> G2   |  M = diffuse
# Drug expectation:  Palbo (CDK4/6i) -> G1 arrest -> up G1-frac
#                    RO    (CDK1i)   -> G2 arrest -> up G2-frac (double-positive)
GATE_SNR = 2.0
# A construct's red channel can only resolve S vs G2 if red turns on in a real
# fraction of an asynchronous (DMSO) population. Below this DMSO red bright-frac
# the 2-colour gate collapses to green-only (G1) and cannot call S/G2 -> flag
# the construct as not red-gateable rather than fabricating phase calls.
MIN_RED_GATE_FRAC = 0.05

# pBJ285 SiRhP far-red is at background -> treat as LABELLING FAILED: hold all
# red-channel / drug-response conclusions for pBJ285; report green/488 only.
PBJ285_LABELLING_FAILED = True

ALL_ROWS = ["B", "D", "F", "C", "E", "G"]


def replicate_index(construct: str, row: str) -> int:
    """1-based replicate id = position of `row` in the construct's row list."""
    return CONSTRUCTS[construct]["rows"].index(row) + 1


def expected_wells() -> list[tuple[str, str, int]]:
    """(construct, row, col) for all 36 analysis wells."""
    out = []
    for name, c in CONSTRUCTS.items():
        for r in c["rows"]:
            for col in c["cols"]:
                out.append((name, r, col))
    return out
