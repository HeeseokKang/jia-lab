"""
ppt_na_compare.py
=================
One-row, four-column PPT comparison figure for the 2026-06-02 green NA sweep:

    [ BF_full (NA0.2) | phase NA0.2 @knee | phase NA0.4 @knee | phase NA0.8 @knee ]

* Col 1: raw full-brightfield reference (NA0.2 = most visible by eye).
* Cols 2-4: half-circle 2-axis phase reconstruction at each NA's chosen knee
  alpha, read straight from the committed analysis tree (no re-solving).

Knee alphas come from analysis/tables/global_summary.csv, so they track the
pipeline's choice INCLUDING any manual KNEE_OVERRIDE in dpc_tian2015.py
(NA08 was overridden to 3e-3 on 2026-06-04; the auto heuristic over-regularized).

Display: each phase panel gets its OWN grayscale clim = [black point, white point]
in radians, set to that panel's 1st / 99th percentile (a robust min/max that
ignores the brightest/darkest 1% of pixels). The exact clim numbers are printed
in each panel title so the scaling is explicit and reproducible. They differ per
panel because the three NA reconstructions use different knee regularizations
(alpha = 0.03 / 0.01 / 0.003) => different absolute phase magnitudes, which are
therefore NOT directly comparable; this view compares resolution / detail / noise
at each NA's best reconstruction. The brightfield panel uses its own clim too.

Run:  conda activate fucci-analysis
      cd /home/heeseok/github/jia-lab/qpm-analysis
      python -m src.dpc.ppt_na_compare
"""

import os
import csv
import glob

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/data/Project_Data/QPM/20260602_DPC_test"
ANALYSIS = os.path.join(DATA_DIR, "analysis")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
NA_OBJ   = 0.40
NA_ILLUM = {"NA02": 0.20, "NA04": 0.40, "NA08": 0.80}
ORDER    = ["NA02", "NA04", "NA08"]
OUT_PNG  = os.path.join(ANALYSIS, "figures", "ppt_na_compare_1x4.png")


def disp(img, p=(1, 99)):
    lo, hi = np.percentile(img, p)
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def knee_alpha_map():
    """{'NA02': 0.03, ...} from the pipeline's global summary (half_circle rows)."""
    out = {}
    with open(os.path.join(ANALYSIS, "tables", "global_summary.csv")) as f:
        for row in csv.DictReader(f):
            if row["pattern"] == "half_circle" and row["reconstructable"] == "True":
                out[row["na_tag"]] = float(row["knee_alpha"])
    return out


def phase_at_knee(na_tag, alpha):
    """Load phase_alpha_<a>.tif for the knee alpha (filename uses '%.0e')."""
    fname = f"phase_alpha_{alpha:.0e}.tif"
    path = os.path.join(ANALYSIS, na_tag, "half_circle", "reconstruction", fname)
    return tifffile.imread(path).astype("float64")


def bf_full(na_tag):
    hits = glob.glob(os.path.join(RAW_DIR, f"*FOV4_green_BF_full_{na_tag}*.tif"))
    if len(hits) != 1:
        raise FileNotFoundError(f"BF_full {na_tag}: {hits}")
    return tifffile.imread(hits[0]).astype("float64")


def main():
    knees = knee_alpha_map()
    phases = {na: phase_at_knee(na, knees[na]) for na in ORDER}

    fig, ax = plt.subplots(1, 4, figsize=(18, 5.6))
    bf = bf_full("NA02")
    blo, bhi = np.percentile(bf, [1, 99])
    ax[0].imshow(bf, cmap="gray", clim=[blo, bhi])
    ax[0].set_title(f"Brightfield (NA 0.2 ref)\nclim [{blo:.0f}, {bhi:.0f}] counts", fontsize=13)
    for k, na in enumerate(ORDER, start=1):
        lo, hi = np.percentile(phases[na], [1, 99])           # per-panel clim (1/99 pct)
        ax[k].imshow(phases[na], cmap="gray", clim=[lo, hi])
        ax[k].set_title(f"Phase  NA {NA_ILLUM[na]:.1f}  (α={knees[na]:.0e})\n"
                        f"clim [{lo:+.2f}, {hi:+.2f}] rad", fontsize=13)
    for a in ax:
        a.axis("off")
    fig.text(0.5, 0.045,
             "Grayscale clim = [black point, white point], set per panel to its 1st–99th "
             "percentile (robust min/max). Panels use different knee α, so absolute",
             ha="center", fontsize=10.5, color="0.25")
    fig.text(0.5, 0.015,
             "phase magnitudes are not directly comparable — compare resolution / detail / "
             "noise, not brightness.",
             ha="center", fontsize=10.5, color="0.25")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}  (knees={knees}, per-panel clim shown)")


if __name__ == "__main__":
    main()
