"""
white_vs_green.py
=================
White (FOV3, RGB[1,1,1] broadband) vs Green (FOV4, RGB[0,1,0], 530 nm) DPC/QPM
comparison on the 2026-06-02 dataset, reusing the faithful reference engine.

IMPORTANT CAVEAT — NOT a matched same-FOV pair.
    FOV3 (white) and FOV4 (green) are DIFFERENT physical fields of view (different
    cells). No FOV in this dataset was imaged in BOTH white and green. So this
    comparison shows the channel/wavelength *qualitative* difference but is
    confounded by (a) different cells and (b) different exposure (white FOV3:
    NA02 20ms / NA04 5ms / NA08 10ms; green FOV4: 20ms uniform). A clean
    wavelength-only comparison needs a matched same-FOV reacquisition.

White is reconstructed at a NOMINAL lambda = 530 nm (its green sub-band); for a
true broadband source lambda is ill-defined, which is exactly why green-only is
preferred for quantitative phase.

Outputs: /data/Project_Data/QPM/20260602_DPC_test/analysis/white_vs_green/
    white_vs_green_<NA>.png   (rows white/green x cols BF/DPC/phase)
    green_full_fig5.png       (full paper-Fig5 row set for green)
    phase_stats.csv
"""

import os
import glob
import csv

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dpc.dpc_tian2015 import (
    VariableNADPCSolver, WAVELENGTH, NA_OBJ, PIXEL_SIZE, ROTATION,
    NA_ILLUM, REG_U, _disp, F,
)

DATA_DIR = "/data/Project_Data/QPM/20260602_DPC_test"
OUT = os.path.join(DATA_DIR, "analysis", "white_vs_green")
REG_P = 1e-2   # fixed (NA04 knee) for a fair white/green comparison

CH = {
    "white": dict(
        label="white  FOV3  RGB[1,1,1] broadband",
        na={"NA02": "NA02_1", "NA04": "NA04_1", "NA08": "NA08_1"},
        tags={"th": "tophalf", "bh": "bottomhalf", "lh": "lefthalf",
              "rh": "righthalf", "full": "full"},
        glob="*FOV3_BF_{tag}_{na}*.tif"),
    "green": dict(
        label="green  FOV4  RGB[0,1,0]  530nm",
        na={"NA02": "NA02", "NA04": "NA04", "NA08": "NA08"},
        tags={"th": "th", "bh": "bh", "lh": "lh", "rh": "rh", "full": "full"},
        glob="*FOV4_green_BF_{tag}_{na}*.tif"),
}


def find(ch, tagkey, na):
    c = CH[ch]
    pat = c["glob"].format(tag=c["tags"][tagkey], na=c["na"][na])
    # raw TIFFs may live directly under the dataset or under a raw/ subfolder
    hits = []
    for base in (os.path.join(DATA_DIR, "raw"), DATA_DIR):
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            break
    if len(hits) != 1:
        raise FileNotFoundError(f"{ch} {tagkey} {na}: {hits}")
    return hits[0]


def recon(ch, na, reg_p=REG_P):
    th, bh, lh, rh = [tifffile.imread(find(ch, t, na)).astype("float64")
                      for t in ["th", "bh", "lh", "rh"]]
    stack = np.asarray([th, bh, lh, rh])      # order = rotation [0,180,90,270]
    solver = VariableNADPCSolver(stack, WAVELENGTH, NA_OBJ, 0.0, PIXEL_SIZE,
                                 ROTATION, na_source=NA_ILLUM[na])
    solver.setTikhonovRegularization(REG_U, reg_p)
    phase = solver.solve()[0].imag            # 2-axis joint (uses all 4 sources)
    bf = tifffile.imread(find(ch, "full", na)).astype("float64")
    dpc_tb = (th - bh) / (th + bh + 1e-9)
    dpc_lr = (lh - rh) / (lh + rh + 1e-9)
    # Hp[0] = top source (TB-axis WOTF), Hp[2] = left source (LR-axis WOTF)
    return dict(phase=phase, bf=bf, dpc_tb=dpc_tb, dpc_lr=dpc_lr,
                Hp_tb=solver.Hp[0], Hp_lr=solver.Hp[2], solver=solver)


def per_na_compare(na, rows):
    w = recon("white", na)
    g = recon("green", na)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    for i, (ch, r) in enumerate([("white", w), ("green", g)]):
        ax[i, 0].imshow(_disp(r["bf"]), cmap="gray")
        ax[i, 1].imshow(r["dpc_lr"], cmap="gray", clim=[-0.2, 0.2])
        ax[i, 2].imshow(r["phase"], cmap="gray", clim=[-1.0, 1.0])
        ax[i, 0].set_ylabel(CH[ch]["label"], fontsize=11)
        for j in range(3):
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, t in enumerate(["raw BF (full)", "DPC (left-right)",
                           "phase reconstruction (reg_p=1e-2)"]):
        ax[0, j].set_title(t, fontsize=12)
    fig.suptitle(f"{na} (σ={NA_ILLUM[na]/NA_OBJ:.2f})  —  white(FOV3) vs green(FOV4)  "
                 f"[DIFFERENT fields, not matched]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, f"white_vs_green_{na}.png"), dpi=120,
                bbox_inches="tight"); plt.close(fig)

    for ch, r in [("white", w), ("green", g)]:
        rows.append(dict(na=na, sigma=round(NA_ILLUM[na]/NA_OBJ, 3), channel=ch,
                         bf_mean=float(r["bf"].mean()),
                         dpc_lr_std=float(r["dpc_lr"].std()),
                         phase_std=float(r["phase"].std()),
                         phase_p1=float(np.percentile(r["phase"], 1)),
                         phase_p99=float(np.percentile(r["phase"], 99))))


def green_full_fig5():
    """Full paper-Fig-5 chain for green, showing BOTH DPC axes: BF, DPC top-bottom,
    DPC left-right, phase WOTF (TB), phase WOTF (LR), phase reconstruction (2-axis).
    cols = NA sweep."""
    nas = ["NA02", "NA04", "NA08"]
    recs = {na: recon("green", na) for na in nas}
    rowfns = [
        ("raw BF (full)",
         lambda r: (_disp(r["bf"]), "gray", None)),
        ("DPC top-bottom (intermediate)",
         lambda r: (r["dpc_tb"], "gray", [-0.2, 0.2])),
        ("DPC left-right (intermediate)",
         lambda r: (r["dpc_lr"], "gray", [-0.2, 0.2])),
        ("Phase WOTF $H_p$ top-bottom",
         lambda r: (np.fft.fftshift(r["Hp_tb"].imag), "jet", [-0.8, 0.8])),
        ("Phase WOTF $H_p$ left-right",
         lambda r: (np.fft.fftshift(r["Hp_lr"].imag), "jet", [-0.8, 0.8])),
        ("Phase reconstruction (2-axis, FINAL)",
         lambda r: (r["phase"], "gray", [-1.0, 1.0])),
    ]
    fig, ax = plt.subplots(len(rowfns), len(nas), figsize=(4*len(nas), 4*len(rowfns)))
    for j, na in enumerate(nas):
        ax[0, j].set_title(f"{na}  σ={NA_ILLUM[na]/NA_OBJ:.2f}", fontsize=12)
        for i, (_, fn) in enumerate(rowfns):
            img, cm, cl = fn(recs[na])
            ax[i, j].imshow(img, cmap=cm, clim=cl); ax[i, j].axis("off")
    for i, (t, _) in enumerate(rowfns):
        ax[i, 0].text(-0.08, 0.5, t, rotation=90, va="center", ha="right",
                      transform=ax[i, 0].transAxes, fontsize=11)
    fig.suptitle("Green (530nm) — full Fig5 chain, BOTH DPC axes. Final = bottom row "
                 "(2-axis quantitative phase); middle rows are DIAGNOSTICS.", fontsize=13)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "green_full_fig5.png"), dpi=120,
                bbox_inches="tight"); plt.close(fig)


def two_axis_explainer(na="NA04"):
    """Why two axes: each single-axis DPC has a zero LINE along its own axis
    (TB misses the horizontal line, LR the vertical); combining fills everything
    but the origin. Reproduces the lesson of paper Fig 3 on green NA04."""
    r = recon("green", na)
    fs = lambda x: np.log(np.abs(np.fft.fftshift(F(x))) + 1e-3)
    comb = np.fft.fftshift(np.sqrt(np.abs(r["Hp_tb"]) ** 2 + np.abs(r["Hp_lr"]) ** 2))
    cols = [
        ("top-bottom axis", r["dpc_tb"], fs(r["dpc_tb"]),
         np.fft.fftshift(r["Hp_tb"].imag)),
        ("left-right axis", r["dpc_lr"], fs(r["dpc_lr"]),
         np.fft.fftshift(r["Hp_lr"].imag)),
    ]
    fig, ax = plt.subplots(3, 3, figsize=(13, 13))
    for j, (title, dpc, spec, tf) in enumerate(cols):
        ax[0, j].imshow(dpc, cmap="gray", clim=[-0.2, 0.2]); ax[0, j].set_title(
            f"{title}\nDPC image")
        ax[1, j].imshow(spec, cmap="viridis"); ax[1, j].set_title("DPC Fourier spectrum\n(dark = missing-freq line)")
        ax[2, j].imshow(tf, cmap="jet", clim=[-0.8, 0.8]); ax[2, j].set_title("Phase WOTF (zero line on axis)")
    # third column = 2-axis combination
    ax[0, 2].imshow(r["phase"], cmap="gray", clim=[-1, 1])
    ax[0, 2].set_title("2-AXIS combined\nphase reconstruction (FINAL)")
    ax[1, 2].imshow(comb, cmap="viridis"); ax[1, 2].set_title("Combined coverage\n$\\sqrt{|H_{TB}|^2+|H_{LR}|^2}$ (only origin missing)")
    ax[2, 2].axis("off")
    ax[2, 2].text(0.5, 0.5,
                  "Single axis → a zero LINE\nalong its own direction.\n\n"
                  "TB + LR (= 2-axis) fills the\ngaps → isotropic phase.\n\n"
                  "This is paper Fig 3's point.",
                  ha="center", va="center", fontsize=12)
    for a in ax.flat:
        if a.has_data():
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"What '2-axis' means — green {na} (σ={NA_ILLUM[na]/NA_OBJ:.2f})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, f"two_axis_explainer_{na}.png"), dpi=120,
                bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for na in ["NA02", "NA04", "NA08"]:
        print(f"[compare] {na}")
        per_na_compare(na, rows)
    green_full_fig5()
    two_axis_explainer("NA04")
    with open(os.path.join(OUT, "phase_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print("\n[done]", OUT)
    for r in rows:
        print(f"  {r['na']} {r['channel']:5s}: BF_mean={r['bf_mean']:.0f}  "
              f"DPC_std={r['dpc_lr_std']:.4f}  phase_std={r['phase_std']:.4f}")


if __name__ == "__main__":
    main()
