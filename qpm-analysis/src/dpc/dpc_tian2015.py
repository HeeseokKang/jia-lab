"""
dpc_tian2015.py
===============
Reference-faithful DPC / quantitative-phase reconstruction following
Tian & Waller, *Optics Express* 23(9):11394 (2015), applied to the
2026-06-02 **green-channel (530 nm)** FOV4 NA-sweep dataset.

Design contract ("paper implementation as-is")
----------------------------------------------
* The Waller-lab reference engine ``example/dpc_algorithm.py::DPCSolver`` is
  imported **verbatim and used unchanged** for:
    - ``normalization``  (per-image self-normalization by a large uniform_filter
      local mean -> /mean -> -1; i.e. **NO blank/empty division is required**),
    - ``WOTFGen``        (Eq 6-8 weak-object transfer functions),
    - ``solve``          (Eq 12-13 joint absorption+phase 2x2 Tikhonov inversion).
* The **only** deviation is ``VariableNADPCSolver.sourceGen``, which decouples
  the illumination (source) NA from the objective (pupil) NA so the NA sweep
  (paper Fig 5, sigma = NA_illum / NA_obj) can be modeled. When
  ``na_source == na`` it reduces **exactly** to the reference sourceGen.

System parameters (this dataset)
--------------------------------
* lambda = 0.530 um  -- SCI Dome GREEN channel, datasheet-confirmed (not a fallback).
* objective 10x / 0.40 NA (real; acquisition YAML mis-records 0.3, a Squid default).
* camera pixel 6.5 um, mag 10x  ->  effective pixel 0.65 um.
* DPC order [top, bottom, left, right] matches rotation [0, 180, 90, 270] deg.
* illumination NA is in the FILENAME (NA02/NA04/NA08), not metadata -- but this
  tag is meaningful ONLY for half_circle/full shots (operator dials the global
  Array-NA to match). For half_annulus the tag is an operator note only: the
  firmware always emits the fixed ring ha.<h>.50.95 (inner 0.50 / outer 0.80),
  which sits entirely outside the NA_OBJ=0.40 pupil -> darkfield, NOT BF-DPC
  reconstructable. All half_annulus patterns are therefore skipped.

Output layout (per-NA folders, dataset-side under analysis/)
------------------------------------------------------------
    /data/Project_Data/QPM/20260602_DPC_test/analysis/
        NA02/ NA04/ NA08/
            half_circle/  (and half_annulus/ where reconstructable)
                dpc/             dpc_LR.tif, dpc_TB.tif, dpc_pair.png
                wotf/            phase_wotf.png  (LR, TB, combined)
                reconstruction/  phase_alpha_<a>.tif (alpha sweep) + .png montage,
                                 opl_knee_nm.tif, lcurve.png, alpha_sweep.csv
        figures/  fig5_style_na_sweep.png, fig5_style_na_sweep_annulus.png  (cross-NA)
        tables/   global_summary.csv, params.json

The alpha (Tikhonov reg_p) L-curve sweep finds, per (NA, pattern), the corner
(knee) alpha that best trades data-misfit against solution norm.
"""

import os
import sys
import glob
import json
import csv
import shutil

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# import the reference engine VERBATIM (no copy, no edit)
#
# We pin to the Tian & Waller 2015 reference (``example/tian2015/``). The
# Chen-Phillips-Waller 2018 code (``example/chen2018_aberration/``) is kept
# alongside for reference only -- it is NOT a drop-in replacement: its
# ``setRegularizationParameters``/``solve(method=...)`` API differs from the
# 2015 engine this pipeline (and ``VariableNADPCSolver``) is built on.
# ---------------------------------------------------------------------------
_EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "example", "tian2015")
)
sys.path.insert(0, _EXAMPLE_DIR)
from dpc_algorithm import DPCSolver, pupilGen, naxis, F  # noqa: E402

# ---------------------------------------------------------------------------
# system constants
# ---------------------------------------------------------------------------
WAVELENGTH = 0.530          # um, SCI Dome green (datasheet)
NA_OBJ     = 0.40           # objective NA (10x/0.4)
NA_DOME    = 0.80           # SCI Dome maximum illumination NA (datasheet)
MAG        = 10.0
PIXEL_CAM  = 6.5            # um
PIXEL_SIZE = PIXEL_CAM / MAG  # 0.65 um
ROTATION   = [0, 180, 90, 270]            # -> [top, bottom, left, right]
TAG_ORDER  = ["th", "bh", "lh", "rh"]     # half-circle, same order as ROTATION
TAG_ORDER_ANNULUS = ["hat", "hab", "hal", "har"]  # half-annulus, same order
NA_ILLUM   = {"NA02": 0.20, "NA04": 0.40, "NA08": 0.80}
# Half-annulus illumination is a FIXED firmware command, NOT scaled by the
# filename NA tag. Every half-annulus snap on 2026-06-02 used the SCI Dome mode
# ha.<h>.50.95  ->  inner NA 0.50, outer NA 0.95 (clipped to the dome max NA_DOME
# = 0.80). Verified from the microscope-PC lighting.py _DEFAULT_UNIFIED_MODES (all
# five annulus modes hardcode annulus=[0.5, 0.95], unchanged since 2026-05-15) and
# corroborated by the raw images (annulus background = 2-9% of full BF -> darkfield;
# the 4 halves differ -> ha. was accepted, not silently disabled). The NA02/NA04/
# NA08 token in annulus FILENAMES is an operator snap_tag ONLY -- it does not set
# the illumination NA for annulus shots (firmware always emits ha.<h>.50.95).
ANNULUS_INNER_NA = 0.50
ANNULUS_OUTER_NA = min(0.95, NA_DOME)   # 0.80 after dome clip

REG_U      = 1e-1           # fixed absorption reg (reference); we sweep phase reg
ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]   # phase reg_p sweep
REG_P_PANEL = 5e-3          # fixed reg_p for the fair cross-NA Fig-5 panel

# Manual knee override (expert visual judgment, 2026-06-04). The automatic
# L-curve triangle/chord heuristic (knee_index) is only reliable on a sharply
# cornered, convex-toward-origin L-curve. For NA08 the L-curve is a gentle,
# almost-straight diagonal with a double bend, so the max-distance rule returned
# alpha=0.1 — which the montage shows is heavily over-regularized (cells nearly
# gone). The true elbow / best phase contrast is at alpha=3e-3 (confirmed by eye
# against phase_alpha_montage.png). When a tag is listed here this alpha replaces
# the auto-detected knee for the summary, L-curve marker, montage marker and OPL
# output; the auto knee is still computed and recorded per-row (is_knee_auto) and
# drawn on the L-curve so the override stays auditable. (NA04 keeps its clean
# auto knee 1e-2; NA02's L-curve is too flat to pick reliably, left at auto.)
KNEE_OVERRIDE = {"NA08": 3e-3}

DATA_DIR = "/data/Project_Data/QPM/20260602_DPC_test"
OUT_DIR  = "/data/Project_Data/QPM/20260602_DPC_test/analysis"


# ---------------------------------------------------------------------------
# the ONLY extension of the reference: decouple source NA from objective NA
# ---------------------------------------------------------------------------
class VariableNADPCSolver(DPCSolver):
    """Reference ``DPCSolver`` with the illumination (source) NA decoupled from
    the objective (pupil) NA. Only ``sourceGen`` is overridden; normalization,
    WOTFGen and solve are the reference's, unchanged. For ``na_source == na`` it
    is identical to the reference (matched sigma = 1)."""

    def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, rotation,
                 na_source=None, na_dome=NA_DOME, dpc_num=4):
        self.na_source = na if na_source is None else na_source
        self.na_dome   = na_dome
        super().__init__(dpc_imgs, wavelength, na, na_in, pixel_size,
                         rotation, dpc_num)

    def sourceGen(self):
        self.source = []
        src_pupil = pupilGen(self.fxlin, self.fylin, self.wavelength,
                             min(self.na_source, self.na_dome), na_in=self.na_in)
        for rotIdx in range(self.dpc_num):
            self.source.append(np.zeros((self.dpc_imgs.shape[-2:])))
            rotdegree = self.rotation[rotIdx]
            if rotdegree < 180:
                self.source[-1][self.fylin[:, naxis] * np.cos(np.deg2rad(rotdegree)) + 1e-15 >=
                                self.fxlin[naxis, :] * np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= src_pupil
            else:
                self.source[-1][self.fylin[:, naxis] * np.cos(np.deg2rad(rotdegree)) + 1e-15 <
                                self.fxlin[naxis, :] * np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= src_pupil
                self.source[-1] += src_pupil
        self.source = np.asarray(self.source)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _find_one(na_tag, pat_tag):
    pat = f"*FOV4_green_BF_{pat_tag}_{na_tag}*.tif"
    # raw TIFFs may live directly under the dataset or under a raw/ subfolder
    hits = []
    for base in (os.path.join(DATA_DIR, "raw"), DATA_DIR):
        hits = glob.glob(os.path.join(base, pat))
        if hits:
            break
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 file for {pat_tag} {na_tag}, got {hits}")
    return hits[0]


def load_dpc_stack(na_tag, tags):
    """4-image stack in reference order [top, bottom, left, right]."""
    return np.asarray([tifffile.imread(_find_one(na_tag, t)).astype("float64")
                       for t in tags])


def annulus_is_brightfield(na_tag):
    """BF-DPC reconstructable only when the annular ring overlaps the objective
    pupil (inner_na < NA_obj). The fixed firmware ring is [0.50, 0.80], so
    inner 0.50 > NA_OBJ 0.40 -> PURE DARKFIELD for every NA tag (brightfield WOTF
    B0 = 0, not BF-DPC reconstructable). na_tag is kept for signature/print
    compatibility but no longer affects the ring (annulus NA is firmware-fixed,
    not set by the filename tag)."""
    inner, outer = ANNULUS_INNER_NA, ANNULUS_OUTER_NA
    return (inner < NA_OBJ), inner, outer


# ---------------------------------------------------------------------------
# helpers: solver build, forward residual, L-curve knee
# ---------------------------------------------------------------------------
def build_solver(na_tag, tags, na_in):
    stack = load_dpc_stack(na_tag, tags)
    solver = VariableNADPCSolver(stack, WAVELENGTH, NA_OBJ, na_in, PIXEL_SIZE,
                                 ROTATION, na_source=NA_ILLUM[na_tag])
    return solver


def solve_at(solver, reg_p):
    """Return (absorption, phase) at a given phase regularization reg_p."""
    solver.setTikhonovRegularization(REG_U, reg_p)
    res = solver.solve()[0]
    return res.real, res.imag


def data_misfit(solver, absorption, phase):
    """Forward-model residual ||I_meas - (Hu*mu + Hp*phi)|| over the 4 sources,
    in Fourier space (DC pixel excluded). Quantifies how well the recovered
    (mu, phi) reproduce the measured normalized images = the L-curve x-axis."""
    Ahat = F(absorption)
    Phat = F(phase)
    res2 = 0.0
    for d in range(solver.dpc_imgs.shape[0]):
        pred = solver.Hu[d] * Ahat + solver.Hp[d] * Phat
        meas = F(solver.dpc_imgs[d])
        diff = meas - pred
        diff[0, 0] = 0.0           # ignore DC
        res2 += float(np.sum(np.abs(diff) ** 2))
    return np.sqrt(res2) / phase.size


def knee_index(log_res, log_sol):
    """L-curve corner = point farthest from the chord joining the two endpoints
    (Triangle/again robust heuristic). Inputs are log-scaled arrays."""
    x = np.asarray(log_res, float); y = np.asarray(log_sol, float)
    xn = (x - x.min()) / (np.ptp(x) + 1e-12)
    yn = (y - y.min()) / (np.ptp(y) + 1e-12)
    p1 = np.array([xn[0], yn[0]]); p2 = np.array([xn[-1], yn[-1]])
    d = p2 - p1; d = d / (np.linalg.norm(d) + 1e-12)
    dist = []
    for i in range(len(xn)):
        v = np.array([xn[i], yn[i]]) - p1
        dist.append(np.linalg.norm(v - v.dot(d) * d))
    return int(np.argmax(dist))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_tif(path, arr):
    tifffile.imwrite(path, arr.astype("float32"), imagej=True)


def _disp(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def save_dpc(solver_imgs_raw, outdir, na_tag, tags):
    """Both axes: DPC_LR=(left-right)/(left+right), DPC_TB=(top-bottom)/(top+bottom).
    tags = [top, bottom, left, right] filenames."""
    top, bot, left, right = [tifffile.imread(_find_one(na_tag, t)).astype("float64")
                             for t in tags]
    dpc_tb = (top - bot) / (top + bot + 1e-9)
    dpc_lr = (left - right) / (left + right + 1e-9)
    save_tif(os.path.join(outdir, "dpc_TB.tif"), dpc_tb)
    save_tif(os.path.join(outdir, "dpc_LR.tif"), dpc_lr)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    for a, img, t in zip(ax, [dpc_tb, dpc_lr], ["DPC top-bottom", "DPC left-right"]):
        im = a.imshow(img, cmap="gray", clim=[-0.2, 0.2]); a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, ticks=[-0.2, 0, 0.2])
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dpc_pair.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)
    return dpc_tb, dpc_lr


def save_wotf(solver, outdir):
    """Phase WOTF for both axes. Hp[0]=top (TB axis), Hp[2]=left (LR axis);
    combined = sqrt(sum_j |Hp_j|^2) (paper Fig 3 2-axis coverage). All imaginary."""
    hp_tb = np.fft.fftshift(solver.Hp[0].imag)
    hp_lr = np.fft.fftshift(solver.Hp[2].imag)
    comb  = np.fft.fftshift(np.sqrt(np.sum(np.abs(solver.Hp) ** 2, axis=0)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, t, cm, cl in [
        (ax[0], hp_tb, "Phase WOTF $H_p$ top-bottom", "jet", [-0.8, 0.8]),
        (ax[1], hp_lr, "Phase WOTF $H_p$ left-right", "jet", [-0.8, 0.8]),
        (ax[2], comb,  "Combined 2-axis $\\sqrt{\\Sigma|H_j|^2}$", "viridis", None)]:
        im = a.imshow(img, cmap=cm, clim=cl); a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "phase_wotf.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)


def save_reconstruction_sweep(solver, outdir, na_tag, pattern):
    """Run the alpha (reg_p) sweep, save each phase TIFF, the montage, the
    L-curve, and the per-(NA,pattern) CSV. Returns the summary row."""
    phases, rows = {}, []
    for a in ALPHA_GRID:
        absn, phase = solve_at(solver, a)
        phases[a] = phase
        save_tif(os.path.join(outdir, f"phase_alpha_{a:.0e}.tif"), phase)
        res = data_misfit(solver, absn, phase)
        sol = float(np.sqrt(np.sum(phase ** 2)))
        rows.append(dict(alpha=a, phase_std=float(phase.std()), residual=res,
                         solution_norm=sol,
                         phase_p1=float(np.percentile(phase, 1)),
                         phase_p99=float(np.percentile(phase, 99))))

    ki_auto = knee_index([np.log10(r["residual"] + 1e-30) for r in rows],
                         [np.log10(r["solution_norm"] + 1e-30) for r in rows])
    if na_tag in KNEE_OVERRIDE:
        knee_alpha = KNEE_OVERRIDE[na_tag]
        ki = ALPHA_GRID.index(knee_alpha)
    else:
        knee_alpha = ALPHA_GRID[ki_auto]
        ki = ki_auto
    for i, r in enumerate(rows):
        r["is_knee"] = (i == ki)
        r["is_knee_auto"] = (i == ki_auto)

    # OPL at knee alpha
    save_tif(os.path.join(outdir, "opl_knee_nm.tif"),
             phases[knee_alpha] * WAVELENGTH * 1000.0 / (2 * np.pi))

    # montage
    n = len(ALPHA_GRID); cols = 3; rows_n = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    for i, a in enumerate(ALPHA_GRID):
        axx = ax.flat[i]
        axx.imshow(phases[a], cmap="gray", clim=[-1, 1])
        axx.set_title(f"α=reg_p={a:.0e}" + ("  ← knee" if a == knee_alpha else ""),
                      fontsize=10)
        axx.axis("off")
    for i in range(n, rows_n * cols):
        ax.flat[i].axis("off")
    fig.suptitle(f"{na_tag} {pattern} — phase vs Tikhonov α (reg_p), clim ±1 rad",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "phase_alpha_montage.png"), dpi=110,
                bbox_inches="tight"); plt.close(fig)

    # L-curve
    fig, a = plt.subplots(figsize=(6.5, 5.5))
    res = [r["residual"] for r in rows]; sol = [r["solution_norm"] for r in rows]
    a.loglog(res, sol, "o-", color="steelblue")
    for r in rows:
        a.annotate(f"{r['alpha']:.0e}", (r["residual"], r["solution_norm"]),
                   fontsize=8, textcoords="offset points", xytext=(4, 4))
    a.loglog(res[ki], sol[ki], "s", ms=14, mfc="none", mec="red", mew=2,
             label=f"knee α={knee_alpha:.0e}"
                   + (" (manual)" if na_tag in KNEE_OVERRIDE else ""))
    if na_tag in KNEE_OVERRIDE and ki_auto != ki:
        a.loglog(res[ki_auto], sol[ki_auto], "o", ms=13, mfc="none", mec="0.5",
                 mew=2, label=f"auto knee α={ALPHA_GRID[ki_auto]:.0e} (rejected)")
    a.set_xlabel("data misfit  ||I − (Hu·μ+Hp·φ)||"); a.set_ylabel("solution norm  ||φ||")
    a.set_title(f"{na_tag} {pattern} — L-curve (phase reg_p)"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "lcurve.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)

    # per-(NA,pattern) CSV
    with open(os.path.join(outdir, "alpha_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)

    knee = rows[ki]
    return dict(knee_alpha=knee_alpha, knee_phase_std=knee["phase_std"],
                knee_p1=knee["phase_p1"], knee_p99=knee["phase_p99"],
                phases=phases, knee=knee_alpha)


# ---------------------------------------------------------------------------
# cross-NA Fig-5 panel (fixed reg_p for fair comparison)
# ---------------------------------------------------------------------------
def fig5_style_panel(results, path):
    nas = list(results.keys())
    if not nas:
        return
    fig, ax = plt.subplots(4, len(nas), figsize=(4 * len(nas), 15))
    if len(nas) == 1:
        ax = ax.reshape(4, 1)
    row_titles = ["Brightfield (full)", "DPC (left-right)",
                  "Phase WOTF $H_p$ (L-R)", f"Phase recon (reg_p={REG_P_PANEL:.0e})"]
    for j, na in enumerate(nas):
        r = results[na]; sigma = NA_ILLUM[na] / NA_OBJ
        ax[0, j].imshow(_disp(r["bf"]), cmap="gray")
        ax[0, j].set_title(f"{na}  (σ = {sigma:.2f})", fontsize=12)
        ax[1, j].imshow(r["dpc_lr"], cmap="gray", clim=[-0.2, 0.2])
        ax[2, j].imshow(np.fft.fftshift(r["Hp_lr"].imag), cmap="jet", clim=[-0.8, 0.8])
        ax[3, j].imshow(r["phase"], cmap="gray", clim=[-1.0, 1.0])
        for i in range(4):
            ax[i, j].axis("off")
    for i, t in enumerate(row_titles):
        ax[i, 0].text(-0.08, 0.5, t, rotation=90, va="center", ha="right",
                      transform=ax[i, 0].transAxes, fontsize=12)
    fig.suptitle("Green (530 nm) DPC NA-sweep — faithful Tian&Waller 2015", fontsize=14)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def process(na_tag, pattern, global_rows, panel_results):
    sigma = NA_ILLUM[na_tag] / NA_OBJ
    if pattern == "half_circle":
        tags, na_in = TAG_ORDER, 0.0
    else:
        ok, inner, outer = annulus_is_brightfield(na_tag)
        tags, na_in = TAG_ORDER_ANNULUS, inner
        if not ok:
            print(f"  {na_tag} {pattern}: ring {inner:.2f}-{outer:.2f} outside "
                  f"pupil {NA_OBJ} -> DARKFIELD, skipped")
            global_rows.append(dict(na_tag=na_tag, pattern=pattern, sigma=round(sigma, 3),
                                    reconstructable=False, knee_alpha="",
                                    knee_phase_std="", knee_p1="", knee_p99=""))
            return

    base = os.path.join(OUT_DIR, na_tag, pattern)
    d_dpc = os.path.join(base, "dpc")
    d_wotf = os.path.join(base, "wotf")
    d_rec = os.path.join(base, "reconstruction")
    for d in (d_dpc, d_wotf, d_rec):
        os.makedirs(d, exist_ok=True)

    solver = build_solver(na_tag, tags, na_in)
    dpc_tb, dpc_lr = save_dpc(None, d_dpc, na_tag, tags)
    save_wotf(solver, d_wotf)
    summ = save_reconstruction_sweep(solver, d_rec, na_tag, pattern)

    print(f"  {na_tag} {pattern}: knee α={summ['knee_alpha']:.0e}  "
          f"phase_std={summ['knee_phase_std']:.4f}  "
          f"p1/p99=[{summ['knee_p1']:.3f},{summ['knee_p99']:.3f}]")
    global_rows.append(dict(na_tag=na_tag, pattern=pattern, sigma=round(sigma, 3),
                            reconstructable=True, knee_alpha=summ["knee_alpha"],
                            knee_phase_std=summ["knee_phase_std"],
                            knee_p1=summ["knee_p1"], knee_p99=summ["knee_p99"]))

    # cross-NA panel (only for half_circle, fixed reg_p)
    if pattern == "half_circle":
        _, phase_panel = solve_at(solver, REG_P_PANEL)
        bf = tifffile.imread(_find_one(na_tag, "full")).astype("float64")
        panel_results[na_tag] = dict(phase=phase_panel, bf=bf, dpc_lr=dpc_lr,
                                     Hp_lr=solver.Hp[2])


def main():
    # fresh rebuild of the analysis tree (everything here is regenerable)
    for na in ("NA02", "NA04", "NA08"):
        shutil.rmtree(os.path.join(OUT_DIR, na), ignore_errors=True)
    for legacy in ("phase",):           # old flat layout, superseded by per-NA folders
        shutil.rmtree(os.path.join(OUT_DIR, legacy), ignore_errors=True)
    for sub in ("figures", "tables"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    global_rows, panel = [], {}
    for na in ("NA02", "NA04", "NA08"):
        print(f"[NA] {na} (σ={NA_ILLUM[na]/NA_OBJ:.2f})")
        process(na, "half_circle", global_rows, panel)
        process(na, "half_annulus", global_rows, panel)

    fig5_style_panel(panel, os.path.join(OUT_DIR, "figures", "fig5_style_na_sweep.png"))

    with open(os.path.join(OUT_DIR, "tables", "global_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(global_rows[0].keys())); w.writeheader()
        w.writerows(global_rows)

    params = dict(
        wavelength_um=WAVELENGTH, wavelength_source="SCI Dome green datasheet (530 nm)",
        na_obj=NA_OBJ, na_dome=NA_DOME, mag=MAG, pixel_size_um=PIXEL_SIZE,
        rotation_deg=ROTATION, tag_order=TAG_ORDER, tag_order_annulus=TAG_ORDER_ANNULUS,
        annulus_inner_na=ANNULUS_INNER_NA, annulus_outer_na=ANNULUS_OUTER_NA,
        annulus_na_source="firmware ha.<h>.50.95 (fixed; filename NA tag ignored for annulus)",
        na_illum=NA_ILLUM,
        reg_u_fixed=REG_U, alpha_grid_reg_p=ALPHA_GRID, reg_p_panel=REG_P_PANEL,
        engine="example/dpc_algorithm.py::DPCSolver (verbatim)",
        extension="VariableNADPCSolver.sourceGen decouples source NA from objective NA",
        blank_required=False, axes=["top-bottom", "left-right"],
        data_dir=DATA_DIR, fov="FOV4 green")
    with open(os.path.join(OUT_DIR, "tables", "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print("\n[done] per-NA outputs under", OUT_DIR)


if __name__ == "__main__":
    main()
