#!/usr/bin/env python
"""
dpc_tian2015_run.py  --  generic, data-agnostic DPC / quantitative-phase runner
following Tian & Waller, Opt. Express 23(9):11394 (2015).

For Bill Jia (and anyone) to run on their OWN 4 half-circle captures.
----------------------------------------------------------------------
This is a thin, generic driver around the Waller-Lab reference engine
``example/tian2015/dpc_algorithm.py::DPCSolver``, which is imported VERBATIM
(byte-identical, unmodified) -- exactly like the lab baseline
``src/dpc/dpc_tian2015.py``. The only added math is ``VariableNADPCSolver``
(below), which decouples the illumination NA from the objective NA so an
under-/over-filled condenser (sigma = NA_illum / NA_obj != 1) can be modeled.
When NA_illum == NA_obj it reduces EXACTLY to the reference engine.

Unlike the lab baseline (which is hard-wired to the 2026-06-02 FOV4 green
dataset: fixed filenames, fixed NA02/04/08 tags, fixed output tree), this runner
takes any 4 TIFFs and any optics via command-line flags.

Capture spec (standard qDPC)
----------------------------
4 brightfield images, each a HALF-CIRCLE of the condenser/LED-array lit:
    top (0 deg), bottom (180 deg), left (90 deg), right (270 deg).
No blank / empty-field image is needed: the reference ``normalization()``
self-normalizes every image (local-mean divide -> mean divide -> -1).

The final result is the quantitative PHASE  phi (radians), and its optical path
length  OPL = phi * lambda / (2 pi)  in nm. The two raw DPC images (top-bottom,
left-right) are intermediate, direction-dependent, QUALITATIVE views.

Run (CWD-independent -- run it from anywhere on the server)
----------------------------------------------------------
    conda activate fucci-analysis        # numpy scipy tifffile matplotlib

    # (a) point at a folder; files auto-matched by tag (top/bottom/left/right,
    #     or th/bh/lh/rh, or t/b/l/r, ... -- case-insensitive)
    python dpc_tian2015_run.py --data /path/to/tiffs --out /path/to/out \
        --wavelength 0.530 --na 0.40 --mag 10 --pixel-cam 6.5

    # (b) or name each file explicitly (most robust)
    python dpc_tian2015_run.py \
        --top T.tif --bottom B.tif --left L.tif --right R.tif --out OUT

    # add  --alpha-sweep  for a Tikhonov-alpha L-curve sweep + knee detection
    # add  --na-illum 0.30  if the condenser is under-filled (sigma < 1)
    # add  --gpu            (see "GPU" in example/runnable/README.md)

Citation: L. Tian and L. Waller, "Quantitative differential phase contrast
imaging in an LED array microscope," Opt. Express 23(9), 11394-11403 (2015).
"""
import os
import re
import sys
import csv
import glob
import json
import argparse

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Import the Waller-Lab reference engine VERBATIM.
# Resolve its path relative to THIS file (example/runnable/ -> example/tian2015/)
# so the runner works from any working directory.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_DIR = os.path.normpath(os.path.join(_HERE, "..", "tian2015"))
if not os.path.isfile(os.path.join(_ENGINE_DIR, "dpc_algorithm.py")):
    sys.exit(f"[error] reference engine not found at {_ENGINE_DIR}/dpc_algorithm.py\n"
             f"        This runner expects to live in example/runnable/ next to "
             f"the vendored example/tian2015/ folder.")
sys.path.insert(0, _ENGINE_DIR)
from dpc_algorithm import DPCSolver, pupilGen, naxis, F  # noqa: E402

ROTATION = [0, 180, 90, 270]   # -> [top, bottom, left, right], reference order


# --------------------------------------------------------------------------- #
# The ONLY extension of the reference: decouple source NA from objective NA.
# Identical to src/dpc/dpc_tian2015.py::VariableNADPCSolver. With na_source == na
# it is byte-identical to the reference sourceGen.
# --------------------------------------------------------------------------- #
class VariableNADPCSolver(DPCSolver):
    def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, rotation,
                 na_source=None, na_dome=None, dpc_num=4):
        self.na_source = na if na_source is None else na_source
        self.na_dome = na_dome
        super().__init__(dpc_imgs, wavelength, na, na_in, pixel_size,
                         rotation, dpc_num)

    def sourceGen(self):
        self.source = []
        na_src = self.na_source if self.na_dome is None else min(self.na_source, self.na_dome)
        src_pupil = pupilGen(self.fxlin, self.fylin, self.wavelength, na_src, na_in=self.na_in)
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


# --------------------------------------------------------------------------- #
# Generic 4-image loader: auto-match by filename tag, or take explicit paths.
# --------------------------------------------------------------------------- #
# Aliases per role. Single/double-letter aliases are matched as WHOLE tokens
# (split on non-alphanumerics) to avoid false hits; >=4-letter aliases also
# match as substrings. Extend freely for your naming convention.
ROLE_ALIASES = {
    "top":    ["top", "th", "t", "u", "north", "tophalf", "top_half"],
    "bottom": ["bottom", "bh", "b", "d", "south", "bottomhalf", "bottom_half"],
    "left":   ["left", "lh", "l", "w", "west", "lefthalf", "left_half"],
    "right":  ["right", "rh", "r", "e", "east", "righthalf", "right_half"],
}
_ROLES = ["top", "bottom", "left", "right"]


def _role_of(filename):
    """Return the role whose alias matches this filename, or None / 'ambiguous'."""
    name = os.path.basename(filename).lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    hits = []
    for role, aliases in ROLE_ALIASES.items():
        matched = False
        for a in aliases:
            if a in tokens:                 # whole-token match (handles th/bh/lh/rh, t/b/l/r)
                matched = True
            elif len(a) >= 4 and a in name:  # substring for long, unambiguous aliases
                matched = True
        if matched:
            hits.append(role)
    if len(hits) == 1:
        return hits[0]
    return "ambiguous" if hits else None


def auto_match(data_dir):
    """Find exactly one TIFF per role in data_dir (searches data_dir and data_dir/raw)."""
    tiffs = []
    for base in (data_dir, os.path.join(data_dir, "raw")):
        tiffs += glob.glob(os.path.join(base, "*.tif")) + glob.glob(os.path.join(base, "*.tiff"))
    tiffs = sorted(set(tiffs))
    by_role = {r: [] for r in _ROLES}
    for f in tiffs:
        r = _role_of(f)
        if r in by_role:
            by_role[r].append(f)
    paths = {}
    problems = []
    for r in _ROLES:
        if len(by_role[r]) == 1:
            paths[r] = by_role[r][0]
        elif len(by_role[r]) == 0:
            problems.append(f"  {r:<6}: no file matched")
        else:
            problems.append(f"  {r:<6}: {len(by_role[r])} files matched -> "
                            + ", ".join(os.path.basename(x) for x in by_role[r]))
    if problems:
        sys.exit("[error] could not auto-match 4 half-circle TIFFs in "
                 f"{data_dir}\n" + "\n".join(problems)
                 + "\n  -> pass them explicitly with --top/--bottom/--left/--right")
    return paths


def load_stack(paths):
    """[top, bottom, left, right] float64 stack."""
    imgs = [tifffile.imread(paths[r]).astype("float64") for r in _ROLES]
    shapes = {im.shape for im in imgs}
    if len(shapes) != 1:
        sys.exit(f"[error] the 4 images have different shapes: {shapes}")
    return np.asarray(imgs)


# --------------------------------------------------------------------------- #
# Reconstruction helpers (generic ports of src/dpc/dpc_tian2015.py)
# --------------------------------------------------------------------------- #
def solve_at(solver, reg_u, reg_p):
    solver.setTikhonovRegularization(reg_u, reg_p)
    res = solver.solve()[0]
    return res.real, res.imag      # absorption, phase


def data_misfit(solver, absorption, phase):
    """||I - (Hu*mu + Hp*phi)|| over the 4 sources in Fourier space (DC excluded)."""
    Ahat, Phat = F(absorption), F(phase)
    res2 = 0.0
    for d in range(solver.dpc_imgs.shape[0]):
        diff = F(solver.dpc_imgs[d]) - (solver.Hu[d] * Ahat + solver.Hp[d] * Phat)
        diff[0, 0] = 0.0
        res2 += float(np.sum(np.abs(diff) ** 2))
    return np.sqrt(res2) / phase.size


def knee_index(log_res, log_sol):
    """L-curve corner = farthest point from the endpoint chord."""
    x, y = np.asarray(log_res, float), np.asarray(log_sol, float)
    xn = (x - x.min()) / (np.ptp(x) + 1e-12)
    yn = (y - y.min()) / (np.ptp(y) + 1e-12)
    p1 = np.array([xn[0], yn[0]]); p2 = np.array([xn[-1], yn[-1]])
    d = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-12)
    dist = [np.linalg.norm((np.array([xn[i], yn[i]]) - p1)
                           - (np.array([xn[i], yn[i]]) - p1).dot(d) * d)
            for i in range(len(xn))]
    return int(np.argmax(dist))


def _disp(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)


def save_tif(path, arr):
    tifffile.imwrite(path, np.asarray(arr).astype("float32"), imagej=True)


def save_dpc(stack, outdir):
    top, bot, left, right = stack
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


def save_wotf(solver, outdir):
    hp_tb = np.fft.fftshift(solver.Hp[0].imag)
    hp_lr = np.fft.fftshift(solver.Hp[2].imag)
    comb = np.fft.fftshift(np.sqrt(np.sum(np.abs(solver.Hp) ** 2, axis=0)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, t, cm, cl in [
        (ax[0], hp_tb, "Phase WOTF $H_p$ top-bottom", "jet", [-0.8, 0.8]),
        (ax[1], hp_lr, "Phase WOTF $H_p$ left-right", "jet", [-0.8, 0.8]),
        (ax[2], comb, "Combined 2-axis $\\sqrt{\\Sigma|H_j|^2}$", "viridis", None)]:
        im = a.imshow(img, cmap=cm, clim=cl); a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "phase_wotf.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)


def save_phase(phase, opl_nm, outdir):
    save_tif(os.path.join(outdir, "phase_rad.tif"), phase)
    save_tif(os.path.join(outdir, "opl_nm.tif"), opl_nm)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    im0 = ax[0].imshow(phase, cmap="gray"); ax[0].set_title("phase phi (rad)")
    ax[0].axis("off"); fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(opl_nm, cmap="viridis"); ax[1].set_title("OPL (nm)")
    ax[1].axis("off"); fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "phase.png"), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)


def alpha_sweep(solver, reg_u, outdir, wavelength_um,
                grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)):
    phases, rows = {}, []
    for a in grid:
        absn, phase = solve_at(solver, reg_u, a)
        phases[a] = phase
        save_tif(os.path.join(outdir, f"phase_alpha_{a:.0e}.tif"), phase)
        rows.append(dict(alpha=a, phase_std=float(phase.std()),
                         residual=data_misfit(solver, absn, phase),
                         solution_norm=float(np.sqrt(np.sum(phase ** 2)))))
    ki = knee_index([np.log10(r["residual"] + 1e-30) for r in rows],
                    [np.log10(r["solution_norm"] + 1e-30) for r in rows])
    knee = grid[ki]
    for i, r in enumerate(rows):
        r["is_knee"] = (i == ki)
    # montage
    n = len(grid); cols = 3; rows_n = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows_n, cols, figsize=(4 * cols, 4 * rows_n))
    for i, a in enumerate(grid):
        ax.flat[i].imshow(phases[a], cmap="gray", clim=[-1, 1])
        ax.flat[i].set_title(f"alpha={a:.0e}" + ("  <- knee" if a == knee else ""), fontsize=10)
        ax.flat[i].axis("off")
    for i in range(n, rows_n * cols):
        ax.flat[i].axis("off")
    fig.suptitle("phase vs Tikhonov alpha (reg_p), clim +/-1 rad", fontsize=13)
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
             label=f"knee alpha={knee:.0e}")
    a.set_xlabel("data misfit  ||I-(Hu*mu+Hp*phi)||"); a.set_ylabel("solution norm ||phi||")
    a.set_title("L-curve (phase reg_p)"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "lcurve.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)
    with open(os.path.join(outdir, "alpha_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    return knee, phases[knee]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with the 4 half-circle TIFFs (auto-matched by tag)")
    ap.add_argument("--top"); ap.add_argument("--bottom")
    ap.add_argument("--left"); ap.add_argument("--right")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--wavelength", type=float, default=0.530, help="um (default 0.530, SCI Dome green)")
    ap.add_argument("--na", type=float, default=0.40, help="objective NA (default 0.40)")
    ap.add_argument("--na-illum", type=float, default=0.80,
                    help="illumination (source) NA (default 0.80, SCI Dome max). "
                         "Pass equal to --na for sigma=1.")
    ap.add_argument("--na-dome", type=float, default=None, help="clip source NA to this max (optional)")
    ap.add_argument("--mag", type=float, default=10.0, help="objective magnification (default 10)")
    ap.add_argument("--pixel-cam", type=float, default=6.5, help="camera pixel um (default 6.5)")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="effective pixel um; overrides pixel-cam/mag if given")
    ap.add_argument("--reg-u", type=float, default=1e-1, help="absorption Tikhonov reg (default 1e-1)")
    ap.add_argument("--reg-p", type=float, default=1e-2, help="phase Tikhonov reg (default 1e-2)")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="sweep phase reg over an L-curve and auto-pick the knee")
    ap.add_argument("--gpu", action="store_true",
                    help="request GPU; see example/runnable/README.md 'GPU notes'")
    args = ap.parse_args()

    if args.gpu:
        try:
            import cupy  # noqa: F401
            print("[gpu] cupy is installed, but the faithful Tian reference solver is "
                  "numpy-based; this single-shot reconstruction runs on CPU (already "
                  "sub-second). GPU's lever here is batch throughput -- see README.")
        except Exception:
            print("[gpu] cupy not installed -> running on CPU (no change to results). "
                  "See README 'GPU notes' to install + benchmark.")

    # resolve inputs
    if args.top and args.bottom and args.left and args.right:
        paths = {"top": args.top, "bottom": args.bottom,
                 "left": args.left, "right": args.right}
    elif args.data:
        paths = auto_match(args.data)
        print("[match] " + "  ".join(f"{r}={os.path.basename(paths[r])}" for r in _ROLES))
    else:
        ap.error("provide either --data DIR or all of --top/--bottom/--left/--right")

    pixel_size = args.pixel_size if args.pixel_size else args.pixel_cam / args.mag
    na_illum = args.na_illum
    os.makedirs(args.out, exist_ok=True)

    stack = load_stack(paths)
    solver = VariableNADPCSolver(stack, args.wavelength, args.na, 0.0, pixel_size,
                                 ROTATION, na_source=na_illum, na_dome=args.na_dome)

    save_dpc(stack, args.out)
    save_wotf(solver, args.out)

    if args.alpha_sweep:
        reg_p, phase = alpha_sweep(solver, args.reg_u, args.out, args.wavelength)
        print(f"[sweep] knee phase reg_p = {reg_p:.0e}")
    else:
        _, phase = solve_at(solver, args.reg_u, args.reg_p)
        reg_p = args.reg_p

    opl_nm = phase * args.wavelength * 1000.0 / (2 * np.pi)
    save_phase(phase, opl_nm, args.out)

    params = dict(method="Tian&Waller 2015 (reference DPCSolver, verbatim)",
                  inputs={r: os.path.abspath(paths[r]) for r in _ROLES},
                  wavelength_um=args.wavelength, na_obj=args.na, na_illum=na_illum,
                  sigma=na_illum / args.na, mag=args.mag, pixel_size_um=pixel_size,
                  reg_u=args.reg_u, reg_p=reg_p, alpha_sweep=bool(args.alpha_sweep),
                  rotation_deg=ROTATION, order=_ROLES,
                  engine="example/tian2015/dpc_algorithm.py::DPCSolver (verbatim)",
                  extension="VariableNADPCSolver.sourceGen (source NA decoupled from objective NA)")
    with open(os.path.join(args.out, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"[done] phase (rad) std={phase.std():.4f}  "
          f"p1/p99=[{np.percentile(phase, 1):.3f},{np.percentile(phase, 99):.3f}]")
    print(f"[done] outputs in {args.out}  (phase_rad.tif, opl_nm.tif, dpc_*.tif, *.png)")


if __name__ == "__main__":
    main()
