#!/usr/bin/env python
"""
dpc_chen2018_run.py  --  generic runner for aberration-corrected qDPC
following Chen, Phillips & Waller, Opt. Express 26(25):32888 (2018).

For Bill Jia (and anyone) to run on their OWN capture.
-------------------------------------------------------
This is a thin, CWD-independent wrapper around the lab's self-contained Python
port of the Chen-2018 MATLAB pipeline, ``src/dpc/dpc_chen2018_aberration.py``.
It JOINTLY recovers the complex field (absorption + phase) AND the pupil
aberration (Zernike coefficients), so it corrects for an aberrated/defocused
objective that the plain Tian-2015 reconstruction (dpc_tian2015_run.py) assumes
away.

Provenance: the Waller repo ships the aberration algorithm only in MATLAB; its
bundled Python (example/chen2018_aberration/dpc_algorithm.py) is plain DPC+TV
with NO aberration recovery. The engine used here is the lab's analytic-gradient
L-BFGS port (numpy + scipy only), validated against the upstream
dataset_DPC_with_aberration.mat. See example/README.md for the full note.

Capture spec (this is NOT the standard 4-half set)
--------------------------------------------------
4 images at sigma = 1 (NA_illum = NA_obj), full half-disk (no annulus):
    top    half-circle  (rotation 0)     tag: th / top
    bottom half-circle  (rotation 180)   tag: bh / bottom
    left   half-circle  (rotation 90)    tag: lh / left
    single on-axis LED  (one central LED) tag: led / single / dc / center
The 4th half-circle (right) of standard DPC is REPLACED by the single LED, which
carries the coherent contrast needed to disambiguate the pupil phase. No blank
needed (self-normalizing).

Run (from anywhere on the server)
---------------------------------
    conda activate fucci-analysis        # numpy scipy tifffile matplotlib

    # (a) point at a folder; files auto-matched by tag
    python dpc_chen2018_run.py --data /path/to/tiffs --out /path/to/out \
        --wavelength 0.530 --na 0.40 --mag 10 --pixel-cam 6.5

    # (b) or name each file explicitly (most robust)
    python dpc_chen2018_run.py \
        --top T.tif --bottom B.tif --left L.tif --led LED.tif --out OUT

    # tuning:  --num-zernike 21  --reg-u 1e-1  --reg-p 5e-3
    #          --max-outer 50  --inner-maxiter 10
    # add  --gpu   (see example/runnable/README.md 'GPU notes')

Outputs: phase_rad.tif, opl_nm.tif, absorption.tif, pupil_phase_rad.tif,
zernike_coeffs.csv, loss.csv, summary.png.

Citation: M. Chen, Z. F. Phillips, and L. Waller, "Quantitative differential
phase contrast (DPC) microscopy with computational aberration correction,"
Opt. Express 26(25), 32888-32899 (2018).
"""
import os
import sys

# --------------------------------------------------------------------------- #
# Honest GPU handling (consistent with dpc_tian2015_run.py). The Chen engine is
# an iterative numpy/scipy solver (L-BFGS over the pupil, FFTs each step); that
# iterative loop -- not a single Tian reconstruction -- is where a cupy backend
# could actually pay off. cupy is NOT installed in this lab's fucci-analysis env
# yet, so this build runs on CPU. We strip --gpu here so the underlying CLI is
# unchanged, and print the status. See README 'GPU notes' for install + the
# benchmark-first plan.
# --------------------------------------------------------------------------- #
_gpu = "--gpu" in sys.argv
if _gpu:
    sys.argv.remove("--gpu")

# --------------------------------------------------------------------------- #
# Make the lab engine importable regardless of CWD:
#   example/runnable/ -> ../../src  (qpm-analysis/src)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if not os.path.isfile(os.path.join(_SRC, "dpc", "dpc_chen2018_aberration.py")):
    sys.exit(f"[error] lab engine not found at {_SRC}/dpc/dpc_chen2018_aberration.py\n"
             f"        This runner expects to live in example/runnable/ inside the "
             f"qpm-analysis repo.")
sys.path.insert(0, _SRC)

from dpc.dpc_chen2018_aberration import main as _engine_main  # noqa: E402


if __name__ == "__main__":
    if _gpu:
        try:
            import cupy  # noqa: F401
            print("[gpu] cupy is installed. NOTE: this build's Chen solver still runs "
                  "on CPU -- a cupy backend for the iterative pupil loop is a measured "
                  "follow-up (see README). Running on CPU now.")
        except Exception:
            print("[gpu] cupy not installed -> running on CPU (no change to results). "
                  "The iterative pupil loop is the part that could benefit; see README "
                  "'GPU notes' to install + benchmark.")
    _engine_main()
