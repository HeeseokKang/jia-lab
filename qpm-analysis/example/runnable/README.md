# `example/runnable/` — ready-to-run DPC reconstructors (lab-authored)

> **This folder is NOT vendored / third-party.** Unlike its sibling folders
> `example/tian2015/` and `example/chen2018_aberration/` (which hold the
> **unmodified** Waller-Lab reference code — do not edit), everything here is
> **lab-authored**: thin, generic, data-agnostic drivers meant for **anyone to
> run on their own captures** (built for Bill Jia, June 2026). They wrap the
> reference engines without altering them.

Two reconstructors, one per paper:

| Script | Paper | Capture | What you get |
|---|---|---|---|
| `dpc_tian2015_run.py` | Tian & Waller 2015 | **4 half-circles** (top/bottom/left/right) | quantitative phase φ, assumes an ideal pupil |
| `dpc_chen2018_run.py` | Chen, Phillips & Waller 2018 | **3 half-circles + 1 on-axis LED** | phase φ **+** recovered pupil aberration (Zernike) |

Both **self-normalize each image** — **no blank / empty-field shot is needed** —
and both find your files from any working directory (paths resolve relative to
the script, so you can run them from anywhere on the server with absolute
`--data`/`--out`).

The final quantitative result of each is the **phase φ (radians)** and its
**optical path length** `OPL = φ·λ/2π` (nm). The two raw DPC images
(top−bottom, left−right) are intermediate, direction-dependent, *qualitative*.

---

## Setup

```bash
conda activate fucci-analysis        # numpy, scipy, tifffile, matplotlib
```

(Any env with `numpy scipy tifffile matplotlib` works.)

---

## Specifying your input & output paths

Both runners take their input the **same way** — pick *one* of two modes:

**Mode A — point at a folder (auto-match):**

```bash
--data /path/to/folder
```

- Drop the half-circle TIFFs in that folder (a `raw/` subfolder inside it is
  also searched).
- Each file is assigned to a role by a **case-insensitive tag in its filename**:
  `top/bottom/left/right`, or `th/bh/lh/rh`, or single letters `t/b/l/r`
  (matched as whole tokens), plus `north/south/west/east`, `tophalf`, … .
- Exactly **one** file must match each role. If a role has zero or several
  matches, the run **aborts and tells you which role is missing/ambiguous** —
  then fall back to Mode B.

**Mode B — name each file explicitly (most robust):**

```bash
--top T.tif --bottom B.tif --left L.tif --right R.tif      # Tian
--top T.tif --bottom B.tif --left L.tif --led  LED.tif     # Chen (led replaces right)
```

Use this when your filenames carry no tag, or the tags are ambiguous.

**Output (required):**

```bash
--out /path/to/output_dir
```

Created if it doesn't exist; all results are written there.

Paths may be absolute or relative (the script resolves its own engine relative
to itself, so you can run from any working directory). **If you give neither
`--data` nor a full set of explicit files, the runner stops with an error — it
never guesses.**

---

## 1. Tian & Waller 2015 — baseline qDPC (4 half-circles)

Capture, full half-disk per image:

| image | LED-array pattern | rotation |
|---|---|---|
| top    | top half-circle    | 0°   |
| bottom | bottom half-circle | 180° |
| left   | left half-circle   | 90°  |
| right  | right half-circle  | 270° |

```bash
# (a) point at a folder — files auto-matched by tag
#     (top/bottom/left/right, or th/bh/lh/rh, or t/b/l/r, case-insensitive)
python dpc_tian2015_run.py --data /path/to/tiffs --out /path/to/out \
    --wavelength 0.530 --na 0.40 --mag 10 --pixel-cam 6.5

# (b) explicit (most robust if your filenames are unusual)
python dpc_tian2015_run.py \
    --top T.tif --bottom B.tif --left L.tif --right R.tif --out OUT

# Tikhonov α L-curve sweep + automatic knee pick:
python dpc_tian2015_run.py --data DIR --out OUT --alpha-sweep

# under-/over-filled condenser (σ ≠ 1): set the illumination NA separately
python dpc_tian2015_run.py --data DIR --out OUT --na 0.40 --na-illum 0.30
```

**Engine:** imports `example/tian2015/dpc_algorithm.py::DPCSolver` **verbatim**.
The only added math is `VariableNADPCSolver.sourceGen`, which decouples the
illumination NA from the objective NA (when they are equal it is byte-identical
to the reference). Same engine the lab baseline `src/dpc/dpc_tian2015.py` uses.

**Outputs:** `phase_rad.tif`, `opl_nm.tif`, `dpc_TB.tif`, `dpc_LR.tif`,
`dpc_pair.png`, `phase_wotf.png`, `phase.png`, `params.json`
(+ `phase_alpha_*.tif`, `phase_alpha_montage.png`, `lcurve.png`,
`alpha_sweep.csv` with `--alpha-sweep`).

Key flags: `--reg-u` (absorption reg, default 1e-1), `--reg-p` (phase reg,
default 1e-2), `--pixel-size` (override `pixel-cam/mag` directly).

---

## 2. Chen, Phillips & Waller 2018 — aberration-corrected qDPC

Use this when the objective may be **aberrated or slightly defocused**: it
jointly recovers the field **and** the pupil phase (Zernike coefficients), so
the phase is not corrupted by a non-ideal pupil.

Capture (**not** the standard 4-half set), σ = 1, full half-disk:

| image | pattern | rotation | role |
|---|---|---|---|
| top    | top half-circle    | 0°   | DPC |
| bottom | bottom half-circle | 180° | DPC |
| left   | left half-circle   | 90°  | DPC |
| LED    | **single on-axis LED** | — | coherent aberration contrast (replaces the right half) |

```bash
# (a) folder, auto-matched (led tag: led / single / dc / center / on_axis)
python dpc_chen2018_run.py --data /path/to/tiffs --out /path/to/out

# (b) explicit
python dpc_chen2018_run.py \
    --top T.tif --bottom B.tif --left L.tif --led LED.tif --out OUT

# tuning
python dpc_chen2018_run.py --data DIR --out OUT \
    --num-zernike 21 --reg-u 1e-1 --reg-p 5e-3 --max-outer 50 --inner-maxiter 10
```

**Engine:** the lab's self-contained Python port of the Chen-2018 **MATLAB**
pipeline (`src/dpc/dpc_chen2018_aberration.py`; analytic-gradient L-BFGS pupil
recovery, validated against the upstream `dataset_DPC_with_aberration.mat`).
⚠ The Waller repo's *Python* (`example/chen2018_aberration/`) is plain DPC+TV
with **no** aberration recovery — this runner does **not** use it. See
`example/README.md`.

**Outputs:** `phase_rad.tif`, `opl_nm.tif`, `absorption.tif`,
`pupil_phase_rad.tif`, `zernike_coeffs.csv`, `loss.csv`, `summary.png`.

---

## GPU notes (answering "can we improve with GPU?")

Short answer: **possible, but the win depends entirely on the workload — and we
have a measured reason to not assume a big speedup.**

- **cupy is not installed** in `fucci-analysis` yet. Both runners accept `--gpu`;
  with cupy absent they **fall back to CPU** and say so (results unchanged). So
  `--gpu` is a no-op on this box today — it's a placeholder for the backend
  below, not a hidden GPU path.
- **Tian (single reconstruction): GPU barely helps.** It's a handful of FFTs +
  one 2×2-per-frequency Tikhonov inversion on a single image — already
  sub-second on CPU. The lever there is **batch throughput** (many FOVs /
  timepoints), not GPU per-frame.
- **Chen (aberration loop): this is where GPU *could* pay off** — it's iterative
  (≈50 outer × L-BFGS pupil updates, FFTs each step). But the optimizer
  (`scipy.optimize` L-BFGS-B) lives on CPU, so a cupy backend means a
  host↔device transfer around the FFT-heavy cost/gradient each call; for modest
  image sizes that transfer can eat the gain.
- **Measured datapoint (don't over-promise):** in this lab's *tracking* pipeline,
  a cupy GPU port of the FFT-dominated stage gave only **~1.7×** end-to-end
  (FFT was 41.9% of runtime) — Amdahl-limited. DPC reconstruction is a different
  workload, but it's a reason to **benchmark before committing**, not assume 10×.

**To actually try it (follow-up):**
1. `conda install -c conda-forge cupy` (match the CUDA toolkit; this box has an
   RTX 4000 Ada).
2. Wire a cupy `xp`-backend into the Chen forward model / WOTF FFTs (the part the
   lab owns), keep the optimizer on CPU, and **measure CPU vs GPU** on a real
   capture before declaring a speedup.

Heeseok can run that benchmark and report back — happy to build the cupy backend
once cupy is in the env and we have a target image size.

---

## License / provenance

- `dpc_chen2018_run.py` → lab-authored engine (BSD-spirit port). Safe to use.
- `dpc_tian2015_run.py` → imports the **vendored** `example/tian2015/` engine,
  which has **no upstream license** (all rights reserved). Fine for internal lab
  use as an attributed academic reference; see the license note at the bottom of
  `example/README.md` before any public redistribution.

Citations: see `example/README.md`.
