# `example/` — Waller-Lab reference DPC implementations (third-party)

This folder vendors the **unmodified** reference DPC/QPM code from the Waller
Lab (UC Berkeley). It is **not our code.** Two upstream versions are kept
side by side, one per subfolder:

| Subfolder | Upstream repo | Paper | License |
|---|---|---|---|
| `tian2015/` | [`Waller-Lab/DPC`](https://github.com/Waller-Lab/DPC) → `python_code/` | Tian & Waller 2015 | **none** (all rights reserved) |
| `chen2018_aberration/` | [`Waller-Lab/DPC_withAberrationCorrection`](https://github.com/Waller-Lab/DPC_withAberrationCorrection) → `python_code/` | Chen, Phillips & Waller 2018 | **BSD 3-Clause** (`LICENSE.txt`) |

> **`runnable/` is the one folder here that is NOT vendored.** It holds
> **lab-authored**, generic, ready-to-run reconstructors (`dpc_tian2015_run.py`,
> `dpc_chen2018_run.py`) for running either paper on **your own** 4-image
> capture — built for Bill Jia (June 2026). They wrap the vendored engines
> without modifying them. See `runnable/README.md`.

## `tian2015/` — the engine we actually use

- `dpc_algorithm.py` — the `DPCSolver` engine (pupil/source generation,
  weak-object transfer functions, Tikhonov inversion). **Verbatim, byte-identical
  to upstream `python_code/dpc_algorithm.py`.**
- `main_dpc.ipynb` — the reference driver notebook.
- **Upstream status:** repository **archived 2019-04-24** (read-only).
- **License:** the 2015 repo ships **no LICENSE file** → under default copyright,
  "no license" = all rights reserved (see the gray-area note at the bottom).

`src/dpc/dpc_tian2015.py` adds `example/tian2015/` to `sys.path` and imports
`DPCSolver` **verbatim** (no edits). Our only extension lives in our own file
(`VariableNADPCSolver.sourceGen`), which subclasses the reference class to
decouple the illumination NA from the objective NA. Keeping the reference here
unchanged makes the "follow the paper's implementation exactly" contract
auditable.

## `chen2018_aberration/` — newer reference, kept for study only

- `dpc_algorithm.py` — the **upgraded** `DPCSolver` (~260 lines vs 2015's ~85):
  adds Total-Variation deconvolution (`deconvTV`, `_softThreshold`) alongside
  Tikhonov, and a richer regularization API.
- `main_dpc.ipynb` — the upgraded driver notebook.
- `LICENSE.txt` — **BSD 3-Clause** (Copyright 2018, Waller Lab). This is the
  license that governs everything in this subfolder.

**Not a drop-in replacement for `tian2015/`.** The 2018 Python API diverged from
the 2015 engine our baseline code is built on:

- `setTikhonovRegularization(reg_u, reg_p)` → renamed `setRegularizationParameters(...)`
- `solve(xini=..., plot_verbose=...)` → `solve(method="Tikhonov"|"TV", tv_order=..., tv_max_iter=...)`

Swapping `dpc_tian2015.py` onto this engine would require real code changes, not
a path swap — so the baseline keeps importing `tian2015/`.

⚠ **Two different Python files — don't confuse them.** The warning below is
about the **vendored** file in *this* subfolder, NOT our lab engine:

| File | Aberration correction? |
|---|---|
| **`example/chen2018_aberration/dpc_algorithm.py`** (vendored, this subfolder) | **NO** — standard DPC + TV only |
| **`src/dpc/dpc_chen2018_aberration.py`** (lab-authored) | **YES** — the full algorithm |

⚠ **The vendored `example/chen2018_aberration/dpc_algorithm.py` is NOT the
aberration-correction algorithm.** It is the standard DPC solver + TV. The
actual computational-aberration-correction code of Chen 2018 exists upstream
**only in MATLAB** (`main_dpc.m`, `gradientPupil.m`, `DPC_L2.m`,
`genTransferFunction.m`, `genZernikePoly.m` in
`Waller-Lab/DPC_withAberrationCorrection`).

✅ **Our `src/dpc/dpc_chen2018_aberration.py` DOES have the aberration
correction.** It is a self-contained **Python port of those five MATLAB files**
(Zernike pupil basis ← `genZernikePoly.m`; closed-form field solve ← `DPC_L2.m`;
analytic-gradient L-BFGS pupil recovery ← `gradientPupil.m`; alternating
field/pupil outer loop ← `main_dpc.m`), validated against the upstream
`dataset_DPC_with_aberration.mat` (smooth monotone loss descent, stable Zernike
recovery). It does **not** import the vendored Python file above — the whole
algorithm is integrated into that one lab file.

> **⚠️ Vendored Python ≠ aberration correction.** The 2018 repo's headline
> feature — the joint complex-field + pupil-aberration estimation — is
> implemented in the repo's **MATLAB** code only (`matlab_code/`, requires the
> external `minFunc` package). Only the *Python* was vendored here, and the
> Waller Python `dpc_algorithm.py` in **this** subfolder is the improved
> *standard* DPC reconstruction, **without** the aberration-correction
> algorithm. (Our `src/dpc/dpc_chen2018_aberration.py` re-implements that MATLAB
> algorithm in Python — see the ✅ above.)

## Citations (requested by upstream)

1. L. Tian and L. Waller, "Quantitative differential phase contrast imaging in
   an LED array microscope," *Opt. Express* **23**(9), 11394–11403 (2015).
2. Z. F. Phillips, M. Chen, and L. Waller, "Single-shot quantitative phase
   microscopy with color-multiplexed differential phase contrast (cDPC),"
   *PLOS ONE* **12**(2): e0171228 (2017).
3. M. Chen, Z. F. Phillips, and L. Waller, "Quantitative differential phase
   contrast (DPC) microscopy with computational aberration correction,"
   *Opt. Express* **26**(25), 32888–32899 (2018).

## ⚠️ License status — read before public release

- **`chen2018_aberration/`** ships an explicit **BSD 3-Clause** license → safe to
  redistribute with attribution (keep `LICENSE.txt`).
- **`tian2015/`** has **no license** upstream. Under default copyright that means
  all rights reserved — redistribution is not explicitly granted. We retain it
  only as an unmodified, attributed academic reference for internal
  reproducibility. If `jia-lab` becomes **public**, the options are: (1) keep as
  attributed reference; (2) vendor out (download at setup instead of committing);
  or (3) ask the Waller Lab for an explicit license. This is a maintainer
  decision (Heeseok), not an automated one.
