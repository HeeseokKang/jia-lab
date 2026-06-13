"""
dpc_chen2018_aberration.py
==========================
Quantitative DPC with **computational aberration correction**, a faithful
Python port of the reference implementation of

    M. Chen, Z. F. Phillips, and L. Waller,
    "Quantitative differential phase contrast (DPC) microscopy with
     computational aberration correction,"
    Opt. Express 26(25), 32888-32899 (2018).

This is the aberration-correcting **upgrade** of the baseline 2015 pipeline in
``dpc_tian2015.py`` (Tian & Waller, Opt. Express 23(9):11394, 2015). The two
form a lineage:

    dpc_tian2015.py            baseline qDPC: 4 half-circles (T/B/L/R), known
                               (assumed ideal) pupil, single Tikhonov inversion.
    dpc_chen2018_aberration.py joint recovery of complex field AND pupil
                               aberration from 3 half-circles + 1 on-axis LED.

PROVENANCE NOTE (important, read before "faithful" claims)
----------------------------------------------------------
The Waller-Lab repo ``DPC_withAberrationCorrection`` ships the aberration
algorithm **only as MATLAB** (``main_dpc.m``, ``gradientPupil.m``, ``DPC_L2.m``,
``genTransferFunction.m``, ``genZernikePoly.m``). Its bundled *Python*
``dpc_algorithm.py`` (vendored here under ``example/chen2018_aberration/``) is
the standard DPC + TV solver and contains **no** aberration recovery. So this
module is a **port of the MATLAB pipeline**, not a wrapper around vendored
Python -- it is self-contained (numpy + scipy only) on purpose, so there is no
hidden dependency on which ``dpc_algorithm.py`` happens to be on ``sys.path``.

WHAT MAPS TO THE PAPER / REFERENCE MATLAB
-----------------------------------------
  * Acquisition = 3 half-circles (rotations [0, 180, 90] = top/bottom/left) +
    1 single on-axis LED (DC pixel). The 4th half-circle (right, 270 deg) of
    standard DPC is **replaced** by the on-axis LED, which carries the coherent
    contrast needed to disambiguate the pupil phase.   -> main_dpc.m
  * Zernike pupil  P(c) = |P| * exp(i * sum_m c_m Z_m), OSA/ANSI ordering with
    the first three modes (piston + 2 tilts) discarded as unrecoverable /
    ambiguous.                                          -> genZernikePoly.m
  * Closed-form Tikhonov field recovery for a given pupil (2x2 per-frequency
    inversion).                                         -> DPC_L2.m
  * Complex-pupil weak-object transfer functions.       -> genTransferFunction.m
  * Pupil recovery by L-BFGS using the **analytic** gradient (closed-form
    back-propagation, returned together with the cost).  -> gradientPupil.m
  * Alternating outer loop: (field | pupil) then (pupil | field).  -> main_dpc.m

This port uses the ANALYTIC gradient (paper Eq. 8); the earlier version of this
file used a finite-difference numeric gradient, which converged shallowly.

System defaults are this lab's microscope: 10x / NA 0.40 objective, SCI Dome
green 530 nm, 6.5 um camera pixel (-> 0.65 um effective), sigma = 1
(NA_illum = NA_obj = 0.40), full half-disk illumination (no annulus).
"""
import os
import csv
import glob
import argparse

import numpy as np
from math import factorial
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# FFT primitives + pupil/grid generators (verbatim from the Waller-Lab engine)
# --------------------------------------------------------------------------- #
naxis = np.newaxis
F  = lambda x: np.fft.fft2(x)
IF = lambda x: np.fft.ifft2(x)


def _gen_grid(size, dx):
    xlin = np.arange(size, dtype="complex128")
    return (xlin - size // 2) * dx


def pupil_gen(fxlin, fylin, wavelength, na, na_in=0.0):
    pupil = np.array(fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 <= (na / wavelength) ** 2)
    if na_in != 0.0:
        pupil[fxlin[naxis, :] ** 2 + fylin[:, naxis] ** 2 < (na_in / wavelength) ** 2] = 0.0
    return pupil


def _reflect_source(s):
    """S(-u) in FFT (DC-at-origin) layout. Direct port of the MATLAB
    ``rot90(padarray(fftshift(s),[1,1],'post'),2)`` -> crop -> ``ifftshift``
    trick, which point-reflects a centred array correctly for even/odd sizes."""
    ss = np.fft.fftshift(s)
    ss = np.pad(ss, ((0, 1), (0, 1)), mode="constant")
    ss = np.rot90(ss, 2)
    ss = ss[:-1, :-1]
    return np.fft.ifftshift(ss)


def _gen_transfer_function(source, pupil):
    """Weak-object transfer functions for a (possibly complex/aberrated) pupil.
    Returns (Hi, Hr) = (phase TF, absorption TF). Port of genTransferFunction.m."""
    source_f = _reflect_source(source)
    FP_cFSP = np.conj(F(source_f * pupil)) * F(pupil)
    real_tf = 2.0 * IF(FP_cFSP.real)
    imag_tf = 2.0 * IF(1j * FP_cFSP.imag)
    DC = (source_f * np.abs(pupil) ** 2).sum()
    if DC == 0:
        DC = 1.0
    return 1j * imag_tf / DC, real_tf / DC      # (Hi, Hr)


# --------------------------------------------------------------------------- #
# Zernike basis on the pupil grid  -- port of genZernikePoly.m (drops 1st 3)
# --------------------------------------------------------------------------- #
def _gen_zernike_basis(Fx, Fy, na, wavelength, highest_order):
    """Columns = OSA/ANSI Zernike modes sampled on the pupil; the first three
    (piston + two tilts) are discarded. ``Fx, Fy`` are 2-D frequency grids in
    the same FFT layout as the pupil. Returns (Npix, highest_order-3) float64."""
    rho = np.sqrt(Fx ** 2 + Fy ** 2) / (na / wavelength)
    theta = np.arctan2(Fy, Fx)
    pupil = (Fx ** 2 + Fy ** 2 <= (na / wavelength) ** 2).astype("float64")
    cols = []
    for j in range(highest_order):
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
        m = 2 * j - n * (n + 2)
        am = abs(m)
        R = np.zeros_like(rho)
        for k in range((n - am) // 2 + 1):
            num = ((-1) ** k) * factorial(n - k)
            den = factorial(k) * factorial((n + am) // 2 - k) * factorial((n - am) // 2 - k)
            R += (num / den) * rho ** (n - 2 * k)
        ang = np.sin(am * theta) if m < 0 else np.cos(am * theta)
        cols.append((pupil * R * ang).ravel())
    basis = np.asarray(cols).T              # (Npix, highest_order)
    return basis[:, 3:]                     # discard piston + 2 tilts


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class Chen2018AberrationDPC:
    """Joint complex-field + pupil-aberration recovery from DPC measurements.

    Parameters
    ----------
    dpc_imgs : (N, H, W) array
        N = (dpc_num half-circles) + (1 if use_single_led). Order must match
        ``rotation`` for the half-circles, single LED last.
    wavelength, na, pixel_size : floats (um, NA, um/px)
    rotation : list[float]
        Half-circle boundary angles, e.g. [0, 180, 90] (top/bottom/left).
    na_illum : float or None
        Illumination NA of the half-disks (default = na, i.e. sigma = 1).
    na_inner : float
        Inner NA for annular illumination (default 0 = full half-disk).
    num_zernike : int
        Highest OSA order generated; first 3 dropped -> (num_zernike-3) coeffs.
    use_single_led : bool
        If True the last image is the on-axis LED (DC pixel source).
    normalize : bool
        Apply the reference self-normalization (no blank needed). Set False if
        the input is already background-subtracted / DC-removed.
    """

    def __init__(self, dpc_imgs, wavelength, na, pixel_size, rotation,
                 na_illum=None, na_inner=0.0, num_zernike=21,
                 use_single_led=True, normalize=True):
        self.wavelength = wavelength
        self.na = na
        self.na_illum = na if na_illum is None else na_illum
        self.na_inner = na_inner
        self.pixel_size = pixel_size
        self.rotation = rotation
        self.use_single_led = use_single_led
        self.imgs = np.asarray(dpc_imgs, dtype="float64").copy()
        H, W = self.imgs.shape[-2:]

        self.fxlin = np.fft.ifftshift(_gen_grid(W, 1.0 / W / pixel_size))
        self.fylin = np.fft.ifftshift(_gen_grid(H, 1.0 / H / pixel_size))
        Fx = np.real(self.fxlin)[naxis, :] * np.ones((H, 1))
        Fy = np.real(self.fylin)[:, naxis] * np.ones((1, W))

        if normalize:
            self._normalize()
        self.fIDPC = np.asarray([F(im) for im in self.imgs])

        # objective (amplitude) pupil + Zernike basis live on the objective NA
        self.pupil = pupil_gen(self.fxlin, self.fylin, wavelength, na).astype("complex128")
        self.zernike_basis = _gen_zernike_basis(Fx, Fy, na, wavelength, num_zernike)
        self.n_modes = self.zernike_basis.shape[1]
        self._build_sources()

    # ----- self-normalization (no blank/empty needed) ---------------------- #
    def _normalize(self):
        for img in self.imgs:
            img /= uniform_filter(img, size=img.shape[0] // 2)
            img /= img.mean()
            img -= 1.0

    # ----- sources: (n_half) half-circles + optional on-axis LED ----------- #
    def _build_sources(self):
        n_half = len(self.imgs) - 1 if self.use_single_led else len(self.imgs)
        illum = pupil_gen(self.fxlin, self.fylin, self.wavelength,
                          self.na_illum, na_in=self.na_inner)
        H, W = self.imgs.shape[-2:]
        src = []
        for i in range(n_half):
            s = np.zeros((H, W))
            rot = self.rotation[i]
            if rot < 180:
                s[self.fylin[:, naxis] * np.cos(np.deg2rad(rot)) + 1e-15 >=
                  self.fxlin[naxis, :] * np.sin(np.deg2rad(rot))] = 1.0
                s *= illum
            else:
                s[self.fylin[:, naxis] * np.cos(np.deg2rad(rot)) + 1e-15 <
                  self.fxlin[naxis, :] * np.sin(np.deg2rad(rot))] = -1.0
                s *= illum
                s += illum
            src.append(s.real.astype("float64"))
        if self.use_single_led:
            led = np.zeros((H, W))
            led[0, 0] = 1.0                  # on-axis LED = DC pixel in FFT layout
            src.append(led)
        self.source = np.asarray(src)

    # ----- aberrated complex pupil ----------------------------------------- #
    def _pupil_phase(self, coeffs):
        return (self.zernike_basis @ np.asarray(coeffs, dtype="float64")).reshape(self.pupil.shape)

    def _aberrated_pupil(self, coeffs):
        return self.pupil * np.exp(1j * self._pupil_phase(coeffs))

    # ----- sub-problem 1: fix pupil, recover (absorption, phase) -- DPC_L2.m  #
    def _dpc_l2(self, coeffs, reg):
        pupil_ab = self._aberrated_pupil(coeffs)
        Hi, Hr = [], []
        for s in range(self.source.shape[0]):
            hi, hr = _gen_transfer_function(self.source[s], pupil_ab)
            Hi.append(hi); Hr.append(hr)
        Hi = np.asarray(Hi); Hr = np.asarray(Hr)
        M11 = (np.abs(Hr) ** 2).sum(0) + reg[0]
        M12 = (Hr.conj() * Hi).sum(0)
        M21 = (Hi.conj() * Hr).sum(0)
        M22 = (np.abs(Hi) ** 2).sum(0) + reg[1]
        det = M11 * M22 - M12 * M21
        I1 = (self.fIDPC * Hr.conj()).sum(0)
        I2 = (self.fIDPC * Hi.conj()).sum(0)
        amplitude = IF((I1 * M22 - I2 * M12) / det).real
        phase = IF((I2 * M11 - I1 * M21) / det).real
        return amplitude, phase

    # ----- sub-problem 2: fix field, recover pupil -- gradientPupil.m -------- #
    # Returns (cost, analytic gradient w.r.t. Zernike coeffs). f_amp/f_phase are
    # the FFTs of the current absorption/phase estimate, set before calling.
    def _grad_pupil(self, coeffs):
        pupil_est = self._aberrated_pupil(coeffs)
        f = 0.0
        g = np.zeros(self.n_modes, dtype="complex128")
        for s in range(self.source.shape[0]):
            source_f = _reflect_source(self.source[s])
            DC = (source_f * np.abs(pupil_est) ** 2).sum()
            if DC == 0:
                DC = 1.0
            f_sp = F(source_f * pupil_est)
            f_p = F(pupil_est)
            H_first = f_sp.conj() * f_p
            H_second = f_p.conj() * f_sp
            residual = self.fIDPC[s] - (
                IF(H_first + H_second) * self.f_amp
                + 1j * IF(H_first - H_second) * self.f_phase) / DC
            f += 0.5 * float(np.sum(np.abs(residual) ** 2))
            backprop_1 = F(self.f_amp.conj() * residual)
            backprop_2 = F(-1j * self.f_phase.conj() * residual)
            grad_pupil = (IF(f_sp * (backprop_1 + backprop_2))
                          + source_f * IF(f_p * (backprop_1 - backprop_2))) / DC
            g -= self.zernike_basis.T @ (-1j * pupil_est.conj() * grad_pupil).ravel()
        return f, g.real

    # ----- alternating outer loop -- main_dpc.m ---------------------------- #
    def solve(self, reg=(1e-1, 5e-3), max_outer=50, inner_maxiter=10, verbose=True):
        coeffs = np.zeros(self.n_modes, dtype="float64")
        loss = []
        for it in range(max_outer):
            amp, ph = self._dpc_l2(coeffs, reg)
            self.f_amp = F(amp)
            self.f_phase = F(ph)
            res = minimize(self._grad_pupil, coeffs, jac=True, method="L-BFGS-B",
                           options={"maxiter": inner_maxiter, "ftol": 1e-30,
                                    "gtol": 1e-30, "maxcor": 50})
            coeffs = res.x
            loss.append(float(res.fun))
            if verbose:
                print(f"  [outer {it + 1:02d}/{max_outer}] loss={res.fun:.4e}  "
                      f"c[:3]={np.round(coeffs[:3], 4)}")
        amp, ph = self._dpc_l2(coeffs, reg)
        return dict(absorption=amp, phase=ph, zernike=coeffs,
                    pupil_phase=self._pupil_phase(coeffs), loss=np.asarray(loss))


# --------------------------------------------------------------------------- #
# turnkey runner: capture 3 half-circles + 1 LED, then run this
# --------------------------------------------------------------------------- #
# Expected capture (this lab, 10x / NA 0.40, SCI Dome green 530 nm):
#   top    half-circle  (rotation 0)     filename tag: th / top
#   bottom half-circle  (rotation 180)   filename tag: bh / bottom
#   left   half-circle  (rotation 90)    filename tag: lh / left
#   single on-axis LED                   filename tag: led / single / dc / center
# Illumination NA of the half-disks = NA_obj = 0.40 (sigma = 1), full half-disk
# (no annulus). The single LED = one central LED only.
_TAG_PATTERNS = {
    "top":    ["*_th_*", "*_top*", "*top_half*", "*tophalf*"],
    "bottom": ["*_bh_*", "*_bottom*", "*bottom_half*", "*bottomhalf*"],
    "left":   ["*_lh_*", "*_left*", "*left_half*", "*lefthalf*"],
    "led":    ["*_led_*", "*single*", "*_dc_*", "*center*", "*on_axis*", "*onaxis*"],
}


def _find_one(data_dir, patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat + ".tif")))
        if not hits:
            hits = sorted(glob.glob(os.path.join(data_dir, "**", pat + ".tif"),
                                    recursive=True))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"ambiguous match for {patterns}: {hits}")
    raise FileNotFoundError(f"no file matched any of {patterns} under {data_dir}")


def run(top, bottom, left, led, out_dir,
        wavelength=0.530, na=0.40, na_illum=0.40, mag=10.0, pixel_cam=6.5,
        num_zernike=21, reg=(1e-1, 5e-3), max_outer=50, inner_maxiter=10,
        normalize=True, verbose=True):
    """Load the 4 capture files (3 half-circle + 1 LED), run Chen-2018
    aberration-corrected DPC, and write the recovered phase/OPL/absorption,
    the recovered pupil aberration, Zernike coefficients and loss curve."""
    import tifffile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [top, bottom, left, led]
    print("[capture] using:")
    for name, p in zip(["top(0)", "bottom(180)", "left(90)", "single-LED"], paths):
        print(f"    {name:12s} {p}")
    imgs = np.asarray([tifffile.imread(p).astype("float64") for p in paths])

    pixel_size = pixel_cam / mag
    solver = Chen2018AberrationDPC(
        imgs, wavelength, na, pixel_size, rotation=[0, 180, 90],
        na_illum=na_illum, na_inner=0.0, num_zernike=num_zernike,
        use_single_led=True, normalize=normalize)
    res = solver.solve(reg=reg, max_outer=max_outer, inner_maxiter=inner_maxiter,
                       verbose=verbose)

    os.makedirs(out_dir, exist_ok=True)
    phase = res["phase"]
    opl_nm = phase * wavelength * 1000.0 / (2 * np.pi)
    tifffile.imwrite(os.path.join(out_dir, "phase_rad.tif"), phase.astype("float32"), imagej=True)
    tifffile.imwrite(os.path.join(out_dir, "opl_nm.tif"), opl_nm.astype("float32"), imagej=True)
    tifffile.imwrite(os.path.join(out_dir, "absorption.tif"), res["absorption"].astype("float32"), imagej=True)
    tifffile.imwrite(os.path.join(out_dir, "pupil_phase_rad.tif"),
                     np.fft.fftshift(res["pupil_phase"]).astype("float32"), imagej=True)

    with open(os.path.join(out_dir, "zernike_coeffs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["osa_index_from_3", "coeff_rad"])
        for i, c in enumerate(res["zernike"]):
            w.writerow([i + 3, c])
    with open(os.path.join(out_dir, "loss.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["outer_iter", "loss"])
        for i, l in enumerate(res["loss"]):
            w.writerow([i + 1, l])

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    im0 = ax[0, 0].imshow(phase, cmap="gray"); ax[0, 0].set_title("recovered phase (rad)")
    ax[0, 0].axis("off"); fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    im1 = ax[0, 1].imshow(res["absorption"], cmap="gray"); ax[0, 1].set_title("recovered absorption")
    ax[0, 1].axis("off"); fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    im2 = ax[1, 0].imshow(np.fft.fftshift(res["pupil_phase"]), cmap="jet")
    ax[1, 0].set_title("recovered pupil aberration (rad)"); ax[1, 0].axis("off")
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].plot(np.arange(1, len(res["loss"]) + 1), np.log10(res["loss"] + 1e-30), "bo-")
    ax[1, 1].set_xlabel("outer iteration"); ax[1, 1].set_ylabel("log10(loss)")
    ax[1, 1].set_title("convergence"); ax[1, 1].set_aspect("auto")
    fig.suptitle("Chen-2018 aberration-corrected DPC", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"[done] wrote phase/OPL/absorption/pupil + zernike/loss + summary.png to {out_dir}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder holding the 4 TIFFs (auto-matched by tag)")
    ap.add_argument("--top"); ap.add_argument("--bottom")
    ap.add_argument("--left"); ap.add_argument("--led")
    ap.add_argument("--out", required=True, help="output directory (dataset-side analysis/)")
    ap.add_argument("--wavelength", type=float, default=0.530)
    ap.add_argument("--na", type=float, default=0.40)
    ap.add_argument("--na-illum", type=float, default=0.40)
    ap.add_argument("--mag", type=float, default=10.0)
    ap.add_argument("--pixel-cam", type=float, default=6.5)
    ap.add_argument("--num-zernike", type=int, default=21)
    ap.add_argument("--reg-u", type=float, default=1e-1)
    ap.add_argument("--reg-p", type=float, default=5e-3)
    ap.add_argument("--max-outer", type=int, default=50)
    ap.add_argument("--inner-maxiter", type=int, default=10)
    ap.add_argument("--no-normalize", action="store_true",
                    help="input is already background-subtracted / DC-removed")
    args = ap.parse_args()

    if args.data:
        top = args.top or _find_one(args.data, _TAG_PATTERNS["top"])
        bottom = args.bottom or _find_one(args.data, _TAG_PATTERNS["bottom"])
        left = args.left or _find_one(args.data, _TAG_PATTERNS["left"])
        led = args.led or _find_one(args.data, _TAG_PATTERNS["led"])
    else:
        if not all([args.top, args.bottom, args.left, args.led]):
            ap.error("provide either --data DIR or all of --top/--bottom/--left/--led")
        top, bottom, left, led = args.top, args.bottom, args.left, args.led

    run(top, bottom, left, led, args.out,
        wavelength=args.wavelength, na=args.na, na_illum=args.na_illum,
        mag=args.mag, pixel_cam=args.pixel_cam, num_zernike=args.num_zernike,
        reg=(args.reg_u, args.reg_p), max_outer=args.max_outer,
        inner_maxiter=args.inner_maxiter, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()
