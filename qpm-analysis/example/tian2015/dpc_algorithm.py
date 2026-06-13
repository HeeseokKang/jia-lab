# ── DPC Phase Reconstruction  (Tian & Waller, Optics Express 23(9), 2015) ──
# Implements the Weak-Object Transfer Function (WOTF) / Tikhonov solver from the paper.
# This is the *original* 2015 engine: Tikhonov regularization only, float64 throughout.
#
# How it differs from chen2018_aberration/dpc_algorithm.py:
#   • No TV regularization (chen2018 adds deconvTV + ADMM).
#   • float64 here vs float32 in chen2018.
#   • dpc_num is HARD-CODED to 4 in __init__ (ignores the constructor argument) —
#     chen2018 correctly stores the passed value.
#   • No _softThreshold helper; solve() is a single Tikhonov path (no method dispatch).
#   • No docstrings — this is the unmodified Waller-Lab 2015 source.

import numpy as np
from scipy.ndimage import uniform_filter
pi    = np.pi
naxis = np.newaxis
F     = lambda x: np.fft.fft2(x)   # forward 2-D DFT  (spatial → frequency)
IF    = lambda x: np.fft.ifft2(x)  # inverse 2-D DFT  (frequency → spatial)

def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
    # Creates a binary coherent-transfer-function (CTF) pupil mask in frequency space.
    # All spatial frequencies |f| ≤ NA/λ are set to 1 (objective can collect them);
    # everything outside the circle is 0 (blocked by the objective aperture stop).
    # na_in > 0 punches out a central disk → annular pupil (used for sourceGen with na_in).
    pupil = np.array(fxlin[naxis, :]**2+fylin[:, naxis]**2 <= (na/wavelength)**2)
    if na_in != 0.0:
        pupil[fxlin[naxis, :]**2+fylin[:, naxis]**2 < (na_in/wavelength)**2] = 0.0
    return pupil

def _genGrid(size, dx):
    # Returns a *centered* 1-D coordinate vector of length `size` with step `dx`.
    # Index 0 → coordinate −(size//2)·dx, index size//2 → coordinate 0 (DC).
    # dtype complex128 (float64 real/imag) — note chen2018 uses complex64 here.
    xlin = np.arange(size, dtype='complex128')
    return (xlin-size//2)*dx

class DPCSolver:
    def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, rotation, dpc_num=4):
        self.wavelength = wavelength
        self.na         = na
        self.na_in      = na_in        # illumination inner NA (0 = full half-disk)
        self.pixel_size = pixel_size
        self.dpc_num    = 4            # ← always 4; the dpc_num argument is silently ignored
        self.rotation   = rotation     # list of half-plane boundary angles [deg], e.g. [0,90,180,270]
        # Frequency axes in cycles/µm, centered so DC = 0, then re-ordered to FFT layout
        # (DC at index 0) via ifftshift so element-wise ops with FFT outputs are aligned.
        self.fxlin      = np.fft.ifftshift(_genGrid(dpc_imgs.shape[-1], 1.0/dpc_imgs.shape[-1]/self.pixel_size))
        self.fylin      = np.fft.ifftshift(_genGrid(dpc_imgs.shape[-2], 1.0/dpc_imgs.shape[-2]/self.pixel_size))
        self.dpc_imgs   = dpc_imgs.astype('float64')
        self.normalization()           # self-normalize each raw DPC image in-place
        self.pupil      = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.sourceGen()               # half-plane source masks  →  self.source
        self.WOTFGen()                 # absorption + phase TFs   →  self.Hu, self.Hp

    def setTikhonovRegularization(self, reg_u = 1e-6, reg_p = 1e-6):
        # reg_u: regularization weight on absorption (u); reg_p: on phase (p).
        # Increase to suppress noise at the cost of spatial resolution.
        # Must be called before solve(); there are no defaults set in __init__.
        self.reg_u      = reg_u
        self.reg_p      = reg_p

    def normalization(self):
        # Self-normalization — no blank/empty-field reference needed.
        # This mimics the DPC ratio (I_top − I_bot)/(I_top + I_bot) which is
        # inherently self-normalizing.  The two steps are:
        #   1. Divide by uniform_filter (local mean over a window = half the image width)
        #      to flatten slowly-varying background (lamp non-uniformity, vignetting).
        #   2. Divide by the global mean (≈ DC term) so the image is dimensionless
        #      and centred near 1, then subtract 1 to shift DC to 0.
        # Result: each image ≈ (I − ⟨I⟩) / ⟨I⟩ — fractional intensity deviation,
        # centred on 0, in a form directly comparable to the WOTF model.
        for img in self.dpc_imgs:
            img          /= uniform_filter(img, size=img.shape[0]//2)
            meanIntensity = img.mean()
            img          /= meanIntensity        # normalize intensity with DC term
            img          -= 1.0                  # subtract the DC term

    def sourceGen(self):
        # Builds a binary half-plane illumination mask in frequency space for each
        # DPC exposure.  Each mask represents which spatial frequencies are illuminated
        # by that half of the LED array, clipped to the objective pupil.
        #
        # The dividing line between lit and dark halves passes through the origin at
        # angle θ (rotation[rotIdx]) measured from the fy-axis:
        #   fy·cos(θ) = fx·sin(θ)
        #
        # For θ < 180°:  lit side is  fy·cos(θ) ≥ fx·sin(θ)  (direct threshold).
        #   θ=0°  → fy ≥ 0 → top half
        #   θ=90° → fx ≤ 0 → left half
        #
        # For θ ≥ 180°:  the complement half is built by the 3-step trick:
        #   (a) set −1 where the opposite condition holds,
        #   (b) multiply by pupil  (−pupil on that side, 0 elsewhere),
        #   (c) add full pupil     (0 on that side, +pupil on the desired side).
        #   θ=180° → bottom half (fy ≤ 0)
        #   θ=270° → right half  (fx ≥ 0)
        #
        # The +1e-15 epsilon prevents numerical ties exactly on the boundary line.
        self.source = []
        pupil       = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na, na_in=self.na_in)
        for rotIdx in range(self.dpc_num):
            self.source.append(np.zeros((self.dpc_imgs.shape[-2:])))
            rotdegree = self.rotation[rotIdx]
            if rotdegree < 180:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= pupil
            else:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= pupil
                self.source[-1] += pupil
        self.source = np.asarray(self.source)

    def WOTFGen(self):
        # Computes the Weak-Object Transfer Functions (WOTF) — Eqs. (5–6) of the paper.
        #
        # Under the weak-object (Born) approximation, the DPC image in Fourier space is:
        #   Î_DPC(f) = Ĥ_u(f)·û(f) + Ĥ_p(f)·p̂(f)
        # where û = absorption component, p̂ = phase component (what we want to recover).
        #
        # FSP_cFP = FFT(S·P) · FFT(P)*
        #   The product of two Fourier-domain quantities (source-weighted pupil vs pupil).
        #   Its IFFT gives the cross-correlation (S·P) ⊛ P as a function of frequency
        #   shift — this is the raw WOTF kernel before splitting into absorption/phase.
        #
        # I0 = Σ S·|P|²  — total DC intensity from the lit aperture; normalises the TFs.
        #
        # Absorption TF:  Hu = (2/I0) · IFFT( Re[FSP_cFP] )
        #   The real (even-symmetric) part of the cross-correlation maps to absorption.
        #
        # Phase TF:  Hp = (2j/I0) · IFFT( j·Im[FSP_cFP] )
        #               = −(2/I0) · IFFT( Im[FSP_cFP] )
        #   The imaginary (odd-symmetric) part maps to phase.  The extra j factors ensure
        #   Hp is purely imaginary (anti-Hermitian in Fourier space), consistent with the
        #   physical constraint that phase contributes only an odd-symmetry signal to the
        #   DPC ratio.
        #
        # Both Hu and Hp are 2-D arrays indexed by the same (H, W) grid as the images;
        # element-wise multiplication with FFT'd images in solve() performs per-frequency
        # inner products across the illumination stack.
        self.Hu = []
        self.Hp = []
        for rotIdx in range(self.source.shape[0]):
            FSP_cFP  = F(self.source[rotIdx]*self.pupil)*F(self.pupil).conj()
            I0       = (self.source[rotIdx]*self.pupil*self.pupil.conj()).sum()
            self.Hu.append(2.0*IF(FSP_cFP.real)/I0)
            self.Hp.append(2.0j*IF(1j*FSP_cFP.imag)/I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def solve(self, xini=None, plot_verbose=False, **kwargs):
        # Tikhonov least-squares inversion in Fourier space — Eq. (7) of the paper.
        # xini and plot_verbose are unused stub parameters (vestigial from an earlier API).
        #
        # The forward model across all dpc_num illuminations stacks as:
        #   [Hu₀ Hp₀]   [û]   [Î₀]
        #   [Hu₁ Hp₁] · [p̂] = [Î₁]   (each row = one DPC direction, in Fourier space)
        #   [ ⋮   ⋮ ]         [ ⋮]
        #
        # Tikhonov normal equations (solved independently at each frequency pixel):
        #   (ÂᴴÂ + λI) x̂ = Âᴴ ŷ
        #
        # AHA is the 2×2 Gramian  ÂᴴÂ + λI, stored as a flat list of 4 elements:
        #   AHA = [ ΣHu*·Hu + reg_u,  ΣHu*·Hp  ]
        #         [ ΣHp*·Hu,          ΣHp*·Hp + reg_p ]
        # where Σ sums over the dpc_num illumination directions.
        #
        # AHy = Âᴴ ŷ (the right-hand side):
        #   AHy[0] = Σ Hu*·Î_DPC   (matched-filter output for absorption)
        #   AHy[1] = Σ Hp*·Î_DPC   (matched-filter output for phase)
        #
        # Cramer's rule solves the 2×2 system analytically per frequency pixel:
        #   û = (AHA[3]·AHy[0] − AHA[1]·AHy[1]) / det
        #   p̂ = (AHA[0]·AHy[1] − AHA[2]·AHy[0]) / det
        # IFFT converts û and p̂ back to the spatial domain.
        #
        # The outer loop over frame_index handles time-lapse stacks where dpc_imgs
        # contains multiple sets of dpc_num images interleaved in time.
        # Result encoding: real part = absorption,  imag part = phase.
        dpc_result  = []
        AHA         = [(self.Hu.conj()*self.Hu).sum(axis=0)+self.reg_u,            (self.Hu.conj()*self.Hp).sum(axis=0),\
                       (self.Hp.conj()*self.Hu).sum(axis=0)           , (self.Hp.conj()*self.Hp).sum(axis=0)+self.reg_p]
        determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
        for frame_index in range(self.dpc_imgs.shape[0]//self.dpc_num):
            fIntensity = np.asarray([F(self.dpc_imgs[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
            AHy        = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0), (self.Hp.conj()*fIntensity).sum(axis=0)])
            absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
            phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
            dpc_result.append(absorption+1.0j*phase)

        return np.asarray(dpc_result)
