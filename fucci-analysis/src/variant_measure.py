"""Stage 3 -- per-cell reporter intensity with per-image background + erosion sweep.

For each analysis well: load the BF label mask, the green (488) and red (2nd
reporter) snapshots. Estimate per-image background from non-cell pixels
(mask == 0): bg_median and a robust bg_sigma (1.4826 * MAD). For every cell and
every erosion level in cfg.EROSION_SWEEP_PX, erode the cell mask to approximate
the nuclear region and record raw mean, background-subtracted mean, and
SNR = (mean - bg_median) / bg_sigma for both channels.

Output <analysis>/per_cell_measurements.csv (long form; one row per cell x erosion).
Background subtraction only -- NO flat-field (Heeseok 2026-06-10). Also writes
bg_summary.csv so any illumination gradient can be flagged as QC, not corrected.

Run:  conda activate fucci-analysis
      python fucci-analysis/src/variant_measure.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion, find_objects

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import variant_config as cfg  # noqa: E402


def background_stats(img: np.ndarray, cell_mask_any: np.ndarray) -> tuple[float, float, float]:
    """median, robust sigma (1.4826*MAD), and frac of background pixels."""
    bg = img[~cell_mask_any]
    if bg.size == 0:
        return float("nan"), float("nan"), 0.0
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(bg))
    return med, sigma, float(bg.size / img.size)


LOCAL_BG_MARGIN_PX = 40  # half-window around each cell bbox to sample local background


def _local_bg(img: np.ndarray, cell_any: np.ndarray, sl: tuple,
              g_med_global: float, g_sig_global: float) -> tuple[float, float]:
    """Local background (median, robust sigma) from non-cell pixels in a window
    around the cell bbox. Removes the illumination gradient that corrupts a global
    background at this field size. Falls back to global if the window is too crowded."""
    ys, xs = sl
    y0 = max(0, ys.start - LOCAL_BG_MARGIN_PX); y1 = min(img.shape[0], ys.stop + LOCAL_BG_MARGIN_PX)
    x0 = max(0, xs.start - LOCAL_BG_MARGIN_PX); x1 = min(img.shape[1], xs.stop + LOCAL_BG_MARGIN_PX)
    win = img[y0:y1, x0:x1]
    bgmask = ~cell_any[y0:y1, x0:x1]
    bgpix = win[bgmask]
    if bgpix.size < 200:
        return g_med_global, g_sig_global
    med = float(np.median(bgpix))
    mad = float(np.median(np.abs(bgpix - med)))
    sig = 1.4826 * mad if mad > 0 else g_sig_global
    return med, sig


def measure_well(mask: np.ndarray, green: np.ndarray, red: np.ndarray,
                 meta: dict) -> tuple[list[dict], dict]:
    cell_any = mask > 0
    green = green.astype(np.float64); red = red.astype(np.float64)
    # global background kept for the gradient QC note (NOT used for per-cell values)
    g_med, g_sig, g_frac = background_stats(green, cell_any)
    r_med, r_sig, r_frac = background_stats(red, cell_any)

    slices = find_objects(mask)
    rows: list[dict] = []
    for lbl, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sub = mask[sl] == lbl
        area = int(sub.sum())
        g_crop = green[sl]; r_crop = red[sl]
        # per-cell LOCAL background (gradient-robust); primary for all per-cell values
        g_lmed, g_lsig = _local_bg(green, cell_any, sl, g_med, g_sig)
        r_lmed, r_lsig = _local_bg(red, cell_any, sl, r_med, r_sig)
        for e in cfg.EROSION_SWEEP_PX:
            er = binary_erosion(sub, iterations=e)
            if not er.any():
                er = sub  # tiny cell: fall back to full footprint
            nuc_area = int(er.sum())
            g_mean = float(g_crop[er].mean())
            r_mean = float(r_crop[er].mean())
            rows.append({
                **meta,
                "cell_id": lbl, "area_px": area, "erosion_px": e,
                "nuclear_area_px": nuc_area,
                "green_raw": g_mean, "red_raw": r_mean,
                "green_local_bg": g_lmed, "red_local_bg": r_lmed,
                "green_bgsub": g_mean - g_lmed, "red_bgsub": r_mean - r_lmed,
                "green_snr": (g_mean - g_lmed) / g_lsig if g_lsig > 0 else float("nan"),
                "red_snr": (r_mean - r_lmed) / r_lsig if r_lsig > 0 else float("nan"),
                # global-background versions kept for QC comparison only
                "green_bgsub_global": g_mean - g_med, "red_bgsub_global": r_mean - r_med,
            })
    bg_row = {**meta, "green_bg_med": g_med, "green_bg_sigma": g_sig,
              "red_bg_med": r_med, "red_bg_sigma": r_sig,
              "green_bg_frac": g_frac, "red_bg_frac": r_frac,
              "n_cells": int(mask.max())}
    return rows, bg_row


def main() -> None:
    manifest = pd.read_csv(cfg.ANALYSIS_DIR / "manifest_analysis.csv")
    # pivot channel paths per well
    wide = manifest.pivot_table(
        index=["construct", "well", "drug", "replicate"],
        columns="role", values="path", aggfunc="first").reset_index()

    all_rows: list[dict] = []
    bg_rows: list[dict] = []
    t0 = time.time()
    for i, r in wide.iterrows():
        tag = f"{r['construct']}_{r['well']}"
        mask_path = cfg.MASKS_DIR / f"{tag}_BF_mask.npy"
        if not mask_path.exists():
            print(f"[WARN] missing mask {mask_path}; run variant_segment.py first")
            continue
        mask = np.load(mask_path)
        green = tifffile.imread(r["green"])
        red = tifffile.imread(r["red"])
        meta = {"construct": r["construct"], "well": r["well"],
                "drug": r["drug"], "replicate": int(r["replicate"]),
                "green_channel": cfg.CONSTRUCTS[r["construct"]]["green"],
                "red_channel": cfg.CONSTRUCTS[r["construct"]]["red"]}
        rows, bg_row = measure_well(mask, green, red, meta)
        all_rows.extend(rows)
        bg_rows.append(bg_row)
        print(f"[{i+1:02d}/{len(wide)}] {tag} drug={r['drug']} rep={r['replicate']} "
              f"cells={bg_row['n_cells']}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    out = cfg.ANALYSIS_DIR / "per_cell_measurements.csv"
    df.to_csv(out, index=False)
    print(f"\n[SAVED] {out}  rows={len(df)} "
          f"(cells x {len(cfg.EROSION_SWEEP_PX)} erosion levels)")

    bg = pd.DataFrame(bg_rows)
    bg_out = cfg.ANALYSIS_DIR / "bg_summary.csv"
    bg.to_csv(bg_out, index=False)
    print(f"[SAVED] {bg_out}")


if __name__ == "__main__":
    main()
