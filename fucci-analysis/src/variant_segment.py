"""Stage 1 -- BF segmentation (Cellpose-SAM) for every analysis well.

Reads manifest_analysis.csv, runs Cellpose-SAM (cpsam) on each BF snapshot, and
writes an integer label mask per well to <analysis>/masks/<construct>_<well>_BF_mask.npy
(heavy, dataset-side). Resumable: existing masks are skipped unless --overwrite.
Writes seg_summary.csv (n_cells per well) for QC.

Run:  conda activate fucci-analysis
      python fucci-analysis/src/variant_segment.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from cellpose import models

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import variant_config as cfg  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cellpose-SAM BF segmentation for the FUCCI variant run.")
    ap.add_argument("--overwrite", action="store_true", help="re-segment wells that already have a mask")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg.MASKS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg.ANALYSIS_DIR / "manifest_analysis.csv")
    bf = manifest[manifest["role"] == "bf"].sort_values(["construct", "well"]).reset_index(drop=True)
    print(f"[SEG] {len(bf)} BF wells to segment (overwrite={args.overwrite})")

    print("[CELLPOSE] loading Cellpose-SAM (cpsam)")
    model = models.CellposeModel(gpu=True, model_type="cpsam")

    rows: list[dict] = []
    t0 = time.time()
    for i, r in bf.iterrows():
        tag = f"{r['construct']}_{r['well']}"
        mask_path = cfg.MASKS_DIR / f"{tag}_BF_mask.npy"
        if mask_path.exists() and not args.overwrite:
            masks = np.load(mask_path)
            n = int(masks.max())
            print(f"[{i+1:02d}/{len(bf)}] {tag}: skip (exists, n_cells={n})")
        else:
            img = tifffile.imread(r["path"])
            masks, _flows, _styles = model.eval(img, diameter=None)
            n = int(masks.max())
            np.save(mask_path, masks.astype(np.int32))
            print(f"[{i+1:02d}/{len(bf)}] {tag}: n_cells={n}  ({time.time()-t0:.0f}s elapsed)")
        rows.append({"construct": r["construct"], "well": r["well"],
                     "drug": r["drug"], "replicate": r["replicate"],
                     "n_cells": n, "mask_path": str(mask_path)})

    summary = pd.DataFrame(rows)
    out = cfg.ANALYSIS_DIR / "seg_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n[SAVED] {out}")
    print(summary.groupby(["construct", "drug"])["n_cells"].agg(["mean", "min", "max"]).round(0).to_string())
    print(f"[DONE] total {time.time()-t0:.0f}s, masks in {cfg.MASKS_DIR}")


if __name__ == "__main__":
    main()
