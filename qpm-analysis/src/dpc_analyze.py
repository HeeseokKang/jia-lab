"""Mean-balanced DPC analysis for the 20260513 Hela QPM dataset.

Pipeline:
  1. Load BF + 4 directional LED images as float64.
  2. Mean-balance each image (divide by its own global mean) before forming DPC.
     This removes a measured ~11% LED asymmetry that would otherwise contaminate
     DPC_x with a +0.11 offset that is NOT phase signal.
  3. Decompose pixel std into a background region (BF > 70th percentile) and
     a candidate-cell region (BF < 30th percentile). Background dominates raw
     std (~92% of pixels), so std must be reported separately to be useful as
     a contrast proxy.

DPC std is a contrast proxy ONLY — not a quantitative phase metric. Quantitative
phase requires WOTF-based Tikhonov reconstruction (next pipeline step).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

EPS = 1e-6
BG_PERCENTILE = 70.0
CELL_PERCENTILE = 30.0

DIRECTIONS = ("bffull", "top", "bottom", "left", "right")


def _find_one(folder: Path, key: str) -> Path:
    candidates = sorted(p for p in folder.glob(f"*{key}*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No '*{key}*.tif' in {folder}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous '*{key}*.tif' in {folder}: {[p.name for p in candidates]}"
        )
    return candidates[0]


def load_condition(folder: Path) -> dict[str, np.ndarray]:
    """Load BF + 4 directional LED images as float64 arrays."""
    return {
        key: tifffile.imread(_find_one(folder, key)).astype(np.float64)
        for key in DIRECTIONS
    }


def _std_in_mask(arr: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.std(arr[mask]))


def compute_dpc_balanced(
    T: np.ndarray,
    B: np.ndarray,
    L: np.ndarray,
    R: np.ndarray,
    bf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Mean-balance LED pairs, then form DPC = (A-B)/(A+B).

    Returns (dpc_y, dpc_x, stats). If `bf` is provided, std is also reported
    separately for background (BF > p70) and cell-candidate (BF < p30) regions.
    """
    T_mean, B_mean = float(np.mean(T)), float(np.mean(B))
    L_mean, R_mean = float(np.mean(L)), float(np.mean(R))

    # Asymmetry as a percentage of pair average.
    tb_pair = 0.5 * (T_mean + B_mean)
    lr_pair = 0.5 * (L_mean + R_mean)
    tb_asym = 100.0 * (T_mean - B_mean) / tb_pair if tb_pair else float("nan")
    lr_asym = 100.0 * (L_mean - R_mean) / lr_pair if lr_pair else float("nan")

    # Raw DPC (no balancing) — used only to report the offset artifact.
    dpc_y_raw = (T - B) / (T + B + EPS)
    dpc_x_raw = (L - R) / (L + R + EPS)

    # Mean-balanced DPC: divide each image by its own global mean first.
    Tn, Bn = T / T_mean, B / B_mean
    Ln, Rn = L / L_mean, R / R_mean
    dpc_y = (Tn - Bn) / (Tn + Bn + EPS)
    dpc_x = (Ln - Rn) / (Ln + Rn + EPS)

    stats: dict[str, float] = {
        "T_mean": T_mean,
        "B_mean": B_mean,
        "L_mean": L_mean,
        "R_mean": R_mean,
        "TB_asym_pct": tb_asym,
        "LR_asym_pct": lr_asym,
        "DPC_y_raw_mean": float(np.mean(dpc_y_raw)),
        "DPC_x_raw_mean": float(np.mean(dpc_x_raw)),
        "DPC_y_bal_std": float(np.std(dpc_y)),
        "DPC_x_bal_std": float(np.std(dpc_x)),
    }

    if bf is not None:
        p70 = float(np.percentile(bf, BG_PERCENTILE))
        p30 = float(np.percentile(bf, CELL_PERCENTILE))
        bg_mask = bf > p70
        cell_mask = bf < p30
        stats.update(
            {
                "DPC_y_std_bg": _std_in_mask(dpc_y, bg_mask),
                "DPC_y_std_cell": _std_in_mask(dpc_y, cell_mask),
                "DPC_x_std_bg": _std_in_mask(dpc_x, bg_mask),
                "DPC_x_std_cell": _std_in_mask(dpc_x, cell_mask),
                "bg_fraction_y": float(bg_mask.mean()),
                "bg_fraction_x": float(bg_mask.mean()),
            }
        )
    else:
        for k in (
            "DPC_y_std_bg",
            "DPC_y_std_cell",
            "DPC_x_std_bg",
            "DPC_x_std_cell",
            "bg_fraction_y",
            "bg_fraction_x",
        ):
            stats[k] = float("nan")

    return dpc_y, dpc_x, stats


def _save_overview(
    bf: np.ndarray, dpc_y: np.ndarray, dpc_x: np.ndarray, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(bf, cmap="gray")
    axes[0].set_title("BF (bffull)")
    # Symmetric scale for DPC visualization — most balanced DPC values lie well
    # within +/-0.2 after mean-balancing.
    vmax = float(np.percentile(np.abs(dpc_y), 99))
    vmax = max(vmax, 0.05)
    axes[1].imshow(dpc_y, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"DPC_y (balanced, ±{vmax:.2f})")
    vmax_x = float(np.percentile(np.abs(dpc_x), 99))
    vmax_x = max(vmax_x, 0.05)
    axes[2].imshow(dpc_x, cmap="RdBu_r", vmin=-vmax_x, vmax=vmax_x)
    axes[2].set_title(f"DPC_x (balanced, ±{vmax_x:.2f})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_condition(
    folder: Path, out_dir: Path, save_npy: bool = True
) -> dict[str, object]:
    """Run the full pipeline on one condition folder. Returns one-row dict."""
    imgs = load_condition(folder)
    dpc_y, dpc_x, stats = compute_dpc_balanced(
        imgs["top"], imgs["bottom"], imgs["left"], imgs["right"], bf=imgs["bffull"]
    )

    cond_out = out_dir / folder.name
    cond_out.mkdir(parents=True, exist_ok=True)
    if save_npy:
        np.save(cond_out / "dpc_y_bal.npy", dpc_y.astype(np.float32))
        np.save(cond_out / "dpc_x_bal.npy", dpc_x.astype(np.float32))
    _save_overview(imgs["bffull"], dpc_y, dpc_x, cond_out / "dpc_overview.png")

    row: dict[str, object] = {"folder": folder.name}
    row.update(stats)
    return row


def _discover_condition_folders(root: Path) -> list[Path]:
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        has_bf = any(child.glob("*bffull*.tif"))
        has_top = any(child.glob("*top*.tif"))
        if has_bf and has_top:
            out.append(child)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: python dpc_analyze.py <data_root> [<out_root>]", file=sys.stderr
        )
        return 2
    data_root = Path(argv[1]).expanduser().resolve()
    if len(argv) >= 3:
        out_root = Path(argv[2]).expanduser().resolve()
    else:
        out_root = data_root / "_dpc_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    folders = _discover_condition_folders(data_root)
    print(f"[dpc_analyze] Found {len(folders)} condition folders under {data_root}")

    rows: list[dict[str, object]] = []
    failed: list[str] = []
    for i, folder in enumerate(folders, 1):
        try:
            print(f"  [{i:2d}/{len(folders)}] {folder.name} ...", flush=True)
            row = analyze_condition(folder, out_root)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 — keep batch alive
            print(f"  WARNING: failed on {folder.name}: {exc}")
            traceback.print_exc()
            failed.append(folder.name)

    if not rows:
        print("[dpc_analyze] No conditions produced output.")
        return 1

    df = pd.DataFrame(rows)
    table_path = out_root / "dpc_master_table.csv"
    df.to_csv(table_path, index=False)
    print(f"[dpc_analyze] Wrote {table_path}  ({len(df)} rows)")
    if failed:
        print(f"[dpc_analyze] Skipped: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
