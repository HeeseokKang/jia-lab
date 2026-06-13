#!/usr/bin/env python3
"""Save :func:`src.io.show_fov_qc` to ``figures/qc/*.png`` for IDE / diff review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import parse_dataset, show_fov_qc  # noqa: E402


def _parse_timepoints(s: str | None) -> list[int] | None:
    if not s or not s.strip():
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_ROOT / "data" / "raw",
        help="Dataset root (symlink ok). Default: <repo>/data/raw",
    )
    p.add_argument("--region", default="R0", help="Region label, e.g. R0")
    p.add_argument("--fov", type=int, default=0, help="FOV index")
    p.add_argument(
        "--timepoints",
        type=str,
        default="",
        help="Comma-separated frame indices, e.g. 0,12,24. Empty = default triplet.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: figures/qc/<region>_fov<n>_qc.png",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--figsize",
        type=str,
        default="14,8",
        help='Matplotlib figsize as "W,H" in inches',
    )
    args = p.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")

    fs_parts = [float(x.strip()) for x in args.figsize.split(",")]
    if len(fs_parts) != 2:
        raise SystemExit('--figsize must be "W,H", e.g. 14,8')
    figsize = (fs_parts[0], fs_parts[1])

    out = args.output
    if out is None:
        out_dir = _REPO_ROOT / "figures" / "qc"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_region = args.region.replace("/", "_")
        out = out_dir / f"{safe_region}_fov{args.fov}_qc.png"
    else:
        out = Path(out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

    tps = _parse_timepoints(args.timepoints)
    df = parse_dataset(data_root)
    if df.empty:
        raise SystemExit(f"No KTR TIFFs found under {data_root}")

    fig = show_fov_qc(df, region=args.region, fov=args.fov, timepoints=tps, figsize=figsize)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
