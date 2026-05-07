from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


DATA_DIR = Path(
    "/data/Project_Data/Voltage_CellCycle/20260413_FUCCI_Timelapse/"
    "timelapse_plus_bf_2026-04-16_12-11-45.768458/0"
)
WELL_PREFIX = "R0_0_0_3T3_FUCCI_"
CHANNELS = ("BF", "561", "647")


def image_stats(img: np.ndarray) -> dict[str, float]:
    return {
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
    }


def quadrant_means(img: np.ndarray) -> dict[str, float]:
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    return {
        "TL": float(img[:h2, :w2].mean()),
        "TR": float(img[:h2, w2:].mean()),
        "BL": float(img[h2:, :w2].mean()),
        "BR": float(img[h2:, w2:].mean()),
    }


def gradient_percent(q: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(q.values()), dtype=float)
    q_mean = float(vals.mean())
    max_min_pct = float((vals.max() - vals.min()) / q_mean * 100.0) if q_mean else 0.0

    top = (q["TL"] + q["TR"]) / 2.0
    bottom = (q["BL"] + q["BR"]) / 2.0
    left = (q["TL"] + q["BL"]) / 2.0
    right = (q["TR"] + q["BR"]) / 2.0

    tb_pct = float((bottom - top) / q_mean * 100.0) if q_mean else 0.0
    lr_pct = float((right - left) / q_mean * 100.0) if q_mean else 0.0

    return {
        "max_min_pct": max_min_pct,
        "top_bottom_pct": tb_pct,
        "left_right_pct": lr_pct,
    }


def main() -> None:
    channel_images: dict[str, np.ndarray] = {}

    for ch in CHANNELS:
        path = DATA_DIR / f"{WELL_PREFIX}{ch}.tiff"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        img = tifffile.imread(path)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}: {path}")
        channel_images[ch] = img

    print("[A1.4] t=0, well=R0_0 channel intensity stats")
    for ch in CHANNELS:
        s = image_stats(channel_images[ch])
        print(
            f"  {ch}: min={s['min']:.0f}, max={s['max']:.0f}, "
            f"mean={s['mean']:.3f}, std={s['std']:.3f}"
        )

    q647 = quadrant_means(channel_images["647"])
    g647 = gradient_percent(q647)
    print("\n[A1.4] 647 quadrant means")
    print(
        "  "
        f"TL={q647['TL']:.3f}, TR={q647['TR']:.3f}, "
        f"BL={q647['BL']:.3f}, BR={q647['BR']:.3f}"
    )
    print("[A1.4] 647 gradient (%)")
    print(
        "  "
        f"max-min/mean={g647['max_min_pct']:.3f}%, "
        f"bottom-top={g647['top_bottom_pct']:.3f}%, "
        f"right-left={g647['left_right_pct']:.3f}%"
    )


if __name__ == "__main__":
    main()
