# 파일: fucci-analysis/src/inspect_background.py
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_root = Path("/mnt/nas1/Projects/Voltage_CellCycle/Data/20260413_FUCCI_Timelapse/timelapse_plus_bf_2026-04-16_12-11-45.768458")

# 647 frame 100 로드
f647 = sorted(data_root.rglob("*FUCCI_647.tiff"))[99]
img = tifffile.imread(str(f647))

print(f"Shape: {img.shape}, dtype: {img.dtype}")
print(f"Min: {img.min()}, Max: {img.max()}")
print(f"Mean: {img.mean():.1f}, Std: {img.std():.1f}")

# 이미지를 4분할해서 각 구역 평균 intensity 비교
h, w = img.shape
q = {
    "top-left":     img[:h//2, :w//2].mean(),
    "top-right":    img[:h//2, w//2:].mean(),
    "bottom-left":  img[h//2:, :w//2].mean(),
    "bottom-right": img[h//2:, w//2:].mean(),
}
print("\n=== Quadrant mean intensities ===")
for k, v in q.items():
    print(f"  {k}: {v:.1f}")