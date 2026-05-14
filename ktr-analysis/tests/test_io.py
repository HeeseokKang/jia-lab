from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Running as `python tests/test_io.py` puts `tests/` on sys.path first; add repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.io import parse_dataset  # noqa: E402

RAW_ROOT = (_REPO_ROOT / "data" / "raw").resolve()


def main() -> None:
    print(f"[INFO] RAW_ROOT = {RAW_ROOT}")
    assert RAW_ROOT.exists(), f"Missing raw root: {RAW_ROOT}"
    assert RAW_ROOT.is_dir(), f"RAW_ROOT is not a directory: {RAW_ROOT}"

    # 1) Parse dataset
    df = parse_dataset(RAW_ROOT)
    print(f"[INFO] parsed rows = {len(df):,}")

    expected_cols = ["timepoint", "region", "fov", "z", "channel", "filepath"]
    assert list(df.columns) == expected_cols, f"Unexpected columns: {list(df.columns)}"
    assert not df.empty, "parse_dataset returned an empty DataFrame"

    # 2) Basic schema checks
    assert df["timepoint"].dtype.kind in "iu", (
        f"timepoint dtype should be integer-like, got {df['timepoint'].dtype}"
    )
    assert df["fov"].dtype.kind in "iu", (
        f"fov dtype should be integer-like, got {df['fov'].dtype}"
    )
    assert df["z"].dtype.kind in "iu", f"z dtype should be integer-like, got {df['z'].dtype}"
    assert df["filepath"].map(lambda x: Path(x).is_absolute()).all(), (
        "filepath must be absolute paths"
    )

    # 3) Expected categories
    regions = set(df["region"].unique())
    channels = set(df["channel"].unique())
    print("[INFO] regions:", sorted(regions))
    print("[INFO] channels:", sorted(channels))
    assert regions == {"R0", "R1"}, f"Unexpected regions: {regions}"
    assert channels == {"BF", "mKate2", "mTagBFP2"}, f"Unexpected channels: {channels}"

    # 4) Timepoint folder integrity
    tp_dirs = sorted(
        [p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    timepoints = [int(p.name) for p in tp_dirs]
    print(f"[INFO] numeric timepoint folders = {len(timepoints)}")
    print(f"[INFO] first 5 timepoints = {timepoints[:5]}")
    print(f"[INFO] last 5 timepoints = {timepoints[-5:]}")
    assert timepoints[0] == 0, f"First timepoint should be 0, got {timepoints[0]}"
    assert timepoints == list(range(timepoints[-1] + 1)), (
        "Timepoint folders are not contiguous from 0..N"
    )

    # 5) Compare metadata.json to actual folders
    meta_path = tp_dirs[0] / "metadata.json"
    assert meta_path.exists(), f"Missing metadata.json: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta_num_tp = meta.get("num_time_points")
    print(f"[INFO] metadata num_time_points = {meta_num_tp}")
    if meta_num_tp is not None and meta_num_tp != len(tp_dirs):
        print(
            f"[WARN] metadata says {meta_num_tp} timepoints, "
            f"but actual folders = {len(tp_dirs)}"
        )

    # 6) Per-timepoint file count checks
    per_tp_counts = df.groupby("timepoint").size()
    assert (per_tp_counts == 96).all(), (
        f"Each timepoint should have 96 TIFFs, got counts:\n{per_tp_counts.value_counts()}"
    )
    per_tp_channel = df.groupby(["timepoint", "channel"]).size().unstack(fill_value=0)
    assert (per_tp_channel == 32).all().all(), (
        "Each timepoint should have 32 TIFFs per channel (2 regions x 16 fovs)"
    )

    # 7) Per-(timepoint, region, fov, z, channel) uniqueness
    dupes = df.duplicated(subset=["timepoint", "region", "fov", "z", "channel"])
    assert not dupes.any(), f"Found duplicated index rows:\n{df.loc[dupes].head()}"

    # 8) Sanity check on one known example
    sample = df[
        (df["timepoint"] == 0)
        & (df["region"] == "R0")
        & (df["fov"] == 0)
        & (df["z"] == 0)
    ].sort_values(["channel"])
    print("[INFO] sample rows at timepoint 0, R0, fov 0:")
    print(sample.to_string(index=False))
    assert set(sample["channel"]) == {"BF", "mKate2", "mTagBFP2"}, "Sample triplet missing"

    print("\n[PASS] parse_dataset looks consistent.")


if __name__ == "__main__":
    main()
