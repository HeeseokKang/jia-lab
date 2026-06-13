"""Stage 0 -- build and validate the snapshot manifest for the 20260610 run.

Parses filenames `2026-06-10_HH-MM-SS_<construct>_<well>_<channel>_<int>pct.tif`
into one row per image: construct, well, row, col, drug, replicate, channel,
role (green/red/bf), exposure, acquisition time, path. Applies the D900
benchmark-time filter (early titration excluded from the analysis manifest) and
checks 36-well x 3-channel completeness.

Writes `<analysis>/manifest.csv` and prints a completeness report.

Run:  conda activate fucci-analysis
      python fucci-analysis/src/variant_index.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import variant_config as cfg  # noqa: E402  (same src/ dir)

FNAME_RE = re.compile(
    r"^2026-06-10_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<construct>D900|pBJ281|pBJ284|pBJ285)_"
    r"(?P<row>[A-G])(?P<col>\d+)_"
    r"(?P<channel>BF|488|561|635|405)_"
    r"(?P<pct>\d+)pct\.tif$"
)


def role_for(construct: str, channel: str) -> str:
    if channel == "BF":
        return "bf"
    c = cfg.CONSTRUCTS[construct]
    if channel == c["green"]:
        return "green"
    if channel == c["red"]:
        return "red"
    return "other"


def build_manifest() -> pd.DataFrame:
    rows: list[dict] = []
    for tif in sorted(cfg.DATASET_DIR.glob("2026-06-10_*pct.tif")):
        m = FNAME_RE.match(tif.name)
        if not m:
            rows.append({"path": str(tif), "parse_ok": False})
            continue
        construct = m.group("construct")
        time = m.group("time")
        col = int(m.group("col"))
        row = m.group("row")

        # D900: keep only the consistent benchmark set; tag the early titration.
        d900_set = None
        if construct == "D900":
            d900_set = "benchmark" if time >= cfg.D900_BENCHMARK_MIN_HHMMSS else "titration"

        channel = m.group("channel")
        rows.append({
            "construct": construct,
            "well": f"{row}{col}",
            "row": row,
            "col": col,
            "drug": cfg.DRUG_BY_COL.get(col),
            "replicate": (cfg.replicate_index(construct, row)
                          if row in cfg.CONSTRUCTS[construct]["rows"] else None),
            "channel": channel,
            "role": role_for(construct, channel),
            "exposure_ms": cfg.EXPOSURE_MS.get(channel),
            "time": time,
            "d900_set": d900_set,
            "pct": int(m.group("pct")),
            "path": str(tif),
            "parse_ok": True,
        })
    return pd.DataFrame(rows)


def analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that feed the analysis: all variants + D900 benchmark set only."""
    ok = df["parse_ok"] & df["role"].isin(["bf", "green", "red"])
    keep = ok & ((df["construct"] != "D900") | (df["d900_set"] == "benchmark"))
    return df[keep].copy()


def report(df: pd.DataFrame) -> None:
    ana = analysis_subset(df)
    print(f"[INDEX] total tif parsed = {int(df['parse_ok'].sum())} / {len(df)}")
    bad = df[~df.get("parse_ok", False)]
    if len(bad):
        print(f"[WARN] {len(bad)} unparsed files (check naming):")
        for p in bad["path"].head(10):
            print(f"        {p}")

    # per-construct well x role completeness
    print("\n[COMPLETENESS] wells with bf+green+red present per construct")
    for name in cfg.CONSTRUCTS:
        sub = ana[ana["construct"] == name]
        wells = sorted(sub["well"].unique())
        roles_by_well = sub.groupby("well")["role"].agg(set)
        complete = [w for w in wells if {"bf", "green", "red"} <= roles_by_well[w]]
        missing = [w for w in wells if w not in complete]
        exp = [f"{r}{c}" for r in cfg.CONSTRUCTS[name]["rows"]
               for c in cfg.CONSTRUCTS[name]["cols"]]
        not_found = sorted(set(exp) - set(wells))
        flag = "OK" if len(complete) == 9 and not not_found else "CHECK"
        print(f"  {name:7s} complete={len(complete)}/9  "
              f"incomplete={missing}  missing_wells={not_found}  [{flag}]")

    # drug x replicate grid sanity
    print("\n[DESIGN] construct x drug x replicate (count of complete wells)")
    grid = (ana[ana["role"] == "bf"]
            .groupby(["construct", "drug", "replicate"]).size()
            .rename("n").reset_index())
    print(grid.to_string(index=False))

    # D900 titration note
    tit = df[(df["construct"] == "D900") & (df["d900_set"] == "titration")]
    print(f"\n[NOTE] D900 titration snapshots excluded from analysis: {len(tit)} "
          f"files (11:04-11:30 exposure/intensity sweep, acquisition note only)")


def main() -> None:
    cfg.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_manifest()
    out = cfg.ANALYSIS_DIR / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"[SAVED] {out}  rows={len(df)}\n")
    report(df)

    ana_out = cfg.ANALYSIS_DIR / "manifest_analysis.csv"
    analysis_subset(df).to_csv(ana_out, index=False)
    print(f"\n[SAVED] {ana_out}  (analysis subset: variants + D900 benchmark)")


if __name__ == "__main__":
    main()
