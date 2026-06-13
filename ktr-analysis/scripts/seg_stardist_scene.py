"""Stage 03 production per-scene fresh-StarDist runner (H2B / mTagBFP2).

Wraps the validated Stage-02 segmenter (src.segmentation.stardist_validate) for
an arbitrary (region, fov), writing masks to the FULL tree:
    <analysis>/segmentation/full/{region}_fov{fov}/stardist_h2b/t####.npy

Mask-source policy (resolved by data 2026-05-21): tracking division recall
collapses on Bill's nuc_labels (0.30) vs fresh StarDist (0.80), so the validated
production substrate is a fresh StarDist rerun — this is its per-scene runner.

MUST run in the StarDist env (GPU, TF cu12 activate.d hook required):
    source /opt/miniconda/etc/profile.d/conda.sh && conda activate ktr-segtrack
    python scripts/seg_stardist_scene.py --region R0 --fov 1

Restartable: skips a scene whose masks_dir has a _SUCCESS marker (unless --force).
Generates a per-scene run-config YAML under RUNTIME/configs for provenance, then
calls stardist_validate.run() in-process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# repo root on path so `import src...` works under `python scripts/x.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation.io_runtime import (  # noqa: E402
    RUNTIME_KTR,
    load_project_yaml,
    resolve_dataset_paths,
)

MODEL = "2D_versatile_fluo"
INPUT_CHANNEL = "mTagBFP2"
MATRIX_CELL = "stardist_h2b"
EXPECTED_FRAMES = 567  # 0..566; reconciled in Stage 01 (registry says 569 / metadata 576)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--stage", choices=["full", "validation"], default="full")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project = load_project_yaml()
    dataset = project["datasets"]["active"]
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_ktr_stardist-h2b-{args.region}_fov{args.fov}"

    cfg = {"dataset": dataset}
    dpaths = resolve_dataset_paths(project, cfg)
    analysis = dpaths["analysis_root"]
    masks_dir = analysis / "segmentation" / args.stage / f"{args.region}_fov{args.fov}" / MATRIX_CELL
    success_marker = masks_dir / "_SUCCESS"
    if success_marker.exists() and not args.force:
        print(f"[seg-scene] SKIP {args.region}/fov{args.fov}: _SUCCESS present ({success_marker})")
        return 0

    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "pipeline": "ktr",
        "operation": "segmentation",
        "domain": "Voltage_CellCycle",
        "dataset": dataset,
        "operator": "heeseok",
        "agent": "claude-code",
        "subset": {"region": args.region, "fov": args.fov, "timepoints": "all"},
        "matrix_cell": MATRIX_CELL,
        "overrides": {
            "pipeline.segmenter.name": "stardist",
            "pipeline.segmenter.model": MODEL,
            "pipeline.segmenter.input_channel": INPUT_CHANNEL,
        },
        "output_paths": {
            "masks_dir": str(masks_dir),
            "log": f"logs/{run_id}__segmentation.jsonl",
        },
        "intent": f"Stage 03 production: fresh StarDist rerun for tracking, {args.region}/fov{args.fov}",
    }
    cfg_dir = RUNTIME_KTR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{run_id}.yaml"
    cfg_path.write_text(yaml.safe_dump(run_config, sort_keys=False))
    print(f"[seg-scene] {args.region}/fov{args.fov} -> {masks_dir}\n[seg-scene] config: {cfg_path}")

    from src.segmentation.stardist_validate import run  # lazy: pulls TF

    t0 = time.time()
    rc = run(str(cfg_path))
    dt = time.time() - t0
    if rc != 0:
        print(f"[seg-scene] FAILED rc={rc} after {dt:.1f}s")
        return rc

    n = len(list(masks_dir.glob("t*.npy")))
    success_marker.write_text(json.dumps({
        "region": args.region, "fov": args.fov, "n_masks": n,
        "expected": EXPECTED_FRAMES, "runtime_sec": round(dt, 1),
        "sec_per_frame": round(dt / max(n, 1), 3),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2))
    print(f"[seg-scene] {args.region}/fov{args.fov} done: {n} masks in {dt:.1f}s "
          f"({dt / max(n, 1):.2f}s/f)")
    if n != EXPECTED_FRAMES:
        print(f"[seg-scene] WARN n_masks={n} != expected {EXPECTED_FRAMES} (check coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
