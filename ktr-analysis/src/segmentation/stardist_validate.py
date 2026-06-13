"""Stage 02 matrix cell: fresh StarDist on one channel of R0/fov0.

CLI:
    python -m src.segmentation.stardist_validate --config <run_config.yaml>

Mask outputs land in <dataset>/analysis/segmentation/validation/R0_fov0/<cell>/
JSONL log lands in RUNTIME_KTR/logs/<run_id>__segmentation.jsonl and is
mirrored into <dataset>/analysis/logs/ per Q2 dual-write.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile

from src.io import parse_dataset, get_fov_timeseries
from src.segmentation.io_runtime import (
    JsonlLogger,
    code_commit,
    dataset_log_mirror_path,
    host_tag,
    load_project_yaml,
    load_run_config,
    resolve_dataset_paths,
    runtime_log_path,
    utc_now_iso,
)


def _normalize_for_stardist(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def run(config_path: str) -> int:
    project = load_project_yaml()
    cfg = load_run_config(config_path)

    run_id = cfg["run_id"]
    matrix_cell = cfg["matrix_cell"]
    overrides = cfg.get("overrides", {})
    input_channel = overrides.get("pipeline.segmenter.input_channel", "mTagBFP2")
    model_name = overrides.get("pipeline.segmenter.model", "2D_versatile_fluo")
    region = cfg["subset"]["region"]
    fov = int(cfg["subset"]["fov"])
    timepoints = cfg["subset"]["timepoints"]

    dataset_paths = resolve_dataset_paths(project, cfg)
    masks_dir = Path(cfg["output_paths"]["masks_dir"])
    masks_dir.mkdir(parents=True, exist_ok=True)

    log_path = runtime_log_path(run_id, "segmentation")
    mirror_path = dataset_log_mirror_path(dataset_paths["analysis_root"], run_id, "segmentation")

    # Resolve input TIFFs for this region/fov/channel.
    df = parse_dataset(dataset_paths["acquisition_dir"])
    series = get_fov_timeseries(df, region=region, fov=fov)
    paths = series[input_channel]
    if isinstance(timepoints, list):
        wanted = [(tp, paths[tp]) for tp in timepoints]
    else:
        wanted = list(enumerate(paths))

    # Lazy heavy imports.
    from stardist.models import StarDist2D

    started_at = utc_now_iso()
    with JsonlLogger(log_path) as log:
        log.header(
            run_id=run_id,
            pipeline="ktr",
            operation="segmentation",
            dataset=cfg["dataset"],
            subset={"region": region, "fov": fov, "timepoints": timepoints},
            matrix_cell=matrix_cell,
            segmenter={"name": "stardist", "model": model_name},
            input_channel=input_channel,
            started_at=started_at,
            agent=cfg.get("agent", "claude-code"),
            host=host_tag(),
            code_commit=code_commit(),
            config_path=cfg["_config_path"],
        )

        t0 = time.time()
        model = StarDist2D.from_pretrained(model_name)
        n_objects_total = 0

        for tp, tiff_path in wanted:
            f0 = time.time()
            img = tifffile.imread(tiff_path)
            norm = _normalize_for_stardist(img)
            labels, _ = model.predict_instances(norm, verbose=False)
            mask_path = masks_dir / f"t{int(tp):04d}.npy"
            np.save(mask_path, labels.astype(np.uint32))
            n_obj = int(np.unique(labels[labels != 0]).size)
            n_objects_total += n_obj

            log.step(
                step="frame",
                frame_index=int(tp),
                tiff_path=tiff_path,
                mask_path=str(mask_path),
                n_objects=n_obj,
                mean_diameter_px=_mean_diameter_px(labels),
                frame_seconds=round(time.time() - f0, 3),
                h2b_p10=float(np.percentile(img, 10)),
            )

        log.summary(
            status="ok",
            run_id=run_id,
            ended_at=utc_now_iso(),
            duration_sec=round(time.time() - t0, 3),
            n_frames=len(wanted),
            n_objects_total=n_objects_total,
            output_paths={"masks_dir": str(masks_dir)},
            qc={
                "mean_objects_per_frame": (n_objects_total / max(len(wanted), 1)),
                "frames_with_zero_objects": 0,  # populated by validator later
            },
        )

    # Q2 dual-write: mirror runtime log into dataset-side analysis dir.
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_path.write_bytes(log_path.read_bytes())
    return 0


def _mean_diameter_px(labels: np.ndarray) -> float:
    from skimage import measure

    props = measure.regionprops(labels)
    if not props:
        return float("nan")
    return float(np.mean([p.equivalent_diameter_area for p in props]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to run_config.yaml")
    args = p.parse_args()
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
