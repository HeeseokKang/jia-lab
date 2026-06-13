"""Fine-tune StarDist (2D_versatile_fluo) on human-corrected H2B nuclei.

Transfer-learn from the pretrained 2D_versatile_fluo model onto the
seg-correction pairs produced by segnapari (Phase-1 stratified-20,
R0/fov0). Goal = raise nuclear recall (false-negative recovery) so the
downstream Ultrack lineage is less selection-biased. See
project_ktr_labeling_strategy: this is the "Improve" half of the
Measure -> Improve -> Re-measure loop.

Self-contained (own argparse + data loading) so it runs unattended in tmux.

Run (ktr-segtrack env):
    python -m src.segmentation.stardist_finetune \
        --pairs-dir <.../seg_corrections/R0_fov0> \
        --out-dir   <.../analysis/segmentation/models/stardist_ft_YYYYMMDD>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


def _load_pairs(pairs_dir: Path):
    img_dir, msk_dir = pairs_dir / "images", pairs_dir / "masks"
    img_files = sorted(img_dir.glob("*.tif"))
    pairs = []
    for f in img_files:
        m = msk_dir / f.name
        if not m.exists():
            raise FileNotFoundError(f"no mask for {f.name}")
        pairs.append((f, m))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs under {pairs_dir}")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--base-model", default="2D_versatile_fluo")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy imports (TF/StarDist) deferred so --help is cheap.
    from csbdeep.utils import normalize
    from stardist import fill_label_holes
    from stardist.models import Config2D, StarDist2D
    import copy

    pairs = _load_pairs(args.pairs_dir)
    print(f"[data] {len(pairs)} pairs from {args.pairs_dir}", flush=True)

    X = [normalize(tifffile.imread(str(f)).astype(np.float32), 1.0, 99.8) for f, _ in pairs]
    Y = [fill_label_holes(tifffile.imread(str(m)).astype(np.int32)) for _, m in pairs]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    n_val = min(args.n_val, max(1, len(X) // 5))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]
    Xt, Yt = [X[i] for i in trn_idx], [Y[i] for i in trn_idx]
    Xv, Yv = [X[i] for i in val_idx], [Y[i] for i in val_idx]
    print(f"[split] train={len(Xt)} val={len(Xv)} "
          f"(val frames: {[pairs[i][0].name for i in val_idx]})", flush=True)

    # Load pretrained, clone its architecture exactly, copy weights -> fine-tune.
    pre = StarDist2D.from_pretrained(args.base_model)
    conf = copy.deepcopy(pre.config)
    conf.train_epochs = args.epochs
    conf.train_steps_per_epoch = args.steps_per_epoch
    conf.train_batch_size = args.batch_size
    conf.train_patch_size = (args.patch, args.patch)
    conf.train_reduce_lr = {"factor": 0.5, "patience": 40, "min_delta": 0}
    conf.train_tensorboard = False  # tensorboard not installed in ktr-segtrack; avoid TBNotInstalledError

    model = StarDist2D(conf, name=args.out_dir.name, basedir=str(args.out_dir.parent))
    model.keras_model.set_weights(pre.keras_model.get_weights())
    print(f"[model] cloned {args.base_model} -> {args.out_dir} "
          f"(n_rays={conf.n_rays}, grid={conf.grid})", flush=True)

    # Standard flip/rotate + mild intensity augmentation.
    def augmenter(x, y):
        axes = tuple(range(x.ndim))
        if rng.random() < 0.5:
            ax = int(rng.integers(x.ndim)); x = np.flip(x, ax); y = np.flip(y, ax)
        k = int(rng.integers(4)); x = np.rot90(x, k); y = np.rot90(y, k)
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
        return x, y

    model.train(Xt, Yt, validation_data=(Xv, Yv), augmenter=augmenter)
    print("[train] done; optimizing thresholds on val ...", flush=True)
    model.optimize_thresholds(Xv, Yv)

    elapsed = time.time() - t0
    summary = {
        "base_model": args.base_model,
        "n_pairs": len(pairs),
        "n_train": len(Xt),
        "n_val": len(Xv),
        "val_frames": [pairs[i][0].name for i in val_idx],
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size,
        "patch": args.patch,
        "thresholds": dict(model.thresholds._asdict()),
        "model_dir": str(args.out_dir),
        "elapsed_sec": round(elapsed, 1),
        "status": "ok",
    }
    (args.out_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] {elapsed/60:.1f} min -> {args.out_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
