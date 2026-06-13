# QC figures (local)

PNG files in this folder are generated for visual review in the IDE (e.g. Cursor) without VNC.

Example:

```bash
cd ~/github/jia-lab/ktr-analysis
conda activate ktr-analysis
python scripts/render_fov_qc.py --region R0 --fov 0
python scripts/render_fov_qc.py --region R0 --fov 0 --timepoints 0,10,20,30
```

Outputs default to `figures/qc/<region>_fov<n>_qc.png`. These PNGs are gitignored so large batches do not clutter the repo; keep them on disk for your session.
