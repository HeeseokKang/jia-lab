from pathlib import Path
import sys

import matplotlib.pyplot as plt
from skimage import exposure, io


# Add repository root to import path so shared/config modules resolve consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import DATASET_1_RAW, RESULT_ROOT  # noqa: E402
from shared.utils import get_clean_file_list  # noqa: E402


CHANNELS = ("BF", "561", "647")
FRAME_NUMBER = 100


def main() -> None:
    data_root = Path(DATASET_1_RAW)
    if not data_root.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {data_root}")

    clean = get_clean_file_list(data_root)
    frame_idx = FRAME_NUMBER - 1

    for channel in CHANNELS:
        if len(clean[channel]) <= frame_idx:
            raise ValueError(
                f"Not enough cleaned frames for channel {channel}. "
                f"Found {len(clean[channel])}, requested frame {FRAME_NUMBER}."
            )

    bf_img = io.imread(str(clean["BF"][frame_idx]))
    c561_img = io.imread(str(clean["561"][frame_idx]))
    c647_img = io.imread(str(clean["647"][frame_idx]))

    # Increase 647 contrast to expose background non-uniformity/noise more clearly.
    c647_enhanced = exposure.equalize_adapthist(c647_img, clip_limit=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(bf_img, cmap="gray")
    axes[0].set_title(f"BF (Frame {FRAME_NUMBER})")
    axes[1].imshow(c561_img, cmap="magma")
    axes[1].set_title(f"561 (Frame {FRAME_NUMBER})")
    axes[2].imshow(c647_enhanced, cmap="inferno")
    axes[2].set_title(f"647 Enhanced (Frame {FRAME_NUMBER})")

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Visual QC - Dataset 1", fontsize=14)
    fig.tight_layout()

    output_dir = RESULT_ROOT / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qc_sample_frame_100.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[QC] Saved visual QC figure: {output_path}")
    print(f"[QC] BF frame source: {clean['BF'][frame_idx]}")
    print(f"[QC] 561 frame source: {clean['561'][frame_idx]}")
    print(f"[QC] 647 frame source: {clean['647'][frame_idx]}")


if __name__ == "__main__":
    main()
