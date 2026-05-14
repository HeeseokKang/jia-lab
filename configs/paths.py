from pathlib import Path

# 1) System-level root paths from the lab storage plan.
NAS_ROOT = Path("/mnt/nas1/Projects/Voltage_CellCycle/Data")
SCRATCH_ROOT = Path("/data/Project_Data/Voltage_CellCycle")

# 2) Dataset 1 canonical raw-data location.
DATASET_1_NAME = "20260413_FUCCI_Timelapse"
# Include the exact acquisition subfolder containing source images.
DATASET_1_RAW = NAS_ROOT / DATASET_1_NAME / "timelapse_plus_bf_2026-04-16_12-11-45.768458"

# 3) Dataset 2 registration for upcoming drug-screen analysis.
DATASET_2_NAME = "20260420_FUCCI_TL_Drug"
DATASET_2_RAW = NAS_ROOT / DATASET_2_NAME  # Confirm this mount point against actual NAS layout.

# 4) Analysis output directory.
# Large intermediate outputs may be redirected to scratch storage if needed.
REPO_ROOT = Path(__file__).parent.parent
RESULT_ROOT = REPO_ROOT / "fucci-analysis" / "analysis" / "20260413_validation"

# Ensure the output folder exists before downstream scripts run.
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

print(f"[CONFIG] Loaded paths. Primary dataset: {DATASET_1_RAW}")

# 5) QPM (quantitative phase microscopy) dataset registration.
QPM_ROOT = Path("/data/Project_Data/QPM")
QPM_20260513 = QPM_ROOT / "20260513_Hela_p15"
QPM_RESULTS = Path(__file__).parent.parent / "qpm-analysis" / "analysis" / "20260513_Hela_p15"
