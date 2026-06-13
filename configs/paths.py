import os
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

# 5b) QPM heavy-artifact outputs root (added 2026-05-18, additive — see
# qpm-analysis/README.md "Outputs policy"). Canonical location for generated
# heavy artifacts (HTML, figures, large per-cell CSVs, .npy/.npz arrays) is
# co-located with each dataset, separate from immutable raw TIFFs:
#     /data/Project_Data/QPM/<dataset>/outputs/{calibration,dpc,phase,...}/
# Raw TIFFs at QPM_ROOT/<dataset>/ remain read-only; only the outputs/ sibling
# is writable. Existing outputs under qpm-analysis/analysis/ remain in place
# until stage-by-stage migration; this layer is opt-in for new scripts.
QPM_OUTPUTS_KINDS = (
    "calibration",
    "dpc",
    "phase",
    "segmentation",
    "tables",
    "figures",
    "reports",
    "logs",
)


def qpm_outputs(dataset_name: str) -> dict[str, Path]:
    """Return canonical output subpaths for a QPM dataset.

    Use in new QPM scripts as:
        from configs.paths import qpm_outputs
        out = qpm_outputs("20260513_Hela_p15")
        figures_dir = out["figures"]

    Does not create directories; provisioning happens at run time, not import
    time. Legacy scripts continue to write to QPM_RESULTS until migrated.
    """
    root = QPM_ROOT / dataset_name / "outputs"
    return {kind: root / kind for kind in QPM_OUTPUTS_KINDS}

# 5c) KTR (ERK-KTR + H2B timelapse) dataset registration and analysis root
# helper (referenced by ktr-analysis/README.md as
# `ktr_analysis_root(dataset_name)`). Mirrors qpm_outputs()
# shape: dataset-side heavy artifacts land here; logs / reports / session
# summaries go to RUNTIME_KTR instead, via runtime_paths(RUNTIME_KTR).
KTR_DATASET_NAME = "20260505_ERKKTR_H2B_BF_Timelapse"
KTR_RAW_DATASET = SCRATCH_ROOT / KTR_DATASET_NAME
KTR_ACTIVE_ACQUISITION = (
    KTR_RAW_DATASET / "timelapse_2026-05-05_18-12-11.466141"
)
KTR_ANALYSIS_KINDS = (
    "inspection",
    "segmentation",
    "tracking",
    "ratio",
    "qc",
    "figures",
    "tables",
    "reports",
    "logs",
)


def ktr_analysis_root(dataset_name: str) -> dict[str, Path]:
    """Return canonical dataset-side analysis subpaths for a KTR dataset.

    Use as:
        from configs.paths import ktr_analysis_root
        out = ktr_analysis_root("20260505_ERKKTR_H2B_BF_Timelapse")
        seg_dir = out["segmentation"]

    Does not create directories. Heavy artifacts (mask .npy, figures, tables,
    meta.json) live here. Logs / reports / session summaries live in
    RUNTIME_KTR (use ``runtime_paths(RUNTIME_KTR)``).
    """
    root = SCRATCH_ROOT / dataset_name / "analysis"
    return {kind: root / kind for kind in KTR_ANALYSIS_KINDS}


# 6) Runtime layer — opt-in, for provenance logs / reports / run-configs that
# should not live in the repo. Override the root with the RESEARCH_RUNTIME_ROOT
# environment variable; defaults to a repo-local ``runtime/`` directory.
# Layout: RUNTIME_ROOT/<domain>/<pipeline>/{active,logs,reports,figures,cache,sessions,configs}
RUNTIME_ROOT = Path(os.environ.get("RESEARCH_RUNTIME_ROOT", REPO_ROOT / "runtime"))
RUNTIME_FUCCI = RUNTIME_ROOT / "Voltage_CellCycle" / "fucci"
RUNTIME_KTR = RUNTIME_ROOT / "Voltage_CellCycle" / "ktr"
RUNTIME_QPM = RUNTIME_ROOT / "QPM" / "qpm"
RUNTIME_SHARED = RUNTIME_ROOT / "shared"


def runtime_paths(pipeline_root: Path) -> dict[str, Path]:
    """Return the canonical 7 runtime subpaths for a pipeline namespace.

    Use in new scripts as: paths = runtime_paths(RUNTIME_FUCCI).
    Does not create directories; the structure is provisioned by runtime
    setup, not by per-script side effects.
    """
    return {
        "active": pipeline_root / "active",
        "logs": pipeline_root / "logs",
        "reports": pipeline_root / "reports",
        "figures": pipeline_root / "figures",
        "cache": pipeline_root / "cache",
        "sessions": pipeline_root / "sessions",
        "configs": pipeline_root / "configs",
    }
