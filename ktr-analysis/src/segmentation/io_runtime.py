"""Stage 02 runtime plumbing: configs, paths, JSONL logger.

Lives next to the segmenter modules so that `python -m
src.segmentation.stardist_validate --config <yaml>` can load the per-run
config, resolve dataset paths, and emit a JSONL log conforming to
shared/logging_convention.md without any extra wiring.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

# Lab-wide paths come from the configs package at the monorepo root.
# tests/test_io.py already prepends the repo root to sys.path; segmenter
# modules will rely on the standard `python -m src...` invocation from the
# ktr-analysis root, with `configs/` resolvable via the parent jia-lab
# directory at runtime.
# Runtime root for provenance logs / run-configs. Override with the
# RESEARCH_RUNTIME_ROOT environment variable; defaults to a repo-local
# ``runtime/`` directory (jia-lab/runtime).
_RUNTIME_ROOT = Path(
    os.environ.get(
        "RESEARCH_RUNTIME_ROOT", Path(__file__).resolve().parents[3] / "runtime"
    )
)
RUNTIME_KTR = _RUNTIME_ROOT / "Voltage_CellCycle" / "ktr"
PROJECT_YAML_DEFAULT = RUNTIME_KTR / "configs" / "project.yaml"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def host_tag() -> str:
    return f"{getpass.getuser()}@{socket.gethostname()}"


def code_commit(repo: Path | None = None) -> str | None:
    """Return short git SHA of the ktr-analysis repo, or None if unavailable."""
    repo = repo or Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode == 0:
        return out.stdout.strip() or None
    return None


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_run_config(config_path: Path | str) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(Path(config_path).resolve())
    return cfg


def load_project_yaml(path: Path | str = PROJECT_YAML_DEFAULT) -> dict[str, Any]:
    return load_yaml(path)


def resolve_dataset_paths(project: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve the input acquisition dir and dataset analysis root for a run."""
    raw_data_root = Path(project["paths"]["raw_data_root"])
    dataset_name = run_config["dataset"]
    raw_dataset = raw_data_root / dataset_name
    active = project["datasets"]["registry"]
    if isinstance(active, list):
        entry = next(d for d in active if d["name"] == dataset_name)
    else:
        entry = active
    acquisition_dir = raw_dataset / entry["acquisition_subdir"]
    analysis_root = raw_dataset / "analysis"
    return {
        "raw_dataset": raw_dataset,
        "acquisition_dir": acquisition_dir,
        "analysis_root": analysis_root,
    }


class JsonlLogger:
    """Append-only JSONL writer following shared/logging_convention.md.

    One header line, many step/warning/error lines, exactly one summary line.
    The summary writer flushes and closes; do not write after calling it.
    """

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.log_path, "w", buffering=1)
        self._n_steps = 0
        self._n_warnings = 0
        self._n_errors = 0
        self._closed = False

    def write(self, event: Mapping[str, Any]) -> None:
        self._fp.write(json.dumps(event, default=str))
        self._fp.write("\n")

    def header(self, **fields: Any) -> None:
        evt = {"event": "header", "schema": 1, **fields}
        self.write(evt)

    def step(self, **fields: Any) -> None:
        evt = {"event": "step", "ts": utc_now_iso(), **fields}
        self.write(evt)
        self._n_steps += 1

    def warning(self, message: str, **fields: Any) -> None:
        evt = {"event": "warning", "ts": utc_now_iso(), "message": message, **fields}
        self.write(evt)
        self._n_warnings += 1

    def error(self, message: str, **fields: Any) -> None:
        evt = {"event": "error", "ts": utc_now_iso(), "message": message, **fields}
        self.write(evt)
        self._n_errors += 1

    def summary(self, status: str, **fields: Any) -> None:
        evt = {
            "event": "summary",
            "ts": utc_now_iso(),
            "status": status,
            "n_steps": self._n_steps,
            "warnings": self._n_warnings,
            "errors": self._n_errors,
            **fields,
        }
        self.write(evt)
        self._fp.flush()
        self._fp.close()
        self._closed = True

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            status = "failed" if exc_type is not None else "partial"
            self.summary(status=status, abnormal_close=True)


def runtime_log_path(run_id: str, operation: str) -> Path:
    return RUNTIME_KTR / "logs" / f"{run_id}__{operation}.jsonl"


def dataset_log_mirror_path(analysis_root: Path, run_id: str, operation: str) -> Path:
    return analysis_root / "logs" / f"{run_id}__{operation}.jsonl"


def write_mirror(src: Path, dst: Path) -> None:
    """Mirror the runtime log into the dataset analysis dir (Q2 dual-write)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
