"""Stage 02 metric kit.

v2 (2026-05-18): eight GT-quality metrics, one cell-table row per
(matrix_cell, frame). Functions take numpy label arrays directly so they can
be reused for both per-frame mask .npy files and per-scene zarr slices.

v3 (2026-05-20): metric-family split (ADDITIVE). Two families now:
  * GT-quality family — H2B rows, nucleus-vs-nucleus: the original eight
    metrics, orchestrated by ``compute_cell_metrics`` (behaviour unchanged).
  * Containment family — BF rows, whole-cell-vs-DAPI-nuclei: NEW functions
    (``dapi_coverage``, ``nuclei_per_cell_histogram``, ``assignment_uniqueness``,
    ``orphan_bf_rate``, optional ``bf_vs_cytoring_iou``) orchestrated by
    ``compute_bf_containment_metrics``. Route with ``compute_stage02_metrics``.
Nucleus-IoU / nucleus-boundary metrics on a BF *whole-cell* mask are a metric
category error (STAGE02_CONTRACT.md §B, decision S0), so BF rows do NOT use
IoU / Hausdorff / merge-split; they use containment/coverage instead.

NOTE (docs-first alignment): this module implements the v3 metric layer only.
The ``compare_masks.py`` orchestration swap (to a heterogeneous per-family
table, which would also change the smoke shape) is a separate, still-deferred
edit — NOT done here, so the existing smoke contract is unchanged.

See STAGE02_CONTRACT.md §4 (v2) and §B–§F (v3) for the full definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage import measure


# ----------------------------------------------------------------------------
# 4.1  IoU distribution
# ----------------------------------------------------------------------------

def pairwise_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels_a, labels_b, iou_matrix) for non-background labels."""
    labels_a = np.unique(mask_a)
    labels_a = labels_a[labels_a != 0]
    labels_b = np.unique(mask_b)
    labels_b = labels_b[labels_b != 0]
    if labels_a.size == 0 or labels_b.size == 0:
        return labels_a, labels_b, np.zeros((labels_a.size, labels_b.size), dtype=np.float32)

    area_a = {int(l): int((mask_a == l).sum()) for l in labels_a}
    area_b = {int(l): int((mask_b == l).sum()) for l in labels_b}
    inter = np.zeros((labels_a.size, labels_b.size), dtype=np.int64)
    for i, la in enumerate(labels_a):
        a_pixels = mask_a == la
        overlapped = mask_b[a_pixels]
        vals, counts = np.unique(overlapped[overlapped != 0], return_counts=True)
        for v, c in zip(vals, counts):
            j = np.searchsorted(labels_b, v)
            if j < labels_b.size and labels_b[j] == v:
                inter[i, j] = c
    union = (
        np.array([area_a[int(l)] for l in labels_a])[:, None]
        + np.array([area_b[int(l)] for l in labels_b])[None, :]
        - inter
    )
    iou = inter.astype(np.float32) / np.maximum(union.astype(np.float32), 1.0)
    return labels_a, labels_b, iou


def iou_distribution_frame(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    """Per-frame IoU summary: best-match IoU per cell in mask_a vs mask_b."""
    _, _, iou = pairwise_overlap(mask_a, mask_b)
    if iou.size == 0:
        return {"n_cells_a": int((np.unique(mask_a) != 0).sum()), "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
    best = iou.max(axis=1) if iou.shape[1] > 0 else np.zeros(iou.shape[0])
    return {
        "n_cells_a": int(iou.shape[0]),
        "mean": float(best.mean()),
        "median": float(np.median(best)),
        "p10": float(np.percentile(best, 10)),
        "p90": float(np.percentile(best, 90)),
    }


# ----------------------------------------------------------------------------
# 4.2 / 4.3  Merge and split counts
# ----------------------------------------------------------------------------

def merge_split_counts(mask_a: np.ndarray, mask_b: np.ndarray, iou_threshold: float = 0.3) -> dict[str, int]:
    """Count merges (one in A overlaps 2+ in B) and splits (symmetric)."""
    _, _, iou = pairwise_overlap(mask_a, mask_b)
    if iou.size == 0:
        return {"merge": 0, "split": 0}
    merge = int(((iou > iou_threshold).sum(axis=1) >= 2).sum())  # rows with >=2 matches
    split = int(((iou > iou_threshold).sum(axis=0) >= 2).sum())  # cols with >=2 matches
    return {"merge": merge, "split": split}


# ----------------------------------------------------------------------------
# 4.4  Temporal cell-count consistency
# ----------------------------------------------------------------------------

def temporal_count_consistency(masks: Sequence[np.ndarray]) -> dict[str, float]:
    """Object-count time series stats across frames in one matrix cell."""
    counts = np.array([int((np.unique(m) != 0).sum()) for m in masks])
    if counts.size == 0:
        return {"mean": np.nan, "std": np.nan, "max_abs_delta": np.nan, "jump_30pct_frames": 0}
    diffs = np.abs(np.diff(counts))
    rolling = np.array([np.median(counts[max(0, i - 5): i + 6]) for i in range(counts.size)])
    jumps = int((np.abs(counts - rolling) > 0.3 * np.maximum(rolling, 1)).sum())
    return {
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "max_abs_delta": float(diffs.max()) if diffs.size else 0.0,
        "jump_30pct_frames": jumps,
    }


# ----------------------------------------------------------------------------
# 4.5  Mitosis robustness (heuristic; no GT available)
# ----------------------------------------------------------------------------

def mitosis_candidate_frames(h2b_series: Sequence[np.ndarray], pct_threshold: float = 99.0) -> list[int]:
    """Frames where the 99th-percentile H2B intensity spikes (round-up proxy)."""
    p99 = np.array([float(np.percentile(im, pct_threshold)) for im in h2b_series])
    if p99.size < 3:
        return []
    rolling = np.array([np.median(p99[max(0, i - 5): i + 6]) for i in range(p99.size)])
    return [int(i) for i, (v, r) in enumerate(zip(p99, rolling)) if v > 1.10 * max(r, 1.0)]


def mitosis_robustness(
    masks: Sequence[np.ndarray],
    h2b_series: Sequence[np.ndarray],
    window: int = 2,
) -> dict[str, float | int]:
    """For each candidate frame, did the mask resolve a split within ±window?"""
    cand = mitosis_candidate_frames(h2b_series)
    if not cand:
        return {"n_candidates": 0, "resolution_rate": np.nan, "vanish_rate": np.nan}
    resolved = 0
    vanished = 0
    for t in cand:
        if t < 1 or t >= len(masks) - 1:
            continue
        before = int((np.unique(masks[t - 1]) != 0).sum())
        at = int((np.unique(masks[t]) != 0).sum())
        after_count = max(
            int((np.unique(masks[min(t + w, len(masks) - 1)]) != 0).sum())
            for w in range(1, window + 1)
        )
        if after_count >= before + 1:
            resolved += 1
        if at < max(before - 1, 0):
            vanished += 1
    n = len(cand)
    return {
        "n_candidates": n,
        "resolution_rate": resolved / n if n else np.nan,
        "vanish_rate": vanished / n if n else np.nan,
    }


# ----------------------------------------------------------------------------
# 4.6  Boundary Hausdorff (95th percentile)
# ----------------------------------------------------------------------------

def boundary_hausdorff95_frame(
    mask_a: np.ndarray, mask_b: np.ndarray, comparable: bool = True
) -> float:
    """95th-percentile boundary Hausdorff between two masks.

    Hausdorff-invalid sentinel policy (v3, 2026-05-20 — resolves
    STAGE02_CONTRACT.md §C). When no comparable boundary exists this returns
    **np.nan, never np.inf**. "No comparable boundary" covers three cases:
      1. an empty mask on either side,
      2. no extractable boundary pixels on either side, and
      3. an explicitly-declared incomparable object pairing (``comparable=False``)
         — e.g. a BF *whole-cell* mask scored against an H2B *nuclear* reference.
         That cross-class pairing produced the smoke's spurious
         ``stardist_bf H95=422``: a real, finite, but meaningless distance
         between two different object classes (a metric-collapse artifact).

    Why np.nan and not np.inf (one choice, applied consistently):
      * every downstream reduction here is nan-aware (``np.nanmedian`` /
        ``np.nanmean`` / ``np.nanpercentile``) — NaN is silently skipped,
        whereas inf poisons percentiles and means;
      * the semantics are "not applicable / not computed", which NaN expresses;
        inf would assert a real-but-unbounded distance, which is false here;
      * ``compare_masks._jsonable`` already maps non-finite floats → null, so
        NaN serialises cleanly without a special case.
    Under the v3 family split, BF rows do not call this metric at all (they use
    the containment family); ``comparable=False`` is the defensive guard for any
    direct cross-class call.
    """
    if not comparable:
        return np.nan
    bin_a = mask_a > 0
    bin_b = mask_b > 0
    if not bin_a.any() or not bin_b.any():
        return np.nan
    dt_a = distance_transform_edt(~bin_a)
    dt_b = distance_transform_edt(~bin_b)
    # Boundary pixels: pixels in A that are not 8-neighbor-interior.
    edge_a = bin_a ^ (bin_a & np.roll(bin_a, 1, axis=0) & np.roll(bin_a, -1, axis=0) & np.roll(bin_a, 1, axis=1) & np.roll(bin_a, -1, axis=1))
    edge_b = bin_b ^ (bin_b & np.roll(bin_b, 1, axis=0) & np.roll(bin_b, -1, axis=0) & np.roll(bin_b, 1, axis=1) & np.roll(bin_b, -1, axis=1))
    d_ab = dt_b[edge_a]
    d_ba = dt_a[edge_b]
    if d_ab.size == 0 or d_ba.size == 0:
        return np.nan
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


# ----------------------------------------------------------------------------
# 4.7  Downstream-track proxy (not full tracking)
# ----------------------------------------------------------------------------

def _centroids(mask: np.ndarray) -> np.ndarray:
    props = measure.regionprops(mask)
    if not props:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([p.centroid for p in props], dtype=np.float32)


def downstream_track_proxy(masks: Sequence[np.ndarray], iou_threshold: float = 0.5, displacement_max_px: float = 20.0) -> dict[str, float]:
    """Three sub-metrics: displacement consistency, bijection rate, area CV.

    All computed on adjacent-frame pairs without an actual tracker.
    """
    if len(masks) < 2:
        return {"displacement_within_max_frac": np.nan, "bijection_rate": np.nan, "area_cv_median": np.nan}
    disp_within = []
    bijection_flags = []
    area_track: dict[int, list[int]] = {}

    next_track_id = 0
    prev_label_to_track: dict[int, int] = {}

    for t in range(len(masks) - 1):
        a, b = masks[t], masks[t + 1]
        labels_a, labels_b, iou = pairwise_overlap(a, b)
        if iou.size == 0:
            bijection_flags.append(False)
            continue

        # bijection: every label_a has exactly one label_b match with IoU > thr, no shared
        a_match_count = (iou > iou_threshold).sum(axis=1)
        b_match_count = (iou > iou_threshold).sum(axis=0)
        bijection = (
            labels_a.size > 0
            and (a_match_count == 1).all()
            and (b_match_count <= 1).all()
        )
        bijection_flags.append(bool(bijection))

        # displacement: for each matched pair, centroid distance
        cents_a = _centroids(a)
        cents_b = _centroids(b)
        for i, la in enumerate(labels_a):
            j = int(iou[i].argmax())
            if iou[i, j] <= iou_threshold:
                continue
            d = float(np.linalg.norm(cents_a[i] - cents_b[j]))
            disp_within.append(d < displacement_max_px)

            # area CV tracking: thread the label across frames
            track_id = prev_label_to_track.get(int(la))
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                area_track[track_id] = [int((a == la).sum())]
            area_track[track_id].append(int((b == labels_b[j]).sum()))
            # update for next iteration
            prev_label_to_track[int(labels_b[j])] = track_id

        # reset to next-frame indexing
        new_map = {}
        for la_int, tid in prev_label_to_track.items():
            if la_int in [int(l) for l in labels_b]:
                new_map[la_int] = tid
        prev_label_to_track = new_map

    cvs = []
    for tid, series in area_track.items():
        if len(series) < 2:
            continue
        arr = np.array(series, dtype=np.float32)
        if arr.mean() > 0:
            cvs.append(float(arr.std() / arr.mean()))

    return {
        "displacement_within_max_frac": float(np.mean(disp_within)) if disp_within else np.nan,
        "bijection_rate": float(np.mean(bijection_flags)) if bijection_flags else np.nan,
        "area_cv_median": float(np.median(cvs)) if cvs else np.nan,
    }


# ----------------------------------------------------------------------------
# 4.8  Dim-nucleus failure rate
# ----------------------------------------------------------------------------

def dim_frame_indices(h2b_series: Sequence[np.ndarray], lower_pct_within: float = 10.0, dim_pct_across: float = 5.0) -> list[int]:
    """Frames whose `lower_pct_within`-percentile H2B intensity is in the
    bottom `dim_pct_across`% across the time series."""
    p_within = np.array([float(np.percentile(im, lower_pct_within)) for im in h2b_series])
    if p_within.size == 0:
        return []
    cutoff = float(np.percentile(p_within, dim_pct_across))
    return [int(i) for i, v in enumerate(p_within) if v <= cutoff]


def dim_nucleus_failure_rate(
    masks: Sequence[np.ndarray],
    h2b_series: Sequence[np.ndarray],
    reference_masks: Sequence[np.ndarray],
    centroid_tol_px: float = 10.0,
) -> dict[str, float]:
    """Fraction of reference-mask cells not recovered by this cell's mask on dim frames."""
    dim_idx = dim_frame_indices(h2b_series)
    if not dim_idx:
        return {"dim_frame_count": 0, "failure_rate": np.nan}
    missed = 0
    total = 0
    for t in dim_idx:
        if t >= len(masks) or t >= len(reference_masks):
            continue
        ref_cents = _centroids(reference_masks[t])
        cand_cents = _centroids(masks[t])
        total += int(ref_cents.shape[0])
        if ref_cents.size == 0:
            continue
        if cand_cents.size == 0:
            missed += int(ref_cents.shape[0])
            continue
        d = cdist(ref_cents, cand_cents)
        nearest = d.min(axis=1)
        missed += int((nearest > centroid_tol_px).sum())
    return {
        "dim_frame_count": len(dim_idx),
        "failure_rate": float(missed / total) if total else np.nan,
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

@dataclass
class CellResult:
    cell_name: str
    iou: dict[str, float]
    merge_split: dict[str, int]
    temporal_count: dict[str, float]
    mitosis: dict[str, float | int]
    boundary: dict[str, float]
    downstream_proxy: dict[str, float]
    dim_failure: dict[str, float]

    def flat_row(self) -> dict[str, float | int | str]:
        return {
            "cell": self.cell_name,
            "iou_mean": self.iou["mean"],
            "iou_p10": self.iou["p10"],
            "iou_p90": self.iou["p90"],
            "merge_count": self.merge_split["merge"],
            "split_count": self.merge_split["split"],
            "count_mean": self.temporal_count["mean"],
            "count_jump_30pct_frames": self.temporal_count["jump_30pct_frames"],
            "mitosis_resolution_rate": self.mitosis["resolution_rate"],
            "boundary_h95_median": self.boundary["median"],
            "track_proxy_bijection_rate": self.downstream_proxy["bijection_rate"],
            "track_proxy_displacement_frac": self.downstream_proxy["displacement_within_max_frac"],
            "track_proxy_area_cv": self.downstream_proxy["area_cv_median"],
            "dim_nucleus_failure_rate": self.dim_failure["failure_rate"],
        }


def compute_cell_metrics(
    cell_name: str,
    candidate_masks: Sequence[np.ndarray],
    reference_masks: Sequence[np.ndarray],
    h2b_series: Sequence[np.ndarray],
) -> CellResult:
    """Compute the GT-quality family (8 metrics) for one matrix cell vs the
    nuclear reference.

    v3: this is the **GT-quality family** (H2B rows, nucleus-vs-nucleus). BF
    rows use ``compute_bf_containment_metrics`` instead; ``compute_stage02_metrics``
    routes between the two. Behaviour for H2B rows is unchanged from v2.
    """
    iou_per = [iou_distribution_frame(c, r) for c, r in zip(candidate_masks, reference_masks)]
    ms_per = [merge_split_counts(c, r) for c, r in zip(candidate_masks, reference_masks)]
    # GT-quality family is nucleus-vs-nucleus, so boundaries are comparable.
    h95_per = [boundary_hausdorff95_frame(c, r, comparable=True) for c, r in zip(candidate_masks, reference_masks)]

    iou_arr = {
        "mean": float(np.nanmean([x["mean"] for x in iou_per])),
        "median": float(np.nanmedian([x["median"] for x in iou_per])),
        "p10": float(np.nanmin([x["p10"] for x in iou_per])),
        "p90": float(np.nanmax([x["p90"] for x in iou_per])),
    }
    return CellResult(
        cell_name=cell_name,
        iou=iou_arr,
        merge_split={
            "merge": int(sum(x["merge"] for x in ms_per)),
            "split": int(sum(x["split"] for x in ms_per)),
        },
        temporal_count=temporal_count_consistency(candidate_masks),
        mitosis=mitosis_robustness(candidate_masks, h2b_series),
        boundary={
            "median": float(np.nanmedian(h95_per)) if h95_per else float("nan"),
            "p95": float(np.nanpercentile(h95_per, 95)) if h95_per else float("nan"),
        },
        downstream_proxy=downstream_track_proxy(candidate_masks),
        dim_failure=dim_nucleus_failure_rate(candidate_masks, h2b_series, reference_masks),
    )


# ============================================================================
# v3 amendment (2026-05-20) — BF containment / coverage family
#
# BF rows are WHOLE-CELL recovery benchmarks, not nucleus-boundary benchmarks
# (STAGE02_CONTRACT.md §B, decision S0). We therefore score BF masks by how
# well they CONTAIN the DAPI/H2B nuclei (coverage/recall, nuclei-per-cell,
# assignment uniqueness, orphan rate) rather than by nucleus IoU.
#
# Locked decisions wired in here:
#   #1  assignment rule = majority-area τ=0.5 + centroid-in-cell fallback
#   S0  BF = whole-cell vs nuclei (containment, not co-IoU)
#   §F  orphan-BF and coverage are REPORTED, not gated (dim-DAPI false negatives)
# In every BF function, ``nuclei_mask`` is the DAPI/H2B GT and ``cell_mask`` is
# the candidate BF whole-cell segmentation.
# ============================================================================

def assign_nuclei_to_cells(
    nuclei_mask: np.ndarray, cell_mask: np.ndarray, *, tau: float = 0.5
) -> tuple[dict[int, int], dict[int, dict[int, float]]]:
    """Assign each DAPI nucleus to a whole-cell label (decision #1, LOCKED).

    Primary rule = **majority-area**: a nucleus is assigned to the cell that
    contains ≥ ``tau`` of the nucleus area. Fallback = **centroid-in-cell**:
    when no cell reaches ``tau``, assign to the cell label under the nucleus
    centroid (0 = background → orphan nucleus / missed DAPI).

    τ=0.5 is our calibration choice, not a value lifted from a paper; the
    nucleus→whole-cell assignment pattern itself is standard practice.

    Returns ``(assignment, overlap_frac)`` where
      * ``assignment``  : ``{nucleus_label: cell_label}`` (cell_label 0 = orphan)
      * ``overlap_frac``: ``{nucleus_label: {cell_label: frac_of_nucleus_in_cell}}``
    """
    nuc_labels = np.unique(nuclei_mask)
    nuc_labels = nuc_labels[nuc_labels != 0]
    assignment: dict[int, int] = {}
    overlap_frac: dict[int, dict[int, float]] = {}
    h, w = cell_mask.shape
    for n in nuc_labels:
        npix = nuclei_mask == n
        area = int(npix.sum())
        if area == 0:
            continue
        cell_vals = cell_mask[npix]
        nz = cell_vals[cell_vals != 0]
        vals, counts = np.unique(nz, return_counts=True)
        fracs = {int(v): float(c) / area for v, c in zip(vals, counts)}
        overlap_frac[int(n)] = fracs
        if fracs:
            best = max(fracs, key=fracs.get)
            if fracs[best] >= tau:
                assignment[int(n)] = best
                continue
        # centroid-in-cell fallback (sanity cross-check per decision #1)
        ys, xs = np.nonzero(npix)
        cy = int(min(max(round(float(ys.mean())), 0), h - 1))
        cx = int(min(max(round(float(xs.mean())), 0), w - 1))
        assignment[int(n)] = int(cell_mask[cy, cx])
    return assignment, overlap_frac


def dapi_coverage(nuclei_mask: np.ndarray, cell_mask: np.ndarray, *, tau: float = 0.5) -> dict[str, float]:
    """DAPI coverage / recall (PRIMARY BF metric) + missed-DAPI rate.

    recall = fraction of DAPI nuclei assigned to some BF cell.
    missed_dapi_rate = 1 − recall (the failure mode Bill named).
    """
    assignment, _ = assign_nuclei_to_cells(nuclei_mask, cell_mask, tau=tau)
    n = len(assignment)
    n_recovered = int(sum(1 for c in assignment.values() if c != 0))
    return {
        "n_nuclei": n,
        "n_recovered": n_recovered,
        "recall": (n_recovered / n) if n else np.nan,
        "missed_dapi_rate": (1.0 - n_recovered / n) if n else np.nan,
    }


def nuclei_per_cell_histogram(nuclei_mask: np.ndarray, cell_mask: np.ndarray, *, tau: float = 0.5) -> dict[str, int]:
    """Histogram of #nuclei assigned per BF cell. Expect a peak at 1, a tail at
    2; bucket 0 = orphan BF cell, ≥3 = under-segmentation."""
    assignment, _ = assign_nuclei_to_cells(nuclei_mask, cell_mask, tau=tau)
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0]
    per_cell = {int(c): 0 for c in cell_labels}
    for c in assignment.values():
        if c != 0 and int(c) in per_cell:
            per_cell[int(c)] += 1
    hist = {"cells_0": 0, "cells_1": 0, "cells_2": 0, "cells_3plus": 0}
    for c in cell_labels:
        k = per_cell[int(c)]
        if k == 0:
            hist["cells_0"] += 1
        elif k == 1:
            hist["cells_1"] += 1
        elif k == 2:
            hist["cells_2"] += 1
        else:
            hist["cells_3plus"] += 1
    hist["n_cells"] = int(cell_labels.size)
    return hist


def assignment_uniqueness(
    nuclei_mask: np.ndarray, cell_mask: np.ndarray, *, tau: float = 0.5, split_min_frac: float = 0.2
) -> dict[str, float]:
    """Assignment uniqueness vs split (over-seg) rate.

    unique_rate = fraction of nuclei with exactly one cell at ≥ ``tau`` (cleanly,
    uniquely assigned). split_rate = fraction of nuclei straddling ≥2 cells each
    above ``split_min_frac`` — i.e. the BF mask over-segments relative to the
    nucleus (a nucleus cut by a BF cell boundary).
    """
    _, overlap_frac = assign_nuclei_to_cells(nuclei_mask, cell_mask, tau=tau)
    n = len(overlap_frac)
    n_unique = 0
    n_split = 0
    for fracs in overlap_frac.values():
        if sum(1 for f in fracs.values() if f >= tau) == 1:
            n_unique += 1
        if sum(1 for f in fracs.values() if f >= split_min_frac) >= 2:
            n_split += 1
    return {
        "n_nuclei": n,
        "n_unique": n_unique,
        "n_split": n_split,
        "unique_rate": (n_unique / n) if n else np.nan,
        "split_rate": (n_split / n) if n else np.nan,
    }


def orphan_bf_rate(nuclei_mask: np.ndarray, cell_mask: np.ndarray, *, tau: float = 0.5) -> dict[str, float]:
    """Fraction of BF cells with zero assigned nuclei.

    REPORTED, NOT GATED (STAGE02_CONTRACT.md §C/§F): there was no FACS sort, so
    the DAPI GT has dim-DAPI false negatives that inflate orphan-BF on the
    precision side. Surface it as a characterization, never as a pass/fail gate.
    """
    assignment, _ = assign_nuclei_to_cells(nuclei_mask, cell_mask, tau=tau)
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels != 0]
    assigned = {int(c) for c in assignment.values() if c != 0}
    n_orphan = int(sum(1 for c in cell_labels if int(c) not in assigned))
    return {
        "n_cells": int(cell_labels.size),
        "n_orphan": n_orphan,
        "orphan_rate": (n_orphan / cell_labels.size) if cell_labels.size else np.nan,
    }


def bf_vs_cytoring_iou(bf_cell_mask: np.ndarray, cytoring_cell_mask: np.ndarray) -> dict[str, float]:
    """OPTIONAL whole-cell-vs-whole-cell IoU (BF vs Bill's cyto-ring whole-cell).

    IoU is VALID here because both operands are whole-cell objects — unlike
    nucleus-IoU on a BF mask. The cyto-ring whole-cell layer is not yet wired
    into ``compare_masks.py`` (the reference zarr currently exposes
    ``nuc_labels`` only), so this is provided as a ready-to-call helper that the
    deferred orchestration edit can use once a whole-cell reference is loaded.
    """
    return iou_distribution_frame(bf_cell_mask, cytoring_cell_mask)


@dataclass
class BFContainmentResult:
    """Containment-family result for one BF matrix cell. Mirrors ``CellResult``
    (also exposes ``flat_row``) but carries the BF-specific columns; the two
    families intentionally have different schemas — see ``compute_stage02_metrics``."""

    cell_name: str
    coverage: dict[str, float]
    nuclei_per_cell: dict[str, float]
    uniqueness: dict[str, float]
    orphan: dict[str, float]
    bf_vs_cytoring: dict[str, float] | None

    def flat_row(self) -> dict[str, float | int | str]:
        return {
            "cell": self.cell_name,
            "metric_family": "containment",
            "dapi_recall": self.coverage["recall"],
            "missed_dapi_rate": self.coverage["missed_dapi_rate"],
            "nuclei_per_cell_1_frac": self.nuclei_per_cell["cells_1_frac"],
            "nuclei_per_cell_2plus_frac": self.nuclei_per_cell["cells_2plus_frac"],
            "assignment_unique_rate": self.uniqueness["unique_rate"],
            "assignment_split_rate": self.uniqueness["split_rate"],
            "orphan_bf_rate": self.orphan["orphan_rate"],
            "bf_vs_cytoring_iou_mean": (self.bf_vs_cytoring or {}).get("mean", np.nan),
        }


def compute_bf_containment_metrics(
    cell_name: str,
    candidate_cell_masks: Sequence[np.ndarray],
    nuclei_reference_masks: Sequence[np.ndarray],
    *,
    tau: float = 0.5,
    split_min_frac: float = 0.2,
    cytoring_cell_masks: Sequence[np.ndarray] | None = None,
) -> BFContainmentResult:
    """Compute the containment family for one BF matrix cell over a frame series.

    Per-frame counts are summed, then rates are recomputed from the totals (so
    frames with more nuclei weight proportionally, and empty frames are inert).
    ``nuclei_reference_masks`` are the DAPI/H2B GT nuclei (in ``compare_masks``
    these are the same ``reference_masks`` used by the GT-quality family).
    """
    n_nuc = n_recovered = 0
    n_unique = n_split = 0
    n_cells = n_orphan = 0
    hist = {"cells_0": 0, "cells_1": 0, "cells_2": 0, "cells_3plus": 0}
    for cm, nm in zip(candidate_cell_masks, nuclei_reference_masks):
        cov = dapi_coverage(nm, cm, tau=tau)
        n_nuc += cov["n_nuclei"]
        n_recovered += cov["n_recovered"]
        uq = assignment_uniqueness(nm, cm, tau=tau, split_min_frac=split_min_frac)
        n_unique += uq["n_unique"]
        n_split += uq["n_split"]
        h = nuclei_per_cell_histogram(nm, cm, tau=tau)
        for k in ("cells_0", "cells_1", "cells_2", "cells_3plus"):
            hist[k] += h[k]
        orp = orphan_bf_rate(nm, cm, tau=tau)
        n_cells += orp["n_cells"]
        n_orphan += orp["n_orphan"]

    total_cells = hist["cells_0"] + hist["cells_1"] + hist["cells_2"] + hist["cells_3plus"]
    coverage = {
        "n_nuclei": n_nuc,
        "n_recovered": n_recovered,
        "recall": (n_recovered / n_nuc) if n_nuc else np.nan,
        "missed_dapi_rate": (1.0 - n_recovered / n_nuc) if n_nuc else np.nan,
    }
    uniqueness = {
        "n_nuclei": n_nuc,
        "unique_rate": (n_unique / n_nuc) if n_nuc else np.nan,
        "split_rate": (n_split / n_nuc) if n_nuc else np.nan,
    }
    nuclei_per_cell = {
        **hist,
        "n_cells": total_cells,
        "cells_1_frac": (hist["cells_1"] / total_cells) if total_cells else np.nan,
        "cells_2plus_frac": ((hist["cells_2"] + hist["cells_3plus"]) / total_cells) if total_cells else np.nan,
    }
    orphan = {
        "n_cells": n_cells,
        "n_orphan": n_orphan,
        "orphan_rate": (n_orphan / n_cells) if n_cells else np.nan,
    }
    bf_vs_cytoring = None
    if cytoring_cell_masks is not None:
        ious = [bf_vs_cytoring_iou(bf, cr) for bf, cr in zip(candidate_cell_masks, cytoring_cell_masks)]
        bf_vs_cytoring = {
            "mean": float(np.nanmean([x["mean"] for x in ious])) if ious else np.nan,
            "median": float(np.nanmedian([x["median"] for x in ious])) if ious else np.nan,
        }
    return BFContainmentResult(
        cell_name=cell_name,
        coverage=coverage,
        nuclei_per_cell=nuclei_per_cell,
        uniqueness=uniqueness,
        orphan=orphan,
        bf_vs_cytoring=bf_vs_cytoring,
    )


# ----------------------------------------------------------------------------
# v3 family-aware dispatcher
# ----------------------------------------------------------------------------

def metric_family_for_cell(cell_name: str) -> str:
    """Route a matrix cell to its v3 metric family.

    BF rows (``*_bf``) → ``"containment"`` (whole-cell vs DAPI nuclei).
    H2B rows (everything else) → ``"gt_quality"`` (nucleus-vs-nucleus).
    """
    return "containment" if cell_name.lower().endswith("_bf") else "gt_quality"


def compute_stage02_metrics(
    cell_name: str,
    candidate_masks: Sequence[np.ndarray],
    reference_masks: Sequence[np.ndarray],
    h2b_series: Sequence[np.ndarray],
    *,
    tau: float = 0.5,
    split_min_frac: float = 0.2,
    cytoring_cell_masks: Sequence[np.ndarray] | None = None,
) -> CellResult | BFContainmentResult:
    """v3 family-aware entry point. Returns a ``CellResult`` (GT-quality) or a
    ``BFContainmentResult`` (containment); both expose ``flat_row()``.

    ``reference_masks`` are the DAPI/H2B nuclei in BOTH families: nucleus-vs-
    nucleus for H2B rows, and the nuclei to be contained for BF rows.

    NOTE (docs-first): ``compare_masks.py`` still calls ``compute_cell_metrics``
    directly for all four cells (the v2 path). Switching it to this dispatcher —
    which also turns the metric table into a per-family schema and updates the
    smoke shape — is the next DEFERRED edit; intentionally not done here.
    """
    if metric_family_for_cell(cell_name) == "containment":
        return compute_bf_containment_metrics(
            cell_name,
            candidate_cell_masks=candidate_masks,
            nuclei_reference_masks=reference_masks,
            tau=tau,
            split_min_frac=split_min_frac,
            cytoring_cell_masks=cytoring_cell_masks,
        )
    return compute_cell_metrics(cell_name, candidate_masks, reference_masks, h2b_series)
