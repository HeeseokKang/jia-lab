"""Click-to-recover segmentation-correction tool — reusable napari widget.

Human-in-the-loop **false-negative recovery** for the H2B reference
segmentation. The reference StarDist masks already cover the easy nuclei; this
tool exists to recover the ones it *misses* (dim / isolated / mitotic / clearly
absent nuclei) so a fine-tuned model stops silently dropping them — which is
where downstream selection bias enters the biology.

It is **not** a generic labeling platform and **not** the lineage annotator
(``src/tracking/gt_annotator.py``); it shares that tool's proven ergonomics
(viewer-level click handler, continuous autosave, session-resume, docked panel)
but its unit of work is "fix this frame's nuclei", not "follow this cell".

Design (optimized for the fastest practical correction loop):
  * **one click = one recovered nucleus.** Click a missed nucleus; the tool
    grows a label from the local H2B intensity (falling back to a fixed disk for
    dim nuclei) and stamps it as a *new* label, writing **only into background**
    pixels so it never clobbers a neighbouring existing nucleus.
  * **split / merge use napari's native Labels tools.** Pick up the brush /
    eraser / fill (the layer's own buttons or keys ``2``/``3``/``4``) and the
    click-to-add handler auto-suppresses (it only fires while the masks layer is
    in pan/zoom mode). Native ``Ctrl+Z`` undoes brush edits; ``u`` undoes the
    last click-add. No custom split/merge UI is invented.
  * **per-frame commit → fine-tunable pairs.** ``c`` commits the current frame:
    it writes the H2B image and the corrected instance mask as a matched
    ``images/tXXXX.tif`` / ``masks/tXXXX.tif`` pair (the StarDist/Cellpose
    fine-tuning layout) and records the edit in ``manifest.csv``. Re-launch
    reloads committed masks so a frame is never re-done.

The deliverable is the ``images/`` + ``masks/`` pair set + ``manifest.csv``;
the on-screen layers are a convenience view, refreshed defensively so a napari
styling hiccup never blocks a data write.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

MANIFEST_COLUMNS = [
    "timepoint", "n_before", "n_after", "n_added", "done", "updated_at",
]
POINTS_COLUMNS = [
    "timepoint", "centroid_y", "centroid_x", "new_label", "area", "mode",
]

_CLICK_MOVE_TOL = 3.0  # data-px; drags larger than this are pans, not clicks


class SegCorrector:
    """State machine + napari wiring for click-driven false-negative recovery.

    Parameters
    ----------
    viewer:
        napari viewer already holding the H2B image + editable masks layers.
    image_layer:
        H2B raw image layer, shape ``(T, Y, X)`` — the model input. Used both to
        grow new-nucleus labels and to export the fine-tuning image.
    masks_layer:
        Editable Labels layer, shape ``(T, Y, X)``, the corrected reference
        masks (committed corrections pre-loaded by the launcher on resume).
    out_dir:
        ``seg_corrections/`` root; ``images/``, ``masks/``, ``manifest.csv``,
        ``added_points.csv`` are written underneath.
    t_offset:
        Real dataset timepoint of window frame 0 (``--t0``); committed file
        names use the true timepoint ``t_offset + frame_index``.
    disk_radius, grow_window:
        Fallback-disk radius (px) for dim nuclei, and the half-window (px) the
        intensity-grow looks within. Operator-tunable from the panel.
    """

    def __init__(
        self,
        viewer,
        image_layer,
        masks_layer,
        out_dir: Path | str,
        t_offset: int = 0,
        region: Optional[str] = None,
        fov: Optional[int] = None,
        disk_radius: int = 7,
        grow_window: int = 21,
        timepoints: Optional[list] = None,
    ) -> None:
        self.viewer = viewer
        self.image_layer = image_layer
        self.masks_layer = masks_layer
        # edit the masks in place + refresh(); never reassign the whole TYX stack
        self.masks = np.asarray(masks_layer.data)
        self.raw = np.asarray(image_layer.data)
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "images"
        self.mask_dir = self.out_dir / "masks"
        self.t_offset = int(t_offset)
        # Optional explicit per-frame timepoints (for non-contiguous frame
        # selections, e.g. a stratified sample). When None the legacy
        # contiguous mapping ``t_offset + frame_idx`` is used.
        self.timepoints = None if timepoints is None else [int(t) for t in timepoints]
        self._tp_to_idx = (None if self.timepoints is None
                           else {tp: i for i, tp in enumerate(self.timepoints)})
        self.region = region
        self.fov = fov
        self.disk_radius = int(disk_radius)
        self.grow_window = int(grow_window)

        self.points: list[dict] = []          # every click-add (provenance)
        self.manifest: dict[int, dict] = {}    # tp -> manifest row
        self._undo: list[tuple] = []           # (frame_idx, y0, x0, prior_patch, label)
        self._status_label = None
        self._n_before: dict[int, int] = {}    # tp -> original nucleus count

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(exist_ok=True)
        self.mask_dir.mkdir(exist_ok=True)
        self._load_existing()
        # snapshot the pre-correction nucleus count per loaded frame (the true
        # baseline for n_added auditing); manifest values from resume win.
        for f in range(self.masks.shape[0]):
            tp = self._tp(f)
            self._n_before.setdefault(tp, self._count(f))

        self.added_layer = viewer.add_points(
            np.empty((0, 3)), name="recovered nuclei", size=16,
            border_color="#00ff88", face_color="transparent", symbol="cross",
        )
        self._wire_mouse()
        self._wire_keys()
        self._refresh_points()

    # ---- frame / geometry helpers ----------------------------------------
    @property
    def _frame(self) -> int:
        try:
            return int(self.viewer.dims.current_step[0])
        except Exception:
            return 0

    def _tp(self, frame_idx: int) -> int:
        """True dataset timepoint for a stack frame index.

        Uses the explicit ``timepoints`` list when provided (non-contiguous
        selections), else the legacy contiguous ``t_offset + frame_idx``.
        """
        if self.timepoints is not None:
            return int(self.timepoints[frame_idx])
        return self.t_offset + frame_idx

    def _count(self, frame_idx: int) -> int:
        return int(len(np.unique(self.masks[frame_idx])) - 1)  # minus background

    def _next_label(self, frame_idx: int) -> int:
        return int(self.masks[frame_idx].max()) + 1

    # ---- core action: add a missed nucleus -------------------------------
    def add_nucleus(self, y: int, x: int, frame_idx: Optional[int] = None) -> None:
        """Stamp a new label for a missed nucleus at ``(y, x)`` on the frame.

        Grows the label from local H2B intensity; only background pixels are
        written, so an existing neighbour is never overwritten. Falls back to a
        fixed disk (dim nuclei where thresholding finds nothing).
        """
        f = self._frame if frame_idx is None else int(frame_idx)
        H, W = self.masks[f].shape
        if not (0 <= y < H and 0 <= x < W):
            return self._status("cursor outside image — ignored")
        if int(self.masks[f, y, x]) != 0:
            return self._status(
                f"already segmented here (label {int(self.masks[f, y, x])}). "
                "Use the brush to split/merge instead.")

        r = self.grow_window
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        win = self.raw[f, y0:y1, x0:x1].astype(np.float32)
        cy, cx = y - y0, x - x0

        blob, mode = self._grow_blob(win, cy, cx)
        # restrict to currently-background pixels only
        prior = self.masks[f, y0:y1, x0:x1]
        write = blob & (prior == 0)
        if not write.any():
            return self._status("nothing to add (all pixels already labelled)")

        label = self._next_label(f)
        prior_copy = prior.copy()
        prior[write] = label  # in-place edit of self.masks view
        area = int(write.sum())
        self._undo.append((f, y0, x0, prior_copy, label))

        tp = self._tp(f)
        self.points.append({
            "timepoint": tp, "centroid_y": float(y), "centroid_x": float(x),
            "new_label": label, "area": area, "mode": mode,
        })
        self.masks_layer.refresh()
        self._save_points()
        self._refresh_points()
        self._status(f"+nucleus t={tp} label={label} area={area}px ({mode})  "
                     f"[added this frame: {self._added_on(f)}]")

    def _grow_blob(self, win: np.ndarray, cy: int, cx: int) -> tuple[np.ndarray, str]:
        """Return (boolean blob in window coords, mode-tag).

        Intensity-grow: Otsu-threshold the local window, keep the connected
        component containing the click, bounded to a plausible nucleus size.
        Fall back to a disk when the grow is empty (dim) or runs away (touching
        a bright neighbour / debris).
        """
        disk = self._disk(win.shape, cy, cx, self.disk_radius)
        try:
            if float(win.max() - win.min()) < 1e-6:
                return disk, "disk(flat)"
            thr = threshold_otsu(win)
            fg = win > thr
            if not fg[cy, cx]:
                # click sits below local threshold (dim nucleus) -> disk
                return disk, "disk(dim)"
            lab, _ = ndi.label(fg)
            comp = lab == lab[cy, cx]
            area = int(comp.sum())
            disk_area = int(disk.sum())
            # runaway (merged with a neighbour / huge) -> safer disk
            if area > 6 * disk_area:
                return disk, "disk(runaway)"
            if area < max(8, disk_area // 4):
                return disk | comp, "grow+disk"
            return comp, "grow"
        except Exception:
            return disk, "disk(fallback)"

    @staticmethod
    def _disk(shape: tuple[int, int], cy: int, cx: int, radius: int) -> np.ndarray:
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    def _added_on(self, frame_idx: int) -> int:
        tp = self._tp(frame_idx)
        return sum(1 for p in self.points if p["timepoint"] == tp)

    # ---- commit / save / undo --------------------------------------------
    def commit_frame(self, mark_done: bool = True) -> None:
        """Export the current frame as a fine-tunable (image, mask) pair."""
        f = self._frame
        tp = self._tp(f)
        n_before = self._n_before.get(tp, self._count(f))
        mask = self._relabel(self.masks[f])
        tifffile.imwrite(self.img_dir / f"t{tp:04d}.tif", self.raw[f].astype(np.uint16))
        tifffile.imwrite(self.mask_dir / f"t{tp:04d}.tif", mask.astype(np.uint16))
        self.manifest[tp] = {
            "timepoint": tp, "n_before": int(n_before), "n_after": self._count(f),
            "n_added": self._added_on(f), "done": bool(mark_done),
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        self._save_manifest()
        verb = "committed (done)" if mark_done else "checkpointed"
        self._status(f"{verb} t={tp}: {n_before}->{self._count(f)} nuclei "
                     f"(+{self._added_on(f)})  -> masks/t{tp:04d}.tif")

    def checkpoint_frame(self) -> None:
        self.commit_frame(mark_done=False)

    def undo(self) -> None:
        if not self._undo:
            return self._status("nothing to undo (click-add)")
        f, y0, x0, prior_patch, label = self._undo.pop()
        h, w = prior_patch.shape
        self.masks[f, y0:y0 + h, x0:x0 + w] = prior_patch
        # drop the matching provenance row (last one for this frame+label)
        for i in range(len(self.points) - 1, -1, -1):
            if self.points[i]["new_label"] == label and \
               self.points[i]["timepoint"] == self._tp(f):
                self.points.pop(i)
                break
        self.masks_layer.refresh()
        self._save_points()
        self._refresh_points()
        self._status(f"undid add (label {label}, t={self._tp(f)})")

    @staticmethod
    def _relabel(frame: np.ndarray) -> np.ndarray:
        """Make instance ids contiguous 1..N (clean fine-tuning labels)."""
        out = np.zeros_like(frame, dtype=np.int32)
        for new_id, lab in enumerate(np.unique(frame[frame > 0]), start=1):
            out[frame == lab] = new_id
        return out

    # ---- persistence + resume --------------------------------------------
    def _load_existing(self) -> None:
        man = self.out_dir / "manifest.csv"
        if man.exists():
            try:
                for _, r in pd.read_csv(man).iterrows():
                    self.manifest[int(r["timepoint"])] = r.to_dict()
                    self._n_before[int(r["timepoint"])] = int(r["n_before"])
            except Exception:
                pass
        pts = self.out_dir / "added_points.csv"
        if pts.exists():
            try:
                self.points = pd.read_csv(pts)[POINTS_COLUMNS].to_dict("records")
            except Exception:
                pass

    def _save_points(self) -> None:
        pd.DataFrame(self.points, columns=POINTS_COLUMNS).to_csv(
            self.out_dir / "added_points.csv", index=False)

    def _save_manifest(self) -> None:
        rows = [self.manifest[k] for k in sorted(self.manifest)]
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(
            self.out_dir / "manifest.csv", index=False)

    # ---- view -------------------------------------------------------------
    def _refresh_points(self) -> None:
        try:
            if not self.points:
                self.added_layer.data = np.empty((0, 3))
                return
            df = pd.DataFrame(self.points)
            data = df[["timepoint", "centroid_y", "centroid_x"]].to_numpy(float)
            if self._tp_to_idx is not None:
                # non-contiguous selection: map true timepoint -> stack z-index
                data[:, 0] = [self._tp_to_idx.get(int(tp), -1) for tp in data[:, 0]]
                data = data[data[:, 0] >= 0]
            else:
                data[:, 0] -= self.t_offset
            self.added_layer.data = data
        except Exception as exc:
            self._status(f"(overlay refresh skipped: {exc})")

    def _status(self, msg: str) -> None:
        print(f"[seg-corrector] {msg}")
        if self._status_label is not None:
            try:
                self._status_label.value = msg
            except Exception:
                pass

    def done_frames(self) -> list[int]:
        return sorted(tp for tp, r in self.manifest.items() if r.get("done"))

    # ---- napari wiring ----------------------------------------------------
    def _wire_mouse(self) -> None:
        corr = self

        def on_mouse(viewer, event):
            if event.button != 1:
                return
            # only act as add-handler while masks layer is NOT in an edit mode,
            # so native brush/fill/erase keep the mouse for split/merge fixes
            try:
                if str(corr.masks_layer.mode) != "pan_zoom":
                    return
            except Exception:
                pass
            start = np.asarray(event.position[1:])
            dragged = False
            yield
            while event.type == "mouse_move":
                if np.hypot(*(np.asarray(event.position[1:]) - start)) > _CLICK_MOVE_TOL:
                    dragged = True
                yield
            if not dragged:
                pos = event.position
                corr.add_nucleus(int(round(pos[1])), int(round(pos[2])), int(round(pos[0])))

        self.viewer.mouse_drag_callbacks.append(on_mouse)

    def add_at_cursor(self) -> None:
        try:
            pos = self.viewer.cursor.position
            self.add_nucleus(int(round(pos[-2])), int(round(pos[-1])))
        except Exception as exc:
            self._status(f"cursor add failed: {exc}")

    def _wire_keys(self) -> None:
        v = self.viewer
        v.bind_key("f", lambda _v: self.add_at_cursor(), overwrite=True)
        v.bind_key("u", lambda _v: self.undo(), overwrite=True)
        v.bind_key("c", lambda _v: self.commit_frame(True), overwrite=True)
        v.bind_key("s", lambda _v: self.checkpoint_frame(), overwrite=True)


def attach_seg_corrector(
    viewer,
    image_layer,
    masks_layer,
    out_dir: Path | str,
    t_offset: int = 0,
    region: Optional[str] = None,
    fov: Optional[int] = None,
    disk_radius: int = 7,
    grow_window: int = 21,
    timepoints: Optional[list] = None,
) -> SegCorrector:
    """Build the corrector, dock its control panel, and return it."""
    corr = SegCorrector(
        viewer, image_layer, masks_layer, out_dir, t_offset=t_offset,
        region=region, fov=fov, disk_radius=disk_radius, grow_window=grow_window,
        timepoints=timepoints,
    )

    from magicgui.widgets import CheckBox, Container, Label, PushButton, SpinBox

    status = Label(value="")
    corr._status_label = status

    rad = SpinBox(value=disk_radius, min=2, max=30, label="fallback disk r (px)")
    rad.changed.connect(lambda val: setattr(corr, "disk_radius", int(val)))
    win = SpinBox(value=grow_window, min=7, max=61, label="grow window (px)")
    win.changed.connect(lambda val: setattr(corr, "grow_window", int(val)))

    def _btn(text, fn):
        b = PushButton(text=text)
        b.changed.connect(lambda *_: fn())
        return b

    panel = Container(widgets=[
        Label(value="Click a MISSED nucleus = recover it (one click = one nucleus)."),
        Label(value="Split/merge: use the Labels brush (keys 2/3/4); Ctrl+Z undoes those."),
        rad, win,
        _btn("add at cursor  f", corr.add_at_cursor),
        _btn("undo last add  u", corr.undo),
        _btn("commit frame (done)  c", lambda: corr.commit_frame(True)),
        _btn("checkpoint frame  s", corr.checkpoint_frame),
        status,
    ])
    viewer.window.add_dock_widget(panel.native, name="Seg correction", area="right")
    done = corr.done_frames()
    corr._status(f"ready. committed frames: {len(done)}"
                 + (f" (last t={done[-1]})" if done else ""))
    return corr
