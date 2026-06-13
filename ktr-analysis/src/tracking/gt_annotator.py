"""Click-to-annotate GT-lineage tool — reusable napari widget (Stage 03).

Manual ground-truth lineage annotation, but **click-driven instead of
typed**. The annotator removes the error-prone transcription step: you click
the nucleus you are tracking and the tool reads the StarDist ``mask_label``
under the click, computes the label centroid, and records the row. You never
type a label id or a coordinate.

Design (locked with operator 2026-05-21):
  * **per-frame, no auto-advance** — the rhythm is: click the cell, press the
    right-arrow to step one frame, click again. Two deliberate actions; nothing
    happens on its own, so a stray click/keystroke can't run away.
  * **events are editable state, not fire-and-forget** — ``d``/``x``/``l`` set
    the *last* recorded row's event; press another (or ``n``) to overwrite it.
    Nothing is committed until you move on, and ``u`` undoes the last row
    entirely. The event also recolors the point so you see what was recorded.
  * **dataset-agnostic / reusable** — takes a labels layer, an output CSV path,
    and a timepoint offset. No KTR-specific paths inside; FUCCI or any other
    per-cell-time-series pipeline can reuse it unchanged.

Output is the locked GT-lineage template schema (see ``GT_SCHEMA_COLUMNS``),
autosaved continuously to ``gt_lineage_filled.csv`` so work is never lost. The
CSV is the deliverable; the on-screen points layer is a convenience view and is
rebuilt defensively (a napari styling hiccup never blocks the data write).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Locked GT-lineage template columns (matches gt_lineage_template.csv).
GT_SCHEMA_COLUMNS = [
    "gt_cell_id", "timepoint", "mask_label", "centroid_x", "centroid_y",
    "event", "parent_gt_cell_id", "notes",
]

VALID_EVENTS = ("none", "division", "death", "leaves_fov", "enters")

# Per-event point face color (event read at a glance); background stays the
# continuity-seed yellow so "none" rows look like the seeds the operator follows.
_EVENT_COLOR = {
    "none": "#ffcc00",        # yellow
    "division": "#ff44aa",    # magenta
    "death": "#ff3333",       # red
    "leaves_fov": "#33aaff",  # blue
    "enters": "#33ffaa",      # green
}
# A small glyph appended to the text label so the event is legible without color.
_EVENT_GLYPH = {
    "none": "", "division": "÷", "death": "†", "leaves_fov": "→", "enters": "←",
}

_CLICK_MOVE_TOL = 3.0  # data-px; drags larger than this are pans, not clicks


class GtLineageAnnotator:
    """State machine + napari wiring for click-driven GT-lineage annotation.

    Parameters
    ----------
    viewer:
        The napari viewer (already holding raw + masks + seed layers).
    masks_layer:
        The StarDist labels layer, shape ``(T, Y, X)`` over the displayed
        window. ``mask_label`` is sampled from ``masks_layer.data``.
    out_csv:
        Where ``gt_lineage_filled.csv`` is autosaved.
    t_offset:
        Real dataset timepoint of window frame 0 (e.g. ``--t0``). Recorded
        ``timepoint`` = ``t_offset + frame_index`` so the CSV carries true
        frames even when the window starts mid-movie.
    start_id:
        Initial active ``gt_cell_id`` (default 1 = first continuity seed).
    region, fov:
        Optional provenance, written into the ``notes`` column on each row.
    """

    def __init__(
        self,
        viewer,
        masks_layer,
        out_csv: Path | str,
        t_offset: int = 0,
        start_id: int = 1,
        region: Optional[str] = None,
        fov: Optional[int] = None,
        daughter_id_start: int = 101,
    ) -> None:
        self.viewer = viewer
        self.masks_layer = masks_layer
        self.out_csv = Path(out_csv)
        self.t_offset = int(t_offset)
        self.active_id = int(start_id)
        self.region = region
        self.fov = fov
        # Daughter gt_cell_ids start here so they never collide with the
        # predefined continuity-seed numbers (1..24) the operator annotates.
        self.daughter_id_start = int(daughter_id_start)

        self.records: list[dict] = []
        self._parents: dict[int, object] = {}  # daughter gt_cell_id -> mother
        self._status_label = None  # set by the dock widget
        self._id_widget = None     # spinbox; kept in sync with active_id

        # resume an in-progress annotation if the CSV already exists
        if self.out_csv.exists():
            try:
                prev = pd.read_csv(self.out_csv)
                self.records = prev[GT_SCHEMA_COLUMNS].to_dict("records")
                # rebuild daughter->mother links so id allocation stays correct
                for r in self.records:
                    p = r.get("parent_gt_cell_id")
                    if p is not None and not pd.isna(p):
                        self._parents[int(r["gt_cell_id"])] = int(p)
            except Exception:
                pass

        # dedicated points layer for the annotation overlay
        self.points_layer = viewer.add_points(
            np.empty((0, 3)),
            name="GT annotations",
            size=14,
            border_color="white",
            face_color=_EVENT_COLOR["none"],
            symbol="o",
        )
        self._wire_mouse()
        self._wire_keys()
        self._refresh_points()

    # ---- geometry helpers -------------------------------------------------
    @property
    def _masks(self) -> np.ndarray:
        return np.asarray(self.masks_layer.data)

    def _label_centroid(self, frame_idx: int, label: int) -> tuple[float, float]:
        frame = self._masks[frame_idx]
        ys, xs = np.where(frame == label)
        if len(xs) == 0:
            return float("nan"), float("nan")
        return float(xs.mean()), float(ys.mean())  # (centroid_x, centroid_y)

    # ---- core actions -----------------------------------------------------
    def handle_click(self, position) -> None:
        """Record one row from a click at world position ``(t, y, x)``."""
        masks = self._masks
        t_idx = int(round(position[0]))
        y_idx = int(round(position[1]))
        x_idx = int(round(position[2]))
        T, H, W = masks.shape
        if not (0 <= t_idx < T and 0 <= y_idx < H and 0 <= x_idx < W):
            return self._status("click outside image — ignored")

        label = int(masks[t_idx, y_idx, x_idx])
        if label == 0:
            return self._status("clicked background (label 0) — ignored")

        cx, cy = self._label_centroid(t_idx, label)
        note = ""
        if self.region is not None and self.fov is not None:
            note = f"{self.region}/fov{self.fov}"
        self.records.append({
            "gt_cell_id": self.active_id,
            "timepoint": self.t_offset + t_idx,
            "mask_label": label,
            "centroid_x": cx,
            "centroid_y": cy,
            "event": "none",
            "parent_gt_cell_id": self._parent_of(self.active_id),
            "notes": note,
        })
        self._after_change(f"id {self.active_id}  t={self.t_offset + t_idx}  label={label}")

    def set_event(self, event: str) -> None:
        if event not in VALID_EVENTS:
            return
        if not self.records:
            return self._status("no row yet — click a cell first")
        self.records[-1]["event"] = event
        self._after_change(f"event '{event}' on last row (id {self.records[-1]['gt_cell_id']})")

    def mark_division(self) -> None:
        """Mark the clicked row as a division and spawn two daughter ids.

        The dividing cell is taken from the **last clicked row** (not the active
        spinbox), so it is always the cell you just marked. Sets that row's
        ``event=division`` (the mother ends here), creates two new
        ``gt_cell_id``s (≥ ``daughter_id_start``, so no clash with seed numbers)
        with ``parent_gt_cell_id`` = the mother, and switches the active id to
        the first daughter. The second daughter is announced; reach it with ``]``.

        **Idempotent:** if the last row is already a division it does nothing —
        a stray second press can't spawn a duplicate set of daughters. Use ``n``
        to undo the event if it was wrong.
        """
        if not self.records:
            return self._status("no row yet — click the mother cell first")
        mother = int(self.records[-1]["gt_cell_id"])
        if self.records[-1]["event"] == "division" and any(int(p) == mother for p in self._parents.values()):
            return self._status("this division already has daughters — not spawning again (press n to clear)")
        self.records[-1]["event"] = "division"
        d1 = self._next_daughter_id()
        d2 = d1 + 1
        self._parents[d1] = mother
        self._parents[d2] = mother
        self.set_active_id(d1)  # routes through the spinbox sync
        self._after_change(f"division of id {mother} → daughters {d1}, {d2}  (now annotating {d1}; press ] for {d2})")

    def clear_event(self) -> None:
        self.set_event("none")

    def undo(self) -> None:
        if not self.records:
            return self._status("nothing to undo")
        gone = self.records.pop()
        self._after_change(f"undid last row (id {gone['gt_cell_id']}, t={gone['timepoint']})")

    def set_active_id(self, gid: int) -> None:
        self.active_id = int(gid)
        # keep the panel spinbox in sync (hotkeys d/[/]/c change active_id too)
        if self._id_widget is not None:
            try:
                if int(self._id_widget.value) != self.active_id:
                    self._id_widget.value = self.active_id
            except Exception:
                pass
        self._status(f"active gt_cell_id = {self.active_id}")
        self._update_status_widget()

    def next_id(self) -> None:
        self.set_active_id(self.active_id + 1)

    def prev_id(self) -> None:
        self.set_active_id(max(1, self.active_id - 1))

    def new_id(self) -> None:
        self.set_active_id(self._next_free_id())

    # ---- parent bookkeeping ----------------------------------------------
    def _parent_of(self, gid: int):
        return self._parents.get(gid, pd.NA)

    def _next_free_id(self) -> int:
        used = {int(r["gt_cell_id"]) for r in self.records}
        used |= set(self._parents.keys())
        return (max(used) + 1) if used else 1

    def _next_daughter_id(self) -> int:
        """First free daughter id at or above ``daughter_id_start`` (and above
        any daughters already allocated), so daughters never reuse a seed id."""
        floor = self.daughter_id_start - 1
        existing = list(self._parents.keys()) + [floor]
        return max(existing) + 1

    # ---- persistence + view ----------------------------------------------
    def save(self) -> None:
        df = pd.DataFrame(self.records, columns=GT_SCHEMA_COLUMNS)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_csv, index=False)

    def _after_change(self, msg: str) -> None:
        self.save()
        self._refresh_points()
        self._status(f"{msg}   [rows: {len(self.records)}]")
        self._update_status_widget()

    def _refresh_points(self) -> None:
        try:
            if not self.records:
                self.points_layer.data = np.empty((0, 3))
                return
            df = pd.DataFrame(self.records)
            data = df[["timepoint", "centroid_y", "centroid_x"]].to_numpy(float)
            data[:, 0] -= self.t_offset  # back to window frame index for display
            tags = [f"{int(g)}{_EVENT_GLYPH.get(e, '')}" for g, e in zip(df["gt_cell_id"], df["event"])]
            colors = [_EVENT_COLOR.get(e, "#ffffff") for e in df["event"]]
            self.points_layer.data = data
            self.points_layer.features = pd.DataFrame({"tag": tags})
            self.points_layer.face_color = colors
            self.points_layer.text = {"string": "{tag}", "size": 9, "color": "white", "anchor": "upper_left"}
        except Exception as exc:  # never let a viz hiccup block the data write
            self._status(f"(points overlay refresh skipped: {exc})")

    def _status(self, msg: str) -> None:
        print(f"[gt-annotator] {msg}")
        if self._status_label is not None:
            try:
                self._status_label.value = msg
            except Exception:
                pass

    def _update_status_widget(self) -> None:
        if self._status_label is None:
            return
        par = self._parents.get(self.active_id)
        par_str = f"parent {int(par)}" if par is not None and not pd.isna(par) else "parent —"
        last = self.records[-1] if self.records else None
        tail = (f" | last: id={int(last['gt_cell_id'])} t={int(last['timepoint'])} ev={last['event']}"
                if last else "")
        try:
            self._status_label.value = f"active id: {self.active_id} ({par_str}) | rows: {len(self.records)}{tail}"
        except Exception:
            pass

    # ---- napari wiring ----------------------------------------------------
    def _wire_mouse(self) -> None:
        annotator = self

        def on_mouse(viewer, event):
            if event.button != 1:  # left button only
                return
            start = np.asarray(event.position[1:])  # (y, x) at press
            dragged = False
            yield
            while event.type == "mouse_move":
                if np.hypot(*(np.asarray(event.position[1:]) - start)) > _CLICK_MOVE_TOL:
                    dragged = True
                yield
            if not dragged:
                annotator.handle_click(event.position)

        self.viewer.mouse_drag_callbacks.append(on_mouse)

    def _wire_keys(self) -> None:
        v = self.viewer
        v.bind_key("d", lambda _v: self.mark_division(), overwrite=True)
        v.bind_key("x", lambda _v: self.set_event("death"), overwrite=True)
        v.bind_key("l", lambda _v: self.set_event("leaves_fov"), overwrite=True)
        v.bind_key("n", lambda _v: self.clear_event(), overwrite=True)
        v.bind_key("u", lambda _v: self.undo(), overwrite=True)
        v.bind_key("s", lambda _v: self.save(), overwrite=True)
        v.bind_key("]", lambda _v: self.next_id(), overwrite=True)
        v.bind_key("[", lambda _v: self.prev_id(), overwrite=True)
        v.bind_key("c", lambda _v: self.new_id(), overwrite=True)


def attach_gt_annotator(
    viewer,
    masks_layer,
    out_csv: Path | str,
    t_offset: int = 0,
    start_id: int = 1,
    region: Optional[str] = None,
    fov: Optional[int] = None,
    daughter_id_start: int = 101,
) -> GtLineageAnnotator:
    """Build the annotator, dock its control panel, and return it.

    The panel mirrors every hotkey as a button (so the tool is usable without
    memorizing keys) and shows a live status line.
    """
    ann = GtLineageAnnotator(
        viewer, masks_layer, out_csv, t_offset=t_offset,
        start_id=start_id, region=region, fov=fov,
        daughter_id_start=daughter_id_start,
    )

    from magicgui.widgets import Container, Label, PushButton, SpinBox

    status = Label(value="")
    ann._status_label = status

    id_box = SpinBox(value=start_id, min=1, max=10_000, label="active gt_cell_id")
    id_box.changed.connect(lambda val: ann.set_active_id(int(val)))
    ann._id_widget = id_box  # set_active_id now keeps the box in sync (incl. hotkeys)

    def _btn(text, fn):
        b = PushButton(text=text)
        b.changed.connect(lambda *_: fn())
        return b

    panel = Container(widgets=[
        Label(value="Click a nucleus = record this frame. → steps a frame."),
        id_box,
        _btn("prev id  [", ann.prev_id),
        _btn("next id  ]", ann.next_id),
        _btn("new id  c", ann.new_id),
        _btn("division  d", ann.mark_division),
        _btn("death  x", lambda: ann.set_event("death")),
        _btn("leaves_fov  l", lambda: ann.set_event("leaves_fov")),
        _btn("clear event  n", ann.clear_event),
        _btn("undo last  u", ann.undo),
        _btn("save now  s", ann.save),
        status,
    ])
    ann._update_status_widget()
    viewer.window.add_dock_widget(panel.native, name="GT lineage annotator", area="right")
    return ann
