"""Drawing helpers: live-feed overlay (FR-16) and per-billet snapshot card.

All colours are BGR.  Pure functions — they copy their input and never touch
disk or global state.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from billetvision.vision.measure import Measurement

STATUS_COLOR = {
    "PASS": (0, 200, 0),
    "FAIL": (0, 0, 220),
    "REWORK": (0, 140, 255),
    "REVIEW": (0, 215, 255),
}
IDLE_COLOR = (200, 200, 200)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img: np.ndarray, text: str, org: tuple, color: tuple, scale: float = 0.6) -> None:
    """Draw ``text`` with a dark backing box so it reads on any background."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, 2)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - base - 2))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y - 2), _FONT, scale, color, 2, cv2.LINE_AA)


def dims_text(meas: Measurement) -> str:
    """Compact measurement string in mm, e.g. ``L 323.0  W 133.1  H 133.1 mm``."""
    if meas.shape == "round" and meas.diameter_mm is not None:
        ov = f"  ov {meas.ovality:.2f}%" if meas.ovality is not None else ""
        return f"L {meas.length_mm:.1f}  D {meas.diameter_mm:.1f}{ov} mm"
    return f"L {meas.length_mm:.1f}  W {meas.width_mm:.1f}  H {meas.height_mm:.1f} mm"


def draw_live_overlay(
    frame: np.ndarray,
    roi_box: Sequence[int],
    contour: Optional[np.ndarray],
    meas: Optional[Measurement],
    provisional_status: Optional[str],
    last_result: Optional[dict],
    stats_line: str,
    camera_ok: bool = True,
) -> np.ndarray:
    """Return an annotated copy of ``frame`` for the MJPEG stream.

    Args:
        frame: BGR frame.
        roi_box: ``(x1, y1, x2, y2)`` inspection ROI in pixels.
        contour: Current billet contour (full-frame px) or None.
        meas: Current per-frame measurement (mm) or None.
        provisional_status: Live verdict from the current frame (colours the box).
        last_result: ``{"billet_id","status","seq"}`` of the last finalised billet.
        stats_line: FPS/latency text for the top-left corner.
        camera_ok: False draws a CAMERA LOST banner.
    """
    out = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in roi_box[:4])
    cv2.rectangle(out, (x1, y1), (x2, y2), (90, 90, 90), 1)

    if contour is not None:
        color = STATUS_COLOR.get(provisional_status or "", IDLE_COLOR)
        cv2.drawContours(out, [contour], -1, color, 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        if meas is not None:
            _label(out, dims_text(meas), (x, y - 6), color)

    _label(out, stats_line, (10, 28), IDLE_COLOR, 0.55)

    if last_result:
        color = STATUS_COLOR.get(last_result.get("status", ""), IDLE_COLOR)
        text = f"#{last_result.get('seq', '?')}  {last_result.get('billet_id', '?')}  {last_result.get('status', '')}"
        _label(out, text, (out.shape[1] - 360, 28), color, 0.7)

    if not camera_ok:
        _label(out, "CAMERA LOST - reconnecting", (out.shape[1] // 2 - 200, out.shape[0] // 2), (0, 0, 255), 0.9)
    return out


def draw_billet_card(
    color_crop: np.ndarray,
    contour_local: Optional[np.ndarray],
    meas: Measurement,
    status: str,
    billet_id: str,
    reasons: Sequence[str],
    mask_source: str = "classical",
) -> np.ndarray:
    """Annotated snapshot of one billet: outline, mm values, ID, verdict, reasons, mask source."""
    color = STATUS_COLOR.get(status, IDLE_COLOR)
    img = color_crop.copy()
    if contour_local is not None:
        cv2.drawContours(img, [contour_local], -1, color, 2)
    banner = [f"{billet_id}  {status}  [mask: {mask_source}]", dims_text(meas)]
    banner += [r[:90] for r in list(reasons)[:3]]
    pad = 22 * len(banner) + 8
    canvas = np.full((img.shape[0] + pad, max(img.shape[1], 520), 3), 20, dtype=np.uint8)
    canvas[pad:pad + img.shape[0], : img.shape[1]] = img
    for i, line in enumerate(banner):
        cv2.putText(canvas, line, (8, 20 + 22 * i), _FONT, 0.55, color if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return canvas
