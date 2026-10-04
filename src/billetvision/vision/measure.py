"""Dimensional measurements: minAreaRect, ellipse fit, ovality, diagonal difference, camber.

All returned measurements are in **millimetres**.
Pixel-to-mm conversion happens exclusively via ``calibrate.px_to_mm`` — never elsewhere.

Coordinate frame: image origin (0,0) top-left, +X right, +Y down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from billetvision.vision.calibrate import px_to_mm
from billetvision.vision.defects import edge_irregularity_mm

logger = logging.getLogger(__name__)


# Contour points sit on the centres of the outermost foreground pixels, so every
# contour-derived extent is one pixel short of the object's true size (a mask
# that is N pixels wide has contour points N-1 apart).
_PIXEL_CENTRE_BIAS_PX = 1.0


@dataclass
class Measurement:
    """Geometric measurements of a single billet in millimetres."""

    shape: str = "square"  # "square" | "round"
    length_mm: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0
    diameter_mm: Optional[float] = None
    ovality: Optional[float] = None        # (max_d - min_d) / nominal_d * 100  [%]
    diag_diff_mm: Optional[float] = None   # |d1 - d2|  (rhomboidity) [mm]
    camber_mm: Optional[float] = None
    cross_section_var_mm: Optional[float] = None
    edge_irregularity_mm: Optional[float] = None
    surface_anomaly_score: float = 0.0
    defects: list[str] = field(default_factory=list)

    # Raw pixel dimensions before mm conversion (useful for debugging)
    _length_px: float = field(default=0.0, repr=False)
    _width_px: float = field(default=0.0, repr=False)
    _height_px: float = field(default=0.0, repr=False)


def _contour_to_rect_dims(
    contour: np.ndarray,
    travel_axis: Optional[str] = None,
) -> tuple[float, float, float]:
    """Return (length_px, width_px, angle_deg) from minAreaRect.

    Args:
        contour: OpenCV contour array (N,1,2).
        travel_axis: ``"x"`` or ``"y"`` if the conveyor runs along that image
            axis.  Length is then the rectangle side along the belt and width
            the side across it, even for short or clipped pieces where the
            across-belt side is the longer one.  None = length is the longer side.

    Returns:
        Tuple of (length_px, width_px, angle_deg), each side corrected for the
        pixel-centre bias.
    """
    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), angle = rect
    if travel_axis in ("x", "y"):
        box = cv2.boxPoints(rect)
        edge = box[2] - box[1]  # direction of the side whose length is rw
        axis = 0 if travel_axis == "x" else 1
        along_is_rw = abs(edge[axis]) >= abs(edge[1 - axis])
        length, width = (rw, rh) if along_is_rw else (rh, rw)
    else:
        length, width = max(rw, rh), min(rw, rh)
    return (
        float(length) + _PIXEL_CENTRE_BIAS_PX,
        float(width) + _PIXEL_CENTRE_BIAS_PX,
        float(angle),
    )


def _mean_width_px(contour: np.ndarray, length_px: float) -> float:
    """Mean width (px) = filled pixel area / length (px), in image pixel coordinates.

    minAreaRect is an envelope: on a tilted or slightly ragged edge (photo noise,
    JPEG, staircase pixels) every 1 px bump on either side widens it.  The area
    of a rectangle divided by its length is its exact width and barely moves
    with such bumps, so the smaller of the two is used.
    """
    if length_px <= 0:
        return float("inf")
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x, y]]], dtype=contour.dtype)], -1, 255, cv2.FILLED)
    return cv2.countNonZero(mask) / float(length_px)


def _diagonal_diff_px(contour: np.ndarray) -> float:
    """Compute diagonal difference (rhomboidity) from the minAreaRect corners.

    Returns:
        |d1 - d2| in pixels, where d1/d2 are the two diagonals.
    """
    box = cv2.boxPoints(cv2.minAreaRect(contour))  # shape (4,2)
    d1 = float(np.linalg.norm(box[0] - box[2]))
    d2 = float(np.linalg.norm(box[1] - box[3]))
    return abs(d1 - d2)


def _camber_px(contour: np.ndarray, n_slices: int = 20) -> float:
    """Estimate camber (maximum deviation from the best-fit axis) in pixels.

    Samples the centroid of ``n_slices`` vertical strips and fits a line; the
    maximum residual is the camber.

    Args:
        contour: OpenCV contour.
        n_slices: Number of cross-section slices along the billet length.

    Returns:
        Camber in pixels.
    """
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 2 or bh < 2:
        return 0.0

    mask = np.zeros((y + bh + 1, x + bw + 1), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    step = bw / n_slices
    centroids: list[tuple[float, float]] = []
    for i in range(n_slices):
        x0 = int(x + i * step)
        x1 = int(x + (i + 1) * step)
        strip = mask[y : y + bh, x0:x1]
        ys_idx, _ = np.where(strip > 0)
        if len(ys_idx) == 0:
            continue
        cx_strip = (x0 + x1) / 2.0
        cy_strip = float(np.mean(ys_idx)) + y
        centroids.append((cx_strip, cy_strip))

    if len(centroids) < 3:
        return 0.0

    pts = np.array(centroids, dtype=np.float64)
    xs, ys = pts[:, 0], pts[:, 1]
    # Fit line y = a*x + b
    coeffs = np.polyfit(xs, ys, 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    residuals = np.abs(ys - (a * xs + b))
    return float(np.max(residuals))


def _cross_section_variation_px(contour: np.ndarray, n_slices: int = 20) -> float:
    """Estimate max cross-section width variation along the length in pixels."""
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 2 or bh < 2:
        return 0.0

    mask = np.zeros((y + bh + 1, x + bw + 1), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    step = bw / n_slices
    widths: list[float] = []
    for i in range(n_slices):
        x0 = int(x + i * step)
        x1 = int(x + (i + 1) * step)
        strip = mask[y : y + bh, x0:x1]
        col_sums = np.sum(strip > 0, axis=0)
        widths.append(float(np.max(col_sums)) if col_sums.size else 0.0)

    if not widths:
        return 0.0
    return float(max(widths) - min(widths))


def measure_rect(
    contour: np.ndarray,
    mm_per_px: float,
    height_px: Optional[float] = None,
    travel_axis: Optional[str] = None,
) -> Measurement:
    """Measure a rectangular (square) billet from its contour.

    Args:
        contour: Segmented billet contour in pixel coordinates.
        mm_per_px: Calibrated millimetres-per-pixel scale factor.
        height_px: If the physical height cannot be measured from the top-view
            image (e.g. side profile is not visible), pass None and width_mm is
            used as a proxy; pass the measured height in pixels otherwise.

    Returns:
        Measurement with all rectangular fields populated.
    """
    length_px, width_px, _ = _contour_to_rect_dims(contour, travel_axis)
    width_px = min(width_px, _mean_width_px(contour, length_px))
    # If height is not observable from the top view, assume square cross-section.
    h_px = height_px if height_px is not None else width_px

    length_mm = px_to_mm(length_px, mm_per_px)
    width_mm = px_to_mm(width_px, mm_per_px)
    height_mm = px_to_mm(h_px, mm_per_px)

    diag_diff_mm = px_to_mm(_diagonal_diff_px(contour), mm_per_px)
    camber_mm = px_to_mm(_camber_px(contour), mm_per_px)
    cs_var_mm = px_to_mm(_cross_section_variation_px(contour), mm_per_px)

    return Measurement(
        shape="square",
        length_mm=round(float(length_mm), 2),
        width_mm=round(float(width_mm), 2),
        height_mm=round(float(height_mm), 2),
        diag_diff_mm=round(float(diag_diff_mm), 3),
        camber_mm=round(float(camber_mm), 3),
        cross_section_var_mm=round(float(cs_var_mm), 3),
        edge_irregularity_mm=edge_irregularity_mm(contour, mm_per_px, "square"),
        _length_px=length_px,
        _width_px=width_px,
        _height_px=h_px,
    )


def measure_round(
    contour: np.ndarray,
    mm_per_px: float,
    travel_axis: Optional[str] = None,
) -> Measurement:
    """Measure a round billet from its contour using ellipse fitting.

    Args:
        contour: Segmented billet contour in pixel coordinates.
        mm_per_px: Calibrated millimetres-per-pixel scale factor.

    Returns:
        Measurement with diameter, ovality, and round-specific fields.
    """
    # Length comes from the bounding rect long side (or belt-speed method externally)
    length_px, _, _ = _contour_to_rect_dims(contour, travel_axis)
    length_mm = px_to_mm(length_px, mm_per_px)

    # Fit ellipse — requires ≥5 points
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, _), (minor_axis, major_axis), _ = ellipse
        max_d_px = float(max(minor_axis, major_axis)) + _PIXEL_CENTRE_BIAS_PX
        min_d_px = float(min(minor_axis, major_axis)) + _PIXEL_CENTRE_BIAS_PX
    else:
        # Fallback: use enclosing circle
        _, radius = cv2.minEnclosingCircle(contour)
        max_d_px = min_d_px = float(radius * 2) + _PIXEL_CENTRE_BIAS_PX

    max_d_mm = px_to_mm(max_d_px, mm_per_px)
    min_d_mm = px_to_mm(min_d_px, mm_per_px)
    diameter_mm = (max_d_mm + min_d_mm) / 2.0
    # Ovality = (max - min) / mean * 100  [%]
    ovality = (
        ((max_d_mm - min_d_mm) / diameter_mm * 100.0) if diameter_mm > 0 else 0.0
    )

    camber_mm = px_to_mm(_camber_px(contour), mm_per_px)
    cs_var_mm = px_to_mm(_cross_section_variation_px(contour), mm_per_px)

    return Measurement(
        shape="round",
        length_mm=round(float(length_mm), 2),
        width_mm=round(float(diameter_mm), 2),
        height_mm=round(float(diameter_mm), 2),
        diameter_mm=round(float(diameter_mm), 2),
        ovality=round(float(ovality), 3),
        camber_mm=round(float(camber_mm), 3),
        cross_section_var_mm=round(float(cs_var_mm), 3),
        edge_irregularity_mm=edge_irregularity_mm(contour, mm_per_px, "round"),
        _length_px=length_px,
        _width_px=max_d_px,
        _height_px=max_d_px,
    )


def measure(
    contour: np.ndarray,
    mm_per_px: float,
    shape: str = "square",
    height_px: Optional[float] = None,
    travel_axis: Optional[str] = None,
) -> Measurement:
    """Unified entry point: dispatch to measure_rect or measure_round.

    Args:
        contour: Segmented billet contour.
        mm_per_px: Calibrated mm/px scale.
        shape: "square" or "round".
        height_px: Height in pixels (rectangular mode only).
        travel_axis: Conveyor axis in the image (``"x"``/``"y"``) or None.

    Returns:
        Measurement dataclass in millimetres.
    """
    if shape == "round":
        return measure_round(contour, mm_per_px, travel_axis=travel_axis)
    return measure_rect(contour, mm_per_px, height_px=height_px, travel_axis=travel_axis)
