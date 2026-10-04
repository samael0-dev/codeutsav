"""Optional Vision-Transformer segmentation fallback (MobileSAM, a TinyViT-based SAM).

Classical thresholding stays the primary segmenter.  This module only runs on the
per-billet finalizer thread, on the tracker's sharpest frames, when the classical
contour looks wrong (not rectangular enough, or grossly the wrong size).  The ViT
produces a mask; the measurement itself is still classical (``measure()`` on the
mask's contour), so the pixel-to-mm conversion stays in ``vision/calibrate.py``.

Everything heavy (torch, mobile_sam, the weights) is imported lazily.  With the
feature disabled, or the libraries/weights missing, nothing here can raise: the
caller keeps the classical result.

Coordinate frame: image origin (0,0) top-left, +X right, +Y down.  Boxes are
``(x1, y1, x2, y2)`` pixels inside the image they are given with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import cv2
import numpy as np

from billetvision.vision.measure import measure
from billetvision.vision.track import TrackedBillet, median_measurement

logger = logging.getLogger(__name__)

# A ViT contour replaces the classical one only if it is at least this much more
# rectangle-like (fill ratio), so a ViT that merely agrees changes nothing.
_MIN_FILL_GAIN = 0.02
_BOX_PAD_FRAC = 0.15  # prompt box = classical bbox grown by this fraction per side

SegmentFn = Callable[[np.ndarray, Optional[tuple]], Optional[np.ndarray]]


@dataclass
class VitConfig:
    """``vit_fallback`` section of config.yaml."""

    enabled: bool = False
    model: str = "vit_t"                       # MobileSAM registry key
    weights_path: str = "weights/mobile_sam.pt"
    device: str = "cpu"
    max_calls_per_billet: int = 3              # best frames re-segmented per billet
    min_fill_ratio: float = 0.85               # contour area / minAreaRect area below this = suspect
    max_size_error_pct: float = 15.0           # gross width/diameter error vs nominal = suspect

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "VitConfig":
        raw = dict(raw or {})
        trigger = dict(raw.pop("trigger", None) or {})
        known = {k: v for k, v in {**raw, **trigger}.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class VitResult:
    """Outcome of ``refine_with_vit`` for one billet."""

    source: str = "classical"   # "classical" | "vit" | "unresolved" (suspect, ViT could not fix it)
    note: str = ""

    @property
    def mask_source(self) -> str:
        """Value reported in the annotated card / detail JSON."""
        return "vit" if self.source == "vit" else "classical"


# ---------------------------------------------------------------------------
# Model access (lazy, cached, never raises)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_predictor(model: str, weights_path: str, device: str) -> Optional[Any]:
    """Load a MobileSAM predictor once; a failure is cached too (None, no retry spam)."""
    try:
        import torch  # noqa: F401
        from mobile_sam import SamPredictor, sam_model_registry
    except ImportError as exc:
        logger.warning("ViT fallback unavailable (%s) - install requirements-vit.txt", exc)
        return None
    path = Path(weights_path)
    if not path.is_file():
        logger.warning("ViT weights not found at %s - download mobile_sam.pt (see requirements-vit.txt)", path)
        return None
    try:
        sam = sam_model_registry[model](checkpoint=str(path))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)
    except Exception:
        logger.exception("Could not load the ViT model %r from %s", model, path)
        return None


def segment_vit(
    bgr: np.ndarray,
    box: Optional[tuple] = None,
    *,
    model: str = "vit_t",
    weights_path: str = "weights/mobile_sam.pt",
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Segment the object inside ``box`` with MobileSAM.

    Args:
        bgr: Input BGR image.
        box: ``(x1, y1, x2, y2)`` prompt in ``bgr`` pixels; None = whole image.

    Returns:
        Binary uint8 mask (0/255), same size as ``bgr``; None if the libraries or
        weights are missing, inference fails, or the mask is empty.
    """
    predictor = _load_predictor(model, weights_path, device)
    if predictor is None:
        return None
    try:
        import torch

        h, w = bgr.shape[:2]
        prompt = np.asarray(box if box is not None else (0, 0, w - 1, h - 1), dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks, _, _ = predictor.predict(box=prompt, multimask_output=False)
        mask = (np.asarray(masks[0]) > 0).astype(np.uint8) * 255
        return mask if cv2.countNonZero(mask) else None
    except Exception:
        logger.exception("ViT inference failed")
        return None


def make_segment_fn(cfg: VitConfig) -> SegmentFn:
    """Bind ``cfg`` to ``segment_vit`` -> ``fn(bgr, box) -> mask | None``."""
    return lambda bgr, box: segment_vit(
        bgr, box, model=cfg.model, weights_path=cfg.weights_path, device=cfg.device
    )


# ---------------------------------------------------------------------------
# Trigger and repair (pure apart from calling ``segment_fn``)
# ---------------------------------------------------------------------------

def fill_ratio(contour: np.ndarray) -> float:
    """Contour area divided by its minAreaRect area (1.0 = a perfect rectangle)."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    rect_area = float(rw) * float(rh)
    return float(cv2.contourArea(contour)) / rect_area if rect_area > 0 else 0.0


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask, or None."""
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(found, key=cv2.contourArea) if found else None


def suspect_reason(contour: np.ndarray, meas: Any, tol: Mapping[str, Any], cfg: VitConfig) -> Optional[str]:
    """Why the classical segmentation looks wrong, or None if it looks fine.

    Two checks: the contour is much less rectangular than a billet should be, or
    the measured width/diameter is grossly (``max_size_error_pct``) off nominal.
    Ordinary out-of-tolerance billets (a few percent) are NOT suspect.
    """
    fill = fill_ratio(contour)
    if fill < cfg.min_fill_ratio:
        return f"contour fill ratio {fill:.2f} < {cfg.min_fill_ratio:.2f}"
    for key, value in (("width_nominal_mm", meas.width_mm), ("diameter_nominal_mm", meas.diameter_mm)):
        nominal = tol.get(key)
        if nominal and value is not None:
            err = abs(value - nominal) / nominal * 100.0
            if err > cfg.max_size_error_pct:
                return f"{key.split('_nominal')[0]} {value:.1f} mm is {err:.0f}% off nominal {nominal:.1f} mm"
    return None


def _repair_sample(sample: Any, segment_fn: SegmentFn, mm_per_px: float, shape: str) -> Optional[Any]:
    """Re-segment one sample with the ViT; return a corrected copy, or None if no better."""
    crop = sample.extras.get("color_crop")
    if crop is None:
        return None
    ch, cw = crop.shape[:2]
    x, y, w, h = cv2.boundingRect(sample.contour)
    px, py = int(_BOX_PAD_FRAC * w), int(_BOX_PAD_FRAC * h)
    box = (max(0, x - px), max(0, y - py), min(cw - 1, x + w + px), min(ch - 1, y + h + py))
    mask = segment_fn(crop, box)
    contour = largest_contour(mask) if mask is not None else None
    if contour is None or fill_ratio(contour) < fill_ratio(sample.contour) + _MIN_FILL_GAIN:
        return None
    meas = measure(contour, mm_per_px, shape=shape, travel_axis="x")
    meas.surface_anomaly_score = sample.measurement.surface_anomaly_score
    return replace(
        sample,
        contour=contour,
        measurement=meas,
        area_px=float(cv2.contourArea(contour)),
        extras={**sample.extras, "mask_source": "vit"},
    )


def refine_with_vit(
    tb: TrackedBillet,
    cfg: VitConfig,
    tol: Mapping[str, Any],
    mm_per_px: float,
    shape: str,
    segment_fn: Optional[SegmentFn] = None,
) -> VitResult:
    """Re-segment a billet's best frames with the ViT when the classical result looks wrong.

    On success ``tb.top_samples`` and ``tb.measurement`` are replaced (median over
    the repaired frames).  Otherwise ``tb`` is left untouched.  Never raises.

    Args:
        tb: Finalised billet (sharpest samples first).
        cfg: ``vit_fallback`` settings; ``enabled=False`` returns immediately.
        tol: Active tolerance profile (for the nominal size check).
        mm_per_px: Calibrated scale, forwarded to ``measure()``.
        shape: ``"square"`` or ``"round"``.
        segment_fn: ``fn(bgr, box) -> mask | None``; defaults to the MobileSAM one.
    """
    if not cfg.enabled or not tb.top_samples:
        return VitResult()
    try:
        reason = suspect_reason(tb.top_samples[0].contour, tb.measurement, tol, cfg)
        if reason is None:
            return VitResult()
        logger.info("Classical segmentation looks wrong (%s) - trying the ViT fallback", reason)
        fn = segment_fn or make_segment_fn(cfg)
        fixed: dict[int, Any] = {}
        for i, sample in enumerate(tb.top_samples[: max(cfg.max_calls_per_billet, 0)]):
            repaired = _repair_sample(sample, fn, mm_per_px, shape)
            if repaired is not None:
                fixed[i] = repaired
        if not fixed:
            return VitResult("unresolved", f"segmentation uncertain: {reason}")
        new_meas, _ = median_measurement(list(fixed.values()), best_n=len(fixed))
        if suspect_reason(next(iter(fixed.values())).contour, new_meas, tol, cfg) is not None:
            return VitResult("unresolved", f"segmentation uncertain: {reason}")
        tb.top_samples = [fixed.get(i, s) for i, s in enumerate(tb.top_samples)]
        tb.measurement = new_meas
        tb.best_contour = tb.top_samples[0].contour
        logger.info("ViT fallback fixed the segmentation on %d frame(s)", len(fixed))
        return VitResult("vit", reason)
    except Exception:
        logger.exception("ViT refinement failed - keeping the classical result")
        return VitResult("unresolved", "segmentation uncertain: ViT refinement error")


def clear_model_cache() -> None:
    """Forget the cached model (tests, or after changing weights)."""
    _load_predictor.cache_clear()
