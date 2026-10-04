"""ViT fallback: trigger, repair, graceful degradation.  The real model is never loaded."""
from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from billetvision.vision import vit_segment as V
from billetvision.vision.measure import measure
from billetvision.vision.track import FrameSample, TrackedBillet

TOL = {"width_nominal_mm": 130.0, "width_tol_mm": 1.0}
CROP_W, CROP_H = 500, 250
GOOD = np.array([(50, 60), (450, 60), (450, 190), (50, 190)], dtype=np.int32).reshape(-1, 1, 2)   # 400 x 130 px
BITTEN = np.array(  # same bar with a big bite out of it (classical mask broke on a glare patch)
    [(50, 60), (450, 60), (450, 190), (250, 190), (250, 120), (50, 120)], dtype=np.int32
).reshape(-1, 1, 2)


def _measure(contour: np.ndarray, mm_per_px: float = 1.0):
    return measure(contour, mm_per_px, shape="square", travel_axis="x")


def _sample(contour: np.ndarray, sharp: float) -> FrameSample:
    crop = np.full((CROP_H, CROP_W, 3), 40, dtype=np.uint8)
    return FrameSample(
        frame_gray=crop[..., 0], contour=contour, measurement=_measure(contour), sharpness_score=sharp,
        extras={"color_crop": crop}, area_px=float(cv2.contourArea(contour)),
    )


def _billet(contour: np.ndarray, n: int = 5) -> TrackedBillet:
    samples = [_sample(contour, 100.0 - i) for i in range(n)]
    return TrackedBillet(
        track_id=1, measurement=samples[0].measurement, best_frame_gray=samples[0].frame_gray,
        best_contour=contour, centroid_path=[], frame_count=n, top_samples=samples,
    )


def _mask_fn(contour: np.ndarray):
    """A fake ViT that always answers with ``contour`` filled; counts its calls."""
    calls = []

    def fn(bgr, box):
        calls.append(box)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        return mask

    fn.calls = calls
    return fn


CFG = V.VitConfig(enabled=True, max_calls_per_billet=3)


# --- config ---------------------------------------------------------------

def test_config_defaults_off_and_flattens_trigger():
    assert V.VitConfig.from_dict(None).enabled is False
    cfg = V.VitConfig.from_dict({"enabled": True, "trigger": {"min_fill_ratio": 0.9}, "bogus": 1})
    assert cfg.enabled and cfg.min_fill_ratio == 0.9


def test_repo_config_has_vit_fallback_disabled_by_default():
    import yaml

    raw = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))["vit_fallback"]
    assert V.VitConfig.from_dict(raw).enabled is False


# --- trigger --------------------------------------------------------------

def test_clean_rectangle_is_not_suspect():
    assert V.suspect_reason(GOOD, _measure(GOOD), TOL, CFG) is None


def test_bitten_contour_is_suspect_by_fill_ratio():
    assert "fill ratio" in V.suspect_reason(BITTEN, _measure(BITTEN), TOL, CFG)


def test_gross_size_error_is_suspect_but_small_deviation_is_not():
    narrow = np.array([(50, 60), (450, 60), (450, 160), (50, 160)], dtype=np.int32).reshape(-1, 1, 2)  # 100 px
    assert "off nominal" in V.suspect_reason(narrow, _measure(narrow), TOL, CFG)
    slightly_wide = np.array([(50, 60), (450, 60), (450, 194), (50, 194)], dtype=np.int32).reshape(-1, 1, 2)  # ~3 %
    assert V.suspect_reason(slightly_wide, _measure(slightly_wide), TOL, CFG) is None


# --- refine_with_vit ------------------------------------------------------

def test_disabled_flag_changes_nothing_and_never_calls_the_model():
    tb = _billet(BITTEN)
    before = (tb.measurement, list(tb.top_samples))
    fn = _mask_fn(GOOD)
    res = V.refine_with_vit(tb, V.VitConfig(enabled=False), TOL, 1.0, "square", segment_fn=fn)
    assert res.source == "classical" and res.mask_source == "classical"
    assert fn.calls == [] and (tb.measurement, tb.top_samples) == before


def test_fallback_not_triggered_when_classical_is_fine():
    tb = _billet(GOOD)
    fn = _mask_fn(GOOD)
    res = V.refine_with_vit(tb, CFG, TOL, 1.0, "square", segment_fn=fn)
    assert res.source == "classical" and fn.calls == []


def test_fallback_repairs_a_bad_segmentation_within_one_percent():
    tb = _billet(BITTEN)
    fn = _mask_fn(GOOD)
    res = V.refine_with_vit(tb, CFG, TOL, 1.0, "square", segment_fn=fn)
    assert res.source == "vit" and res.mask_source == "vit"
    assert len(fn.calls) == CFG.max_calls_per_billet          # capped, not all 5 samples
    assert abs(tb.measurement.width_mm - 130.0) / 130.0 < 0.01
    assert abs(tb.measurement.length_mm - 400.0) / 400.0 < 0.01
    assert tb.top_samples[0].extras["mask_source"] == "vit"
    assert "mask_source" not in tb.top_samples[-1].extras     # un-repaired frames stay classical


def test_unresolved_when_vit_returns_nothing_or_no_improvement_or_raises():
    for fn in (lambda bgr, box: None, _mask_fn(BITTEN)):
        tb = _billet(BITTEN)
        before = tb.measurement
        res = V.refine_with_vit(tb, CFG, TOL, 1.0, "square", segment_fn=fn)
        assert res.source == "unresolved" and "segmentation uncertain" in res.note
        assert res.mask_source == "classical" and tb.measurement is before

    def boom(bgr, box):
        raise RuntimeError("cuda out of memory")

    res = V.refine_with_vit(_billet(BITTEN), CFG, TOL, 1.0, "square", segment_fn=boom)
    assert res.source == "unresolved"


def test_zero_call_budget_is_unresolved_without_calling_the_model():
    fn = _mask_fn(GOOD)
    cfg = V.VitConfig(enabled=True, max_calls_per_billet=0)
    assert V.refine_with_vit(_billet(BITTEN), cfg, TOL, 1.0, "square", segment_fn=fn).source == "unresolved"
    assert fn.calls == []


# --- model access ---------------------------------------------------------

def test_missing_torch_degrades_to_none(monkeypatch):
    V.clear_model_cache()
    monkeypatch.setitem(sys.modules, "torch", None)       # makes `import torch` raise ImportError
    assert V.segment_vit(np.zeros((20, 20, 3), np.uint8), (1, 1, 10, 10)) is None
    V.clear_model_cache()


def test_missing_weights_degrade_to_none(monkeypatch, tmp_path):
    V.clear_model_cache()
    for name in ("torch", "mobile_sam"):                  # pretend both import fine
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    sys.modules["mobile_sam"].SamPredictor = object
    sys.modules["mobile_sam"].sam_model_registry = {}
    out = V.segment_vit(np.zeros((20, 20, 3), np.uint8), weights_path=str(tmp_path / "nope.pt"))
    assert out is None
    V.clear_model_cache()
