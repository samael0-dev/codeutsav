"""Uploaded still photo of a billet: OCR + dimensions through the real pipeline (image source settings)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from billetvision import inputs
from billetvision.pipeline import BilletVisionPipeline, _deep_merge, still_image_notes
from billetvision.vision.measure import _mean_width_px

SCALE = 0.5            # mm per px of the synthetic photo
W, H = 2600, 1300


def _photo(length_mm=1000.0, width_mm=130.0, ident="H123456", marker=True, x0=250, angle=3.0) -> np.ndarray:
    """Phone-style photo: textured, unevenly lit table, tilted steel bar, white painted ID, 50 mm ArUco."""
    rng = np.random.default_rng(1)
    img = np.full((H, W, 3), 200, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 18, (H, W)).astype(np.float32), (0, 0), 6)[..., None]
    img += np.linspace(-15, 15, W, dtype=np.float32)[None, :, None]
    length, width = length_mm / SCALE, width_mm / SCALE
    cx, cy = x0 + length / 2, H / 2 + 80
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [cv2.boxPoints(((cx, cy), (length, width), angle)).astype(np.int32)], 255)
    steel = np.full((H, W, 3), (95, 92, 90), np.float32) + rng.normal(0, 8, (H, W, 3)).astype(np.float32)
    img = np.where((mask > 0)[..., None], steel, img)
    text = np.zeros((H, W), np.uint8)
    cv2.putText(text, f"HEAT: {ident}", (x0 + 250, int(cy + 25)), cv2.FONT_HERSHEY_DUPLEX, 2.6, 255, 6, cv2.LINE_AA)
    text = cv2.warpAffine(text, cv2.getRotationMatrix2D((cx, cy), -angle, 1.0), (W, H)) & mask
    img = np.where((text > 128)[..., None], np.float32(235), img)
    if marker:
        side, q = int(50 / SCALE), 20
        tile = np.full((side + 2 * q, side + 2 * q), 255, np.uint8)
        tile[q:q + side, q:q + side] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side)
        img[40:40 + tile.shape[0], 60:60 + tile.shape[1]] = tile[..., None]
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _analyse(tmp_path: Path, photo: np.ndarray) -> list:
    """Run ``photo`` exactly as an image upload would (``inputs.image_overrides``)."""
    h, w = photo.shape[:2]
    ov = _deep_merge(inputs.image_overrides(tmp_path / "p.jpg", {"width": w, "height": h}), {
        "vision": {"marker_size_mm": 50.0},
        "logging": {"db_path": str(tmp_path / "b.db"), "csv_path": str(tmp_path / "b.csv"),
                    "xlsx_path": str(tmp_path / "b.xlsx"), "snapshot_dir": str(tmp_path / "snap")},
    })
    pipe = BilletVisionPipeline("config/config.yaml", overrides=ov)
    return pipe.run_offline([photo] * inputs.IMAGE_REPEAT, fps=15.0)


def test_photo_reads_id_and_dimensions_within_one_percent(tmp_path):
    rows = _analyse(tmp_path, _photo())
    assert len(rows) == 1
    r = rows[0]
    assert r["billet_id"] == "H123456"
    assert abs(r["length_mm"] - 1000.0) / 1000.0 < 0.01
    assert abs(r["width_mm"] - 130.0) / 130.0 < 0.01
    assert r["status"] == "PASS", r["fail_reasons"]


def test_photo_out_of_tolerance_fails_with_reason(tmp_path):
    rows = _analyse(tmp_path, _photo(width_mm=134.0, ident="H654321"))
    assert len(rows) == 1 and rows[0]["billet_id"] == "H654321"
    assert abs(rows[0]["width_mm"] - 134.0) / 134.0 < 0.01
    assert rows[0]["status"] == "FAIL" and "width" in rows[0]["fail_reasons"]


def test_photo_billet_touching_edge_is_logged_as_review(tmp_path):
    rows = _analyse(tmp_path, _photo(length_mm=1400.0, x0=600, ident="H777888"))
    assert len(rows) == 1                                   # previously dropped silently
    assert rows[0]["status"] == "REVIEW"
    assert "edge of the photo" in rows[0]["fail_reasons"]
    assert rows[0]["billet_id"] == "H777888"


def test_photo_without_marker_is_review_not_fail(tmp_path):
    rows = _analyse(tmp_path, _photo(marker=False, ident="H246810"))
    assert len(rows) == 1
    assert rows[0]["status"] == "REVIEW"
    assert "no calibration marker" in rows[0]["fail_reasons"]
    assert rows[0]["billet_id"] == "H246810"


def test_still_image_notes():
    assert still_image_notes(True, "marker", 0.5) == []
    assert len(still_image_notes(False, "stored", 0.5)) == 2
    assert "0.5000 mm/px" in still_image_notes(True, "default", 0.5)[0]


@pytest.mark.parametrize("angle", [0.0, 3.0, 12.0])
def test_mean_width_ignores_edge_bumps(angle):
    """2 px bumps on the long edges of a tilted bar barely move the mean width (minAreaRect jumps ~4 px)."""
    img = np.zeros((900, 1400), np.uint8)
    box = cv2.boxPoints(((700, 450), (1000, 130), angle))
    cv2.fillPoly(img, [box.astype(np.int32)], 255)
    drawn_width = cv2.countNonZero(img) / 1000.0           # what fillPoly really drew (~131 px)
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    for a, b in sorted(sides, key=lambda s: -np.linalg.norm(s[1] - s[0]))[:2]:   # the two long sides
        for t in np.linspace(0.05, 0.95, 15):
            p = a + t * (b - a)
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2, 255, -1)
    contour = max(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    assert abs(_mean_width_px(contour, 1000.0) - drawn_width) < 0.5
    assert min(cv2.minAreaRect(contour)[1]) + 1 - drawn_width > 2.0
