"""Upload / simulated-demo input sources: FrameSource still images, validation, API and an end-to-end run."""
from __future__ import annotations

import time
import unittest.mock
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from billetvision import inputs
from billetvision import synthetic as S
from billetvision.capture.frame_source import FrameSource
from billetvision.pipeline import BilletVisionPipeline, _deep_merge

with unittest.mock.patch("billetvision.pipeline.BilletVisionPipeline.start"), \
        unittest.mock.patch("billetvision.pipeline.BilletVisionPipeline.stop"):
    from billetvision.api.main import app
    from billetvision.pipeline import pipeline


def _still(width_mm: float = 130.0, length_mm: float = 300.0) -> np.ndarray:
    """One frame: calibration marker plus a labelled billet fully inside the frame."""
    rng = np.random.default_rng(1)
    img = S.background(rng)
    spec = S.PropSpec("T", "square", length_mm, width_mm, width_mm, None, "H123456", "PASS")
    lp, wp = length_mm / S.SCALE_MM_PER_PX, width_mm / S.SCALE_MM_PER_PX
    tex = S.billet_texture(spec, int(lp) + 2, int(wp) + 2, rng)
    cy = (S.ROI_BOX[1] + S.ROI_BOX[3]) / 2
    S.draw_bar(img, tex, 300.0, 300.0 + lp, cy - wp / 2, cy + wp / 2)
    return img


def test_frame_source_single_image_repeats_then_ends(tmp_path):
    path = tmp_path / "one.png"
    cv2.imwrite(str(path), _still())
    src = FrameSource(source=str(path), loop=False, fps=200, repeat=5)
    assert src.mode == "folder"
    assert len(list(src)) == 5


def test_safe_upload_path_sanitises_and_rejects(tmp_path):
    p = inputs.safe_upload_path("../../evil name?.MP4", "video", tmp_path)
    assert p.parent == tmp_path and p.suffix.lower() == ".mp4" and ".." not in p.name and " " not in p.name
    with pytest.raises(inputs.UploadError):
        inputs.safe_upload_path("notes.txt", "video", tmp_path)
    with pytest.raises(inputs.UploadError):
        inputs.safe_upload_path("clip.mp4", "image", tmp_path)


def test_validators_reject_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(inputs.UploadError):
        inputs.validate_video(bad)
    bad_img = tmp_path / "bad.png"
    bad_img.write_bytes(b"not an image")
    with pytest.raises(inputs.UploadError):
        inputs.validate_image(bad_img)


def test_large_image_is_downscaled_and_roi_scales_with_frame(tmp_path):
    path = tmp_path / "big.png"
    cv2.imwrite(str(path), np.zeros((4800, 6400, 3), np.uint8))
    meta = inputs.validate_image(path)
    assert max(meta["width"], meta["height"]) == inputs._MAX_IMAGE_SIDE
    ov = inputs.image_overrides(path, meta)
    x1, y1, x2, y2 = ov["vision"]["roi_box"]
    assert 0 < x1 < x2 <= meta["width"] and 0 < y1 < y2 <= meta["height"]
    assert ov["capture"]["loop"] is False and ov["capture"]["repeat"] == inputs.IMAGE_REPEAT


def test_demo_overrides_loop_with_demo_scene_geometry():
    ov = inputs.demo_overrides(Path("x.mp4"))
    assert ov["capture"]["loop"] is True
    assert ov["vision"]["length_mode"] == "direct" and ov["system"]["batch_prefix"] == "DEMO"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "UPLOAD_DIR", tmp_path / "uploads")
    with unittest.mock.patch.object(pipeline, "start"), unittest.mock.patch.object(pipeline, "stop"):
        with TestClient(app) as c:
            yield c


def test_api_rejects_bad_extension_and_corrupt_file(client):
    assert client.post("/api/source/video?filename=a.txt", content=b"x").status_code == 415
    r = client.post("/api/source/video?filename=a.mp4", content=b"not a video")
    assert r.status_code == 422
    assert not list((inputs.UPLOAD_DIR).glob("*"))          # rejected files are cleaned up
    assert client.post("/api/source/image?filename=a.png", content=b"").status_code == 400


def test_api_image_upload_switches_source(client):
    ok, buf = cv2.imencode(".png", _still())
    with unittest.mock.patch.object(pipeline, "switch_source") as sw:
        r = client.post("/api/source/image?filename=billet.png", content=buf.tobytes())
    assert r.status_code == 200, r.text
    overrides, info = sw.call_args.args
    assert info.kind == "image" and info.label == "billet.png"
    assert Path(overrides["capture"]["source"]).is_file()


def test_api_upload_size_limit(client, monkeypatch):
    monkeypatch.setitem(inputs.MAX_UPLOAD_BYTES, "image", 10)
    r = client.post("/api/source/image?filename=a.png", content=b"x" * 100)
    assert r.status_code == 413
    assert not list(inputs.UPLOAD_DIR.glob("*"))


def test_switch_failure_restores_previous_source(tmp_path):
    pipe = BilletVisionPipeline("config/config.yaml", overrides={"marker": "old"})
    starts = []

    def fake_start(*a, **k):
        starts.append(dict(pipe.overrides))
        if len(starts) == 1:
            raise RuntimeError("boom")

    with unittest.mock.patch.object(pipe, "start", fake_start), unittest.mock.patch.object(pipe, "stop"):
        with pytest.raises(RuntimeError):
            pipe.switch_source({"marker": "new"}, inputs.SourceInfo("image", "x"))
    assert starts == [{"marker": "new"}, {"marker": "old"}] and pipe.overrides == {"marker": "old"}


def test_uploaded_image_end_to_end_produces_one_record(tmp_path):
    """Real pipeline: a still billet image is measured, ID-read, verdict-ed and logged once."""
    path = tmp_path / "billet.png"
    cv2.imwrite(str(path), _still(width_mm=130.0))
    meta = inputs.validate_image(path)
    ov = _deep_merge(inputs.image_overrides(path, meta), {"logging": {
        "db_path": str(tmp_path / "b.db"), "csv_path": str(tmp_path / "b.csv"),
        "xlsx_path": str(tmp_path / "b.xlsx"), "snapshot_dir": str(tmp_path / "snap"),
    }})
    pipe = BilletVisionPipeline("config/config.yaml")
    pipe.switch_source(ov, inputs.SourceInfo("image", "billet.png"))
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not (pipe.ended and pipe.stats.total >= 1):
            time.sleep(0.2)
        assert pipe.ended and pipe.stats.total == 1
        status = pipe.get_status()
        assert status["scale_source"] == "marker"
        assert abs(status["mm_per_px"] - S.SCALE_MM_PER_PX) < 0.01
    finally:
        pipe.stop()
    from billetvision.logging_.db import fetch_recent
    (rec,) = fetch_recent(str(tmp_path / "b.db"), n=None)
    assert rec["batch_id"].startswith("UPLOAD-")
    assert abs(rec["width_mm"] - 130.0) / 130.0 < 0.01
    assert abs(rec["length_mm"] - 300.0) / 300.0 < 0.01


def test_demo_scene_billets_fit_the_view_with_clear_gaps():
    sc = S.DEMO_SCENE
    length_px = 1000.0 / sc.scale
    x1, y1, x2, y2 = sc.roi
    assert length_px < (x2 - x1) - 60            # whole billet inside the ROI -> measured directly
    assert 130.0 / sc.scale < 0.25 * S.FRAME_H   # billet is a modest part of the view, not a wall
    assert sc.gap_mm / sc.scale > 300            # visibly separate pieces
    assert sc.overrides()["vision"]["length_mode"] == "direct"
    assert inputs.demo_overrides(Path("x.mp4"))["vision"]["marker_size_mm"] == sc.marker_mm


def test_demo_scene_measures_within_one_percent(tmp_path):
    prop = S.demo_props()[3]                     # PROP-04: 131.8 mm wide, 1.8 mm over nominal
    ov = _deep_merge(S.DEMO_SCENE.overrides(), {"logging": {
        "db_path": str(tmp_path / "b.db"), "csv_path": str(tmp_path / "b.csv"),
        "xlsx_path": str(tmp_path / "b.xlsx"), "snapshot_dir": str(tmp_path / "s")}})
    pipe = BilletVisionPipeline("config/config.yaml", overrides=ov)
    (rec,) = pipe.run_offline(S.render_belt_frames([prop], scene=S.DEMO_SCENE), fps=S.FPS)
    assert abs(rec["width_mm"] - prop.width_mm) / prop.width_mm < 0.01
    assert abs(rec["length_mm"] - prop.length_mm) / prop.length_mm < 0.01
    assert rec["billet_id"] == prop.heat_id
    assert rec["status"] == "REWORK"             # 1.8 mm deviation is inside the 2x-tolerance rework band


def test_demo_lineup_has_pass_and_fail_with_reasons(tmp_path):
    """Default tolerances (130 +/-1, 1000 +/-10): in-spec pieces PASS, the three planted ones FAIL with reasons."""
    props = S.show_props()
    pick = [props[0], props[3], props[5], props[7]]       # PASS, wide, narrow, short
    ov = _deep_merge(S.DEMO_SCENE.overrides(), {"logging": {
        "db_path": str(tmp_path / "b.db"), "csv_path": str(tmp_path / "b.csv"),
        "xlsx_path": str(tmp_path / "b.xlsx"), "snapshot_dir": str(tmp_path / "s")}})
    rows = BilletVisionPipeline("config/config.yaml", overrides=ov).run_offline(
        S.render_belt_frames(pick, scene=S.DEMO_SCENE), fps=S.FPS)
    assert [r["status"] for r in rows] == ["PASS", "FAIL", "FAIL", "FAIL"]
    assert rows[0]["fail_reasons"] == ""
    assert "width" in rows[1]["fail_reasons"] and "130.00 ± 1.00" in rows[1]["fail_reasons"]
    assert "width" in rows[2]["fail_reasons"]
    assert "length" in rows[3]["fail_reasons"] and "1000.00 ± 10.00" in rows[3]["fail_reasons"]


def test_source_status_has_boot_id_so_open_pages_detect_a_restart(client):
    a = client.get("/api/source").json()["boot_id"]
    assert a and a == client.get("/api/source").json()["boot_id"]


def test_demo_replays_ids_so_duplicate_check_is_off():
    ov = inputs.demo_overrides(Path("x.mp4"))
    assert ov["system"]["detect_duplicates"] is False
    assert "detect_duplicates" not in inputs.image_overrides(Path("x.png"), {"width": 640, "height": 480})["system"]


def test_alert_reset_clears_history_and_active():
    from billetvision.alerts.manager import AlertManager
    am = AlertManager()
    am.trigger("H1", "FAIL", ["width too big"])
    assert am.history and am.active_alert
    am.reset()
    assert am.history == [] and am.active_alert is None


def _stamp_clusters(tex) -> int:
    """Number of separate dark-text clusters along the bar (columns containing text pixels)."""
    dark = (tex.min(axis=2) < 70).any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, dark, False].astype(np.int8)))
    runs = list(zip(edges[::2], edges[1::2]))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < 80:     # glyph gaps inside one label
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return len(merged)


def test_realistic_billet_is_stamped_once_and_plain_twice():
    spec = S.show_props()[0]
    rng = np.random.default_rng(1)
    plain = S.billet_texture(spec, 1113, 146, rng, look="plain")
    real = S.billet_texture(spec, 1113, 146, np.random.default_rng(1), look="realistic", tone=-10, stamp_frac=0.2)
    assert _stamp_clusters(plain) == 2 and _stamp_clusters(real) == 1


def test_demo_billets_differ_in_tone_position_and_size():
    rng = np.random.default_rng(7)
    variants = [S._variant(rng) for _ in range(8)]
    assert len({v["tone"] for v in variants}) > 4 and len({round(v["dy"]) for v in variants}) > 4
    assert len({round(v["stamp"], 2) for v in variants}) > 4
    widths = {p.width_mm for p in S.show_props() if p.width_mm}
    lengths = {p.length_mm for p in S.show_props()}
    assert max(widths) - min(widths) > 15 and min(lengths) < 950      # visibly thick / thin / short pieces


def test_default_scene_is_unchanged_by_the_realistic_look():
    """The accuracy-report scene keeps its exact old rendering (two stamps, no variation)."""
    assert S.DEFAULT_SCENE.look == "plain" and S.DEMO_SCENE.look == "realistic"
    a = next(S.render_belt_frames(S.demo_props()[:1], seed=3))
    b = next(S.render_belt_frames(S.demo_props()[:1], seed=3, scene=S.DEFAULT_SCENE))
    assert np.array_equal(a, b)
