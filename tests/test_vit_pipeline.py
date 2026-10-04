"""ViT fallback wired into the pipeline: off = identical, unresolved = REVIEW, mask source reported."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from billetvision import synthetic as S
from billetvision.pipeline import BilletVisionPipeline, _deep_merge
from billetvision.vision import vit_segment as V

_COMPARED = ("billet_id", "length_mm", "width_mm", "height_mm", "status", "fail_reasons")


def _run(out: Path, vit: dict | None, monkeypatch=None, fake=None):
    extra = {"logging": {
        "db_path": str(out / "b.db"), "csv_path": str(out / "b.csv"),
        "xlsx_path": str(out / "b.xlsx"), "snapshot_dir": str(out / "snap"),
    }}
    if vit is not None:
        extra["vit_fallback"] = vit
    if fake is not None:
        monkeypatch.setattr("billetvision.pipeline.refine_with_vit", fake)
    pipe = BilletVisionPipeline("config/config.yaml", overrides=_deep_merge(S.PIPELINE_OVERRIDES, extra))
    rows = pipe.run_offline(S.render_belt_frames([S.demo_props()[0]], seed=3), fps=S.FPS)
    detail = json.loads((out / "snap" / "000001_detail.json").read_text())
    return rows, detail


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("vit_off"), vit=None)


def test_default_run_is_classical_and_unchanged(baseline):
    rows, detail = baseline
    assert len(rows) == 1 and rows[0]["status"] == "PASS"
    assert detail["mask_source"] == "classical"


def test_enabled_but_classical_is_fine_gives_identical_records(baseline, tmp_path, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("ViT must not be called for a healthy classical segmentation")

    monkeypatch.setattr(V, "make_segment_fn", lambda cfg: must_not_run)
    rows, detail = _run(tmp_path, {"enabled": True})
    assert [{k: r[k] for k in _COMPARED} for r in rows] == [{k: r[k] for k in _COMPARED} for r in baseline[0]]
    assert detail["mask_source"] == "classical"


def test_unresolved_suspect_segmentation_goes_to_review_with_reason(tmp_path, monkeypatch):
    fake = lambda *a, **k: V.VitResult("unresolved", "segmentation uncertain: contour fill ratio 0.70 < 0.85")
    rows, detail = _run(tmp_path, None, monkeypatch, fake)
    assert rows[0]["status"] == "REVIEW"
    assert "segmentation uncertain" in rows[0]["fail_reasons"]
    assert detail["status"] == "REVIEW" and detail["mask_source"] == "classical"


def test_vit_mask_source_is_reported_in_detail_json(tmp_path, monkeypatch):
    rows, detail = _run(tmp_path, None, monkeypatch, lambda *a, **k: V.VitResult("vit", "test"))
    assert rows[0]["status"] == "PASS"
    assert detail["mask_source"] == "vit"
