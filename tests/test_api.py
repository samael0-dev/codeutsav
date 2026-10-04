"""Tests for FastAPI endpoints and the pipeline support classes.

The pipeline's start/stop (which opens video files and launches threads)
is mocked out so tests are fast and offline-safe.
"""
from __future__ import annotations

import json
import sqlite3
import unittest.mock
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Patch pipeline.start / stop before importing the app so the lifespan
# does not try to open a camera or video file.
# ---------------------------------------------------------------------------

with unittest.mock.patch(
    "billetvision.pipeline.BilletVisionPipeline.start"
), unittest.mock.patch(
    "billetvision.pipeline.BilletVisionPipeline.stop"
):
    from billetvision.api.main import app
    from billetvision.pipeline import pipeline, PipelineStats

from billetvision.logging_.db import (
    InspectionRecord,
    init_db,
    insert_record,
    open_writer_connection,
    commit,
)
from billetvision.alerts.manager import Alert


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan (start/stop) mocked out."""
    with unittest.mock.patch.object(pipeline, "start"), \
         unittest.mock.patch.object(pipeline, "stop"):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def _insert_n(db: Path, n: int) -> None:
    conn = open_writer_connection(db)
    for i in range(n):
        insert_record(conn, InspectionRecord.make(
            billet_seq=i,
            billet_id=f"A{i:06d}",
            status="PASS" if i % 3 != 0 else "FAIL",
        ))
    commit(conn)
    conn.close()


# ---------------------------------------------------------------------------
# PipelineStats (unit, no HTTP)
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_snapshot_defaults(self):
        s = PipelineStats()
        snap = s.snapshot()
        assert snap["fps"] == 0.0
        assert snap["total"] == 0
        assert snap["pass_rate"] == pytest.approx(0.0, abs=1e-3)

    def test_tick_frame_updates_latency(self):
        s = PipelineStats()
        s.tick_frame(42.5)
        assert s.snapshot()["latency_ms"] == pytest.approx(42.5)

    def test_add_result_counts(self):
        s = PipelineStats()
        s.add_result("PASS", True)
        s.add_result("FAIL", False)
        snap = s.snapshot()
        assert snap["total"] == 2
        assert snap["by_status"]["PASS"] == 1
        assert snap["by_status"]["FAIL"] == 1
        assert snap["ocr_rate"] == pytest.approx(0.5)

    def test_pass_rate(self):
        s = PipelineStats()
        for _ in range(3):
            s.add_result("PASS", True)
        s.add_result("FAIL", False)
        snap = s.snapshot()
        assert snap["pass_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_dict(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        body = r.json()
        assert "fps" in body
        assert "total" in body
        assert "pass_rate" in body


# ---------------------------------------------------------------------------
# /api/log
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_default(self, client, tmp_db, tmp_path):
        _insert_n(tmp_db, 5)
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.get("/api/log")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 5

    def test_log_n_param(self, client, tmp_db):
        _insert_n(tmp_db, 20)
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.get("/api/log?n=3")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_log_status_filter(self, client, tmp_db):
        _insert_n(tmp_db, 9)  # every 3rd is FAIL
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.get("/api/log?status=FAIL")
        assert r.status_code == 200
        rows = r.json()
        assert all(row["status"] == "FAIL" for row in rows)

    def test_log_count(self, client, tmp_db):
        _insert_n(tmp_db, 7)
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.get("/api/log/count")
        assert r.status_code == 200
        assert r.json()["count"] == 7


# ---------------------------------------------------------------------------
# /api/export
# ---------------------------------------------------------------------------

class TestExport:
    def test_csv_missing_returns_404(self, client, tmp_path):
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"csv_path": str(tmp_path / "nonexistent.csv")},
        ):
            r = client.get("/api/export/csv")
        assert r.status_code == 404

    def test_xlsx_missing_returns_404(self, client, tmp_path):
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"xlsx_path": str(tmp_path / "nonexistent.xlsx")},
        ):
            r = client.get("/api/export/xlsx")
        assert r.status_code == 404

    def test_csv_present_returns_file(self, client, tmp_path):
        csv = tmp_path / "log.csv"
        csv.write_text("timestamp,billet_seq\n2026-01-01,1\n")
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"csv_path": str(csv)},
        ):
            r = client.get("/api/export/csv")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]

    def test_xlsx_present_returns_file(self, client, tmp_path):
        import pandas as pd
        xlsx = tmp_path / "log.xlsx"
        pd.DataFrame({"a": [1]}).to_excel(str(xlsx), index=False)
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"xlsx_path": str(xlsx)},
        ):
            r = client.get("/api/export/xlsx")
        assert r.status_code == 200
        ct = r.headers["content-type"]
        assert "spreadsheet" in ct or "officedocument" in ct


# ---------------------------------------------------------------------------
# /api/tolerances
# ---------------------------------------------------------------------------

class TestTolerances:
    def test_get_all(self, client):
        with unittest.mock.patch.object(
            pipeline,
            "get_tolerances",
            return_value={"square_130": {"width_nominal_mm": 130.0}},
        ):
            r = client.get("/api/tolerances")
        assert r.status_code == 200
        assert "square_130" in r.json()

    def test_get_profile(self, client):
        with unittest.mock.patch.object(
            pipeline,
            "get_tolerances",
            return_value={"square_130": {"width_nominal_mm": 130.0}},
        ):
            r = client.get("/api/tolerances/square_130")
        assert r.status_code == 200
        assert r.json()["width_nominal_mm"] == 130.0

    def test_get_missing_profile_404(self, client):
        with unittest.mock.patch.object(pipeline, "get_tolerances", return_value={}):
            r = client.get("/api/tolerances/nonexistent")
        assert r.status_code == 404

    def test_put_tolerances(self, client):
        updated = {"width_nominal_mm": 135.0, "width_tol_mm": 1.5}
        with unittest.mock.patch.object(
            pipeline, "update_profile_tolerances", return_value=updated
        ):
            r = client.put(
                "/api/tolerances/square_130",
                json={"updates": {"width_nominal_mm": 135.0}},
            )
        assert r.status_code == 200
        assert r.json()["width_nominal_mm"] == 135.0

    def test_put_missing_profile_404(self, client):
        with unittest.mock.patch.object(
            pipeline, "update_profile_tolerances", side_effect=KeyError("nope")
        ):
            r = client.put("/api/tolerances/nope", json={"updates": {}})
        assert r.status_code == 404

    def test_activate_profile(self, client):
        with unittest.mock.patch.object(pipeline, "set_active_profile"):
            r = client.put("/api/tolerances/square_130/activate")
        assert r.status_code == 200
        assert r.json()["active_profile"] == "square_130"

    def test_activate_missing_404(self, client):
        with unittest.mock.patch.object(
            pipeline, "set_active_profile", side_effect=KeyError("nope")
        ):
            r = client.put("/api/tolerances/nope/activate")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/review
# ---------------------------------------------------------------------------

class TestReview:
    def test_approve_review(self, client, tmp_db):
        _insert_n(tmp_db, 3)
        # Force one record to REVIEW
        with sqlite3.connect(str(tmp_db)) as conn:
            conn.execute("UPDATE records SET status='REVIEW' WHERE billet_seq=1")
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.post("/api/review/1", json={"action": "approve"})
        assert r.status_code == 200
        assert r.json()["new_status"] == "PASS"

    def test_reject_review(self, client, tmp_db):
        _insert_n(tmp_db, 3)
        with sqlite3.connect(str(tmp_db)) as conn:
            conn.execute("UPDATE records SET status='REVIEW' WHERE billet_seq=2")
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.post(
                "/api/review/2",
                json={"action": "reject", "notes": "visually bent"},
            )
        assert r.status_code == 200
        assert r.json()["new_status"] == "FAIL"

    def test_invalid_action_400(self, client, tmp_db):
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.post("/api/review/1", json={"action": "skip"})
        assert r.status_code == 400

    def test_missing_seq_404(self, client, tmp_db):
        init_db(tmp_db)  # empty
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.post("/api/review/9999", json={"action": "approve"})
        assert r.status_code == 404

    def test_pending_reviews(self, client, tmp_db):
        _insert_n(tmp_db, 5)
        with sqlite3.connect(str(tmp_db)) as conn:
            conn.execute("UPDATE records SET status='REVIEW' WHERE billet_seq IN (0,2)")
        with unittest.mock.patch(
            "billetvision.api.main._log_cfg",
            return_value={"db_path": str(tmp_db)},
        ):
            r = client.get("/api/review/pending")
        assert r.status_code == 200
        assert len(r.json()) == 2


# ---------------------------------------------------------------------------
# /api/alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def _seed_alerts(self):
        pipeline.alert_manager.history.clear()
        pipeline.alert_manager.active_alert = None
        pipeline.alert_manager.trigger("A000001", "FAIL", ["width out of range"])
        pipeline.alert_manager.trigger("A000002", "REVIEW", ["OCR uncertain"])

    def test_get_alerts(self, client):
        self._seed_alerts()
        r = client.get("/api/alerts")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_active_alert(self, client):
        self._seed_alerts()
        r = client.get("/api/alerts/active")
        assert r.status_code == 200
        body = r.json()
        assert body["billet_id"] == "A000002"

    def test_clear_active_alert(self, client):
        self._seed_alerts()
        r = client.post("/api/alerts/clear")
        assert r.status_code == 200
        assert pipeline.alert_manager.active_alert is None

    def test_clear_alert_history(self, client):
        self._seed_alerts()
        r = client.post("/api/alerts/clear?history=true")
        assert r.status_code == 200
        assert pipeline.alert_manager.active_alert is None
        assert client.get("/api/alerts").json() == []

    def test_no_active_alert_returns_null(self, client):
        pipeline.alert_manager.active_alert = None
        r = client.get("/api/alerts/active")
        assert r.status_code == 200
        assert r.json() is None
