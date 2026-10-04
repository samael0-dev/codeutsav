"""FastAPI application: MJPEG video, WebSocket events, REST for log/tolerances/export/review."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from billetvision import artifacts, inputs
from billetvision.api.mjpeg import frame_generator
from billetvision.api.ws import ws_manager
from billetvision.logging_.db import (
    count_records,
    fetch_record,
    fetch_recent,
    update_record,
)
from billetvision.ocr.validate import correct_and_validate
from billetvision.pipeline import SourceUnavailable, pipeline

logger = logging.getLogger(__name__)

_CFG_PATH = Path("config/config.yaml")
_WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
_BOOT_ID = uuid.uuid4().hex   # changes on every server start; lets an open dashboard notice a restart
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _log_cfg() -> Dict[str, Any]:
    with _CFG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("logging", {})


def _db_path() -> str:
    return _log_cfg().get("db_path", "data/outputs/billetvision.db")


def _snapshot_dir() -> Path:
    """Directory holding per-billet images/JSON (the running pipeline's, else config's)."""
    if pipeline.is_running:
        return pipeline.snapshot_dir
    return Path(_log_cfg().get("snapshot_dir", "data/outputs/snapshots"))


def _split_reasons(text: Optional[str]) -> List[str]:
    return [r.strip() for r in (text or "").split(";") if r.strip()]


def _enrich(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add dashboard conveniences to a DB row (schema columns are left untouched)."""
    out = dict(row)
    out["reasons"] = _split_reasons(row.get("fail_reasons"))
    path = row.get("image_path")
    out["image_url"] = f"/snapshots/{Path(path).name}" if path else None
    return out


# ---------------------------------------------------------------------------
# Lifespan: start/stop pipeline around the server lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    pipeline.set_event_loop(loop)
    try:
        pipeline.start(event_loop=loop)
    except Exception as exc:
        logger.error("Pipeline failed to start: %s", exc)
    yield
    pipeline.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BilletVision API", version="1.0.0", lifespan=lifespan)

if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def get_index():
    index = _WEB_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BilletVision — pipeline running</h1>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/snapshots/{name}", include_in_schema=False)
def get_snapshot(name: str):
    """Serve a per-billet artifact (annotated card, ID crop, frame, detail JSON)."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    path = _snapshot_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(str(path))


# ---------------------------------------------------------------------------
# MJPEG video stream
# ---------------------------------------------------------------------------

@app.get("/video", include_in_schema=False)
def video_feed():
    """MJPEG stream of the annotated live feed."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


@app.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """Real-time billet events and telemetry (JSON messages)."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; pipeline pushes unsolicited events
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "BilletVision",
        **pipeline.get_status(),
    }


# ---------------------------------------------------------------------------
# Input source: uploaded video / image, simulated demo, back to the camera
# ---------------------------------------------------------------------------

def _source_status() -> Dict[str, Any]:
    snap = pipeline.stats.snapshot()
    status = pipeline.get_status()
    return {
        **pipeline.source_info.as_dict(),
        "boot_id": _BOOT_ID,
        "running": pipeline.is_running,
        "ended": pipeline.ended,
        "camera_ok": pipeline.camera_ok,
        "frames_done": pipeline.frames_done,
        "results": snap["total"],
        "by_status": snap["by_status"],
        "mm_per_px": status["mm_per_px"],
        "scale_source": status["scale_source"],
    }


def _switch(overrides: Dict[str, Any], info: inputs.SourceInfo, require_open: bool = False) -> Dict[str, Any]:
    try:
        pipeline.switch_source(overrides, info, require_open=require_open)
    except SourceUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not start this source: {exc}")
    return _source_status()


async def _save_upload(request: Request, dest: Path, limit: int) -> None:
    """Stream the raw request body to ``dest`` (no multipart dependency); enforce ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=f"File too large (max {limit // (1024 * 1024)} MB)")
    size = 0
    try:
        with dest.open("wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File too large (max {limit // (1024 * 1024)} MB)")
                await run_in_threadpool(fh.write, chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty upload")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


async def _upload(request: Request, filename: str, kind: str) -> Dict[str, Any]:
    try:
        dest = inputs.safe_upload_path(filename, kind)
    except inputs.UploadError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    await _save_upload(request, dest, inputs.MAX_UPLOAD_BYTES[kind])
    try:
        if kind == "video":
            meta = await run_in_threadpool(inputs.validate_video, dest)
            overrides = inputs.video_overrides(dest, meta)
        else:
            meta = await run_in_threadpool(inputs.validate_image, dest)
            overrides = inputs.image_overrides(dest, meta)
    except inputs.UploadError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    info = inputs.SourceInfo(kind, Path(filename).name, frames_total=meta.get("frames") or (inputs.IMAGE_REPEAT if kind == "image" else None))
    return await run_in_threadpool(_switch, overrides, info)


@app.get("/api/source")
def get_source():
    """What is being analysed right now, and how far along it is."""
    return _source_status()


@app.post("/api/source/video")
async def upload_video(request: Request, filename: str = Query(default="video.mp4", max_length=200)):
    """Upload a video (raw request body) and analyse it from the start."""
    return await _upload(request, filename, "video")


@app.post("/api/source/image")
async def upload_image(request: Request, filename: str = Query(default="image.jpg", max_length=200)):
    """Upload a still image (raw request body) and analyse it."""
    return await _upload(request, filename, "image")


@app.post("/api/source/demo")
def start_demo():
    """Run the simulated conveyor (synthetic billets with known dimensions), looped."""
    try:
        path = inputs.ensure_demo_video()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not render the demo video: {exc}")
    return _switch(inputs.demo_overrides(path), inputs.SourceInfo("demo", "Simulated conveyor demo"))


@app.post("/api/source/camera")
def use_camera(index: int = Query(default=0, ge=0, le=9)):
    """Analyse live camera ``index``.  The device is opened once (by the pipeline); if it
    does not open the previous source is restored and a 422 is returned.  Clicking this
    while that camera is already streaming is a no-op; if it has been lost, it retries."""
    info = inputs.SourceInfo("camera", f"Camera {index}")
    cur = pipeline.source_info
    if pipeline.is_running and cur.kind == "camera" and cur.label == info.label and pipeline.camera_ok:
        return _source_status()
    error: Optional[HTTPException] = None
    for backend in _camera_backends():
        try:
            return _switch(inputs.camera_overrides(index, backend), info, require_open=True)
        except HTTPException as exc:
            if exc.status_code != 422:      # a real failure, not just "this backend gave nothing"
                raise
            logger.warning("Camera %d via %s backend failed: %s", index, backend or "default", exc.detail)
            error = exc
    raise error


def _camera_backends() -> List[Optional[str]]:
    """Capture backends to try in order.  On Windows Media Foundation (OpenCV's default) often
    opens a camera but never delivers frames, so DirectShow goes first."""
    return ["dshow", "msmf"] if sys.platform == "win32" else [None]


@app.post("/api/source/reset")
def reset_source():
    """Go back to the source configured in config/config.yaml (camera or demo file)."""
    return _switch({}, inputs.SourceInfo("configured", "Configured source"))


# ---------------------------------------------------------------------------
# Stats / KPI
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    """Live pipeline performance counters."""
    return pipeline.stats.snapshot()


@app.get("/api/kpi")
def get_kpi():
    """Alias for /api/stats for compatibility."""
    return pipeline.stats.snapshot()


# ---------------------------------------------------------------------------
# Log (SQLite read)
# ---------------------------------------------------------------------------

@app.get("/api/log")
def get_log(
    n: int = Query(default=50, ge=1, le=1000),
    status: Optional[str] = Query(default=None, pattern="^(PASS|FAIL|REWORK|REVIEW)$"),
    q: Optional[str] = Query(default=None, max_length=64, description="substring of billet/batch ID (heat lookup)"),
):
    """Return the most-recent ``n`` inspection records (newest first)."""
    try:
        return [_enrich(r) for r in fetch_recent(_db_path(), n=n, status=status, q=q)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/logs")
def get_logs(
    n: int = Query(default=50, ge=1, le=1000),
    status: Optional[str] = Query(default=None, pattern="^(PASS|FAIL|REWORK|REVIEW)$"),
    q: Optional[str] = Query(default=None, max_length=64),
):
    """Alias for /api/log for backwards compatibility."""
    return get_log(n=n, status=status, q=q)


@app.get("/api/log/count")
def log_count():
    return {"count": count_records(_db_path())}


# ---------------------------------------------------------------------------
# Drill-down (FR-21)
# ---------------------------------------------------------------------------

@app.get("/api/billet/{billet_seq}")
def get_billet(billet_seq: int):
    """Everything known about one billet: record, measurements, per-frame data, OCR, decision."""
    row = fetch_record(_db_path(), billet_seq)
    if row is None:
        raise HTTPException(status_code=404, detail=f"billet_seq {billet_seq} not found")
    return {"record": _enrich(row), "detail": artifacts.load_detail(_snapshot_dir(), billet_seq)}


# ---------------------------------------------------------------------------
# Export (file download)
# ---------------------------------------------------------------------------

@app.get("/api/export/csv")
def export_csv():
    """Download the full inspection log as CSV."""
    cfg = _log_cfg()
    path = Path(cfg.get("csv_path", "data/outputs/billet_log.csv"))
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV log not found")
    return FileResponse(
        str(path),
        media_type="text/csv",
        filename=path.name,
    )


@app.get("/api/export/xlsx")
def export_xlsx():
    """Download the full inspection log as Excel."""
    cfg = _log_cfg()
    path = Path(cfg.get("xlsx_path", "data/outputs/billet_log.xlsx"))
    if not path.exists():
        raise HTTPException(status_code=404, detail="XLSX log not found")
    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


@app.get("/api/export/json")
def export_json():
    """Download the complete inspection log (every day, from SQLite) as JSON."""
    try:
        rows = list(reversed(fetch_recent(_db_path(), n=None)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=json.dumps(rows, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="billet_log.json"'},
    )


@app.get("/api/export/{file_format}")
def export_log(file_format: str):
    """Download log in requested format (csv, xlsx or json)."""
    fmt = file_format.lower()
    if fmt == "csv":
        return export_csv()
    elif fmt == "xlsx":
        return export_xlsx()
    raise HTTPException(status_code=404, detail=f"Unsupported format '{file_format}'. Use 'csv', 'xlsx' or 'json'.")


# ---------------------------------------------------------------------------
# Tolerances / profiles
# ---------------------------------------------------------------------------

@app.get("/api/profiles")
def get_profiles():
    """List the configured billet profiles."""
    path = Path("config/billet_profiles.yaml")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("profiles", [])


@app.get("/api/tolerances")
def get_tolerances():
    """Return all tolerance profiles."""
    return pipeline.get_tolerances()


@app.get("/api/tolerances/{profile}")
def get_profile(profile: str):
    tols = pipeline.get_tolerances()
    if profile not in tols:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")
    return tols[profile]


class ToleranceUpdate(BaseModel):
    updates: Dict[str, Any]


def _validated_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Tolerance values must be finite, non-negative numbers (the shape is fixed per profile)."""
    clean: Dict[str, Any] = {}
    for key, value in updates.items():
        if key == "shape":
            raise HTTPException(status_code=400, detail="'shape' cannot be changed at runtime")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value != value:
            raise HTTPException(status_code=400, detail=f"'{key}' must be a non-negative number")
        clean[key] = float(value)
    return clean


@app.put("/api/tolerances/{profile}")
def update_tolerances(profile: str, body: ToleranceUpdate):
    """Patch tolerance values for a profile (applied live; restart reverts to the YAML)."""
    updates = _validated_updates(body.updates)
    try:
        return pipeline.update_profile_tolerances(profile, updates)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")


@app.post("/api/tolerances")
def update_tolerances_post(data: Dict[str, Any]):
    """Alternative POST endpoint for tolerance update."""
    profile = data.get("profile_id", "square_130")
    updates = _validated_updates(data.get("tolerances", {}))
    try:
        updated = pipeline.update_profile_tolerances(profile, updates)
        return {"status": "updated", "active_profile": profile, "tolerances": updated}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")


@app.put("/api/tolerances/{profile}/activate")
def activate_profile(profile: str):
    """Switch the active inspection profile."""
    try:
        pipeline.set_active_profile(profile)
        return {"active_profile": profile}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile}' not found")


# ---------------------------------------------------------------------------
# Calibration (FR-3)
# ---------------------------------------------------------------------------

class RecalibrateRequest(BaseModel):
    marker_size_mm: Optional[float] = None


@app.get("/api/calibration")
def get_calibration():
    """Current scale (mm/px) and calibration metadata."""
    return pipeline.get_calibration()


@app.post("/api/calibration/recalibrate")
def recalibrate(body: Optional[RecalibrateRequest] = None):
    """Re-calibrate from the ArUco marker visible in the live feed and save it."""
    size = body.marker_size_mm if body else None
    if size is not None and size <= 0:
        raise HTTPException(status_code=400, detail="marker_size_mm must be positive")
    try:
        return pipeline.recalibrate(marker_size_mm=size)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Review queue (FR-20)
# ---------------------------------------------------------------------------

class ReviewResolution(BaseModel):
    action: str          # "approve" | "reject"
    notes: Optional[str] = None
    corrected_id: Optional[str] = None   # operator-typed ID (approve)


def _apply_review(db_path: str, billet_seq: int, fields: Dict[str, Any]) -> bool:
    """Write through the pipeline's single log writer when it owns this DB, else directly."""
    writer = pipeline.writer
    if writer is not None and str(writer.db_path) == str(Path(db_path)):
        return pipeline.apply_review(billet_seq, fields)
    return update_record(db_path, billet_seq, fields)


@app.post("/api/review/{billet_seq}")
def resolve_review(billet_seq: int, body: ReviewResolution):
    """Resolve a REVIEW-status billet as PASS (approve) or FAIL (reject).

    ``corrected_id`` (approve) replaces the unconfirmed ID; it must match the
    configured heat-ID format.  Rejection notes become the fail reason.  The
    change is written to SQLite, CSV and XLSX.
    """
    if body.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    db_path = _db_path()
    row = fetch_record(db_path, billet_seq)
    if row is None:
        raise HTTPException(status_code=404, detail=f"billet_seq {billet_seq} not found")

    fields: Dict[str, Any] = {}
    if body.action == "approve":
        fields["status"] = "PASS"
        fields["fail_reasons"] = ""
        if body.corrected_id:
            corrected, matched, _ = correct_and_validate(body.corrected_id.strip().upper(), pipeline.id_regex)
            if not matched:
                raise HTTPException(
                    status_code=422,
                    detail=f"ID '{body.corrected_id}' does not match the format {pipeline.id_regex}",
                )
            fields["billet_id"] = corrected
            fields["ocr_confidence"] = 1.0
            labels = [d.strip() for d in (row.get("defects") or "").split(",") if d.strip()]
            fields["defects"] = ", ".join([*labels, "manual_id"])
    else:
        fields["status"] = "FAIL"
        fields["fail_reasons"] = body.notes or "Rejected by operator"

    if not _apply_review(db_path, billet_seq, fields):
        raise HTTPException(status_code=404, detail=f"billet_seq {billet_seq} not found")
    return {"billet_seq": billet_seq, "new_status": fields["status"], "billet_id": fields.get("billet_id", row["billet_id"])}


@app.get("/api/review/pending")
def pending_reviews(n: int = Query(default=50, ge=1, le=500)):
    """List records still in REVIEW status (with ID-crop URL for the operator)."""
    out = []
    for row in fetch_recent(_db_path(), n=n, status="REVIEW"):
        item = _enrich(row)
        crop = artifacts.artifact_name(row["billet_seq"], "_id.png")
        item["id_crop_url"] = f"/snapshots/{crop}" if (_snapshot_dir() / crop).is_file() else None
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def get_alerts(n: int = Query(default=50, ge=1, le=500)):
    """Return the last ``n`` alerts (newest first)."""
    history = pipeline.alert_manager.history[-n:]
    return [dataclasses.asdict(a) for a in reversed(history)]


@app.post("/api/alerts/clear")
def clear_active_alert(history: bool = Query(default=False)):
    """Dismiss the active alert banner; with ``history=true`` also clear the alert history."""
    if history:
        pipeline.alert_manager.clear_history()
    else:
        pipeline.alert_manager.clear_active()
    return {"ok": True}


@app.get("/api/alerts/active")
def active_alert():
    a = pipeline.alert_manager.active_alert
    return dataclasses.asdict(a) if a else None
