"""BilletVision end-to-end inspection pipeline.

Wires together capture → preprocess → segment → track → ID read → decision
→ log → annotate → MJPEG + WebSocket broadcast.

Threads
-------
* capture thread  (inside ``FrameSource``) — drop-oldest queue, never blocks.
* ``BVPipeline``  — per-frame vision (segment / measure / track / annotate).
* ``BVFinalizer`` — per-billet work (ID read, verdict, artifacts, log, alert),
  kept off the vision thread so slow OCR can never stall the live feed.
* ``LogWriter``   — the single log-writer thread.

The FastAPI app calls ``pipeline.start()`` on startup and ``pipeline.stop()`` on
shutdown.  The WS event loop is injected via ``set_event_loop()`` so the sync
threads can broadcast safely.
"""
from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import yaml

from billetvision import artifacts
from billetvision.alerts.manager import AlertManager
from billetvision.alerts.webhook import build_notifiers
from billetvision.annotate import draw_billet_card, draw_live_overlay
from billetvision.api.mjpeg import latest_frame
from billetvision.api.ws import ws_manager
from billetvision.capture.frame_source import FrameSource
from billetvision.decision.engine import Verdict, evaluate, load_tolerances
from billetvision.inputs import SourceInfo, frame_overrides
from billetvision.logging_.db import InspectionRecord, billet_id_exists, fetch_recent, max_billet_seq
from billetvision.logging_.writer import LogWriter
from billetvision.ocr.billet_id import IdReadout, read_billet_id
from billetvision.ocr.reader import OcrReader
from billetvision.vision.calibrate import CalibrationData, calibrate_from_marker, warp_undistort_frame
from billetvision.vision.defects import classify_defects, surface_anomaly_score
from billetvision.vision.length import length_from_belt_speed
from billetvision.vision.measure import Measurement, measure
from billetvision.vision.preprocess import to_gray
from billetvision.vision.segment import BackgroundModel, segment
from billetvision.vision.track import CentroidTracker, TrackedBillet
from billetvision.vision.vit_segment import VitConfig, VitResult, refine_with_vit

logger = logging.getLogger(__name__)

_CROP_PAD_PX = 30          # context kept around each billet crop
_CAMERA_LOST_S = 3.0       # no frames for this long → camera lost
_RECONNECT_S = 2.0         # retry interval while the camera is lost
_FIRST_FRAME_S = 25.0      # a freshly selected camera gets this long to deliver its first frame
_RESULT_HOLD_S = 6.0       # how long the last result stays on the video overlay


class SourceUnavailable(RuntimeError):
    """The requested input could not be opened (e.g. no camera at that index)."""


def still_image_notes(complete: bool, scale_source: str, mm_per_px: float) -> List[str]:
    """Reasons an uploaded photo's mm values are not trustworthy (empty = trust them)."""
    notes: List[str] = []
    if not complete:
        notes.append("billet touches the edge of the photo — dimensions may be truncated; "
                     "photograph the whole billet")
    if scale_source != "marker":
        notes.append(f"no calibration marker in the photo — mm values use the stored scale "
                     f"({mm_per_px:.4f} mm/px) and are not verified for this camera")
    return notes


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` updated recursively with ``extra`` (inputs untouched)."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Pipeline stats (read by the API)
# ---------------------------------------------------------------------------

class PipelineStats:
    """Counters and rates updated by the processing threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.billet_ms: float = 0.0
        self.total: int = 0
        self.by_status: Dict[str, int] = {"PASS": 0, "FAIL": 0, "REWORK": 0, "REVIEW": 0}
        self.ocr_ok: int = 0
        self._fps_frames: int = 0
        self._fps_t0: float = time.monotonic()

    def tick_frame(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_ms = latency_ms
            self._fps_frames += 1
            elapsed = time.monotonic() - self._fps_t0
            if elapsed >= 1.0:
                self.fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_t0 = time.monotonic()

    def add_result(self, status: str, ocr_matched: bool, billet_ms: float = 0.0) -> None:
        with self._lock:
            self.total += 1
            self.by_status[status] = self.by_status.get(status, 0) + 1
            if ocr_matched:
                self.ocr_ok += 1
            # running mean of per-billet inspection latency
            self.billet_ms += (billet_ms - self.billet_ms) / self.total

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = max(self.total, 1)
            pass_count = self.by_status.get("PASS", 0)
            return {
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "billet_latency_ms": round(self.billet_ms, 1),
                "total": self.total,
                "pass_rate": round(pass_count / total, 4),
                "ocr_rate": round(self.ocr_ok / total, 4),
                "by_status": dict(self.by_status),
            }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class BilletVisionPipeline:
    """Single-instance inspection pipeline.

    Usage (managed by FastAPI lifespan)::

        pipeline = BilletVisionPipeline("config/config.yaml")
        pipeline.start(event_loop)
        ...
        pipeline.stop()
    """

    def __init__(
        self,
        config_path: str | Path = "config/config.yaml",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self.overrides: Dict[str, Any] = overrides or {}
        self._cfg: Dict[str, Any] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._final_thread: Optional[threading.Thread] = None
        self._final_queue: "queue.Queue[Optional[TrackedBillet]]" = queue.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._switch_lock = threading.Lock()

        # Components — created in start()
        self._source: Optional[FrameSource] = None
        self._tracker: Optional[CentroidTracker] = None
        self._ocr: Optional[OcrReader] = None
        self._writer: Optional[LogWriter] = None
        self._bg: Optional[BackgroundModel] = None
        self.alert_manager = AlertManager()
        self.stats = PipelineStats()

        # Config-derived values
        self._mm_per_px: float = 1.0
        self._calib: Optional[CalibrationData] = None
        self._scale_source = "default"           # marker | stored | default
        self._cal_path = Path("config/calibration.json")
        self._needs_warp = False
        self._profile_name: str = "square_130"
        self._profile_shape: str = "square"
        self._tolerances: Dict[str, Any] = {}
        self._id_regex: str = r"^[A-Z]\d{5,7}$"
        self._min_confidence: float = 0.60
        self._vit = VitConfig()                  # optional ViT segmentation fallback (off by default)
        self._batch_id: str = ""
        self._billet_seq: int = 0
        self._detect_duplicates = True
        self._roi_fitted = True            # False until the first frame when vision.auto_roi is set
        self._snapshot_dir: Path = Path("data/outputs/snapshots")
        self._db_path: str = "data/outputs/billetvision.db"
        self._roi: List[int] = [0, 0, 9999, 9999]
        self._length_mode = "direct"
        self._belt_speed = 250.0
        self._direction = "left_to_right"

        # Runtime state
        self.camera_ok = True
        self.ended = False                       # non-looping source exhausted
        self.frames_done = 0                     # frames processed since the source started
        self.source_info = SourceInfo("configured", "Configured source")
        self._last_raw: Optional[np.ndarray] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_result_t = 0.0
        self._seen_ids: set = set()
        self._last_stats_broadcast: float = 0.0
        self.offline_stats: Dict[str, float] = {}   # filled by run_offline()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        if self._running:
            return
        if event_loop is not None:
            self._loop = event_loop
        self._load_config()
        try:
            self._init_components()
        except Exception:
            if self._source is not None:  # do not leak a running capture thread
                self._source.stop()
            raise
        self._stop_event.clear()
        self.ended = False
        self.frames_done = 0
        self._running = True
        self._final_thread = threading.Thread(
            target=self._finalizer_loop, name="BVFinalizer", daemon=True
        )
        self._final_thread.start()
        self._thread = threading.Thread(
            target=self._process_loop, name="BVPipeline", daemon=True
        )
        self._thread.start()
        logger.info("BilletVision pipeline started")

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._source:
            self._source.release()   # frees the camera / file handle for the next source
        if self._final_thread:  # let queued billets finish before the writer closes
            self._final_queue.put(None)
            self._final_thread.join(timeout=timeout)
        if self._writer:
            self._writer.stop(timeout=15.0)
        self.alert_manager.close()
        logger.info("BilletVision pipeline stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def get_status(self) -> Dict[str, Any]:
        """Runtime status for the API/dashboard."""
        return {
            "pipeline_running": self._running,
            "camera_ok": self.camera_ok,
            "source_ended": self.ended,
            "source": str(self._cfg.get("capture", {}).get("source", "")),
            "profile": self._profile_name,
            "length_mode": self._length_mode,
            "mm_per_px": self._mm_per_px,
            "calibrated_at": self._calib.calibrated_at if self._calib else None,
            "scale_source": self._scale_source,
            "xlsx_status": self._writer.xlsx_status if self._writer else "N/A",
            "queue_depth": self._writer.queue_depth if self._writer else 0,
        }

    @property
    def source_opened(self) -> bool:
        """Whether the current input actually opened (a webcam can fail to)."""
        return self._source is not None and self._source.opened

    def switch_source(self, overrides: Dict[str, Any], info: SourceInfo, require_open: bool = False) -> None:
        """Stop, point the pipeline at a different input, and start it again.

        Stats, the on-video result and the alerts are reset so each run starts
        clean; the log files keep appending.  If the new source cannot be started
        (or, with ``require_open``, did not open), the previous one is restored and
        the error is re-raised.
        """
        with self._switch_lock:
            prev = (self.overrides, self.source_info)
            self.stop()
            self.overrides, self.source_info = overrides, info
            info.started_at = time.time()
            self.stats = PipelineStats()
            self._last_result, self._last_result_t = None, 0.0
            self.alert_manager.reset()
            try:
                self.start()
                if require_open:
                    if not self.source_opened:
                        raise SourceUnavailable(
                            f"Could not open {info.label} (is it connected and not used by another app?)"
                        )
                    if not self._source.wait_first_frame(_FIRST_FRAME_S):
                        raise SourceUnavailable(
                            f"{info.label} opened but delivered no frames within {_FIRST_FRAME_S:.0f} s "
                            "(privacy shutter, another app using it, or a driver problem)"
                        )
            except Exception:
                logger.exception("Could not start source %r - restoring the previous one", info.label)
                self._restore(prev)
                raise
        self._broadcast_sync({"type": "source_changed", **info.as_dict()})

    def _restore(self, prev) -> None:
        """Best-effort return to the previous source after a failed switch."""
        self.stop()
        self.overrides, self.source_info = prev
        try:
            self.start()
        except Exception:
            logger.exception("Could not restore the previous source either")

    # ------------------------------------------------------------------
    # Config / tolerance access for the API
    # ------------------------------------------------------------------

    @property
    def snapshot_dir(self) -> Path:
        return self._snapshot_dir

    @property
    def writer(self) -> Optional[LogWriter]:
        return self._writer

    @property
    def id_regex(self) -> str:
        return self._id_regex

    def get_tolerances(self) -> Dict[str, Any]:
        return self._tolerances

    def update_profile_tolerances(
        self, profile: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if profile not in self._tolerances:
            raise KeyError(profile)
        self._tolerances[profile].update(updates)
        return self._tolerances[profile]

    def set_active_profile(self, profile: str) -> None:
        if profile not in self._tolerances:
            raise KeyError(profile)
        self._profile_name = profile
        self._profile_shape = self._tolerances[profile].get("shape", "square")

    def get_calibration(self) -> Dict[str, Any]:
        """Current scale/calibration summary."""
        cal = self._calib
        return {
            "mm_per_px": self._mm_per_px,
            "px_per_mm": 1.0 / self._mm_per_px if self._mm_per_px else None,
            "marker_type": cal.marker_type if cal else None,
            "marker_size_mm": cal.marker_size_mm if cal else None,
            "calibrated_at": cal.calibrated_at if cal else None,
            "reprojection_error": cal.reprojection_error if cal else None,
            "rectified": self._needs_warp,
        }

    def recalibrate(
        self, marker_size_mm: Optional[float] = None, save: bool = True
    ) -> Dict[str, Any]:
        """Re-run marker calibration on the newest raw frame (FR-3, from the UI).

        Raises:
            RuntimeError: no frame has been received yet.
            ValueError: no ArUco marker visible in the frame.
        """
        frame = self._last_raw
        if frame is None:
            raise RuntimeError("No frame available yet — is the camera running?")
        old = self._calib
        cal = calibrate_from_marker(
            frame,
            marker_size_mm=marker_size_mm or self._marker_size_mm(),
            marker_dict=(old.marker_type if old and old.marker_type else "DICT_4X4_50"),
            camera_matrix=old.get_camera_matrix() if old else None,
            dist_coeffs=old.get_dist_coeffs() if old else None,
        )
        self._apply_calibration(cal)
        self._scale_source = "marker"
        if save:
            cal.save(self._cal_path)
        logger.info("Recalibrated: 1 px = %.4f mm", cal.mm_per_pixel)
        return self.get_calibration()

    # ------------------------------------------------------------------
    # Internal init
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        with self._config_path.open(encoding="utf-8") as fh:
            self._cfg = _deep_merge(yaml.safe_load(fh) or {}, self.overrides)

        vis = self._cfg.get("vision", {})
        self._profile_name = vis.get("active_profile", "square_130")
        self._roi = [int(v) for v in vis.get("roi_box", [0, 0, 9999, 9999])]
        self._length_mode = str(vis.get("length_mode", "direct"))
        self._belt_speed = float(vis.get("conveyor_speed_mm_s", 250.0))
        self._direction = str(vis.get("direction", "left_to_right"))

        tol_path = Path(self._cfg.get("decision", {}).get("tolerances_file", "config/tolerances.yaml"))
        self._tolerances = load_tolerances(tol_path)
        self._profile_shape = self._tolerances.get(self._profile_name, {}).get("shape", "square")

        ocr_cfg = self._cfg.get("ocr", {})
        self._id_regex = ocr_cfg.get("heat_id_regex", r"^[A-Z]\d{5,7}$")
        self._min_confidence = float(ocr_cfg.get("min_confidence", 0.60))
        self._vit = VitConfig.from_dict(self._cfg.get("vit_fallback"))

        log_cfg = self._cfg.get("logging", {})
        self._snapshot_dir = Path(log_cfg.get("snapshot_dir", "data/outputs/snapshots"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = log_cfg.get("db_path", "data/outputs/billetvision.db")

        self.alert_manager.notifiers = build_notifiers(self._cfg.get("alerts", {}))
        self.alert_manager.debounce_s = float(self._cfg.get("alerts", {}).get("debounce_s", 10.0))

        self._detect_duplicates = bool(self._cfg.get("system", {}).get("detect_duplicates", True))
        prefix = str(self._cfg.get("system", {}).get("batch_prefix", "BATCH"))
        self._batch_id = datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%d-%H%M")
        self._cal_path = Path(vis.get("calibration_file", "config/calibration.json"))
        self._load_calibration()

    def _load_calibration(self) -> None:
        if self._cal_path.exists():
            try:
                self._apply_calibration(CalibrationData.load(self._cal_path))
                self._scale_source = "stored"
                return
            except Exception as exc:
                logger.warning("Could not load calibration: %s", exc)
        logger.warning("No valid calibration found — defaulting mm_per_px=0.5")
        self._calib = None
        self._mm_per_px = 0.5  # rough default for demo props
        self._scale_source = "default"
        self._needs_warp = False

    def _apply_calibration(self, cal: CalibrationData) -> None:
        """Adopt ``cal``: scale in mm/px plus whether frames must be rectified."""
        H = cal.get_homography_matrix()
        dist = cal.get_dist_coeffs()
        # A homography fitted to one small marker extrapolates poorly across the whole
        # belt (a sub-pixel corner error becomes a percent-level scale error far from
        # the marker), so it is only applied when vision.use_homography is set.
        use_h = bool(self._cfg.get("vision", {}).get("use_homography", False))
        needs_warp = bool(
            (use_h and H is not None and not np.allclose(H, np.eye(3), atol=1e-6))
            or (dist is not None and np.any(dist != 0))
        )
        self._calib = cal
        self._mm_per_px = float(cal.mm_per_pixel)
        self._needs_warp = needs_warp

    def _init_components(self, start_source: bool = True) -> None:
        vis_cfg = self._cfg.get("vision", {})
        log_cfg = self._cfg.get("logging", {})

        if start_source:
            self._source = self._make_source()
            self._source.start()
            if vis_cfg.get("auto_calibrate", True):
                self._auto_calibrate(int(vis_cfg.get("calibration_attempts", 30)))

        self._build_tracker()
        self._roi_fitted = not vis_cfg.get("auto_roi", False)
        self._bg = BackgroundModel() if vis_cfg.get("background_subtraction", True) else None
        self._ocr = OcrReader.from_config(self._cfg.get("ocr", {}))

        self._writer = LogWriter(
            db_path=self._db_path,
            csv_path=log_cfg.get("csv_path", "data/outputs/billet_log.csv"),
            xlsx_path=log_cfg.get("xlsx_path", "data/outputs/billet_log.xlsx"),
            rotate_daily=bool(log_cfg.get("rotate_daily", True)),
        )
        self._writer.start()
        self._billet_seq = max_billet_seq(self._db_path)  # never reuse a seq across restarts
        self._seen_ids.clear()
        self.camera_ok = True

        logger.info(
            "Pipeline components ready — profile=%s mm_per_px=%.4f length_mode=%s",
            self._profile_name, self._mm_per_px, self._length_mode,
        )

    def _build_tracker(self) -> None:
        """(Re)create the tracker with entry/exit lines derived from the current ROI."""
        vis_cfg = self._cfg.get("vision", {})
        x1, _, x2, _ = self._roi
        margin = int(vis_cfg.get("entry_margin_px", 20))
        ltr = self._direction == "left_to_right"
        self._tracker = CentroidTracker(
            entry_x=(x1 + margin) if ltr else (x2 - margin),
            exit_x=(x2 - margin) if ltr else (x1 + margin),
            max_distance_px=100.0,
            max_lost_frames=15,
            best_n=int(vis_cfg.get("best_n_frames", 5)),
            direction=self._direction,
        )

    def _fit_to_frame(self, frame: np.ndarray) -> None:
        """Size the ROI and minimum contour area to the real frame (``vision.auto_roi``)."""
        h, w = frame.shape[:2]
        fit = frame_overrides(w, h)
        self._roi = [int(v) for v in fit["roi_box"]]
        self._cfg.setdefault("vision", {})["min_contour_area"] = fit["min_contour_area"]
        self._build_tracker()
        self._roi_fitted = True
        logger.info("ROI fitted to the %dx%d frame: %s", w, h, self._roi)

    def _make_source(self) -> FrameSource:
        cap_cfg = self._cfg.get("capture", {})
        return FrameSource(
            source=cap_cfg.get("source", 0),
            maxsize=cap_cfg.get("queue_maxsize", 5),
            loop=bool(cap_cfg.get("loop", True)),  # loop video files for the demo
            fps=cap_cfg.get("fps_target", 15),
            repeat=int(cap_cfg.get("repeat", 1)),
            backend=cap_cfg.get("backend"),
        )

    def _marker_size_mm(self) -> float:
        """Physical marker side: ``vision.marker_size_mm`` if configured, else the stored one, else 50."""
        cfg = self._cfg.get("vision", {}).get("marker_size_mm")
        old = self._calib
        return float(cfg or (old.marker_size_mm if old and old.marker_size_mm else 50.0))

    def _try_marker_calibration(self, frame: np.ndarray) -> bool:
        """Adopt the scale from a calibration marker visible in ``frame``; False if none."""
        old = self._calib
        try:
            cal = calibrate_from_marker(
                frame,
                marker_size_mm=self._marker_size_mm(),
                marker_dict=(old.marker_type if old and old.marker_type else "DICT_4X4_50"),
                camera_matrix=old.get_camera_matrix() if old else None,
                dist_coeffs=old.get_dist_coeffs() if old else None,
            )
        except ValueError:
            return False
        if old is not None and abs(cal.mm_per_pixel - old.mm_per_pixel) / old.mm_per_pixel > 0.05:
            logger.warning(
                "Marker scale %.4f mm/px differs from stored calibration %.4f — using the marker",
                cal.mm_per_pixel, old.mm_per_pixel,
            )
        self._apply_calibration(cal)
        self._scale_source = "marker"
        logger.info("Auto-calibrated from marker: 1 px = %.4f mm", cal.mm_per_pixel)
        return True

    def _auto_calibrate(self, attempts: int = 30) -> None:
        """If a calibration marker is visible at startup, derive mm/px from it."""
        assert self._source is not None
        for _ in range(attempts):
            ok, frame = self._source.read(timeout=0.5)
            if ok and frame is not None and self._try_marker_calibration(frame):
                return
        logger.info("No calibration marker in view — using stored calibration")

    # ------------------------------------------------------------------
    # Processing loop (vision thread)
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        assert self._source and self._tracker
        last_frame_t = time.monotonic()
        last_reconnect = 0.0
        frame_no = 0

        while not self._stop_event.is_set():
            ok, frame = self._source.read(timeout=0.5)
            now = time.monotonic()
            if not ok or frame is None:
                last_reconnect = self._on_no_frame(now, last_frame_t, last_reconnect)
                continue
            if not self.camera_ok:
                self._set_camera(True)
            last_frame_t = now
            self._last_raw = frame
            if not self._roi_fitted:
                self._fit_to_frame(frame)
            frame_no += 1
            self.frames_done = frame_no
            t0 = time.perf_counter()

            try:
                finalized, annotated = self._process_frame(frame, self._frame_time(frame_no, now))
                for tb in finalized:
                    self._final_queue.put(tb)
            except Exception as exc:
                logger.exception("Frame processing error: %s", exc)
                annotated = frame.copy()

            latest_frame.update(annotated)
            self.stats.tick_frame((time.perf_counter() - t0) * 1000)
            self._maybe_broadcast_stats()

        logger.info("Pipeline processing loop exited")

    def _frame_time(self, frame_no: int, now: float) -> float:
        """Timestamp (s) for belt-speed timing.

        File/folder sources use frame index / fps (deterministic, immune to
        pacing jitter); a live camera uses the monotonic clock.
        """
        src = self._source
        if src is not None and src.mode != "webcam" and src.fps > 0:
            return frame_no / float(src.fps)
        return now

    def _on_no_frame(self, now: float, last_frame_t: float, last_reconnect: float) -> float:
        """Handle an empty read: end-of-stream, camera loss, reconnect. Returns last reconnect time."""
        src = self._source
        assert src is not None and self._tracker is not None
        live = src.mode == "webcam"
        if getattr(src, "warming_up", False):
            return last_reconnect   # still waking up: reconnecting now would restart the warm-up
        if not live and not self.loop_enabled and not src.is_alive():
            if not self.ended:
                self.ended = True
                for tb in self._tracker.flush():
                    self._final_queue.put(tb)
                logger.info("Frame source finished (end of stream)")
            return last_reconnect
        if now - last_frame_t >= _CAMERA_LOST_S:
            if self.camera_ok:
                self._set_camera(False)
            if now - last_reconnect >= _RECONNECT_S:
                self._reconnect()
                return now
        return last_reconnect

    @property
    def loop_enabled(self) -> bool:
        return bool(self._cfg.get("capture", {}).get("loop", True))

    @property
    def _still_image(self) -> bool:
        return bool(self._cfg.get("vision", {}).get("still_image", False))

    def _reconnect(self) -> None:
        """Replace the frame source after a camera loss (best effort)."""
        logger.warning("Camera lost — attempting to reconnect")
        try:
            if self._source:
                self._source.release()   # the old handle must be freed or the device stays busy
            self._source = self._make_source()
            self._source.start()
        except Exception as exc:
            logger.error("Reconnect failed: %s", exc)

    def _set_camera(self, ok: bool) -> None:
        """Record camera state, raise/clear the operator alert, notify the dashboard."""
        self.camera_ok = ok
        if ok:
            self.alert_manager.trigger("-", "CAMERA_RESTORED", ["Camera feed restored"], kind="camera", sound=False)
        else:
            self.alert_manager.trigger(
                "-", "CAMERA_LOST", [f"No frames received for {_CAMERA_LOST_S:.0f} s — check camera"], kind="camera"
            )
        self._broadcast_sync({"type": "camera_status", "status": "ok" if ok else "lost"})
        a = self.alert_manager.active_alert
        if a is not None:
            self._broadcast_sync({"type": "alert", **a.to_payload(), "sound": a.sound})

    def _inside_roi(self, x: int, y: int, w: int, h: int, margin: int = 3) -> bool:
        """True if the bbox lies fully inside the ROI (not clipped by its border)."""
        x1, y1, x2, y2 = self._roi
        return x > x1 + margin and y > y1 + margin and x + w < x2 - margin and y + h < y2 - margin

    def _process_frame(self, frame: np.ndarray, t: float):
        """Segment, measure and track one frame; returns (finalised billets, annotated frame)."""
        vis = self._cfg.get("vision", {})
        if self._needs_warp and self._calib is not None:
            frame = warp_undistort_frame(frame, self._calib)
        gray = to_gray(frame)
        if self._bg is not None:
            self._bg.apply(gray)

        roi_area = max((self._roi[2] - self._roi[0]) * (self._roi[3] - self._roi[1]), 1)
        seg = segment(
            frame,
            hot_billet_mode=bool(vis.get("hot_billet_mode", False)),
            min_area_frac=float(vis.get("min_contour_area", 5000)) / roi_area,
            roi=self._roi,
            background=self._bg,
            preprocess_kwargs={"auto_exposure": bool(vis.get("auto_exposure", True))},
        )

        detections: list = []
        meas: Optional[Measurement] = None
        provisional: Optional[str] = None
        if seg is not None:
            det, meas = self._make_detection(frame, gray, seg)
            if det is not None:
                detections.append(det)
                if meas is not None and self._profile_name in self._tolerances:
                    provisional = evaluate(meas, self._profile_name, self._tolerances).status

        finalized = self._tracker.update(detections, timestamp=t)
        annotated = draw_live_overlay(
            frame,
            self._roi,
            seg.contour if seg is not None else None,
            meas,
            provisional,
            self._current_result(t),
            self._stats_line(),
            self.camera_ok,
        )
        return finalized, annotated

    def _make_detection(self, frame: np.ndarray, gray: np.ndarray, seg):
        """Build the tracker detection tuple (and per-frame measurement) for a segment."""
        m = cv2.moments(seg.contour)
        if m["m00"] <= 0:
            return None, None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = seg.bounding_rect
        complete = self._inside_roi(x, y, w, h)
        x0, y0 = max(0, x - _CROP_PAD_PX), max(0, y - _CROP_PAD_PX)
        x1, y1 = min(gray.shape[1], x + w + _CROP_PAD_PX), min(gray.shape[0], y + h + _CROP_PAD_PX)
        gray_crop = gray[y0:y1, x0:x1].copy()
        contour_local = seg.contour - np.array([[[x0, y0]]], dtype=seg.contour.dtype)

        meas: Optional[Measurement] = None
        # Direct mode needs the whole billet in view; belt-speed mode accepts clipped
        # frames because only the cross-section (not the length) is read from them.
        # A still photo has no later frame to wait for: measure it, and flag it REVIEW.
        if complete or self._length_mode == "belt_speed" or self._still_image:
            meas = measure(seg.contour, self._mm_per_px, shape=self._profile_shape, travel_axis="x")
            if complete:
                mask = np.zeros(gray_crop.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour_local], -1, 255, cv2.FILLED)
                meas.surface_anomaly_score = round(surface_anomaly_score(gray_crop, mask), 4)
        extras = {
            "x_range": (x, x + w),
            "crop_origin": (x0, y0),
            "color_crop": frame[y0:y1, x0:x1].copy() if meas is not None else None,
            "complete": complete,
        }
        return ((cx, cy), contour_local, gray_crop, meas, extras), meas

    def _current_result(self, t: float) -> Optional[Dict[str, Any]]:
        return self._last_result if t - self._last_result_t < _RESULT_HOLD_S else None

    def _stats_line(self) -> str:
        snap = self.stats.snapshot()
        return f"FPS {snap['fps']:.1f}  {snap['latency_ms']:.0f} ms  #{snap['total']}  1px={self._mm_per_px:.3f}mm"

    # ------------------------------------------------------------------
    # Per-billet handling (finalizer thread)
    # ------------------------------------------------------------------

    def _finalizer_loop(self) -> None:
        while True:
            tb = self._final_queue.get()
            if tb is None:
                return
            try:
                self._handle_billet(tb)
            except Exception as exc:  # one bad billet must not stop the line
                logger.exception("Billet handling failed: %s", exc)

    def _apply_length_mode(self, tb: TrackedBillet) -> None:
        """Replace the direct length with belt-speed x time-in-view when configured."""
        if self._length_mode != "belt_speed":
            return
        x1, _, x2, _ = self._roi
        length = length_from_belt_speed(tb.timeline, (x1 + x2) / 2.0, self._belt_speed, self._direction)
        if length is None:
            logger.warning("Belt-speed length unavailable (head/tail not seen crossing) — using direct length")
            return
        tb.measurement.length_mm = round(length, 2)

    def _handle_billet(self, tb: TrackedBillet) -> None:
        t0 = time.perf_counter()
        self._billet_seq += 1
        seq = self._billet_seq
        tol = self._tolerances.get(self._profile_name, {})
        seg_fix = refine_with_vit(tb, self._vit, tol, self._mm_per_px, self._profile_shape)
        meas = tb.measurement
        self._apply_length_mode(tb)
        meas.defects = classify_defects(
            camber_mm=meas.camber_mm,
            cross_section_var_mm=meas.cross_section_var_mm,
            surface_anomaly=meas.surface_anomaly_score,
            edge_irregularity=meas.edge_irregularity_mm,
            tol=tol,
        )

        readout = read_billet_id(
            [s.frame_gray for s in tb.top_samples],
            self._ocr,
            self._id_regex,
            min_confidence=self._min_confidence,
            fixed_roi=self._cfg.get("vision", {}).get("id_roi"),
            max_frames=int(self._cfg.get("ocr", {}).get("max_frames", 3)),
        )
        billet_id = readout.text or f"UNREAD-{seq:06d}"
        duplicate = (
            self._detect_duplicates and readout.status == "PASS" and self._is_duplicate(billet_id)
        )
        if duplicate:
            meas.defects = [*meas.defects, "duplicate_id"]

        verdict = evaluate(meas, self._profile_name, self._tolerances, ocr_status=readout.status)
        reasons = list(verdict.reasons)
        if readout.status == "REVIEW":
            msg = readout.reason(self._min_confidence, self._id_regex)
            reasons = [msg] if verdict.status == "REVIEW" else [*reasons, msg]
        if seg_fix.source == "unresolved":  # suspect outline the ViT could not fix: never pass it silently
            reasons = [*reasons, seg_fix.note]
            if verdict.status == "PASS":
                verdict = Verdict(status="REVIEW", reasons=reasons)
        if self._still_image:
            notes = still_image_notes(
                bool(tb.top_samples[0].extras.get("complete", True)), self._scale_source, self._mm_per_px
            )
            if notes:  # the mm values cannot be trusted, so neither PASS nor FAIL is a safe verdict
                reasons = [*reasons, *notes]
                verdict = Verdict(status="REVIEW", reasons=reasons)
        t_verdict = time.perf_counter()

        image_path, frame_names = self._save_artifacts(
            tb, seq, meas, verdict.status, billet_id, reasons, readout, seg_fix
        )
        processing_ms = (time.perf_counter() - t0) * 1000
        record = InspectionRecord.make(
            billet_seq=seq,
            billet_id=billet_id,
            batch_id=self._batch_id,
            length_mm=meas.length_mm,
            width_mm=meas.width_mm,
            height_mm=meas.height_mm,
            diameter_mm=meas.diameter_mm,
            ovality=meas.ovality,
            diag_diff_mm=meas.diag_diff_mm,
            defects=", ".join(meas.defects),
            ocr_confidence=readout.confidence,
            status=verdict.status,
            fail_reasons="; ".join(reasons),
            image_path=str(image_path),
            processing_ms=round(processing_ms, 1),
        )
        assert self._writer is not None
        self._writer.submit(record)
        self._seen_ids.add(billet_id)
        self._publish(record, reasons, readout, duplicate, image_path, t_verdict)
        self._write_detail(tb, seq, meas, verdict.status, reasons, readout, frame_names, record, seg_fix)
        logger.info("Billet #%d %s %s (%.0f ms)", seq, billet_id, verdict.status, processing_ms)

    def _is_duplicate(self, billet_id: str) -> bool:
        return billet_id in self._seen_ids or billet_id_exists(self._db_path, billet_id)

    def _publish(
        self,
        record: InspectionRecord,
        reasons: List[str],
        readout: IdReadout,
        duplicate: bool,
        image_path: Path,
        t_verdict: float,
    ) -> None:
        """Alert, update stats/overlay and push the result to the dashboard."""
        status = record.status
        alert = None
        if status != "PASS":
            alert = self.alert_manager.trigger(
                billet_id=record.billet_id, status=status, reasons=reasons, image_path=str(image_path),
                latency_ms=round((time.perf_counter() - t_verdict) * 1000, 2),
            )
        if duplicate:
            self.alert_manager.trigger(
                record.billet_id, "DUPLICATE_ID", [f"ID {record.billet_id} was already logged"],
                kind="warning", sound=False,
            )
        self.stats.add_result(status, readout.status == "PASS", record.processing_ms)
        self._last_result = {"billet_id": record.billet_id, "status": status, "seq": record.billet_seq}
        self._last_result_t = time.monotonic()

        self._broadcast_sync({
            "type": "inspection_result",
            "billet_seq": record.billet_seq,
            "billet_id": record.billet_id,
            "batch_id": record.batch_id,
            "status": status,
            "length_mm": record.length_mm,
            "width_mm": record.width_mm,
            "height_mm": record.height_mm,
            "diameter_mm": record.diameter_mm,
            "ovality": record.ovality,
            "diag_diff_mm": record.diag_diff_mm,
            "defects": record.defects,
            "fail_reasons": reasons,
            "ocr_confidence": record.ocr_confidence,
            "processing_ms": record.processing_ms,
            "timestamp": record.timestamp,
            "image_url": f"/snapshots/{image_path.name}",
        })
        if alert is not None:
            self._broadcast_sync({"type": "alert", **alert.to_payload(), "sound": alert.sound,
                                  "latency_ms": alert.latency_ms})

    def _save_artifacts(
        self,
        tb: TrackedBillet,
        seq: int,
        meas: Measurement,
        status: str,
        billet_id: str,
        reasons: List[str],
        readout: IdReadout,
        seg_fix: VitResult,
    ):
        """Write the annotated card, ID crop (REVIEW) and thumbnails; return (card path, thumb names)."""
        best = tb.top_samples[0]
        color = best.extras.get("color_crop")
        if color is None:
            color = cv2.cvtColor(best.frame_gray, cv2.COLOR_GRAY2BGR)
        card = draw_billet_card(color, best.contour, meas, status, billet_id, reasons, seg_fix.mask_source)
        path = artifacts.save_snapshot(self._snapshot_dir, seq, card)
        if status == "REVIEW" or readout.status == "REVIEW":
            artifacts.save_id_crop(self._snapshot_dir, seq, readout.crop)
        thumbs = [s.extras.get("color_crop") for s in tb.top_samples[:3]]
        names = artifacts.save_frames(self._snapshot_dir, seq, [t for t in thumbs if t is not None])
        return path, names

    def _write_detail(
        self,
        tb: TrackedBillet,
        seq: int,
        meas: Measurement,
        status: str,
        reasons: List[str],
        readout: IdReadout,
        frame_names: List[str],
        record: InspectionRecord,
        seg_fix: VitResult,
    ) -> None:
        """Persist the drill-down JSON consumed by ``GET /api/billet/{seq}``."""
        def fields(m: Measurement) -> Dict[str, Any]:
            return {
                "length_mm": m.length_mm, "width_mm": m.width_mm, "height_mm": m.height_mm,
                "diameter_mm": m.diameter_mm, "ovality": m.ovality, "diag_diff_mm": m.diag_diff_mm,
                "camber_mm": m.camber_mm, "cross_section_var_mm": m.cross_section_var_mm,
                "edge_irregularity_mm": m.edge_irregularity_mm,
                "surface_anomaly_score": m.surface_anomaly_score,
            }
        id_crop = self._snapshot_dir / artifacts.artifact_name(seq, "_id.png")
        artifacts.save_detail(self._snapshot_dir, seq, {
            "billet_seq": seq,
            "billet_id": record.billet_id,
            "status": status,
            "reasons": reasons,
            "profile": self._profile_name,
            "tolerances": dict(self._tolerances.get(self._profile_name, {})),
            "length_mode": self._length_mode,
            "mm_per_px": self._mm_per_px,
            "mask_source": seg_fix.mask_source,
            "measurement": fields(meas),
            "defects": list(meas.defects),
            "frames": [
                {"rank": i + 1, "sharpness": round(s.sharpness_score, 1), **fields(s.measurement)}
                for i, s in enumerate(tb.top_samples)
            ],
            "frames_observed": tb.frame_count,
            "ocr": {
                "text": readout.text, "confidence": readout.confidence, "status": readout.status,
                "format_valid": readout.matched, "source": readout.source,
                "candidates": readout.candidates, "frames_read": readout.frames_read,
            },
            "images": {
                "card": f"/snapshots/{artifacts.artifact_name(seq, '.jpg')}",
                "id_crop": f"/snapshots/{id_crop.name}" if id_crop.exists() else None,
                "frames": [f"/snapshots/{n}" for n in frame_names],
            },
        })

    # ------------------------------------------------------------------
    # Offline (synchronous) processing — accuracy runs and tests
    # ------------------------------------------------------------------

    def run_offline(self, frames: Iterable[np.ndarray], fps: float = 15.0) -> List[Dict[str, Any]]:
        """Process ``frames`` synchronously and return the logged records (oldest first).

        No capture/vision/finalizer threads are used and time advances by
        ``1 / fps`` per frame, so results are deterministic.  The log writer
        still runs, so SQLite/CSV/XLSX output is exercised exactly as live.
        """
        if self._running:
            raise RuntimeError("run_offline cannot be used while the pipeline is running")
        self._load_config()
        self._init_components(start_source=False)
        assert self._tracker is not None and self._writer is not None
        first_seq = self._billet_seq
        it = iter(frames)
        head = list(itertools.islice(it, 30))
        if self._cfg.get("vision", {}).get("auto_calibrate", True):
            for frame in head:
                if self._try_marker_calibration(frame):
                    break
        frame_ms: List[float] = []
        try:
            for n, frame in enumerate(itertools.chain(head, it), start=1):
                self._last_raw = frame
                t0 = time.perf_counter()
                finalized, _ = self._process_frame(frame, n / fps)
                frame_ms.append((time.perf_counter() - t0) * 1000)
                for tb in finalized:
                    self._handle_billet(tb)
            for tb in self._tracker.flush():
                self._handle_billet(tb)
        finally:
            self.offline_stats = {
                "frames": len(frame_ms),
                "mean_frame_ms": float(np.mean(frame_ms)) if frame_ms else 0.0,
                "p95_frame_ms": float(np.percentile(frame_ms, 95)) if frame_ms else 0.0,
            }
            self._writer.stop(timeout=30.0)
            self.alert_manager.close()
        rows = fetch_recent(self._db_path, n=None)
        return [r for r in reversed(rows) if r["billet_seq"] > first_seq]

    # ------------------------------------------------------------------
    # Operator corrections (review queue)
    # ------------------------------------------------------------------

    def apply_review(self, billet_seq: int, fields: Dict[str, Any]) -> bool:
        """Write an operator correction through the single log writer."""
        if self._writer is None:
            return False
        return self._writer.update_record(billet_seq, fields)

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    def _maybe_broadcast_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats_broadcast >= 1.0:
            self._last_stats_broadcast = now
            self._broadcast_sync({"type": "kpi_update", **self.stats.snapshot()})

    def _broadcast_sync(self, message: Dict[str, Any]) -> None:
        """Fire-and-forget broadcast from a sync processing thread."""
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(message), self._loop)
            except Exception as exc:
                logger.debug("WS broadcast error: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton — shared with the API
# ---------------------------------------------------------------------------

pipeline = BilletVisionPipeline()
