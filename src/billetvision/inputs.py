"""Operator-selectable input sources: uploaded video, uploaded image, simulated demo.

Turns a source choice into pipeline config overrides.  The ROI and the minimum
contour area are scaled to the frame size, because ``config.yaml`` is tuned for a
1280x720 belt camera and an arbitrary upload can be any resolution.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from billetvision import synthetic

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
DEMO_DIR = Path("data/raw")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = {"video": 500 * 1024 * 1024, "image": 25 * 1024 * 1024}

IMAGE_REPEAT = 8          # frames a still image is fed to the tracker (best-N needs several)
_ROI_MARGIN = 0.03        # fraction of the frame left outside the ROI
_MAX_IMAGE_SIDE = 4096    # larger stills are downscaled on save (keep phone detail: 1 px must be << tolerance)
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


class UploadError(ValueError):
    """The uploaded file is unusable (wrong type, unreadable, too large)."""


@dataclass
class SourceInfo:
    """What the pipeline is currently analysing (shown on the dashboard)."""

    kind: str                     # configured | camera | video | image | demo
    label: str
    frames_total: Optional[int] = None
    started_at: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "frames_total": self.frames_total}


def safe_upload_path(filename: str, kind: str, root: Optional[Path] = None) -> Path:
    """Timestamped, sanitised destination for an upload; raises UploadError on a bad extension."""
    name = _SAFE_CHARS.sub("_", Path(filename or "").name).strip("._") or kind
    ext = Path(name).suffix.lower()
    allowed = VIDEO_EXT if kind == "video" else IMAGE_EXT
    if ext not in allowed:
        raise UploadError(f"Unsupported {kind} type '{ext or '?'}'. Allowed: {', '.join(sorted(allowed))}")
    root = root or UPLOAD_DIR   # looked up at call time so tests can redirect it
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"


def validate_video(path: Path) -> Dict[str, Any]:
    """Open ``path`` and return ``{width, height, fps, frames}``; raises UploadError if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok or frame is None:
            raise UploadError("Could not decode this video (unsupported codec or corrupt file)")
        h, w = frame.shape[:2]
        return {
            "width": w, "height": h,
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None,
        }
    finally:
        cap.release()


def validate_image(path: Path) -> Dict[str, Any]:
    """Decode ``path``, downscale it in place if huge, return ``{width, height}``."""
    img = cv2.imread(str(path))
    if img is None:
        raise UploadError("Could not decode this image (unsupported format or corrupt file)")
    h, w = img.shape[:2]
    big = max(h, w)
    if big > _MAX_IMAGE_SIDE:
        k = _MAX_IMAGE_SIDE / big
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        cv2.imwrite(str(path), img)
    return {"width": w, "height": h}


def frame_overrides(width: int, height: int) -> Dict[str, Any]:
    """ROI (3% margin) and minimum contour area scaled to a ``width`` x ``height`` frame."""
    mx, my = int(width * _ROI_MARGIN), int(height * _ROI_MARGIN)
    return {
        "roi_box": [mx, my, width - mx, height - my],
        "min_contour_area": max(500, int(0.004 * width * height)),
    }


def camera_overrides(index: int, backend: Optional[str] = None) -> Dict[str, Any]:
    """Live camera: the ROI is fitted to the first real frame (``vision.auto_roi``), so the
    device is opened exactly once.  Length mode etc. stay whatever ``config.yaml`` says."""
    return {
        "system": {"batch_prefix": "LIVE"},
        "capture": {"source": index, "loop": False, "fps_target": 15, "backend": backend},
        "vision": {"auto_roi": True},
    }


def video_overrides(path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse every frame once (paced at <= 15 fps, so slower than real time for fast clips)."""
    return {
        "system": {"batch_prefix": "UPLOAD"},
        "capture": {"source": str(path), "loop": False, "fps_target": min(meta["fps"], 15.0)},
        "vision": {**frame_overrides(meta["width"], meta["height"]), "calibration_attempts": 3},
    }


def image_overrides(path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    """A still photo: no conveyor, so the billet is measured even if it touches the frame
    edge (flagged REVIEW instead of dropped) and the tracker lines sit on the ROI edges."""
    return {
        "system": {"batch_prefix": "UPLOAD"},
        "capture": {"source": str(path), "loop": False, "fps_target": 15, "repeat": IMAGE_REPEAT},
        "vision": {
            **frame_overrides(meta["width"], meta["height"]),
            "calibration_attempts": 1,
            "still_image": True,
            "entry_margin_px": 0,
            "length_mode": "direct",
        },
    }


def demo_video_path() -> Path:
    """Cache path of the demo video (named after its scene/line-up; resolved lazily)."""
    return DEMO_DIR / synthetic.demo_video_name()


def demo_overrides(path: Path) -> Dict[str, Any]:
    """Synthetic belt (``synthetic.DEMO_SCENE``), looped forever."""
    out = synthetic.DEMO_SCENE.overrides()
    out["system"] = {"batch_prefix": "DEMO", "detect_duplicates": False}   # the loop replays the same IDs
    out["capture"] = {"source": str(path), "loop": True, "fps_target": synthetic.FPS}
    return out


def ensure_demo_video(path: Optional[Path] = None) -> Path:
    """Return the cached synthetic belt video, rendering it first if it is missing."""
    path = path or demo_video_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    logger.info("Rendering simulated belt video to %s (one-off, ~2 min)", path)
    tmp = path.with_name(path.stem + ".part" + path.suffix)
    synthetic.write_video(synthetic.render_belt_frames(synthetic.show_props(), scene=synthetic.DEMO_SCENE), tmp)
    os.replace(tmp, path)   # never leave a half-written file that looks valid
    return path
