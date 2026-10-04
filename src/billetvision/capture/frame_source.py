"""Unified FrameSource implementation supporting webcam, video file, and image folder modes with bounded drop-oldest queue."""
import logging
import sys
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from billetvision.capture.queue import DropOldestQueue

logger = logging.getLogger(__name__)

_MAX_READ_FAILURES = 30      # consecutive failed webcam reads (~1 s) before a streaming capture gives up
_READ_RETRY_S = 0.03
_WARMUP_S = 20.0             # a camera that has not produced its first frame yet gets this long to wake up
_WARMUP_FAILURES = int(_WARMUP_S / _READ_RETRY_S)
_BACKENDS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def natural_sort_key(path: Path) -> List[Union[int, str]]:
    """Natural alphanumeric sort key (e.g. frame_2 before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


class FrameSource:
    """Frame source supporting webcam (cv2.VideoCapture(0)), video file, and image folder modes.
    
    Uses a bounded drop-oldest queue (default maxsize=4) to ensure capture latency never grows.
    """

    def __init__(
        self,
        source: Union[str, int, Path, None] = None,
        mode: Optional[str] = None,
        maxsize: int = 4,
        loop: bool = False,
        fps: Optional[float] = None,
        delay: float = 0.0,
        pace: bool = True,
        repeat: int = 1,
        backend: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.maxsize = int(maxsize)
        self.repeat = max(1, int(repeat))   # folder/image mode: frames emitted per image
        # webcam: "dshow" | "msmf" | None.  None means DirectShow on Windows, where the OpenCV
        # default (Media Foundation) often opens a camera but never delivers a frame.
        self.backend = backend or ("dshow" if sys.platform == "win32" else None)
        self.frames_received = 0
        self.loop = bool(loop)
        self._custom_fps = float(fps) if fps is not None else None
        self.delay = float(delay)
        self.pace = bool(pace)

        # Resolve mode and source
        if mode is not None:
            resolved_mode = str(mode).strip().lower()
        elif source is None:
            resolved_mode = "webcam"
        elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            resolved_mode = "webcam"
        elif Path(str(source)).is_dir() or Path(str(source)).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            resolved_mode = "folder"   # a directory of images, or one still image
        else:
            resolved_mode = "video"

        if resolved_mode in ("webcam", "camera"):
            self.mode = "webcam"
            raw_index = kwargs.get("index", 0 if source is None else source)
            self.source = int(raw_index)
        elif resolved_mode in ("video", "videofile", "file"):
            self.mode = "video"
            raw_path = kwargs.get("path", source)
            if raw_path is None:
                raise ValueError("Video mode requires a file path")
            self.source = Path(str(raw_path))
        elif resolved_mode in ("folder", "image_folder", "images", "dir"):
            self.mode = "folder"
            raw_dir = kwargs.get("dir", kwargs.get("path", source))
            if raw_dir is None:
                raise ValueError("Image folder mode requires a directory path")
            self.source = Path(str(raw_dir))
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'webcam', 'video', or 'image folder'.")

        self.queue = DropOldestQueue(maxsize=self.maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self._running = False
        self._eof = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._image_files: List[Path] = []
        self._effective_fps: float = 25.0
        self._pace_fps: float = 25.0       # rate the capture thread reads at
        self._release_on_exit = False      # worker frees the device itself if it was stuck in a read
        self.opened = False                # webcam: whether the device actually opened

    def start(self) -> "FrameSource":
        """Initialize the underlying source and start the background capture thread."""
        if self._started and self._running:
            return self

        self._stop_event.clear()
        self._eof = False

        if self.mode == "webcam":
            flag = getattr(cv2, _BACKENDS.get(str(self.backend).lower(), ""), None) if self.backend else None
            self._cap = cv2.VideoCapture(self.source, flag) if flag is not None else cv2.VideoCapture(self.source)
            self.opened = bool(self._cap.isOpened())
            logger.info("Webcam %d: backend=%s opened=%s", self.source, self.backend or "default", self.opened)
            if not self.opened:
                logger.warning("Could not open webcam index %d", self.source)
            else:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep the driver queue short (no-op on some backends)
            val = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0
            native = val if val and val > 0 else 30.0
            self._effective_fps = self._custom_fps or native
            self._pace_fps = native   # read at the camera's own rate; the drop-oldest queue keeps the newest

        elif self.mode == "video":
            if not self.source.exists():
                raise FileNotFoundError(f"Video file not found: {self.source}")
            self._cap = cv2.VideoCapture(str(self.source))
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.source}")
            val = self._cap.get(cv2.CAP_PROP_FPS)
            self._effective_fps = self._custom_fps or (val if val and val > 0 else 25.0)
            self._pace_fps = self._effective_fps
            self.opened = True

        elif self.mode == "folder":
            if self.source.is_file():
                found = [self.source]
            elif self.source.is_dir():
                found = sorted(
                    [p for p in self.source.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS],
                    key=natural_sort_key,
                )
            else:
                raise NotADirectoryError(f"Directory not found: {self.source}")
            self._image_files = [p for p in found for _ in range(self.repeat)]
            if not self._image_files:
                raise RuntimeError(f"No supported images found in folder: {self.source}")
            if self.delay > 0:
                self._effective_fps = 1.0 / self.delay
            else:
                self._effective_fps = self._custom_fps or 15.0
            self._pace_fps = self._effective_fps
            self.opened = True

        self._running = True
        self._started = True
        self._thread = threading.Thread(target=self._capture_worker, name="FrameSourceWorker", daemon=True)
        self._thread.start()
        return self

    def _capture_worker(self) -> None:
        """Worker thread pushing frames into the bounded drop-oldest queue."""
        interval = (1.0 / self._pace_fps) if (self.pace and self._pace_fps > 0) else 0.0

        if self.mode in ("webcam", "video"):
            failures = 0
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                if self._cap is None or not self._cap.isOpened():
                    break
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    if self.mode == "webcam":
                        failures += 1   # a dropped frame is not a lost camera, and a waking one is not dead
                        limit = _MAX_READ_FAILURES if self.frames_received else _WARMUP_FAILURES
                        if failures < limit:
                            time.sleep(_READ_RETRY_S)
                            continue
                        break
                    if self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self._cap.read()
                        if not ret or frame is None:
                            break
                    else:
                        break
                failures = 0

                self.queue.put(frame)
                self.frames_received += 1

                if interval > 0 and not self._stop_event.is_set():
                    elapsed = time.perf_counter() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        elif self.mode == "folder":
            idx = 0
            total = len(self._image_files)
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                if idx >= total:
                    if self.loop:
                        idx = 0
                    else:
                        break

                img_path = self._image_files[idx]
                idx += 1
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    self.queue.put(frame)

                effective_delay = self.delay if self.delay > 0 else (interval if self.pace else 0.0)
                if effective_delay > 0 and not self._stop_event.is_set():
                    elapsed = time.perf_counter() - t0
                    sleep_time = effective_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        self._eof = True
        self._running = False
        if self._release_on_exit:
            self._close_cap()

    def read(self, timeout: Optional[float] = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the bounded queue.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) if a frame was read,
            or (False, None) if the source has ended / timed out.
        """
        if not self._started:
            self.start()

        start_time = time.perf_counter()
        while True:
            try:
                frame = self.queue.get(timeout=0.02)
                return True, frame
            except queue.Empty:
                if self._eof and self.queue.empty():
                    return False, None
                if not self._running and self.queue.empty():
                    return False, None
                if timeout is not None and (time.perf_counter() - start_time) >= timeout:
                    return False, None

    @property
    def warming_up(self) -> bool:
        """Webcam that is open but has not delivered its first frame yet (still waking up)."""
        return self.mode == "webcam" and self._running and self.frames_received == 0

    def wait_first_frame(self, timeout: float) -> bool:
        """Block until the first frame arrives; False if the source died or ``timeout`` passed."""
        end = time.perf_counter() + timeout
        while self.frames_received == 0:
            if not self._running or time.perf_counter() >= end:
                return self.frames_received > 0
            time.sleep(0.05)
        return True

    def get_frame(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """Convenience method returning frame directly, or None if unavailable/ended."""
        ret, frame = self.read(timeout=timeout)
        return frame if ret else None

    def stop(self, timeout: Optional[float] = 1.0) -> None:
        """Signal capture thread to stop and wait for completion."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._running = False

    def _close_cap(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def release(self) -> None:
        """Stop capture and release the device/file.

        Releasing a capture while another thread is blocked in ``read()`` can crash
        OpenCV, so if the worker has not exited in time it frees the device itself
        when its read returns.
        """
        self.stop(timeout=3.0)
        if self._thread is not None and self._thread.is_alive():
            self._release_on_exit = True
        else:
            self._close_cap()
        self._started = False
        self._image_files = []

    def close(self) -> None:
        """Alias for release."""
        self.release()

    @property
    def dropped_frames(self) -> int:
        """Total number of frames dropped due to buffer limit (maxsize=4)."""
        return self.queue.dropped_count

    @property
    def fps(self) -> float:
        """Effective FPS of the source."""
        return self._effective_fps

    def is_alive(self) -> bool:
        """Whether the capture worker is currently running."""
        return self._running

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames until stream ends."""
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
