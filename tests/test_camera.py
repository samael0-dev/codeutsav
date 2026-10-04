"""Live-camera path with a fake device: capture resilience, release, checked switching, ROI fitting, API.

No real camera is ever opened - ``cv2.VideoCapture`` is replaced by ``FakeCap``.
"""
from __future__ import annotations

import threading
import time
import unittest.mock
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from billetvision import inputs
from billetvision.capture import frame_source as fs
from billetvision.capture.frame_source import FrameSource

with unittest.mock.patch("billetvision.pipeline.BilletVisionPipeline.start"), \
        unittest.mock.patch("billetvision.pipeline.BilletVisionPipeline.stop"):
    from billetvision.api.main import app
    from billetvision.pipeline import BilletVisionPipeline, SourceUnavailable, pipeline

FRAME = np.full((48, 64, 3), 90, np.uint8)


class FakeCap:
    """Scripted capture device: ``script`` items are True (a frame) or False (a failed read)."""

    def __init__(self, script, opened=True, block: threading.Event | None = None):
        self.script, self.opened, self.block = list(script), opened, block
        self.released, self.props = False, {}

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.block is not None:
            self.block.wait(5)
        if self.script and self.script.pop(0):
            return True, FRAME.copy()
        return False, None

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


def _source(cap: FakeCap) -> FrameSource:
    with unittest.mock.patch.object(fs.cv2, "VideoCapture", return_value=cap):
        return FrameSource(source=0, mode="webcam", pace=False, maxsize=50).start()


def _drain(src: FrameSource, want: int, timeout: float = 5.0) -> int:
    got, end = 0, time.time() + timeout
    while got < want and time.time() < end:
        ok, _ = src.read(timeout=0.2)
        got += 1 if ok else 0
    return got


def test_webcam_survives_transient_read_failures():
    cap = FakeCap([True] + [False] * 5 + [True, True])
    src = _source(cap)
    assert _drain(src, 3) == 3            # the dropped frames did not end the capture
    src.release()


def test_webcam_gives_up_after_sustained_failure_so_the_pipeline_can_reconnect():
    src = _source(FakeCap([True]))
    assert _drain(src, 1) == 1
    end = time.time() + 5
    while src.is_alive() and time.time() < end:
        time.sleep(0.05)
    assert not src.is_alive()
    src.release()


def test_opened_flag_and_buffer_hint():
    cap = FakeCap([True])
    src = _source(cap)
    assert src.opened and cap.props.get(cv2.CAP_PROP_BUFFERSIZE) == 1
    src.release()
    assert not _source(FakeCap([], opened=False)).opened


def test_release_frees_the_device():
    cap = FakeCap([True] * 200)
    src = _source(cap)
    assert _drain(src, 1) == 1
    src.release()
    assert cap.released


def test_release_defers_to_the_worker_when_it_is_stuck_in_a_read():
    gate = threading.Event()
    cap = FakeCap([True] * 5, block=gate)
    src = _source(cap)
    with unittest.mock.patch.object(FrameSource, "stop", lambda self, timeout=None: self._stop_event.set()):
        src.release()
    assert not cap.released               # never release under a blocked read()
    gate.set()
    end = time.time() + 5
    while not cap.released and time.time() < end:
        time.sleep(0.05)
    assert cap.released                   # ...the worker frees it once the read returns


def test_reconnect_releases_the_old_device():
    pipe = BilletVisionPipeline("config/config.yaml")
    old, new = unittest.mock.Mock(), unittest.mock.Mock()
    pipe._source = old
    with unittest.mock.patch.object(pipe, "_make_source", return_value=new):
        pipe._reconnect()
    old.release.assert_called_once()
    new.start.assert_called_once()
    assert pipe._source is new


def test_unopened_camera_rolls_back_to_the_previous_source():
    pipe = BilletVisionPipeline("config/config.yaml", overrides={"marker": "old"})
    starts = []

    def fake_start(*a, **k):
        starts.append(dict(pipe.overrides))
        pipe._source = SimpleNamespace(opened=len(starts) > 1)   # the camera fails to open; the restore works

    with unittest.mock.patch.object(pipe, "start", fake_start), unittest.mock.patch.object(pipe, "stop"):
        with pytest.raises(SourceUnavailable):
            pipe.switch_source(inputs.camera_overrides(3), inputs.SourceInfo("camera", "Camera 3"), require_open=True)
    assert starts[0]["capture"]["source"] == 3 and starts[1] == {"marker": "old"}
    assert pipe.overrides == {"marker": "old"} and pipe.source_info.kind == "configured"


def test_auto_roi_fits_the_real_frame_and_rebuilds_the_tracker():
    pipe = BilletVisionPipeline("config/config.yaml", overrides=inputs.camera_overrides(0))
    pipe._load_config()
    pipe._build_tracker()
    old_tracker = pipe._tracker
    pipe._fit_to_frame(np.zeros((480, 640, 3), np.uint8))
    x1, y1, x2, y2 = pipe._roi
    assert 0 < x1 < x2 <= 640 and 0 < y1 < y2 <= 480
    assert pipe._tracker is not old_tracker and pipe._tracker.exit_x < 640 and pipe._roi_fitted


def test_camera_overrides_open_once_and_leave_length_mode_to_config():
    ov = inputs.camera_overrides(2)
    assert ov["capture"]["source"] == 2 and ov["vision"] == {"auto_roi": True}
    assert ov["system"]["batch_prefix"] == "LIVE"


@pytest.fixture()
def client():
    with unittest.mock.patch.object(pipeline, "start"), unittest.mock.patch.object(pipeline, "stop"):
        with TestClient(app) as c:
            yield c


def _running(kind="camera", label="Camera 0", camera_ok=True):
    return (
        unittest.mock.patch.object(pipeline, "source_info", inputs.SourceInfo(kind, label)),
        unittest.mock.patch.object(type(pipeline), "is_running", new_callable=unittest.mock.PropertyMock, return_value=True),
        unittest.mock.patch.object(pipeline, "camera_ok", camera_ok),
    )


def test_api_camera_opens_once_with_the_chosen_index(client):
    with unittest.mock.patch.object(pipeline, "switch_source") as sw:
        assert client.post("/api/source/camera?index=1").status_code == 200
    overrides, info = sw.call_args.args
    assert overrides["capture"]["source"] == 1 and info.label == "Camera 1"
    assert sw.call_args.kwargs["require_open"] is True


def test_api_missing_camera_is_a_422_not_a_crash(client):
    with unittest.mock.patch.object(pipeline, "switch_source", side_effect=SourceUnavailable("Could not open Camera 0")):
        r = client.post("/api/source/camera")
    assert r.status_code == 422 and "Could not open" in r.json()["detail"]


def test_api_same_working_camera_is_a_noop_but_a_lost_one_retries(client):
    a, b, c = _running(camera_ok=True)
    with a, b, c, unittest.mock.patch.object(pipeline, "switch_source") as sw:
        assert client.post("/api/source/camera?index=0").status_code == 200
        assert not sw.called
    a, b, c = _running(camera_ok=False)
    with a, b, c, unittest.mock.patch.object(pipeline, "switch_source") as sw:
        assert client.post("/api/source/camera?index=0").status_code == 200
        assert sw.called                                   # lost -> the button forces a reopen


def test_api_other_camera_index_switches(client):
    a, b, c = _running(camera_ok=True)
    with a, b, c, unittest.mock.patch.object(pipeline, "switch_source") as sw:
        client.post("/api/source/camera?index=2")
    assert sw.called and sw.call_args.args[0]["capture"]["source"] == 2


def test_source_status_reports_camera_health(client):
    assert "camera_ok" in client.get("/api/source").json()


def test_fake_camera_end_to_end_fits_roi_streams_and_releases_on_stop(tmp_path):
    """Real pipeline + fake 640x360 camera: ROI fitted to the frame, frames flow, device freed on stop."""
    from billetvision import synthetic as S
    from billetvision.pipeline import _deep_merge

    frame = cv2.resize(S.background(np.random.default_rng(1)), (640, 360))
    cap = FakeCap([True] * 100000)
    cap.read = lambda: (True, frame.copy())            # endless frames at the camera's own pace
    ov = _deep_merge(inputs.camera_overrides(0), {"logging": {
        "db_path": str(tmp_path / "b.db"), "csv_path": str(tmp_path / "b.csv"),
        "xlsx_path": str(tmp_path / "b.xlsx"), "snapshot_dir": str(tmp_path / "s")}})
    pipe = BilletVisionPipeline("config/config.yaml")
    with unittest.mock.patch.object(fs.cv2, "VideoCapture", return_value=cap):
        pipe.switch_source(ov, inputs.SourceInfo("camera", "Camera 0"), require_open=True)
        try:
            end = time.time() + 30
            while pipe.frames_done < 5 and time.time() < end:
                time.sleep(0.1)
            assert pipe.frames_done >= 5 and pipe.source_opened and pipe.camera_ok
            assert pipe._roi[2] <= 640 and pipe._roi[3] <= 360 and pipe._roi_fitted
        finally:
            pipe.stop()
    assert cap.released


# ---- warm-up: a camera that fails its first reads while waking up must not be declared dead ----

def test_slow_waking_camera_is_not_given_up_on():
    """60 failed reads (~1.8 s) before the first frame: the old ~1 s limit killed the capture thread."""
    cap = FakeCap([False] * 60 + [True] * 5)
    src = _source(cap)
    assert src.warming_up                           # open, no frame yet
    assert src.wait_first_frame(10)                 # ...and it does wake up
    assert _drain(src, 3) >= 3 and not src.warming_up
    src.release()


def test_wait_first_frame_times_out_for_a_camera_that_never_delivers():
    src = _source(FakeCap([False] * 100000))
    assert src.warming_up and not src.wait_first_frame(0.3)
    src.release()


def test_no_lost_alert_or_reconnect_while_the_camera_is_still_warming_up(tmp_path):
    pipe = BilletVisionPipeline("config/config.yaml")
    pipe._tracker = unittest.mock.Mock()
    pipe._source = SimpleNamespace(mode="webcam", warming_up=True, is_alive=lambda: True)
    with unittest.mock.patch.object(pipe, "_reconnect") as rec:
        last = pipe._on_no_frame(now=100.0, last_frame_t=90.0, last_reconnect=0.0)   # 10 s of silence
    assert last == 0.0 and not rec.called and pipe.camera_ok
    assert not pipe.alert_manager.history


def test_camera_that_opens_but_never_streams_rolls_back_with_a_clear_error():
    pipe = BilletVisionPipeline("config/config.yaml", overrides={"marker": "old"})
    calls = []

    def fake_start(*a, **k):
        calls.append(dict(pipe.overrides))
        pipe._source = SimpleNamespace(opened=True, wait_first_frame=lambda t: len(calls) > 1)

    with unittest.mock.patch.object(pipe, "start", fake_start), unittest.mock.patch.object(pipe, "stop"):
        with pytest.raises(SourceUnavailable, match="no frames"):
            pipe.switch_source(inputs.camera_overrides(0), inputs.SourceInfo("camera", "Camera 0"), require_open=True)
    assert pipe.overrides == {"marker": "old"}


def test_api_falls_back_to_directshow_when_the_default_backend_gives_nothing(client):
    seen = []

    def fake_switch(overrides, info, require_open=False):
        seen.append(overrides["capture"]["backend"])
        if overrides["capture"]["backend"] is None:
            raise SourceUnavailable("Camera 0 opened but delivered no frames")

    with unittest.mock.patch("billetvision.api.main._camera_backends", return_value=[None, "dshow"]), \
            unittest.mock.patch.object(pipeline, "switch_source", side_effect=fake_switch):
        assert client.post("/api/source/camera").status_code == 200
    assert seen == [None, "dshow"]


def test_api_reports_the_error_when_every_backend_fails(client):
    with unittest.mock.patch("billetvision.api.main._camera_backends", return_value=[None, "dshow"]), \
            unittest.mock.patch.object(pipeline, "switch_source", side_effect=SourceUnavailable("no frames")):
        r = client.post("/api/source/camera")
    assert r.status_code == 422 and "no frames" in r.json()["detail"]


def test_capture_backend_flag_is_passed_to_opencv():
    cap = FakeCap([True] * 50)
    with unittest.mock.patch.object(fs.cv2, "VideoCapture", return_value=cap) as vc:
        src = FrameSource(source=0, mode="webcam", pace=False, backend="dshow").start()
    assert vc.call_args.args == (0, cv2.CAP_DSHOW)
    src.release()
