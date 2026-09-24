import numpy as np
import pytest
import cv2

from safeguard.capture import CaptureError, ImageSource, VideoSource, open_source


class FakeCapture:
    def __init__(self, frames=(), opened=True, total=0):
        self.frames = iter(frames)
        self.opened = opened
        self.total = total
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def get(self, _):
        return self.total

    def release(self):
        self.released = True


def _video(tmp_path):
    path = tmp_path / "video.mp4"
    path.touch()
    return str(path)


def test_file_returns_none_at_eof_and_releases(monkeypatch, tmp_path):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    capture = FakeCapture([frame], total=1)
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: capture)
    with VideoSource(_video(tmp_path)) as source:
        assert source.read() is frame
        assert source.read() is None
        assert source.read() is None
    assert capture.released


def test_disconnected_camera_is_not_eof(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: capture)
    with VideoSource(0) as source:
        with pytest.raises(CaptureError, match="desconectado"):
            source.read()
    assert capture.released


def test_network_open_and_read_have_timeouts_and_errors_hide_credentials(monkeypatch):
    calls = []

    def create_capture(*args):
        calls.append(args)
        return FakeCapture(opened=False)

    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", create_capture)
    with pytest.raises(CaptureError) as error:
        VideoSource("rtsp://admin:secret@camera.example/live", timeout_ms=1200).open()
    assert len(calls[0]) == 3
    assert calls[0][2][1::2] == [1200, 1200]
    assert "secret" not in str(error.value)
    assert "camera.example" not in str(error.value)


def test_corrupted_or_empty_file_raises_readable_error(monkeypatch, tmp_path):
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: FakeCapture())
    with VideoSource(_video(tmp_path)) as source:
        with pytest.raises(CaptureError, match="corrompido"):
            source.read()


def test_truncated_file_is_not_silently_treated_as_completed(monkeypatch, tmp_path):
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: FakeCapture([frame], total=30))
    with VideoSource(_video(tmp_path)) as source:
        source.read()
        with pytest.raises(CaptureError, match="antes do esperado"):
            source.read()


def test_missing_file_and_unopened_read_are_actionable(tmp_path):
    with pytest.raises(CaptureError, match="não encontrado"):
        VideoSource(str(tmp_path / "missing.mp4")).open()
    with pytest.raises(CaptureError, match="não foi aberta"):
        VideoSource(0).read()


@pytest.mark.parametrize("source", ["", " ", -1, True, None])
def test_invalid_source_rejected(source):
    with pytest.raises(ValueError):
        VideoSource(source)


def test_capture_release_even_when_processing_raises(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: capture)
    with pytest.raises(RuntimeError):
        with VideoSource("0"):
            raise RuntimeError("processing failed")
    assert capture.released


def test_image_source_reads_unicode_filename_once_and_can_reopen(tmp_path):
    path = tmp_path / "inspeção_工人.PNG"
    frame = np.full((20, 30, 3), 123, np.uint8)
    success, encoded = cv2.imencode(".png", frame)
    assert success
    encoded.tofile(str(path))
    source = open_source(str(path))
    assert isinstance(source, ImageSource)
    with source:
        np.testing.assert_array_equal(source.read(), frame)
        assert source.timestamp_seconds == 0
        assert source.read() is None
    with pytest.raises(CaptureError, match="não foi aberta"):
        source.read()
    with source:
        np.testing.assert_array_equal(source.read(), frame)


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_image_source_rejects_empty_or_corrupt_data(tmp_path, payload):
    path = tmp_path / "bad.jpg"
    path.write_bytes(payload)
    source = ImageSource(str(path))
    with pytest.raises(CaptureError, match="corrompida"):
        source.open()
    with pytest.raises(CaptureError, match="não foi aberta"):
        source.read()


def test_missing_image_is_a_clear_capture_error(tmp_path):
    with pytest.raises(CaptureError, match="ler a imagem"):
        ImageSource(str(tmp_path / "missing.png")).open()


def test_remote_image_suffix_does_not_select_local_image_loader():
    source = open_source("https://camera.example/snapshot.png")
    assert isinstance(source, VideoSource) and source.is_stream


class TimedCapture(FakeCapture):
    def __init__(self, positions, fps):
        super().__init__([np.zeros((20, 30, 3), np.uint8)] * len(positions), total=len(positions))
        self.positions, self.fps = iter(positions), fps

    def get(self, key):
        if key == cv2.CAP_PROP_FRAME_COUNT:
            return self.total
        if key == cv2.CAP_PROP_FPS:
            return self.fps
        if key == cv2.CAP_PROP_POS_MSEC:
            value = next(self.positions)
            if isinstance(value, Exception):
                raise value
            return value
        return 0


@pytest.mark.parametrize("positions,fps,expected", [
    ([0, 100, 200], 30, [0., .1, .2]),
    ([0, 0, 0], 25, [0., .04, .08]),
    ([float("nan"), -1, None], 20, [0., .05, .1]),
    ([0, cv2.error("unsupported"), 0], 10, [0., .1, .2]),
    ([0, 100, 50, 0], 10, [0., .1, .2, .3]),
    ([0, 0, 100], 0, [0., None, .1]),
    ([float("nan"), float("nan")], None, [None, None]),
])
def test_file_timestamps_use_media_clock_or_fps_fallback(monkeypatch, tmp_path, positions, fps, expected):
    capture = TimedCapture(positions, fps)
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: capture)
    timestamps = []
    with VideoSource(_video(tmp_path)) as source:
        for _ in positions:
            source.read()
            timestamps.append(source.timestamp_seconds)
    assert timestamps == expected and capture.released


def test_live_camera_leaves_timestamp_none_for_monotonic_event_clock(monkeypatch):
    capture = FakeCapture([np.zeros((20, 30, 3), np.uint8)])
    monkeypatch.setattr("safeguard.capture.cv2.VideoCapture", lambda *_: capture)
    with VideoSource(0) as source:
        source.read()
        assert source.timestamp_seconds is None
