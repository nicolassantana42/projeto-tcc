import numpy as np
import pytest

from safeguard.capture import CaptureError, VideoSource


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
