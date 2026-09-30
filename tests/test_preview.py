import cv2
import numpy as np
import pytest

from epi_monitor.ui import preview


@pytest.mark.parametrize("shape, expected", [((1080, 1920, 3), (405, 720, 3)),
                                             ((1920, 1080, 3), (720, 405, 3)),
                                             ((480, 640, 3), (480, 640, 3))])
def test_jpeg_preview_bounds_long_edge_and_preserves_evidence(shape, expected):
    original = np.zeros(shape, dtype=np.uint8)
    original[20:40, 30:60] = 180
    before = original.copy()
    content = preview.encode_preview(original)
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == expected
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("rate", [3, 10, 25, 30])
def test_pacer_counts_inference_time_inside_target_period(monkeypatch, rate):
    now = [1.0]
    monkeypatch.setattr(preview, "monotonic", lambda: now[0])
    pacer = preview.AnalysisPacer(rate)
    assert pacer.ready()
    pacer.started()
    now[0] += 0.5 / rate
    assert not pacer.ready()
    # A pipeline taking longer than the target is immediately eligible, instead
    # of being forced to wait another complete period after finishing.
    now[0] += 1 / rate
    assert pacer.ready()
    pacer.started()
    assert not pacer.ready()


@pytest.mark.parametrize("invalid", [True, 0, -1, 5, 30.0, "30"])
def test_invalid_analysis_limit_is_rejected(invalid):
    with pytest.raises(ValueError, match="limite"):
        preview.AnalysisPacer(invalid)


def test_preview_encoding_failure_has_actionable_error(monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(ValueError, match="codificar"):
        preview.encode_preview(np.zeros((10, 10, 3), dtype=np.uint8))
