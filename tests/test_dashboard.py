"""Exercise real Streamlit reruns without a camera, weights or network access."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from streamlit.testing.v1 import AppTest

from safeguard.capture import VideoSource
from safeguard.inference import YOLODetector


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "safeguard" / "ui" / "app.py"


def button(app, label):
    """Use visible labels because sidebar buttons have a different tree order."""
    return next(element for element in app.button if element.label == label)


def assert_no_exception(app):
    assert not app.exception, [element.message for element in app.exception]


@pytest.fixture
def dashboard(monkeypatch):
    def forbid_external_resource(*args, **kwargs):
        pytest.fail("This dashboard test must not load a model or open a camera.")

    monkeypatch.setattr(YOLODetector, "load", forbid_external_resource)
    monkeypatch.setattr(VideoSource, "open", forbid_external_resource)
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert_no_exception(app)
    try:
        yield app
    finally:
        # Release the session watchdog even if an assertion fails before Stop.
        runtime = app.session_state["runtime"]
        if runtime is not None:
            runtime.close()
            runtime._watchdog.join(timeout=1)


def test_preview_start_threshold_export_and_stop(dashboard):
    app = dashboard
    assert app.session_state["runtime"] is None
    assert app.session_state["latest_frame"] is None
    assert button(app, "Preparar exportação").disabled
    assert button(app, "■ Parar").disabled

    button(app, "▶ Iniciar").click().run()
    assert_no_exception(app)
    runtime = app.session_state["runtime"]
    assert runtime.illustrative
    assert runtime.capture is None
    assert runtime.pipeline is None
    assert not runtime.closed
    assert button(app, "▶ Iniciar").disabled
    assert not button(app, "■ Parar").disabled
    assert not button(app, "Preparar exportação").disabled
    assert all(metric.value == "—" for metric in app.metric[:3])
    first_frame = app.session_state["latest_result"].frame_index

    # Both widgets rerun the same active session instead of reopening resources.
    app.slider(key="confidence").set_value(.65)
    app.slider(key="iou").set_value(.30)
    app.run()
    assert_no_exception(app)
    assert app.session_state["runtime"] is runtime
    assert app.session_state["latest_result"].frame_index == first_frame + 1
    assert app.session_state["history"][-1]["thresholds"] == {
        "confidence": pytest.approx(.65), "iou": pytest.approx(.30),
    }

    button(app, "Preparar exportação").click().run()
    assert_no_exception(app)
    export = app.session_state["export"]
    assert export["snapshot"].startswith(b"\x89PNG\r\n\x1a\n")
    with ZipFile(io.BytesIO(export["report"])) as archive:
        assert set(archive.namelist()) == {"report.json", "detections.csv", "snapshot.png"}
        assert archive.read("snapshot.png") == export["snapshot"]
        document = json.loads(archive.read("report.json"))
    assert document["metadata"]["mode"] == "illustrative_preview"
    assert document["metadata"]["model"] is None
    assert document["metadata"]["confidence_at_export"] == pytest.approx(.65)
    assert document["metadata"]["iou_at_export"] == pytest.approx(.30)
    assert document["retained_frames"] == len(app.session_state["history"])
    assert 0 < document["retained_frames"] <= 300
    for frame in document["frames"]:
        assert frame["mode"] == "illustrative_preview"
        assert frame["inference_ms"] is None
        assert frame["pipeline_ms"] is None
        assert frame["observed_fps"] is None
        assert all(item["confidence"] is None for item in frame["detections"])

    last_frame = app.session_state["latest_result"].frame_index
    button(app, "■ Parar").click().run()
    assert_no_exception(app)
    assert app.session_state["runtime"] is None
    assert runtime.closed
    runtime._watchdog.join(timeout=1)
    assert not runtime._watchdog.is_alive()
    assert app.session_state["latest_result"].frame_index == last_frame
    assert app.session_state["export"]["report"] == export["report"]
    assert not button(app, "▶ Iniciar").disabled
    assert button(app, "■ Parar").disabled


def test_missing_video_upload_is_actionable_and_does_not_start(dashboard):
    app = dashboard
    source = next(element for element in app.selectbox if element.label == "Fonte de entrada")
    source.select("Arquivo de vídeo").run()
    assert_no_exception(app)

    button(app, "▶ Iniciar").click().run()
    assert_no_exception(app)
    assert app.session_state["runtime"] is None
    assert app.session_state["latest_frame"] is None
    assert not app.session_state["history"]
    assert "Selecione um arquivo de vídeo antes de iniciar." in app.session_state["notice"]
    assert any("Selecione um arquivo de vídeo" in element.value for element in app.info)
    assert not button(app, "▶ Iniciar").disabled
    assert button(app, "■ Parar").disabled
    assert button(app, "Preparar exportação").disabled
