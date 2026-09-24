"""Exercise real Streamlit reruns without a camera, weights or network access."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest
import requests
import smtplib
from streamlit.testing.v1 import AppTest

from safeguard.capture import VideoSource
from safeguard.inference import YOLODetector
from safeguard.events import EventService, EventStore
from safeguard.types import Detection
from safeguard.ui import alert_panels


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "safeguard" / "ui" / "app.py"


def button(app, label):
    """Use visible labels because sidebar buttons have a different tree order."""
    return next(element for element in app.button if element.label == label)


def assert_no_exception(app):
    assert not app.exception, [element.message for element in app.exception]


@pytest.fixture
def dashboard(monkeypatch, tmp_path):
    def forbid_external_resource(*args, **kwargs):
        pytest.fail("This dashboard test must not load a model or open a camera.")

    monkeypatch.setattr(YOLODetector, "load", forbid_external_resource)
    monkeypatch.setattr(VideoSource, "open", forbid_external_resource)
    monkeypatch.setenv("SAFEGUARD_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(alert_panels, "_secret", lambda section, key, fallback="": fallback)
    monkeypatch.setattr(requests.Session, "request", forbid_external_resource)
    monkeypatch.setattr(smtplib, "SMTP", forbid_external_resource)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbid_external_resource)
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
    app.selectbox(key="source_type").select("Prévia ilustrativa").run()
    assert app.session_state["runtime"] is None
    assert app.session_state["latest_frame"] is None
    assert button(app, "Preparar exportação").disabled
    assert button(app, "■ Parar").disabled
    assert button(app, "Salvar imagem agora").disabled

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
    assert runtime.event_service is None
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
    assert not list(alert_panels.reports_root().joinpath("occurrences").glob("*/event.json"))


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


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def configure_camera(app, *, telegram=False):
    for label, value in {
        "Identificador da câmera": "entrada-07",
        "Nome da câmera": "Câmera da entrada",
        "Local / setor": "Unidade 2 • Galpão A",
        "Token do bot": "123456:FAKE_TEST_TOKEN_NOT_REAL",
        "Chat ID de destino": "-123456789",
        "Access token OAuth2 SMTP": "FAKE_OAUTH_TEST_SECRET",
        "Senha de aplicativo SMTP": "FAKE_SMTP_TEST_SECRET",
    }.items():
        element(app, "text_input", label).set_value(value)
    element(app, "checkbox", "Ativar Telegram para novas ocorrências").set_value(telegram)
    element(app, "selectbox", "Quando registrar uma ocorrência").select("person")
    button(app, "Salvar configurações").click().run()
    assert_no_exception(app)
    assert not app.error


def test_initial_screen_explains_capture_and_channels_stay_disabled(dashboard):
    app = dashboard
    assert app.selectbox(key="source_type").value == "Webcam local"
    assert [tab.label for tab in app.tabs] == ["Monitoramento", "Ocorrências", "Alertas e integrações"]
    assert any("As pessoas detectadas aparecem aqui" in info.value for info in app.info)
    assert app.session_state["runtime"] is None
    assert button(app, "Salvar imagem agora").disabled
    assert button(app, "Enviar teste aos canais ativos").disabled
    assert not app.session_state["telegram_config"].enabled
    assert not app.session_state["email_config"].enabled
    assert element(app, "radio", "Finalidade do modelo").value == "EPI treinado"
    assert not element(app, "checkbox", "Exibir painel completo").value
    assert app.session_state["alert_settings"]["trigger"] == "ppe"


def test_settings_survive_restart_without_secrets_or_channel_activation(dashboard):
    configure_camera(dashboard, telegram=True)
    assert dashboard.session_state["telegram_config"].enabled
    assert not button(dashboard, "Enviar teste aos canais ativos").disabled
    settings_file = alert_panels.reports_root() / "settings.json"
    raw = settings_file.read_text(encoding="utf-8")
    settings = json.loads(raw)
    assert settings["camera_id"] == "entrada-07"
    assert settings["location"] == "Unidade 2 • Galpão A"
    assert settings["telegram_chat_id"] == "-123456789"
    assert set(settings) == set(alert_panels.DEFAULTS)
    for secret in ("FAKE_TEST_TOKEN", "FAKE_OAUTH_TEST_SECRET", "FAKE_SMTP_TEST_SECRET"):
        assert secret not in raw
    assert not list((settings_file.parent / "occurrences").glob("*/event.json"))

    restarted = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert_no_exception(restarted)
    assert restarted.session_state["alert_settings"]["location"] == settings["location"]
    assert restarted.session_state["telegram_config"].token == ""
    assert restarted.session_state["email_config"].password == ""
    assert restarted.session_state["email_config"].access_token == ""
    assert not restarted.session_state["telegram_config"].enabled
    assert button(restarted, "Enviar teste aos canais ativos").disabled


@pytest.fixture
def detected_dashboard(dashboard, monkeypatch):
    """Keep the actual Pipeline/EventService/render/store with fake model input."""
    opened, closed, sent = [], [], []
    frame = np.full((200, 320, 3), 40, dtype=np.uint8)

    class FakeDetector:
        names = {0: "person"}
        device = "cpu"

        def predict(self, image, confidence=None, iou=None):
            assert image.shape == frame.shape
            return [Detection(0, "person", .91, (50, 20, 150, 185))]

    class FakeTelegramSender:
        def __init__(self, config):
            assert config.enabled

        def send(self, event):
            assert Path(event["snapshot_path"]).is_file()
            sent.append(event)
            return "Mock accepted"

    def open_capture(capture):
        opened.append(capture.source)
        capture.timestamp_seconds = 0.
        return capture

    original_process = EventService.process

    def deterministic_process(service, result, annotated_frame, now=None):
        # One simulated second between observations exercises real confirmation
        # and cooldown logic without sleeps or changing the global clock.
        return original_process(service, result, annotated_frame, now=float(result.frame_index))

    monkeypatch.setattr(YOLODetector, "load", lambda detector: FakeDetector())
    monkeypatch.setattr(VideoSource, "open", open_capture)
    monkeypatch.setattr(VideoSource, "read", lambda capture: frame.copy())
    monkeypatch.setattr(VideoSource, "close", lambda capture: closed.append(capture.source))
    monkeypatch.setattr(EventService, "process", deterministic_process)
    monkeypatch.setattr(alert_panels, "TelegramSender", FakeTelegramSender)
    element(dashboard, "radio", "Finalidade do modelo").set_value("Demo COCO").run()
    yield dashboard, opened, closed, sent


def test_detection_saves_annotated_occurrence_with_location_and_mock_telegram(detected_dashboard):
    app, opened, closed, sent = detected_dashboard
    configure_camera(app, telegram=True)
    app.selectbox(key="source_type").select("Arquivo de vídeo").run()
    element(app, "radio", "Abrir vídeo").set_value("Caminho no computador").run()
    element(app, "text_input", "Caminho do vídeo").set_value("fixture-only.avi")
    button(app, "▶ Iniciar").click().run()
    assert_no_exception(app)
    runtime = app.session_state["runtime"]
    assert runtime is not None and not runtime.illustrative
    assert opened == ["fixture-only.avi"]
    assert runtime.last_event is None
    store = EventStore(alert_panels.reports_root() / "occurrences")
    assert store.list_events() == []
    app.run()
    assert store.list_events() == []
    app.run()
    assert_no_exception(app)
    assert runtime.last_event is not None
    dispatcher = runtime.dispatcher
    button(app, "■ Parar").click().run()
    assert dispatcher.close(wait=True, timeout=2)
    assert closed == ["fixture-only.avi"]
    assert len(sent) == 1
    events = store.list_events()
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "person"
    assert event["camera_id"] == "entrada-07"
    assert event["camera_name"] == "Câmera da entrada"
    assert event["location"] == "Unidade 2 • Galpão A"
    assert event["counts"] == {"person": 1}
    assert event["detections"][0]["confidence"] == pytest.approx(.91)
    assert event["deliveries"]["telegram"]["status"] == "accepted"
    stored_image = cv2.imread(event["snapshot_path"])
    assert stored_image is not None and stored_image.shape[0] > 200
    assert np.std(stored_image[:200]) > 0  # Actual annotation, not the flat input.
    assert json.loads(Path(event["metadata_path"]).read_text(encoding="utf-8"))["id"] == event["id"]
    assert app.session_state["latest_result"].counts == {"person": 1}
    assert any("Última imagem salva" in success.value for success in app.success)


def test_manual_snapshot_keeps_captured_camera_after_settings_change(detected_dashboard):
    app, _, _, sent = detected_dashboard
    configure_camera(app)
    element(app, "checkbox", "Salvar ocorrências automaticamente").uncheck()
    button(app, "Salvar configurações").click().run()
    button(app, "▶ Iniciar").click().run()
    assert_no_exception(app)
    assert app.session_state["runtime"].event_service is None
    button(app, "■ Parar").click().run()
    for label, value in {"Identificador da câmera": "outra-camera", "Nome da câmera": "Outra câmera", "Local / setor": "Outro local"}.items():
        element(app, "text_input", label).set_value(value)
    button(app, "Salvar configurações").click().run()
    button(app, "Salvar imagem agora").click().run()
    assert_no_exception(app)
    assert not app.error
    events = EventStore(alert_panels.reports_root() / "occurrences").list_events()
    assert len(events) == 1
    assert events[0]["kind"] == "manual"
    assert events[0]["camera_id"] == "entrada-07"
    assert events[0]["camera_name"] == "Câmera da entrada"
    assert events[0]["location"] == "Unidade 2 • Galpão A"
    assert events[0]["deliveries"] == {}
    assert not sent


def test_storage_failure_is_visible_and_detection_continues(detected_dashboard, monkeypatch):
    app, _, _, sent = detected_dashboard
    configure_camera(app)

    def disk_unavailable(*args, **kwargs):
        raise OSError("Simulated disk full")

    monkeypatch.setattr(EventStore, "save", disk_unavailable)
    button(app, "▶ Iniciar").click().run()
    app.run()
    app.run()
    assert_no_exception(app)
    runtime = app.session_state["runtime"]
    assert runtime is not None and not runtime.closed
    assert any("Falha ao salvar ocorrência" in error.value for error in app.error)
    assert not sent
    previous = app.session_state["latest_result"].frame_index
    app.run()
    assert_no_exception(app)
    assert app.session_state["latest_result"].frame_index == previous + 1
    assert runtime.last_event is None


def test_connection_test_persists_test_image_and_mock_channel_status(detected_dashboard, monkeypatch):
    app, opened, _, sent = detected_dashboard
    configure_camera(app, telegram=True)
    dispatchers = []
    dispatcher_class = alert_panels.NotificationDispatcher

    def tracked_dispatcher(*args, **kwargs):
        dispatcher = dispatcher_class(*args, **kwargs)
        dispatchers.append(dispatcher)
        return dispatcher

    monkeypatch.setattr(alert_panels, "NotificationDispatcher", tracked_dispatcher)
    button(app, "Enviar teste aos canais ativos").click().run()
    assert_no_exception(app)
    assert not app.error
    assert len(dispatchers) == 1
    assert dispatchers[0].close(wait=True, timeout=2)
    assert not opened
    assert len(sent) == 1
    event = EventStore(alert_panels.reports_root() / "occurrences").list_events()[0]
    assert event["kind"] == "manual"
    assert event["location"] == "Unidade 2 • Galpão A"
    assert "Teste de conexão" in event["reasons"][0]
    assert event["detections"] == []
    assert event["deliveries"]["telegram"]["status"] == "accepted"


@pytest.mark.parametrize("negative", [False, True])
def test_image_executes_real_cascade_and_saves_only_explicit_unsafe(dashboard, monkeypatch, tmp_path, negative):
    """Exercise image decoding, factory, two-stage inference and UI persistence."""
    app = dashboard
    image_path = tmp_path / "câmera.png"
    cv2.imencode(".png", np.full((300, 500, 3), 40, dtype=np.uint8))[1].tofile(str(image_path))
    calls, loaded = [], []

    class Detector:
        device = "cpu"

        def __init__(self, stage):
            self.stage = stage
            self.names = {0: "person"} if stage == "person" else {0: "helmet", 1: "vest", 2: "no_helmet"}

        def predict(self, frame, confidence=None, iou=None):
            assert frame.shape == (300, 500, 3)
            calls.append((self.stage, confidence, iou))
            if self.stage == "person":
                return [Detection(0, "person", .95, (40, 20, 160, 280))]
            return [Detection(2 if negative else 0, "no_helmet" if negative else "helmet", .9, (70, 25, 130, 60)),
                    Detection(1, "vest", .9, (65, 100, 135, 180))]

    def load(detector):
        stage = "person" if detector.config.model_path == "models/yolo11n.pt" else "ppe"
        loaded.append(stage)
        return Detector(stage)

    monkeypatch.setattr(YOLODetector, "load", load)
    app.selectbox(key="source_type").select("Imagem").run()
    element(app, "radio", "Abrir imagem").set_value("Caminho no computador").run()
    element(app, "text_input", "Caminho da imagem").set_value(str(image_path))
    button(app, "▶ Iniciar").click().run()
    assert_no_exception(app)
    assert loaded == ["person", "ppe"]
    assert calls == [("person", .4, .45), ("ppe", .4, .45)]
    result = app.session_state["latest_result"]
    assert result.ppe_executed
    assert result.assessments[0].status == ("unsafe" if negative else "ok")
    assert app.session_state["history"][-1]["ppe_executed"]
    events = EventStore(alert_panels.reports_root() / "occurrences").list_events()
    assert len(events) == int(negative)
    if negative:
        assert events[0]["kind"] == "ppe"
        assert "observação única" in events[0]["reasons"][0]
        assert events[0]["deliveries"] == {}
    app.run()  # EOF releases image source and keeps the evidence visible.
    assert_no_exception(app)
    assert app.session_state["runtime"] is None
    assert app.session_state["latest_result"] is result
    assert len(calls) == 2
