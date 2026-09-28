"""No test in this module contacts n8n or any external network."""

import io
import threading
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

import safeguard.notifications as notifications
from safeguard.notifications import (
    N8nConfig,
    N8nWebhookSender,
    NotificationDispatcher,
    NotificationError,
    event_caption,
)


WEBHOOK = "https://n8n.example.com/webhook/secret-path-segment"
WEBHOOK_SECRET = "secret-path-segment"


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in notification tests.")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def event(tmp_path):
    snapshot = tmp_path / "retained.jpg"
    Image.new("RGB", (640, 480), (32, 48, 64)).save(snapshot)
    return {
        "id": "20260918-event-001",
        "timestamp_utc": "2026-09-18T12:00:00+00:00",
        "camera_id": "camera-1",
        "camera_name": "Portaria",
        "location": "Galpão A / entrada norte",
        "kind": "ppe",
        "reasons": ["Sem capacete"],
        "model_mode": "ppe_model",
        "counts": {"person": 1, "no-helmet": 1},
        "snapshot_path": str(snapshot),
        "metadata_path": str(tmp_path / "retained.json"),
        "detections": [],
    }


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else {"ok": True}
        self.closed = False

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response=None, failure=None):
        self.response = response or FakeResponse()
        self.failure = failure
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.response

    def close(self):
        self.closed = True


def n8n_sender(monkeypatch, response=None, failure=None):
    session = FakeSession(response, failure)
    monkeypatch.setattr(notifications.requests, "Session", lambda: session)
    return N8nWebhookSender(N8nConfig(True, WEBHOOK)), session


def test_disabled_config_requires_nothing_and_hides_webhook_url():
    N8nConfig().validate()
    assert WEBHOOK not in repr(N8nConfig(True, WEBHOOK))


@pytest.mark.parametrize("webhook_url", [
    "http://evil.example.com/hook",
    WEBHOOK + "\n",
    "ftp://n8n.example/hook",
    "",
])
def test_n8n_config_rejects_malformed_webhook_without_echo(webhook_url):
    with pytest.raises(ValueError) as caught:
        N8nConfig(True, webhook_url).validate()
    assert WEBHOOK_SECRET not in str(caught.value)


def test_n8n_sends_small_photo_with_event_location_and_safe_filename(monkeypatch, event):
    sender, session = n8n_sender(monkeypatch)
    original = open(event["snapshot_path"], "rb").read()
    result = sender.send(event)
    assert "Aceito" in result and "n8n" in result
    assert len(session.calls) == 1
    url, args = session.calls[0]
    assert url == WEBHOOK
    assert args["timeout"] == (5.0, 15.0)
    assert args["allow_redirects"] is False
    assert args["data"]["event_id"] == event["id"]
    assert "Galpão A / entrada norte" in args["data"]["location"]
    assert "Portaria" in args["data"]["camera_name"]
    assert "Sem capacete" in args["data"]["reasons"]
    assert "2026-09-18T12:00:00+00:00" in args["data"]["timestamp_utc"]
    assert event["snapshot_path"] not in args["data"]["caption"]
    filename, photo, mime = args["files"]["photo"]
    assert filename == "safeguard-event.jpg" and mime == "image/jpeg"
    assert len(photo) <= 10_000_000
    assert Image.open(io.BytesIO(photo)).format == "JPEG"
    assert open(event["snapshot_path"], "rb").read() == original
    assert session.response.closed
    sender.close()
    assert session.closed


def test_caption_remains_bounded_with_unicode_and_long_reasons(event):
    event.update(location="😀" * 1000, reasons=["Ocorrência " * 1000])
    caption = event_caption(event)
    assert len(caption.encode("utf-16-le")) <= 2048
    assert "Câmera: Portaria" in caption
    assert "Data/hora UTC:" in caption


def test_oversized_dimensions_and_aspect_ratio_are_normalized(monkeypatch, event):
    Image.new("RGB", (12000, 30), (10, 20, 30)).save(event["snapshot_path"])
    sender, session = n8n_sender(monkeypatch)
    sender.send(event)
    photo = Image.open(io.BytesIO(session.calls[0][1]["files"]["photo"][1]))
    assert max(photo.size) <= 1920
    assert sum(photo.size) <= 10000
    assert max(photo.size) / min(photo.size) <= 20


@pytest.mark.parametrize("variant", ["missing", "relative", "corrupt"])
def test_invalid_local_snapshot_is_not_sent(monkeypatch, event, variant):
    sender, session = n8n_sender(monkeypatch)
    if variant == "missing":
        event["snapshot_path"] += ".missing"
    elif variant == "relative":
        event["snapshot_path"] = "relative.jpg"
    else:
        with open(event["snapshot_path"], "wb") as target:
            target.write(b"not an image")
    with pytest.raises(NotificationError, match="Snapshot"):
        sender.send(event)
    assert not session.calls


@pytest.mark.parametrize("response,code", [
    (FakeResponse(401), "webhook_auth"),
    (FakeResponse(403), "webhook_auth"),
    (FakeResponse(429), "webhook_rate"),
    (FakeResponse(302), "webhook_rejected"),
    (FakeResponse(400), "webhook_rejected"),
    (FakeResponse(payload={"ok": False, "description": WEBHOOK_SECRET}), "webhook_response"),
    (FakeResponse(payload={"accepted": False, "detail": WEBHOOK_SECRET}), "webhook_response"),
])
def test_n8n_rejects_provider_errors_without_leaking_or_retrying(monkeypatch, event, response, code):
    sender, session = n8n_sender(monkeypatch, response=response)
    with pytest.raises(NotificationError) as caught:
        sender.send(event)
    assert caught.value.code == code
    assert WEBHOOK_SECRET not in str(caught.value)
    assert "https://" not in str(caught.value)
    assert len(session.calls) == 1


@pytest.mark.parametrize("failure,code", [
    (requests.Timeout(WEBHOOK), "webhook_timeout"),
    (requests.ConnectionError(WEBHOOK_SECRET), "webhook_network"),
])
def test_n8n_network_errors_are_safe_and_not_retried(monkeypatch, event, failure, code):
    sender, session = n8n_sender(monkeypatch, failure=failure)
    with pytest.raises(NotificationError) as caught:
        sender.send(event)
    assert caught.value.code == code
    assert WEBHOOK_SECRET not in str(caught.value)
    assert len(session.calls) == 1


def test_disabled_sender_does_not_attempt_network(event):
    with pytest.raises(NotificationError, match="desativado"):
        N8nWebhookSender(N8nConfig()).send(event)


class MemoryStore:
    def __init__(self):
        self.history = []

    def set_delivery(self, event_id, channel, status, detail=""):
        self.history.append((event_id, channel, status, detail))


def test_dispatcher_records_n8n_failure_without_leaking_secrets(event):
    store = MemoryStore()

    def fail(item):
        raise RuntimeError(WEBHOOK_SECRET)

    dispatcher = NotificationDispatcher({"n8n": SimpleNamespace(send=fail)}, store)
    assert dispatcher.enqueue(event)
    assert not dispatcher.enqueue(event)
    assert dispatcher.close(wait=True)
    assert {row[1]: row[2] for row in store.history} == {"n8n": "failed"}
    assert WEBHOOK_SECRET not in repr(store.history)
    assert dispatcher.snapshot()["pending"] == 0
    assert dispatcher.snapshot()["failed"] == 1


def test_queue_full_is_explicit_and_close_preserves_pending_work(event):
    started, release = threading.Event(), threading.Event()
    store, received = MemoryStore(), []

    def slow_send(item):
        started.set()
        assert release.wait(timeout=2)
        received.append(item["id"])
        return "accepted"

    dispatcher = NotificationDispatcher({"n8n": SimpleNamespace(send=slow_send)}, store, max_queue=1)
    try:
        assert dispatcher.enqueue(event)
        assert started.wait(timeout=1)
        assert dispatcher.enqueue({**event, "id": "second"})
        assert not dispatcher.enqueue({**event, "id": "overflow"})
        assert any(row[0] == "overflow" and row[2] == "failed" and "queue_full" in row[3] for row in store.history)
        assert not dispatcher.close(wait=False)
        assert not dispatcher.enqueue({**event, "id": "closed"})
        release.set()
        assert dispatcher.close(wait=True)
        assert received == [event["id"], "second"]
        assert dispatcher.snapshot()["pending"] == 0
        assert not dispatcher.snapshot()["worker_alive"]
    finally:
        release.set()
        dispatcher.close(wait=True)


def test_queue_copies_event_before_caller_mutates_it(event):
    started, release = threading.Event(), threading.Event()
    received = []

    def collect(item):
        started.set()
        assert release.wait(timeout=2)
        received.append(item["reasons"][:])
        return "accepted"

    dispatcher = NotificationDispatcher({"n8n": SimpleNamespace(send=collect)}, MemoryStore())
    try:
        assert dispatcher.enqueue(event)
        assert started.wait(timeout=1)
        event["reasons"][0] = "Mutated"
        release.set()
        assert dispatcher.close(wait=True)
        assert received == [["Sem capacete"]]
    finally:
        release.set()
        dispatcher.close(wait=True)


def test_close_drains_event_accepted_after_worker_queue_timeout(monkeypatch, event):
    timed_out, release = threading.Event(), threading.Event()
    original_queue = notifications.queue.Queue

    class PausingQueue(original_queue):
        first_get = True

        def get(self, *args, **kwargs):
            if self.first_get:
                self.first_get = False
                timed_out.set()
                assert release.wait(timeout=2)
                raise notifications.queue.Empty
            return super().get(*args, **kwargs)

    monkeypatch.setattr(notifications.queue, "Queue", PausingQueue)
    received, store = [], MemoryStore()
    sender = SimpleNamespace(send=lambda item: received.append(item["id"]))
    dispatcher = NotificationDispatcher({"n8n": sender}, store)
    try:
        assert timed_out.wait(timeout=1)
        assert dispatcher.enqueue(event)
        assert not dispatcher.close(wait=False)
        release.set()
        assert dispatcher.close(wait=True)
        assert received == [event["id"]]
        assert dispatcher.snapshot()["queued"] == 0
        assert dispatcher.snapshot()["pending"] == 0
        assert store.history[-1][2] == "accepted"
    finally:
        release.set()
        dispatcher.close(wait=True)


def test_store_failure_is_visible_and_does_not_kill_worker(event):
    def fail_store(*args, **kwargs):
        raise OSError("disk error")

    dispatcher = NotificationDispatcher(
        {"n8n": SimpleNamespace(send=lambda item: "accepted")},
        SimpleNamespace(set_delivery=fail_store),
    )
    assert dispatcher.enqueue(event)
    assert dispatcher.close(wait=True)
    assert dispatcher.snapshot()["store_errors"] == 2
    assert dispatcher.snapshot()["accepted"] == 1


def test_no_channels_means_no_queueing(event):
    store = MemoryStore()
    dispatcher = NotificationDispatcher({}, store)
    assert not dispatcher.enqueue(event)
    assert dispatcher.close(wait=True)
    assert not store.history


def test_notification_error_only_exposes_allowlisted_detail():
    assert WEBHOOK_SECRET not in str(NotificationError(WEBHOOK_SECRET))
    assert NotificationError(WEBHOOK_SECRET).code == "unknown"
