"""No test in this module contacts Telegram or any SMTP service."""

import base64
import io
import ssl
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

import safeguard.notifications as notifications
from safeguard.notifications import (
    EmailConfig,
    EmailSender,
    NotificationDispatcher,
    NotificationError,
    TelegramConfig,
    TelegramSender,
    event_caption,
)


TOKEN = "123456:secret_bot_token"
PASSWORD = "secret_smtp_password"
ACCESS_TOKEN = "secret_access_token"


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in notification tests.")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(notifications.smtplib, "SMTP", forbidden)
    monkeypatch.setattr(notifications.smtplib, "SMTP_SSL", forbidden)


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
        self.payload = payload if payload is not None else {"ok": True, "result": {"message_id": 42}}
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


def telegram_sender(monkeypatch, response=None, failure=None):
    session = FakeSession(response, failure)
    monkeypatch.setattr(notifications.requests, "Session", lambda: session)
    return TelegramSender(TelegramConfig(True, TOKEN, "-100123456789")), session


def email_config(**changes):
    return replace(EmailConfig(
        enabled=True, host="smtp.example.com", username="sender@example.com",
        password=PASSWORD, sender="sender@example.com", recipient="security@example.com",
    ), **changes)


class FakeSMTP:
    def __init__(self, failure_at=None, failure=None, auth_code=235, refused=None):
        self.operations = []
        self.failure_at = failure_at
        self.failure = failure
        self.auth_code = auth_code
        self.refused = refused or {}
        self.message = None
        self.closed = False

    def _call(self, name, *args, **kwargs):
        self.operations.append((name, args, kwargs))
        if self.failure_at == name:
            raise self.failure

    def ehlo(self):
        self._call("ehlo")
        return 250, b"ok"

    def starttls(self, **kwargs):
        self._call("starttls", **kwargs)
        return 220, b"ok"

    def login(self, *args):
        self._call("login", *args)

    def docmd(self, *args):
        self._call("docmd", *args)
        return self.auth_code, b"Provider text must never be logged"

    def send_message(self, message, **kwargs):
        self._call("send_message", **kwargs)
        self.message = message
        return self.refused

    def close(self):
        self._call("close")
        self.closed = True


def smtp_factory(monkeypatch, client, secure=False):
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return client

    monkeypatch.setattr(notifications.smtplib, "SMTP_SSL" if secure else "SMTP", factory)
    return calls


def test_disabled_configs_require_nothing_and_hide_credentials():
    TelegramConfig().validate()
    EmailConfig().validate()
    assert TOKEN not in repr(TelegramConfig(True, TOKEN, "123456"))
    assert PASSWORD not in repr(email_config())
    assert "sender@example.com" not in repr(email_config())
    assert ACCESS_TOKEN not in repr(email_config(access_token=ACCESS_TOKEN))


@pytest.mark.parametrize("token,chat_id", [
    ("https://fake/token", "123"), (TOKEN + "\n", "123"),
    (TOKEN, "https://chat"), (TOKEN, "123\r\nfoo"), (TOKEN, "0"),
])
def test_telegram_config_rejects_malformed_credentials_without_echo(token, chat_id):
    with pytest.raises(ValueError) as caught:
        TelegramConfig(True, token, chat_id).validate()
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize("changes", [
    {"recipient": "one@example.com,two@example.com"},
    {"recipient": "x@example.com\r\nBcc: extra@example.com"},
    {"sender": "Somebody <one@example.com>"},
    {"username": "user\x01injection"},
    {"host": "smtp://example.com"},
    {"security": "none"},
    {"port": 0},
    {"port": True},
    {"auth_mode": "unknown"},
    {"auth_mode": "oauth2", "access_token": "token with space"},
])
def test_email_configuration_rejects_unsafe_values(changes):
    with pytest.raises(ValueError):
        email_config(**changes).validate()


@pytest.mark.parametrize("host", ["smtp.office365.com", "smtp-mail.outlook.com", "SMTP.OFFICE365.COM."])
def test_outlook_requires_oauth2(host):
    with pytest.raises(ValueError, match="OAuth2"):
        email_config(host=host).validate()
    email_config(host=host, auth_mode="oauth2", access_token=ACCESS_TOKEN).validate()


def test_telegram_sends_small_photo_with_event_location_and_safe_filename(monkeypatch, event):
    sender, session = telegram_sender(monkeypatch)
    original = open(event["snapshot_path"], "rb").read()
    result = sender.send(event)
    assert "Aceito" in result and "42" in result
    assert len(session.calls) == 1
    url, args = session.calls[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    assert args["timeout"] == (5.0, 15.0)
    assert args["allow_redirects"] is False
    assert args["data"]["chat_id"] == "-100123456789"
    assert "Galpão A / entrada norte" in args["data"]["caption"]
    assert "Portaria" in args["data"]["caption"]
    assert "Sem capacete" in args["data"]["caption"]
    assert "2026-09-18T12:00:00+00:00" in args["data"]["caption"]
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
    sender, session = telegram_sender(monkeypatch)
    sender.send(event)
    photo = Image.open(io.BytesIO(session.calls[0][1]["files"]["photo"][1]))
    assert max(photo.size) <= 1920
    assert sum(photo.size) <= 10000
    assert max(photo.size) / min(photo.size) <= 20


@pytest.mark.parametrize("variant", ["missing", "relative", "corrupt"])
def test_invalid_local_snapshot_is_not_sent(monkeypatch, event, variant):
    sender, session = telegram_sender(monkeypatch)
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
    (FakeResponse(401), "telegram_auth"),
    (FakeResponse(403), "telegram_auth"),
    (FakeResponse(429), "telegram_rate"),
    (FakeResponse(302), "telegram_rejected"),
    (FakeResponse(400), "telegram_rejected"),
    (FakeResponse(payload={"ok": False, "description": TOKEN}), "telegram_response"),
    (FakeResponse(payload={"ok": True, "result": {"message_id": TOKEN}}), "telegram_response"),
    (FakeResponse(payload={"ok": True, "result": {"message_id": True}}), "telegram_response"),
    (FakeResponse(payload=ValueError(TOKEN)), "telegram_response"),
])
def test_telegram_rejects_provider_errors_without_leaking_or_retrying(monkeypatch, event, response, code):
    sender, session = telegram_sender(monkeypatch, response=response)
    with pytest.raises(NotificationError) as caught:
        sender.send(event)
    assert caught.value.code == code
    assert TOKEN not in str(caught.value)
    assert "https://" not in str(caught.value)
    assert len(session.calls) == 1


@pytest.mark.parametrize("failure,code", [
    (requests.Timeout(f"https://api.telegram.org/bot{TOKEN}"), "telegram_timeout"),
    (requests.ConnectionError(TOKEN), "telegram_network"),
])
def test_telegram_network_errors_are_safe_and_not_retried(monkeypatch, event, failure, code):
    sender, session = telegram_sender(monkeypatch, failure=failure)
    with pytest.raises(NotificationError) as caught:
        sender.send(event)
    assert caught.value.code == code
    assert TOKEN not in str(caught.value)
    assert len(session.calls) == 1


def test_disabled_senders_do_not_attempt_network(event):
    with pytest.raises(NotificationError, match="desativado"):
        TelegramSender(TelegramConfig()).send(event)
    with pytest.raises(NotificationError, match="desativado"):
        EmailSender(EmailConfig()).send(event)


def test_email_uses_verified_starttls_before_auth_and_attaches_image(monkeypatch, event):
    client = FakeSMTP()
    factory_calls = smtp_factory(monkeypatch, client)
    result = EmailSender(email_config()).send(event)
    assert "Aceito" in result
    names = [operation[0] for operation in client.operations]
    assert names == ["ehlo", "starttls", "ehlo", "login", "send_message", "close"]
    context = client.operations[1][2]["context"]
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert factory_calls[0][1]["timeout"] == 15.0
    assert "Galpão A / entrada norte" in client.message.get_body().get_content()
    attachments = list(client.message.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_content_type() == "image/jpeg"
    assert attachments[0].get_filename() == "safeguard-event.jpg"
    assert client.closed


def test_email_ssl_and_outlook_xoauth2_do_not_use_password_login(monkeypatch, event):
    client = FakeSMTP()
    calls = smtp_factory(monkeypatch, client, secure=True)
    config = email_config(host="smtp.office365.com", security="ssl", port=465,
                          auth_mode="oauth2", access_token=ACCESS_TOKEN)
    EmailSender(config).send(event)
    names = [operation[0] for operation in client.operations]
    assert names == ["ehlo", "docmd", "send_message", "close"]
    command = next(operation for operation in client.operations if operation[0] == "docmd")
    assert command[1][0] == "AUTH"
    mechanism, encoded = command[1][1].split(" ", 1)
    assert mechanism == "XOAUTH2"
    assert base64.b64decode(encoded).decode() == f"user=sender@example.com\x01auth=Bearer {ACCESS_TOKEN}\x01\x01"
    assert calls[0][1]["context"].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("failure_at,failure,code", [
    ("starttls", notifications.smtplib.SMTPNotSupportedError(PASSWORD), "email_tls"),
    ("login", notifications.smtplib.SMTPAuthenticationError(535, PASSWORD.encode()), "email_auth"),
    ("send_message", TimeoutError(PASSWORD), "email_timeout"),
    ("send_message", notifications.smtplib.SMTPRecipientsRefused({"security@example.com": (550, PASSWORD)}), "email_recipient"),
])
def test_email_error_messages_do_not_include_server_or_secret_text(monkeypatch, event, failure_at, failure, code):
    client = FakeSMTP(failure_at=failure_at, failure=failure)
    smtp_factory(monkeypatch, client)
    with pytest.raises(NotificationError) as caught:
        EmailSender(email_config()).send(event)
    assert caught.value.code == code
    assert PASSWORD not in str(caught.value)
    assert client.closed
    assert [operation[0] for operation in client.operations].count("send_message") <= 1


def test_outlook_oauth_error_is_not_misreported_as_success(monkeypatch, event):
    client = FakeSMTP(auth_code=334)
    smtp_factory(monkeypatch, client)
    with pytest.raises(NotificationError, match="autenticação"):
        EmailSender(email_config(host="smtp.office365.com", auth_mode="oauth2", access_token=ACCESS_TOKEN)).send(event)
    assert client.message is None and client.closed


def test_smtp_recipient_refusal_dictionary_is_failure(monkeypatch, event):
    client = FakeSMTP(refused={"security@example.com": (550, "denied")})
    smtp_factory(monkeypatch, client)
    with pytest.raises(NotificationError) as caught:
        EmailSender(email_config()).send(event)
    assert caught.value.code == "email_recipient"


class MemoryStore:
    def __init__(self):
        self.history = []

    def set_delivery(self, event_id, channel, status, detail=""):
        self.history.append((event_id, channel, status, detail))


def test_dispatcher_independent_channels_safe_errors_and_no_retry(event):
    store = MemoryStore()
    attempts = []

    def fail(item):
        attempts.append("telegram")
        raise RuntimeError(TOKEN)

    def accept(item):
        attempts.append("email")
        return PASSWORD

    dispatcher = NotificationDispatcher({"telegram": SimpleNamespace(send=fail), "email": SimpleNamespace(send=accept)}, store)
    assert dispatcher.enqueue(event)
    assert not dispatcher.enqueue(event)
    assert dispatcher.close(wait=True)
    assert attempts == ["telegram", "email"]
    assert {row[1]: row[2] for row in store.history} == {"telegram": "failed", "email": "accepted"}
    assert TOKEN not in repr(store.history) and PASSWORD not in repr(store.history)
    assert dispatcher.snapshot()["pending"] == 0
    assert dispatcher.snapshot()["failed"] == 1
    assert dispatcher.snapshot()["accepted"] == 1


def test_queue_full_is_explicit_and_close_preserves_pending_work(event):
    started, release = threading.Event(), threading.Event()
    store, received = MemoryStore(), []

    def slow_send(item):
        started.set()
        assert release.wait(timeout=2)
        received.append(item["id"])
        return "accepted"

    dispatcher = NotificationDispatcher({"telegram": SimpleNamespace(send=slow_send)}, store, max_queue=1)
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

    dispatcher = NotificationDispatcher({"telegram": SimpleNamespace(send=collect)}, MemoryStore())
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
    dispatcher = NotificationDispatcher({"telegram": sender}, store)
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


def test_store_failure_is_visible_and_does_not_kill_other_channels(event):
    def fail_store(*args, **kwargs):
        raise OSError("disk error")

    dispatcher = NotificationDispatcher(
        {"telegram": SimpleNamespace(send=lambda item: "accepted")},
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
    assert TOKEN not in str(NotificationError(TOKEN))
    assert NotificationError(TOKEN).code == "unknown"
