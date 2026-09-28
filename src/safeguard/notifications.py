"""Opt-in n8n webhook delivery and a bounded, in-memory queue.

Successful sends mean the webhook accepted the payload, not that Telegram,
e-mail or other downstream channels completed delivery. Credentials and
remote error bodies are deliberately excluded from persisted delivery details.
"""

from __future__ import annotations

import copy
import io
import math
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError


_ERRORS = {
    "disabled": "Canal desativado; nenhum envio realizado.",
    "snapshot": "Snapshot local ausente, inválido ou grande demais para envio.",
    "webhook_timeout": "Tempo limite do webhook n8n; resultado incerto. Verifique a automação antes de reenviar.",
    "webhook_network": "Falha de conexão com o webhook n8n; verifique a rede e o fluxo antes de reenviar.",
    "webhook_auth": "O webhook n8n recusou a autenticação. Revise a URL e credenciais do fluxo.",
    "webhook_rate": "O webhook n8n limitou os envios. Aguarde e revise o intervalo entre alertas.",
    "webhook_rejected": "O webhook n8n recusou a solicitação. Revise a URL e o fluxo.",
    "webhook_response": "O webhook n8n retornou resposta sem confirmação válida de aceitação.",
    "queue_full": "queue_full: fila de alertas cheia; evento salvo, mas envio não agendado.",
    "closed": "dispatcher_closed: sessão encerrada; envio não agendado.",
    "unknown": "Falha no canal de alerta; detalhes remotos omitidos para proteger credenciais.",
}
_MAX_PHOTO_BYTES = 10_000_000
_HTTP_TIMEOUT = (5.0, 15.0)
_ACCEPTED_DETAIL = "Aceito pelo provedor; entrega e leitura não confirmadas."


class NotificationError(RuntimeError):
    """A public, allowlisted error; arbitrary provider text is never retained."""

    def __init__(self, code: str = "unknown") -> None:
        self.code = code if code in _ERRORS else "unknown"
        super().__init__(_ERRORS[self.code])


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _webhook_url(value: str) -> bool:
    if not isinstance(value, str) or _has_control(value):
        return False
    parsed = urlparse(value.strip())
    if parsed.username or parsed.password or not parsed.netloc:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    return False


@dataclass(frozen=True)
class N8nConfig:
    enabled: bool = False
    webhook_url: str = field(default="", repr=False)

    def validate(self) -> None:
        if not self.enabled:
            return
        if not _webhook_url(self.webhook_url):
            raise ValueError("Informe a URL HTTPS do webhook n8n (HTTP somente em localhost).")


def _plain(value: Any, limit: int = 160) -> str:
    return " ".join(str(value or "Não informado").split())[:limit]


def event_caption(event: Mapping[str, Any]) -> str:
    """Describe only approved event fields; never include paths or stream URLs."""
    kind = {"ppe": "Alerta de EPI", "person": "Pessoa detectada", "manual": "Registro manual"}.get(
        event.get("kind"), "Evento de detecção",
    )
    reasons = event.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    counts = event.get("counts") or {}
    if not isinstance(counts, Mapping):
        counts = {}
    text = "\n".join((
        f"SafeGuard | {kind}",
        f"Câmera: {_plain(event.get('camera_name'))}",
        f"Local: {_plain(event.get('location'))}",
        f"Data/hora UTC: {_plain(event.get('timestamp_utc'), 50)}",
        f"Motivo: {_plain('; '.join(str(item) for item in reasons), 300)}",
        f"Modo: {_plain(event.get('model_mode'), 60)}",
        f"Detecções no frame: {_plain(', '.join(f'{label}: {count}' for label, count in counts.items()), 150)}",
        f"Evento: {_plain(event.get('id'), 70)}",
        "Requer revisão humana; contagens não representam pessoas únicas.",
    ))
    return text.encode("utf-16-le")[:2048].decode("utf-16-le", errors="ignore")


def _snapshot_jpeg(event: Mapping[str, Any]) -> bytes:
    """Create a small upload copy without editing the retained evidence."""
    try:
        path = Path(event.get("snapshot_path", ""))
        if not path.is_absolute() or not path.is_file() or path.stat().st_size > 50_000_000:
            raise NotificationError("snapshot")
        with Image.open(path) as source:
            if source.format != "JPEG" or source.width * source.height > 40_000_000:
                raise NotificationError("snapshot")
            source.thumbnail((1920, 1920))
            picture = source.convert("RGB")
        width, height = picture.size
        if max(width, height) / min(width, height) > 20:
            padded_size = (max(width, math.ceil(height / 20)), max(height, math.ceil(width / 20)))
            padded = Image.new("RGB", padded_size, (16, 22, 32))
            padded.paste(picture, ((padded.width - width) // 2, (padded.height - height) // 2))
            picture = padded
        for quality in (90, 75, 55, 35):
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=quality, optimize=True)
            if buffer.tell() <= _MAX_PHOTO_BYTES:
                return buffer.getvalue()
            picture.thumbnail((max(1, picture.width // 2), max(1, picture.height // 2)))
        raise NotificationError("snapshot")
    except NotificationError:
        raise
    except (OSError, ValueError, TypeError, UnidentifiedImageError, Image.DecompressionBombError):
        raise NotificationError("snapshot") from None


class Sender(Protocol):
    def send(self, event: Mapping[str, Any]) -> str: ...


class DeliveryStore(Protocol):
    def set_delivery(self, event_id: str, channel: str, status: str, detail: str = "") -> Any: ...


def _webhook_event_fields(event: Mapping[str, Any]) -> dict[str, str]:
    reasons = event.get("reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    counts = event.get("counts") or {}
    fields = {
        "event_id": _plain(event.get("id"), 70),
        "camera_id": _plain(event.get("camera_id"), 80),
        "camera_name": _plain(event.get("camera_name")),
        "location": _plain(event.get("location")),
        "timestamp_utc": _plain(event.get("timestamp_utc"), 50),
        "kind": _plain(event.get("kind"), 40),
        "model_mode": _plain(event.get("model_mode"), 60),
        "caption": event_caption(event),
        "reasons": _plain("; ".join(str(item) for item in reasons), 300),
    }
    if isinstance(counts, Mapping):
        fields["counts"] = _plain(", ".join(f"{label}: {count}" for label, count in counts.items()), 150)
    return fields


class N8nWebhookSender:
    def __init__(self, config: N8nConfig) -> None:
        config.validate()
        self.config = config
        self._session = requests.Session()

    def send(self, event: Mapping[str, Any]) -> str:
        if not self.config.enabled:
            raise NotificationError("disabled")
        photo = _snapshot_jpeg(event)
        try:
            response = self._session.post(
                self.config.webhook_url.strip(),
                data=_webhook_event_fields(event),
                files={"photo": ("safeguard-event.jpg", photo, "image/jpeg")},
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
            try:
                if response.status_code in {401, 403}:
                    raise NotificationError("webhook_auth")
                if response.status_code == 429:
                    raise NotificationError("webhook_rate")
                if not 200 <= response.status_code < 300:
                    raise NotificationError("webhook_rejected")
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            finally:
                response.close()
            if payload is not None:
                if isinstance(payload, dict) and payload.get("ok") is False:
                    raise NotificationError("webhook_response")
                if isinstance(payload, dict) and payload.get("accepted") is False:
                    raise NotificationError("webhook_response")
            return "Aceito pelo webhook n8n; canais downstream não confirmados neste aplicativo."
        except NotificationError:
            raise
        except requests.Timeout:
            raise NotificationError("webhook_timeout") from None
        except requests.RequestException:
            raise NotificationError("webhook_network") from None
        except (ValueError, TypeError):
            raise NotificationError("webhook_response") from None

    def close(self) -> None:
        self._session.close()


class NotificationDispatcher:
    """One daemon worker; each enabled channel is attempted once per event."""

    def __init__(self, senders: dict[str, Sender], store: DeliveryStore, max_queue: int = 20) -> None:
        if type(max_queue) is not int or max_queue < 1:
            raise ValueError("O limite da fila deve ser um inteiro positivo.")
        self._senders = dict(senders)
        self._store = store
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._lock = threading.RLock()
        self._closing = threading.Event()
        self._seen: set[str] = set()
        self._accepting = True
        self._active = 0
        self._counts = {"pending": 0, "accepted": 0, "failed": 0, "store_errors": 0}
        self._thread = threading.Thread(target=self._run, name="safeguard-notifications", daemon=True)
        self._thread.start()

    def _record(self, event_id: str, channel: str, status: str, detail: str = "") -> None:
        try:
            self._store.set_delivery(event_id, channel, status, detail[:400])
        except Exception:
            self._counts["store_errors"] += 1

    def enqueue(self, event: Mapping[str, Any]) -> bool:
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("O evento precisa ter um identificador antes de entrar na fila.")
        with self._lock:
            if event_id in self._seen or not self._senders:
                return False
            if not self._accepting:
                for channel in self._senders:
                    self._record(event_id, channel, "failed", _ERRORS["closed"])
                    self._counts["failed"] += 1
                return False
            try:
                self._queue.put_nowait(copy.deepcopy(dict(event)))
            except queue.Full:
                for channel in self._senders:
                    self._record(event_id, channel, "failed", _ERRORS["queue_full"])
                    self._counts["failed"] += 1
                return False
            self._seen.add(event_id)
            for channel in self._senders:
                self._record(event_id, channel, "pending", "Aguardando tentativa de envio.")
                self._counts["pending"] += 1
            return True

    def _run(self) -> None:
        try:
            while True:
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    with self._lock:
                        if self._closing.is_set() and self._queue.empty():
                            return
                    continue
                with self._lock:
                    self._active = 1
                try:
                    for channel, sender in self._senders.items():
                        status, detail = "accepted", _ACCEPTED_DETAIL
                        try:
                            sender.send(event)
                        except NotificationError as error:
                            status, detail = "failed", _ERRORS.get(error.code, _ERRORS["unknown"])
                        except Exception:
                            status, detail = "failed", _ERRORS["unknown"]
                        with self._lock:
                            self._record(event["id"], channel, status, detail)
                            self._counts["pending"] -= 1
                            self._counts[status] += 1
                finally:
                    with self._lock:
                        self._active = 0
                    self._queue.task_done()
        finally:
            for sender in self._senders.values():
                close = getattr(sender, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    def close(self, wait: bool = False, timeout: float = 5.0) -> bool:
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("O tempo de espera deve ser finito e não negativo.")
        with self._lock:
            self._accepting = False
            self._closing.set()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._counts,
                "queued": self._queue.qsize(),
                "active": self._active,
                "accepting": self._accepting,
                "worker_alive": self._thread.is_alive(),
                "channels": tuple(self._senders),
            }
