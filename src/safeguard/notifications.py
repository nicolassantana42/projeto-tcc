"""Opt-in notification transports and a bounded, in-memory delivery queue.

Successful sends mean that the provider accepted the message, never that a
person received or read it. No automatic retry is made: a timeout may happen
after the provider accepted a message. Credentials and remote error bodies
are deliberately excluded from persisted delivery details.
"""

from __future__ import annotations

import base64
import copy
import io
import math
import queue
import re
import smtplib
import ssl
import threading
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path
from typing import Any, Mapping, Protocol

import requests
from PIL import Image, UnidentifiedImageError


_ERRORS = {
    "disabled": "Canal desativado; nenhum envio realizado.",
    "snapshot": "Snapshot local ausente, inválido ou grande demais para envio.",
    "telegram_timeout": "Tempo limite do Telegram; resultado incerto. Verifique o chat antes de reenviar.",
    "telegram_network": "Falha de conexão com o Telegram; verifique a rede e o chat antes de reenviar.",
    "telegram_auth": "Telegram recusou a autenticação ou o acesso ao chat. Revise token e Chat ID.",
    "telegram_rate": "Telegram limitou os envios. Aguarde e revise o intervalo entre alertas.",
    "telegram_rejected": "Telegram recusou a solicitação. Revise Chat ID e permissões do bot.",
    "telegram_response": "Telegram retornou resposta sem confirmação válida de aceitação.",
    "email_auth": "SMTP recusou a autenticação. Revise credenciais, token e permissão SMTP AUTH.",
    "email_tls": "Não foi possível estabelecer TLS com o servidor SMTP.",
    "email_recipient": "O servidor SMTP recusou o destinatário.",
    "email_timeout": "Tempo limite do SMTP; resultado incerto. Verifique a caixa antes de reenviar.",
    "email_network": "Falha no envio SMTP. Revise servidor e conexão; verifique a caixa antes de reenviar.",
    "queue_full": "queue_full: fila de alertas cheia; evento salvo, mas envio não agendado.",
    "closed": "dispatcher_closed: sessão encerrada; envio não agendado.",
    "unknown": "Falha no canal de alerta; detalhes remotos omitidos para proteger credenciais.",
}
_MAX_PHOTO_BYTES = 10_000_000
_HTTP_TIMEOUT = (5.0, 15.0)
_SMTP_TIMEOUT = 15.0
_ACCEPTED_DETAIL = "Aceito pelo provedor; entrega e leitura não confirmadas."


class NotificationError(RuntimeError):
    """A public, allowlisted error; arbitrary provider text is never retained."""

    def __init__(self, code: str = "unknown") -> None:
        self.code = code if code in _ERRORS else "unknown"
        super().__init__(_ERRORS[self.code])


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _mailbox(value: str) -> bool:
    """Require a single bare mailbox, excluding address lists/header injection."""
    if not isinstance(value, str) or _has_control(value):
        return False
    name, address = parseaddr(value)
    return not name and address == value and bool(
        re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}", value)
    )


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool = False
    token: str = field(default="", repr=False)
    chat_id: str = field(default="", repr=False)

    def validate(self) -> None:
        if not self.enabled:
            return
        if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", self.token):
            raise ValueError("Informe um token válido do bot Telegram.")
        if not re.fullmatch(r"-?[1-9][0-9]*|@[A-Za-z][A-Za-z0-9_]{4,31}", self.chat_id):
            raise ValueError("Informe o Chat ID numérico ou o nome público do canal (@nome).")


@dataclass(frozen=True)
class EmailConfig:
    enabled: bool = False
    host: str = ""
    port: int = 587
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    sender: str = field(default="", repr=False)
    recipient: str = field(default="", repr=False)
    auth_mode: str = "password"
    access_token: str = field(default="", repr=False)
    security: str = "starttls"

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.host or not re.fullmatch(r"[A-Za-z0-9.-]+", self.host):
            raise ValueError("Informe somente o hostname do servidor SMTP.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("A porta SMTP deve estar entre 1 e 65535.")
        if self.security not in {"starttls", "ssl"}:
            raise ValueError("O envio de e-mail exige STARTTLS ou SSL/TLS.")
        if self.auth_mode not in {"password", "oauth2"}:
            raise ValueError("Selecione autenticação por senha ou OAuth2.")
        if not self.username or _has_control(self.username):
            raise ValueError("Informe um usuário SMTP válido.")
        if not _mailbox(self.sender) or not _mailbox(self.recipient):
            raise ValueError("Informe um único e-mail válido para remetente e destinatário.")
        host = self.host.lower().rstrip(".")
        microsoft = any(host == domain or host.endswith("." + domain) for domain in (
            "outlook.com", "office365.com", "outlook.office.com", "hotmail.com", "live.com",
        ))
        if microsoft and self.auth_mode != "oauth2":
            raise ValueError("Outlook/Microsoft 365 exige OAuth2 neste aplicativo; use um token de acesso SMTP.")
        if self.auth_mode == "oauth2":
            if not self.access_token or any(character.isspace() for character in self.access_token) or _has_control(self.access_token):
                raise ValueError("Informe um token de acesso OAuth2 válido para SMTP.")
        elif not self.password or _has_control(self.password):
            raise ValueError("Informe uma senha SMTP válida.")


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
    # Conservative UTF-16 bound also works when location/reasons contain emoji.
    return text.encode("utf-16-le")[:2048].decode("utf-16-le", errors="ignore")


def _snapshot_jpeg(event: Mapping[str, Any]) -> bytes:
    """Create a small upload copy without editing the retained evidence.

    Telegram permits <=10 MB, width + height <=10000 and aspect ratio <=20.
    A 1920 px thumbnail satisfies dimensions; extreme aspect ratios are padded.
    Metadata/EXIF is not copied into the upload.
    """
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


class TelegramSender:
    def __init__(self, config: TelegramConfig) -> None:
        config.validate()
        self.config = config
        self._session = requests.Session()

    def send(self, event: Mapping[str, Any]) -> str:
        if not self.config.enabled:
            raise NotificationError("disabled")
        photo = _snapshot_jpeg(event)
        try:
            response = self._session.post(
                f"https://api.telegram.org/bot{self.config.token}/sendPhoto",
                data={"chat_id": self.config.chat_id, "caption": event_caption(event)},
                files={"photo": ("safeguard-event.jpg", photo, "image/jpeg")},
                timeout=_HTTP_TIMEOUT,
                allow_redirects=False,
            )
            try:
                if response.status_code in {401, 403}:
                    raise NotificationError("telegram_auth")
                if response.status_code == 429:
                    raise NotificationError("telegram_rate")
                if not 200 <= response.status_code < 300:
                    raise NotificationError("telegram_rejected")
                payload = response.json()
            finally:
                response.close()
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise NotificationError("telegram_response")
            result = payload.get("result")
            message_id = result.get("message_id") if isinstance(result, dict) else None
            if type(message_id) is not int or message_id <= 0:
                raise NotificationError("telegram_response")
            return f"Aceito pelo Telegram (mensagem {message_id}); leitura não confirmada."
        except NotificationError:
            raise
        except requests.Timeout:
            raise NotificationError("telegram_timeout") from None
        except requests.RequestException:
            raise NotificationError("telegram_network") from None
        except (ValueError, TypeError):
            raise NotificationError("telegram_response") from None

    def close(self) -> None:
        self._session.close()


class EmailSender:
    def __init__(self, config: EmailConfig) -> None:
        config.validate()
        self.config = config

    def send(self, event: Mapping[str, Any]) -> str:
        if not self.config.enabled:
            raise NotificationError("disabled")
        message = EmailMessage()
        message["Subject"] = "SafeGuard | Evento de monitoramento"
        message["From"] = self.config.sender
        message["To"] = self.config.recipient
        message["Date"] = formatdate(localtime=False)
        message["Message-ID"] = make_msgid()
        message.set_content(event_caption(event))
        message.add_attachment(_snapshot_jpeg(event), maintype="image", subtype="jpeg", filename="safeguard-event.jpg")
        context = ssl.create_default_context()
        client = None
        try:
            if self.config.security == "ssl":
                client = smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=_SMTP_TIMEOUT, context=context)
                client.ehlo()
            else:
                client = smtplib.SMTP(self.config.host, self.config.port, timeout=_SMTP_TIMEOUT)
                client.ehlo()
                client.starttls(context=context)
                client.ehlo()
            if self.config.auth_mode == "oauth2":
                auth = f"user={self.config.username}\x01auth=Bearer {self.config.access_token}\x01\x01"
                encoded = base64.b64encode(auth.encode("utf-8")).decode("ascii")
                code, _ = client.docmd("AUTH", "XOAUTH2 " + encoded)
                if code != 235:
                    raise NotificationError("email_auth")
            else:
                client.login(self.config.username, self.config.password)
            refused = client.send_message(message, from_addr=self.config.sender, to_addrs=[self.config.recipient])
            if refused:
                raise NotificationError("email_recipient")
            return "Aceito pelo servidor SMTP; entrega e leitura não confirmadas."
        except NotificationError:
            raise
        except smtplib.SMTPAuthenticationError:
            raise NotificationError("email_auth") from None
        except smtplib.SMTPRecipientsRefused:
            raise NotificationError("email_recipient") from None
        except (ssl.SSLError, smtplib.SMTPNotSupportedError):
            raise NotificationError("email_tls") from None
        except TimeoutError:
            raise NotificationError("email_timeout") from None
        except (smtplib.SMTPException, OSError):
            raise NotificationError("email_network") from None
        finally:
            if client is not None:
                # Closing the socket cannot turn an accepted DATA reply into a
                # false failure because a later SMTP QUIT timed out.
                try:
                    client.close()
                except Exception:
                    pass


class NotificationDispatcher:
    """One daemon worker; each enabled channel is attempted once per event.

    enqueue() never waits for network/queue capacity. close() stops accepting
    new events but lets the worker drain accepted jobs, even after UI stop.
    The queue is not durable across process exit; persisted pending entries
    must be reviewed manually after a crash rather than retried silently.
    """

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
            # A failed status write must not kill the worker or hide another
            # channel's result. The UI can surface store_errors explicitly.
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
                    # enqueue() may have accepted a job after get() timed out
                    # and before close(). Recheck under the intake lock so
                    # shutdown never abandons that already accepted job.
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
        """Stop intake; optionally wait at most timeout seconds for draining."""
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
