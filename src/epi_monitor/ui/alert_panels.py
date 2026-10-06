"""Occurrence history and notification setup; credentials stay out of disk settings."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

from epi_monitor.events import CameraContext, EventPolicy, EventStore
from epi_monitor.notifications import (
    EmailConfig, EmailSender, NotificationDispatcher, TelegramConfig, TelegramSender,
)
from epi_monitor.types import FrameResult


DEFAULTS = {
    "camera_id": "camera-01", "camera_name": "Câmera 01", "location": "",
    "save_enabled": True, "trigger": "ppe", "confirmation_seconds": 2.0,
    "cooldown_seconds": 60.0, "max_events": 500,
    "telegram_chat_id": "", "email_host": "smtp-mail.outlook.com", "email_port": 587,
    "email_username": "", "email_sender": "", "email_recipient": "",
    "email_auth_mode": "oauth2", "email_security": "starttls",
}


def reports_root() -> Path:
    configured = os.environ.get("EPI_REPORTS_DIR", "reports")
    return Path(configured).expanduser().resolve()


def load_settings() -> dict:
    settings = DEFAULTS.copy()
    try:
        document = json.loads((reports_root() / "settings.json").read_text(encoding="utf-8"))
        if isinstance(document, dict):
            settings.update({key: value for key, value in document.items() if key in DEFAULTS})
        EventPolicy(trigger=settings["trigger"], confirmation_seconds=settings["confirmation_seconds"],
                    cooldown_seconds=settings["cooldown_seconds"], max_events=settings["max_events"])
        CameraContext(settings["camera_id"], settings["camera_name"], settings["location"] or "Local não informado")
        if settings["email_auth_mode"] not in {"password", "oauth2"} or settings["email_security"] not in {"starttls", "ssl"}:
            return DEFAULTS.copy()
        if not isinstance(settings["email_port"], int) or not 1 <= settings["email_port"] <= 65535:
            return DEFAULTS.copy()
        if not .1 <= settings["confirmation_seconds"] <= 60 or not 1 <= settings["cooldown_seconds"] <= 86400:
            return DEFAULTS.copy()
        if not 10 <= settings["max_events"] <= 10000 or not isinstance(settings["save_enabled"], bool):
            return DEFAULTS.copy()
        if any(not isinstance(settings[key], str) for key, default in DEFAULTS.items() if isinstance(default, str)):
            return DEFAULTS.copy()
    except (OSError, ValueError, TypeError, KeyError):
        return DEFAULTS.copy()
    return settings


def save_settings(settings: dict) -> None:
    """Allowlist prevents tokens/passwords and activation flags from reaching disk."""
    root = reports_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = {key: settings[key] for key in DEFAULTS}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary.replace(root / "settings.json")
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _secret(section: str, key: str, fallback=""):
    env_key = "TELEGRAM_BOT_TOKEN" if (section, key) == ("telegram", "token") else f"{section}_{key}".upper()
    if env_key in os.environ:
        return os.environ[env_key]
    try:
        section_values = st.secrets.get(section, {})
        return section_values.get(key, fallback) if hasattr(section_values, "get") else fallback
    except (FileNotFoundError, KeyError, ValueError):
        return fallback


def init_alert_settings():
    if "alert_settings" not in st.session_state:
        st.session_state.alert_settings = load_settings()
    settings = st.session_state.alert_settings
    if "telegram_config" not in st.session_state:
        st.session_state.telegram_config = TelegramConfig(
            token=str(_secret("telegram", "token")),
            chat_id=str(_secret("telegram", "chat_id", settings["telegram_chat_id"])),
        )
    if "email_config" not in st.session_state:
        values = {key: _secret("email", key, settings.get(f"email_{key}", "")) for key in
                  ("host", "port", "username", "password", "sender", "recipient", "auth_mode", "access_token", "security")}
        try:
            values["port"] = int(values["port"])
        except (TypeError, ValueError):
            values["port"] = 587
        if not 1 <= values["port"] <= 65535:
            values["port"] = 587
        for key, allowed in (("auth_mode", {"password", "oauth2"}), ("security", {"starttls", "ssl"})):
            if values[key] not in allowed:
                values[key] = DEFAULTS[f"email_{key}"]
        values = {key: value if key == "port" else str(value) for key, value in values.items()}
        st.session_state.email_config = EmailConfig(**values)


def event_store() -> EventStore:
    return EventStore(reports_root() / "occurrences", max_events=int(st.session_state.alert_settings["max_events"]))


def camera_context() -> CameraContext:
    settings = st.session_state.alert_settings
    return CameraContext(settings["camera_id"], settings["camera_name"], settings["location"] or "Local não informado")


def event_policy() -> EventPolicy:
    settings = st.session_state.alert_settings
    return EventPolicy(**{key: settings[key] for key in
                          ("trigger", "confirmation_seconds", "cooldown_seconds", "max_events")})


def configured_senders() -> dict:
    channels = {}
    telegram = st.session_state.telegram_config
    email = st.session_state.email_config
    if telegram.enabled:
        telegram.validate()
        channels["telegram"] = TelegramSender(telegram)
    if email.enabled:
        email.validate()
        channels["email"] = EmailSender(email)
    return channels


def render_alert_settings(running: bool):
    settings = st.session_state.alert_settings
    telegram, email = st.session_state.telegram_config, st.session_state.email_config
    st.markdown("**Câmera e alertas**")
    if notice := st.session_state.pop("alert_settings_notice", None):
        st.toast("Configurações salvas.", icon=":material/check_circle:")
        st.success(notice)
    st.caption("Tokens e senhas ficam somente nesta sessão (ou em .streamlit/secrets.toml).")
    if running:
        st.info("Pare o monitoramento para alterar os destinos e a política de ocorrências.")
    with st.form("alert_configuration"):
        left, right = st.columns(2)
        with left:
            camera_id = st.text_input("Identificador da câmera", value=settings["camera_id"], disabled=running)
            camera_name = st.text_input("Nome da câmera", value=settings["camera_name"], disabled=running)
        with right:
            location = st.text_input("Local / setor", value=settings["location"], placeholder="Unidade 1 • Galpão A • Entrada", disabled=running)
            save_enabled = st.checkbox("Salvar ocorrências automaticamente", value=settings["save_enabled"], disabled=running)
        trigger = st.selectbox("Quando registrar uma ocorrência", ["ppe", "person"],
                               index=["ppe", "person"].index(settings["trigger"]),
                               format_func=lambda item: "Possível ausência de EPI (modelo treinado)" if item == "ppe" else "Pessoa detectada (também funciona com COCO)", disabled=running)
        a, b, c = st.columns(3)
        confirmation = a.number_input("Confirmação contínua (segundos)", min_value=.1, max_value=60.0,
                                      value=float(settings["confirmation_seconds"]), step=.5, disabled=running)
        cooldown = b.number_input("Intervalo entre ocorrências (segundos)", min_value=1.0, max_value=86400.0,
                                  value=float(settings["cooldown_seconds"]), step=10.0, disabled=running)
        max_events = c.number_input("Máximo de ocorrências salvas", min_value=10, max_value=10000,
                                    value=int(settings["max_events"]), step=10, disabled=running)
        st.markdown("#### Telegram")
        telegram_enabled = st.checkbox("Ativar Telegram para novas ocorrências", value=telegram.enabled, disabled=running)
        token = st.text_input("Token do bot", value=telegram.token, type="password", disabled=running)
        chat_id = st.text_input("Chat ID de destino", value=telegram.chat_id, disabled=running)
        st.caption("Crie o bot no @BotFather e envie /start para ele. Guia: docs/ALERTS.md.")
        with st.expander("E-mail / Outlook (opcional)"):
            email_enabled = st.checkbox("Ativar e-mail para novas ocorrências", value=email.enabled, disabled=running)
            host = st.text_input("Servidor SMTP", value=email.host, disabled=running)
            port = st.number_input("Porta SMTP", min_value=1, max_value=65535, value=email.port, disabled=running)
            security = st.selectbox("Criptografia", ["starttls", "ssl"], index=["starttls", "ssl"].index(email.security), disabled=running)
            username = st.text_input("Usuário SMTP", value=email.username, disabled=running)
            sender = st.text_input("E-mail remetente", value=email.sender, disabled=running)
            recipient = st.text_input("E-mail destinatário", value=email.recipient, disabled=running)
            auth_mode = st.selectbox("Autenticação", ["oauth2", "password"], index=["oauth2", "password"].index(email.auth_mode),
                                     format_func=lambda value: "OAuth2 (Outlook / Microsoft 365)" if value == "oauth2" else "Senha de aplicativo (outros provedores)", disabled=running)
            access_token = st.text_input("Access token OAuth2 SMTP", value=email.access_token, type="password", disabled=running)
            password = st.text_input("Senha de aplicativo SMTP", value=email.password, type="password", disabled=running)
            st.caption("Outlook usa OAuth2. Esta versão recebe um access token SMTP válido; login Microsoft e renovação "
                       "automática ainda não estão integrados. Não use sua senha comum do Outlook aqui.")
        submitted = st.form_submit_button("Salvar configurações", type="primary", disabled=running)
    if submitted:
        try:
            if not camera_name.strip() or not camera_id.strip() or not location.strip():
                raise ValueError("Preencha identificador, nome da câmera e local / setor.")
            CameraContext(camera_id.strip(), camera_name.strip(), location.strip())
            EventPolicy(trigger, confirmation, cooldown, int(max_events))
            next_telegram = TelegramConfig(telegram_enabled, token.strip(), chat_id.strip())
            next_email = EmailConfig(enabled=email_enabled, host=host.strip(), port=int(port), username=username.strip(),
                                     password=password, sender=sender.strip(), recipient=recipient.strip(),
                                     auth_mode=auth_mode, access_token=access_token.strip(), security=security)
            next_telegram.validate()
            next_email.validate()
            updated = {"camera_id": camera_id.strip(), "camera_name": camera_name.strip(), "location": location.strip(),
                       "save_enabled": save_enabled, "trigger": trigger, "confirmation_seconds": confirmation,
                       "cooldown_seconds": cooldown, "max_events": int(max_events), "telegram_chat_id": chat_id.strip(),
                       **{f"email_{key}": value for key, value in asdict(next_email).items() if f"email_{key}" in DEFAULTS}}
            save_settings(updated)
            st.session_state.alert_settings = updated
            st.session_state.telegram_config = next_telegram
            st.session_state.email_config = next_email
            st.session_state.alert_settings_notice = "Configurações salvas. Canais ativados valem para esta sessão; inicie a captura para usá-los."
            # Refresh widget defaults and monitor summaries after committing the form.
            # Without this rerun, a subsequent edit can be reset by a changed default.
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))
    enabled = [name for name, config in (("Telegram", st.session_state.telegram_config), ("E-mail", st.session_state.email_config)) if config.enabled]
    st.caption("Canais ativos nesta sessão: " + (", ".join(enabled) if enabled else "nenhum — imagens ficam somente no computador"))
    if enabled and not st.session_state.alert_settings["save_enabled"]:
        st.warning("Canal configurado, mas o envio automático está desligado porque 'Salvar ocorrências automaticamente' está desmarcado.")
    if st.button("Enviar teste aos canais ativos", disabled=running or not enabled):
        _send_test()
    st.caption("O teste faz um envio real; salvar configurações não envia mensagens.")


def _send_test():
    try:
        frame = np.full((360, 640, 3), (35, 48, 60), dtype=np.uint8)
        cv2.putText(frame, "Deteccao de EPIs - TESTE DE CONEXAO", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, .75, (80, 235, 180), 2)
        result = FrameResult(frame, [], {}, [], 0.0, 0.0, 0)
        store = event_store()
        event = store.save(result, frame, camera_context(), kind="manual", reasons=["Teste de conexão solicitado na interface"], demo_mode=True)
        dispatcher = NotificationDispatcher(configured_senders(), store)
        try:
            queued = dispatcher.enqueue(event)
        finally:
            dispatcher.close(wait=False)
        if queued:
            st.success("Teste colocado na fila. O status de aceitação ou falha aparece em Ocorrências.")
        else:
            st.error("O teste foi salvo, mas não entrou na fila. Consulte o status em Ocorrências.")
    except Exception:
        st.error("Não foi possível preparar o teste. Verifique os campos e a permissão de escrita em reports.")


KIND_LABELS = {"ppe": "Sem EPI", "person": "Pessoa", "manual": "Manual"}
DELIVERY_LABELS = {"pending": "Na fila", "accepted": "Aceito pelo provedor", "failed": "Falhou", "queue_full": "Fila cheia"}
REVIEW_LABELS = {"confirmed": ("Confirmada", "unsafe"), "dismissed": ("Falso positivo", "neutral"), None: ("Pendente", "uncertain")}


def _review(event: dict):
    review = event.get("review")
    status = review.get("status") if isinstance(review, dict) else None
    return status if status in REVIEW_LABELS else None


def _local_time(event: dict) -> str:
    try:
        return datetime.fromisoformat(event["timestamp_utc"]).astimezone().strftime("%d/%m/%Y %H:%M:%S")
    except (ValueError, KeyError):
        return str(event.get("timestamp_utc", ""))


def render_occurrences():
    from epi_monitor.ui import theme

    try:
        store = event_store()
        events = store.list_events(limit=100)
    except OSError:
        st.error("Não foi possível ler o histórico. Verifique a pasta reports e as permissões de acesso.")
        return
    head, refresh = st.columns([4, 1], vertical_alignment="center")
    reviewed = [_review(event) for event in events]
    confirmed, dismissed = reviewed.count("confirmed"), reviewed.count("dismissed")
    precision = f" · precisão revisada {100 * confirmed / (confirmed + dismissed):.0f}%" if confirmed + dismissed else ""
    head.markdown(f"**{len(events)} ocorrência(s)** · {len(events) - confirmed - dismissed} pendente(s) de revisão{precision}")
    refresh.button("Atualizar histórico", key="refresh_occurrences", width="stretch")
    if not events:
        st.info("Nenhuma ocorrência salva. Elas aparecem aqui quando alguém é detectado sem EPI, "
                "ou ao usar 'Salvar imagem agora' no Monitor.")
        return

    query = st.text_input("Buscar", placeholder="Câmera, local, motivo ou ID", key="occurrence_search",
                          label_visibility="collapsed").strip().lower()
    filtered = [event for event in events if not query or query in " ".join(
        str(event.get(field, "")) for field in ("camera_name", "location", "id", "reasons")).lower()]
    if not filtered:
        st.caption("Nenhuma ocorrência corresponde à busca.")
        return
    st.dataframe([{
        "Data e hora": _local_time(event),
        "Tipo": KIND_LABELS.get(event.get("kind"), str(event.get("kind"))),
        "Câmera": event.get("camera_name", ""),
        "Local": event.get("location", ""),
        "Revisão": REVIEW_LABELS[_review(event)][0],
        "ID": event["id"][:8],
    } for event in filtered], hide_index=True, width="stretch", height=min(38 * len(filtered) + 40, 300))

    labels = {event["id"]: f"{_local_time(event)} · {event.get('camera_name', '')} · {event['id'][:8]}" for event in filtered}
    selected = st.selectbox("Ocorrência", list(labels), format_func=labels.get, key="selected_occurrence")
    event = next(item for item in filtered if item["id"] == selected)
    image_col, detail_col = st.columns([1.6, 1], gap="large")
    content = None
    with image_col:
        try:
            content = Path(event["snapshot_path"]).read_bytes()
            st.image(content, width="stretch")
        except OSError:
            st.warning("A imagem foi removida ou está indisponível no disco.")
    with detail_col:
        review_label, review_tone = REVIEW_LABELS[_review(event)]
        st.markdown(f"**{event.get('camera_name', '')}** · {event.get('location', '')}  \n{_local_time(event)}")
        st.markdown(f"{KIND_LABELS.get(event.get('kind'), '')} · revisão: {theme.pill(review_label, review_tone)}",
                    unsafe_allow_html=True)
        st.write(" · ".join(event.get("reasons", [])))
        confirm_col, dismiss_col = st.columns(2)
        decision = None
        if confirm_col.button("Confirmar", key=f"confirm_{event['id']}", width="stretch", type="primary",
                              disabled=_review(event) == "confirmed"):
            decision = "confirmed"
        if dismiss_col.button("Descartar", key=f"dismiss_{event['id']}", width="stretch",
                              help="Marca como falso positivo; o registro é mantido para estatística.",
                              disabled=_review(event) == "dismissed"):
            decision = "dismissed"
        if decision is not None:
            try:
                store.set_review(event["id"], decision)
                st.rerun(scope="fragment")
            except (OSError, ValueError):
                st.error("Não foi possível salvar a revisão. Verifique a pasta reports.")
        for channel, delivery in event.get("deliveries", {}).items():
            status = delivery.get("status", "")
            st.caption(f"{channel}: {DELIVERY_LABELS.get(status, status)} — {delivery.get('detail', '')}")
        if content is not None:
            st.download_button("Baixar imagem JPG", content, file_name=f"{event['id']}.jpg", mime="image/jpeg",
                               key=f"photo_{event['id']}", width="stretch")
        st.download_button("Baixar registro JSON", json.dumps(event, ensure_ascii=False, indent=2),
                           file_name=f"{event['id']}.json", mime="application/json", key=f"json_{event['id']}",
                           width="stretch")
        st.caption(f"ID: {event['id']} · arquivos em {reports_root() / 'occurrences'}")
