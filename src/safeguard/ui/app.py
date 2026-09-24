"""Session-isolated Streamlit dashboard. Run through the Streamlit CLI."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import streamlit as st

from safeguard.capture import CaptureError, VideoSource, open_source, ImageSource
from safeguard.config import InferenceConfig
from safeguard.inference import YOLODetector
from safeguard.pipeline import Pipeline
from safeguard.rendering import render_frame
from safeguard.reporting import build_report, encode_snapshot, frame_record
from safeguard.types import Detection, FrameResult
from safeguard.events import EventService
from safeguard.notifications import NotificationDispatcher
from safeguard.ui.alert_panels import (
    camera_context, configured_senders, event_policy, event_store, init_alert_settings,
    render_alert_settings, render_occurrences, reports_root,
)


HISTORY_LIMIT = 300
IDLE_TIMEOUT_SECONDS = 45
PREVIEW = "Prévia ilustrativa"

STYLE = """
<style>
.stApp {background: #0b1220; color: #e4edf4;}
[data-testid="stHeader"] {background: #0b1220dd;}
[data-testid="stSidebar"] {background: #101b2b; border-right: 1px solid #263449;}
[data-testid="stSidebar"] .stMarkdown p {color: #aabace;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1560px;}
h1,h2,h3 {letter-spacing: -.025em; color: #eff7fc !important;}
h1 {font-size: 2.05rem !important; line-height: 1.15 !important;}
.eyebrow {color: #4bd8b0; font-size: .72rem; letter-spacing: .17em; font-weight: 700;}
.muted {color: #99abc0; font-size: .9rem;}
.brand {font-size: 1.45rem; font-weight: 750; letter-spacing: -.04em; color:#edf8fa;}
.brand span {color: #52dab4;}
.pill {display:inline-block; padding: 5px 11px; border-radius: 24px; font-size:.72rem;
       font-weight:650; border: 1px solid #2b4a4d; color:#76e6c6; background:#173333;}
[data-testid="stMetric"] {background: #132135; border: 1px solid #27384e;
    border-radius: 12px; padding: 15px 18px; min-height: 109px;}
[data-testid="stMetricLabel"] {color:#a1b5cb; font-size:.82rem;}
[data-testid="stMetricValue"] {color:#eff9ff; font-weight:650;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#283a50 !important; border-radius:12px;}
.stButton button[kind="primary"] {background:#45d4ab; color:#09271f; border:0; font-weight:700;}
.stButton button {border-radius:8px;}
div[data-testid="stCaptionContainer"] {color:#91a8be;}
.frame-placeholder {height:330px; display:flex; align-items:center; justify-content:center;
  border:1px dashed #355069; border-radius:12px; color:#9bb0c8; background:#0e1a2b;}
</style>
"""


def _illustrative_result(frame_index: int) -> FrameResult:
    """A code-drawn warehouse illustration; it is never passed to a model."""
    frame = np.full((540, 960, 3), (36, 28, 19), dtype=np.uint8)
    cv2.rectangle(frame, (0, 365), (960, 540), (55, 44, 31), -1)
    for x in (45, 300, 570, 855):
        cv2.rectangle(frame, (x, 50), (x + 12, 365), (84, 70, 49), -1)
    for y in (90, 220, 350):
        cv2.line(frame, (40, y), (910, y), (97, 76, 52), 9)
    for x, y, w in ((76, 109, 170), (335, 123, 172), (607, 108, 186), (79, 238, 167), (617, 237, 179)):
        cv2.rectangle(frame, (x, y), (x + w, y + 92), (65, 82, 97), -1)
        cv2.line(frame, (x + w // 2, y), (x + w // 2, y + 92), (83, 101, 117), 3)
    cv2.line(frame, (30, 510), (905, 411), (69, 169, 211), 5)
    cv2.line(frame, (80, 540), (940, 441), (69, 169, 211), 5)
    for x, y, helmet in ((341, 191, True), (664, 218, False)):
        cv2.circle(frame, (x, y), 25, (167, 193, 217), -1)
        cv2.rectangle(frame, (x - 38, y + 32), (x + 38, y + 133), (52, 173, 209), -1)
        cv2.line(frame, (x - 21, y + 38), (x - 21, y + 127), (203, 222, 225), 7)
        cv2.line(frame, (x + 21, y + 38), (x + 21, y + 127), (203, 222, 225), 7)
        cv2.line(frame, (x - 35, y + 92), (x + 35, y + 92), (203, 222, 225), 7)
        for offset in (-23, 23):
            cv2.line(frame, (x + offset, y + 136), (x + offset, y + 222), (132, 111, 82), 18)
        if helmet:
            cv2.ellipse(frame, (x, y - 10), (29, 22), 0, 180, 360, (66, 216, 242), -1)
            cv2.line(frame, (x - 34, y - 7), (x + 34, y - 7), (66, 216, 242), 6)
        color = (155, 221, 73) if helmet else (84, 173, 244)
        cv2.rectangle(frame, (x - 57, y - 40), (x + 57, y + 239), color, 2)
        label = "EPI ilustrativo" if helmet else "Alerta ilustrativo"
        cv2.putText(frame, label, (x - 57, y - 52), cv2.FONT_HERSHEY_SIMPLEX, .57, color, 2)
    cv2.rectangle(frame, (0, 0), (960, 41), (25, 22, 16), -1)
    cv2.putText(frame, "PREVIA ILUSTRATIVA  /  SEM INFERENCIA DE MODELO", (23, 27),
                cv2.FONT_HERSHEY_SIMPLEX, .63, (198, 222, 233), 1, cv2.LINE_AA)
    return FrameResult(
        frame=frame,
        detections=[
            Detection(0, "Pessoa ilustrativa", 0.0, (284, 151, 398, 430)),
            Detection(0, "Pessoa ilustrativa", 0.0, (607, 178, 721, 457)),
            Detection(1, "Capacete ilustrativo", 0.0, (307, 157, 375, 188)),
        ],
        counts={"Pessoa ilustrativa": 2, "Capacete ilustrativo": 1},
        alerts=["Exemplo visual de alerta. Nenhuma avaliação de EPI foi executada."],
        inference_ms=0.0, pipeline_ms=0.0, frame_index=frame_index,
    )


class _SessionRuntime:
    """Own resources per browser session; release after an idle heartbeat.

    The watchdog never calls Streamlit. Driver read timeouts remain the capture
    layer's responsibility; a native driver that hangs can delay final cleanup.
    """

    def __init__(self, *, source_label, capture=None, pipeline=None, temporary_path=None, illustrative=False,
                 event_service=None, dispatcher=None):
        self.source_label = source_label
        self.capture = capture
        self.pipeline = pipeline
        self.temporary_path = temporary_path
        self.illustrative = illustrative
        self.event_service = event_service
        self.dispatcher = dispatcher
        self.last_event = None
        self.event_error = ""
        self.annotated_frame = None
        self.closed = False
        self.frame_index = 0
        self.frame_times = deque(maxlen=30)
        self.last_heartbeat = time.monotonic()
        self._lock = threading.RLock()
        self._finished = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="safeguard-session-cleanup")
        self._watchdog.start()

    @property
    def device(self):
        return "Ilustração" if self.illustrative else str(self.pipeline.detector.device)

    @property
    def observed_fps(self):
        if len(self.frame_times) < 2:
            return None
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 0 else None

    def step(self, confidence, iou):
        with self._lock:
            if self.closed:
                return None
            self.last_heartbeat = time.monotonic()
            if self.illustrative:
                self.frame_index += 1
                result = _illustrative_result(self.frame_index)
            else:
                frame = self.capture.read()
                if frame is None:
                    return None
                result = self.pipeline.process(frame, confidence=confidence, iou=iou)
            self.annotated_frame = result.frame if self.illustrative else render_frame(result)
            if not self.illustrative and self.event_service is not None:
                try:
                    if isinstance(self.capture, ImageSource):
                        kind = self.event_service.policy.trigger
                        if kind == "person":
                            reasons = ["Imagem estática, observação única: pessoa detectada."] if result.counts.get("person") else []
                        else:
                            reasons = [f"Imagem estática, observação única: {'; '.join(item.reasons)}"
                                       for item in result.assessments if item.status == "unsafe"]
                        event = self.event_service.store.save(result, self.annotated_frame, self.event_service.context,
                                                              kind, reasons, self.event_service.demo_mode) if reasons else None
                    elif self.capture.is_file and getattr(self.capture, "timestamp_seconds", None) is None:
                        self.event_service.reset_confirmation()
                        event = None
                        self.event_error = "Vídeo sem tempo legível: confirmação temporal indisponível. Detecção continua."
                    else:
                        event = self.event_service.process(result, self.annotated_frame,
                                                           now=getattr(self.capture, "timestamp_seconds", None))
                    if event is not None:
                        self.last_event = event
                        if self.dispatcher is not None:
                            self.dispatcher.enqueue(event)
                    if not (self.capture.is_file and getattr(self.capture, "timestamp_seconds", None) is None):
                        self.event_error = self.event_service.store.retention_warning
                except Exception:
                    self.event_error = "Falha ao salvar ocorrência. Verifique espaço e permissão na pasta reports. A detecção continua."
            self.frame_times.append(time.monotonic())
            self.last_heartbeat = time.monotonic()
            return result

    def _watch(self):
        while not self._finished.wait(5):
            if time.monotonic() - self.last_heartbeat > IDLE_TIMEOUT_SECONDS:
                self.close()
                return

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self._finished.set()
            if self.capture is not None:
                try:
                    self.capture.close()
                except Exception:
                    pass  # Best effort on a disconnected native driver.
            self.capture = None
            self.pipeline = None
            if self.dispatcher is not None:
                self.dispatcher.close(wait=False)
            if self.temporary_path:
                try:
                    Path(self.temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass


def _stop(message="Monitoramento encerrado. Captura liberada."):
    runtime = st.session_state.get("runtime")
    if runtime is not None:
        runtime.close()
    st.session_state.runtime = None
    st.session_state.notice = message


def _start(source_type, upload, model_path, model_mode, device, camera_index, stream_url, video_path="",
           person_model_path="models/yolo11n.pt"):
    temporary_path = None
    capture = None
    dispatcher = None
    try:
        if source_type == PREVIEW:
            runtime = _SessionRuntime(source_label=PREVIEW, illustrative=True)
        else:
            if source_type in {"Arquivo de vídeo", "Imagem"}:
                if upload is None and not video_path.strip():
                    raise ValueError("Selecione uma imagem antes de iniciar." if source_type == "Imagem" else "Selecione um arquivo de vídeo antes de iniciar.")
                if video_path.strip():
                    source = str(Path(video_path).expanduser())
                else:
                    suffix = Path(upload.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(prefix="safeguard_", suffix=suffix, delete=False) as file:
                        temporary_path = file.name
                        file.write(upload.getbuffer())
                    source = temporary_path
            elif source_type == "Webcam local":
                source = int(camera_index)
            else:
                if urlsplit(stream_url.strip()).scheme not in {"rtsp", "rtsps", "http", "https"}:
                    raise ValueError("Informe uma URL RTSP, RTSPS, HTTP ou HTTPS válida.")
                source = stream_url.strip()
            if not model_path.strip():
                raise ValueError("Informe o caminho do modelo.")
            configuration = InferenceConfig(
                model_path=model_path.strip(), device=device,
                confidence=st.session_state.confidence, iou=st.session_state.iou,
            )
            with st.spinner("Carregando modelo e abrindo a fonte…"):
                if model_mode == "EPI treinado":
                    from safeguard.factory import create_cascade
                    pipeline = create_cascade(person_model_path, model_path.strip(), device,
                                              confidence=st.session_state.confidence, iou=st.session_state.iou)
                    detector = pipeline.detector
                else:
                    detector = YOLODetector(configuration).load()
                    pipeline = Pipeline(detector, demo_mode=True)
                capture = open_source(source).open()
            settings = st.session_state.alert_settings
            service = None
            if settings["save_enabled"]:
                store = event_store()
                service = EventService(store, camera_context(), event_policy(), demo_mode=model_mode == "Demo COCO",
                                       model_names=getattr(pipeline, "names", detector.names))
                senders = configured_senders()
                dispatcher = NotificationDispatcher(senders, store) if senders else None
            runtime = _SessionRuntime(
                source_label=source_type, capture=capture, pipeline=pipeline,
                temporary_path=temporary_path, event_service=service, dispatcher=dispatcher,
            )
        st.session_state.runtime = runtime
        st.session_state.history = deque(maxlen=HISTORY_LIMIT)
        st.session_state.latest_result = None
        st.session_state.latest_frame = None
        st.session_state.last_event = None
        st.session_state.observed_fps = None
        st.session_state.illustrative = runtime.illustrative
        st.session_state.active_device = runtime.device
        st.session_state.active_metadata = {
            "source": source_type,
            "mode": "illustrative_preview" if runtime.illustrative else model_mode,
            "model": None if runtime.illustrative else Path(model_path).name,
            "person_model": Path(person_model_path).name if model_mode == "EPI treinado" else None,
            "device": runtime.device,
            "history_limit_frames": HISTORY_LIMIT,
            "camera_id": camera_context().camera_id,
            "camera_name": camera_context().name,
            "location": camera_context().location,
        }
        st.session_state.export = None
        st.session_state.notice = ""
    except Exception as error:
        if dispatcher is not None:
            dispatcher.close(wait=False)
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass
        # Capture URLs can contain credentials; show a generic source failure.
        if isinstance(error, CaptureError):
            st.session_state.notice = "Falha na captura. Verifique a fonte, suas permissões e a conexão."
        else:
            st.session_state.notice = f"Não foi possível iniciar: {error}"


def _live_panel(compact=False):
    runtime = st.session_state.get("runtime")
    if runtime is not None:
        if runtime.closed:
            _stop("Sessão pausada por inatividade. Inicie novamente para reabrir a captura.")
            st.rerun()
        try:
            result = runtime.step(st.session_state.confidence, st.session_state.iou)
            if result is None:
                _stop("Fim do vídeo ou fonte encerrada. O último frame está disponível para exportação.")
                st.rerun()
            st.session_state.latest_result = result
            st.session_state.latest_frame = runtime.annotated_frame
            st.session_state.observed_fps = runtime.observed_fps
            st.session_state.active_device = runtime.device
            st.session_state.active_metadata["device"] = runtime.device
            record = frame_record(
                result, illustrative=runtime.illustrative, observed_fps=runtime.observed_fps,
            )
            record["thresholds"] = {"confidence": st.session_state.confidence, "iou": st.session_state.iou}
            st.session_state.history.append(record)
            if runtime.last_event is not None:
                st.session_state.last_event = runtime.last_event
        except Exception as error:
            if isinstance(error, CaptureError):
                message = "Captura interrompida. Verifique a webcam, o arquivo ou a conexão do stream."
            else:
                message = f"Processamento interrompido: {error}"
            _stop(message)
            st.rerun()

    result = st.session_state.get("latest_result")
    illustrative = st.session_state.get("illustrative", False) if result is not None else st.session_state.get("source_type") == PREVIEW
    active = runtime is not None and not runtime.closed
    has_metrics = result is not None and not illustrative
    counts = result.counts if result is not None else {}
    fps = st.session_state.get("observed_fps")
    pipeline_fps = 1000 / result.pipeline_ms if has_metrics and result.pipeline_ms > 0 else None
    if compact:
        metrics = st.columns(2)
        metrics[0].metric("FPS observado", f"{fps:.1f}" if has_metrics and fps is not None else "—")
        metrics[1].metric("Inferência total", f"{result.inference_ms:.1f} ms" if has_metrics else "—")
        frame = st.session_state.get("latest_frame")
        if frame is not None:
            st.image(frame, channels="BGR", width="stretch")
            st.caption(f"Frame {result.frame_index} · {sum(result.counts.values())} caixas · "
                       f"Dispositivo: {st.session_state.active_device}")
            if result.assessments:
                status_names = {"ok": "EPI detectado", "unsafe": "Não seguro — revisar", "uncertain": "Inconclusivo"}
                st.dataframe([{"Pessoa no frame": item.index, "Resultado": status_names[item.status],
                               "Evidências": "; ".join(item.reasons)} for item in result.assessments],
                              hide_index=True, width="stretch")
            elif not illustrative:
                st.caption("Nenhuma pessoa avaliada para EPI neste frame.")
        else:
            st.info("Selecione uma imagem, vídeo ou câmera e clique em Iniciar. As pessoas detectadas aparecem aqui com caixas e nível de confiança.")
        if illustrative:
            st.caption("Prévia ilustrativa: desenhos sem inferência de modelo.")
        if runtime is not None and runtime.event_error:
            st.error(runtime.event_error)
        if runtime is not None and runtime.dispatcher is not None:
            delivery = runtime.dispatcher.snapshot()
            st.caption(f"Envios: {delivery['pending']} pendentes · {delivery['accepted']} aceitos · {delivery['failed']} falharam.")
            if delivery["store_errors"]:
                st.warning("Falha ao gravar o status de envios. Verifique o armazenamento.")
        if st.session_state.get("last_event"):
            event = st.session_state.last_event
            st.success(f"Última imagem salva: {event['camera_name']} · {event['location']}. Consulte Ocorrências.")
        return
    metrics = st.columns(4)
    metrics[0].metric("FPS observado", f"{fps:.1f}" if has_metrics and fps is not None else "—")
    metrics[1].metric("Capacidade do pipeline", f"{pipeline_fps:.1f} FPS" if pipeline_fps else "—")
    metrics[2].metric("Latência de inferência", f"{result.inference_ms:.1f} ms" if has_metrics else "—")
    metrics[3].metric("Detecções no frame", sum(counts.values()) if result else "—")
    st.caption("FPS observado: média de até 30 frames, incluindo captura, renderização e interface (até ~10 FPS). "
               "Capacidade do pipeline = 1.000 / tempo de processamento; não inclui a interface.")
    if runtime is not None and runtime.event_error:
        st.error(runtime.event_error)
    if runtime is not None and runtime.dispatcher is not None:
        delivery = runtime.dispatcher.snapshot()
        st.caption(f"Envios nesta sessão: {delivery['pending']} pendentes · {delivery['accepted']} aceitos · "
                   f"{delivery['failed']} falharam. Detalhes por imagem em Ocorrências.")
        if delivery["store_errors"]:
            st.warning("Não foi possível atualizar o status de alguns envios no disco. Verifique o armazenamento.")
    if st.session_state.get("last_event"):
        event = st.session_state.last_event
        st.success(f"Última imagem salva: {event['camera_name']} · {event['location']} · {event['timestamp_utc']}. Veja a aba Ocorrências.")

    left, right = st.columns([2.35, 1], gap="large")
    with left, st.container(border=True):
        title_col, status_col = st.columns([3, 1])
        title_col.markdown("#### Visão ao vivo")
        status = "PRÉVIA ILUSTRATIVA" if illustrative else ("AO VIVO" if active else "PAUSADO" if result else "PRONTO PARA INICIAR")
        status_col.markdown(f'<span class="pill">{status}</span>', unsafe_allow_html=True)
        frame = st.session_state.get("latest_frame")
        if frame is None:
            st.info("Selecione Webcam local, Arquivo de vídeo ou RTSP / IP e clique em Iniciar. "
                    "As pessoas detectadas aparecem aqui com caixas e nível de confiança.")
            if st.session_state.get("source_type") == PREVIEW:
                st.image(_illustrative_result(0).frame, channels="BGR", width="stretch")
        else:
            st.image(frame, channels="BGR", width="stretch")
            st.caption(f"Frame {result.frame_index:,} · {st.session_state.active_metadata['source']} · "
                       f"Dispositivo: {st.session_state.active_device}")
        if illustrative and result is not None:
            st.info("Prévia desenhada por código, sem câmera e sem modelo. Caixas e contagens são ilustrativas; "
                    "não há métricas de desempenho nem avaliação real de EPI.")
        else:
            st.caption("Caixas representam resultados do modelo selecionado. Alertas exigem revisão humana.")

    with right:
        with st.container(border=True):
            st.markdown("#### Objetos por classe")
            st.caption("Contagem no frame atual")
            if counts:
                st.dataframe(
                    [{"Classe": label, "Quantidade": count} for label, count in sorted(counts.items())],
                    hide_index=True, width="stretch",
                )
            else:
                st.caption("Nenhuma detecção no frame atual." if result else "As classes aparecerão após o início da captura.")
            st.caption("Detecções por frame. Pessoas repetidas em frames diferentes não são pessoas únicas.")
        with st.container(border=True):
            st.markdown("#### Central de atenção")
            if result and result.alerts:
                for alert in result.alerts[:5]:
                    st.warning(str(alert))
            elif result:
                st.success("Nenhum alerta de classe neste frame.")
                st.caption("Ausência de alerta não comprova conformidade de EPI.")
            else:
                st.info("Aguardando início da sessão.")
        st.caption(f"Histórico em memória: {len(st.session_state.history)} / {HISTORY_LIMIT} frames.")


def main():
    st.set_page_config(page_title="SafeGuard | Visão e segurança", page_icon="🟢", layout="wide")
    init_alert_settings()
    for key, value in {
        "runtime": None, "history": deque(maxlen=HISTORY_LIMIT), "latest_result": None,
        "latest_frame": None, "notice": "", "export": None, "confidence": .4, "iou": .45,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
    runtime = st.session_state.runtime
    running = runtime is not None and not runtime.closed

    with st.sidebar:
        compact = not st.checkbox("Exibir painel completo", value=False)
        if not compact:
            st.markdown(STYLE, unsafe_allow_html=True)
        st.markdown("#### Configuração da sessão")
        source_type = st.selectbox("Fonte de entrada", ["Webcam local", "Imagem", "Arquivo de vídeo", "RTSP / IP", PREVIEW], key="source_type", disabled=running)
        upload, camera_index, stream_url, video_path = None, 0, "", ""
        if source_type == "Webcam local":
            camera_index = st.number_input("Índice da webcam", min_value=0, max_value=20, value=0, disabled=running)
            st.caption("A webcam deve estar conectada ao computador que executa o servidor.")
        elif source_type == "Arquivo de vídeo":
            file_mode = st.radio("Abrir vídeo", ["Enviar arquivo", "Caminho no computador"], disabled=running)
            if file_mode == "Enviar arquivo":
                upload = st.file_uploader("Vídeo", type=["mp4", "avi", "mov", "mkv", "webm"], disabled=running)
            else:
                video_path = st.text_input("Caminho do vídeo", placeholder="C:/videos/camera.mp4", disabled=running)
        elif source_type == "Imagem":
            file_mode = st.radio("Abrir imagem", ["Enviar arquivo", "Caminho no computador"], disabled=running)
            if file_mode == "Enviar arquivo":
                upload = st.file_uploader("Imagem", type=["jpg", "jpeg", "png", "bmp", "webp"], disabled=running)
            else:
                video_path = st.text_input("Caminho da imagem", disabled=running)
        elif source_type == "RTSP / IP":
            stream_url = st.text_input("URL do stream", type="password", placeholder="rtsp://…", disabled=running)
        model_mode = st.radio("Finalidade do modelo", ["EPI treinado", "Demo COCO"], disabled=running or source_type == PREVIEW)
        default_path = "models/yolo11n.pt" if model_mode == "Demo COCO" else "models/ppe/best.pt"
        model_path = st.text_input("Caminho do modelo", value=default_path, key=f"model_{model_mode}", disabled=running or source_type == PREVIEW)
        person_model_path = "models/yolo11n.pt"
        if model_mode == "EPI treinado":
            person_model_path = st.text_input("Modelo de pessoas — primeira etapa", value=person_model_path, disabled=running)
        device = st.selectbox("Dispositivo", ["auto", "cpu", "cuda:0", "mps"], disabled=running or source_type == PREVIEW)
        if model_mode == "Demo COCO":
            st.caption("COCO demonstra detecção geral; não é um modelo de identificação de EPIs.")
        else:
            st.caption("Pessoa → segundo YOLO de EPI. Informe pesos treinados e confira suas métricas antes da demonstração.")
            if not Path(model_path).exists():
                st.warning("Pesos de EPI ainda não disponíveis neste caminho. Treine o modelo ou selecione os pesos gerados em runs/train.")
            elif Path(model_path).name == "baseline-public.pt":
                st.warning("Este modelo público teve baixa detecção de coletes na avaliação. Use apenas como comparação experimental.")
        st.divider()
        st.markdown("#### Ajustes em tempo real")
        st.slider("Confiança mínima", min_value=.05, max_value=.95, step=.05, key="confidence")
        st.slider("IoU / sobreposição", min_value=.05, max_value=.95, step=.05, key="iou")
        start_col, stop_col = st.columns(2)
        if start_col.button("▶ Iniciar", type="primary", width="stretch", disabled=running):
            _start(source_type, upload, model_path, model_mode, device, camera_index, stream_url, video_path, person_model_path)
            st.rerun()
        if stop_col.button("■ Parar", width="stretch", disabled=not running):
            _stop()
            st.rerun()
        st.caption("Cada sessão possui sua própria captura. Ao sair, a liberação por inatividade ocorre em aproximadamente 45–50 s.")
        settings = st.session_state.alert_settings
        st.divider()
        st.write(f"📍 {settings['camera_name']} · {settings['location'] or 'Configure o local na aba Alertas e integrações'}")
        st.caption("Gravação automática: " + ("ativa" if settings["save_enabled"] else "desativada"))

    st.title("Detecção de pessoas e EPIs")
    st.caption("Capacete e colete por pessoa. EPI detectado, não seguro com evidência explícita, ou inconclusivo.")
    if st.session_state.notice:
        st.info(st.session_state.notice)

    monitor_tab, occurrences_tab, integrations_tab = st.tabs(["Monitoramento", "Ocorrências", "Alertas e integrações"])
    with monitor_tab:
        settings = st.session_state.alert_settings
        policy_text = "pessoas detectadas" if settings["trigger"] == "person" else "possível ausência de EPI"
        if settings["save_enabled"]:
            st.caption(f"Registro automático de {policy_text}: confirmação {settings['confirmation_seconds']:g}s do vídeo/câmera · "
                       f"intervalo {settings['cooldown_seconds']:g}s. Imagens: {reports_root() / 'occurrences'}")
            if source_type == "Imagem":
                st.caption("Imagem estática: salva uma observação única, sem confirmação temporal.")
        else:
            st.caption("Registro automático e alertas externos desativados. Use 'Salvar imagem agora' para registrar um frame.")
        if not settings["location"]:
            st.info("Cadastre câmera e local / setor na aba Alertas e integrações para identificar as imagens salvas.")
        if model_mode == "Demo COCO" and settings["trigger"] == "ppe":
            st.info("Para salvar presença de pessoas com COCO, escolha 'Pessoa detectada' na aba Alertas e integrações. "
                    "A regra de falta de EPI precisa de um modelo treinado.")
        @st.fragment(run_every=.1 if running else None)
        def live_fragment():
            _live_panel(compact=compact)
        live_fragment()
        if st.button("Salvar imagem agora", disabled=st.session_state.latest_frame is None or st.session_state.get("illustrative", True)):
            try:
                metadata = st.session_state.active_metadata
                from safeguard.events import CameraContext
                context = CameraContext(metadata["camera_id"], metadata["camera_name"], metadata["location"])
                event = event_store().save(st.session_state.latest_result, st.session_state.latest_frame, context,
                                           kind="manual", reasons=["Captura manual solicitada na interface"],
                                           demo_mode=metadata["mode"] == "Demo COCO")
                st.session_state.last_event = event
                st.success("Imagem salva no computador. Abra a aba Ocorrências para visualizar ou baixar.")
            except Exception:
                st.error("Não foi possível salvar a imagem. Verifique a pasta reports e o espaço em disco.")
        _render_export()
    with occurrences_tab:
        @st.fragment(run_every=2 if running else None)
        def history_fragment():
            render_occurrences()
        history_fragment()
    with integrations_tab:
        render_alert_settings(running)


def _render_export():
    with st.expander("Exportar evidências da sessão", expanded=False):
        st.caption("O ZIP reúne CSV, JSON e snapshot do último frame processado. O histórico contém até "
                   f"{HISTORY_LIMIT} frames; não representa uma gravação completa. URLs e credenciais não são exportadas.")
        if st.button("Preparar exportação", disabled=st.session_state.latest_frame is None) and st.session_state.latest_frame is not None:
            snapshot = encode_snapshot(st.session_state.latest_frame)
            st.session_state.export = {
                "snapshot": snapshot,
                "report": build_report(
                    st.session_state.history, snapshot_png=snapshot,
                    metadata={
                        **st.session_state.active_metadata,
                        "confidence_at_export": st.session_state.confidence,
                        "iou_at_export": st.session_state.iou,
                    },
                ),
                "name": "safeguard_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        export = st.session_state.export
        if export:
            col1, col2 = st.columns(2)
            col1.download_button("↓ Snapshot PNG", export["snapshot"], file_name=f"{export['name']}.png", mime="image/png", width="stretch")
            col2.download_button("↓ Relatório ZIP", export["report"], file_name=f"{export['name']}.zip", mime="application/zip", width="stretch")
            st.caption("Exportação congelada no instante de preparação. Prepare novamente para atualizar.")


if __name__ == "__main__":
    main()
