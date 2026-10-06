"""Session-isolated Streamlit dashboard. Run through the Streamlit CLI."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import streamlit as st

from epi_monitor.capture import CaptureError, VideoSource, open_source, ImageSource
from epi_monitor.config import DEFAULT_PERSON_MODEL, DEFAULT_PPE_MODEL, InferenceConfig
from epi_monitor.detection import canonical_label
from epi_monitor.inference import YOLODetector
from epi_monitor.pipeline import Pipeline
from epi_monitor.rendering import render_frame
from epi_monitor.reporting import build_report, encode_snapshot, frame_record
from epi_monitor.types import Detection, FrameResult
from epi_monitor.events import EventService
from epi_monitor.notifications import NotificationDispatcher
from epi_monitor.ui.preview import AnalysisPacer, encode_preview, preview_markup
from epi_monitor.ui import profiles, theme
from epi_monitor.ui.alert_panels import (
    camera_context, configured_senders, event_policy, event_store, init_alert_settings,
    render_alert_settings, render_occurrences, reports_root,
)


HISTORY_LIMIT = 300
IDLE_TIMEOUT_SECONDS = 45
PREVIEW = "Prévia ilustrativa"
ANALYSIS_RATES = {"3 análises/s · econômico": 3, "10 análises/s · padrão": 10,
                  "25 análises/s": 25, "30 análises/s": 30}



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
                 event_service=None, dispatcher=None, analysis_rate=10):
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
        self.preview_bytes = None
        self.preview_error = ""
        self.pacer = AnalysisPacer(analysis_rate)
        self.closed = False
        self.frame_index = 0
        self.frame_times = deque(maxlen=30)
        self.last_heartbeat = time.monotonic()
        self._lock = threading.RLock()
        self._finished = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="epi-session-cleanup")
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

    def should_process(self):
        self.last_heartbeat = time.monotonic()
        return isinstance(self.capture, ImageSource) or self.pacer.ready()

    def step(self, confidence, iou):
        with self._lock:
            if self.closed:
                return None
            self.last_heartbeat = time.monotonic()
            self.pacer.started()
            if self.illustrative:
                self.frame_index += 1
                result = _illustrative_result(self.frame_index)
            else:
                frame = self.capture.read()
                if frame is None:
                    return None
                result = self.pipeline.process(frame, confidence=confidence, iou=iou)
            self.annotated_frame = result.frame if self.illustrative else render_frame(result)
            try:
                self.preview_bytes = encode_preview(self.annotated_frame)
                self.preview_error = ""
            except (cv2.error, ValueError):
                self.preview_bytes = None
                self.preview_error = "Prévia reduzida indisponível; a detecção e o registro de evidências continuam."
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
           person_model_path=DEFAULT_PERSON_MODEL):
    temporary_path = None
    capture = None
    dispatcher = None
    analysis_rate = ANALYSIS_RATES[st.session_state.analysis_rate]
    try:
        if source_type == PREVIEW:
            runtime = _SessionRuntime(source_label=PREVIEW, illustrative=True, analysis_rate=analysis_rate)
        else:
            if source_type in {"Arquivo de vídeo", "Imagem"}:
                if upload is None and not video_path.strip():
                    raise ValueError("Selecione uma imagem antes de iniciar." if source_type == "Imagem" else "Selecione um arquivo de vídeo antes de iniciar.")
                if video_path.strip():
                    source = str(Path(video_path).expanduser())
                else:
                    suffix = Path(upload.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(prefix="epi_", suffix=suffix, delete=False) as file:
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
            if model_mode == "EPI treinado":
                from epi_monitor.factory import create_cascade
                pipeline = create_cascade(person_model_path, model_path.strip(), device, imgsz=profiles.IMGSZ,
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
                analysis_rate=analysis_rate,
            )
        st.session_state.runtime = runtime
        st.session_state.history = deque(maxlen=HISTORY_LIMIT)
        st.session_state.latest_result = None
        st.session_state.latest_frame = None
        st.session_state.latest_preview = None
        st.session_state.preview_error = ""
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
            "analysis_rate_limit": analysis_rate,
            "analysis_target_period_seconds": 1 / analysis_rate,
            "camera_id": camera_context().camera_id,
            "camera_name": camera_context().name,
            "location": camera_context().location,
            # Read the loaded PPE model, not the cascade's fixed output vocabulary.
            "ppe_model_classes": ([] if runtime.illustrative or model_mode != "EPI treinado" else
                                  list(pipeline.ppe_detector.names.values())),
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


STATUS_NAMES = {"ok": "EPIs OK", "unsafe": "Sem EPI", "uncertain": "Inconclusivo"}
METRICS_PATH = Path("models/ppe/epi.metrics.json")


def _render_ppe_capabilities():
    metadata = st.session_state.get("active_metadata", {})
    if metadata.get("mode") != "EPI treinado":
        return
    classes = {canonical_label(name) for name in metadata.get("ppe_model_classes", [])}
    supported = [f"sem {name.lower()}" for kind, name in theme.EQUIPMENT.items() if f"no_{kind}" in classes]
    st.caption("Classes de ausência disponíveis no modelo carregado: " + (", ".join(supported) or "nenhuma") + ". "
               "EPI não visto e sem classe de ausência aparece como inconclusivo.")


def _equipment_status(item, equipment):
    if equipment in item.uncertain:
        return "uncertain", "⚠️ Não detectado"
    if equipment in item.absent:
        return "unsafe", "❌ Ausente"
    if equipment in item.present:
        return "ok", "✅ Detectado"
    return "uncertain", "⚠️ Não detectado"


def _render_ppe_assessments(result, details=True):
    if not result.assessments:
        st.caption("Nenhuma pessoa no quadro.")
        return
    evaluated = [kind for kind in theme.EQUIPMENT
                 if any(kind in item.present + item.absent + item.uncertain for item in result.assessments)]
    symbol = {"ok": "✅", "unsafe": "❌", "uncertain": "⚠️"}
    rows = []
    for item in result.assessments:
        items = "".join(f"<span>{symbol[_equipment_status(item, kind)[0]]} {theme.EQUIPMENT[kind]}</span>" for kind in evaluated)
        rows.append(f'<div class="person"><b>Pessoa {item.index}</b><span class="items">{items}</span>'
                    f'{theme.pill(STATUS_NAMES[item.status], item.status)}</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    if not details:
        return  # A table per live frame costs more than the inference itself.
    with st.expander("Detalhes"):
        st.dataframe([{
            "Pessoa": item.index,
            **{theme.EQUIPMENT[kind]: _equipment_status(item, kind)[1] for kind in evaluated},
            "Resultado": STATUS_NAMES[item.status],
            "Evidências": "; ".join(item.reasons),
        } for item in result.assessments], hide_index=True, width="stretch")


def _step_runtime(runtime):
    """Advance one analysis tick; stop the session on end of source or error."""
    try:
        result = runtime.step(st.session_state.confidence, st.session_state.iou)
        if result is None:
            _stop("Imagem analisada." if isinstance(runtime.capture, ImageSource) else
                  "Fim do vídeo ou fonte encerrada. O último frame está disponível para exportação.")
            st.rerun()
        st.session_state.latest_result = result
        st.session_state.latest_frame = runtime.annotated_frame
        st.session_state.latest_preview = runtime.preview_bytes
        st.session_state.preview_error = runtime.preview_error
        st.session_state.observed_fps = runtime.observed_fps
        st.session_state.active_device = runtime.device
        st.session_state.active_metadata["device"] = runtime.device
        record = frame_record(result, illustrative=runtime.illustrative, observed_fps=runtime.observed_fps)
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


def _live_panel():
    runtime = st.session_state.get("runtime")
    if runtime is not None and runtime.closed:
        _stop("Sessão pausada por inatividade. Inicie novamente para reabrir a captura.")
        st.rerun()
    if runtime is not None and runtime.should_process():
        _step_runtime(runtime)

    result = st.session_state.get("latest_result")
    illustrative = st.session_state.get("illustrative", False) if result is not None else st.session_state.get("source_type") == PREVIEW
    active = runtime is not None and not runtime.closed
    has_metrics = result is not None and not illustrative
    people = result.assessments if has_metrics else []
    fps = st.session_state.get("observed_fps")

    metrics = st.columns(4)
    metrics[0].metric("Pessoas", len(people) if has_metrics else "—")
    metrics[1].metric("EPIs completos", sum(item.status == "ok" for item in people) if has_metrics else "—")
    metrics[2].metric("Alertas", sum(item.status == "unsafe" for item in people) if has_metrics else "—")
    metrics[3].metric("FPS", f"{fps:.1f}" if has_metrics and fps is not None else "—")

    if runtime is not None and runtime.event_error:
        st.error(runtime.event_error)
    if runtime is not None and runtime.dispatcher is not None:
        delivery = runtime.dispatcher.snapshot()
        st.caption(f"Envios: {delivery['pending']} pendentes · {delivery['accepted']} aceitos · {delivery['failed']} falharam.")
        if delivery["store_errors"]:
            st.warning("Falha ao gravar o status de envios. Verifique o armazenamento.")
    if st.session_state.get("last_event"):
        event = st.session_state.last_event
        st.success(f"Ocorrência salva: {event['camera_name']} · {event['location']}. Veja a aba Ocorrências.")

    left, right = st.columns([2, 1], gap="large")
    with left:
        frame = st.session_state.get("latest_frame")
        _show_preview(frame)
        if frame is None:
            st.info("Escolha a fonte na barra lateral e clique em ▶ Iniciar.")
        else:
            st.caption(f"{'Ao vivo' if active else 'Pausado'} · quadro {result.frame_index:,} · "
                       f"{st.session_state.active_metadata['source']}"
                       + (f" · inferência {result.inference_ms:.0f} ms" if has_metrics else ""))
        if illustrative and result is not None:
            st.info("Prévia ilustrativa: desenho sem câmera e sem modelo.")
    with right, st.container(border=True):
        st.markdown("**Pessoas detectadas**")
        if has_metrics:
            _render_ppe_assessments(result, details=not active or st.session_state.active_metadata.get("source") == "Imagem")
            _render_ppe_capabilities()
        else:
            st.caption("Os resultados aparecem aqui após iniciar.")


def _show_preview(frame=None):
    content = st.session_state.get("latest_preview") if frame is not None else None
    st.markdown(preview_markup(content), unsafe_allow_html=True)
    if st.session_state.get("preview_error"):
        st.caption(st.session_state.preview_error)


def _remember_model_path(key):
    st.session_state.model_paths[key] = st.session_state[key]


def _model_path_input(label, *, default, key, disabled):
    # Streamlit drops keys for widgets that disappear when the backend changes.
    # A separate non-widget map preserves each profile's user-entered paths.
    if "model_paths" not in st.session_state:
        st.session_state.model_paths = {}
    if key not in st.session_state:
        st.session_state[key] = st.session_state.model_paths.get(key, default)
    return st.text_input(label, key=key, disabled=disabled,
                         on_change=_remember_model_path, args=(key,))


def _model_summary():
    try:
        classes = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["classes"]
        scores = " · ".join(f"{theme.EQUIPMENT[kind]} {100 * values['mAP50']:.0f}%" for kind, values in classes.items())
        return f"Modelo YOLO11n · mAP50 no teste: {scores}"
    except (OSError, ValueError, KeyError):
        return "Modelo YOLO11n"


def main():
    st.set_page_config(page_title="Monitor de EPIs", page_icon="🦺", layout="wide")
    st.markdown(theme.CSS, unsafe_allow_html=True)
    init_alert_settings()
    for key, value in {
        "runtime": None, "history": deque(maxlen=HISTORY_LIMIT), "latest_result": None,
        "latest_frame": None, "latest_preview": None, "preview_error": "",
        "notice": "", "export": None, "confidence": .4, "iou": .45,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
    runtime = st.session_state.runtime
    running = runtime is not None and not runtime.closed

    with st.sidebar:
        st.markdown("### 🦺 Monitor de EPIs")
        source_type = st.selectbox("Fonte de entrada", ["Imagem", "Arquivo de vídeo", "Webcam local", "RTSP / IP", PREVIEW],
                                   key="source_type", disabled=running)
        upload, camera_index, stream_url, video_path = None, 0, "", ""
        if source_type == "Webcam local":
            camera_index = st.number_input("Índice da webcam", min_value=0, max_value=20, value=0, disabled=running)
        elif source_type == "Arquivo de vídeo":
            file_mode = st.radio("Abrir vídeo", ["Enviar arquivo", "Caminho no computador"], disabled=running, horizontal=True)
            if file_mode == "Enviar arquivo":
                upload = st.file_uploader("Vídeo", type=["mp4", "avi", "mov", "mkv", "webm"], disabled=running)
            else:
                video_path = st.text_input("Caminho do vídeo", value="data/demo/demo_obra.mp4", disabled=running)
        elif source_type == "Imagem":
            file_mode = st.radio("Abrir imagem", ["Enviar arquivo", "Caminho no computador"], disabled=running, horizontal=True)
            if file_mode == "Enviar arquivo":
                upload = st.file_uploader("Imagem", type=["jpg", "jpeg", "png", "bmp", "webp"], disabled=running)
            else:
                video_path = st.text_input("Caminho da imagem", disabled=running)
        elif source_type == "RTSP / IP":
            stream_url = st.text_input("URL do stream", type="password", placeholder="rtsp://…", disabled=running)

        st.slider("Confiança mínima", min_value=.05, max_value=.95, step=.05, key="confidence")
        start_col, stop_col = st.columns(2)
        start_clicked = start_col.button("▶ Iniciar", type="primary", width="stretch", disabled=running)
        if stop_col.button("■ Parar", width="stretch", disabled=not running):
            _stop()
            st.rerun()

        with st.expander("Avançado"):
            model_mode = st.radio("Finalidade do modelo", ["EPI treinado", "Demo COCO"], disabled=running or source_type == PREVIEW)
            if "backend" not in st.session_state:
                st.session_state.backend = profiles.recommended_profile()
            backend = st.selectbox("Execução", [profiles.PYTORCH, profiles.OPENVINO],
                                   key="backend", disabled=running or source_type == PREVIEW)
            use_openvino = backend == profiles.OPENVINO
            person_default = profiles.OPENVINO_PERSON if use_openvino else DEFAULT_PERSON_MODEL
            ppe_default = profiles.OPENVINO_PPE if use_openvino else DEFAULT_PPE_MODEL
            default_path = person_default if model_mode == "Demo COCO" else ppe_default
            model_path = _model_path_input("Caminho do modelo", default=default_path, key=f"model_{model_mode}_{backend}",
                                           disabled=running or source_type == PREVIEW)
            person_model_path = person_default
            if model_mode == "EPI treinado":
                speed = st.radio("Velocidade", ["Precisa · 2 modelos", "Rápida · 1 modelo"], key="speed_mode",
                                 disabled=running, help="Rápida usa a classe Person do próprio modelo de EPI "
                                 "(~40% mais FPS). Ótima em cenas de obra; em outros ambientes pode perder pessoas.")
                person_model_path = _model_path_input("Modelo de pessoas — primeira etapa", default=person_default,
                                                     key=f"person_model_{backend}",
                                                     disabled=running or speed.startswith("Rápida"))
                if speed.startswith("Rápida"):
                    person_model_path = model_path
            if use_openvino:
                device = "cpu"
                st.caption("OpenVINO otimizado para CPU, modelos em 480 pixels; não usa quantização.")
                if not profiles.openvino_available():
                    st.warning("Perfil OpenVINO indisponível: faltam a dependência ou os modelos exportados. Use PyTorch.")
            else:
                device = st.selectbox("Dispositivo", ["auto", "cpu", "cuda:0", "mps"], disabled=running or source_type == PREVIEW)
            analysis_rate = st.selectbox("Limite de análises por segundo", list(ANALYSIS_RATES), index=1,
                                         key="analysis_rate", disabled=running)
            st.caption("É um limite, não uma garantia de FPS. Pare para mudar.")
            st.slider("IoU / sobreposição", min_value=.05, max_value=.95, step=.05, key="iou")
            if model_mode == "Demo COCO":
                st.caption("COCO detecta pessoas, mas não identifica EPIs.")
            elif not Path(model_path).exists():
                st.warning("Pesos de EPI não encontrados neste caminho.")
        if start_clicked:
            with st.spinner("Carregando modelo…"):
                _start(source_type, upload, model_path, model_mode, device, camera_index, stream_url, video_path,
                       person_model_path)
            st.rerun()

        settings = st.session_state.alert_settings
        st.divider()
        if st.session_state.telegram_config.enabled:
            telegram_status = "ativo para novas ocorrências" if settings["save_enabled"] else "configurado; envio automático desligado"
        else:
            telegram_status = "desativado"
        st.caption(f"📍 {settings['camera_name']} · {settings['location'] or 'local não informado'}")
        st.caption(f"Gravação: {'ativa' if settings['save_enabled'] else 'desativada'} · Telegram: {telegram_status}.")
        st.caption(_model_summary())

    st.title("Monitor de EPIs")
    st.markdown('<p class="muted">Capacete, colete e bota por pessoa · YOLO11</p>', unsafe_allow_html=True)
    if st.session_state.notice:
        st.info(st.session_state.notice)

    monitor_tab, occurrences_tab, integrations_tab = st.tabs(["Monitor", "Ocorrências", "Configurações"])
    with monitor_tab:
        if model_mode == "Demo COCO" and st.session_state.alert_settings["trigger"] == "ppe":
            st.info("Para salvar presença de pessoas com COCO, escolha 'Pessoa detectada' em Configurações.")

        @st.fragment(run_every=1 / ANALYSIS_RATES[analysis_rate] if running else None)
        def live_fragment():
            _live_panel()
        live_fragment()
        if st.button("Salvar imagem agora",
                     disabled=st.session_state.latest_frame is None or st.session_state.get("illustrative", True)):
            try:
                metadata = st.session_state.active_metadata
                from epi_monitor.events import CameraContext
                context = CameraContext(metadata["camera_id"], metadata["camera_name"], metadata["location"])
                event = event_store().save(st.session_state.latest_result, st.session_state.latest_frame, context,
                                           kind="manual", reasons=["Captura manual solicitada na interface"],
                                           demo_mode=metadata["mode"] == "Demo COCO")
                st.session_state.last_event = event
                st.success("Imagem salva. Abra a aba Ocorrências para visualizar ou baixar.")
            except Exception:
                st.error("Não foi possível salvar a imagem. Verifique a pasta reports e o espaço em disco.")
        _render_export()
    with occurrences_tab:
        @st.fragment
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
                "name": "deteccao_epi_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        export = st.session_state.export
        if export:
            col1, col2 = st.columns(2)
            col1.download_button("↓ Snapshot PNG", export["snapshot"], file_name=f"{export['name']}.png", mime="image/png", width="stretch")
            col2.download_button("↓ Relatório ZIP", export["report"], file_name=f"{export['name']}.zip", mime="application/zip", width="stretch")
            st.caption("Exportação congelada no instante de preparação. Prepare novamente para atualizar.")


if __name__ == "__main__":
    main()
