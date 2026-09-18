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

from safeguard.capture import CaptureError, VideoSource
from safeguard.config import InferenceConfig
from safeguard.inference import YOLODetector
from safeguard.pipeline import Pipeline
from safeguard.rendering import render_frame
from safeguard.reporting import build_report, encode_snapshot, frame_record
from safeguard.types import Detection, FrameResult


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

    def __init__(self, *, source_label, capture=None, pipeline=None, temporary_path=None, illustrative=False):
        self.source_label = source_label
        self.capture = capture
        self.pipeline = pipeline
        self.temporary_path = temporary_path
        self.illustrative = illustrative
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


def _start(source_type, upload, model_path, model_mode, device, camera_index, stream_url):
    temporary_path = None
    capture = None
    try:
        if source_type == PREVIEW:
            runtime = _SessionRuntime(source_label=PREVIEW, illustrative=True)
        else:
            if source_type == "Arquivo de vídeo":
                if upload is None:
                    raise ValueError("Selecione um arquivo de vídeo antes de iniciar.")
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
                detector = YOLODetector(configuration).load()
                capture = VideoSource(source).open()
                pipeline = Pipeline(detector, demo_mode=model_mode == "Demo COCO")
            runtime = _SessionRuntime(
                source_label=source_type, capture=capture, pipeline=pipeline,
                temporary_path=temporary_path,
            )
        st.session_state.runtime = runtime
        st.session_state.history = deque(maxlen=HISTORY_LIMIT)
        st.session_state.latest_result = None
        st.session_state.latest_frame = None
        st.session_state.observed_fps = None
        st.session_state.illustrative = runtime.illustrative
        st.session_state.active_device = runtime.device
        st.session_state.active_metadata = {
            "source": source_type,
            "mode": "illustrative_preview" if runtime.illustrative else model_mode,
            "model": None if runtime.illustrative else Path(model_path).name,
            "device": runtime.device,
            "history_limit_frames": HISTORY_LIMIT,
        }
        st.session_state.export = None
        st.session_state.notice = ""
    except Exception as error:
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


def _live_panel():
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
            st.session_state.latest_frame = result.frame if runtime.illustrative else render_frame(result)
            st.session_state.observed_fps = runtime.observed_fps
            st.session_state.active_device = runtime.device
            st.session_state.active_metadata["device"] = runtime.device
            record = frame_record(
                result, illustrative=runtime.illustrative, observed_fps=runtime.observed_fps,
            )
            record["thresholds"] = {"confidence": st.session_state.confidence, "iou": st.session_state.iou}
            st.session_state.history.append(record)
        except Exception as error:
            if isinstance(error, CaptureError):
                message = "Captura interrompida. Verifique a webcam, o arquivo ou a conexão do stream."
            else:
                message = f"Processamento interrompido: {error}"
            _stop(message)
            st.rerun()

    result = st.session_state.get("latest_result")
    illustrative = st.session_state.get("illustrative", True)
    active = runtime is not None and not runtime.closed
    has_metrics = result is not None and not illustrative
    counts = result.counts if result is not None else {}
    fps = st.session_state.get("observed_fps")
    pipeline_fps = 1000 / result.pipeline_ms if has_metrics and result.pipeline_ms > 0 else None
    metrics = st.columns(4)
    metrics[0].metric("FPS observado", f"{fps:.1f}" if has_metrics and fps is not None else "—")
    metrics[1].metric("Capacidade do pipeline", f"{pipeline_fps:.1f} FPS" if pipeline_fps else "—")
    metrics[2].metric("Latência de inferência", f"{result.inference_ms:.1f} ms" if has_metrics else "—")
    metrics[3].metric("Detecções no frame", sum(counts.values()) if result else "—")
    st.caption("FPS observado: média de até 30 frames, incluindo captura, renderização e interface (até ~10 FPS). "
               "Capacidade do pipeline = 1.000 / tempo de processamento; não inclui a interface.")

    left, right = st.columns([2.35, 1], gap="large")
    with left, st.container(border=True):
        title_col, status_col = st.columns([3, 1])
        title_col.markdown("#### Visão ao vivo")
        status = "PRÉVIA ILUSTRATIVA" if illustrative else ("AO VIVO" if active else "PAUSADO")
        status_col.markdown(f'<span class="pill">{status}</span>', unsafe_allow_html=True)
        frame = st.session_state.get("latest_frame")
        if frame is None:
            st.image(_illustrative_result(0).frame, channels="BGR", width="stretch")
            st.caption("Ilustração da interface. Escolha uma fonte e pressione Iniciar para executar.")
        else:
            st.image(frame, channels="BGR", width="stretch")
            st.caption(f"Frame {result.frame_index:,} · {st.session_state.active_metadata['source']} · "
                       f"Dispositivo: {st.session_state.active_device}")
        if illustrative:
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
    st.markdown(STYLE, unsafe_allow_html=True)
    for key, value in {
        "runtime": None, "history": deque(maxlen=HISTORY_LIMIT), "latest_result": None,
        "latest_frame": None, "notice": "", "export": None, "confidence": .4, "iou": .45,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
    runtime = st.session_state.runtime
    running = runtime is not None and not runtime.closed

    with st.sidebar:
        st.markdown('<div class="brand">Safe<span>Guard</span></div>', unsafe_allow_html=True)
        st.caption("VISÃO COMPUTACIONAL · TCC")
        st.divider()
        st.markdown("#### Configuração da sessão")
        source_type = st.selectbox("Fonte de entrada", [PREVIEW, "Webcam local", "Arquivo de vídeo", "RTSP / IP"], disabled=running)
        upload, camera_index, stream_url = None, 0, ""
        if source_type == "Webcam local":
            camera_index = st.number_input("Índice da webcam", min_value=0, max_value=20, value=0, disabled=running)
            st.caption("A webcam deve estar conectada ao computador que executa o servidor.")
        elif source_type == "Arquivo de vídeo":
            upload = st.file_uploader("Vídeo", type=["mp4", "avi", "mov", "mkv", "webm"], disabled=running)
        elif source_type == "RTSP / IP":
            stream_url = st.text_input("URL do stream", type="password", placeholder="rtsp://…", disabled=running)
        model_mode = st.radio("Finalidade do modelo", ["Demo COCO", "EPI treinado"], disabled=running or source_type == PREVIEW)
        default_path = "models/yolo11n.pt" if model_mode == "Demo COCO" else "models/ppe/best.pt"
        model_path = st.text_input("Caminho do modelo", value=default_path, key=f"model_{model_mode}", disabled=running or source_type == PREVIEW)
        device = st.selectbox("Dispositivo", ["auto", "cpu", "cuda:0", "mps"], disabled=running or source_type == PREVIEW)
        if model_mode == "Demo COCO":
            st.caption("COCO demonstra detecção geral; não é um modelo de identificação de EPIs.")
        else:
            st.caption("Use pesos treinados no dataset de EPI. Confira as classes e as métricas de validação.")
        st.divider()
        st.markdown("#### Ajustes em tempo real")
        st.slider("Confiança mínima", min_value=.05, max_value=.95, step=.05, key="confidence")
        st.slider("IoU / sobreposição", min_value=.05, max_value=.95, step=.05, key="iou")
        start_col, stop_col = st.columns(2)
        if start_col.button("▶ Iniciar", type="primary", width="stretch", disabled=running):
            _start(source_type, upload, model_path, model_mode, device, camera_index, stream_url)
            st.rerun()
        if stop_col.button("■ Parar", width="stretch", disabled=not running):
            _stop()
            st.rerun()
        st.caption("Cada sessão possui sua própria captura. Ao sair, a liberação por inatividade ocorre em aproximadamente 45–50 s.")

    st.markdown('<div class="eyebrow">INTELIGÊNCIA VISUAL PARA AMBIENTES DE TRABALHO</div>', unsafe_allow_html=True)
    st.title("Monitoramento inteligente")
    st.markdown('<p class="muted">Detecção em tempo real, indicadores claros e evidências prontas para sua apresentação.</p>', unsafe_allow_html=True)
    if st.session_state.notice:
        st.info(st.session_state.notice)

    @st.fragment(run_every=.1 if running else None)
    def live_fragment():
        _live_panel()

    live_fragment()
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
