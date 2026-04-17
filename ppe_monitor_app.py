#!/usr/bin/env python3
# ============================================================
#  ppe_monitor_app.py  —  SafeGuard EPI Monitor
#  Aplicativo Desktop Profissional — CustomTkinter + OpenCV + YOLOv8
#  TCC: Sistema Inteligente de Monitoramento de EPI
#       Visão Computacional e IA em Tempo Real
#
#  Execução:
#    python ppe_monitor_app.py
#
#  Gerar .exe:
#    pyinstaller --onefile --windowed ppe_monitor_app.py
# ============================================================

import json
import os
import queue
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# ── Garante que a raiz do projeto esteja no sys.path ─────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # aplica patch PyTorch 2.6 logo de início
from config import (
    MODEL_PATH, CONFIDENCE, IOU_THRESH,
    VIOLATIONS_DIR, LOGS_DIR,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    ALERT_COOLDOWN_SECONDS,
    HEAD_REGION_RATIO, HELMET_IOU_THRESHOLD,
    REQUIRE_HELMET, REQUIRE_VEST,
    DEMO_MODE,
)
from src.ai.detector     import EPIDetector, FrameResult
from src.rules.ppe_rules import PPERulesEngine, draw_ppe_status, PersonStatus
from src.alerts.logger   import ViolationLogger
from src.alerts.telegram import TelegramAlerter

# ── Tema e aparência ──────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paleta de cores da aplicação ──────────────────────────────
C = {
    "bg_primary":   "#0d1117",   # fundo principal
    "bg_secondary": "#161b22",   # sidebar
    "bg_card":      "#1c2230",   # cartões
    "bg_input":     "#21262d",   # inputs
    "accent_blue":  "#1f6feb",   # azul industrial
    "accent_green": "#238636",   # verde conforme
    "accent_red":   "#da3633",   # vermelho alerta
    "accent_amber": "#d29922",   # amarelo atenção
    "text_primary": "#e6edf3",   # texto principal
    "text_muted":   "#8b949e",   # texto secundário
    "border":       "#30363d",   # bordas
    "sidebar_w":    240,         # largura sidebar
}

APP_TITLE   = "SafeGuard EPI Monitor"
APP_VERSION = "v2.0.0"
COMPANY     = "Indústria ABC — Unidade São Paulo"

# ─────────────────────────────────────────────────────────────
#  UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

def load_violations_json() -> list:
    """Carrega o histórico de violações do JSON."""
    p = LOGS_DIR / "violations.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def cv2_to_ctk(frame: np.ndarray,
               target_w: int,
               target_h: int) -> ctk.CTkImage:
    """Converte frame OpenCV (BGR) para CTkImage escalado."""
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    pil  = pil.resize((target_w, target_h), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil,
                        size=(target_w, target_h))


# ─────────────────────────────────────────────────────────────
#  THREAD DE VÍDEO
# ─────────────────────────────────────────────────────────────

class VideoThread(threading.Thread):
    """
    Thread dedicada à captura e processamento de vídeo.
    Publica resultados em uma queue não-bloqueante para a UI.
    Isso evita qualquer travamento da interface gráfica.
    """

    def __init__(self,
                 camera_index: int,
                 detector:     EPIDetector,
                 rules_engine: PPERulesEngine,
                 vlogger:      ViolationLogger,
                 telegram:     TelegramAlerter,
                 result_queue: queue.Queue,
                 use_dshow:    bool = True):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.detector     = detector
        self.rules        = rules_engine
        self.vlogger      = vlogger
        self.telegram     = telegram
        self.result_queue = result_queue
        self.use_dshow    = use_dshow
        self._running     = threading.Event()
        self._running.set()

        # Estatísticas em tempo real
        self.fps_deque    = deque(maxlen=30)
        self.fps          = 0.0
        self.frame_count  = 0
        self.error_msg    = ""

    def run(self):
        """Loop principal da thread de vídeo."""
        backend = cv2.CAP_DSHOW if self.use_dshow and sys.platform == "win32" else 0
        cap = cv2.VideoCapture(self.camera_index + backend
                               if backend else self.camera_index)

        if not cap.isOpened():
            # Tenta sem backend específico como fallback
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.error_msg = f"Câmera {self.camera_index} não disponível"
            self.result_queue.put({"type": "error", "msg": self.error_msg})
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._running.is_set():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1

            # ── Detecção ───────────────────────────────────────
            frame_result = self.detector.detect(frame)
            statuses     = self.rules.evaluate(frame_result)

            # ── Anotação de status EPI ─────────────────────────
            annotated = frame_result.annotated_frame.copy()
            annotated = draw_ppe_status(annotated, statuses)

            # ── Violações ──────────────────────────────────────
            violators = [s for s in statuses if not s.is_compliant]
            saved_path = None
            if violators:
                saved_path = self.vlogger.log_violation(annotated, statuses)
                viols_flat = []
                for s in violators:
                    viols_flat.extend(s.violations)
                self.telegram.send_violation_alert(
                    violations=list(set(viols_flat)),
                    image_path=saved_path,
                    person_count=len(violators),
                )

            # ── FPS ────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            self.fps_deque.append(elapsed)
            if self.fps_deque:
                self.fps = 1.0 / (sum(self.fps_deque) / len(self.fps_deque))

            # ── Envia resultado para UI ────────────────────────
            # Descarta frame anterior se UI não consumiu (evita lag)
            if self.result_queue.qsize() < 2:
                self.result_queue.put({
                    "type":       "frame",
                    "frame":      annotated,
                    "statuses":   statuses,
                    "fps":        round(self.fps, 1),
                    "persons":    len(frame_result.persons),
                    "violators":  len(violators),
                    "saved_path": saved_path,
                    "frame_num":  self.frame_count,
                })

        cap.release()

    def stop(self):
        self._running.clear()


# ─────────────────────────────────────────────────────────────
#  JANELA DE LOGIN
# ─────────────────────────────────────────────────────────────

class LoginWindow(ctk.CTk):
    """
    Tela de login profissional com validação de credenciais.
    Credenciais de demo: admin / 1234
    """

    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} — Login")
        self.geometry("480x600")
        self.resizable(False, False)
        self.configure(fg_color=C["bg_primary"])
        self._center_window(480, 600)
        self.success = False
        self._build_ui()

    def _center_window(self, w: int, h: int):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # ── Container central ─────────────────────────────────
        frame = ctk.CTkFrame(self, fg_color=C["bg_secondary"],
                             corner_radius=16,
                             border_width=1, border_color=C["border"])
        frame.place(relx=0.5, rely=0.5, anchor="center",
                    relwidth=0.88, relheight=0.90)

        # ── Ícone / Logo ──────────────────────────────────────
        icon_lbl = ctk.CTkLabel(frame, text="🛡️",
                                font=ctk.CTkFont(size=64))
        icon_lbl.pack(pady=(40, 4))

        ctk.CTkLabel(frame, text=APP_TITLE,
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=C["text_primary"]).pack()

        ctk.CTkLabel(frame,
                     text="Monitoramento Inteligente de EPI · Indústria 4.0",
                     font=ctk.CTkFont(size=11),
                     text_color=C["text_muted"]).pack(pady=(2, 0))

        ctk.CTkLabel(frame, text=APP_VERSION,
                     font=ctk.CTkFont(size=10),
                     text_color=C["accent_blue"]).pack(pady=(0, 28))

        # ── Separador ─────────────────────────────────────────
        sep = ctk.CTkFrame(frame, height=1, fg_color=C["border"])
        sep.pack(fill="x", padx=30, pady=(0, 28))

        # ── Campos ────────────────────────────────────────────
        input_w = 320

        ctk.CTkLabel(frame, text="Usuário",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C["text_muted"],
                     anchor="w").pack(padx=50, fill="x")

        self.entry_user = ctk.CTkEntry(
            frame, width=input_w, height=42,
            placeholder_text="Digite seu usuário",
            fg_color=C["bg_input"], border_color=C["border"],
            font=ctk.CTkFont(size=13))
        self.entry_user.pack(pady=(4, 16))
        self.entry_user.insert(0, "admin")

        ctk.CTkLabel(frame, text="Senha",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C["text_muted"],
                     anchor="w").pack(padx=50, fill="x")

        self.entry_pass = ctk.CTkEntry(
            frame, width=input_w, height=42,
            placeholder_text="Digite sua senha",
            show="●",
            fg_color=C["bg_input"], border_color=C["border"],
            font=ctk.CTkFont(size=13))
        self.entry_pass.pack(pady=(4, 8))
        self.entry_pass.insert(0, "1234")

        # ── Mensagem de erro ──────────────────────────────────
        self.lbl_error = ctk.CTkLabel(frame, text="",
                                      font=ctk.CTkFont(size=11),
                                      text_color=C["accent_red"])
        self.lbl_error.pack(pady=(0, 12))

        # ── Botão entrar ──────────────────────────────────────
        self.btn_login = ctk.CTkButton(
            frame, text="  Entrar no Sistema  →",
            width=input_w, height=46,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=C["accent_blue"],
            hover_color="#1a5cba",
            corner_radius=8,
            command=self._do_login)
        self.btn_login.pack(pady=(0, 20))

        # ── Dica de demo ──────────────────────────────────────
        ctk.CTkLabel(frame,
                     text="Demo: admin / 1234",
                     font=ctk.CTkFont(size=10),
                     text_color=C["text_muted"]).pack()

        # Enter no teclado faz login
        self.entry_pass.bind("<Return>", lambda _: self._do_login())
        self.entry_user.bind("<Return>", lambda _: self.entry_pass.focus())

    def _do_login(self):
        user = self.entry_user.get().strip()
        pwd  = self.entry_pass.get().strip()

        self.btn_login.configure(text="Verificando...", state="disabled")
        self.after(600, lambda: self._validate(user, pwd))

    def _validate(self, user: str, pwd: str):
        if user == "admin" and pwd == "1234":
            self.success = True
            self.destroy()
        else:
            self.lbl_error.configure(text="⚠  Usuário ou senha incorretos")
            self.btn_login.configure(
                text="  Entrar no Sistema  →", state="normal")
            self.entry_pass.delete(0, "end")
            self.entry_pass.focus()


# ─────────────────────────────────────────────────────────────
#  WIDGET: CARTÃO KPI
# ─────────────────────────────────────────────────────────────

class KPICard(ctk.CTkFrame):
    """Cartão de métrica com ícone, valor grande e rótulo."""

    def __init__(self, master, icon: str, label: str,
                 value: str = "—", color: str = "#4fc3f7", **kw):
        super().__init__(master, fg_color=C["bg_card"],
                         corner_radius=12,
                         border_width=1, border_color=C["border"], **kw)
        self._color = color

        ctk.CTkLabel(self, text=icon,
                     font=ctk.CTkFont(size=28)).pack(pady=(18, 4))
        self._val_lbl = ctk.CTkLabel(self, text=value,
                                     font=ctk.CTkFont(size=32, weight="bold"),
                                     text_color=color)
        self._val_lbl.pack()
        ctk.CTkLabel(self, text=label,
                     font=ctk.CTkFont(size=11),
                     text_color=C["text_muted"]).pack(pady=(4, 18))

    def set_value(self, v: str):
        self._val_lbl.configure(text=str(v))


# ─────────────────────────────────────────────────────────────
#  WIDGET: LINHA DE LOG
# ─────────────────────────────────────────────────────────────

class LogRow(ctk.CTkFrame):
    """Uma linha formatada para o feed de logs de auditoria."""

    def __init__(self, master, ts: str, event: str,
                 level: str = "INFO", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        colors = {"INFO": C["text_muted"], "WARN": C["accent_amber"],
                  "ALERT": C["accent_red"], "OK": C["accent_green"]}
        c = colors.get(level, C["text_muted"])
        ctk.CTkLabel(self, text=ts,
                     font=ctk.CTkFont(size=10, family="Courier"),
                     text_color=C["text_muted"], width=140,
                     anchor="w").pack(side="left", padx=(8, 4))
        ctk.CTkLabel(self, text=f"[{level}]",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=c, width=52,
                     anchor="w").pack(side="left", padx=4)
        ctk.CTkLabel(self, text=event,
                     font=ctk.CTkFont(size=10),
                     text_color=C["text_primary"],
                     anchor="w").pack(side="left", padx=4)


# ─────────────────────────────────────────────────────────────
#  JANELA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class MainApp(ctk.CTk):
    """
    Janela principal do SafeGuard EPI Monitor.
    Layout: Sidebar fixa à esquerda + área de conteúdo por abas.
    """

    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE}  |  {COMPANY}")
        self.geometry("1280x760")
        self.minsize(1100, 680)
        self.configure(fg_color=C["bg_primary"])
        self._center_window(1280, 760)

        # ── Componentes de back-end ───────────────────────────
        self.detector     = None
        self.rules_engine = None
        self.vlogger      = ViolationLogger()
        self.telegram     = TelegramAlerter()
        self.video_thread = None
        self.result_queue = queue.Queue(maxsize=4)

        # ── Estado da aplicação ───────────────────────────────
        self._monitoring   = False
        self._current_tab  = "monitor"
        self._cam_index    = ctk.IntVar(value=0)
        self._confidence   = ctk.DoubleVar(value=CONFIDENCE)
        self._iou          = ctk.DoubleVar(value=IOU_THRESH)
        self._model_var    = ctk.StringVar(value=Path(MODEL_PATH).name)
        self._req_helmet   = ctk.BooleanVar(value=REQUIRE_HELMET)
        self._req_vest     = ctk.BooleanVar(value=REQUIRE_VEST)
        self._tg_token     = ctk.StringVar(value=TELEGRAM_TOKEN)
        self._tg_chat      = ctk.StringVar(value=TELEGRAM_CHAT_ID)
        self._demo_badge   = DEMO_MODE
        self._audit_events = []  # [(ts, level, msg)]

        # ── Estatísticas acumuladas ────────────────────────────
        self._stats = {
            "frames":       0,
            "persons":      0,
            "violations":   0,
            "sessions":     0,
            "detect_times": deque(maxlen=200),
        }

        # ── Construção da UI ──────────────────────────────────
        self._build_sidebar()
        self._build_content_area()
        self._switch_tab("monitor")  # deve vir APÓS _build_content_area

        # ── Inicializa detector em background ─────────────────
        self._add_audit("Sistema iniciado", "INFO")
        threading.Thread(target=self._init_detector, daemon=True).start()

        # ── Loop de atualização da UI ─────────────────────────
        self._poll_queue()

    # ─── Layout ──────────────────────────────────────────────

    def _center_window(self, w: int, h: int):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_sidebar(self):
        """Monta a barra lateral fixa."""
        sb = ctk.CTkFrame(self, width=C["sidebar_w"],
                          fg_color=C["bg_secondary"],
                          corner_radius=0,
                          border_width=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        self._sidebar = sb

        # Logo + nome
        ctk.CTkLabel(sb, text="🛡️",
                     font=ctk.CTkFont(size=36)).pack(pady=(28, 0))
        ctk.CTkLabel(sb, text=APP_TITLE,
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C["text_primary"],
                     wraplength=C["sidebar_w"] - 20).pack(pady=(4, 0))
        ctk.CTkLabel(sb, text=COMPANY,
                     font=ctk.CTkFont(size=9),
                     text_color=C["text_muted"],
                     wraplength=C["sidebar_w"] - 20).pack(pady=(2, 0))

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(
            fill="x", padx=16, pady=16)

        # Menu de navegação
        menu_items = [
            ("📹", "Monitoramento ao Vivo", "monitor"),
            ("📊", "Dashboard",              "dashboard"),
            ("📋", "Relatórios",             "reports"),
            ("⚙️",  "Configurações",          "settings"),
            ("📜", "Logs de Auditoria",      "audit"),
            ("🎓", "Sobre o TCC",            "about"),
        ]
        self._menu_btns = {}
        for icon, label, key in menu_items:
            btn = ctk.CTkButton(
                sb, text=f"  {icon}  {label}",
                anchor="w", height=40,
                fg_color="transparent",
                hover_color=C["bg_card"],
                text_color=C["text_muted"],
                font=ctk.CTkFont(size=12),
                corner_radius=8,
                command=lambda k=key: self._switch_tab(k))
            btn.pack(fill="x", padx=10, pady=2)
            self._menu_btns[key] = btn

        # Highlight inicial (sem chamar _switch_tab — _tabs ainda nao existe)
        for k, btn in self._menu_btns.items():
            if k == "monitor":
                btn.configure(fg_color=C["bg_card"],
                              text_color=C["text_primary"])
            else:
                btn.configure(fg_color="transparent",
                              text_color=C["text_muted"])

        # Espaçador
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)

        # Status câmera
        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(
            fill="x", padx=16, pady=(0, 12))

        status_row = ctk.CTkFrame(sb, fg_color="transparent")
        status_row.pack(fill="x", padx=14, pady=(0, 6))
        self._cam_dot = ctk.CTkLabel(status_row, text="●",
                                     font=ctk.CTkFont(size=14),
                                     text_color=C["accent_red"])
        self._cam_dot.pack(side="left")
        self._cam_status_lbl = ctk.CTkLabel(
            status_row, text="Câmera offline",
            font=ctk.CTkFont(size=11), text_color=C["text_muted"])
        self._cam_status_lbl.pack(side="left", padx=6)

        self._fps_lbl = ctk.CTkLabel(sb, text="FPS: —",
                                     font=ctk.CTkFont(size=10),
                                     text_color=C["text_muted"])
        self._fps_lbl.pack(pady=(0, 8))

        # Botão Iniciar / Parar
        self._btn_toggle = ctk.CTkButton(
            sb, text="▶  Iniciar Monitoramento",
            height=44, corner_radius=8,
            fg_color=C["accent_green"],
            hover_color="#1a6e28",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._toggle_monitoring)
        self._btn_toggle.pack(fill="x", padx=14, pady=(0, 18))

    def _build_content_area(self):
        """Monta a área principal com as abas."""
        self._content = ctk.CTkFrame(self, fg_color=C["bg_primary"],
                                     corner_radius=0)
        self._content.pack(side="left", fill="both", expand=True)

        self._tabs: dict[str, ctk.CTkFrame] = {}
        self._tabs["monitor"]   = self._build_tab_monitor()
        self._tabs["dashboard"] = self._build_tab_dashboard()
        self._tabs["reports"]   = self._build_tab_reports()
        self._tabs["settings"]  = self._build_tab_settings()
        self._tabs["audit"]     = self._build_tab_audit()
        self._tabs["about"]     = self._build_tab_about()

    # ─── Abas ────────────────────────────────────────────────

    def _build_tab_monitor(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        # HUD superior
        hud = ctk.CTkFrame(tab, fg_color=C["bg_secondary"],
                           corner_radius=0, height=52)
        hud.pack(fill="x")
        hud.pack_propagate(False)

        def _hud_item(parent, icon, label, init="—"):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(side="left", padx=18, pady=6)
            ctk.CTkLabel(f, text=f"{icon}  {label}",
                         font=ctk.CTkFont(size=10),
                         text_color=C["text_muted"]).pack(anchor="w")
            lbl = ctk.CTkLabel(f, text=init,
                               font=ctk.CTkFont(size=14, weight="bold"),
                               text_color=C["text_primary"])
            lbl.pack(anchor="w")
            return lbl

        self._hud_fps     = _hud_item(hud, "⚡", "FPS")
        self._hud_persons = _hud_item(hud, "👤", "Pessoas")
        self._hud_alerts  = _hud_item(hud, "⚠️",  "Alertas hoje", "0")
        self._hud_mode    = _hud_item(hud, "🤖", "Modelo",
                                      Path(MODEL_PATH).name)

        # Relógio à direita
        self._hud_clock = ctk.CTkLabel(hud,
                                       text="",
                                       font=ctk.CTkFont(size=11),
                                       text_color=C["text_muted"])
        self._hud_clock.pack(side="right", padx=18)
        self._update_clock()

        # Feed de vídeo
        video_frame = ctk.CTkFrame(tab, fg_color="#000000",
                                   corner_radius=0)
        video_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self._video_label = ctk.CTkLabel(video_frame, text="",
                                         fg_color="transparent")
        self._video_label.pack(fill="both", expand=True)

        # Placeholder antes de iniciar
        self._placeholder_lbl = ctk.CTkLabel(
            video_frame,
            text="📷\n\nClique em  ▶ Iniciar Monitoramento\npara ativar a câmera",
            font=ctk.CTkFont(size=16),
            text_color=C["text_muted"])
        self._placeholder_lbl.place(relx=0.5, rely=0.5, anchor="center")

        return tab

    def _build_tab_dashboard(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        # Título
        ctk.CTkLabel(tab, text="📊  Dashboard de Monitoramento",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=C["text_primary"]).pack(
                         anchor="w", padx=24, pady=(18, 4))
        ctk.CTkLabel(tab,
                     text="Indicadores de desempenho e conformidade em tempo real",
                     font=ctk.CTkFont(size=11),
                     text_color=C["text_muted"]).pack(anchor="w", padx=24)

        # ── KPIs ──────────────────────────────────────────────
        kpi_row = ctk.CTkFrame(tab, fg_color="transparent")
        kpi_row.pack(fill="x", padx=20, pady=(16, 8))

        self._kpi_alerts    = KPICard(kpi_row, "⚠️",  "Alertas Hoje",
                                      "0", C["accent_red"])
        self._kpi_conform   = KPICard(kpi_row, "✅", "Taxa Conformidade",
                                      "—", C["accent_green"])
        self._kpi_persons   = KPICard(kpi_row, "👤", "Pessoas Monitoradas",
                                      "0", "#4fc3f7")
        self._kpi_det_time  = KPICard(kpi_row, "⚡", "Tempo Médio Det.",
                                      "—", C["accent_amber"])

        for card in (self._kpi_alerts, self._kpi_conform,
                     self._kpi_persons, self._kpi_det_time):
            card.pack(side="left", expand=True, fill="both",
                      padx=6, ipady=8)

        # ── Gráfico (matplotlib embed) ────────────────────────
        chart_row = ctk.CTkFrame(tab, fg_color="transparent")
        chart_row.pack(fill="both", expand=True, padx=20, pady=8)

        self._chart_frame = ctk.CTkFrame(chart_row,
                                         fg_color=C["bg_card"],
                                         corner_radius=12,
                                         border_width=1,
                                         border_color=C["border"])
        self._chart_frame.pack(fill="both", expand=True, padx=4)

        self._chart_label = ctk.CTkLabel(
            self._chart_frame,
            text="📈  Gráfico disponível após iniciar monitoramento",
            font=ctk.CTkFont(size=12),
            text_color=C["text_muted"])
        self._chart_label.pack(expand=True)

        # ── Últimas violações ─────────────────────────────────
        ctk.CTkLabel(tab,
                     text="📸  Últimas Violações",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=C["text_primary"]).pack(
                         anchor="w", padx=24, pady=(0, 6))

        self._violations_scroll = ctk.CTkScrollableFrame(
            tab, height=130, fg_color=C["bg_card"],
            corner_radius=12)
        self._violations_scroll.pack(fill="x", padx=20, pady=(0, 12))

        self._refresh_dashboard()
        return tab

    def _build_tab_reports(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        ctk.CTkLabel(tab, text="📋  Relatórios e Exportação",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=C["text_primary"]).pack(
                         anchor="w", padx=24, pady=(18, 4))
        ctk.CTkLabel(tab,
                     text="Exporte dados para análise e apresentação na banca",
                     font=ctk.CTkFont(size=11),
                     text_color=C["text_muted"]).pack(anchor="w", padx=24)

        # Botões de exportação
        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=16)

        ctk.CTkButton(btn_row,
                      text="📄  Exportar CSV",
                      width=180, height=42,
                      fg_color=C["accent_blue"],
                      command=self._export_csv).pack(side="left", padx=6)

        ctk.CTkButton(btn_row,
                      text="🖨️  Gerar Relatório TXT",
                      width=200, height=42,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._export_report).pack(side="left", padx=6)

        ctk.CTkButton(btn_row,
                      text="🔄  Atualizar Tabela",
                      width=160, height=42,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._refresh_reports_table).pack(
                          side="left", padx=6)

        # Tabela de registros
        cols_frame = ctk.CTkFrame(tab, fg_color=C["bg_card"],
                                  corner_radius=12,
                                  border_width=1,
                                  border_color=C["border"])
        cols_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # Header da tabela
        hdr = ctk.CTkFrame(cols_frame, fg_color=C["bg_input"],
                           corner_radius=0)
        hdr.pack(fill="x")
        for col, w in [("Data/Hora", 170), ("Pessoas", 80),
                       ("Violações", 260), ("Arquivo", 220)]:
            ctk.CTkLabel(hdr, text=col, width=w,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=C["text_muted"],
                         anchor="w").pack(side="left", padx=10, pady=8)

        # Scroll de linhas
        self._report_scroll = ctk.CTkScrollableFrame(
            cols_frame, fg_color="transparent", corner_radius=0)
        self._report_scroll.pack(fill="both", expand=True)

        self._report_rows = []
        self._refresh_reports_table()
        return tab

    def _build_tab_settings(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        scroll = ctk.CTkScrollableFrame(tab, fg_color=C["bg_primary"])
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        def section(title: str):
            ctk.CTkLabel(scroll, text=title,
                         font=ctk.CTkFont(size=14, weight="bold"),
                         text_color=C["text_primary"]).pack(
                             anchor="w", padx=20, pady=(20, 6))
            ctk.CTkFrame(scroll, height=1,
                         fg_color=C["border"]).pack(
                             fill="x", padx=20, pady=(0, 10))

        def row_label(parent, text):
            ctk.CTkLabel(parent, text=text,
                         font=ctk.CTkFont(size=12),
                         text_color=C["text_muted"],
                         width=160, anchor="w").pack(side="left", padx=(20, 8))

        # ── Câmera ────────────────────────────────────────────
        section("📷  Câmera")

        r1 = ctk.CTkFrame(scroll, fg_color="transparent")
        r1.pack(fill="x")
        row_label(r1, "Índice da câmera:")
        self._cam_combo = ctk.CTkComboBox(
            r1, values=["0", "1", "2", "3", "4"],
            variable=None, width=80,
            command=lambda v: self._cam_index.set(int(v)))
        self._cam_combo.set("0")
        self._cam_combo.pack(side="left")

        ctk.CTkButton(scroll,
                      text="🔍  Diagnosticar Câmeras Disponíveis",
                      height=36, width=280,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._diagnose_cameras).pack(
                          anchor="w", padx=20, pady=(8, 0))

        self._diag_result = ctk.CTkLabel(
            scroll, text="",
            font=ctk.CTkFont(size=10),
            text_color=C["text_muted"],
            wraplength=700, justify="left")
        self._diag_result.pack(anchor="w", padx=20, pady=4)

        ctk.CTkButton(scroll,
                      text="📁  Usar Vídeo de Arquivo (fallback)",
                      height=36, width=240,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._pick_video_file).pack(
                          anchor="w", padx=20, pady=(4, 0))

        # ── Modelo ────────────────────────────────────────────
        section("🤖  Modelo YOLOv8")

        r2 = ctk.CTkFrame(scroll, fg_color="transparent")
        r2.pack(fill="x")
        row_label(r2, "Modelo:")
        ctk.CTkComboBox(
            r2, values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            variable=self._model_var, width=160).pack(side="left")

        # Confidence
        r3 = ctk.CTkFrame(scroll, fg_color="transparent")
        r3.pack(fill="x", pady=(10, 0))
        row_label(r3, "Confiança:")
        sl_conf = ctk.CTkSlider(r3, from_=0.1, to=0.9,
                                variable=self._confidence,
                                width=200)
        sl_conf.pack(side="left")
        lbl_conf = ctk.CTkLabel(r3, text=f"{CONFIDENCE:.2f}",
                                width=40, text_color=C["text_muted"])
        lbl_conf.pack(side="left", padx=8)
        sl_conf.configure(command=lambda v: lbl_conf.configure(
            text=f"{float(v):.2f}"))

        # IoU
        r4 = ctk.CTkFrame(scroll, fg_color="transparent")
        r4.pack(fill="x", pady=(8, 0))
        row_label(r4, "IoU Threshold:")
        sl_iou = ctk.CTkSlider(r4, from_=0.1, to=0.9,
                               variable=self._iou, width=200)
        sl_iou.pack(side="left")
        lbl_iou = ctk.CTkLabel(r4, text=f"{IOU_THRESH:.2f}",
                               width=40, text_color=C["text_muted"])
        lbl_iou.pack(side="left", padx=8)
        sl_iou.configure(command=lambda v: lbl_iou.configure(
            text=f"{float(v):.2f}"))

        # ── Regras EPI ────────────────────────────────────────
        section("🦺  Regras de EPI")

        r5 = ctk.CTkFrame(scroll, fg_color="transparent")
        r5.pack(fill="x")
        ctk.CTkCheckBox(r5, text="Capacete obrigatório",
                        variable=self._req_helmet).pack(
                            side="left", padx=20)
        ctk.CTkCheckBox(r5, text="Colete obrigatório",
                        variable=self._req_vest).pack(
                            side="left", padx=20)

        # ── Telegram ──────────────────────────────────────────
        section("📱  Alertas Telegram")

        r6 = ctk.CTkFrame(scroll, fg_color="transparent")
        r6.pack(fill="x")
        row_label(r6, "Token do Bot:")
        ctk.CTkEntry(r6, textvariable=self._tg_token,
                     width=340, show="●").pack(side="left")

        r7 = ctk.CTkFrame(scroll, fg_color="transparent")
        r7.pack(fill="x", pady=(8, 0))
        row_label(r7, "Chat ID:")
        ctk.CTkEntry(r7, textvariable=self._tg_chat,
                     width=200).pack(side="left")

        ctk.CTkButton(scroll, text="🔔  Testar Conexão Telegram",
                      height=36, width=220,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._test_telegram).pack(
                          anchor="w", padx=20, pady=(10, 0))

        self._tg_status = ctk.CTkLabel(
            scroll, text="",
            font=ctk.CTkFont(size=11),
            text_color=C["text_muted"])
        self._tg_status.pack(anchor="w", padx=20, pady=4)

        # Botão salvar
        ctk.CTkButton(scroll, text="💾  Salvar Configurações",
                      height=40, width=200,
                      fg_color=C["accent_blue"],
                      command=self._save_settings).pack(
                          anchor="w", padx=20, pady=(16, 20))

        return tab

    def _build_tab_audit(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(18, 8))
        ctk.CTkLabel(hdr, text="📜  Logs de Auditoria",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=C["text_primary"]).pack(side="left")
        ctk.CTkButton(hdr, text="🗑  Limpar",
                      width=90, height=30,
                      fg_color=C["bg_card"],
                      border_width=1, border_color=C["border"],
                      command=self._clear_audit).pack(side="right")

        # Header de colunas
        col_hdr = ctk.CTkFrame(tab, fg_color=C["bg_input"],
                               corner_radius=0)
        col_hdr.pack(fill="x", padx=20)
        for col, w in [("Timestamp", 150), ("Nível", 70), ("Evento", 600)]:
            ctk.CTkLabel(col_hdr, text=col, width=w,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=C["text_muted"],
                         anchor="w").pack(side="left", padx=8, pady=6)

        self._audit_scroll = ctk.CTkScrollableFrame(
            tab, fg_color=C["bg_card"],
            corner_radius=0)
        self._audit_scroll.pack(fill="both", expand=True,
                                padx=20, pady=(0, 16))

        return tab

    def _build_tab_about(self) -> ctk.CTkFrame:
        tab = ctk.CTkFrame(self._content, fg_color=C["bg_primary"])

        card = ctk.CTkFrame(tab, fg_color=C["bg_card"],
                            corner_radius=16,
                            border_width=1, border_color=C["border"])
        card.place(relx=0.5, rely=0.5, anchor="center",
                   relwidth=0.72, relheight=0.82)

        ctk.CTkLabel(card, text="🛡️",
                     font=ctk.CTkFont(size=56)).pack(pady=(36, 4))
        ctk.CTkLabel(card, text=APP_TITLE,
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=C["text_primary"]).pack()
        ctk.CTkLabel(card, text=APP_VERSION,
                     font=ctk.CTkFont(size=12),
                     text_color=C["accent_blue"]).pack(pady=(2, 16))

        ctk.CTkFrame(card, height=1, fg_color=C["border"]).pack(
            fill="x", padx=40)

        info = (
            "Sistema Inteligente de Monitoramento de Equipamentos\n"
            "de Proteção Individual — Visão Computacional e IA em Tempo Real\n\n"
            "Tecnologias:  YOLOv8 · OpenCV · CustomTkinter · Python 3.10+\n"
            "Arquitetura:  Detecção YOLO + Motor de Regras Espaciais (IoU)\n"
            "Alertas:      Telegram Bot API\n"
            "Dashboard:    Streamlit (modo web)\n\n"
            f"Modelo ativo : {Path(MODEL_PATH).name}\n"
            f"Diretório:    {ROOT}"
        )
        ctk.CTkLabel(card, text=info,
                     font=ctk.CTkFont(size=12),
                     text_color=C["text_muted"],
                     justify="center").pack(pady=20)

        ctk.CTkFrame(card, height=1, fg_color=C["border"]).pack(
            fill="x", padx=40)
        ctk.CTkLabel(card,
                     text="TCC — Engenharia / Ciência da Computação\n"
                          "Orientado por: SafeGuard Systems",
                     font=ctk.CTkFont(size=10),
                     text_color=C["text_muted"],
                     justify="center").pack(pady=16)

        return tab

    # ─── Navegação ───────────────────────────────────────────

    def _switch_tab(self, key: str):
        """Troca a aba visível e destaca o item do menu."""
        for k, f in self._tabs.items():
            f.pack_forget()

        self._tabs[key].pack(fill="both", expand=True)
        self._current_tab = key

        # Highlight no menu
        for k, btn in self._menu_btns.items():
            if k == key:
                btn.configure(fg_color=C["bg_card"],
                              text_color=C["text_primary"])
            else:
                btn.configure(fg_color="transparent",
                              text_color=C["text_muted"])

        if key == "dashboard":
            self._refresh_dashboard()
        elif key == "reports":
            self._refresh_reports_table()

    # ─── Back-end ────────────────────────────────────────────

    def _init_detector(self):
        """Carrega o modelo YOLO em thread separada (não trava UI)."""
        try:
            self.detector = EPIDetector(
                model_path=str(ROOT / self._model_var.get()))
            self.rules_engine = PPERulesEngine(
                require_helmet=self._req_helmet.get(),
                require_vest=self._req_vest.get())
            self._add_audit(f"Modelo {self._model_var.get()} carregado", "OK")
        except Exception as e:
            self._add_audit(f"Falha ao carregar modelo: {e}", "WARN")

    def _toggle_monitoring(self):
        if self._monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self):
        if self.detector is None:
            self._show_toast("⏳ Aguarde — modelo ainda carregando...")
            return

        # Recria engine com configurações atuais
        self.rules_engine = PPERulesEngine(
            require_helmet=self._req_helmet.get(),
            require_vest=self._req_vest.get())

        cam = self._cam_index.get()
        self.video_thread = VideoThread(
            camera_index=cam,
            detector=self.detector,
            rules_engine=self.rules_engine,
            vlogger=self.vlogger,
            telegram=self.telegram,
            result_queue=self.result_queue,
        )
        self.video_thread.start()
        self._monitoring = True
        self._stats["sessions"] += 1

        self._btn_toggle.configure(
            text="⏹  Parar Monitoramento",
            fg_color=C["accent_red"],
            hover_color="#a02020")
        self._placeholder_lbl.place_forget()
        self._add_audit(f"Monitoramento iniciado | câmera {cam}", "INFO")

    def _stop_monitoring(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self._monitoring = False
        self._btn_toggle.configure(
            text="▶  Iniciar Monitoramento",
            fg_color=C["accent_green"],
            hover_color="#1a6e28")
        self._cam_dot.configure(text_color=C["accent_red"])
        self._cam_status_lbl.configure(text="Câmera offline")
        self._fps_lbl.configure(text="FPS: —")
        self._hud_fps.configure(text="—")
        self._placeholder_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._add_audit("Monitoramento encerrado", "INFO")

    # ─── Loop de polling da queue ─────────────────────────────

    def _poll_queue(self):
        """Consome resultados da VideoThread e atualiza a UI (30ms)."""
        try:
            while not self.result_queue.empty():
                data = self.result_queue.get_nowait()

                if data["type"] == "error":
                    self._stop_monitoring()
                    self._show_toast(f"❌ {data['msg']}")
                    self._add_audit(data["msg"], "WARN")

                elif data["type"] == "frame":
                    self._update_from_frame(data)

        except queue.Empty:
            pass
        finally:
            self.after(30, self._poll_queue)

    def _update_from_frame(self, data: dict):
        """Atualiza todos os widgets com dados do frame processado."""
        frame      = data["frame"]
        fps        = data["fps"]
        persons    = data["persons"]
        violators  = data["violators"]
        frame_num  = data["frame_num"]

        # Estatísticas acumuladas
        self._stats["frames"]  = frame_num
        self._stats["persons"] += persons

        # ── Status câmera ──────────────────────────────────────
        self._cam_dot.configure(text_color=C["accent_green"])
        self._cam_status_lbl.configure(text=f"Câmera {self._cam_index.get()} ativa")
        self._fps_lbl.configure(text=f"FPS: {fps}")

        # ── HUD ───────────────────────────────────────────────
        self._hud_fps.configure(text=str(fps))
        self._hud_persons.configure(text=str(persons))
        today_count = self.vlogger.get_today_count()
        self._hud_alerts.configure(
            text=str(today_count),
            text_color=C["accent_red"] if today_count > 0 else C["accent_green"])

        # ── Vídeo ─────────────────────────────────────────────
        if self._current_tab == "monitor":
            lbl = self._video_label
            w   = max(lbl.winfo_width(),  640)
            h   = max(lbl.winfo_height(), 480)
            img = cv2_to_ctk(frame, w, h)
            lbl.configure(image=img)
            lbl._image = img  # evita GC

        # ── KPIs no dashboard ─────────────────────────────────
        self._kpi_alerts.set_value(str(today_count))
        self._kpi_persons.set_value(str(self._stats["persons"]))

        total = load_violations_json()
        conf_pct = 0.0
        if frame_num > 0:
            total_persons_seen = self._stats["persons"]
            total_viols = sum(r.get("violators", 0) for r in total)
            if total_persons_seen > 0:
                conf_pct = max(0.0,
                    100 - (total_viols / total_persons_seen * 100))
        self._kpi_conform.set_value(f"{conf_pct:.0f}%")

    # ─── Diagnóstico de câmeras ───────────────────────────────

    def _diagnose_cameras(self):
        """Testa câmeras de índice 0 a 5 e mostra resultado."""
        self._diag_result.configure(text="🔍 Diagnosticando... aguarde")
        self.update()

        def _run():
            found  = []
            failed = []
            for idx in range(6):
                cap = cv2.VideoCapture(
                    idx + cv2.CAP_DSHOW
                    if sys.platform == "win32" else idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        found.append(f"  ✅ Índice {idx} — {w}x{h} (CAP_DSHOW)")
                    else:
                        found.append(f"  ⚠️  Índice {idx} — abre mas sem frame")
                else:
                    # Tenta sem DSHOW
                    cap2 = cv2.VideoCapture(idx)
                    if cap2.isOpened():
                        ret, frame = cap2.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            found.append(
                                f"  ✅ Índice {idx} — {w}x{h} (backend padrão)")
                        cap2.release()
                    else:
                        failed.append(idx)
                cap.release()

            lines = ["Câmeras detectadas:"] + (found if found else ["  Nenhuma"])
            if failed:
                lines.append(
                    f"\nÍndices sem câmera: {failed}\n"
                    "Dica: Verifique permissões de câmera em\n"
                    "Configurações → Privacidade → Câmera (Windows 11)")
            msg = "\n".join(lines)
            self.after(0, lambda: self._diag_result.configure(text=msg))
            if found:
                first_idx = int(found[0].split("Índice ")[1].split(" ")[0])
                self.after(0, lambda: self._cam_combo.set(str(first_idx)))
                self.after(0, lambda: self._cam_index.set(first_idx))

        threading.Thread(target=_run, daemon=True).start()

    def _pick_video_file(self):
        """Abre diálogo para selecionar arquivo de vídeo."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Selecionar Vídeo de Teste",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv"),
                       ("Todos", "*.*")])
        if path:
            self._cam_index.set(path)  # VideoThread aceita string path tb
            self._cam_combo.set(f"arquivo: {Path(path).name}")
            self._add_audit(f"Vídeo selecionado: {path}", "INFO")

    # ─── Configurações ────────────────────────────────────────

    def _save_settings(self):
        """Aplica configurações sem reiniciar o app."""
        # Recarrega detector se modelo mudou
        new_model = self._model_var.get()
        if new_model != Path(MODEL_PATH).name:
            threading.Thread(target=self._init_detector, daemon=True).start()

        self._add_audit("Configurações salvas", "INFO")
        self._show_toast("✅ Configurações salvas!")

    def _test_telegram(self):
        self._tg_status.configure(
            text="📡 Testando...", text_color=C["text_muted"])
        token   = self._tg_token.get()
        chat_id = self._tg_chat.get()

        def _run():
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            try:
                r = requests.post(url, json={
                    "chat_id": chat_id,
                    "text": "✅ SafeGuard EPI Monitor — conexão validada!"
                }, timeout=8)
                ok = r.status_code == 200
            except Exception:
                ok = False

            msg   = "✅ Telegram conectado!" if ok else "❌ Falha — verifique token e chat_id"
            color = C["accent_green"] if ok else C["accent_red"]
            self.after(0, lambda: self._tg_status.configure(
                text=msg, text_color=color))

        threading.Thread(target=_run, daemon=True).start()

    # ─── Dashboard / Relatórios ───────────────────────────────

    def _refresh_dashboard(self):
        """Atualiza os KPIs e o gráfico do dashboard."""
        records     = load_violations_json()
        today       = date.today().isoformat()
        today_count = sum(1 for r in records
                          if r.get("timestamp", "").startswith(today))
        self._kpi_alerts.set_value(str(today_count))

        # Tentativa de embed do matplotlib
        self._try_embed_chart(records)

        # Últimas violações
        for w in self._violations_scroll.winfo_children():
            w.destroy()

        recent = sorted(records, key=lambda x: x.get("timestamp", ""),
                        reverse=True)[:8]
        if not recent:
            ctk.CTkLabel(self._violations_scroll,
                         text="Nenhuma violação registrada",
                         text_color=C["text_muted"]).pack(pady=12)
            return

        for rec in recent:
            row = ctk.CTkFrame(self._violations_scroll,
                               fg_color=C["bg_input"],
                               corner_radius=8)
            row.pack(fill="x", pady=3, padx=4)

            ts    = rec.get("timestamp", "")[:19].replace("T", " ")
            viols = ", ".join(rec.get("violations", []))
            n     = rec.get("violators", 1)

            ctk.CTkLabel(row, text=f"🕐 {ts}",
                         font=ctk.CTkFont(size=11),
                         text_color=C["text_muted"],
                         width=160, anchor="w").pack(
                             side="left", padx=10, pady=6)
            ctk.CTkLabel(row, text=f"👤 {n}",
                         font=ctk.CTkFont(size=11),
                         text_color=C["text_primary"]).pack(
                             side="left", padx=8)
            ctk.CTkLabel(row, text=f"⛔ {viols}",
                         font=ctk.CTkFont(size=11),
                         text_color=C["accent_red"]).pack(
                             side="left", padx=8)

    def _try_embed_chart(self, records: list):
        """Tenta renderizar gráfico matplotlib no dashboard."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            # Limpa frame anterior
            for w in self._chart_frame.winfo_children():
                w.destroy()

            # Dados: alertas por hora do dia
            hours = [0] * 24
            for r in records:
                ts = r.get("timestamp", "")
                try:
                    h = int(ts[11:13])
                    hours[h] += r.get("violators", 1)
                except Exception:
                    pass

            fig, ax = plt.subplots(figsize=(7, 2.4),
                                   facecolor=C["bg_card"])
            ax.set_facecolor(C["bg_card"])
            bars = ax.bar(range(24), hours,
                          color=[C["accent_red"] if h > 0
                                 else C["bg_input"] for h in hours],
                          edgecolor="none", width=0.7)
            ax.set_xlabel("Hora do dia", color=C["text_muted"],
                          fontsize=9)
            ax.set_ylabel("Violações", color=C["text_muted"],
                          fontsize=9)
            ax.set_title("Violações por Hora",
                         color=C["text_primary"], fontsize=11)
            ax.tick_params(colors=C["text_muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])
            fig.tight_layout(pad=1.2)

            canvas = FigureCanvasTkAgg(fig, master=self._chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True,
                                        padx=8, pady=8)
            plt.close(fig)

        except Exception:
            # matplotlib não instalado ou erro — mostra texto
            self._chart_label.configure(
                text="📊 Instale matplotlib para visualizar o gráfico")

    def _refresh_reports_table(self):
        """Recarrega a tabela da aba Relatórios."""
        for w in self._report_scroll.winfo_children():
            w.destroy()

        records = sorted(load_violations_json(),
                         key=lambda x: x.get("timestamp", ""),
                         reverse=True)
        if not records:
            ctk.CTkLabel(self._report_scroll,
                         text="Nenhum registro encontrado.",
                         text_color=C["text_muted"]).pack(pady=16)
            return

        for rec in records:
            row = ctk.CTkFrame(self._report_scroll,
                               fg_color="transparent")
            row.pack(fill="x")
            sep = ctk.CTkFrame(self._report_scroll,
                               height=1, fg_color=C["border"])
            sep.pack(fill="x")

            ts    = rec.get("timestamp", "")[:19].replace("T", " ")
            n     = str(rec.get("violators", 1))
            viols = ", ".join(rec.get("violations", []))
            fname = rec.get("image_file", "—")

            for val, w in [(ts, 170), (n, 80), (viols, 260), (fname, 220)]:
                ctk.CTkLabel(row, text=val, width=w,
                             font=ctk.CTkFont(size=11),
                             text_color=C["text_primary"],
                             anchor="w").pack(
                                 side="left", padx=10, pady=6)

    def _export_csv(self):
        """Exporta histórico de violações como CSV."""
        from tkinter import filedialog
        records = load_violations_json()
        if not records:
            self._show_toast("⚠️  Sem dados para exportar")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"violacoes_epi_{date.today()}.csv")
        if not path:
            return

        lines = ["Data/Hora,Pessoas,Violações,Arquivo"]
        for r in records:
            ts    = r.get("timestamp", "")[:19].replace("T", " ")
            n     = r.get("violators", 1)
            viols = "|".join(r.get("violations", []))
            fname = r.get("image_file", "")
            lines.append(f"{ts},{n},{viols},{fname}")

        Path(path).write_text("\n".join(lines), encoding="utf-8-sig")
        self._show_toast(f"✅ CSV exportado: {Path(path).name}")
        self._add_audit(f"CSV exportado: {path}", "INFO")

    def _export_report(self):
        """Gera relatório TXT formatado."""
        from tkinter import filedialog
        records = load_violations_json()

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt")],
            initialfile=f"relatorio_epi_{date.today()}.txt")
        if not path:
            return

        today_viols = sum(1 for r in records
                          if r.get("timestamp", "")
                          .startswith(date.today().isoformat()))
        lines = [
            "=" * 60,
            "  RELATÓRIO DE MONITORAMENTO EPI",
            f"  {APP_TITLE}  {APP_VERSION}",
            f"  Empresa: {COMPANY}",
            f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "=" * 60,
            "",
            "RESUMO:",
            f"  Total de violações registradas : {len(records)}",
            f"  Violações hoje                  : {today_viols}",
            f"  Frames processados              : {self._stats['frames']}",
            "",
            "─" * 60,
            "HISTÓRICO DETALHADO:",
            "─" * 60,
        ]
        for r in sorted(records,
                        key=lambda x: x.get("timestamp", ""),
                        reverse=True):
            ts    = r.get("timestamp", "")[:19].replace("T", " ")
            n     = r.get("violators", 1)
            viols = ", ".join(r.get("violations", []))
            fname = r.get("image_file", "")
            lines.append(f"[{ts}]  {n} pessoa(s) — {viols}")
            lines.append(f"  Arquivo: {fname}")

        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self._show_toast(f"✅ Relatório gerado: {Path(path).name}")
        self._add_audit(f"Relatório exportado: {path}", "INFO")

    # ─── Logs de Auditoria ────────────────────────────────────

    def _add_audit(self, msg: str, level: str = "INFO"):
        ts  = datetime.now().strftime("%d/%m %H:%M:%S")
        row = LogRow(self._audit_scroll, ts, msg, level)
        row.pack(fill="x", pady=1)
        self._audit_events.append((ts, level, msg))
        # Scroll para o fundo
        self.after(50, self._audit_scroll._parent_canvas.yview_moveto, 1.0)

    def _clear_audit(self):
        for w in self._audit_scroll.winfo_children():
            w.destroy()
        self._audit_events.clear()

    # ─── Utilitários UI ───────────────────────────────────────

    def _update_clock(self):
        now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
        self._hud_clock.configure(text=now)
        self.after(1000, self._update_clock)

    def _show_toast(self, msg: str, duration_ms: int = 3000):
        """Exibe uma notificação temporária no canto superior direito."""
        toast = ctk.CTkLabel(self, text=f"  {msg}  ",
                             font=ctk.CTkFont(size=12),
                             fg_color=C["bg_card"],
                             text_color=C["text_primary"],
                             corner_radius=8,
                             padx=12, pady=8)
        toast.place(relx=0.99, rely=0.04, anchor="ne")
        self.after(duration_ms, toast.destroy)

    def on_close(self):
        """Encerra a thread de vídeo antes de fechar."""
        self._stop_monitoring()
        self.destroy()


# ─────────────────────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    # Tela de Login
    login = LoginWindow()
    login.mainloop()

    if not login.success:
        return  # Usuário fechou sem logar

    # Janela Principal
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
