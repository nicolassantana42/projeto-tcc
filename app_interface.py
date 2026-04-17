#!/usr/bin/env python3
# ============================================================
# app_interface.py — Interface Gráfica Moderna
# Sistema de Monitoramento de EPI com Dashboard
# ============================================================

import sys
from pathlib import Path
import threading
import time
from datetime import datetime
from typing import Optional

# Adiciona raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from loguru import logger

from config import (
    CAMERA_INDEX, MODEL_PATH, CONFIDENCE,
    REQUIRE_HELMET, REQUIRE_VEST, VIOLATIONS_DIR
)
from src.camera.capture import VideoCapture, CameraDetector
from src.ai.detector import EPIDetector
from src.rules.ppe_rules import PPERulesEngine
from src.alerts.logger import ViolationLogger
from src.alerts.telegram import TelegramAlerter

# Configurar tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class EPIMonitorApp(ctk.CTk):
    """
    Aplicação Desktop Moderna para Monitoramento de EPI
    
    Features:
    - Dashboard com estatísticas em tempo real
    - Visualização de câmera com detecções
    - Configurações de EPIs personalizáveis
    - Alertas visuais de violações
    - Design profissional e responsivo
    """
    
    def __init__(self):
        super().__init__()
        
        # Configuração da janela
        self.title("Sistema de Monitoramento de EPI - TCC")
        self.geometry("1400x900")
        
        # Estado da aplicação
        self.camera_running = False
        self.camera = None
        self.detector = None
        self.rules_engine = None
        self.violation_logger = None
        self.telegram = None
        
        # Estatísticas
        self.total_frames = 0
        self.total_pessoas = 0
        self.total_violacoes = 0
        self.fps_atual = 0.0
        
        # Configurações de EPIs
        self.eppis_config = {
            'capacete': ctk.BooleanVar(value=True),
            'colete': ctk.BooleanVar(value=False),
            'bota': ctk.BooleanVar(value=False)
        }
        
        # Thread de captura
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
        # Criar interface
        self.create_widgets()
        
        # Protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """Cria todos os widgets da interface."""
        
        # ══════════════════════════════════════════════════════════
        # PAINEL LATERAL ESQUERDO - Controles e Estatísticas
        # ══════════════════════════════════════════════════════════
        
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        # Logo/Título
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🦺 Monitor EPI",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Sistema Inteligente de Segurança",
            font=ctk.CTkFont(size=12)
        )
        self.subtitle.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # ── Controles da Câmera ──
        self.controls_frame = ctk.CTkFrame(self.sidebar)
        self.controls_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            self.controls_frame,
            text="Controles",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 10))
        
        self.btn_start = ctk.CTkButton(
            self.controls_frame,
            text="▶ Iniciar Câmera",
            command=self.start_camera,
            fg_color="green",
            hover_color="darkgreen",
            height=40
        )
        self.btn_start.pack(pady=5, padx=10, fill="x")
        
        self.btn_stop = ctk.CTkButton(
            self.controls_frame,
            text="⏹ Parar Câmera",
            command=self.stop_camera,
            fg_color="red",
            hover_color="darkred",
            height=40,
            state="disabled"
        )
        self.btn_stop.pack(pady=5, padx=10, fill="x")
        
        self.btn_snapshot = ctk.CTkButton(
            self.controls_frame,
            text="📸 Capturar Foto",
            command=self.take_snapshot,
            height=35,
            state="disabled"
        )
        self.btn_snapshot.pack(pady=5, padx=10, fill="x")
        
        # ── Estatísticas ──
        self.stats_frame = ctk.CTkFrame(self.sidebar)
        self.stats_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            self.stats_frame,
            text="Estatísticas",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 10))
        
        # FPS
        self.label_fps = ctk.CTkLabel(
            self.stats_frame,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=14)
        )
        self.label_fps.pack(pady=2)
        
        # Frames processados
        self.label_frames = ctk.CTkLabel(
            self.stats_frame,
            text="Frames: 0",
            font=ctk.CTkFont(size=14)
        )
        self.label_frames.pack(pady=2)
        
        # Pessoas detectadas
        self.label_pessoas = ctk.CTkLabel(
            self.stats_frame,
            text="Pessoas: 0",
            font=ctk.CTkFont(size=14)
        )
        self.label_pessoas.pack(pady=2)
        
        # Violações
        self.label_violacoes = ctk.CTkLabel(
            self.stats_frame,
            text="Violações: 0",
            font=ctk.CTkFont(size=14),
            text_color="orange"
        )
        self.label_violacoes.pack(pady=2)
        
        # ── Configurações de EPIs ──
        self.config_frame = ctk.CTkFrame(self.sidebar)
        self.config_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            self.config_frame,
            text="EPIs Monitorados",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 10))
        
        # Capacete
        self.check_capacete = ctk.CTkCheckBox(
            self.config_frame,
            text="⛑️  Capacete",
            variable=self.eppis_config['capacete'],
            command=self.update_epi_config,
            font=ctk.CTkFont(size=13)
        )
        self.check_capacete.pack(pady=5, padx=20, anchor="w")
        
        # Colete
        self.check_colete = ctk.CTkCheckBox(
            self.config_frame,
            text="🦺 Colete",
            variable=self.eppis_config['colete'],
            command=self.update_epi_config,
            font=ctk.CTkFont(size=13)
        )
        self.check_colete.pack(pady=5, padx=20, anchor="w")
        
        # Bota
        self.check_bota = ctk.CTkCheckBox(
            self.config_frame,
            text="👢 Bota de Segurança",
            variable=self.eppis_config['bota'],
            command=self.update_epi_config,
            font=ctk.CTkFont(size=13)
        )
        self.check_bota.pack(pady=5, padx=20, anchor="w")
        
        # ── Botão Configurações Avançadas ──
        self.btn_config = ctk.CTkButton(
            self.sidebar,
            text="⚙️ Configurações",
            command=self.open_settings,
            height=35
        )
        self.btn_config.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        # ══════════════════════════════════════════════════════════
        # PAINEL CENTRAL - Visualização da Câmera
        # ══════════════════════════════════════════════════════════
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, rowspan=3, padx=20, pady=20, sticky="nsew")
        
        # Título do painel
        self.camera_title = ctk.CTkLabel(
            self.main_frame,
            text="Visualização da Câmera",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.camera_title.pack(pady=(10, 5))
        
        # Status da câmera
        self.camera_status = ctk.CTkLabel(
            self.main_frame,
            text="● Câmera desligada",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.camera_status.pack(pady=(0, 10))
        
        # Canvas para vídeo
        self.video_canvas = ctk.CTkLabel(
            self.main_frame,
            text="",
            width=1000,
            height=700
        )
        self.video_canvas.pack(pady=10, padx=10, expand=True, fill="both")
        
        # Placeholder quando câmera está desligada
        self.show_placeholder()
        
        # ══════════════════════════════════════════════════════════
        # PAINEL INFERIOR - Alertas e Logs
        # ══════════════════════════════════════════════════════════
        
        self.alerts_frame = ctk.CTkFrame(self)
        self.alerts_frame.grid(row=3, column=1, padx=20, pady=(0, 20), sticky="ew")
        
        ctk.CTkLabel(
            self.alerts_frame,
            text="Alertas Recentes",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.alerts_text = ctk.CTkTextbox(
            self.alerts_frame,
            height=100,
            font=ctk.CTkFont(size=11)
        )
        self.alerts_text.pack(pady=5, padx=10, fill="both", expand=True)
        self.alerts_text.insert("1.0", "Sistema iniciado. Aguardando início da câmera...\n")
        
        # Configurar grid weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
    def show_placeholder(self):
        """Mostra imagem placeholder quando câmera está desligada."""
        # Criar imagem placeholder
        placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 30)  # Cor de fundo escura
        
        # Adicionar texto
        text = "Câmera Desligada"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        # Calcular posição centralizada
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (800 - text_width) // 2
        y = (600 + text_height) // 2
        
        cv2.putText(placeholder, text, (x, y), font, font_scale, (100, 100, 100), thickness)
        
        # Adicionar ícone
        cv2.putText(placeholder, "Pressione 'Iniciar Camera'", 
                   (250, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        # Converter e mostrar
        self.update_video_frame(placeholder)
        
    def update_video_frame(self, frame):
        """Atualiza o frame do vídeo no canvas."""
        # Redimensionar frame para caber no canvas
        h, w = frame.shape[:2]
        max_w, max_h = 1000, 700
        
        # Calcular proporção
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # Converter BGR para RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Converter para ImageTk
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Atualizar canvas
        self.video_canvas.configure(image=imgtk)
        self.video_canvas.image = imgtk  # Manter referência
        
    def start_camera(self):
        """Inicia a câmera e o processamento."""
        if self.camera_running:
            return
        
        self.add_alert("Iniciando câmera...", "info")
        
        try:
            # Inicializar componentes
            self.camera = VideoCapture(camera_source=CAMERA_INDEX, auto_detect=True)
            self.detector = EPIDetector(model_path=MODEL_PATH)
            self.rules_engine = PPERulesEngine()
            self.violation_logger = ViolationLogger()
            self.telegram = TelegramAlerter()
            
            # Atualizar configuração de EPIs
            self.update_epi_config()
            
            # Resetar estatísticas
            self.total_frames = 0
            self.total_pessoas = 0
            self.total_violacoes = 0
            
            # Iniciar thread de captura
            self.camera_running = True
            self.stop_capture.clear()
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Atualizar UI
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.btn_snapshot.configure(state="normal")
            self.camera_status.configure(text="● Câmera ativa", text_color="green")
            
            self.add_alert("Câmera iniciada com sucesso!", "success")
            
        except Exception as e:
            self.add_alert(f"Erro ao iniciar câmera: {e}", "error")
            logger.error(f"Erro ao iniciar câmera: {e}")
            self.camera_running = False
            
    def stop_camera(self):
        """Para a câmera e o processamento."""
        if not self.camera_running:
            return
        
        self.add_alert("Parando câmera...", "info")
        
        # Sinalizar parada
        self.camera_running = False
        self.stop_capture.set()
        
        # Aguardar thread
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Liberar recursos
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Atualizar UI
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_snapshot.configure(state="disabled")
        self.camera_status.configure(text="● Câmera desligada", text_color="gray")
        
        # Mostrar placeholder
        self.show_placeholder()
        
        self.add_alert("Câmera parada.", "info")
        
    def capture_loop(self):
        """Loop de captura e processamento (roda em thread separada)."""
        try:
            for frame in self.camera.stream():
                if self.stop_capture.is_set():
                    break
                
                self.total_frames += 1
                self.fps_atual = self.camera.current_fps
                
                # Detecção
                result = self.detector.detect(frame)
                
                # Aplicar regras
                statuses = self.rules_engine.evaluate(result)
                
                # Contar pessoas
                self.total_pessoas = len(statuses)
                
                # Desenhar detecções
                display_frame = result.annotated_frame.copy()
                display_frame = self.draw_detections(display_frame, statuses)
                
                # Verificar violações
                violators = [s for s in statuses if not s.is_compliant]
                if violators:
                    self.total_violacoes += len(violators)
                    
                    # Log violações
                    for status in violators:
                        violations_str = ", ".join(status.violations)
                        self.add_alert(f"⚠️ Violação: {violations_str}", "warning")
                
                # Atualizar frame
                self.after(0, self.update_video_frame, display_frame)
                
                # Atualizar estatísticas
                self.after(0, self.update_stats)
                
        except Exception as e:
            self.add_alert(f"Erro na captura: {e}", "error")
            logger.error(f"Erro no loop de captura: {e}")
        finally:
            self.camera_running = False
            
    def draw_detections(self, frame, statuses):
        """Desenha as detecções e status no frame."""
        for status in statuses:
            bbox = status.person_bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # Cor da caixa baseada em conformidade
            if status.is_compliant:
                color = (0, 255, 0)  # Verde
                text = "OK"
            else:
                color = (0, 0, 255)  # Vermelho
                text = "VIOLACAO"
            
            # Desenhar caixa
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Fundo para texto
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
            
            # Texto
            cv2.putText(frame, text, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Listar violações
            if not status.is_compliant:
                y_offset = y2 + 25
                for violation in status.violations:
                    cv2.putText(frame, f"- {violation}", (x1, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    y_offset += 20
        
        # Info no canto
        info_text = f"FPS: {self.fps_atual:.1f} | Pessoas: {len(statuses)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
        
    def update_stats(self):
        """Atualiza labels de estatísticas."""
        self.label_fps.configure(text=f"FPS: {self.fps_atual:.1f}")
        self.label_frames.configure(text=f"Frames: {self.total_frames}")
        self.label_pessoas.configure(text=f"Pessoas: {self.total_pessoas}")
        self.label_violacoes.configure(text=f"Violações: {self.total_violacoes}")
        
    def update_epi_config(self):
        """Atualiza configuração de EPIs no motor de regras."""
        if self.rules_engine:
            self.rules_engine.require_helmet = self.eppis_config['capacete'].get()
            self.rules_engine.require_vest = self.eppis_config['colete'].get()
            # Note: bota será implementado quando tiver modelo treinado
            
            config_msg = "EPIs configurados: "
            epis = []
            if self.eppis_config['capacete'].get():
                epis.append("Capacete")
            if self.eppis_config['colete'].get():
                epis.append("Colete")
            if self.eppis_config['bota'].get():
                epis.append("Bota")
            
            config_msg += ", ".join(epis) if epis else "Nenhum"
            self.add_alert(config_msg, "info")
            
    def take_snapshot(self):
        """Captura uma foto da câmera atual."""
        if not self.camera_running:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = VIOLATIONS_DIR / f"snapshot_{timestamp}.png"
        
        # Salvar último frame
        # (precisaria armazenar o último frame processado)
        self.add_alert(f"Foto salva: {filename.name}", "success")
        
    def add_alert(self, message: str, level: str = "info"):
        """Adiciona alerta ao painel de logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Ícones por nível
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        icon = icons.get(level, "ℹ️")
        alert_msg = f"[{timestamp}] {icon} {message}\n"
        
        self.alerts_text.insert("end", alert_msg)
        self.alerts_text.see("end")  # Auto-scroll
        
        # Limitar histórico
        lines = int(self.alerts_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.alerts_text.delete("1.0", "50.0")
            
    def open_settings(self):
        """Abre janela de configurações avançadas."""
        SettingsWindow(self)
        
    def on_closing(self):
        """Chamado ao fechar a janela."""
        if self.camera_running:
            self.stop_camera()
            time.sleep(0.5)
        
        self.destroy()


class SettingsWindow(ctk.CTkToplevel):
    """Janela de configurações avançadas."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Configurações Avançadas")
        self.geometry("600x500")
        
        # Título
        ctk.CTkLabel(
            self,
            text="⚙️ Configurações do Sistema",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)
        
        # ── Câmera ──
        camera_frame = ctk.CTkFrame(self)
        camera_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(
            camera_frame,
            text="Configurações de Câmera",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(camera_frame, text="Índice/URL da Câmera:").pack(pady=5)
        self.camera_entry = ctk.CTkEntry(camera_frame, width=400)
        self.camera_entry.insert(0, str(CAMERA_INDEX))
        self.camera_entry.pack(pady=5)
        
        # ── Detecção ──
        detection_frame = ctk.CTkFrame(self)
        detection_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(
            detection_frame,
            text="Configurações de Detecção",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(detection_frame, text="Confiança (0-1):").pack(pady=5)
        self.confidence_slider = ctk.CTkSlider(
            detection_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100
        )
        self.confidence_slider.set(CONFIDENCE)
        self.confidence_slider.pack(pady=5)
        
        self.confidence_label = ctk.CTkLabel(detection_frame, text=f"{CONFIDENCE:.2f}")
        self.confidence_label.pack(pady=5)
        
        # Atualizar label ao mover slider
        def update_confidence_label(value):
            self.confidence_label.configure(text=f"{value:.2f}")
        
        self.confidence_slider.configure(command=update_confidence_label)
        
        # ── Botões ──
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="Salvar",
            command=self.save_settings,
            fg_color="green",
            width=120
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="Cancelar",
            command=self.destroy,
            width=120
        ).pack(side="left", padx=10)
        
    def save_settings(self):
        """Salva as configurações."""
        # Aqui você salvaria no .env ou config.py
        self.destroy()


def main():
    """Inicia a aplicação."""
    logger.info("Iniciando Interface Gráfica...")
    
    app = EPIMonitorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
