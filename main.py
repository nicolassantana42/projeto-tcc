#!/usr/bin/env python3
# ============================================================
#  main.py — Ponto de Entrada Principal
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  Uso:
#    python main.py
#
#  Teclas durante execução:
#    Q / ESC → Encerra o sistema
#    B       → Executa benchmark de performance
#    S       → Salva frame atual manualmente
#    D       → Ativa/desativa modo debug (região da cabeça)
# ============================================================

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger

# Garante que a raiz do projeto (onde está config.py) está no sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    MODEL_PATH, CONFIDENCE, SHOW_VIDEO, SAVE_FRAMES,
    VIOLATIONS_DIR, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DEMO_MODE
)
from src.camera.capture    import VideoCapture
from src.ai.detector       import EPIDetector
from src.rules.ppe_rules   import PPERulesEngine, draw_ppe_status
from src.alerts.logger     import ViolationLogger
from src.alerts.telegram   import TelegramAlerter


# ── Configuração de logs no console ──────────────────────────
logger.remove()  # Remove handler padrão
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║     SISTEMA DE MONITORAMENTO DE EPI                      ║
║     Visão Computacional e Inteligência Artificial        ║
║     TCC — Engenharia / Ciência da Computação            ║
╠══════════════════════════════════════════════════════════╣
║  Teclas:                                                 ║
║    Q / ESC → Encerrar     B → Benchmark                 ║
║    S       → Salvar frame D → Debug (região da cabeça)  ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def draw_hud(frame, fps: float, frame_count: int,
             violations_today: int, demo_mode: bool) -> None:
    """Desenha o HUD (Heads-Up Display) de informações no frame."""
    h, w = frame.shape[:2]

    # Fundo semi-transparente no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    fps_color = (0, 200, 80) if fps >= 20 else (0, 160, 255) if fps >= 10 else (0, 50, 220)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

    # Contador de frames
    cv2.putText(frame, f"Frame: {frame_count}", (130, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Alertas hoje
    alert_color = (0, 50, 220) if violations_today > 0 else (0, 200, 80)
    cv2.putText(frame, f"Alertas hoje: {violations_today}", (280, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 1, cv2.LINE_AA)

    # Modo demo
    if demo_mode:
        cv2.putText(frame, "MODO DEMO (COCO)", (w - 220, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # Timestamp
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


def run_benchmark(detector: EPIDetector, camera: VideoCapture):
    """Executa benchmark e exibe resultados."""
    logger.info("🔬 Iniciando benchmark (50 runs)...")
    ret, frame = camera.read_frame()
    if not ret:
        logger.error("Não foi possível capturar frame para benchmark.")
        return

    results = detector.benchmark(frame, n_runs=50)
    print("\n" + "─" * 50)
    print("  RESULTADOS DE BENCHMARK")
    print("─" * 50)
    print(f"  Modelo  : {results['model']}")
    print(f"  Runs    : {results['n_runs']}")
    print(f"  Média   : {results['avg_ms']} ms")
    print(f"  Mínimo  : {results['min_ms']} ms")
    print(f"  Máximo  : {results['max_ms']} ms")
    print(f"  Desvio  : {results['std_ms']} ms")
    print(f"  FPS eq. : {results['fps_equiv']}")
    print("─" * 50 + "\n")


def main():
    print_banner()

    # ── Inicialização ─────────────────────────────────────────
    logger.info("🚀 Inicializando componentes...")

    try:
        camera   = VideoCapture()
        detector = EPIDetector(model_path=MODEL_PATH)
        rules    = PPERulesEngine()
        vlogger  = ViolationLogger()
        telegram = TelegramAlerter()
    except Exception as e:
        logger.error(f"Falha na inicialização: {e}")
        sys.exit(1)

    # Teste de conexão Telegram (opcional)
    if telegram.enabled:
        telegram.test_connection()

    # ── Loop principal ────────────────────────────────────────
    logger.info("▶  Iniciando captura de vídeo...")
    logger.info("   Pressione Q ou ESC para encerrar\n")

    frame_count    = 0
    debug_mode     = False
    violations_today = vlogger.get_today_count()

    try:
        for frame in camera.stream():
            frame_count += 1
            fps = camera.current_fps

            # ── Detecção ──────────────────────────────────────
            frame_result = detector.detect(frame)

            # ── Regras EPI ────────────────────────────────────
            statuses = rules.evaluate(frame_result)

            # ── Anotações de status no frame ──────────────────
            display_frame = frame_result.annotated_frame.copy()
            display_frame = draw_ppe_status(display_frame, statuses)

            # ── HUD ───────────────────────────────────────────
            draw_hud(display_frame, fps, frame_count, violations_today, DEMO_MODE)

            # ── Violações ─────────────────────────────────────
            violators = [s for s in statuses if not s.is_compliant]
            if violators:
                all_viols = []
                for s in violators:
                    all_viols.extend(s.violations)

                if SAVE_FRAMES:
                    saved_path = vlogger.log_violation(display_frame, statuses)
                else:
                    saved_path = None

                telegram.send_violation_alert(
                    violations=list(set(all_viols)),
                    image_path=saved_path,
                    person_count=len(violators)
                )
                violations_today = vlogger.get_today_count()

            # ── Exibição ──────────────────────────────────────
            if SHOW_VIDEO:
                cv2.imshow("Sistema EPI — TCC | Q=sair", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), ord('Q'), 27):      # Q ou ESC
                    logger.info("Encerramento solicitado pelo usuário.")
                    break
                elif key in (ord('b'), ord('B')):         # Benchmark
                    run_benchmark(detector, camera)
                elif key in (ord('s'), ord('S')):         # Save manual
                    ts  = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                    out = VIOLATIONS_DIR / f"manual_{ts}.png"
                    cv2.imwrite(str(out), display_frame)
                    logger.info(f"📸 Frame salvo: {out.name}")
                elif key in (ord('d'), ord('D')):         # Debug toggle
                    debug_mode = not debug_mode
                    logger.info(f"Debug: {'ON' if debug_mode else 'OFF'}")

    except KeyboardInterrupt:
        logger.info("\nInterrompido pelo usuário (Ctrl+C)")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("✅ Sistema encerrado.")
        logger.info(f"   Total de frames processados : {frame_count}")
        logger.info(f"   Violações registradas hoje  : {violations_today}")


if __name__ == "__main__":
    main()
