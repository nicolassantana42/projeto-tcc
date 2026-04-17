# ============================================================
#  config.py — Configurações centrais do Sistema EPI
#  TCC: Monitoramento de EPI com Visão Computacional
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Patch PyTorch 2.6 (weights_only=True por padrão) ─────────
# Deve rodar ANTES de qualquer import do ultralytics/YOLO
def _patch_torch_safe_globals():
    try:
        import torch
        # Tenta registrar todas as classes do ultralytics como seguras
        try:
            import ultralytics.nn.tasks   as _t
            import ultralytics.nn.modules as _m
            _safe = [
                getattr(_t, n) for n in dir(_t) if isinstance(getattr(_t, n), type)
            ] + [
                getattr(_m, n) for n in dir(_m) if isinstance(getattr(_m, n), type)
            ]
            torch.serialization.add_safe_globals(_safe)
        except Exception:
            pass
        # Fallback: força weights_only=False globalmente via monkey-patch
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load
    except Exception:
        pass

_patch_torch_safe_globals()
# ─────────────────────────────────────────────────────────────

# ─── Diretórios ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
VIOLATIONS_DIR = BASE_DIR / "violations"
LOGS_DIR       = BASE_DIR / "logs"

VIOLATIONS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Carrega .env (deve estar na raiz do projeto)
load_dotenv(BASE_DIR / ".env")

# ─── Câmera ──────────────────────────────────────────────────
# Suporta tanto índice numérico (0, 1, 2) quanto URLs RTSP/HTTP
camera_index_raw = os.getenv("CAMERA_INDEX", "0")

# Detecta se é URL ou índice numérico
if camera_index_raw.startswith(('rtsp://', 'http://', 'https://')):
    # É uma URL de câmera IP
    CAMERA_INDEX = camera_index_raw
else:
    # É um índice numérico
    try:
        CAMERA_INDEX = int(camera_index_raw)
    except ValueError:
        logger.warning(f"CAMERA_INDEX inválido: {camera_index_raw}, usando 0")
        CAMERA_INDEX = 0

FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))

# ─── Modelo YOLOv8 ───────────────────────────────────────────
#
#  MODO DE OPERAÇÃO:
#  ─────────────────
#  1. DEMO (padrão): usa yolov8n.pt (COCO) → detecta "person" apenas
#     → Ideal para demonstrar a arquitetura. Todas as pessoas
#       aparecerão como "SEM CAPACETE" (esperado, pois o modelo
#       COCO não conhece capacetes/coletes).
#
#  2. PPE REAL: usa modelo customizado treinado em dataset de EPI
#     → Baixe em: https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
#     → Altere MODEL_PATH no .env para o caminho do modelo baixado
#
MODEL_PATH  = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE  = float(os.getenv("CONFIDENCE", 0.45))
IOU_THRESH  = float(os.getenv("IOU_THRESH", 0.45))

# ─── IDs de Classe ───────────────────────────────────────────
#  Modelo COCO padrão (yolov8n/s/m.pt):
PERSON_CLASS_ID = 0       # "person" — sempre presente no COCO

#  Modelo PPE customizado (após download/fine-tuning):
#  Ajuste os IDs conforme o modelo escolhido.
HELMET_CLASS_ID = 80      # placeholder — altere após baixar modelo PPE
VEST_CLASS_ID   = 81      # placeholder — altere após baixar modelo PPE

#  Detecção de modo automático (True = modelo COCO sem helmet/vest)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# ─── Regras de EPI ───────────────────────────────────────────
HEAD_REGION_RATIO    = float(os.getenv("HEAD_REGION_RATIO", 0.30))
HELMET_IOU_THRESHOLD = float(os.getenv("HELMET_IOU_THRESHOLD", 0.15))

REQUIRE_HELMET = os.getenv("REQUIRE_HELMET", "true").lower() == "true"
REQUIRE_VEST   = os.getenv("REQUIRE_VEST",   "false").lower() == "true"

# ─── Alertas ─────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN", 30))

# ─── Dashboard ───────────────────────────────────────────────
DASHBOARD_PORT    = int(os.getenv("DASHBOARD_PORT", 8501))
MAX_VIOLATIONS_UI = 10

# ─── Exibição (OpenCV) ───────────────────────────────────────
SHOW_VIDEO  = os.getenv("SHOW_VIDEO", "true").lower() == "true"
DRAW_BOXES  = True
DRAW_LABELS = True
SAVE_FRAMES = True
