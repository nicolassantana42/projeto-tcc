import os
from pathlib import Path
from dotenv import load_dotenv

# ── Patch PyTorch 2.6 ────────────────────────────────────────
def _patch_torch_safe_globals():
    try:
        import torch
        # Força weights_only=False globalmente para carregar modelos customizados .pt
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load
    except Exception:
        pass

_patch_torch_safe_globals()

# ─── Diretórios ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
VIOLATIONS_DIR = BASE_DIR / "violations"
LOGS_DIR       = BASE_DIR / "logs"

VIOLATIONS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

# ─── Câmera (Fábrica / Switch PoE) ──────────────────────────
# Suporta lista de câmeras IP se necessário futuramente
camera_index_raw = os.getenv("CAMERA_INDEX", "0")

if camera_index_raw.startswith(('rtsp://', 'http://', 'https://')):
    CAMERA_INDEX = camera_index_raw
else:
    try:
        CAMERA_INDEX = int(camera_index_raw)
    except ValueError:
        CAMERA_INDEX = 0

# No Raspberry Pi 5, imgsz=640 é o padrão, 320 para mais performance
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH",  640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))

# ─── Modelo YOLOv8 ───────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE  = float(os.getenv("CONFIDENCE", 0.50)) # Subimos para evitar spam
IOU_THRESH  = float(os.getenv("IOU_THRESH", 0.45))

# ─── IDs de Classe (Ajustados para o seu modelo de EPI) ──────
# IMPORTANTE: Verifique se estes IDs batem com o seu arquivo .yaml de treino
PERSON_CLASS_ID = int(os.getenv("PERSON_ID", 0))
HELMET_CLASS_ID = int(os.getenv("HELMET_ID", 1))
VEST_CLASS_ID   = int(os.getenv("VEST_ID",   2))
BOOT_CLASS_ID   = int(os.getenv("BOOT_ID",   3)) # Novo: Bota de Segurança

# ─── Regras de Engenharia de Prompt ──────────────────────────
REQUIRE_HELMET = os.getenv("REQUIRE_HELMET", "true").lower() == "true"
REQUIRE_VEST   = os.getenv("REQUIRE_VEST",   "true").lower() == "true"
REQUIRE_BOOT   = os.getenv("REQUIRE_BOOT",   "true").lower() == "true"

# ─── Alertas (E-mail e Telegram) ─────────────────────────────
# Configuração SMTP (Estilo Grafana/Monitoramento)
SMTP_SERVER   = os.getenv("SMTP_SERVER",   "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
EMAIL_SENDER  = os.getenv("EMAIL_USER",    "seu_email@gmail.com")
EMAIL_PASS    = os.getenv("EMAIL_PASS",    "sua_senha_app")
EMAIL_RECEIVER = os.getenv("EMAIL_DEST",   "supervisor@fabrica.com.br")

# Telegram (Opcional)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Silenciamento de alertas (Evita 1000 e-mails pela mesma pessoa)
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN", 300)) # 5 Minutos

# ─── Exibição e Operação ─────────────────────────────────────
# No Raspberry Pi na fábrica, deixe SHOW_VIDEO = False
SHOW_VIDEO  = os.getenv("SHOW_VIDEO", "true").lower() == "true"
SAVE_FRAMES = True # Essencial para anexar no e-mail