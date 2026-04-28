# config.py
import os

# ==========================================
# CONFIGURAÇÕES DO MODELO YOLO
# ==========================================
MODEL_PATH = "yolov8n.pt" 
CONFIDENCE_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.45

CLASS_PERSON = 0
CLASS_HELMET = 1
CLASS_BOOT = 2

# ==========================================
# REGRAS DE NEGÓCIO E EPIs (NOVO)
# ==========================================
# Escolha quais EPIs são obrigatórios para considerar a pessoa "Segura"
REQUIRE_HELMET = True
REQUIRE_BOOT = False

# Tempo de bloqueio de emails por câmera (5 minutos = 300 segundos)
ALERT_COOLDOWN_SECONDS = 300 
FRAME_SKIP_HEADLESS = 5 

# ==========================================
# CONFIGURAÇÕES DE EMAIL
# ==========================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "seu_email_remetente@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "sua_senha_de_app_aqui")
EMAIL_RECEIVER = "email_do_supervisor@industria.com"