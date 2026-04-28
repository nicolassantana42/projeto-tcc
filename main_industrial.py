# main_industrial.py
import cv2
import time
import config
from src.ai.detector import PPEDetector
from src.rules.ppe_rules import PPERulesEngine
from src.alerts.email_sender import AsyncEmailSender

def run_headless():
    print("[SISTEMA] Iniciando Monitoramento Industrial (Modo Headless/Raspberry Pi)...")
    
    # 1. Carrega Módulos
    detector = PPEDetector()
    rules = PPERulesEngine()
    email_sender = AsyncEmailSender()

    # 2. Configuração desta Câmera
    CAM_ID = "CAM_001"
    CAM_NAME = "Linha_Montagem_Principal"
    CAM_IP = "192.168.0.50" # Ou a URL RTSP
    SECTOR = "Setor A"

    # Inicia Câmera (0 para USB, ou URL RTSP para câmera IP)
    cap = cv2.VideoCapture(0) 
    
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Perda de sinal da câmera. Tentando reconectar...")
            time.sleep(5)
            cap = cv2.VideoCapture(0)
            continue

        frame_count += 1
        
        # Otimização Crítica (Parte 7): Frame Skipping para salvar CPU do RPi
        if frame_count % config.FRAME_SKIP_HEADLESS != 0:
            continue

        # Processamento Principal
        detections = detector.detect(frame)
        violating_persons = rules.check_violation(detections)

        # Se encontrou infratores
        if violating_persons:
            # Verifica se já passou 10 minutos desde o último alerta (Parte 5)
            if rules.can_send_alert(CAM_ID):
                print(f"[ALERTA] Violação! Iniciando processo de evidência.")
                
                # Gera Imagem com a box vermelha e as infos
                evidence_img = detector.draw_evidence(frame, violating_persons, CAM_NAME, CAM_IP, SECTOR)
                
                # Envia email em background
                email_sender.send_violation_alert(evidence_img, CAM_NAME, SECTOR)
            else:
                # Ocorre durante os 10 minutos de bloqueio. Não faz nada para não floodar.
                pass 

    cap.release()

if __name__ == "__main__":
    try:
        run_headless()
    except KeyboardInterrupt:
        print("\n[SISTEMA] Monitoramento encerrado pelo usuário.")