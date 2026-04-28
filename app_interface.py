# app_interface.py
import cv2
import config
from src.ai.detector import PPEDetector
from src.rules.ppe_rules import PPERulesEngine
from src.alerts.email_sender import AsyncEmailSender

def run_gui():
    print("[SISTEMA] Iniciando Interface Gráfica (Sala de Controle)...")
    
    detector = PPEDetector()
    rules = PPERulesEngine()
    email_sender = AsyncEmailSender()

    CAM_ID = "CAM_GUI_01"
    CAM_NAME = "Teste_Local"
    CAM_IP = "Localhost"
    SECTOR = "Laboratorio"

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Interface processa todos os frames para o vídeo não ficar travado
        detections = detector.detect(frame)
        violating_persons = rules.check_violation(detections)

        status_text = "STATUS: SEGURO (EPI OK)"
        status_color = (0, 255, 0) # Verde

        if violating_persons:
            status_text = "STATUS: VIOLACAO DETECTADA"
            status_color = (0, 0, 255) # Vermelho
            
            # Gera evidência visual na tela (Bounding boxes vermelhas)
            frame = detector.draw_evidence(frame, violating_persons, CAM_NAME, CAM_IP, SECTOR)
            
            # Envio de Alerta também funciona na GUI, respeitando 10 min
            if rules.can_send_alert(CAM_ID):
                print("[GUI] Violação flagrada! Disparando email em background.")
                email_sender.send_violation_alert(frame, CAM_NAME, SECTOR)

        # Feedback visual para pessoas regulares (Opcional: Bounding box verde para quem está OK)
        persons = [d for d in detections if d['class_id'] == config.CLASS_PERSON]
        for p in persons:
            if p not in violating_persons:
                x1, y1, x2, y2 = p['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "EPI OK", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Banner Superior
        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Monitor de Segurança - PPE Detection", frame)

        # Aperte 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gui()