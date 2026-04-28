# src/ai/detector.py
from ultralytics import YOLO
import cv2
import time
import config

class PPEDetector:
    def __init__(self):
        print(f"[AI] Carregando modelo YOLO de: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)

    def detect(self, frame):
        """Processa o frame e retorna as detecções formatadas."""
        # imgsz=320 melhora muito a performance no Raspberry Pi
        results = self.model.predict(
            frame, 
            conf=config.CONFIDENCE_THRESHOLD, 
            iou=config.IOU_THRESHOLD, 
            imgsz=320, 
            verbose=False
        )
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2), 
                    'confidence': conf, 
                    'class_id': cls
                })
        return detections

    def draw_evidence(self, frame, violating_persons, cam_name, cam_ip, sector):
        """
        Parte 2 e 3: Desenha as bounding boxes vermelhas APENAS em quem está 
        violando a regra e adiciona dados de auditoria na imagem.
        """
        annotated_frame = frame.copy()
        
        # 1. Incluir Dados da Ocorrência na Imagem (Evidência)
        timestamp = time.strftime('%d/%m/%Y %H:%M:%S')
        
        # Fundo semi-transparente para o texto ficar legível
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        cv2.putText(annotated_frame, f"CAMERA: {cam_name} (IP: {cam_ip})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"SETOR: {sector}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"DATA/HORA: {timestamp}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. Destacar APENAS os infratores em VERMELHO
        for person in violating_persons:
            x1, y1, x2, y2 = person['bbox']
            # Bounding box vermelha
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Label de Violação
            cv2.putText(annotated_frame, "VIOLACAO EPI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return annotated_frame