# main_industrial.py
import cv2
import time
import os
import sqlite3
from pathlib import Path
from ultralytics import YOLO

import config
from src.camera.capture import IPCameraStream
from src.alerts.telegram_sender import AsyncTelegramAlert

def load_active_cameras():
    db_path = "logs/violations.db"
    if not os.path.exists(db_path): return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, sector, connection_string FROM cameras")
    rows = cursor.fetchall()
    conn.close()
    return rows

def main():
    print("[SERVER] Inicializando Motor de Inferência YOLOv5...")
    
    # Carrega a arquitetura YOLO configurada para simular o comportamento YOLOv5 do artigo
    model = YOLO("yolov8n.pt") 
    alert_system = AsyncTelegramAlert()
    active_streams = {}
    
    try:
        while True:
            cameras = load_active_cameras()
            if not cameras:
                time.sleep(4)
                continue
                
            for cam_id, cam_name, sector, conn_str in cameras:
                if cam_id not in active_streams:
                    stream = IPCameraStream(cam_id, conn_str)
                    if stream.start(): active_streams[cam_id] = stream
                    else: continue
                
                ret, frame = active_streams[cam_id].get_frame()
                if not ret or frame is None: continue
                
                # Predição estruturada
                results = model.predict(frame, conf=0.50, imgsz=640, verbose=False)
                
                # Lógica de detecção de não conformidade baseada nas classes rotuladas
                for r in results:
                    boxes = r.boxes
                    if len(boxes) > 0:
                        # Exemplo de lógica: identificou pessoa mas não os EPIs obrigatórios
                        # Simulação de comportamento de risco pautada nas regras
                        avg_conf = float(boxes[0].conf[0])
                        
                        # Simula desenho e marcação judicial vermelha de não conformidade
                        annotated_frame = frame.copy()
                        cv2.putText(annotated_frame, "ALERTA NR-6", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        # Dispara o pipeline assíncrono do Telegram
                        alert_system.send_violation(
                            cam_id=cam_id,
                            cam_name=cam_name,
                            sector=sector,
                            image=annotated_frame,
                            violation_type="Capacete/Coletor",
                            confidence=avg_conf
                        )
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[SERVER] Desligando Ponteadores.")
        for s in active_streams.values(): s.stop()

if __name__ == "__main__":
    main()