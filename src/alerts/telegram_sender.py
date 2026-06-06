# src/alerts/telegram_sender.py
import requests
import threading
import cv2
import time
import os
import sqlite3
import config

class AsyncTelegramAlert:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_alerts = {}
        self.db_path = "logs/violations.db"
        
        # Puxar credenciais direto do config/.env corporativo
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "SEU_TOKEN_TELEGRAM_AQUI")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI")

    def send_violation(self, cam_id, cam_name, sector, image, violation_type, confidence):
        current_time = time.time()
        last_alert_time = self.last_alerts.get(cam_id, 0)

        # Janela de Cooldown de 5 minutos configurada no config para evitar flood
        if (current_time - last_alert_time) >= config.ALERT_COOLDOWN_SECONDS:
            self.last_alerts[cam_id] = current_time
            
            thread = threading.Thread(
                target=self._process_alert_pipeline, 
                args=(cam_id, cam_name, sector, image, violation_type, confidence)
            )
            thread.daemon = True
            thread.start()
            return True
        return False

    def _process_alert_pipeline(self, cam_id, cam_name, sector, image, violation_type, confidence):
        with self._lock:
            try:
                data_hora_iso = time.strftime('%Y-%m-%dT%H:%M:%S')
                filename = f"evidencia_{cam_id}_{int(time.time())}.jpg"
                filepath = os.path.join("violations", filename)
                cv2.imwrite(filepath, image)

                # Persistência relacional imediata
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO logs_violation (timestamp, camera_id, camera_name, sector, violation_type, confidence, image_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (data_hora_iso, cam_id, cam_name, sector, violation_type, confidence, filepath))
                    conn.commit()

                # Disparo via API do Telegram Core
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                message = (
                    f"🚨 *ALERTA: VIOLAÇÃO DE EPI REGISTRADA*\n"
                    f"----------------------------------------\n"
                    f"• *Local:* {cam_name}\n"
                    f"• *Setor:* {sector}\n"
                    f"• *Infração:* Não conformidade de {violation_type}\n"
                    f"• *Confiança:* {confidence * 100:.1f}%\n"
                    f"----------------------------------------\n"
                    f"⚠️ _Mecanismo de fiscalização automatizado via IA._"
                )
                
                success, img_encoded = cv2.imencode('.jpg', image)
                if success:
                    files = {'photo': (filename, img_encoded.tobytes(), 'image/jpeg')}
                    data = {'chat_id': self.chat_id, 'caption': message, 'parse_mode': 'Markdown'}
                    requests.post(url, files=files, data=data, timeout=10)
                    
            except Exception as e:
                print(f"[ERRO ALERTA SERVER]: {e}")