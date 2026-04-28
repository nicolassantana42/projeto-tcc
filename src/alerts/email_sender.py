# src/alerts/email_sender.py
import smtplib
from email.message import EmailMessage
import threading
import cv2
import time
import config

class AsyncEmailSender:
    def __init__(self):
        self._lock = threading.Lock()

    def send_violation_alert(self, image, cam_name, sector):
        """Inicia o envio em Background (Não trava a detecção YOLO)"""
        thread = threading.Thread(
            target=self._process_email, 
            args=(image, cam_name, sector)
        )
        thread.daemon = True
        thread.start()

    def _process_email(self, image, cam_name, sector):
        with self._lock:
            try:
                # 1. Configurar Mensagem
                msg = EmailMessage()
                msg['Subject'] = f"URGENTE: Violação de EPI - {cam_name} ({sector})"
                msg['From'] = config.EMAIL_SENDER
                msg['To'] = config.EMAIL_RECEIVER
                
                data_hora = time.strftime('%d/%m/%Y %H:%M:%S')
                body = (
                    f"Atenção!\n\n"
                    f"Uma violação de segurança foi detectada pelo sistema.\n"
                    f"Trabalhador flagrado sem EPI completo (Capacete e/ou Bota).\n\n"
                    f"Detalhes da Ocorrência:\n"
                    f"- Câmera: {cam_name}\n"
                    f"- Setor: {sector}\n"
                    f"- Data/Hora: {data_hora}\n\n"
                    f"A evidência em imagem está anexada neste email."
                )
                msg.set_content(body)

                # 2. Converter imagem OpenCV para Anexo
                success, img_encoded = cv2.imencode('.jpg', image)
                if success:
                    msg.add_attachment(img_encoded.tobytes(), maintype='image', subtype='jpeg', filename=f'evidencia_{cam_name}.jpg')

                # 3. Enviar via SMTP
                with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                    server.starttls()
                    server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                    server.send_message(msg)
                    
                print(f"\n[SUCESSO] Email enviado para {config.EMAIL_RECEIVER} (Câmera: {cam_name})\n")
            
            except Exception as e:
                # Parte 4: Garantir que não falhe silenciosamente no log
                print(f"\n[ERRO CRÍTICO] Falha no envio de email. Verifique internet ou credenciais. Detalhe: {e}\n")