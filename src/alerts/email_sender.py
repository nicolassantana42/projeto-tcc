import smtplib
import cv2
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from loguru import logger

# Importa as configurações do seu config.py
from config import (
    SMTP_SERVER, SMTP_PORT, EMAIL_SENDER, 
    EMAIL_PASS, EMAIL_RECEIVER
)

class EmailAlert:
    """
    Responsável pelo empacotamento e envio dos alertas 
    fotográficos via protocolo SMTP.
    """

    @staticmethod
    def send_violation(frame, violations, camera_id="Não Identificada"):
        """
        Monta o e-mail com a foto anexa e os dados da infração.
        """
        try:
            # 1. Cria a estrutura da mensagem
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER
            msg['Subject'] = f"⚠️ [SISTEMA EPI] VIOLAÇÃO: {camera_id}"

            timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
            
            # 2. Corpo do e-mail em HTML (Estilo Profissional)
            lista_html = "".join([f"<li><b>{v}</b></li>" for v in violations])
            
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">Alerta de Segurança do Trabalho</h2>
                <p>Uma irregularidade no uso de EPI foi detectada pelo sistema automático.</p>
                <hr>
                <p><b>📍 Localização:</b> {camera_id}</p>
                <p><b>⏰ Horário:</b> {timestamp}</p>
                <p><b>❌ Irregularidades encontradas:</b></p>
                <ul>{lista_html}</ul>
                <br>
                <p><i>Verifique a imagem em anexo para comprovação visual.</i></p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))

            # 3. Processamento da Imagem (Converte frame OpenCV para anexo)
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.error("Falha ao codificar imagem para o e-mail.")
                return False

            image_attachment = MIMEImage(buffer.tobytes())
            image_attachment.add_header('Content-Disposition', 'attachment', filename=f"violacao_{camera_id}.jpg")
            msg.attach(image_attachment)

            # 4. Conexão com o Servidor e Envio
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()  # Camada de segurança
                server.login(EMAIL_SENDER, EMAIL_PASS)
                server.send_message(msg)
            
            logger.success(f"📧 E-mail enviado com sucesso para {EMAIL_RECEIVER}")
            return True

        except Exception as e:
            logger.error(f"❌ Falha crítica no envio de e-mail: {e}")
            return False