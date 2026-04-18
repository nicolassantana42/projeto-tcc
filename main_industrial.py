import json
import time
import cv2
from datetime import datetime
from loguru import logger

# Importações do seu projeto
from config import MODEL_PATH, ALERT_COOLDOWN_SECONDS
from src.ai.detector import EPIDetector
from src.rules.ppe_rules import PPERulesEngine, draw_ppe_status
from src.alerts.email_sender import EmailAlert

def main():
    logger.info("🏭 Iniciando Monitoramento Industrial (Modo Raspberry Pi)")

    # 1. Inicializa os motores de IA e Regras
    try:
        detector = EPIDetector(model_path=MODEL_PATH)
        rules = PPERulesEngine()
        alert_cooldowns = {} 
        logger.success("✅ Motores de IA e Regras carregados com sucesso.")
    except Exception as e:
        logger.error(f"❌ Falha crítica ao inicializar motores: {e}")
        return

    # 2. Carrega a lista de câmeras do arquivo provisionado
    try:
        with open('cameras.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            cameras = data.get('setor_producao', [])
        
        if not cameras:
            logger.error("❌ Nenhuma câmera encontrada no arquivo cameras.json.")
            return
            
        logger.info(f"📸 {len(cameras)} câmeras carregadas do sistema.")
    except Exception as e:
        logger.error(f"❌ Erro ao ler o arquivo cameras.json: {e}")
        return

    # 3. Loop de Monitoramento Circular (Round-Robin)
    logger.info("▶️ Iniciando ciclo de varredura...")
    while True:
        for cam in cameras:
            cam_id = cam.get('id', 'N/A')
            cam_local = cam.get('local', 'Desconhecido')
            cam_url = cam.get('url')

            logger.info(f"🔎 Analisando: {cam_id} em {cam_local}...")

            # 4. Captura com tratamento de timeout e erro de rede
            try:
                cap = cv2.VideoCapture(cam_url)
                
                # Define timeout de 5 segundos para não travar o loop se a câmera sumir
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                
                ret, frame = cap.read()
                cap.release() # Fecha a conexão imediatamente

                if not ret or frame is None:
                    logger.warning(f"⚠️ Câmera {cam_id} OFFLINE ou Timeout atingido.")
                    continue

            except Exception as e:
                logger.error(f"⚠️ Erro de conexão com {cam_id}: {e}")
                continue

            # 5. Processa Visão Computacional
            result = detector.detect(frame)
            statuses = rules.evaluate(result)

            # Filtra apenas os colaboradores em situação de irregularidade
            violadores = [s for s in statuses if not s.is_compliant]

            if violadores:
                agora = time.time()
                # Puxa o último alerta enviado por ESTA câmera específica
                ultimo_alerta = alert_cooldowns.get(cam_id, 0)

                # Verifica se o tempo de silêncio (cooldown) já passou
                if agora - ultimo_alerta > ALERT_COOLDOWN_SECONDS:
                    logger.critical(f"🚨 VIOLAÇÃO DETECTADA: {cam_id} ({cam_local})")

                    # Prepara a evidência visual (desenha caixas e labels de erro)
                    frame_com_alerta = draw_ppe_status(frame, statuses)
                    
                    # Consolida a lista de EPIs ausentes
                    erros_detectados = []
                    for v in violadores:
                        erros_detectados.extend(v.violations)
                    
                    # Remove duplicatas da lista de erros
                    erros_unicos = list(set(erros_detectados))

                    # 6. Disparo do Alerta Industrial (E-mail)
                    sucesso = EmailAlert.send_violation(
                        frame=frame_com_alerta,
                        violations=erros_unicos,
                        camera_id=f"{cam_id} - {cam_local}"
                    )

                    if sucesso:
                        alert_cooldowns[cam_id] = agora
                        logger.success(f"📧 Alerta enviado para supervisão: {cam_id}")
                else:
                    logger.info(f"⏳ Violação em {cam_id} ignorada devido ao Cooldown ativo.")

        # Pausa estratégica para evitar sobrecarga térmica no Raspberry Pi
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n🛑 Sistema encerrado pelo usuário.")
    except Exception as e:
        logger.critical(f"💥 Erro fatal: {e}")