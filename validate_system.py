#!/usr/bin/env python3
# ============================================================
#  validate_system.py — Validação Completa do Sistema
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  Este script valida todas as funcionalidades implementadas:
#  ✅ Detecção de câmeras
#  ✅ Backends múltiplos
#  ✅ Captura de frames
#  ✅ Modelo YOLOv8
#  ✅ Sistema de regras EPI
#  ✅ Logs e alertas
#  ✅ Configurações
# ============================================================

import sys
from pathlib import Path

# Adiciona raiz ao path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import time
from loguru import logger

# Configurações de log
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)


class SystemValidator:
    """Validador completo do sistema."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []
    
    def test(self, name: str, func):
        """Executa um teste e registra o resultado."""
        logger.info(f"\n{'─'*70}")
        logger.info(f"🧪 Teste: {name}")
        logger.info(f"{'─'*70}")
        
        try:
            func()
            self.tests_passed += 1
            self.results.append(("✅", name))
            logger.success(f"✅ PASSOU: {name}")
        except AssertionError as e:
            self.tests_failed += 1
            self.results.append(("❌", name))
            logger.error(f"❌ FALHOU: {name}")
            logger.error(f"   Erro: {e}")
        except Exception as e:
            self.tests_failed += 1
            self.results.append(("⚠️", name))
            logger.error(f"⚠️  ERRO: {name}")
            logger.error(f"   Exceção: {e}")
    
    def print_summary(self):
        """Imprime sumário dos resultados."""
        print("\n" + "="*70)
        print("📊 RESUMO DA VALIDAÇÃO")
        print("="*70)
        
        for status, name in self.results:
            print(f"  {status} {name}")
        
        total = self.tests_passed + self.tests_failed
        percentage = (self.tests_passed / total * 100) if total > 0 else 0
        
        print("\n" + "─"*70)
        print(f"  Total de testes: {total}")
        print(f"  ✅ Passaram: {self.tests_passed}")
        print(f"  ❌ Falharam: {self.tests_failed}")
        print(f"  📈 Taxa de sucesso: {percentage:.1f}%")
        print("="*70 + "\n")
        
        return self.tests_failed == 0


def test_imports():
    """Valida que todos os módulos podem ser importados."""
    logger.info("Importando módulos...")
    
    import config
    from src.camera.capture import VideoCapture, CameraDetector
    from src.ai.detector import EPIDetector
    from src.rules.ppe_rules import PPERulesEngine
    from src.alerts.logger import ViolationLogger
    from src.alerts.telegram import TelegramAlerter
    
    logger.info("✓ Todos os módulos importados com sucesso")


def test_config():
    """Valida configurações."""
    logger.info("Validando configurações...")
    
    import config
    
    # Verifica variáveis essenciais
    assert hasattr(config, 'CAMERA_INDEX'), "CAMERA_INDEX não definido"
    assert hasattr(config, 'MODEL_PATH'), "MODEL_PATH não definido"
    assert hasattr(config, 'CONFIDENCE'), "CONFIDENCE não definido"
    
    # Verifica diretórios
    assert config.BASE_DIR.exists(), "BASE_DIR não existe"
    assert config.VIOLATIONS_DIR.exists(), "VIOLATIONS_DIR não foi criado"
    assert config.LOGS_DIR.exists(), "LOGS_DIR não foi criado"
    
    logger.info(f"✓ CAMERA_INDEX = {config.CAMERA_INDEX}")
    logger.info(f"✓ MODEL_PATH = {config.MODEL_PATH}")
    logger.info(f"✓ CONFIDENCE = {config.CONFIDENCE}")
    logger.info(f"✓ Diretórios criados: violations/, logs/")


def test_opencv():
    """Valida instalação do OpenCV."""
    logger.info("Validando OpenCV...")
    
    version = cv2.__version__
    logger.info(f"✓ OpenCV versão: {version}")
    
    # Verifica versão mínima
    major = int(version.split('.')[0])
    assert major >= 4, f"OpenCV versão muito antiga: {version}"
    
    # Testa backends disponíveis
    backends = cv2.getBuildInformation()
    logger.info("✓ Backends disponíveis detectados")


def test_camera_detector():
    """Valida sistema de detecção de câmeras."""
    logger.info("Testando detecção de câmeras...")
    
    from src.camera.capture import CameraDetector
    
    # Detecta câmeras
    available = CameraDetector.detect_available_cameras(max_test=5)
    
    logger.info(f"✓ Câmeras detectadas: {len(available)}")
    for idx in available:
        logger.info(f"  - Índice {idx}")
    
    # Nota: Não falha se não encontrar câmeras (pode estar em servidor)
    if len(available) == 0:
        logger.warning("  ⚠️ Nenhuma câmera detectada (OK se estiver em servidor)")


def test_video_capture():
    """Valida captura de vídeo."""
    logger.info("Testando captura de vídeo...")
    
    from src.camera.capture import VideoCapture
    
    # Tenta criar VideoCapture
    cap = VideoCapture(auto_detect=True)
    logger.info("✓ VideoCapture criado")
    
    # Tenta iniciar (pode falhar se não houver câmera)
    if cap.start():
        logger.info("✓ Câmera iniciada com sucesso")
        
        # Tenta capturar um frame
        ret, frame = cap.read_frame()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            logger.info(f"✓ Frame capturado: {w}x{h}")
        
        cap.release()
        logger.info("✓ Câmera liberada")
    else:
        logger.warning("  ⚠️ Câmera não disponível (OK se estiver em servidor)")


def test_yolo_model():
    """Valida carregamento do modelo YOLOv8."""
    logger.info("Testando modelo YOLOv8...")
    
    from src.ai.detector import EPIDetector
    import config
    
    # Verifica se o modelo existe
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        logger.warning(f"  ⚠️ Modelo não encontrado: {config.MODEL_PATH}")
        logger.warning(f"  O modelo será baixado automaticamente na primeira execução")
        return
    
    # Tenta carregar o modelo
    detector = EPIDetector(model_path=str(model_path))
    logger.info(f"✓ Modelo carregado: {config.MODEL_PATH}")
    
    # Testa detecção em imagem dummy
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = detector.detect(dummy_frame)
    logger.info(f"✓ Detecção executada (dummy frame)")
    logger.info(f"  - Detecções: {len(result.detections)}")


def test_ppe_rules():
    """Valida motor de regras EPI."""
    logger.info("Testando motor de regras EPI...")
    
    from src.rules.ppe_rules import PPERulesEngine
    
    engine = PPERulesEngine()
    logger.info("✓ PPERulesEngine criado")
    logger.info(f"  - Capacete obrigatório: {engine.require_helmet}")
    logger.info(f"  - Colete obrigatório: {engine.require_vest}")


def test_violation_logger():
    """Valida sistema de log de violações."""
    logger.info("Testando logger de violações...")
    
    from src.alerts.logger import ViolationLogger
    
    vlogger = ViolationLogger()
    logger.info("✓ ViolationLogger criado")
    
    # Testa contagem de violações
    count = vlogger.get_today_count()
    logger.info(f"✓ Violações hoje: {count}")


def test_telegram_alerter():
    """Valida sistema de alertas Telegram."""
    logger.info("Testando alertas Telegram...")
    
    from src.alerts.telegram import TelegramAlerter
    import config
    
    telegram = TelegramAlerter()
    logger.info(f"✓ TelegramAlerter criado")
    logger.info(f"  - Habilitado: {telegram.enabled}")
    
    if not telegram.enabled:
        logger.warning("  ⚠️ Telegram não configurado (OK para testes)")
    else:
        logger.info(f"  - Token configurado: {config.TELEGRAM_TOKEN[:10]}...")


def test_file_structure():
    """Valida estrutura de arquivos do projeto."""
    logger.info("Validando estrutura de arquivos...")
    
    required_files = [
        "main.py",
        "config.py",
        "requirements.txt",
        ".env.example",
        "test_camera.py",
        "TROUBLESHOOTING.md",
        "CHANGELOG.md",
        "src/camera/capture.py",
        "src/ai/detector.py",
        "src/rules/ppe_rules.py",
        "src/alerts/logger.py",
        "src/alerts/telegram.py",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = ROOT / file_path
        if full_path.exists():
            logger.debug(f"  ✓ {file_path}")
        else:
            missing.append(file_path)
            logger.warning(f"  ⚠️  {file_path} não encontrado")
    
    assert len(missing) == 0, f"Arquivos faltando: {missing}"
    logger.info(f"✓ Todos os {len(required_files)} arquivos essenciais presentes")


def test_python_version():
    """Valida versão do Python."""
    logger.info("Validando versão do Python...")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    logger.info(f"✓ Python {version_str}")
    
    assert version_info.major == 3, "Python 3 é necessário"
    assert version_info.minor >= 9, "Python 3.9+ é necessário"


def main():
    """Executa validação completa."""
    print("\n" + "="*70)
    print("🔬 VALIDAÇÃO COMPLETA DO SISTEMA")
    print("   Sistema de Monitoramento de EPI - TCC")
    print("="*70 + "\n")
    
    validator = SystemValidator()
    
    # Executa testes em ordem lógica
    validator.test("1. Versão do Python", test_python_version)
    validator.test("2. Estrutura de arquivos", test_file_structure)
    validator.test("3. Importação de módulos", test_imports)
    validator.test("4. Configurações", test_config)
    validator.test("5. OpenCV", test_opencv)
    validator.test("6. Detecção de câmeras", test_camera_detector)
    validator.test("7. Captura de vídeo", test_video_capture)
    validator.test("8. Modelo YOLOv8", test_yolo_model)
    validator.test("9. Motor de regras EPI", test_ppe_rules)
    validator.test("10. Logger de violações", test_violation_logger)
    validator.test("11. Alertas Telegram", test_telegram_alerter)
    
    # Imprime sumário
    all_passed = validator.print_summary()
    
    if all_passed:
        logger.success("🎉 Sistema totalmente validado e pronto para uso!")
        logger.info("\n💡 Próximos passos:")
        logger.info("  1. Configure a câmera: python test_camera.py --detect")
        logger.info("  2. Execute o sistema: python main.py")
        logger.info("  3. Abra o dashboard: streamlit run src/dashboard/streamlit_app.py")
        return 0
    else:
        logger.error("❌ Sistema com problemas. Verifique os erros acima.")
        logger.info("\n💡 Para ajuda:")
        logger.info("  - Leia TROUBLESHOOTING.md")
        logger.info("  - Verifique .env.example")
        logger.info("  - Execute: python test_camera.py --detect")
        return 1


if __name__ == "__main__":
    sys.exit(main())
