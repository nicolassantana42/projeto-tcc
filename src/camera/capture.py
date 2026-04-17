# ============================================================
#  src/camera/capture.py — VERSÃO MELHORADA v2.0
#  Módulo de captura de vídeo com OpenCV
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  MELHORIAS:
#    ✅ Detecção automática de câmeras (testa índices 0-10)
#    ✅ Suporte a câmeras IP via RTSP/HTTP (Intelbras, Hikvision, etc)
#    ✅ Múltiplos backends (DirectShow, V4L2, GStreamer, MSMF)
#    ✅ Reconexão automática em caso de falha
#    ✅ Sistema de diagnóstico integrado
#    ✅ Buffer otimizado para reduzir latência
# ============================================================

import sys
import time
import platform
from pathlib import Path
from typing import Optional, Tuple, Union

# Garante que a raiz do projeto esteja no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from loguru import logger
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraDetector:
    """
    Detecta automaticamente câmeras disponíveis no sistema.
    Suporta: USB, Webcam integrada, IP Cameras (RTSP/HTTP).
    """
    
    @staticmethod
    def detect_available_cameras(max_test: int = 10) -> list:
        """
        Testa índices de 0 até max_test para encontrar câmeras disponíveis.
        Retorna lista de índices funcionais.
        """
        logger.info("🔍 Detectando câmeras disponíveis...")
        available = []
        
        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                    logger.info(f"  ✅ Câmera encontrada no índice {idx}")
                cap.release()
        
        if not available:
            logger.warning("  ⚠️  Nenhuma câmera USB/Webcam detectada")
        else:
            logger.success(f"  📷 Total: {len(available)} câmera(s) detectada(s)")
        
        return available
    
    @staticmethod
    def get_best_backend() -> int:
        """
        Retorna o melhor backend disponível para o sistema operacional.
        """
        system = platform.system()
        
        if system == "Windows":
            # Windows: DirectShow é mais estável que MSMF para a maioria dos casos
            return cv2.CAP_DSHOW
        elif system == "Linux":
            # Linux: V4L2 é o padrão
            return cv2.CAP_V4L2
        elif system == "Darwin":  # macOS
            return cv2.CAP_AVFOUNDATION
        else:
            return cv2.CAP_ANY
    
    @staticmethod
    def diagnose_camera_issue(camera_source: Union[int, str]) -> None:
        """
        Fornece diagnóstico detalhado quando uma câmera não pode ser aberta.
        """
        logger.error(f"\n{'='*60}")
        logger.error("❌ DIAGNÓSTICO DE CÂMERA")
        logger.error(f"{'='*60}")
        logger.error(f"Fonte: {camera_source}")
        logger.error(f"Sistema: {platform.system()} {platform.release()}")
        logger.error(f"OpenCV: {cv2.__version__}")
        
        if isinstance(camera_source, int):
            logger.error("\n📋 SOLUÇÕES POSSÍVEIS:")
            logger.error("  1. Verifique se a câmera está conectada fisicamente")
            logger.error("  2. Teste outros índices: CAMERA_INDEX=1 ou CAMERA_INDEX=2")
            logger.error("  3. Feche outros programas usando a câmera (Zoom, Teams, Skype)")
            logger.error("  4. No Windows: Verifique Configurações > Privacidade > Câmera")
            logger.error("  5. Execute o detector automático: python -c 'from src.camera.capture import CameraDetector; CameraDetector.detect_available_cameras()'")
            
            # Tenta detectar automaticamente
            available = CameraDetector.detect_available_cameras(5)
            if available:
                logger.info(f"\n💡 SUGESTÃO: Use CAMERA_INDEX={available[0]} no arquivo .env")
        else:
            logger.error("\n📋 SOLUÇÕES PARA CÂMERAS IP:")
            logger.error("  1. Verifique se a URL está correta")
            logger.error("  2. Teste o stream em um player: VLC Media Player")
            logger.error("  3. Verifique usuário/senha se necessário")
            logger.error("  4. Verifique se a câmera está na mesma rede")
            logger.error("  5. Tente pingar o IP da câmera")
        
        logger.error(f"{'='*60}\n")


class VideoCapture:
    """
    Encapsula a captura de vídeo robusta com suporte a:
      - Webcams USB (índice 0, 1, 2, etc)
      - Câmeras integradas
      - Câmeras IP via RTSP/HTTP (Intelbras, Hikvision, etc)
      - Reconexão automática
      - Múltiplos backends

    Uso básico:
        cap = VideoCapture()
        for frame in cap.stream():
            # processa o frame
            pass
    
    Uso com câmera IP:
        cap = VideoCapture(camera_source="rtsp://admin:senha@192.168.1.100:554/stream")
        for frame in cap.stream():
            # processa o frame
            pass
    """

    def __init__(self,
                 camera_source: Union[int, str] = None,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT,
                 auto_detect: bool = True,
                 buffer_size: int = 1):
        """
        Inicializa o VideoCapture.
        
        Args:
            camera_source: Índice da câmera (0, 1, 2...) ou URL RTSP/HTTP
                          Se None, usa CAMERA_INDEX do config ou detecta automaticamente
            width: Largura desejada do frame
            height: Altura desejada do frame
            auto_detect: Se True, detecta automaticamente a primeira câmera disponível
            buffer_size: Tamanho do buffer (1 = sem buffer, reduz latência)
        """
        # Define a fonte da câmera
        if camera_source is None:
            if auto_detect:
                logger.info("🔍 Modo de detecção automática ativado...")
                available = CameraDetector.detect_available_cameras(10)
                if available:
                    self.camera_source = available[0]
                    logger.success(f"✅ Usando câmera detectada automaticamente: {self.camera_source}")
                else:
                    # Fallback para o índice configurado
                    self.camera_source = CAMERA_INDEX
                    logger.warning(f"⚠️  Nenhuma câmera detectada, usando índice padrão: {self.camera_source}")
            else:
                self.camera_source = CAMERA_INDEX
        else:
            self.camera_source = camera_source
        
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.cap = None
        self._fps_counter = FPSCounter()
        self._backend = CameraDetector.get_best_backend()
        self._reconnection_attempts = 0
        self._max_reconnection_attempts = 3
        
        # Identifica se é câmera IP
        self.is_ip_camera = isinstance(self.camera_source, str) and \
                           any(proto in str(self.camera_source).lower() 
                               for proto in ['rtsp://', 'http://', 'https://'])
        
        if self.is_ip_camera:
            logger.info(f"📡 Modo câmera IP detectado: {self.camera_source}")

    def _try_open_with_backend(self, backend: int) -> bool:
        """
        Tenta abrir a câmera com um backend específico.
        """
        try:
            if isinstance(self.camera_source, int):
                self.cap = cv2.VideoCapture(self.camera_source, backend)
            else:
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if self.cap.isOpened():
                return True
            
            if self.cap:
                self.cap.release()
            return False
        except Exception as e:
            logger.debug(f"Erro ao tentar backend {backend}: {e}")
            return False

    def start(self) -> bool:
        """
        Inicializa a câmera com tentativa de múltiplos backends.
        Retorna True se bem-sucedido.
        """
        logger.info(f"🎬 Inicializando câmera: {self.camera_source}")
        
        # Lista de backends para tentar (em ordem de preferência)
        backends_to_try = [
            self._backend,  # Backend otimizado para o SO
            cv2.CAP_ANY,    # Deixa OpenCV escolher
        ]
        
        # Adiciona backends adicionais no Windows
        if platform.system() == "Windows":
            backends_to_try.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])
        
        # Tenta cada backend
        for backend in backends_to_try:
            logger.debug(f"  Tentando backend: {backend}")
            if self._try_open_with_backend(backend):
                logger.debug(f"  ✅ Backend {backend} funcionou!")
                break
        
        if not self.cap or not self.cap.isOpened():
            CameraDetector.diagnose_camera_issue(self.camera_source)
            return False
        
        # Configurações otimizadas
        try:
            # Define o tamanho do buffer (reduz latência)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Define resolução (pode não funcionar para todas as câmeras)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Para câmeras IP, ajusta configurações de rede
            if self.is_ip_camera:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Mínimo buffer para IP
                # Tenta habilitar TCP para RTSP (mais estável que UDP)
                # Nota: nem todos os drivers suportam isso
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                except:
                    pass
            
            # Lê um frame de teste
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("❌ Câmera aberta mas não fornece frames")
                self.cap.release()
                return False
            
            # Obtém resolução real
            real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_cam = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.success(f"✅ Câmera iniciada com sucesso!")
            logger.info(f"  📐 Resolução: {real_w}x{real_h}")
            logger.info(f"  🎞️  FPS da câmera: {fps_cam if fps_cam > 0 else 'Não disponível'}")
            logger.info(f"  🔧 Backend: {self._backend}")
            
            self._reconnection_attempts = 0
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na configuração da câmera: {e}")
            if self.cap:
                self.cap.release()
            return False

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Lê e retorna o próximo frame.
        Returns: (True, frame) ou (False, None)
        
        Com tentativa de reconexão automática em caso de falha.
        """
        if self.cap is None or not self.cap.isOpened():
            # Tenta reconectar
            if self._reconnection_attempts < self._max_reconnection_attempts:
                self._reconnection_attempts += 1
                logger.warning(f"⚠️  Tentando reconectar... (tentativa {self._reconnection_attempts}/{self._max_reconnection_attempts})")
                if self.start():
                    return self.read_frame()
            return False, None

        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            logger.warning("⚠️  Frame inválido recebido")
            
            # Tenta reconectar para câmeras IP (comum ter drops de conexão)
            if self.is_ip_camera and self._reconnection_attempts < self._max_reconnection_attempts:
                self._reconnection_attempts += 1
                logger.warning(f"  Reconectando câmera IP... (tentativa {self._reconnection_attempts})")
                time.sleep(1)  # Aguarda 1 segundo antes de reconectar
                self.release()
                if self.start():
                    return self.read_frame()
            
            return False, None

        self.fps = self._fps_counter.update()
        self._reconnection_attempts = 0  # Reset em caso de sucesso
        return True, frame

    def stream(self):
        """
        Gerador que itera continuamente sobre os frames.
        Com tratamento robusto de erros e reconexão automática.
        """
        if not self.start():
            logger.error("❌ Não foi possível iniciar a câmera")
            return
        
        consecutive_failures = 0
        max_consecutive_failures = 30  # 30 frames consecutivos falhando = encerra
        
        try:
            while True:
                ret, frame = self.read_frame()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"⚠️  Falha consecutiva #{consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("❌ Muitas falhas consecutivas. Encerrando stream.")
                        break
                    
                    time.sleep(0.1)  # Aguarda antes de tentar novamente
                    continue
                
                consecutive_failures = 0  # Reset em caso de sucesso
                yield frame
                
        except KeyboardInterrupt:
            logger.info("⏹️  Stream interrompido pelo usuário")
        except Exception as e:
            logger.error(f"❌ Erro no stream: {e}")
        finally:
            self.release()

    def release(self):
        """Libera os recursos da câmera."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("📷 Câmera liberada")

    @property
    def current_fps(self) -> float:
        return self._fps_counter.fps


class FPSCounter:
    """
    Calcula o FPS médio usando janela deslizante de 30 frames.
    Suaviza variações momentâneas no processamento.
    """

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

        return self.fps


# ── Função de teste/diagnóstico ──────────────────────────────
def test_camera_system():
    """
    Função de teste para diagnosticar o sistema de câmeras.
    Execute com: python -m src.camera.capture
    """
    logger.info("\n" + "="*60)
    logger.info("🧪 TESTE DO SISTEMA DE CÂMERAS")
    logger.info("="*60 + "\n")
    
    # Detecta câmeras disponíveis
    available = CameraDetector.detect_available_cameras(10)
    
    if not available:
        logger.error("❌ Nenhuma câmera detectada!")
        return
    
    # Testa a primeira câmera encontrada
    logger.info(f"\n🎬 Testando câmera {available[0]}...")
    cap = VideoCapture(camera_source=available[0])
    
    logger.info("📹 Capturando 10 frames de teste...")
    frame_count = 0
    
    for frame in cap.stream():
        frame_count += 1
        logger.info(f"  Frame {frame_count}/10 | Shape: {frame.shape} | FPS: {cap.current_fps:.1f}")
        
        if frame_count >= 10:
            break
    
    logger.success("\n✅ Teste concluído com sucesso!")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    test_camera_system()
