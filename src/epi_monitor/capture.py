"""Captura isolada com EOF finito e falhas explícitas de câmera/rede."""

from pathlib import Path
import math
import cv2
import numpy as np


class CaptureError(RuntimeError):
    """Fonte indisponível, desconectada ou inválida."""


class VideoSource:
    def __init__(self, source: int | str, timeout_ms: int = 5000):
        if isinstance(source, bool) or not isinstance(source, (int, str)):
            raise ValueError("A fonte deve ser um índice de webcam, arquivo ou URL de vídeo.")
        if isinstance(source, str):
            source = source.strip()
            if not source:
                raise ValueError("Informe uma fonte de vídeo.")
            if source.isdecimal():
                source = int(source)
        if isinstance(source, int) and source < 0:
            raise ValueError("O índice da webcam deve ser não negativo.")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("O timeout deve ser um inteiro positivo em milissegundos.")
        self.source = source
        self.timeout_ms = timeout_ms
        self.is_stream = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://"))
        self.is_file = isinstance(source, str) and not self.is_stream
        self._capture = None
        self._frames_read = 0
        self._frame_count = 0.0
        self._fps = 0.0
        self.timestamp_seconds = None

    def open(self) -> "VideoSource":
        self.close()
        self._frames_read = 0
        self.timestamp_seconds = None
        if self.is_file and not Path(self.source).is_file():
            raise CaptureError(f"Arquivo de vídeo não encontrado: {self.source}")
        try:
            if self.is_stream:
                if not hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") or not hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    raise CaptureError("Esta versão do OpenCV não oferece timeout de rede. Atualize o OpenCV.")
                self._capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms,
                ])
            else:
                self._capture = cv2.VideoCapture(self.source)
            if not self._capture.isOpened():
                raise CaptureError("Não foi possível abrir a fonte. Verifique permissões, conexão e codec do vídeo.")
            if not self.is_file:
                # Some camera backends ignore this setting; do not fail capture
                # when low-latency buffering is unavailable on a platform.
                try:
                    self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except (AttributeError, cv2.error):
                    pass
            self._frame_count = self._property(cv2.CAP_PROP_FRAME_COUNT) if self.is_file else 0.0
            self._fps = self._property(cv2.CAP_PROP_FPS) if self.is_file else 0.0
        except Exception as exc:
            self.close()
            if isinstance(exc, CaptureError):
                raise
            # Native OpenCV error strings can contain credentials from RTSP URLs.
            raise CaptureError("Falha ao abrir a fonte de vídeo. Verifique OpenCV, permissões e o endereço informado.") from exc
        return self

    def _property(self, key: int) -> float:
        """Unavailable optional codec metadata must not stop valid frames."""
        try:
            return float(self._capture.get(key))
        except (TypeError, ValueError, OverflowError, cv2.error):
            return math.nan

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise CaptureError("A fonte de vídeo ainda não foi aberta.")
        try:
            success, frame = self._capture.read()
        except Exception as exc:
            raise CaptureError("Falha ao ler a fonte de vídeo. Tente reconectar.") from exc
        if success and frame is not None and frame.size:
            self._frames_read += 1
            if self.is_file:
                position = self._property(cv2.CAP_PROP_POS_MSEC) / 1000
                fallback = (self._frames_read - 1) / self._fps if math.isfinite(self._fps) and self._fps > 0 else None
                if math.isfinite(position) and position >= 0 and (self._frames_read == 1 or position > (self.timestamp_seconds or 0)):
                    self.timestamp_seconds = position
                elif fallback is not None:
                    self.timestamp_seconds = max(self.timestamp_seconds or 0, fallback)
                else:
                    self.timestamp_seconds = None
            return frame
        if not self.is_file:
            raise CaptureError("Webcam/stream desconectado ou sem quadros. Verifique a fonte e tente reconectar.")
        if self._frames_read == 0:
            raise CaptureError("O arquivo não contém quadros decodificáveis; pode estar vazio ou corrompido.")
        if self._frame_count > 0 and self._frames_read < self._frame_count - 1:
            raise CaptureError("O vídeo terminou antes do esperado; o arquivo ou codec pode estar corrompido.")
        return None

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageSource:
    """A still image is exactly one observation, never simulated video time."""

    is_file = True
    is_stream = False
    timestamp_seconds = 0.0

    def __init__(self, source: str):
        self.source = str(source)
        self._frame = None
        self._opened = False

    def open(self):
        self.close()
        try:
            raw = np.fromfile(self.source, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
        except (OSError, cv2.error, ValueError) as error:
            raise CaptureError("Não foi possível ler a imagem. Verifique o arquivo e as permissões.") from error
        if frame is None or not frame.size:
            raise CaptureError("Imagem vazia, corrompida ou em formato não suportado.")
        self._frame, self._opened = frame, True
        return self

    def read(self):
        if not self._opened:
            raise CaptureError("A imagem ainda não foi aberta.")
        frame, self._frame = self._frame, None
        return frame

    def close(self):
        self._frame, self._opened = None, False

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


def open_source(source: int | str):
    """Choose still image or video capture without opening the resource yet."""
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES and "://" not in source:
        return ImageSource(source)
    return VideoSource(source)
