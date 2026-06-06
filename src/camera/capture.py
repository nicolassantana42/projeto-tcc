# src/camera/capture.py
import sys
import time
import platform
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from loguru import logger

class CameraDetector:
    @staticmethod
    def detect_available_cameras(max_test: int = 10) -> list:
        logger.info("🔍 Detectando câmeras disponíveis...")
        available = []
        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                    logger.info(f"   ✅ Câmera encontrada no índice {idx}")
                cap.release()
        return available
    
    @staticmethod
    def get_best_backend() -> int:
        system = platform.system()
        if system == "Windows": return cv2.CAP_DSHOW
        elif system == "Linux": return cv2.CAP_V4L2
        else: return cv2.CAP_ANY

class IPCameraStream:
    def __init__(self, camera_id: str, connection_string: Union[int, str], width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.connection_string = connection_string
        if str(self.connection_string).isdigit():
            self.connection_string = int(self.connection_string)
            
        self.width = width
        self.height = height
        self.cap = None
        self.ret = False
        self.frame = None
        self.is_running = False
        self.thread = None
        self._fps_counter = FPSCounter()
        self._backend = CameraDetector.get_best_backend()

    def start(self) -> bool:
        try:
            if isinstance(self.connection_string, int):
                self.cap = cv2.VideoCapture(self.connection_string, self._backend)
            else:
                self.cap = cv2.VideoCapture(self.connection_string)
                
            if not self.cap or not self.cap.isOpened():
                return False
                
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()
            return True
        except Exception:
            return False

    def _update_frame(self):
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret = ret
                    self.frame = frame
                    self._fps_counter.update()
                else:
                    self.ret = False
                    time.sleep(2)
                    if self.cap: self.cap.open(self.connection_string)
            else:
                time.sleep(1)

    def get_frame(self) -> Tuple[bool, Optional[any]]:
        return self.ret, self.frame

    def stop(self):
        self.is_running = False
        if self.thread: self.thread.join(timeout=1)
        if self.cap and self.cap.isOpened(): self.cap.release()

class FPSCounter:
    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window: self.timestamps.pop(0)
        if len(self.timestamps) > 1:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
        return self.fps