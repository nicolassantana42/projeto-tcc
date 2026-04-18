# ============================================================
#  src/ai/detector.py — Detector Silencioso para Raspberry Pi 5
# ============================================================

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

# Garante que a raiz do projeto esteja no path para importar o config
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MODEL_PATH, CONFIDENCE, IOU_THRESH,
    PERSON_CLASS_ID, HELMET_CLASS_ID, VEST_CLASS_ID, BOOT_CLASS_ID
)

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]

    @property
    def x1(self): return self.bbox[0]
    @property
    def y1(self): return self.bbox[1]
    @property
    def x2(self): return self.bbox[2]
    @property
    def y2(self): return self.bbox[3]

    @property
    def center_x(self): return (self.x1 + self.x2) // 2
    @property
    def center_y(self): return (self.y1 + self.y2) // 2


@dataclass
class FrameResult:
    persons: List[Detection] = field(default_factory=list)
    helmets: List[Detection] = field(default_factory=list)
    vests: List[Detection] = field(default_factory=list)
    boots: List[Detection] = field(default_factory=list)
    all: List[Detection] = field(default_factory=list)
    raw_frame: np.ndarray = None
    annotated_frame: np.ndarray = None


class EPIDetector:
    """
    Detector baseado em YOLOv8 otimizado para execução em borda (Raspberry Pi 5).
    Focado em fornecer dados brutos sem poluição visual.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        from ultralytics import YOLO

        logger.info(f"🤖 Carregando modelo YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> FrameResult:
        """
        Realiza a inferência e retorna apenas os dados. 
        O frame anotado retornará SEMPRE limpo para evitar poluição visual.
        """
        # IMPORTANTE: Criamos uma cópia limpa para o resultado
        clean_frame = frame.copy()
        result = FrameResult(raw_frame=frame.copy(), annotated_frame=clean_frame)

        # Inferência com parâmetros de silenciamento
        predictions = self.model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=IOU_THRESH,
            imgsz=640,
            verbose=False,
            device='cpu',
            show=False,        # Garante que não abra janelas extras
            save=False,        # Garante que não salve fotos intermediárias
        )

        if not predictions or len(predictions[0]) == 0:
            return result

        pred = predictions[0]

        for box in pred.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            det = Detection(
                class_id=class_id,
                class_name=self.class_names.get(class_id, str(class_id)),
                confidence=confidence,
                bbox=[x1, y1, x2, y2]
            )

            result.all.append(det)

            # Organização por categorias para o Rules Engine
            if class_id == PERSON_CLASS_ID:
                result.persons.append(det)
            elif class_id == HELMET_CLASS_ID:
                result.helmets.append(det)
            elif class_id == VEST_CLASS_ID:
                result.vests.append(det)
            elif class_id == BOOT_CLASS_ID:
                result.boots.append(det)

        # Retornamos o frame anotado como uma cópia do frame limpo.
        # Nenhuma caixa vermelha do YOLO original passará por aqui.
        result.annotated_frame = clean_frame.copy()
        return result

    def benchmark(self, frame: np.ndarray, n_runs: int = 50):
        """Mede a performance no hardware atual."""
        import time
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.model.predict(frame, imgsz=640, verbose=False, device='cpu')
            latencies.append((time.perf_counter() - t0) * 1000)
        
        return {
            "model": MODEL_PATH,
            "n_runs": n_runs,
            "avg_ms": np.mean(latencies),
            "fps_equiv": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        }