# ============================================================
#  src/ai/detector.py
#  Módulo de detecção com YOLOv8 (Ultralytics)
#  TCC: Monitoramento de EPI com Visão Computacional
# ============================================================

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

# Garante que a raiz do projeto esteja no sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MODEL_PATH, CONFIDENCE, IOU_THRESH,
    PERSON_CLASS_ID, HELMET_CLASS_ID, VEST_CLASS_ID
)


@dataclass
class Detection:
    """Representa um único objeto detectado no frame."""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]   # [x1, y1, x2, y2]

    @property
    def x1(self) -> int:      return self.bbox[0]
    @property
    def y1(self) -> int:      return self.bbox[1]
    @property
    def x2(self) -> int:      return self.bbox[2]
    @property
    def y2(self) -> int:      return self.bbox[3]
    @property
    def width(self) -> int:   return self.x2 - self.x1
    @property
    def height(self) -> int:  return self.y2 - self.y1
    @property
    def center_x(self) -> int: return (self.x1 + self.x2) // 2
    @property
    def center_y(self) -> int: return (self.y1 + self.y2) // 2
    @property
    def area(self) -> int:    return self.width * self.height


@dataclass
class FrameResult:
    """Resultado completo da análise de um frame."""
    persons:         List[Detection] = field(default_factory=list)
    helmets:         List[Detection] = field(default_factory=list)
    vests:           List[Detection] = field(default_factory=list)
    all:             List[Detection] = field(default_factory=list)
    raw_frame:       np.ndarray      = field(default=None, repr=False)
    annotated_frame: np.ndarray      = field(default=None, repr=False)


class EPIDetector:
    """
    Encapsula o modelo YOLOv8 e a lógica de inferência.

    Suporta dois modos:
    - DEMO:  usa yolov8n.pt (COCO), detecta 'person' apenas
    - REAL:  usa modelo customizado com classes helmet/vest
    """

    def __init__(self, model_path: str = MODEL_PATH):
        logger.info(f"🤖 Carregando modelo YOLOv8: {model_path}")
        try:
            # Compatibilidade com PyTorch 2.6+
            # O PyTorch 2.6 tornou weights_only=True o padrão.
            # Precisamos registrar as classes do ultralytics como seguras.
            try:
                import torch
                import ultralytics.nn.tasks as _tasks
                import ultralytics.nn.modules as _modules

                _safe = []
                for _name in dir(_tasks):
                    _obj = getattr(_tasks, _name)
                    if isinstance(_obj, type):
                        _safe.append(_obj)
                for _name in dir(_modules):
                    _obj = getattr(_modules, _name)
                    if isinstance(_obj, type):
                        _safe.append(_obj)

                torch.serialization.add_safe_globals(_safe)
            except Exception:
                pass  # versões antigas do PyTorch não precisam disso

            from ultralytics import YOLO
            self.model      = YOLO(model_path)
            self.model_path = model_path
            self.class_names = self.model.names
            logger.success(f"✅ Modelo carregado | Classes: {len(self.class_names)}")

            # Detecta automaticamente se está em modo demo (COCO)
            has_helmet = HELMET_CLASS_ID in self.class_names
            has_vest   = VEST_CLASS_ID   in self.class_names
            if not has_helmet or not has_vest:
                logger.warning(
                    "⚠️  MODO DEMO ATIVO: modelo COCO não possui classes de EPI.\n"
                    "   Capacetes e coletes NÃO serão detectados.\n"
                    "   → Para detecção real, baixe um modelo PPE customizado."
                )
            else:
                logger.success("✅ Modelo PPE com classes de EPI detectado!")

        except Exception as e:
            logger.error(f"❌ Falha ao carregar modelo: {e}")
            raise

    def detect(self, frame: np.ndarray) -> FrameResult:
        """Executa detecção em um frame e retorna detecções organizadas."""
        result = FrameResult(raw_frame=frame.copy())

        try:
            from ultralytics import YOLO
            predictions = self.model.predict(
                source=frame,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                verbose=False
            )

            if predictions and len(predictions) > 0:
                pred = predictions[0]

                for box in pred.boxes:
                    class_id   = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    det = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=[x1, y1, x2, y2]
                    )

                    result.all.append(det)

                    if   class_id == PERSON_CLASS_ID: result.persons.append(det)
                    elif class_id == HELMET_CLASS_ID: result.helmets.append(det)
                    elif class_id == VEST_CLASS_ID:   result.vests.append(det)

            result.annotated_frame = self._annotate_frame(frame.copy(), result)

        except Exception as e:
            logger.error(f"Erro durante detecção: {e}")
            result.annotated_frame = frame.copy()

        return result

    def _annotate_frame(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Desenha bounding boxes coloridas no frame."""
        color_map = {
            PERSON_CLASS_ID: (255, 140, 0),    # Laranja
            HELMET_CLASS_ID: (0, 200, 80),     # Verde
            VEST_CLASS_ID:   (0, 210, 210),    # Ciano
        }
        default_color = (180, 180, 180)

        for det in result.all:
            color = color_map.get(det.class_id, default_color)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (det.x1, det.y1 - th - 8), (det.x1 + tw + 6, det.y1), color, -1)
            cv2.putText(frame, label, (det.x1 + 3, det.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def benchmark(self, frame: np.ndarray, n_runs: int = 50) -> dict:
        """Mede performance — dados para o capítulo de resultados do TCC."""
        import time
        times = []
        # Warmup
        self.model.predict(source=frame, verbose=False)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.model.predict(source=frame, conf=CONFIDENCE, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        return {
            "model":     self.model_path,
            "n_runs":    n_runs,
            "avg_ms":    round(avg, 2),
            "min_ms":    round(min(times), 2),
            "max_ms":    round(max(times), 2),
            "fps_equiv": round(1000 / avg, 1),
            "std_ms":    round(float(np.std(times)), 2),
        }
