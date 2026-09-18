"""Objetos de domínio que não dependem de PyTorch ou Ultralytics."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class FrameResult:
    frame: np.ndarray
    detections: list[Detection]
    counts: dict[str, int]
    alerts: list[str]
    inference_ms: float
    pipeline_ms: float
    frame_index: int
