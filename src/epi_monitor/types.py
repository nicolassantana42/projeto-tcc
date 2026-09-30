"""Objetos de domínio que não dependem de PyTorch ou Ultralytics."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .detection import PersonPPEAssessment


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
    assessments: list["PersonPPEAssessment"] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    ppe_executed: bool = False
