"""Configuração validada, independente da interface e do backend."""

from dataclasses import dataclass
import math


def validate_threshold(value: float, name: str) -> float:
    """Normalize a finite probability; reject booleans and silent coercion errors."""
    if isinstance(value, bool):
        raise ValueError(f"{name} deve ser um número entre 0 e 1.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve ser um número entre 0 e 1.") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} deve estar entre 0 e 1.")
    return number


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str = "models/yolo11n.pt"
    device: str = "auto"
    confidence: float = 0.4
    iou: float = 0.45
    imgsz: int = 640

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("Informe o caminho do modelo.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Informe um dispositivo: auto, cpu, cuda:0 ou mps.")
        object.__setattr__(self, "confidence", validate_threshold(self.confidence, "Confiança"))
        object.__setattr__(self, "iou", validate_threshold(self.iou, "IoU"))
        if isinstance(self.imgsz, bool) or not isinstance(self.imgsz, int) or self.imgsz < 32:
            raise ValueError("imgsz deve ser um inteiro de pelo menos 32 pixels.")
