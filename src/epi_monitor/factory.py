"""Compose the actual detectors; UI and CLI share this factory."""
from .config import DEFAULT_PERSON_MODEL, DEFAULT_PPE_MODEL, InferenceConfig
from .inference import YOLODetector
from .detection import CascadePipeline


def create_cascade(person_model=DEFAULT_PERSON_MODEL, ppe_model=DEFAULT_PPE_MODEL,
                   device="auto", imgsz=640, confidence=.4, iou=.45):
    person = YOLODetector(InferenceConfig(model_path=person_model, device=device,
                                         imgsz=imgsz, confidence=confidence, iou=iou)).load()
    equipment = YOLODetector(InferenceConfig(model_path=ppe_model, device=device,
                                            imgsz=imgsz, confidence=confidence, iou=iou)).load()
    return CascadePipeline(person, equipment)
