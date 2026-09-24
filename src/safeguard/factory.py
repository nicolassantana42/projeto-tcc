"""Compose the actual detectors; UI and CLI share this factory."""
from .config import InferenceConfig
from .inference import YOLODetector
from .detection import CascadePipeline


def create_cascade(person_model="models/yolo11n.pt", ppe_model="models/ppe/best.pt",
                   device="auto", imgsz=640, confidence=.4, iou=.45):
    person = YOLODetector(InferenceConfig(model_path=person_model, device=device,
                                         imgsz=imgsz, confidence=confidence, iou=iou)).load()
    equipment = YOLODetector(InferenceConfig(model_path=ppe_model, device=device,
                                            imgsz=imgsz, confidence=confidence, iou=iou)).load()
    return CascadePipeline(person, equipment)
