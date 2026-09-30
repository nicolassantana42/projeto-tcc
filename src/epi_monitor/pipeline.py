"""Orquestração por quadro, sem dependência da UI ou da captura."""

from collections import Counter
from time import perf_counter
from typing import Protocol

import numpy as np

from .preprocessing import preprocess_frame
from .rules import assess_ppe
from .types import Detection, FrameResult


class Detector(Protocol):
    names: dict[int, str]

    def predict(self, frame: np.ndarray, confidence: float | None = None, iou: float | None = None) -> list[Detection]: ...


class Pipeline:
    def __init__(self, detector: Detector, demo_mode: bool = True):
        self.detector = detector
        self.demo_mode = demo_mode
        self.frame_index = 0

    def process(self, frame: np.ndarray, confidence: float | None = None, iou: float | None = None) -> FrameResult:
        started = perf_counter()
        prepared = preprocess_frame(frame)
        inference_started = perf_counter()
        detections = self.detector.predict(prepared, confidence=confidence, iou=iou)
        inference_ms = (perf_counter() - inference_started) * 1000
        counts = dict(Counter(item.label for item in detections))
        alerts = assess_ppe(detections, self.detector.names, self.demo_mode)
        self.frame_index += 1
        return FrameResult(
            frame=frame,
            detections=detections,
            counts=counts,
            alerts=alerts,
            inference_ms=inference_ms,
            pipeline_ms=(perf_counter() - started) * 1000,
            frame_index=self.frame_index,
        )
