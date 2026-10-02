"""Renderização pura em uma cópia do quadro; nenhuma lógica de detecção."""

import unicodedata
import cv2
import numpy as np

from .preprocessing import preprocess_frame
from .types import FrameResult

PALETTE = ((189, 224, 44), (255, 185, 80), (137, 115, 252), (90, 210, 250), (164, 209, 68))
# BGR: equipment found is green-ish, explicit absence is red.
LABEL_STYLE = {
    "helmet": ("Capacete", (80, 200, 60)), "vest": ("Colete", (60, 200, 140)), "boots": ("Bota", (200, 160, 40)),
    "no_helmet": ("Sem capacete", (60, 60, 235)), "no_vest": ("Sem colete", (60, 60, 235)),
    "no_boots": ("Sem bota", (60, 60, 235)),
}


def render_frame(result: FrameResult) -> np.ndarray:
    canvas = preprocess_frame(result.frame).copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(width, height) / 350))
    font_scale = max(0.4, min(0.75, min(width, height) / 800))
    for detection in result.detections:
        if result.assessments and detection.label == "person":
            continue  # The per-person status box below already frames people.
        name, color = LABEL_STYLE.get(detection.label, (detection.label, PALETTE[detection.class_id % len(PALETTE)]))
        x1, y1, x2, y2 = np.clip(detection.bbox, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        text = f"{label} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_y = min(height - baseline - 1, max(text_height + 4, y1 - 7))
        label_x = max(0, min(x1, width - text_width - 8))
        cv2.rectangle(canvas, (label_x, max(0, text_y - text_height - 4)), (min(width - 1, label_x + text_width + 8), min(height - 1, text_y + baseline)), color, -1)
        cv2.putText(canvas, text, (label_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    status_labels = {"ok": "EPIs OK", "unsafe": "SEM EPI", "uncertain": "INCONCLUSIVO"}
    status_colors = {"ok": (65, 200, 70), "unsafe": (50, 55, 240), "uncertain": (30, 205, 245)}
    for assessment in result.assessments:
        x1, y1, x2, y2 = np.clip(assessment.person.bbox, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]).astype(int)
        color = status_colors[assessment.status]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)
        text = f"P{assessment.index} {status_labels[assessment.status]}"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        x = max(0, min(x1, width - text_width - 4))
        y = max(16, min(height - 4, y2 - 8))
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return canvas
