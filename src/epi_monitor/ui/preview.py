"""Presentation-only image encoding and non-blocking analysis pacing."""
from __future__ import annotations

from base64 import b64encode
from time import monotonic

import cv2
import numpy as np


def encode_preview(frame: np.ndarray, *, max_size: int = 720, quality: int = 75) -> bytes:
    """Bound both preview dimensions, without modifying the original evidence."""
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("A prévia precisa de uma imagem BGR não vazia.")
    if max_size < 1 or not 1 <= quality <= 100:
        raise ValueError("Tamanho e qualidade da prévia inválidos.")
    height, width = frame.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    preview = frame
    if scale < 1:
        preview = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Não foi possível codificar a prévia.")
    return encoded.tobytes()


def preview_markup(content: bytes | None = None) -> str:
    """Reserve the same responsive 16:9 viewport, including before capture starts.

    Only our encoded JPEG enters the HTML; camera names, paths and other user
    input are never interpolated. Contain preserves portrait and 4:3 images.
    """
    body = (f'<img alt="Detecções no último frame analisado" '
            f'src="data:image/jpeg;base64,{b64encode(content).decode("ascii")}" '
            'style="display:block;width:100%;height:100%;object-fit:contain">') if content else (
            '<span style="padding:1rem;text-align:center">Aguardando captura</span>')
    return ('<div data-epi-preview="true" style="width:100%;max-width:720px;aspect-ratio:16/9;'
            'overflow:hidden;display:flex;align-items:center;justify-content:center;'
            'background:#0e1a2b;color:#b9cad8;border-radius:8px">' + body + '</div>')


class AnalysisPacer:
    """Limit analysis starts to a target period without blocking the UI.

    Inference time counts toward that period; a slow pipeline does not incur
    another full pause after completing. A skipped UI tick is neither an empty
    observation nor end-of-source and must never reach the event service.
    """

    def __init__(self, analyses_per_second: int = 10):
        if type(analyses_per_second) is not int or analyses_per_second not in (3, 10, 25, 30):
            raise ValueError("O limite precisa ser 3, 10, 25 ou 30 análises por segundo.")
        self.limit = analyses_per_second
        self.next_due = 0.0

    def ready(self) -> bool:
        return monotonic() >= self.next_due

    def started(self) -> None:
        self.next_due = monotonic() + 1 / self.limit
