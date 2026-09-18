"""Contrato da captura: imagens BGR uint8, sem resize/letterbox duplicado."""

import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Validate BGR input and make it contiguous without modifying its pixels.

    Ultralytics owns channel conversion, letterboxing and normalization. Keeping
    these there preserves bounding boxes in the original frame coordinates.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("O quadro deve ser um numpy.ndarray BGR.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("O quadro deve ter formato H x W x 3 (BGR).")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("O quadro está vazio.")
    if frame.dtype != np.uint8:
        raise ValueError("O quadro deve usar uint8, com valores entre 0 e 255.")
    return np.ascontiguousarray(frame)
