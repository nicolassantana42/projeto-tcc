"""Portable session reports, without retaining raw video or source credentials."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile


def frame_record(
    result: Any,
    *,
    illustrative: bool = False,
    observed_fps: float | None = None,
) -> dict[str, Any]:
    """Serialize one processed frame; illustrated frames never claim timings."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frame_index": int(result.frame_index),
        "mode": "illustrative_preview" if illustrative else "model_inference",
        "observed_fps": None if illustrative else observed_fps,
        "inference_ms": None if illustrative else float(result.inference_ms),
        "pipeline_ms": None if illustrative else float(result.pipeline_ms),
        "counts": {str(key): int(value) for key, value in result.counts.items()},
        "alerts": [str(alert) for alert in result.alerts],
        "detections": [
            {
                "class_id": int(item.class_id),
                "label": str(item.label),
                "confidence": None if illustrative else float(item.confidence),
                "bbox": [float(value) for value in item.bbox],
            }
            for item in result.detections
        ],
    }


def encode_snapshot(frame: Any) -> bytes:
    """Encode a BGR frame as PNG; fail clearly if OpenCV cannot encode it."""
    import cv2

    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Não há um frame válido para exportar.")
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("Não foi possível gerar o snapshot PNG.")
    return encoded.tobytes()


def _safe_cell(value: Any) -> Any:
    """Avoid formula execution when an untrusted class label is opened in Excel."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def build_report(
    records: Iterable[Mapping[str, Any]],
    *,
    snapshot_png: bytes | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Return ZIP bytes with frame JSON, detection CSV and optional PNG.

    Each CSV row describes a detection in one frame; empty frames get an empty
    detection row. Repeated detections are not unique people or tracked events.
    Callers should pass only descriptive source metadata, never stream URLs.
    """
    history = list(records)
    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counting_semantics": "Detecções por frame; não representa pessoas únicas.",
        "retained_frames": len(history),
        "metadata": dict(metadata or {}),
        "frames": history,
    }
    csv_buffer = io.StringIO(newline="")
    columns = [
        "timestamp_utc", "frame_index", "mode", "observed_fps",
        "inference_ms", "pipeline_ms", "class_id", "label", "confidence",
        "x1", "y1", "x2", "y2", "alerts",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=columns)
    writer.writeheader()
    for record in history:
        for detection in record.get("detections", []) or [{}]:
            bbox = detection.get("bbox", [None, None, None, None])
            row = {key: record.get(key) for key in columns[:6]}
            row.update({key: detection.get(key) for key in ("class_id", "label", "confidence")})
            row.update(dict(zip(("x1", "y1", "x2", "y2"), bbox)))
            row["alerts"] = " | ".join(str(value) for value in record.get("alerts", []))
            writer.writerow({key: _safe_cell(value) for key, value in row.items()})
    output = io.BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("report.json", json.dumps(document, ensure_ascii=False, indent=2))
        archive.writestr("detections.csv", csv_buffer.getvalue().encode("utf-8-sig"))
        if snapshot_png is not None:
            archive.writestr("snapshot.png", snapshot_png)
    return output.getvalue()
