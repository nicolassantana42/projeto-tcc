"""Evaluate final cascade boxes at fixed thresholds, without claiming mAP.

Annotations follow YOLO detection format (class, center x/y, width, height).
Only canonical classes explicitly declared in the dataset are scored. These
box annotations cannot establish correctness of per-person compliance states.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import yaml

from .config import validate_threshold
from .detection import canonical_label


CANONICAL_CLASSES = ("person", "helmet", "vest", "no_helmet", "no_vest")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_ERROR_RECORDS = 5000


class EvaluationError(ValueError):
    """Invalid local annotations, dataset layout or pipeline output."""


def _names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        value = dict(enumerate(value))
    if not isinstance(value, dict) or not value:
        raise EvaluationError("O YAML deve definir 'names' como lista ou mapeamento de classes.")
    names = {}
    for key, label in value.items():
        if isinstance(key, bool) or not str(key).isdecimal() or not isinstance(label, str) or not label.strip():
            raise EvaluationError("IDs de classe devem ser inteiros não negativos e nomes devem ser textos.")
        identifier = int(key)
        if identifier in names:
            raise EvaluationError("O YAML contém IDs de classe duplicados.")
        names[identifier] = label.strip()
    return names


def _dataset(data: str | Path, split: str) -> tuple[Path, Path, dict[int, str], list[Path], set[Path]]:
    source = Path(data).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationError("Não foi possível ler o YAML local do dataset.") from error
    if not isinstance(document, dict):
        raise EvaluationError("O dataset precisa ser um mapeamento YAML.")
    if "download" in document:
        raise EvaluationError("Prepare o dataset localmente e remova 'download' do YAML antes de avaliar.")
    names = _names(document.get("names"))
    root_value = document.get("path") or str(source.parent)
    if not isinstance(root_value, str):
        raise EvaluationError("O campo 'path' deve ser um caminho local.")
    root = Path(root_value).expanduser()
    root = (root if root.is_absolute() else source.parent / root).resolve()
    entries = document.get(split)
    if not entries:
        raise EvaluationError(f"O YAML precisa definir o split '{split}'.")
    entries = entries if isinstance(entries, list) else [entries]
    images: set[Path] = set()
    protected = {source}
    for entry in entries:
        if not isinstance(entry, str):
            raise EvaluationError("Cada entrada do split deve ser um caminho local.")
        path = Path(entry).expanduser()
        path = (path if path.is_absolute() else root / path).resolve()
        protected.add(path)
        if path.is_dir():
            images.update(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                image = Path(line.strip()).expanduser()
                # Image lists use paths relative to the list, never the process CWD.
                image = (image if image.is_absolute() else path.parent / image).resolve()
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    raise EvaluationError(f"Imagem da lista não encontrada ou formato não aceito: {image}")
                images.add(image)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.add(path)
        else:
            raise EvaluationError(f"Split local não encontrado ou formato não aceito: {path}")
    if not images:
        raise EvaluationError("O split não contém imagens; nenhuma avaliação foi executada.")
    protected.update(images)
    protected.update(_label_path(image).resolve() for image in images)
    return source, root, names, sorted(images, key=lambda path: path.as_posix()), protected


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, part in enumerate(parts[:-1]) if part.lower() == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    # YOLO also permits image/label pairs in the same directory.
    return image.with_suffix(".txt")


def _ground_truth(payload: bytes | None, names: dict[int, str], width: int, height: int,
                  image: Path) -> list[dict[str, Any]]:
    annotations = []
    try:
        rows = payload.decode("utf-8-sig").splitlines() if payload is not None else []
        for line_number, line in enumerate(rows, start=1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError("YOLO detection requires five columns")
            identifier = float(values[0])
            if not math.isfinite(identifier) or not identifier.is_integer() or identifier < 0:
                raise ValueError("class id must be a non-negative integer")
            class_id = int(identifier)
            if class_id not in names:
                raise ValueError("class id is absent from names")
            cx, cy, box_width, box_height = map(float, values[1:])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, box_width, box_height)) or min(box_width, box_height) <= 0:
                raise ValueError("invalid normalized box")
            x1, y1, x2, y2 = cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2
            if min(x1, y1) < -1e-5 or max(x2, y2) > 1 + 1e-5:
                raise ValueError("box extends beyond image")
            annotations.append({
                "class_id": class_id, "label": names[class_id], "canonical_class": canonical_label(names[class_id]),
                "bbox": [max(0, x1) * width, max(0, y1) * height, min(1, x2) * width, min(1, y2) * height],
                "annotation_line": line_number,
            })
    except (UnicodeError, ValueError, OverflowError) as error:
        raise EvaluationError(f"Anotação YOLO inválida para {image.name}; confira IDs e caixas normalizadas.") from error
    return annotations


def _iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _metrics(counts: dict[str, int]) -> dict[str, int | float | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {**counts, "ground_truth": tp + fn, "predictions": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def _latency(values: list[float]) -> dict[str, int | float | None]:
    return {"samples": len(values), "mean": float(np.mean(values)) if values else None,
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "total": float(sum(values))}


def _runtime_metadata(pipeline: Any) -> dict[str, Any]:
    stages = {}
    for name in ("person_detector", "ppe_detector", "detector"):
        detector = getattr(pipeline, name, None)
        if detector is None or (name == "detector" and "person_detector" in stages):
            continue
        configuration = getattr(detector, "config", None)
        stages[name] = {"device": str(getattr(detector, "device", "unknown"))}
        for key in ("model_path", "device", "confidence", "iou", "imgsz"):
            value = getattr(configuration, key, None)
            if isinstance(value, (str, int, float, bool)):
                stages[name]["configured_" + key] = value
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "pipeline": type(pipeline).__name__, "stages": stages}


def _output_path(output: str | Path | None, protected: set[Path], pipeline: Any) -> Path | None:
    """Keep reports from replacing dataset inputs or the evaluated models."""
    if output is None:
        return None
    destination = Path(output).expanduser().resolve()
    for name in ("person_detector", "ppe_detector", "detector"):
        configuration = getattr(getattr(pipeline, name, None), "config", None)
        model_path = getattr(configuration, "model_path", None)
        if model_path:
            model = Path(model_path).expanduser().resolve()
            if model.is_dir() and destination.is_relative_to(model):
                raise EvaluationError("O relatório não pode substituir arquivos do modelo avaliado.")
            protected.add(model)
            if model.suffix.lower() == ".xml":
                protected.add(model.with_suffix(".bin"))
    if destination in protected or (destination.exists() and any(
            path.exists() and os.path.samefile(destination, path) for path in protected)):
        raise EvaluationError("O relatório não pode substituir arquivos do dataset ou do modelo avaliado.")
    return destination


def evaluate_cascade(pipeline: Any, data: str | Path, split: str = "test", confidence: float = .4,
                     iou: float = .45, match_iou: float = .5,
                     output: str | Path | None = "runs/cascade-evaluation.json",
                     max_images: int | None = None) -> dict[str, Any]:
    """Score final boxes with confidence-ordered, per-class one-to-one matching.

    Input timing excludes image I/O and includes the first (possibly cold)
    pipeline call. Missing YOLO label files count as background and are listed
    explicitly for dataset review. Undefined precision/recall are JSON null.
    """
    confidence = validate_threshold(confidence, "Confiança")
    iou = validate_threshold(iou, "IoU de inferência")
    match_iou = validate_threshold(match_iou, "IoU de pareamento")
    if match_iou <= 0:
        raise EvaluationError("IoU de pareamento precisa ser maior que zero.")
    if split not in ("train", "val", "test"):
        raise EvaluationError("Split deve ser train, val ou test.")
    if max_images is not None and (type(max_images) is not int or max_images < 1):
        raise EvaluationError("max_images deve ser um inteiro positivo ou None.")
    source, root, names, all_images, protected = _dataset(data, split)
    destination = _output_path(output, protected, pipeline)
    declared = {canonical_label(label) for label in names.values()}
    classes = [label for label in CANONICAL_CLASSES if label in declared]
    if not classes:
        raise EvaluationError("O dataset não declara nenhuma classe canônica avaliável.")
    images = all_images[:max_images] if max_images is not None else all_images
    totals = {label: {"tp": 0, "fp": 0, "fn": 0} for label in classes}
    ignored_predictions, ignored_annotations = Counter(), Counter()
    missing_labels, empty_labels = [], []
    errors, errors_total = [], 0
    timings, stage_timings = [], defaultdict(list)
    background_images, images_without_scored_labels, low_confidence_predictions = 0, 0, 0
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps({"names": names, "split": split}, sort_keys=True).encode())
    started = perf_counter()

    def record_error(kind, image, detection):
        nonlocal errors_total
        errors_total += 1
        if len(errors) < MAX_ERROR_RECORDS:
            errors.append({"type": kind, "image": str(image), **detection})

    for image in images:
        label_path = _label_path(image)
        try:
            image_bytes = image.read_bytes()
            label_bytes = label_path.read_bytes() if label_path.is_file() else None
        except OSError as error:
            raise EvaluationError(f"Não foi possível ler imagem/anotação: {image}") from error
        relative = image.relative_to(root).as_posix() if image.is_relative_to(root) else image.as_posix()
        fingerprint.update(json.dumps({"image": relative, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                                       "label_sha256": hashlib.sha256(label_bytes).hexdigest() if label_bytes is not None else None}, sort_keys=True).encode())
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or not frame.size:
            raise EvaluationError(f"Imagem corrompida ou não decodificável: {image}")
        annotations = _ground_truth(label_bytes, names, frame.shape[1], frame.shape[0], image)
        if label_bytes is None:
            missing_labels.append(str(label_path))
        elif not annotations:
            empty_labels.append(str(label_path))
        background_images += int(not annotations)
        truths = [item for item in annotations if item["canonical_class"] in classes]
        images_without_scored_labels += int(not truths)
        ignored_annotations.update(item["label"] for item in annotations if item["canonical_class"] not in classes)
        call_started = perf_counter()
        result = pipeline.process(frame, confidence=confidence, iou=iou)
        timings.append((perf_counter() - call_started) * 1000)
        for stage, elapsed in getattr(result, "stage_timings_ms", {}).items():
            if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
                stage_timings[str(stage)].append(float(elapsed))
        predictions = []
        for detection in result.detections:
            try:
                score = float(detection.confidence)
                bbox = [float(value) for value in detection.bbox]
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise EvaluationError("A cascata retornou uma detecção com confiança/caixa inválida.") from error
            if not math.isfinite(score) or not 0 <= score <= 1 or len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise EvaluationError("A cascata retornou uma detecção com confiança/caixa inválida.")
            if score < confidence:
                low_confidence_predictions += 1
                continue
            label = canonical_label(detection.label)
            if label not in classes:
                ignored_predictions[detection.label] += 1
                continue
            predictions.append({"label": detection.label, "canonical_class": label, "confidence": score, "bbox": bbox})
        for label in classes:
            remaining = [item for item in truths if item["canonical_class"] == label]
            for prediction in sorted((item for item in predictions if item["canonical_class"] == label), key=lambda item: item["confidence"], reverse=True):
                overlaps = [_iou(prediction["bbox"], annotation["bbox"]) for annotation in remaining]
                if overlaps and max(overlaps) >= match_iou:
                    del remaining[int(np.argmax(overlaps))]
                    totals[label]["tp"] += 1
                else:
                    totals[label]["fp"] += 1
                    record_error("fp", image, prediction)
            totals[label]["fn"] += len(remaining)
            for annotation in remaining:
                record_error("fn", image, annotation)
    micro = {key: sum(counts[key] for counts in totals.values()) for key in ("tp", "fp", "fn")}
    report = {
        "kind": "cascade_fixed_threshold_detection_evaluation", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"data": str(source), "split": split, "confidence": confidence, "iou": iou,
                   "match_iou": match_iou, "max_images": max_images, "sample_selection": "sorted paths, first N; no shuffle",
                   "error_record_limit": MAX_ERROR_RECORDS},
        "dataset": {"source": "external_dataset", "training_overlap": "unknown", "held_out_verified": False,
                    "available_images": len(all_images), "evaluated_images": len(images), "names": names,
                    "evaluated_set_sha256": fingerprint.hexdigest(),
                    "hash_scope": "names, split and ordered evaluated image/label content hashes; missing labels marked explicitly",
                    "background_images": background_images, "images_without_scored_labels": images_without_scored_labels,
                    "missing_label_files": missing_labels, "empty_label_files": empty_labels},
        "coverage": {"scored_classes": classes, "unannotated_canonical_classes": [label for label in CANONICAL_CLASSES if label not in classes],
                     "ignored_dataset_classes": {identifier: label for identifier, label in names.items() if canonical_label(label) not in classes},
                     "ignored_predictions": {"total": sum(ignored_predictions.values()), "by_label": dict(ignored_predictions)},
                     "ignored_ground_truth": {"total": sum(ignored_annotations.values()), "by_label": dict(ignored_annotations)},
                     "predictions_below_confidence": low_confidence_predictions},
        "per_class": {label: _metrics(counts) for label, counts in totals.items()}, "micro": _metrics(micro),
        "errors": errors, "errors_total": errors_total, "errors_truncated": errors_total > len(errors),
        "pipeline_latency_ms": _latency(timings),
        "stage_latency_ms": {stage: _latency(values) for stage, values in stage_timings.items()},
        "evaluation_wall_time_ms": (perf_counter() - started) * 1000,
        "environment": _runtime_metadata(pipeline),
        "limitations": [
            "Fixed-threshold box precision/recall, not AP or mAP; no confidence sweep.",
            "Only classes declared in YAML are scored; declaration does not prove exhaustive annotation.",
            "Missing label files are treated as background; review the explicit list before interpreting false positives.",
            "Per-person PPE states and equipment-person associations are not evaluated from isolated box labels.",
            "Overlap with model training data is unknown; this report does not establish an independent held-out test.",
            "Latency includes the first pipeline call, excludes image I/O and depends on the recorded hardware.",
        ],
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(destination)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return report
