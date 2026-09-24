"""Explicit, reproducible training, evaluation, export and timing workflows."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import platform
import shutil
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from safeguard.hardware import select_device
from safeguard.dataset_audit import audit_dataset
from safeguard.inference import ModelError, exported_input_shape, resolve_model_path


class WorkflowError(RuntimeError):
    """Actionable failure in an ML workflow."""


def require_optional(modules: list[str], install: str) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise WorkflowError(f"Dependências ausentes: {', '.join(missing)}. Execute: {install}")


def load_yolo(model_path: str) -> Any:
    """Never implicitly download weights during a workflow."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise WorkflowError(
            f"Modelo não encontrado: {path}. Use 'python -m safeguard download' "
            "para a demonstração COCO ou informe pesos de EPI treinados."
        )
    try:
        path = resolve_model_path(path)
    except ModelError as error:
        raise WorkflowError(str(error)) from error
    require_optional(["ultralytics"], 'python -m pip install -e "."')
    if path.suffix.lower() == ".onnx":
        require_optional(["onnx", "onnxruntime"], 'python -m pip install -e ".[onnx]"')
    elif path.is_dir() or path.suffix.lower() == ".xml":
        require_optional(["openvino"], 'python -m pip install -e ".[openvino]"')
    elif path.suffix.lower() == ".engine":
        require_optional(["tensorrt"], "instale TensorRT compatível com sua versão CUDA; consulte docs/ML.md")
    from ultralytics import YOLO

    return YOLO(str(path.resolve()), task="detect")


def environment(device: str) -> dict[str, Any]:
    versions = {}
    for package in ("ultralytics", "torch", "numpy", "onnxruntime", "openvino", "nncf", "tensorrt"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device,
        "versions": versions,
    }
    if str(device).startswith("cuda"):
        import torch

        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_value(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return destination


@contextmanager
def normalized_dataset(
    data: str, *, splits: tuple[str, ...] = ("train", "val"), directory: Path | None = None,
    resolved_splits: dict[str, list[str]] | None = None,
) -> Iterator[str]:
    """Resolve dataset paths relative to its YAML, independent of global YOLO settings."""
    import yaml

    source = Path(data).expanduser().resolve()
    if not source.is_file():
        raise WorkflowError(f"Dataset YAML não encontrado: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document.get("names"):
        raise WorkflowError("O dataset YAML deve definir um mapeamento/lista 'names' com as classes reais.")
    if "download" in document:
        raise WorkflowError("Remova 'download' do YAML e prepare o dataset localmente antes de continuar.")
    root = Path(document.get("path") or source.parent).expanduser()
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    document["path"] = str(root)
    for split in ("train", "val", "test"):
        entries = document.get(split)
        if not entries:
            if split in splits:
                raise WorkflowError(f"Dataset precisa definir o split '{split}'.")
            continue
        paths = entries if isinstance(entries, list) else [entries]
        resolved = []
        for entry in paths:
            path = Path(entry).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            if split in splits and not path.exists():
                raise WorkflowError(f"Split '{split}' não encontrado: {path}. Configure seu dataset real.")
            resolved.append(str(path))
        document[split] = resolved if isinstance(entries, list) else resolved[0]
    # Both train and val keys are required by the downstream detection schema.
    if "train" not in document or "val" not in document:
        raise WorkflowError("Defina 'train' e 'val' no YAML; use dados distintos para treino e avaliação.")
    def prepare(destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            if resolved_splits is not None and split in resolved_splits:
                manifest = destination / f"{split}.txt"
                manifest.write_text("".join(f"{image}\n" for image in resolved_splits[split]), encoding="utf-8")
                document[split] = str(manifest.resolve())
            elif split in document:
                # Ultralytics otherwise interprets most relative TXT entries
                # against the process working directory, not their manifest.
                entries = document[split] if isinstance(document[split], list) else [document[split]]
                normalized_entries = []
                for index, entry in enumerate(entries):
                    original = Path(entry)
                    if original.is_file() and original.suffix.lower() == ".txt":
                        lines = []
                        for line in original.read_text(encoding="utf-8-sig").splitlines():
                            if line.strip() and not line.lstrip().startswith("#"):
                                image = Path(line.strip()).expanduser()
                                lines.append(str((image if image.is_absolute() else original.parent / image).resolve()))
                        manifest = destination / f"{split}-{index}.txt"
                        manifest.write_text("".join(f"{image}\n" for image in lines), encoding="utf-8")
                        normalized_entries.append(str(manifest.resolve()))
                    else:
                        normalized_entries.append(entry)
                document[split] = normalized_entries if isinstance(document[split], list) else normalized_entries[0]
        normalized = destination / "dataset.resolved.yaml"
        normalized.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
        return str(normalized.resolve())

    if directory is not None:
        yield prepare(directory)
    else:
        with tempfile.TemporaryDirectory(prefix="safeguard-data-") as temporary:
            yield prepare(Path(temporary))


def _reserve_run(project: str, name: str) -> Path:
    """Own a fresh run directory so audit/manifests exist before model loading."""
    parent = Path(project).expanduser().resolve()
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise WorkflowError("name deve ser um nome de execução, sem diretórios.")
    parent.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, 100000):
        directory = parent / (name if suffix == 1 else f"{name}{suffix}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue
    raise WorkflowError("Não foi possível reservar uma pasta nova para a execução.")


def _audited_run(data: str, project: str, name: str, *, require_test: bool = False) -> tuple[Path, dict]:
    directory = _reserve_run(project, name)
    report = audit_dataset(data, require_test=require_test, output=directory / "dataset-audit.json")
    if not report["valid"]:
        reasons = "; ".join(item["message"] for item in report["errors"][:3])
        raise WorkflowError(f"Auditoria do dataset falhou: {reasons} Relatório: {report['report_path']}")
    return directory, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"path": str(artifact), "sha256": None}
    if artifact.is_file():
        result["sha256"] = _sha256(artifact)
        if artifact.suffix.lower() == ".xml" and artifact.with_suffix(".bin").is_file():
            result["weights_bin_sha256"] = _sha256(artifact.with_suffix(".bin"))
    elif artifact.is_dir():
        files = {str(item.relative_to(artifact)).replace("\\", "/"): _sha256(item)
                 for item in sorted(artifact.rglob("*")) if item.is_file()}
        result["files"] = files
        result["sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
        result["hash_scope"] = "SHA256 do JSON ordenado de caminhos relativos e hashes dos arquivos."
    return result


def _dataset_provenance(data: str, normalized: str, audit: dict, directory: Path) -> dict:
    return {"dataset_yaml": {"path": str(Path(data).expanduser().resolve()),
                             "sha256": audit["dataset_yaml_sha256"]},
            "resolved_dataset": _fingerprint(normalized),
            "dataset_audit": str(directory / "dataset-audit.json"),
            "split_manifests": {split: _fingerprint(directory / f"{split}.txt")
                                for split in audit["resolved_splits"]}}


def _assert_class_mapping(model: Any, names: dict, device: str, imgsz: int) -> dict[int, str]:
    # Exported backends may initialize lazily when reading names. Never inject
    # dataset names: those would conceal a wrong checkpoint/class ordering.
    if isinstance(getattr(model, "overrides", None), dict):
        model.overrides.update({"device": device, "imgsz": imgsz})
    expected = {int(key): str(value) for key, value in names.items()}
    try:
        actual_names = model.names
        actual = ({int(key): str(value) for key, value in actual_names.items()}
                  if isinstance(actual_names, dict) else dict(enumerate(actual_names)))
    except (AttributeError, TypeError, ValueError) as error:
        raise WorkflowError("O modelo não expõe nomes de classes verificáveis; exporte com metadados antes de avaliar.") from error
    if actual != expected:
        raise WorkflowError(
            f"Classes do modelo incompatíveis com o dataset (IDs, nomes e ordem devem coincidir). "
            f"Modelo: {actual}. Dataset: {expected}. Não renomeie classes para mascarar pesos incompatíveis."
        )
    return actual


def _per_class_metrics(metrics: Any, names: dict, audit: dict, split: str) -> list[dict]:
    box = getattr(metrics, "box", None)
    indices = list(getattr(box, "ap_class_index", []))
    columns = {key: list(getattr(box, attribute, []))
               for key, attribute in (("precision", "p"), ("recall", "r"), ("mAP50", "ap50"), ("mAP50_95", "ap"))}
    rows = []
    counts = audit["splits"][split]["class_counts"]
    for key, name in sorted(names.items(), key=lambda item: int(item[0])):
        identifier = int(key)
        position = indices.index(identifier) if identifier in indices else None
        row = {"class_id": identifier, "name": name, "instances": counts[str(identifier)]["instances"],
               "evaluated": position is not None}
        for metric, values in columns.items():
            row[metric] = float(values[position]) if position is not None and position < len(values) else None
        rows.append(row)
    return rows


def train_model(
    model_path: str, data: str, *, device: str = "auto", epochs: int = 50,
    imgsz: int = 640, batch: int = 8, workers: int = 0, seed: int = 42,
    project: str = "runs/train", name: str = "ppe", amp: bool = False,
) -> dict[str, Any]:
    if epochs < 1 or batch < 1 or workers < 0:
        raise WorkflowError("epochs e batch devem ser positivos; workers deve ser >= 0.")
    if Path(model_path).suffix.lower() != ".pt":
        raise WorkflowError("Treinamento requer pesos PyTorch .pt, não um modelo exportado.")
    directory, audit = _audited_run(data, project, name)
    selected = select_device(device, model_path=model_path)
    with normalized_dataset(data, directory=directory, resolved_splits=audit["resolved_splits"]) as normalized:
        provenance = _dataset_provenance(data, normalized, audit, directory)
        provenance["initial_weights"] = _fingerprint(model_path)
        model = load_yolo(model_path)
        metrics = model.train(
            data=normalized, device=selected, epochs=epochs, imgsz=imgsz, batch=batch,
            workers=workers, seed=seed, deterministic=True, project=str(directory.parent), name=directory.name,
            plots=True, exist_ok=True, amp=amp,
        )
    actual_directory = Path(model.trainer.save_dir).resolve()
    if actual_directory != directory:
        raise WorkflowError(f"O treinador alterou a pasta reservada da execução: {actual_directory}; os manifests estão em {directory}.")
    report = {
        "kind": "training", "environment": environment(selected),
        "config": {"model": model_path, "data": str(Path(data).resolve()), "epochs": epochs,
                   "imgsz": imgsz, "batch": batch, "workers": workers, "seed": seed, "amp": amp},
        "metrics": getattr(metrics, "results_dict", {}),
        "per_class": _per_class_metrics(metrics, audit["class_names"], audit, "val"),
        "provenance": provenance,
        "best_weights": str(directory / "weights" / "best.pt"),
        "last_weights": str(directory / "weights" / "last.pt"),
    }
    provenance["best_weights"] = _fingerprint(report["best_weights"])
    provenance["last_weights"] = _fingerprint(report["last_weights"])
    report["report_path"] = str((directory / "training.json").resolve())
    write_json(report["report_path"], report)
    return report


def validate_model(
    model_path: str, data: str, *, device: str = "auto", imgsz: int = 640,
    batch: int = 1, split: str = "val", confidence: float = 0.001, iou: float = 0.7,
    project: str = "runs/validate", name: str = "ppe", output: str | None = None,
) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise WorkflowError("Avaliação exige split val ou test; treino não é estimativa de generalização.")
    directory, audit = _audited_run(data, project, name, require_test=split == "test")
    selected = select_device(device, model_path=model_path)
    artifact = Path(model_path)
    # Ultralytics maps plain "cpu" to OpenVINO AUTO, which may use Intel GPU.
    openvino = artifact.is_dir() or artifact.suffix.lower() == ".xml" or artifact.name.endswith("_openvino_model")
    backend_device = "intel:cpu" if openvino else selected
    requested_imgsz, requested_batch = imgsz, batch
    with normalized_dataset(data, splits=(split,), directory=directory, resolved_splits=audit["resolved_splits"]) as normalized:
        provenance = _dataset_provenance(data, normalized, audit, directory)
        provenance["weights"] = _fingerprint(model_path)
        model = load_yolo(model_path)
        shape = exported_input_shape(model_path)
        if shape is not None:
            batch, height, width = shape
            if height != width:
                raise WorkflowError("Validação Ultralytics exige uma exportação quadrada; reexporte com --imgsz 640, por exemplo.")
            imgsz = height
        class_names = _assert_class_mapping(model, audit["class_names"], backend_device, imgsz)
        confusion_data: dict[str, Any] = {}

        def capture_confusion(validator: Any) -> None:
            confusion = getattr(validator, "confusion_matrix", None)
            if confusion is not None:
                confusion_data["matrix"] = getattr(confusion, "matrix", None)

        model.add_callback("on_val_end", capture_confusion)
        metrics = model.val(
            data=normalized, device=backend_device, imgsz=imgsz, batch=batch, split=split,
            conf=confidence, iou=iou, plots=True, save_json=True, workers=0,
            project=str(directory.parent), name=directory.name, exist_ok=True,
        )
    if Path(metrics.save_dir).resolve() != directory:
        raise WorkflowError(f"O validador alterou a pasta reservada da execução: {metrics.save_dir}; os manifests estão em {directory}.")
    box = metrics.box
    report = {
        "kind": "validation", "environment": environment(selected),
        "config": {"model": model_path, "data": str(Path(data).resolve()), "imgsz": imgsz,
                   "batch": batch, "requested_imgsz": requested_imgsz, "requested_batch": requested_batch,
                   "split": split, "confidence": confidence, "iou": iou},
        "metrics": {**getattr(metrics, "results_dict", {}),
                    "mAP50": float(box.map50), "mAP50_95": float(box.map),
                    "precision": float(box.mp), "recall": float(box.mr)},
        "per_class": _per_class_metrics(metrics, class_names, audit, split),
        "provenance": provenance,
        "speed_ms_per_image": getattr(metrics, "speed", {}),
        "class_names": class_names,
        "artifacts": {path.name: str(path.resolve()) for path in directory.glob("*")
                      if path.is_file() and path.suffix.lower() in {".png", ".json", ".csv"}},
    }
    # curves_results contains [x, y, x_label, y_label] for PR/F1/P/R curves.
    curves = getattr(metrics, "curves_results", None)
    if curves:
        report["curves"] = dict(zip(getattr(metrics, "curves", []), curves))
    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is not None:
        report["confusion_matrix"] = getattr(confusion, "matrix", None)
    elif confusion_data:
        report["confusion_matrix"] = confusion_data["matrix"]
    report["report_path"] = str(Path(output or directory / "validation.json").resolve())
    write_json(report["report_path"], report)
    return _json_value(report)


def check_tensorrt_precision(precision: str, device: str) -> None:
    """Reject the exporter's silent INT8/FP16 downgrade on unsupported GPUs."""
    if precision == "fp32":
        return
    import tensorrt as trt
    import torch

    capability = "platform_has_fast_int8" if precision == "int8" else "platform_has_fast_fp16"
    with torch.cuda.device(device):
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        if not getattr(builder, capability):
            raise WorkflowError(f"Esta GPU/TensorRT não oferece {precision.upper()} acelerado. Use --precision fp32; o exportador não fará conversão silenciosa.")


def export_model(
    model_path: str, *, format: str = "onnx", precision: str = "fp32",
    data: str | None = None, device: str = "auto", imgsz: int = 640,
    fraction: float = 1.0, batch: int = 1,
) -> dict[str, Any]:
    if format not in {"onnx", "openvino", "engine"}:
        raise WorkflowError("Formato suportado: onnx, openvino ou engine (TensorRT).")
    if precision not in {"fp32", "fp16", "int8"}:
        raise WorkflowError("Precisão suportada: fp32, fp16 ou int8.")
    if precision == "int8" and format == "onnx":
        raise WorkflowError("Neste projeto ONNX exporta FP32/FP16. Para INT8 use OpenVINO ou TensorRT.")
    if precision == "int8" and not data:
        raise WorkflowError("INT8 exige --data com dataset representativo para calibração; nunca use COCO como substituto de EPI.")
    if not 0 < fraction <= 1 or batch < 1:
        raise WorkflowError("fraction deve estar em (0, 1] e batch deve ser positivo.")
    if Path(model_path).suffix.lower() != ".pt":
        raise WorkflowError("Exportação requer pesos PyTorch .pt.")
    calibration_cache = Path(model_path).expanduser().with_suffix(".cache")
    if format == "engine" and precision == "int8" and calibration_cache.exists():
        raise WorkflowError(f"Cache de calibração existente: {calibration_cache}. Mova-o ou remova-o antes de reexportar para garantir calibração com o dataset solicitado.")
    selected = select_device(device, model_path=model_path)
    if format == "engine" and not str(selected).startswith("cuda"):
        raise WorkflowError("TensorRT exige GPU NVIDIA CUDA. Use --device cuda:0 ou exporte OpenVINO/ONNX para CPU.")
    if format == "onnx" and precision == "fp16" and not str(selected).startswith("cuda"):
        raise WorkflowError("ONNX FP16 no Ultralytics 8.3.203 exige CUDA; em CPU use --precision fp32.")
    if format == "openvino":
        selected = "cpu"
        require_optional(["openvino"] + (["nncf"] if precision == "int8" else []),
                         'python -m pip install -e ".[openvino]"')
    else:
        require_optional(["onnx"], 'python -m pip install -e ".[onnx]"')
    if format == "engine":
        require_optional(["tensorrt"], "instale o TensorRT compatível com sua versão CUDA conforme docs/ML.md")
        check_tensorrt_precision(precision, selected)
    kwargs = {"format": format, "imgsz": imgsz, "batch": batch,
              "device": selected, "half": precision == "fp16", "int8": precision == "int8"}
    # Avoid automatic onnxslim/onnxruntime installation by the exporter.
    if format in {"onnx", "engine"}:
        kwargs["simplify"] = False
    if precision == "int8":
        kwargs["fraction"] = fraction
        with normalized_dataset(data, splits=("val",)) as normalized:
            kwargs["data"] = normalized
            artifact = load_yolo(model_path).export(**kwargs)
    else:
        artifact = load_yolo(model_path).export(**kwargs)
    if not artifact or not Path(artifact).exists():
        raise WorkflowError("O exportador não produziu um artefato. Consulte o log anterior.")
    report = {
        "kind": "export", "environment": environment(selected),
        "config": {"model": model_path, "format": format, "precision": precision,
                   "imgsz": imgsz, "batch": batch, "calibration_data": data,
                   "calibration_fraction": fraction if precision == "int8" else None},
        "artifact": str(Path(artifact).resolve()),
        "accuracy": "Não avaliada. Execute validate no artefato e nos pesos originais sobre o mesmo split.",
    }
    write_json(str(artifact).rstrip("/\\") + ".export.json", report)
    return report


def synchronize(device: str) -> None:
    """GPU work must complete before an end-to-end wall timer stops."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
    elif device == "mps":
        import torch

        torch.mps.synchronize()


def latency_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise WorkflowError("Nenhum frame medido. Verifique a fonte e a quantidade de frames.")
    values = np.asarray(samples, dtype=np.float64)
    mean = float(values.mean())
    return {"mean_ms": mean, "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": 1000.0 / mean if mean > 0 else 0.0}


def benchmark_model(
    model_path: str, source: int | str, *, device: str = "auto", imgsz: int = 640,
    confidence: float = 0.4, iou: float = 0.45, frames: int = 100,
    warmup: int = 10, output: str = "runs/benchmark.json", ppe_model: str | None = None,
) -> dict[str, Any]:
    from safeguard.capture import VideoSource
    from safeguard.config import InferenceConfig
    from safeguard.inference import YOLODetector
    from safeguard.pipeline import Pipeline
    from safeguard.rendering import render_frame
    from safeguard.factory import create_cascade
    from safeguard.runner import _output_paths

    if frames < 1 or warmup < 0:
        raise WorkflowError("frames deve ser positivo e warmup deve ser >= 0.")
    destination, _ = _output_paths(source, model_path, ppe_model or model_path, output, None)
    if ppe_model is None:
        detector = YOLODetector(InferenceConfig(model_path=model_path, device=device,
                                              confidence=confidence, iou=iou, imgsz=imgsz)).load()
        pipeline = Pipeline(detector, demo_mode=True)
        detectors = [detector]
    else:
        pipeline = create_cascade(model_path, ppe_model, device, imgsz, confidence, iou)
        detector = pipeline.person_detector
        detectors = [detector, pipeline.ppe_detector]
    inference, processing, end_to_end = [], [], []
    person_times, ppe_times = [], []
    ppe_executed = 0
    with VideoSource(source) as video:
        frame = video.read()
        if frame is None:
            raise WorkflowError("A fonte não contém frames decodificáveis.")
        # Reuse the first frame to warm up without consuming a short video.
        for _ in range(warmup):
            render_frame(pipeline.process(frame))
        for stage_device in dict.fromkeys(stage.device for stage in detectors):
            synchronize(stage_device)
        for _ in range(frames):
            start = time.perf_counter()
            frame = video.read()
            if frame is None:
                break
            result = pipeline.process(frame)
            render_frame(result)
            for stage_device in dict.fromkeys(stage.device for stage in detectors):
                synchronize(stage_device)
            end_to_end.append((time.perf_counter() - start) * 1000)
            inference.append(result.inference_ms)
            processing.append(result.pipeline_ms)
            if ppe_model is not None:
                person_times.append(result.stage_timings_ms["person"])
                if result.ppe_executed:
                    ppe_times.append(result.stage_timings_ms["ppe"])
                    ppe_executed += 1
    report = {
        "kind": "cascade_benchmark" if ppe_model is not None else "benchmark", "environment": environment(detector.device),
        "config": {"model": model_path, "source_type": "camera" if isinstance(source, int) else "video_or_stream",
                   "source_file": str(Path(source).resolve()) if isinstance(source, str) and "://" not in source else None,
                   "imgsz": detector.imgsz, "requested_imgsz": imgsz,
                   "ppe_model": ppe_model,
                   "ppe_imgsz": pipeline.ppe_detector.imgsz if ppe_model is not None else None,
                   "ppe_device": pipeline.ppe_detector.device if ppe_model is not None else None,
                   "confidence": confidence, "iou": iou, "warmup": warmup,
                   "requested_frames": frames, "measured_frames": len(end_to_end)},
        "timings": {"detector_call": latency_summary(inference),
                    "pipeline": latency_summary(processing),
                    "capture_pipeline_render": latency_summary(end_to_end)},
        "scope": {"detector_call": "Chamada predict completa: pré-processamento, modelo e pós-processamento.",
                  "pipeline": "Processamento do frame e regras; exclui captura, renderização e UI.",
                  "capture_pipeline_render": "Captura/leitura/decodificação, pipeline e renderização; exclui UI, gravação de resultados e carregamento do modelo."},
        "accuracy": "Não medida. Benchmark de latência não estima mAP; use validate com dataset anotado.",
    }
    if ppe_model is not None:
        report["ppe_executed_frames"] = ppe_executed
        report["ppe_skipped_frames"] = len(end_to_end) - ppe_executed
        report["stage_timings"] = {"person": latency_summary(person_times),
                                   "ppe": latency_summary(ppe_times) if ppe_times else None}
        report["scope"]["detector_call"] = "Soma das chamadas predict de pessoa e EPI; EPI executa somente quando há pessoa."
        report["scope"]["stage_timings"] = "Person inclui todos os frames medidos; PPE inclui somente frames em que o segundo estágio executou, sem zeros de frames pulados."
    report["report_path"] = str(destination)
    write_json(destination, report)
    return report


OFFICIAL_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt")


def download_model(name: str = "yolo11n.pt", directory: str = "models") -> Path:
    if name not in OFFICIAL_MODELS:
        raise WorkflowError(f"Escolha pesos oficiais suportados: {', '.join(OFFICIAL_MODELS)}")
    target = Path(directory) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1024:
            raise WorkflowError("O download retornou um arquivo vazio ou inválido.")
        temporary.replace(target)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise WorkflowError(f"Falha ao baixar pesos oficiais. Verifique sua conexão: {error}") from error
    return target
