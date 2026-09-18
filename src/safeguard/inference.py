"""Adaptador YOLO com imports tardios e saídas independentes do backend."""

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import InferenceConfig, validate_threshold
from .hardware import select_device
from .preprocessing import preprocess_frame
from .types import Detection

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """Falha recuperável de configuração, carregamento ou inferência."""


def resolve_model_path(model_path: str | Path) -> Path:
    """Ultralytics 8.3.203 recognizes OpenVINO by directory suffix, not .xml."""
    path = Path(model_path).expanduser()
    if path.suffix.lower() == ".xml" or path.is_dir():
        directory = path.parent if path.suffix.lower() == ".xml" else path
        if not directory.name.endswith("_openvino_model"):
            raise ModelError("Mantenha o modelo OpenVINO em um diretório terminado em '_openvino_model', com .xml, .bin e metadata.yaml.")
        xml_files = list(directory.glob("*.xml"))
        if len(xml_files) != 1 or not xml_files[0].with_suffix(".bin").is_file():
            raise ModelError("O diretório OpenVINO precisa de exatamente um par .xml e .bin.")
        return directory.resolve()
    return path.resolve()


def exported_input_shape(model_path: str | Path) -> tuple[int, int, int] | None:
    """Return fixed (batch, height, width), or None for flexible/native inputs."""
    path = resolve_model_path(model_path)
    if path.suffix.lower() == ".pt":
        return None
    if path.suffix.lower() == ".onnx":
        onnx = importlib.import_module("onnx")
        graph = onnx.load(str(path), load_external_data=False).graph
        dims = graph.input[0].type.tensor_type.shape.dim
        if len(dims) == 4 and all(dims[index].dim_value > 0 for index in (0, 2, 3)):
            return int(dims[0].dim_value), int(dims[2].dim_value), int(dims[3].dim_value)
        return None
    metadata = {}
    if path.is_dir():
        import yaml

        metadata_path = path / "metadata.yaml"
        if metadata_path.is_file():
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    elif path.suffix.lower() == ".engine":
        # Engines exported by this Ultralytics version carry a JSON header.
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(4), byteorder="little", signed=True)
            if 0 < length <= 10_000_000:
                try:
                    metadata = json.loads(stream.read(length).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
    if not isinstance(metadata, dict) or metadata.get("args", {}).get("dynamic"):
        return None
    shape = metadata.get("imgsz")
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return int(metadata.get("batch", 1)), int(shape[0]), int(shape[1])
    return None


class YOLODetector:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = "cpu"
        self.names: dict[int, str] = {}
        self.imgsz: int | tuple[int, int] = config.imgsz
        self._model: Any = None

    def _check_dependencies(self, path: Path) -> None:
        required = []
        if path.suffix.lower() == ".onnx":
            required = [("onnx", "onnx"), ("onnxruntime", "onnxruntime (CPU) ou onnxruntime-gpu (CUDA)")]
        elif path.suffix.lower() == ".engine":
            required = [("tensorrt", "TensorRT compatível com seu CUDA e com o engine exportado")]
        elif path.suffix.lower() == ".xml" or path.is_dir():
            required = [("openvino", "openvino>=2024.0.0")]
            xml = path if path.is_file() else next(path.glob("*.xml"), None)
            if xml is None or not xml.with_suffix(".bin").is_file():
                raise ModelError("O modelo OpenVINO precisa dos arquivos .xml e .bin no mesmo diretório.")
        for module, package in required:
            try:
                importlib.import_module(module)
            except (ImportError, OSError, RuntimeError) as exc:
                raise ModelError(f"Dependência de inferência ausente ou incompatível: {package}. Instale o backend antes de abrir este modelo.") from exc

    def _can_fallback(self, exc: Exception) -> bool:
        supported = Path(self.config.model_path).suffix.lower() in {".pt", ".onnx"}
        gpu_failure = any(term in str(exc).lower() for term in ("cuda", "cudnn", "cublas", "mps", "metal", "out of memory"))
        return self.device != "cpu" and supported and gpu_failure

    def _backend_device(self) -> str:
        path = Path(self.config.model_path)
        # Ultralytics interprets plain "cpu" as OpenVINO AUTO. This explicit
        # Intel device keeps the actual runtime consistent with our selection.
        return "intel:cpu" if path.suffix.lower() == ".xml" or path.is_dir() else self.device

    def _build_model(self) -> Any:
        try:
            module = importlib.import_module("ultralytics")
        except (ImportError, OSError) as exc:
            raise ModelError("Ultralytics/PyTorch indisponível. Execute o setup ou instale requirements.txt.") from exc
        path = resolve_model_path(self.config.model_path)
        model = module.YOLO(str(path), task="detect")
        # Accessing names for exported models initializes AutoBackend. Supply
        # the chosen device before this happens, rather than letting it guess.
        if hasattr(model, "overrides"):
            model.overrides.update({"device": self._backend_device(), "imgsz": self.config.imgsz})
        if path.suffix.lower() != ".pt":
            shape = exported_input_shape(path)
            if shape is not None:
                batch, height, width = shape
                if batch != 1:
                    raise ModelError(f"O pipeline processa um frame por vez; reexporte o modelo com --batch 1 (artefato atual: {batch}).")
                self.imgsz = height if height == width else (height, width)
                if self.imgsz != self.config.imgsz:
                    logger.warning("Modelo exportado tem entrada fixa %sx%s; usando esse tamanho em vez de imgsz=%s.", height, width, self.config.imgsz)
                model.overrides["imgsz"] = self.imgsz
            # Initialize once: model.names otherwise builds a temporary backend,
            # and subsequent predict calls would initialize the runtime again.
            predictor = model._smart_load("predictor")(overrides=model.overrides, _callbacks=model.callbacks)
            predictor.setup_model(model=model.model, verbose=False)
            model.predictor = predictor
        names = model.names
        self.names = {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        return model

    def load(self) -> "YOLODetector":
        if self._model is not None:
            return self
        path = Path(self.config.model_path)
        if not path.exists():
            raise ModelError(f"Modelo não encontrado: {path}. Baixe os pesos de demonstração ou informe seus pesos de EPI.")
        if not path.is_dir() and path.suffix.lower() not in {".pt", ".onnx", ".engine", ".xml"}:
            raise ModelError("Formato de modelo inválido. Use .pt, .onnx, .engine ou um diretório OpenVINO.")
        try:
            self._check_dependencies(path)
            self.device = select_device(self.config.device, self.config.model_path)
            try:
                self._model = self._build_model()
            except Exception as exc:
                if not self._can_fallback(exc):
                    raise
                logger.warning("Falha ao inicializar acelerador; carregando o modelo em CPU: %s", exc)
                self.device = "cpu"
                self._model = self._build_model()
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Não foi possível carregar o modelo: {exc}") from exc
        return self

    def _predict_raw(self, frame: np.ndarray, confidence: float, iou: float) -> Any:
        precision = {"half": self.device.startswith("cuda:")} if Path(self.config.model_path).suffix.lower() == ".pt" else {}
        return self._model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            imgsz=self.imgsz,
            device=self._backend_device(),
            verbose=False,
            stream=False,
            **precision,
        )

    def predict(self, frame: np.ndarray, confidence: float | None = None, iou: float | None = None) -> list[Detection]:
        prepared = preprocess_frame(frame)
        conf = validate_threshold(self.config.confidence if confidence is None else confidence, "Confiança")
        overlap = validate_threshold(self.config.iou if iou is None else iou, "IoU")
        self.load()
        try:
            raw = self._predict_raw(prepared, conf, overlap)
        except Exception as exc:
            # A fresh adapter avoids reusing a predictor whose backend still owns
            # GPU tensors. Do not pretend a TensorRT engine supports CPU fallback.
            if self._can_fallback(exc):
                logger.warning("Falha no acelerador; tentando este modelo em CPU: %s", exc)
                try:
                    self.device = "cpu"
                    self._model = self._build_model()
                    raw = self._predict_raw(prepared, conf, overlap)
                except Exception as fallback_exc:
                    raise ModelError(f"Inferência também falhou em CPU: {fallback_exc}") from fallback_exc
            else:
                raise ModelError(f"Falha de inferência: {exc}") from exc
        try:
            return self._convert(raw, prepared.shape[1], prepared.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            raise ModelError(f"Saída incompatível com detecção de objetos: {exc}") from exc

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value.numpy() if hasattr(value, "numpy") else value)

    def _convert(self, results: Any, width: int, height: int) -> list[Detection]:
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        coords = self._array(boxes.xyxy).reshape(-1, 4)
        scores = self._array(boxes.conf).reshape(-1)
        classes = self._array(boxes.cls).reshape(-1)
        if not len(coords) == len(scores) == len(classes):
            raise ValueError("As quantidades de caixas, classes e confianças diferem.")
        detections: list[Detection] = []
        for bbox, score, class_value in zip(coords, scores, classes):
            if not np.isfinite(bbox).all() or not np.isfinite(score) or not np.isfinite(class_value):
                continue
            if not 0 <= score <= 1 or class_value < 0 or int(class_value) != class_value:
                continue
            clipped = np.clip(bbox, [0, 0, 0, 0], [width, height, width, height])
            x1, y1, x2, y2 = map(float, clipped)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_value)
            detections.append(Detection(class_id, self.names.get(class_id, str(class_id)), float(score), (x1, y1, x2, y2)))
        return detections
