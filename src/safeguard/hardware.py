"""Seleção conservadora de hardware por formato de modelo."""

import importlib
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class HardwareError(RuntimeError):
    """O artefato não pode executar no hardware/dependências presentes."""


def _torch_capabilities() -> tuple[bool, bool, int]:
    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda else 0
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        mps = bool(mps_backend and mps_backend.is_available())
        return cuda, mps, count
    except (ImportError, OSError, RuntimeError):
        return False, False, 0


def _onnx_cuda_available() -> bool:
    try:
        runtime = importlib.import_module("onnxruntime")
        return "CUDAExecutionProvider" in runtime.get_available_providers()
    except (ImportError, OSError, RuntimeError):
        return False


def select_device(requested: str = "auto", model_path: str | None = None) -> str:
    """Return an Ultralytics device, falling back only for compatible artifacts.

    TensorRT engines are CUDA-only. OpenVINO uses CPU here. ONNX cannot use
    PyTorch MPS and CUDA requires both PyTorch CUDA and ONNX Runtime GPU.
    """
    choice = requested.strip().lower()
    if choice == "cuda":
        choice = "cuda:0"
    elif choice.isdecimal():
        choice = f"cuda:{int(choice)}"
    if choice not in {"auto", "cpu", "mps"} and not re.fullmatch(r"cuda:\d+", choice):
        raise ValueError("Dispositivo inválido. Use auto, cpu, cuda:0 ou mps.")
    path = Path(model_path) if model_path else None
    suffix = path.suffix.lower() if path else ".pt"
    openvino = bool(path and (suffix == ".xml" or path.name.endswith("_openvino_model") or path.is_dir()))
    cuda, mps, count = _torch_capabilities()
    if suffix == ".engine":
        if choice in {"cpu", "mps"} or not cuda:
            raise HardwareError("TensorRT exige GPU NVIDIA CUDA. Use o modelo .pt ou .onnx para CPU/MPS.")
        index = int(choice.split(":")[1]) if choice.startswith("cuda:") else 0
        if index >= count:
            raise HardwareError(f"GPU CUDA {index} indisponível para o engine TensorRT.")
        return f"cuda:{index}"
    if openvino:
        if choice not in {"auto", "cpu"}:
            logger.warning("Este backend OpenVINO usa CPU; selecionando CPU.")
        return "cpu"
    if suffix == ".onnx":
        cuda = cuda and _onnx_cuda_available()
        mps = False
    if choice == "auto":
        return "cuda:0" if cuda else "mps" if mps else "cpu"
    if choice == "cpu":
        return "cpu"
    available = mps if choice == "mps" else cuda and int(choice.split(":")[1]) < count
    if available:
        return choice
    logger.warning("Dispositivo %s indisponível para este modelo; selecionando CPU.", choice)
    return "cpu"
