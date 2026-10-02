"""Choose available UI defaults without loading weights or changing saved paths."""

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

from epi_monitor.hardware import select_device


PYTORCH = "Padrão (PyTorch)"
OPENVINO = "CPU otimizada (OpenVINO)"
OPENVINO_PERSON = "models/yolo11n_openvino_model"
OPENVINO_PPE = "models/ppe/epi_openvino_model"


def openvino_available() -> bool:
    """Require the runtime and both complete artifacts, not just directories."""
    try:
        if find_spec("openvino") is None:
            return False
        for folder, stem in ((OPENVINO_PERSON, "yolo11n"), (OPENVINO_PPE, "epi")):
            for name in (f"{stem}.xml", f"{stem}.bin", "metadata.yaml"):
                item = Path(folder) / name
                if not item.is_file() or item.stat().st_size == 0:
                    return False
    except (ImportError, OSError, ValueError):
        return False
    return True


@lru_cache(maxsize=1)
def _automatic_device() -> str:
    # CUDA/MPS discovery runs once per server process, never once per frame.
    return select_device("auto")


def recommended_profile() -> str:
    return OPENVINO if openvino_available() and _automatic_device() == "cpu" else PYTORCH
