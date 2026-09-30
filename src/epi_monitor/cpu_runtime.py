"""Bound CPU inference threads without changing training configuration.

PyTorch's intra-op thread budget is process-wide, not per model or per session.
The CLI training workflow runs in its own process and does not call this module.
Avoid running training and the dashboard in the same Python process.
OpenVINO instead receives a thread budget on its own compiled CPU model.
"""

from contextlib import contextmanager
import importlib
import logging
import os
from pathlib import Path
import threading

logger = logging.getLogger(__name__)
_SETUP_LOCK = threading.RLock()


def cpu_thread_budget() -> int:
    """Default to at most four workers, with an explicit deployment override."""
    configured = os.environ.get("EPI_CPU_THREADS")
    if configured is None:
        return min(4, os.cpu_count() or 1)
    value = configured.strip()
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError("EPI_CPU_THREADS deve ser um inteiro positivo, por exemplo 4 ou 8.")
    return int(value)


@contextmanager
def cpu_inference_setup():
    """Apply the budget AFTER Ultralytics' lazy select_device resets threads.

    Serialize our predictor initializations, then configure PyTorch once per
    initialized predictor. There is no setter, backend rebuild, or lock per
    frame. This does not alter OpenCV, inter-op workers or accelerator backends.
    Other libraries in the same process still share PyTorch's intra-op budget.
    """
    budget = cpu_thread_budget()
    with _SETUP_LOCK:
        yield
        torch = importlib.import_module("torch")
        previous = torch.get_num_threads()
        if previous != budget:
            torch.set_num_threads(budget)
        logger.info("Inferência PyTorch CPU: %s threads (limite compartilhado no processo; EPI_CPU_THREADS).", budget)


def configure_openvino_cpu(predictor, model_path: str | Path) -> None:
    """Install a bounded CPU runtime once, before the first real prediction.

    Ultralytics 8.3.203 compiles OpenVINO with an unbounded LATENCY hint. The
    pinned backend exposes ``ov_compiled_model``; replacing that handle retains
    its preprocessing, names and output conversion. No work happens per frame.
    """
    budget = cpu_thread_budget()
    path = Path(model_path).resolve()
    xml = path if path.suffix.lower() == ".xml" else next(path.glob("*.xml"))
    key = (str(xml), budget)
    backend = predictor.model
    with _SETUP_LOCK:
        if getattr(backend, "_epi_cpu_config", None) == key:
            return
        if not hasattr(backend, "ov_compiled_model"):
            raise RuntimeError("Backend OpenVINO incompatível; use a versão Ultralytics fixada em requirements.txt.")
        ov = importlib.import_module("openvino")
        core = ov.Core()
        model = core.read_model(model=str(xml), weights=xml.with_suffix(".bin"))
        parameter = model.get_parameters()[0]
        if parameter.get_layout().empty:
            parameter.set_layout(ov.Layout("NCHW"))
        compiled = core.compile_model(
            model,
            device_name="CPU",
            config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": budget, "NUM_STREAMS": 1},
        )
        backend.ov_compiled_model = compiled
        backend.input_name = compiled.input().get_any_name()
        backend.inference_mode = "LATENCY"
        backend._epi_cpu_config = key
        logger.info("Inferência OpenVINO CPU: %s threads, um stream (EPI_CPU_THREADS).", budget)
