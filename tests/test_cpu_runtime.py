"""CPU runtime contracts; no real models, cameras or GPU are required."""

from types import SimpleNamespace

import numpy as np
import pytest

from epi_monitor import cpu_runtime
from epi_monitor.config import InferenceConfig
from epi_monitor.inference import ModelError, YOLODetector


@pytest.mark.parametrize("logical,expected", [(12, 4), (2, 2), (None, 1)])
def test_default_budget_leaves_room_for_other_work(monkeypatch, logical, expected):
    monkeypatch.delenv("EPI_CPU_THREADS", raising=False)
    monkeypatch.setattr(cpu_runtime.os, "cpu_count", lambda: logical)
    assert cpu_runtime.cpu_thread_budget() == expected


@pytest.mark.parametrize("value,expected", [("1", 1), ("8", 8), (" 4 ", 4)])
def test_explicit_budget(monkeypatch, value, expected):
    monkeypatch.setenv("EPI_CPU_THREADS", value)
    assert cpu_runtime.cpu_thread_budget() == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "True", "auto", "１２"])
def test_invalid_budget_is_actionable(monkeypatch, value):
    monkeypatch.setenv("EPI_CPU_THREADS", value)
    with pytest.raises(ValueError, match="EPI_CPU_THREADS"):
        cpu_runtime.cpu_thread_budget()


def test_budget_is_applied_after_backend_initialization_and_only_when_needed(monkeypatch):
    monkeypatch.setenv("EPI_CPU_THREADS", "4")
    counts = [16]
    changes = []

    def set_threads(count):
        changes.append(count)
        counts[0] = count

    fake_torch = SimpleNamespace(get_num_threads=lambda: counts[0], set_num_threads=set_threads)
    monkeypatch.setattr(cpu_runtime.importlib, "import_module", lambda _: fake_torch)
    with cpu_runtime.cpu_inference_setup():
        counts[0] = 8  # Ultralytics select_device() overrides this during setup.
        assert changes == []
    assert changes == [4]
    with cpu_runtime.cpu_inference_setup():
        pass
    assert changes == [4]


def _runtime(monkeypatch, tmp_path, *, device="cpu"):
    monkeypatch.setenv("EPI_CPU_THREADS", "4")
    path = tmp_path / "model.pt"
    path.touch()
    state = {"threads": 16, "setters": [], "setup": 0, "predict_threads": []}

    def set_threads(count):
        state["threads"] = count
        state["setters"].append(count)

    torch = SimpleNamespace(get_num_threads=lambda: state["threads"], set_num_threads=set_threads)

    class Predictor:
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**overrides)

        def setup_model(self, model, verbose):
            state["setup"] += 1
            state["threads"] = 8
            self.model = model

    class Model:
        names = {0: "person"}

        def __init__(self, *_args, **_kwargs):
            self.overrides = {}
            self.callbacks = {}
            self.predictor = None
            self.model = object()

        def _smart_load(self, _name):
            return Predictor

        def predict(self, **kwargs):
            state["predict_threads"].append(state["threads"])
            return []

    def imported(name):
        return torch if name == "torch" else SimpleNamespace(YOLO=Model)

    monkeypatch.setattr(cpu_runtime.importlib, "import_module", imported)
    monkeypatch.setattr("epi_monitor.inference.select_device", lambda *_: device)
    detector = YOLODetector(InferenceConfig(model_path=str(path)))
    return detector, state


def test_native_cpu_initializes_once_before_any_frame(monkeypatch, tmp_path):
    detector, state = _runtime(monkeypatch, tmp_path)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detector.load()
    detector.predict(frame)
    detector.predict(frame)
    assert state["setup"] == 1
    assert state["setters"] == [4]
    assert state["predict_threads"] == [4, 4]
    assert detector._model.overrides["half"] is False


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_accelerator_loading_does_not_modify_cpu_threads(monkeypatch, tmp_path, device):
    detector, state = _runtime(monkeypatch, tmp_path, device=device)
    monkeypatch.setenv("EPI_CPU_THREADS", "invalid_but_irrelevant_to_gpu")
    detector.load()
    assert state["setup"] == 0
    assert state["setters"] == []
    assert detector._model.predictor is None


def test_invalid_cpu_environment_is_reported_as_model_error(monkeypatch, tmp_path):
    detector, state = _runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("EPI_CPU_THREADS", "0")
    with pytest.raises(ModelError, match="EPI_CPU_THREADS"):
        detector.load()
    assert state["setup"] == 0


@pytest.mark.parametrize("budget", ["4", "8"])
def test_openvino_budget_compiles_cpu_once_and_preserves_backend(monkeypatch, tmp_path, budget):
    monkeypatch.setenv("EPI_CPU_THREADS", budget)
    xml = tmp_path / "model.xml"
    xml.touch()
    xml.with_suffix(".bin").touch()
    calls = []
    layouts = []
    parameter = SimpleNamespace(get_layout=lambda: SimpleNamespace(empty=True), set_layout=layouts.append)
    ov_model = SimpleNamespace(get_parameters=lambda: [parameter])
    compiled = SimpleNamespace(input=lambda: SimpleNamespace(get_any_name=lambda: "images"))

    class Core:
        def read_model(self, **kwargs):
            assert kwargs == {"model": str(xml), "weights": xml.with_suffix(".bin")}
            return ov_model

        def compile_model(self, model, **kwargs):
            calls.append((model, kwargs))
            return compiled

    monkeypatch.setattr(cpu_runtime.importlib, "import_module", lambda _: SimpleNamespace(Core=Core, Layout=lambda value: value))
    backend = SimpleNamespace(ov_compiled_model=object(), names={0: "helmet"}, inference_mode="LATENCY")
    predictor = SimpleNamespace(model=backend)
    cpu_runtime.configure_openvino_cpu(predictor, xml.parent)
    cpu_runtime.configure_openvino_cpu(predictor, xml)
    assert calls == [(ov_model, {"device_name": "CPU", "config": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": int(budget), "NUM_STREAMS": 1}})]
    assert predictor.model is backend
    assert backend.ov_compiled_model is compiled
    assert backend.names == {0: "helmet"}
    assert backend.input_name == "images"
    assert backend.inference_mode == "LATENCY"
    assert layouts == ["NCHW"]


def test_incompatible_openvino_backend_fails_without_silent_unbounded_execution(tmp_path):
    xml = tmp_path / "model.xml"
    xml.touch()
    with pytest.raises(RuntimeError, match="Ultralytics fixada"):
        cpu_runtime.configure_openvino_cpu(SimpleNamespace(model=SimpleNamespace()), xml)
