"""Behavior tests: no network, weights, accelerators or camera required."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from safeguard.config import InferenceConfig
from safeguard.hardware import HardwareError, select_device
from safeguard.inference import ModelError, YOLODetector, exported_input_shape, resolve_model_path
from safeguard.pipeline import Pipeline
from safeguard.preprocessing import preprocess_frame
from safeguard.rendering import render_frame
from safeguard.rules import assess_ppe
from safeguard.types import Detection


@pytest.fixture
def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _det(label="person", bbox=(10, 10, 80, 110), class_id=0):
    return Detection(class_id, label, 0.9, bbox)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, None])
def test_invalid_thresholds_are_rejected(value):
    with pytest.raises(ValueError):
        InferenceConfig(confidence=value)


@pytest.mark.parametrize("bad", [None, [], np.zeros((3, 3)), np.zeros((3, 3, 4), dtype=np.uint8), np.zeros((3, 3, 3)), np.zeros((0, 3, 3), dtype=np.uint8)])
def test_preprocess_rejects_non_bgr_uint8(bad):
    with pytest.raises(ValueError):
        preprocess_frame(bad)


def test_preprocess_preserves_pixels_shape_and_makes_contiguous(frame):
    frame[0, 0] = [3, 8, 240]
    view = frame[:, ::-1]
    prepared = preprocess_frame(view)
    assert prepared.flags.c_contiguous
    assert prepared.shape == view.shape
    np.testing.assert_array_equal(prepared, view)
    assert preprocess_frame(frame) is frame


def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr("safeguard.hardware._torch_capabilities", lambda: (True, True, 1))
    assert select_device() == "cuda:0"
    monkeypatch.setattr("safeguard.hardware._torch_capabilities", lambda: (False, True, 0))
    assert select_device() == "mps"
    monkeypatch.setattr("safeguard.hardware._torch_capabilities", lambda: (False, False, 0))
    assert select_device() == "cpu"
    assert select_device("cuda:0") == "cpu"


def test_exported_artifacts_respect_device_capabilities(monkeypatch):
    monkeypatch.setattr("safeguard.hardware._torch_capabilities", lambda: (True, True, 1))
    monkeypatch.setattr("safeguard.hardware._onnx_cuda_available", lambda: False)
    assert select_device("auto", "model.onnx") == "cpu"
    assert select_device("mps", "model.onnx") == "cpu"
    assert select_device("auto", "model_openvino_model") == "cpu"
    with pytest.raises(HardwareError, match="TensorRT"):
        select_device("cpu", "model.engine")
    assert select_device("auto", "model.engine") == "cuda:0"
    monkeypatch.setattr("safeguard.hardware._torch_capabilities", lambda: (False, True, 0))
    with pytest.raises(HardwareError):
        select_device("auto", "model.engine")


def _fake_model(monkeypatch, tmp_path, names=None, fail_gpu=False):
    path = tmp_path / "model.pt"
    path.touch()
    calls = []
    builds = []

    class FakeYOLO:
        def __init__(self, model_path, task):
            self.names = names or {0: "person", 1: "helmet", 2: "vest"}
            self.overrides = {}
            builds.append(self)

        def predict(self, **kwargs):
            calls.append(kwargs)
            if fail_gpu and kwargs["device"] != "cpu":
                raise RuntimeError("CUDA out of memory")
            boxes = SimpleNamespace(
                xyxy=np.array([[-10, 10, 80, 130], [12, 15, 8, 30], [1, 1, float("nan"), 4]]),
                conf=np.array([0.8, 0.9, 0.7]),
                cls=np.array([0, 1, 2]),
            )
            return [SimpleNamespace(boxes=boxes)]

    original_import = importlib.import_module
    monkeypatch.setattr("safeguard.inference.importlib.import_module", lambda name: SimpleNamespace(YOLO=FakeYOLO) if name == "ultralytics" else original_import(name))
    monkeypatch.setattr("safeguard.inference.select_device", lambda *_: "cuda:0" if fail_gpu else "cpu")
    return str(path), calls, builds


def test_model_load_is_lazy_cached_and_missing_weights_are_actionable(tmp_path):
    detector = YOLODetector(InferenceConfig(model_path=str(tmp_path / "missing.pt")))
    assert detector._model is None
    with pytest.raises(ModelError, match="Modelo não encontrado"):
        detector.load()


def test_model_loads_once_predict_maps_output_and_thresholds(monkeypatch, tmp_path, frame):
    path, calls, builds = _fake_model(monkeypatch, tmp_path)
    detector = YOLODetector(InferenceConfig(model_path=path))
    assert not builds
    assert detector.load() is detector
    detector.load()
    detections = detector.predict(frame, confidence=0, iou=0.8)
    assert len(builds) == 1
    assert detector.names[0] == "person"
    assert calls[0]["conf"] == 0
    assert calls[0]["iou"] == 0.8
    assert calls[0]["half"] is False
    assert builds[0].overrides["device"] == "cpu"
    assert len(detections) == 1
    assert detections[0].bbox == (0.0, 10.0, 80.0, 120.0)
    assert detections[0].label == "person"


def test_gpu_failure_reloads_a_cpu_compatible_model(monkeypatch, tmp_path, frame):
    path, calls, builds = _fake_model(monkeypatch, tmp_path, fail_gpu=True)
    detector = YOLODetector(InferenceConfig(model_path=path))
    assert detector.predict(frame)
    assert detector.device == "cpu"
    assert len(builds) == 2
    assert [call["device"] for call in calls] == ["cuda:0", "cpu"]
    assert [call["half"] for call in calls] == [True, False]


def test_gpu_initialization_failure_falls_back_for_pt(monkeypatch, tmp_path):
    path = tmp_path / "model.pt"
    path.touch()
    monkeypatch.setattr("safeguard.inference.select_device", lambda *_: "cuda:0")
    detector = YOLODetector(InferenceConfig(model_path=str(path)))
    attempted = []

    def build():
        attempted.append(detector.device)
        if detector.device != "cpu":
            raise RuntimeError("CUDA runtime unavailable")
        return SimpleNamespace(names={0: "person"})

    monkeypatch.setattr(detector, "_build_model", build)
    detector.load()
    assert attempted == ["cuda:0", "cpu"]
    assert detector.device == "cpu"


@pytest.mark.parametrize("suffix,module", [(".onnx", "onnx"), (".engine", "tensorrt")])
def test_exported_model_missing_dependency_reports_backend(monkeypatch, tmp_path, suffix, module):
    path = tmp_path / f"model{suffix}"
    path.touch()

    def unavailable(_):
        raise ImportError(module)

    monkeypatch.setattr("safeguard.inference.importlib.import_module", unavailable)
    with pytest.raises(ModelError, match="Dependência de inferência"):
        YOLODetector(InferenceConfig(model_path=str(path))).load()


def test_openvino_requires_xml_and_bin(tmp_path):
    path = tmp_path / "model_openvino_model"
    path.mkdir()
    with pytest.raises(ModelError, match=".xml e .bin"):
        YOLODetector(InferenceConfig(model_path=str(path))).load()


def test_exported_precision_is_not_overridden_and_openvino_cpu_is_explicit(frame):
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return []

    for path in ("model.onnx", "model.xml"):
        detector = YOLODetector(InferenceConfig(model_path=path))
        detector._model = SimpleNamespace(predict=predict)
        detector._predict_raw(frame, 0.4, 0.5)
    assert all("half" not in call for call in calls)
    assert calls[0]["device"] == "cpu"
    assert calls[1]["device"] == "intel:cpu"


def _fake_exported_backend(monkeypatch, shape):
    calls, paths, setups = [], [], []

    class Predictor:
        def __init__(self, overrides, _callbacks):
            self.args = SimpleNamespace(**overrides)

        def setup_model(self, model, verbose):
            setups.append(model)
            self.model = SimpleNamespace(names={0: "helmet"})

    class ExportedYOLO:
        def __init__(self, model_path, task):
            paths.append(model_path)
            self.model = model_path
            self.overrides = {}
            self.callbacks = {}
            self.predictor = None

        def _smart_load(self, name):
            assert name == "predictor"
            return Predictor

        @property
        def names(self):
            assert self.predictor is not None  # backend must already be initialized
            return self.predictor.model.names

        def predict(self, **kwargs):
            calls.append(kwargs)
            # Reproduces the persistent predictor's per-call args overwrite.
            self.predictor.args.imgsz = kwargs["imgsz"]
            return []

    monkeypatch.setattr("safeguard.inference.importlib.import_module", lambda name: SimpleNamespace(YOLO=ExportedYOLO))
    monkeypatch.setattr("safeguard.inference.select_device", lambda *_: "cpu")
    monkeypatch.setattr("safeguard.inference.exported_input_shape", lambda _: shape)
    return calls, paths, setups


@pytest.mark.parametrize("use_xml", [False, True])
def test_openvino_loader_passes_supported_directory_and_builds_backend_once(monkeypatch, tmp_path, frame, use_xml):
    directory = tmp_path / "ppe_openvino_model"
    directory.mkdir()
    xml = directory / "ppe.xml"
    xml.touch()
    xml.with_suffix(".bin").touch()
    calls, paths, setups = _fake_exported_backend(monkeypatch, (1, 640, 640))
    detector = YOLODetector(InferenceConfig(model_path=str(xml if use_xml else directory))).load()
    detector.predict(frame)
    detector.predict(frame)
    assert paths == [str(directory.resolve())]
    assert setups == paths
    assert [call["device"] for call in calls] == ["intel:cpu", "intel:cpu"]


def test_arbitrary_openvino_directory_has_actionable_error(tmp_path):
    directory = tmp_path / "ambiguous"
    directory.mkdir()
    with pytest.raises(ModelError, match="_openvino_model"):
        resolve_model_path(directory)


def test_fixed_onnx_shape_is_preserved_on_second_prediction(monkeypatch, tmp_path, frame):
    path = tmp_path / "model.onnx"
    path.touch()
    calls, paths, setups = _fake_exported_backend(monkeypatch, (1, 320, 320))
    detector = YOLODetector(InferenceConfig(model_path=str(path), imgsz=640)).load()
    detector.predict(frame)
    detector.predict(frame)
    assert detector.config.imgsz == 640  # user request remains immutable
    assert detector.imgsz == 320
    assert [call["imgsz"] for call in calls] == [320, 320]
    assert len(setups) == 1


def test_static_batched_export_is_rejected_for_single_frame_pipeline(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    path.touch()
    _fake_exported_backend(monkeypatch, (4, 640, 640))
    with pytest.raises(ModelError, match="--batch 1"):
        YOLODetector(InferenceConfig(model_path=str(path))).load()


def test_onnx_shape_uses_graph_dimensions_not_untrusted_requested_size(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    path.touch()
    dims = [SimpleNamespace(dim_value=value) for value in (1, 3, 320, 640)]
    graph = SimpleNamespace(input=[SimpleNamespace(type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims))))])
    monkeypatch.setattr("safeguard.inference.importlib.import_module", lambda _: SimpleNamespace(load=lambda *a, **kw: SimpleNamespace(graph=graph)))
    assert exported_input_shape(path) == (1, 320, 640)


def test_empty_pipeline_output_is_valid(frame):
    detector = SimpleNamespace(names={0: "person"}, predict=lambda *_args, **_kwargs: [])
    pipeline = Pipeline(detector)
    first = pipeline.process(frame)
    second = pipeline.process(frame)
    assert first.counts == {}
    assert first.detections == []
    assert first.alerts == []
    assert (first.frame_index, second.frame_index) == (1, 2)


def test_missing_ultralytics_has_readable_error(monkeypatch, tmp_path):
    path = tmp_path / "model.pt"
    path.touch()
    monkeypatch.setattr("safeguard.inference.select_device", lambda *_: "cpu")

    def missing_import(_):
        raise ImportError("not installed")

    monkeypatch.setattr("safeguard.inference.importlib.import_module", missing_import)
    with pytest.raises(ModelError, match="requirements.txt"):
        YOLODetector(InferenceConfig(model_path=str(path))).load()


def test_pipeline_counts_preserves_frame_and_render_is_separate(frame):
    class FakeDetector:
        names = {0: "person"}

        def predict(self, source, confidence=None, iou=None):
            assert confidence == 0.3
            assert iou == 0.6
            return [_det()]

    pipeline = Pipeline(FakeDetector())
    result = pipeline.process(frame, confidence=0.3, iou=0.6)
    assert result.frame is frame
    assert result.frame_index == 1
    assert result.counts == {"person": 1}
    assert result.alerts == []
    assert 0 <= result.inference_ms <= result.pipeline_ms
    assert not frame.any()
    rendered = render_frame(result)
    assert rendered.any()
    assert not frame.any()
    assert not np.shares_memory(rendered, frame)


def test_demo_and_coco_never_accuse_missing_ppe():
    detections = [_det()]
    assert assess_ppe(detections, {0: "person"}, demo_mode=True) == []
    alerts = assess_ppe(detections, {0: "person"}, demo_mode=False)
    assert len(alerts) == 1
    assert "indisponível" in alerts[0]
    assert "possível ausência" not in alerts[0]


def test_ppe_is_associated_to_each_person_not_global():
    names = {0: "person", 1: "hardhat", 2: "safety vest"}
    detections = [
        _det(bbox=(0, 0, 100, 200)),
        _det(bbox=(200, 0, 300, 200)),
        _det("hardhat", (25, 0, 75, 40), 1),
        _det("safety vest", (20, 50, 80, 140), 2),
    ]
    alerts = assess_ppe(detections, names, demo_mode=False)
    assert len(alerts) == 1
    assert "Pessoa 2" in alerts[0]
    assert "capacete e colete" in alerts[0]


def test_overlapping_people_cannot_share_one_helmet_or_vest():
    names = {0: "person", 1: "helmet", 2: "vest"}
    detections = [_det(bbox=(0, 0, 100, 200)), _det(bbox=(0, 0, 100, 200)), _det("helmet", (25, 0, 75, 40), 1), _det("vest", (20, 50, 80, 140), 2)]
    assert len(assess_ppe(detections, names, demo_mode=False)) == 1


def test_negative_ppe_class_does_not_count_as_equipment():
    names = {0: "person", 1: "helmet", 2: "vest", 3: "NO-Hardhat"}
    alerts = assess_ppe([_det(), _det("NO-Hardhat", (20, 10, 60, 30), 3)], names, demo_mode=False)
    assert "capacete e colete" in alerts[0]
