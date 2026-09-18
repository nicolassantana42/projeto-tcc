"""Workflow contracts, without model downloads, GPU or actual training."""

from pathlib import Path
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from safeguard import cli, ml


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "samples"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    source = tmp_path / "dataset.yaml"
    source.write_text("path: samples\ntrain: images/train\nval: images/val\nnames: [helmet]\n", encoding="utf-8")
    return source


def test_dataset_paths_are_relative_to_yaml_and_no_longer_depend_on_yolo_settings(dataset):
    with ml.normalized_dataset(str(dataset)) as path:
        normalized = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert Path(normalized["path"]) == dataset.parent / "samples"
        assert Path(normalized["val"]).is_dir()
    assert not Path(path).exists()


def test_dataset_rejects_missing_images_and_download_hooks(dataset):
    dataset.write_text("path: absent\ntrain: images/train\nval: images/val\nnames: [helmet]\n", encoding="utf-8")
    with pytest.raises(ml.WorkflowError, match="Split 'train'"):
        with ml.normalized_dataset(str(dataset)):
            pass
    dataset.write_text("train: x\nval: y\nnames: [helmet]\ndownload: echo no\n", encoding="utf-8")
    with pytest.raises(ml.WorkflowError, match="Remova 'download'"):
        with ml.normalized_dataset(str(dataset)):
            pass


def test_missing_model_is_not_implicitly_downloaded(tmp_path):
    with pytest.raises(ml.WorkflowError, match="download"):
        ml.load_yolo(str(tmp_path / "yolo11n.pt"))


def test_workflow_openvino_xml_resolves_to_supported_directory(monkeypatch, tmp_path):
    directory = tmp_path / "ppe_openvino_model"
    directory.mkdir()
    xml = directory / "ppe.xml"
    xml.touch()
    xml.with_suffix(".bin").touch()
    paths = []
    monkeypatch.setattr(ml, "require_optional", lambda *a: None)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path, **kw: paths.append(path)))
    ml.load_yolo(str(xml))
    assert paths == [str(directory.resolve())]


@pytest.mark.parametrize("precision,capability", [("int8", "platform_has_fast_int8"), ("fp16", "platform_has_fast_fp16")])
def test_tensorrt_cannot_silently_downgrade_requested_precision(monkeypatch, precision, capability):
    class Logger:
        ERROR = 1

        def __init__(self, level):
            pass

    support = SimpleNamespace(**{capability: False})
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(Logger=Logger, Builder=lambda _: support))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext())))
    with pytest.raises(ml.WorkflowError, match="conversão silenciosa"):
        ml.check_tensorrt_precision(precision, "cuda:0")
    setattr(support, capability, True)
    ml.check_tensorrt_precision(precision, "cuda:0")


@pytest.mark.parametrize("format", ["openvino", "engine"])
def test_int8_requires_explicit_calibration_dataset(format):
    with pytest.raises(ml.WorkflowError, match="representativo"):
        ml.export_model("model.pt", format=format, precision="int8")


def test_onnx_is_not_misrepresented_as_int8():
    with pytest.raises(ml.WorkflowError, match="ONNX exporta FP32/FP16"):
        ml.export_model("model.pt", format="onnx", precision="int8", data="data.yaml")


def test_tensorrt_rejects_stale_calibration_cache_before_export(tmp_path):
    model = tmp_path / "model.pt"
    model.with_suffix(".cache").write_bytes(b"old dataset calibration")
    with pytest.raises(ml.WorkflowError, match="Cache de calibração existente"):
        ml.export_model(str(model), format="engine", precision="int8", data="new.yaml")


@pytest.mark.parametrize("format,precision,match", [("engine", "fp32", "NVIDIA"), ("onnx", "fp16", "FP16")])
def test_gpu_only_export_fails_before_optional_imports(monkeypatch, format, precision, match):
    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    with pytest.raises(ml.WorkflowError, match=match):
        ml.export_model("model.pt", format=format, precision=precision)


def test_export_passes_real_int8_and_normalized_calibration_to_yolo(monkeypatch, tmp_path, dataset):
    calls = {}
    artifact = tmp_path / "ppe_int8_openvino_model"
    artifact.mkdir()

    def export(**kwargs):
        calls.update(kwargs)
        assert Path(kwargs["data"]).is_file()
        return str(artifact)

    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    monkeypatch.setattr(ml, "require_optional", lambda *a: None)
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    monkeypatch.setattr(ml, "load_yolo", lambda path: SimpleNamespace(export=export))
    report = ml.export_model("ppe.pt", format="openvino", precision="int8", data=str(dataset), fraction=0.5)
    assert calls["int8"] is True
    assert calls["half"] is False
    assert calls["fraction"] == 0.5
    assert calls["device"] == "cpu"
    assert report["config"]["precision"] == "int8"
    assert Path(str(artifact) + ".export.json").is_file()


@pytest.mark.parametrize("model_path,expected_device", [("ppe.pt", "cpu"), ("ppe_openvino_model", "intel:cpu"), ("ppe.onnx", "cpu")])
def test_validation_generates_plots_prediction_json_and_numeric_metrics(monkeypatch, tmp_path, dataset, model_path, expected_device):
    calls, callbacks = {}, {}
    metrics = SimpleNamespace(
        save_dir=tmp_path, box=SimpleNamespace(map50=0.8, map=0.6, mp=0.7, mr=0.9),
        results_dict={"metrics/mAP50(B)": 0.8}, speed={"inference": 2.5}, names={0: "helmet"},
        curves=["Precision-Recall(B)"], curves_results=[[np.array([0, 1]), np.array([[1, 0]]), "Recall", "Precision"]],
    )

    def validate(**kwargs):
        calls.update(kwargs)
        callbacks["on_val_end"](SimpleNamespace(confusion_matrix=SimpleNamespace(matrix=np.array([[3, 1], [2, 0]]))))
        (tmp_path / "PR_curve.png").touch()
        return metrics

    fake = SimpleNamespace(val=validate, add_callback=lambda event, callback: callbacks.update({event: callback}))
    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    monkeypatch.setattr(ml, "load_yolo", lambda path: fake)
    monkeypatch.setattr(ml, "exported_input_shape", lambda path: (1, 320, 320) if path.endswith(".onnx") else None)
    report = ml.validate_model(model_path, str(dataset))
    assert calls["plots"] is True and calls["save_json"] is True
    assert calls["device"] == expected_device
    assert calls["imgsz"] == (320 if model_path.endswith(".onnx") else 640)
    assert report["config"]["requested_imgsz"] == 640
    assert calls["conf"] == 0.001
    assert report["metrics"]["mAP50_95"] == 0.6
    assert report["confusion_matrix"] == [[3, 1], [2, 0]]
    assert "PR_curve.png" in report["artifacts"]
    assert (tmp_path / "validation.json").is_file()


def test_training_uses_seed_local_dataset_and_no_implicit_amp_download(monkeypatch, tmp_path, dataset):
    calls = {}

    def train(**kwargs):
        calls.update(kwargs)
        assert Path(kwargs["data"]).is_file()
        return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.2})

    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    monkeypatch.setattr(ml, "load_yolo", lambda path: SimpleNamespace(train=train, trainer=SimpleNamespace(save_dir=tmp_path)))
    report = ml.train_model("ppe.pt", str(dataset), epochs=2, seed=17)
    assert calls["amp"] is False
    assert calls["seed"] == 17
    assert calls["epochs"] == 2
    assert calls["deterministic"] is True
    assert Path(report["report_path"]).is_file()
    assert (tmp_path / "dataset.resolved.yaml").is_file()


def test_benchmark_percentiles_and_fps_use_actual_elapsed_samples():
    summary = ml.latency_summary([10, 20, 30])
    assert summary["p50_ms"] == 20
    assert summary["p95_ms"] == pytest.approx(29)
    assert summary["fps"] == 50
    with pytest.raises(ml.WorkflowError, match="Nenhum frame"):
        ml.latency_summary([])


def test_cli_dispatch_preserves_export_configuration(monkeypatch, capsys):
    calls = {}

    def export(**kwargs):
        calls.update(kwargs)
        return {"artifact": "model_openvino"}

    monkeypatch.setattr(ml, "export_model", export)
    assert cli.main(["export", "--model", "ppe.pt", "--format", "openvino", "--precision", "int8", "--data", "ppe.yaml"]) == 0
    assert calls["model_path"] == "ppe.pt"
    assert calls["format"] == "openvino"
    assert calls["data"] == "ppe.yaml"
    assert "artifact" in capsys.readouterr().out


def test_cli_friendly_error_and_explicit_debug(monkeypatch, capsys):
    def fail(**kwargs):
        raise ml.WorkflowError("Instale a dependência opcional.")

    monkeypatch.setattr(ml, "export_model", fail)
    assert cli.main(["export"]) == 1
    output = capsys.readouterr().err
    assert "Instale a dependência" in output
    assert "Traceback" not in output
    with pytest.raises(ml.WorkflowError):
        cli.main(["--debug", "export"])


def test_download_does_not_replace_existing_weights(tmp_path, monkeypatch):
    model = tmp_path / "yolo11n.pt"
    model.write_bytes(b"existing trusted checkpoint")

    def forbidden(*args, **kwargs):
        raise AssertionError("network must not be called")

    monkeypatch.setattr(ml.urllib.request, "urlopen", forbidden)
    assert ml.download_model(directory=str(tmp_path)) == model


def test_cli_accepts_numeric_camera_and_rtsp_without_modification():
    assert cli.source_value("0") == 0
    assert cli.source_value("rtsp://camera/live") == "rtsp://camera/live"
