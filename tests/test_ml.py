"""Workflow contracts, without model downloads, GPU or actual training."""

from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from safeguard import cli, ml


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "samples"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    for split, color in (("train", "red"), ("val", "blue")):
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (24, 24), color).save(root / "images" / split / "sample.png")
        (root / "labels" / split / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
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
        box=SimpleNamespace(map50=0.8, map=0.6, mp=0.7, mr=0.9,
                            ap_class_index=np.array([0]), p=[0.7], r=[0.9], ap50=[0.8], ap=[0.6]),
        results_dict={"metrics/mAP50(B)": 0.8}, speed={"inference": 2.5}, names={0: "helmet"},
        curves=["Precision-Recall(B)"], curves_results=[[np.array([0, 1]), np.array([[1, 0]]), "Recall", "Precision"]],
    )

    def validate(**kwargs):
        calls.update(kwargs)
        metrics.save_dir = Path(kwargs["project"]) / kwargs["name"]
        callbacks["on_val_end"](SimpleNamespace(confusion_matrix=SimpleNamespace(matrix=np.array([[3, 1], [2, 0]]))))
        (metrics.save_dir / "PR_curve.png").touch()
        return metrics

    fake = SimpleNamespace(names={0: "helmet"}, overrides={}, val=validate,
                           add_callback=lambda event, callback: callbacks.update({event: callback}))
    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    monkeypatch.setattr(ml, "load_yolo", lambda path: fake)
    monkeypatch.setattr(ml, "exported_input_shape", lambda path: (1, 320, 320) if path.endswith(".onnx") else None)
    artifact = tmp_path / model_path
    if model_path.endswith("openvino_model"):
        artifact.mkdir()
        (artifact / "model.xml").write_bytes(b"test xml")
        (artifact / "model.bin").write_bytes(b"test weights")
    else:
        artifact.write_bytes(b"test checkpoint")
    report = ml.validate_model(str(artifact), str(dataset), project=str(tmp_path / "validation"))
    assert calls["plots"] is True and calls["save_json"] is True
    assert calls["device"] == expected_device
    assert calls["imgsz"] == (320 if model_path.endswith(".onnx") else 640)
    assert report["config"]["requested_imgsz"] == 640
    assert calls["conf"] == 0.001
    assert report["metrics"]["mAP50_95"] == 0.6
    assert report["confusion_matrix"] == [[3, 1], [2, 0]]
    assert "PR_curve.png" in report["artifacts"]
    assert Path(report["report_path"]).is_file()
    assert report["per_class"] == [{"class_id": 0, "name": "helmet", "instances": 1,
                                    "evaluated": True, "precision": 0.7, "recall": 0.9,
                                    "mAP50": 0.8, "mAP50_95": 0.6}]
    assert len(report["provenance"]["weights"]["sha256"]) == 64
    assert report["provenance"]["dataset_yaml"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()


def test_training_uses_seed_local_dataset_and_no_implicit_amp_download(monkeypatch, tmp_path, dataset):
    calls = {}
    fake = SimpleNamespace(trainer=SimpleNamespace())

    def train(**kwargs):
        calls.update(kwargs)
        assert Path(kwargs["data"]).is_file()
        directory = Path(kwargs["project"]) / kwargs["name"]
        assert (directory / "dataset-audit.json").is_file()
        assert (directory / "val.txt").is_file()
        fake.trainer.save_dir = directory
        (directory / "weights").mkdir()
        (directory / "weights" / "best.pt").write_bytes(b"test trained checkpoint")
        return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.2})

    fake.train = train
    monkeypatch.setattr(ml, "select_device", lambda *a, **kw: "cpu")
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    monkeypatch.setattr(ml, "load_yolo", lambda path: fake)
    initial_weights = tmp_path / "ppe.pt"
    initial_weights.write_bytes(b"test initial checkpoint")
    report = ml.train_model(str(initial_weights), str(dataset), epochs=2, seed=17, project=str(tmp_path / "train"))
    assert calls["amp"] is False
    assert calls["seed"] == 17
    assert calls["epochs"] == 2
    assert calls["deterministic"] is True
    assert Path(report["report_path"]).is_file()
    directory = Path(report["report_path"]).parent
    assert (directory / "dataset.resolved.yaml").is_file()
    assert report["provenance"]["initial_weights"]["sha256"] == hashlib.sha256(initial_weights.read_bytes()).hexdigest()
    assert report["provenance"]["best_weights"]["sha256"] == hashlib.sha256(b"test trained checkpoint").hexdigest()
    assert report["provenance"]["last_weights"]["sha256"] is None
    assert all(Path(line).is_absolute() for line in (directory / "train.txt").read_text().splitlines())


@pytest.mark.parametrize("operation", [ml.train_model, ml.validate_model])
@pytest.mark.parametrize("problem", ["missing_label", "split_leakage", "corrupt_image"])
def test_invalid_datasets_fail_before_loading_weights_and_preserve_audit(monkeypatch, dataset, tmp_path, operation, problem):
    root = dataset.parent / "samples"
    if problem == "missing_label":
        (root / "labels" / "val" / "sample.txt").unlink()
    elif problem == "split_leakage":
        (root / "images" / "val" / "sample.png").write_bytes((root / "images" / "train" / "sample.png").read_bytes())
    else:
        (root / "images" / "val" / "sample.png").write_bytes(b"not an image")

    def forbidden(*args, **kwargs):
        pytest.fail("Weights must not load before a valid dataset audit")

    monkeypatch.setattr(ml, "load_yolo", forbidden)
    with pytest.raises(ml.WorkflowError, match="Auditoria do dataset falhou"):
        operation("unavailable.pt", str(dataset), project=str(tmp_path / "run"))
    audit = json.loads((tmp_path / "run" / "ppe" / "dataset-audit.json").read_text(encoding="utf-8"))
    assert audit["valid"] is False
    assert audit["errors"]


def test_training_persists_manifest_resolution_and_never_overwrites_old_run(monkeypatch, tmp_path, dataset):
    root = dataset.parent / "samples"
    (root / "manifests").mkdir()
    for split in ("train", "val"):
        (root / "manifests" / f"{split}.txt").write_text(f"../images/{split}/sample.png\n", encoding="utf-8")
    dataset.write_text("path: samples\ntrain: manifests/train.txt\nval: manifests/val.txt\nnames: [helmet]\n", encoding="utf-8")
    output = tmp_path / "output"
    existing = output / "ppe"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("previous experiment", encoding="utf-8")
    fake = SimpleNamespace(trainer=SimpleNamespace())

    def train(**kwargs):
        resolved = yaml.safe_load(Path(kwargs["data"]).read_text(encoding="utf-8"))
        for split in ("train", "val"):
            manifest = Path(resolved[split])
            assert manifest.parent == output / "ppe2"
            assert manifest.read_text(encoding="utf-8").splitlines() == [str((root / "images" / split / "sample.png").resolve())]
        fake.trainer.save_dir = Path(kwargs["project"]) / kwargs["name"]
        return SimpleNamespace(results_dict={})

    fake.train = train
    monkeypatch.setattr(ml, "load_yolo", lambda path: fake)
    monkeypatch.setattr(ml, "select_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ml, "environment", lambda device: {})
    report = ml.train_model("checkpoint.pt", str(dataset), project=str(output))
    assert sentinel.read_text(encoding="utf-8") == "previous experiment"
    assert Path(report["report_path"]).parent == output / "ppe2"
    assert Path(report["provenance"]["split_manifests"]["val"]["path"]).is_file()


def test_normalized_calibration_resolves_relative_txt_entries(dataset):
    root = dataset.parent / "samples"
    (root / "val.txt").write_text("images/val/sample.png\n", encoding="utf-8")
    dataset.write_text("path: samples\ntrain: images/train\nval: val.txt\nnames: [helmet]\n", encoding="utf-8")
    with ml.normalized_dataset(str(dataset), splits=("val",)) as path:
        resolved = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert Path(resolved["val"]).read_text(encoding="utf-8").strip() == str(root / "images" / "val" / "sample.png")


@pytest.mark.parametrize("names", [{0: "person"}, {0: "vest", 1: "helmet"}, {}])
def test_validation_rejects_mismatched_or_reordered_class_names_before_metrics(monkeypatch, tmp_path, dataset, names):
    if len(names) == 2:
        dataset.write_text("path: samples\ntrain: images/train\nval: images/val\nnames: [helmet, vest]\n", encoding="utf-8")
    fake = SimpleNamespace(names=names, overrides={})
    monkeypatch.setattr(ml, "load_yolo", lambda path: fake)
    monkeypatch.setattr(ml, "select_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ml, "exported_input_shape", lambda path: None)
    with pytest.raises(ml.WorkflowError, match="Classes do modelo incompatíveis"):
        ml.validate_model("checkpoint.pt", str(dataset), project=str(tmp_path / "evaluation"))
    assert fake.names == names
    assert "names" not in fake.overrides and "data" not in fake.overrides


def test_lazy_backend_receives_device_before_names_are_read(monkeypatch, tmp_path, dataset):
    class LazyModel:
        overrides = {}

        @property
        def names(self):
            assert self.overrides["device"] == "intel:cpu"
            assert self.overrides["imgsz"] == 320
            return {0: "wrong class"}

    monkeypatch.setattr(ml, "load_yolo", lambda path: LazyModel())
    monkeypatch.setattr(ml, "select_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(ml, "exported_input_shape", lambda path: (1, 320, 320))
    with pytest.raises(ml.WorkflowError, match="Classes do modelo incompatíveis"):
        ml.validate_model("ppe_openvino_model", str(dataset), project=str(tmp_path / "evaluation"))


def test_per_class_metrics_map_actual_ids_and_do_not_impute_unobserved_classes():
    metrics = SimpleNamespace(box=SimpleNamespace(ap_class_index=np.array([2, 0]),
                              p=[0.2, 0.9], r=[0.3, 0.8], ap50=[0.4, 0.7], ap=[0.1, 0.6]))
    audit = {"splits": {"test": {"class_counts": {"0": {"instances": 3}, "1": {"instances": 0}, "2": {"instances": 1}}}}}
    rows = ml._per_class_metrics(metrics, {0: "person", 1: "helmet", 2: "vest"}, audit, "test")
    assert rows[0]["mAP50"] == 0.7 and rows[2]["precision"] == 0.2
    assert rows[1]["evaluated"] is False
    assert rows[1]["mAP50"] is None and rows[1]["precision"] is None


def test_final_validation_requires_test_split_before_model_loading(monkeypatch, tmp_path, dataset):
    monkeypatch.setattr(ml, "load_yolo", lambda _: pytest.fail("Must reject absent test before loading"))
    with pytest.raises(ml.WorkflowError, match="split 'test'"):
        ml.validate_model("checkpoint.pt", str(dataset), split="test", project=str(tmp_path / "evaluation"))


def test_benchmark_percentiles_and_fps_use_actual_elapsed_samples():
    summary = ml.latency_summary([10, 20, 30])
    assert summary["p50_ms"] == 20
    assert summary["p95_ms"] == pytest.approx(29)
    assert summary["fps"] == 50
    with pytest.raises(ml.WorkflowError, match="Nenhum frame"):
        ml.latency_summary([])


@pytest.mark.parametrize("shared_device", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_cascade_benchmark_counts_gating_and_synchronizes_each_distinct_device(monkeypatch, tmp_path, shared_device, fallback):
    from safeguard import capture, factory
    from safeguard.detection import CascadePipeline
    from safeguard.types import Detection

    class FakeDetector:
        def __init__(self, equipment):
            self.equipment = equipment
            self.names = {0: "helmet", 1: "vest"} if equipment else {0: "person"}
            self.device = "cuda:0" if shared_device or not equipment else "cpu"
            self.imgsz = 416 if equipment else 640
            self.calls = 0

        def predict(self, frame, **kwargs):
            self.calls += 1
            if fallback:
                self.device = "cpu"
            if self.equipment:
                return [Detection(0, "helmet", .9, (50, 20, 75, 35)), Detection(1, "vest", .9, (40, 65, 95, 110))]
            return [Detection(0, "person", .9, (30, 15, 110, 210))] if frame[0, 0, 0] else []

    class FakeVideo:
        def __init__(self, source):
            self.frames = iter(np.full((256, 256, 3), value, dtype=np.uint8) for value in (1, 1, 0))
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def read(self):
            return next(self.frames, None)

    person, equipment = FakeDetector(False), FakeDetector(True)
    pipeline = CascadePipeline(person, equipment)
    calls, sync = [], []

    def create(*args):
        calls.append(args)
        return pipeline

    monkeypatch.setattr(factory, "create_cascade", create)
    monkeypatch.setattr(capture, "VideoSource", FakeVideo)
    monkeypatch.setattr(ml, "synchronize", sync.append)
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    report = ml.benchmark_model("person.pt", "fixture.avi", ppe_model="equipment.pt", frames=2, warmup=1,
                                imgsz=640, output=str(tmp_path / "benchmark.json"))
    assert calls == [("person.pt", "equipment.pt", "auto", 640, .4, .45)]
    assert report["kind"] == "cascade_benchmark"
    assert report["config"]["measured_frames"] == 2
    assert report["ppe_executed_frames"] == report["ppe_skipped_frames"] == 1
    assert person.calls == 3 and equipment.calls == 2  # Includes the unmeasured warmup.
    assert report["config"]["ppe_imgsz"] == 416
    assert report["config"]["ppe_device"] == equipment.device
    expected_devices = ["cpu"] if fallback else (["cuda:0"] if shared_device else ["cuda:0", "cpu"])
    assert sync == expected_devices * 3
    assert report["stage_timings"]["person"]["mean_ms"] >= 0
    assert report["stage_timings"]["ppe"]["mean_ms"] >= 0
    assert Path(report["report_path"]).is_file()


def test_default_benchmark_retains_single_detector_without_cascade(monkeypatch, tmp_path):
    from safeguard import capture, factory, inference

    detector = SimpleNamespace(device="cpu", imgsz=320, names={0: "person"}, predict=lambda *args, **kwargs: [])
    detector.load = lambda: detector
    frames = iter([np.zeros((16, 16, 3), dtype=np.uint8)] * 2)
    source = SimpleNamespace(read=lambda: next(frames, None))
    monkeypatch.setattr(capture, "VideoSource", lambda _: nullcontext(source))
    monkeypatch.setattr(inference, "YOLODetector", lambda config: detector)
    monkeypatch.setattr(factory, "create_cascade", lambda *args: pytest.fail("Default benchmark must not initialize PPE"))
    monkeypatch.setattr(ml, "synchronize", lambda _: None)
    monkeypatch.setattr(ml, "environment", lambda device: {"device": device})
    report = ml.benchmark_model("person.pt", 0, frames=1, warmup=0, output=str(tmp_path / "benchmark.json"))
    assert report["kind"] == "benchmark"
    assert report["config"]["imgsz"] == 320
    assert report["config"]["ppe_model"] is None
    assert "stage_timings" not in report


@pytest.mark.parametrize("protected,cascade", [("video.avi", False), ("person.pt", False),
                                               ("video.avi", True), ("person.pt", True), ("equipment.pt", True)])
def test_benchmark_output_cannot_replace_input_or_weights_before_model_loading(monkeypatch, tmp_path, protected, cascade):
    from safeguard import factory, inference

    artifacts = {name: tmp_path / name for name in ("video.avi", "person.pt", "equipment.pt")}
    for name, path in artifacts.items():
        path.write_bytes(f"preserve {name}".encode())
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in artifacts.items()}

    def forbidden(*args, **kwargs):
        pytest.fail("Model loading must follow output path validation")

    monkeypatch.setattr(factory, "create_cascade", forbidden)
    monkeypatch.setattr(inference, "YOLODetector", forbidden)
    with pytest.raises(ValueError, match="não podem substituir"):
        ml.benchmark_model(str(artifacts["person.pt"]), str(artifacts["video.avi"]),
                           ppe_model=str(artifacts["equipment.pt"]) if cascade else None,
                           output=str(artifacts[protected]))
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in artifacts.items()} == hashes


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
