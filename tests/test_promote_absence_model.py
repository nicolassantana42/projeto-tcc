"""Promotion preserves historical weights and rejects mismatched artifacts."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from epi_monitor import config, factory, runner
from epi_monitor.cli import parser


NAMES = {0: "Hardhat", 1: "Safety Vest", 2: "NO-Hardhat", 3: "NO-Safety Vest"}


@pytest.fixture
def promoter(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("promote_absence_model", scripts / "promote_absence_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_names", lambda _: dict(NAMES))
    return module


@pytest.fixture
def training(tmp_path):
    directory = tmp_path / "runs/train/ppe_absence"
    (directory / "weights").mkdir(parents=True)
    weights = directory / "weights/best.pt"
    weights.write_bytes(b"new locally trained checkpoint")
    report = {"kind": "training", "provenance": {"best_weights": {"sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}},
              "per_class": [{"class_id": key, "name": name} for key, name in NAMES.items()]}
    (directory / "training.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "models/ppe").mkdir(parents=True)
    (tmp_path / "models/ppe/best.pt").write_bytes(b"historical 11-class weights")
    (tmp_path / "models/ppe/best_int8_openvino_model").mkdir()
    (tmp_path / "models/ppe/best_int8_openvino_model/best.xml").write_bytes(b"historical INT8")
    return directory


def test_publish_and_repeat_preserve_historical_models(promoter, training, tmp_path):
    first = promoter.promote(training, root=tmp_path)
    assert promoter.promote(training, root=tmp_path) == first
    assert Path(first["model"]).read_bytes() == (training / "weights/best.pt").read_bytes()
    provenance = json.loads(Path(first["provenance"]).read_text(encoding="utf-8"))
    assert provenance["weights_sha256"] == first["sha256"]
    assert (tmp_path / "models/ppe/best.pt").read_bytes() == b"historical 11-class weights"
    assert (tmp_path / "models/ppe/best_int8_openvino_model/best.xml").read_bytes() == b"historical INT8"
    assert not list((tmp_path / "models/ppe").glob("*.part"))


@pytest.mark.parametrize("conflict", ["weights", "provenance"])
def test_divergent_existing_destinations_are_preserved(promoter, training, tmp_path, conflict):
    destination = tmp_path / "models/ppe" / ("absence.pt" if conflict == "weights" else "absence.provenance.json")
    destination.write_bytes(b"user artifact")
    with pytest.raises(promoter.PreparationError, match="divergente"):
        promoter.promote(training, root=tmp_path)
    assert destination.read_bytes() == b"user artifact"
    other = destination.with_name("absence.provenance.json" if conflict == "weights" else "absence.pt")
    assert not other.exists()


@pytest.mark.parametrize("issue", ["missing_report", "wrong_hash", "wrong_names", "duplicate_names", "no_vest_missing"])
def test_invalid_provenance_or_classes_never_publish(promoter, training, tmp_path, monkeypatch, issue):
    report_path = training / "training.json"
    report = json.loads(report_path.read_text())
    if issue == "missing_report":
        report_path.unlink()
    elif issue == "wrong_hash":
        (training / "weights/best.pt").write_bytes(b"modified checkpoint")
    elif issue == "wrong_names":
        monkeypatch.setattr(promoter, "load_names", lambda _: {**NAMES, 3: "unexpected"})
    elif issue == "duplicate_names":
        report["per_class"].append(report["per_class"][0])
        report_path.write_text(json.dumps(report))
    else:
        reduced = {key: name for key, name in NAMES.items() if key != 3}
        monkeypatch.setattr(promoter, "load_names", lambda _: reduced)
        report["per_class"] = [{"class_id": key, "name": name} for key, name in reduced.items()]
        report_path.write_text(json.dumps(report))
    with pytest.raises(promoter.PreparationError):
        promoter.promote(training, root=tmp_path)
    assert not (tmp_path / config.DEFAULT_PPE_MODEL).exists()
    assert not (tmp_path / config.DEFAULT_PPE_MODEL).with_suffix(".provenance.json").exists()


def test_all_detection_entry_points_share_new_default():
    assert config.DEFAULT_PPE_MODEL == "models/ppe/absence.pt"
    for function in (factory.create_cascade, runner.run_detection):
        parameters = inspect.signature(function).parameters
        assert parameters["ppe_model"].default == config.DEFAULT_PPE_MODEL
        assert parameters["person_model"].default == config.DEFAULT_PERSON_MODEL
    for args in (["detect", "--source", "frame.png"], ["evaluate-cascade", "--data", "dataset.yaml"]):
        parsed = parser().parse_args(args)
        assert parsed.ppe_model == config.DEFAULT_PPE_MODEL
        assert parsed.person_model == config.DEFAULT_PERSON_MODEL
