"""Final-box cascade metrics use local fixtures and fake pipeline outputs only."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from safeguard.config import InferenceConfig
import safeguard.evaluation as evaluation
from safeguard.evaluation import EvaluationError, evaluate_cascade
from safeguard.types import Detection, FrameResult


def detection(label="person", score=.9, bbox=(30, 30, 70, 70)):
    return Detection(0, label, score, bbox)


class FakePipeline:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.person_detector = SimpleNamespace(device="cpu", config=InferenceConfig("fake-person.pt"))
        self.ppe_detector = SimpleNamespace(device="cpu", config=InferenceConfig("fake-ppe.pt"))

    def process(self, frame, confidence, iou):
        self.calls.append({"shape": frame.shape, "confidence": confidence, "iou": iou})
        items = next(self.outputs)
        return FrameResult(frame, items, {}, [], 1, 2, len(self.calls),
                           stage_timings_ms={"person": 1.0, "ppe": 2.0})


def dataset(tmp_path, annotations, names=None):
    root = tmp_path / "dataset"
    images, labels = root / "images" / "test", root / "labels" / "test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index, annotation in enumerate(annotations):
        assert cv2.imwrite(str(images / f"{index:03d}.png"), np.full((100, 100, 3), index, np.uint8))
        if annotation is not None:
            (labels / f"{index:03d}.txt").write_text(annotation, encoding="utf-8")
    source = tmp_path / "dataset.yaml"
    source.write_text(yaml.safe_dump({"path": "dataset", "test": "images/test", "names": names or ["person"]}), encoding="utf-8")
    return source


def test_duplicate_predictions_count_once_and_keep_fp_for_inspection(tmp_path):
    data = dataset(tmp_path, ["0 .5 .5 .4 .4\n"])
    pipeline = FakePipeline([[detection(score=.6), detection(score=.95)]])
    output = tmp_path / "output" / "evaluation.json"
    report = evaluate_cascade(pipeline, data, confidence=.5, iou=.3, output=output)
    assert report["per_class"]["person"] == {
        "tp": 1, "fp": 1, "fn": 0, "ground_truth": 1, "predictions": 2, "precision": .5, "recall": 1,
    }
    assert report["errors"][0]["type"] == "fp"
    assert report["errors"][0]["confidence"] == .6
    assert Path(report["errors"][0]["image"]).is_file()
    assert report["errors"][0]["bbox"] == [30, 30, 70, 70]
    assert pipeline.calls == [{"shape": (100, 100, 3), "confidence": .5, "iou": .3}]
    assert json.loads(output.read_text(encoding="utf-8"))["micro"] == report["micro"]
    assert report["pipeline_latency_ms"]["samples"] == 1
    assert report["pipeline_latency_ms"]["p95"] >= 0
    assert report["stage_latency_ms"]["person"]["p50"] == 1
    assert report["environment"]["stages"]["ppe_detector"]["device"] == "cpu"
    assert report["dataset"]["held_out_verified"] is False
    assert "mAP" not in report["micro"]


def test_missed_person_gate_produces_false_negatives_in_final_output(tmp_path):
    data = dataset(tmp_path, ["0 .5 .5 .4 .4\n1 .5 .35 .2 .1\n"], ["Person", "Hardhat"])
    report = evaluate_cascade(FakePipeline([[]]), data, output=None)
    assert report["micro"]["tp"] == report["micro"]["fp"] == 0
    assert report["micro"]["fn"] == 2
    assert report["micro"]["precision"] is None
    assert report["micro"]["recall"] == 0
    assert {item["canonical_class"] for item in report["errors"]} == {"person", "helmet"}


def test_class_mismatch_is_fp_and_fn_despite_identical_box(tmp_path):
    data = dataset(tmp_path, ["1 .5 .5 .4 .4\n"], ["person", "helmet", "vest"])
    report = evaluate_cascade(FakePipeline([[detection("safety vest")]]), data, output=None)
    assert report["per_class"]["helmet"]["fn"] == 1
    assert report["per_class"]["vest"]["fp"] == 1
    assert report["micro"]["tp"] == 0


def test_empty_and_missing_labels_count_background_with_defined_precision_policy(tmp_path):
    data = dataset(tmp_path, ["", None])
    report = evaluate_cascade(FakePipeline([[], [detection()]]), data, output=None)
    assert report["dataset"]["background_images"] == 2
    assert len(report["dataset"]["missing_label_files"]) == 1
    assert len(report["dataset"]["empty_label_files"]) == 1
    assert report["micro"]["fp"] == 1
    assert report["micro"]["precision"] == 0
    assert report["micro"]["recall"] is None


def test_fully_empty_predictions_and_ground_truth_do_not_invent_perfect_scores(tmp_path):
    data = dataset(tmp_path, [""])
    report = evaluate_cascade(FakePipeline([[]]), data, output=None)
    assert report["micro"]["tp"] == report["micro"]["fp"] == report["micro"]["fn"] == 0
    assert report["micro"]["precision"] is report["micro"]["recall"] is None
    assert report["errors"] == []


def test_only_declared_canonical_classes_are_scored_none_is_not_no_vest(tmp_path):
    names = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "none",
             6: "Person", 7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"}
    data = dataset(tmp_path, ["7 .5 .5 .4 .4\n5 .5 .5 .4 .4\n"], names)
    report = evaluate_cascade(FakePipeline([[detection("NO-Hardhat"), detection("no_vest"), detection("gloves")]]), data, output=None)
    assert report["coverage"]["scored_classes"] == ["person", "helmet", "vest", "no_helmet"]
    assert report["coverage"]["unannotated_canonical_classes"] == ["no_vest"]
    assert report["coverage"]["ignored_predictions"] == {"total": 2, "by_label": {"no_vest": 1, "gloves": 1}}
    assert report["coverage"]["ignored_ground_truth"] == {"total": 1, "by_label": {"none": 1}}
    assert report["per_class"]["no_helmet"]["tp"] == 1
    assert report["micro"]["fp"] == 0


def test_confidence_and_matching_iou_are_respected(tmp_path):
    data = dataset(tmp_path, ["0 .5 .5 .4 .4\n"])
    report = evaluate_cascade(FakePipeline([[detection(score=.2), detection(bbox=(30, 30, 50, 70))]]),
                              data, confidence=.4, match_iou=.6, output=None)
    assert report["micro"]["tp"] == 0
    assert report["micro"]["fp"] == report["micro"]["fn"] == 1
    assert report["coverage"]["predictions_below_confidence"] == 1


def test_content_hash_and_first_n_sampling_are_reproducible(tmp_path):
    data = dataset(tmp_path, ["0 .5 .5 .4 .4\n", "", ""])
    first = evaluate_cascade(FakePipeline([[]]), data, max_images=1, output=None)
    repeated = evaluate_cascade(FakePipeline([[]]), data, max_images=1, output=None)
    assert first["dataset"]["evaluated_set_sha256"] == repeated["dataset"]["evaluated_set_sha256"]
    assert first["dataset"]["available_images"] == 3 and first["dataset"]["evaluated_images"] == 1
    label = tmp_path / "dataset" / "labels" / "test" / "000.txt"
    label.write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
    changed = evaluate_cascade(FakePipeline([[]]), data, max_images=1, output=None)
    assert changed["dataset"]["evaluated_set_sha256"] != first["dataset"]["evaluated_set_sha256"]


def test_local_image_lists_are_relative_to_list_file_and_deduplicated(tmp_path, monkeypatch):
    data = dataset(tmp_path, [""])
    listing = tmp_path / "dataset" / "test.txt"
    listing.write_text("./images/test/000.png\n./images/test/000.png\n", encoding="utf-8")
    configuration = yaml.safe_load(data.read_text(encoding="utf-8"))
    configuration["test"] = "test.txt"
    data.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    report = evaluate_cascade(FakePipeline([[]]), data, output=None)
    assert report["dataset"]["evaluated_images"] == 1


def test_error_list_is_bounded_without_losing_metric_counts(tmp_path, monkeypatch):
    data = dataset(tmp_path, ["0 .5 .5 .4 .4\n"])
    monkeypatch.setattr(evaluation, "MAX_ERROR_RECORDS", 2)
    report = evaluate_cascade(FakePipeline([[detection() for _ in range(5)]]), data, output=None)
    assert report["micro"]["fp"] == report["errors_total"] == 4
    assert len(report["errors"]) == 2 and report["errors_truncated"] is True


@pytest.mark.parametrize("annotation", ["9 .5 .5 .4 .4", "0 nan .5 .4 .4", "0 .5 .5 .4", "0 .1 .5 .8 .4"])
def test_invalid_annotations_fail_instead_of_silently_becoming_background(tmp_path, annotation):
    data = dataset(tmp_path, [annotation])
    pipeline = FakePipeline([])
    with pytest.raises(EvaluationError, match="Anotação YOLO inválida"):
        evaluate_cascade(pipeline, data, output=None)
    assert not pipeline.calls


@pytest.mark.parametrize("parameters", [{"max_images": 0}, {"max_images": True}, {"match_iou": 0}, {"confidence": float("nan")}, {"split": "unknown"}])
def test_invalid_parameters_are_rejected_before_loading_models(parameters):
    with pytest.raises(ValueError):
        evaluate_cascade(FakePipeline([]), "missing.yaml", output=None, **parameters)


def test_empty_split_is_an_error_not_a_successful_zero_image_evaluation(tmp_path):
    data = dataset(tmp_path, [])
    with pytest.raises(EvaluationError, match="não contém imagens"):
        evaluate_cascade(FakePipeline([]), data, output=None)


def test_corrupt_image_is_an_error(tmp_path):
    data = dataset(tmp_path, [""])
    (tmp_path / "dataset" / "images" / "test" / "000.png").write_bytes(b"not a picture")
    with pytest.raises(EvaluationError, match="corrompida"):
        evaluate_cascade(FakePipeline([]), data, output=None)


def test_integral_float_class_id_matches_dataset_audit_contract(tmp_path):
    data = dataset(tmp_path, ["0.0 .5 .5 .4 .4\n"])
    report = evaluate_cascade(FakePipeline([[detection()]]), data, output=None)
    assert report["micro"]["tp"] == 1


@pytest.mark.parametrize("identifier", ["0.5", "-1", "nan", "inf"])
def test_nonintegral_or_nonfinite_class_ids_are_rejected(tmp_path, identifier):
    data = dataset(tmp_path, [f"{identifier} .5 .5 .4 .4\n"])
    with pytest.raises(EvaluationError, match="Anotação YOLO inválida"):
        evaluate_cascade(FakePipeline([]), data, output=None)


def test_manifest_comments_match_dataset_audit_contract(tmp_path):
    data = dataset(tmp_path, [""])
    listing = tmp_path / "dataset" / "test.txt"
    listing.write_text("# Evaluation sample\n  # comment\nimages/test/000.png\n", encoding="utf-8")
    configuration = yaml.safe_load(data.read_text(encoding="utf-8"))
    configuration["test"] = "test.txt"
    data.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    report = evaluate_cascade(FakePipeline([[]]), data, output=None)
    assert report["dataset"]["evaluated_images"] == 1


@pytest.mark.parametrize("target", ["yaml", "image", "label", "manifest", "unsampled_image", "unsampled_label"])
def test_report_never_overwrites_dataset_inputs_even_outside_selected_sample(tmp_path, target):
    data = dataset(tmp_path, ["", ""])
    root = tmp_path / "dataset"
    listing = root / "test.txt"
    listing.write_text("images/test/000.png\nimages/test/001.png\n", encoding="utf-8")
    configuration = yaml.safe_load(data.read_text(encoding="utf-8"))
    configuration["test"] = "test.txt"
    data.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    destination = {
        "yaml": data, "manifest": listing, "image": root / "images/test/000.png",
        "label": root / "labels/test/000.txt", "unsampled_image": root / "images/test/001.png",
        "unsampled_label": root / "labels/test/001.txt",
    }[target]
    original = destination.read_bytes()
    pipeline = FakePipeline([])
    with pytest.raises(EvaluationError, match="não pode substituir"):
        evaluate_cascade(pipeline, data, max_images=1, output=destination)
    assert not pipeline.calls
    assert destination.read_bytes() == original


@pytest.mark.parametrize("backend", ["pt", "onnx", "openvino", "xml"])
def test_report_never_overwrites_evaluated_model(tmp_path, backend):
    data = dataset(tmp_path, [""])
    if backend == "openvino":
        model = tmp_path / "ppe_openvino_model"
        model.mkdir()
        destination = model / "metadata.yaml"
    elif backend == "xml":
        model = tmp_path / "ppe.xml"
        destination = model.with_suffix(".bin")
    else:
        model = destination = tmp_path / f"ppe.{backend}"
    destination.write_bytes(b"original model data")
    pipeline = FakePipeline([])
    pipeline.ppe_detector.config = InferenceConfig(str(model))
    with pytest.raises(EvaluationError, match="não pode substituir"):
        evaluate_cascade(pipeline, data, output=destination)
    assert not pipeline.calls
    assert destination.read_bytes() == b"original model data"


@pytest.mark.parametrize("invalid_detection", [
    detection(score=None), detection(bbox=None), detection(bbox=("invalid", 1, 2, 3)),
    detection(score=float("nan")), detection(bbox=(3, 3, 2, 2)),
])
def test_malformed_predictions_raise_evaluation_error(tmp_path, invalid_detection):
    data = dataset(tmp_path, [""])
    with pytest.raises(EvaluationError, match="confiança/caixa inválida"):
        evaluate_cascade(FakePipeline([[invalid_detection]]), data, output=None)
