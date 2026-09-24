"""Dataset audits use local generated images; no model, camera or network."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest
import yaml

from safeguard.dataset_audit import DatasetAuditError, audit_dataset, main


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "assets"
    for index, split in enumerate(("train", "val", "test")):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (32, 24), (index * 70, 100, 150)).save(images / "one.png")
        (labels / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n1 0.5 0.25 0.2 0.2\n", encoding="utf-8")
    source = tmp_path / "config" / "ppe.yaml"
    source.parent.mkdir()
    source.write_text("path: ../assets\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [person, helmet]\nnc: 2\n", encoding="utf-8")
    return source, root


def codes(report):
    return {issue["code"] for issue in report["errors"]}


def test_valid_dataset_resolves_yaml_root_independently_of_cwd_and_reports_counts(dataset, tmp_path, monkeypatch):
    source, root = dataset
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "results" / "audit.json"
    report = audit_dataset(source, require_test=True, output=output)
    assert report["valid"]
    assert report["dataset_root"] == str(root)
    assert report["summary"]["images"] == report["summary"]["valid_images"] == 3
    assert report["summary"]["instances"] == 6
    assert report["splits"]["train"]["class_counts"] == {
        "0": {"name": "person", "images": 1, "instances": 1},
        "1": {"name": "helmet", "images": 1, "instances": 1},
    }
    assert len(report["dataset_yaml_sha256"]) == 64
    assert len(report["splits"]["train"]["files"][0]["sha256"]) == 64
    assert report["splits"]["train"]["files"][0]["dimensions"] == [32, 24]
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_yaml_without_path_is_relative_to_its_directory(dataset):
    source, root = dataset
    source = root / "dataset.yaml"
    source.write_text("train: images/train\nval: images/val\nnames: {0: person, 1: helmet}\n", encoding="utf-8")
    report = audit_dataset(source)
    assert report["valid"]
    assert report["resolved_splits"]["train"] == [str(root / "images" / "train" / "one.png")]


def test_manifests_and_yaml_lists_are_resolved_locally(dataset):
    source, root = dataset
    manifests = root / "manifests"
    manifests.mkdir()
    (manifests / "train.txt").write_text("# own local image\n../images/train/one.png\n", encoding="utf-8")
    config = yaml.safe_load(source.read_text())
    config["train"] = ["manifests/train.txt"]
    config["val"] = str(root / "images" / "val")
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_dataset(source)
    assert report["valid"]
    assert report["resolved_splits"]["train"] == [str(root / "images" / "train" / "one.png")]


def test_test_split_is_required_only_on_request(dataset):
    source, _ = dataset
    source.write_text(source.read_text().replace("test: images/test\n", ""), encoding="utf-8")
    assert audit_dataset(source)["valid"]
    report = audit_dataset(source, require_test=True)
    assert not report["valid"]
    assert any(error["code"] == "missing_split" and error["split"] == "test" for error in report["errors"])


def test_empty_configured_split_is_an_error(dataset):
    source, root = dataset
    (root / "images" / "test" / "one.png").unlink()
    assert "empty_split" in codes(audit_dataset(source))


def test_empty_labels_are_explicit_background_but_missing_labels_require_opt_in(dataset):
    source, root = dataset
    (root / "labels" / "train" / "one.txt").write_text("\n \n", encoding="utf-8")
    (root / "labels" / "val" / "one.txt").unlink()
    strict = audit_dataset(source)
    assert "missing_label" in codes(strict)
    assert strict["splits"]["train"]["background_images"] == strict["splits"]["train"]["empty_labels"] == 1
    assert strict["splits"]["val"]["missing_labels"] == 1
    assert strict["splits"]["val"]["background_images"] == 0
    permitted = audit_dataset(source, allow_background=True)
    assert permitted["valid"]
    assert permitted["splits"]["val"]["missing_labels"] == permitted["splits"]["val"]["background_images"] == 1
    assert any(item["code"] == "missing_label_as_background" for item in permitted["warnings"])


@pytest.mark.parametrize("line", [
    "0.5 0.5 0.5 0.2 0.2",  # Fractional class
    "2 0.5 0.5 0.2 0.2",  # Unknown class
    "-1 0.5 0.5 0.2 0.2",
    "0 nan 0.5 0.2 0.2",
    "0 0.5 0.5 inf 0.2",
    "0 0.5 0.5 0 0.2",
    "0 0.5 0.5 -0.2 0.2",
    "0 1.2 0.5 0.2 0.2",
    "0 0.1 0.5 0.4 0.2",  # Center valid, corner outside image
    "0 0.5 0.9 0.2 0.4",
    "0 0.5 0.5 0.2",
    "0 0.5 0.5 0.2 0.2 0.8 0.8",
    "person 0.5 0.5 0.2 0.2",
])
def test_invalid_detection_labels_are_reported_with_file_and_line(dataset, line):
    source, root = dataset
    label = root / "labels" / "train" / "one.txt"
    label.write_text("\n" + line + "\n", encoding="utf-8")
    report = audit_dataset(source)
    error = next(error for error in report["errors"] if error["code"] == "invalid_label")
    assert error["path"] == str(label)
    assert error["line"] == 2
    assert report["splits"]["train"]["invalid_labels"] == 1
    assert report["splits"]["train"]["background_images"] == 0


def test_full_image_box_and_integral_float_class_are_valid(dataset):
    source, root = dataset
    (root / "labels" / "train" / "one.txt").write_text("0.0 0.5 0.5 1 1\n", encoding="utf-8")
    assert audit_dataset(source)["valid"]


def test_unreadable_label_is_not_counted_as_background(dataset):
    source, root = dataset
    (root / "labels" / "train" / "one.txt").write_bytes(b"\xff\xff")
    report = audit_dataset(source, allow_background=True)
    assert "unreadable_label" in codes(report)
    assert report["splits"]["train"]["background_images"] == 0


@pytest.mark.parametrize("operation", ["corrupt", "truncated", "missing", "too_small"])
def test_bad_images_are_errors_without_crashing(dataset, operation):
    source, root = dataset
    image = root / "images" / "train" / "one.png"
    if operation == "corrupt":
        image.write_bytes(b"not an image")
    elif operation == "truncated":
        image.write_bytes(image.read_bytes()[:45])
    elif operation == "too_small":
        Image.new("RGB", (3, 3), "red").save(image)
    else:
        image.unlink()
        manifest = root / "train.txt"
        manifest.write_text("images/train/one.png\n", encoding="utf-8")
        source.write_text(source.read_text().replace("train: images/train", "train: train.txt"), encoding="utf-8")
    report = audit_dataset(source)
    assert "invalid_image" in codes(report)
    assert report["splits"]["train"]["corrupt_images"] == 1
    assert report["splits"]["train"]["valid_images"] == 0


def test_exact_duplicates_between_splits_are_leakage_even_with_different_names(dataset):
    source, root = dataset
    first = root / "images" / "train" / "one.png"
    copied = root / "images" / "val" / "renamed.png"
    shutil.copyfile(first, copied)
    (root / "labels" / "val" / "renamed.txt").write_text("", encoding="utf-8")
    report = audit_dataset(source)
    assert "split_leakage" in codes(report)
    assert len(report["leakage"]) == 1
    assert {item["split"] for item in report["leakage"][0]["files"]} == {"train", "val"}


def test_identical_bytes_inside_one_split_are_a_warning(dataset):
    source, root = dataset
    directory = root / "images" / "train"
    shutil.copyfile(directory / "one.png", directory / "duplicate.png")
    (root / "labels" / "train" / "duplicate.txt").write_text("", encoding="utf-8")
    report = audit_dataset(source)
    assert report["valid"]
    assert len(report["duplicates_within_split"]) == 1
    assert report["leakage"] == []


@pytest.mark.parametrize("declaration", [
    "names: []", "names: {1: person}", "names: [person, PERSON]",
    "names: {true: person}", "names: {0.5: person}", "names: [person, null]",
    "names: {0: person, '0': helmet}", "names: [person]\nnc: 2",
])
def test_invalid_class_schema_fails(dataset, declaration):
    source, _ = dataset
    source.write_text("path: ../assets\ntrain: images/train\nval: images/val\n" + declaration, encoding="utf-8")
    report = audit_dataset(source)
    assert not report["valid"]
    assert codes(report) & {"invalid_classes", "class_count_mismatch"}


@pytest.mark.parametrize("contents", [
    "[broken", "[]", "names: [person]\nnames: [helmet]", "!!python/object/apply:os.system ['exit 99']",
    "names: [person]\ndownload: echo must-never-execute",
    "names: [person]\npath: https://example.invalid/dataset",
])
def test_unsafe_or_invalid_yaml_returns_failure_report(tmp_path, contents):
    source = tmp_path / "bad.yaml"
    source.write_text(contents, encoding="utf-8")
    output = tmp_path / "audit.json"
    report = audit_dataset(source, output=output)
    assert "invalid_yaml" in codes(report)
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is False


def test_remote_split_or_manifest_entries_never_attempt_a_download(dataset, monkeypatch):
    source, root = dataset
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network request"))
    (root / "train.txt").write_text("https://example.invalid/image.png\n", encoding="utf-8")
    source.write_text("path: ../assets\ntrain: train.txt\nval: https://example.invalid/val.zip\nnames: [person]\n", encoding="utf-8")
    report = audit_dataset(source)
    assert "invalid_manifest_entry" in codes(report)
    assert "invalid_split_path" in codes(report)


def test_missing_yaml_still_writes_an_actionable_report(tmp_path):
    report = audit_dataset(tmp_path / "absent.yaml", output=tmp_path / "audit.json")
    assert report["summary"]["errors"] == 1
    assert "invalid_yaml" in codes(report)


@pytest.mark.parametrize("target", ["yaml", "image", "label", "manifest"])
def test_report_cannot_overwrite_dataset_inputs(dataset, target):
    source, root = dataset
    manifest = root / "train.txt"
    manifest.write_text("images/train/one.png", encoding="utf-8")
    source.write_text(source.read_text().replace("train: images/train", "train: train.txt"), encoding="utf-8")
    destination = {"yaml": source, "image": root / "images/train/one.png", "label": root / "labels/train/one.txt", "manifest": manifest}[target]
    original = destination.read_bytes()
    with pytest.raises(DatasetAuditError, match="não pode substituir"):
        audit_dataset(source, output=destination)
    assert destination.read_bytes() == original


def test_cli_has_nonzero_exit_on_invalid_dataset_and_json_report(dataset, tmp_path, capsys):
    source, root = dataset
    output = tmp_path / "audit.json"
    assert main(["--data", str(source), "--output", str(output), "--require-test"]) == 0
    (root / "labels" / "val" / "one.txt").unlink()
    assert main(["--data", str(source), "--output", str(output)]) == 1
    assert "missing_label" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is False


def test_standalone_script_runs_with_local_dataset(dataset, tmp_path):
    source, _ = dataset
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
    result = subprocess.run([sys.executable, "-X", "utf8", str(script), "--data", str(source), "--output", str(tmp_path / "audit.json")],
                            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 inst" in result.stdout
