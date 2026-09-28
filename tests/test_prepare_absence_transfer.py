"""Class remapping preserves source data, splits, checkpoint and box geometry."""

import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


NAMES = {0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest",
         5: "Person", 6: "Safety Cone", 7: "Safety Vest", 8: "machinery", 9: "vehicle"}


@pytest.fixture
def bootstrap(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_absence_transfer.py"
    spec = importlib.util.spec_from_file_location("prepare_absence_transfer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "load_names", lambda weights: dict(NAMES))
    return module


@pytest.fixture
def dataset(bootstrap, tmp_path):
    root = tmp_path / "data/datasets/ppe-absence"
    labels = "\n".join(f"{index} 0.5 0.5 0.2 0.3" for index in range(5)) + "\n"
    for index, split in enumerate(("train", "valid", "test")):
        images, annotations = root / split / "images", root / split / "labels"
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)
        Image.new("RGB", (20 + index, 24), (index * 50, 30, 20)).save(images / "frame.png")
        (annotations / "frame.txt").write_text(labels, encoding="utf-8")
    source = tmp_path / "data/ppe-absence.yaml"
    source.write_text(json.dumps({"path": "datasets/ppe-absence", "train": "train/images", "val": "valid/images",
                                  "test": "test/images", "names": list(bootstrap.SOURCE_NAMES)}), encoding="utf-8")
    weights = tmp_path / "models/ppe/absence-base.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake checkpoint, never deserialized")
    return root, source, weights


def test_expected_mapping_matches_actual_model_names(bootstrap):
    assert bootstrap.verify_mapping(dict(enumerate(bootstrap.SOURCE_NAMES)), NAMES) == {0: 0, 1: 2, 2: 4, 3: 5, 4: 7}


@pytest.mark.parametrize("changes", [{4: "unknown"}, {6: "no-vest"}, {2: "NO-Safety Vest", 4: "NO-Hardhat"}])
def test_missing_ambiguous_or_reordered_target_classes_rejected(bootstrap, changes):
    with pytest.raises(bootstrap.PreparationError, match="ausente, ambígua ou em ID"):
        bootstrap.verify_mapping(dict(enumerate(bootstrap.SOURCE_NAMES)), {**NAMES, **changes})


def test_wrong_source_id_order_rejected(bootstrap):
    source = dict(enumerate(bootstrap.SOURCE_NAMES))
    source[1], source[2] = source[2], source[1]
    with pytest.raises(bootstrap.PreparationError, match="RF100"):
        bootstrap.verify_mapping(source, NAMES)


def test_model_with_five_outputs_is_not_silently_accepted(bootstrap):
    with pytest.raises(bootstrap.PreparationError, match="dez"):
        bootstrap.verify_mapping(dict(enumerate(bootstrap.SOURCE_NAMES)), dict(enumerate(bootstrap.SOURCE_NAMES)))


def test_remapping_changes_only_ids_and_retains_numeric_coordinate_tokens(bootstrap):
    source = b"1.0 0.500000 0.4 0.200 0.12\r\n2 0.3 0.6 0.15 0.44\r\n"
    assert bootstrap.remap_labels(source, bootstrap.CLASS_MAPPING) == b"2 0.500000 0.4 0.200 0.12\n4 0.3 0.6 0.15 0.44\n"
    assert bootstrap.remap_labels(b"", bootstrap.CLASS_MAPPING) == b""


@pytest.mark.parametrize("payload", [b"5 0.5 0.5 0.2 0.3", b"1.5 0.5 0.5 0.2 0.3", b"1 nan 0.5 0.2 0.3",
                                    b"1 0.5 0.5 -0.2 0.3", b"1 0.5 0.5 0.2", b"a b c d e"])
def test_invalid_labels_are_not_synthesized_or_silently_skipped(bootstrap, payload):
    with pytest.raises(bootstrap.PreparationError):
        bootstrap.remap_labels(payload, bootstrap.CLASS_MAPPING)


def test_prepare_preserves_images_splits_source_and_weights(bootstrap, dataset):
    original, source, weights = dataset
    before = {str(path): path.read_bytes() for path in original.rglob("*") if path.is_file()}
    weights_before = weights.read_bytes()
    report = bootstrap.prepare()
    assert report["images"] == 3
    assert report["splits"] == {"train": 1, "val": 1, "test": 1}
    assert report["class_id_mapping"] == {0: 0, 1: 2, 2: 4, 3: 5, 4: 7}
    derived = Path(report["directory"])
    for split in ("train", "valid", "test"):
        assert (derived / split / "images/frame.png").read_bytes() == (original / split / "images/frame.png").read_bytes()
        ids = [int(line.split()[0]) for line in (derived / split / "labels/frame.txt").read_text().splitlines()]
        assert ids == [0, 2, 4, 5, 7]
    yaml = json.loads(Path(report["dataset_yaml"]).read_text(encoding="utf-8"))
    assert yaml["names"] == list(NAMES.values())
    assert yaml["val"] == "valid/images"
    provenance = json.loads(Path(report["provenance"]).read_text(encoding="utf-8"))
    assert provenance["unannotated_target_class_ids"] == [1, 3, 6, 8, 9]
    assert all(item["source_label_sha256"] != item["derived_label_sha256"] for item in provenance["files"])
    assert weights.read_bytes() == weights_before
    assert {str(path): path.read_bytes() for path in original.rglob("*") if path.is_file()} == before
    assert bootstrap.prepare() == report
    assert not list(derived.rglob("*.part"))


def test_images_are_independent_copies_of_source(bootstrap, dataset):
    original, _, _ = dataset
    report = bootstrap.prepare()
    destination = Path(report["directory"]) / "train/images/frame.png"
    old = (original / "train/images/frame.png").read_bytes()
    destination.write_bytes(b"derived edit")
    assert (original / "train/images/frame.png").read_bytes() == old


def test_existing_edited_file_stops_preflight_before_any_outputs(bootstrap, dataset):
    path = bootstrap.ROOT / "data/datasets/ppe-absence-transfer/test/labels/frame.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"user labels")
    with pytest.raises(bootstrap.PreparationError, match="preservado"):
        bootstrap.prepare()
    assert path.read_bytes() == b"user labels"
    assert not (bootstrap.ROOT / "data/datasets/ppe-absence-transfer/train").exists()
    assert not (bootstrap.ROOT / "data/ppe-absence-transfer.yaml").exists()


def test_source_and_destination_must_not_overlap(bootstrap, dataset):
    original, _, _ = dataset
    with pytest.raises(bootstrap.PreparationError, match="independentes"):
        bootstrap.prepare(directory=original / "derived")


def test_outside_project_paths_are_rejected(bootstrap):
    with pytest.raises(bootstrap.PreparationError, match="dentro do projeto"):
        bootstrap.checked_path(bootstrap.ROOT.parent / "outside")


def test_invalid_source_audit_prevents_weight_loading_and_writes(bootstrap, dataset, monkeypatch):
    original, _, _ = dataset
    (original / "train/labels/frame.txt").unlink()
    monkeypatch.setattr(bootstrap, "load_names", lambda *_: pytest.fail("Invalid dataset must precede model loading"))
    with pytest.raises(bootstrap.PreparationError, match="auditoria"):
        bootstrap.prepare()
    assert not (bootstrap.ROOT / "data/datasets/ppe-absence-transfer").exists()
