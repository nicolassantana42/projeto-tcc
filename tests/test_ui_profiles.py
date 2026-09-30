import pytest

from epi_monitor.ui import profiles


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    person, ppe = tmp_path / "person", tmp_path / "ppe"
    for path, stem in ((person, "yolo11n"), (ppe, "absence")):
        path.mkdir()
        for file in (f"{stem}.xml", f"{stem}.bin", "metadata.yaml"):
            (path / file).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(profiles, "OPENVINO_PERSON", str(person))
    monkeypatch.setattr(profiles, "OPENVINO_PPE", str(ppe))
    monkeypatch.setattr(profiles, "find_spec", lambda name: object())
    return person, ppe


@pytest.mark.parametrize("device, expected", [("cpu", profiles.OPENVINO), ("cuda:0", profiles.PYTORCH), ("mps", profiles.PYTORCH)])
def test_recommended_profile_respects_accelerator_and_complete_artifacts(monkeypatch, artifacts, device, expected):
    monkeypatch.setattr(profiles, "_automatic_device", lambda: device)
    assert profiles.recommended_profile() == expected


def test_incomplete_artifact_or_missing_runtime_keeps_pytorch(monkeypatch, artifacts):
    monkeypatch.setattr(profiles, "_automatic_device", lambda: "cpu")
    person, ppe = artifacts
    (ppe / "absence.bin").write_bytes(b"")
    assert not profiles.openvino_available()
    assert profiles.recommended_profile() == profiles.PYTORCH
    (ppe / "absence.bin").write_bytes(b"weights")
    monkeypatch.setattr(profiles, "find_spec", lambda name: None)
    assert profiles.recommended_profile() == profiles.PYTORCH
