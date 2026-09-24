"""Reproducible PPE bootstrap, without network or untrusted archive writes."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import zipfile

import pytest


@pytest.fixture
def bootstrap(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ppe.py"
    spec = importlib.util.spec_from_file_location("prepare_ppe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        pytest.fail("No real network is permitted in bootstrap tests")

    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden)
    return module


def make_zip(tmp_path, entries):
    path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return path


def test_cached_artifact_is_verified_without_network(bootstrap, tmp_path):
    target = tmp_path / "baseline.pt"
    target.write_bytes(b"cached checkpoint")
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert bootstrap.ensure_download("https://example.invalid/model", target, expected) == target


def test_modified_cache_is_preserved_and_not_redownloaded(bootstrap, tmp_path):
    target = tmp_path / "baseline.pt"
    target.write_bytes(b"user checkpoint")
    with pytest.raises(bootstrap.PreparationError, match="cache"):
        bootstrap.ensure_download("https://example.invalid/model", target, "0" * 64)
    assert target.read_bytes() == b"user checkpoint"


def test_download_is_checked_published_and_leaves_no_temporary_file(bootstrap, monkeypatch, tmp_path):
    payload = b"pinned version"
    calls = []

    def response(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", response)
    target = tmp_path / "model.pt"
    bootstrap.ensure_download("https://example.invalid/pinned/model.pt", target, hashlib.sha256(payload).hexdigest())
    assert target.read_bytes() == payload
    assert calls == [("https://example.invalid/pinned/model.pt", 90)]
    assert list(tmp_path.glob("*.part")) == []


def test_wrong_download_hash_never_becomes_cached_artifact(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"tampered payload"))
    target = tmp_path / "model.pt"
    with pytest.raises(bootstrap.PreparationError, match="SHA256 do download"):
        bootstrap.ensure_download("https://example.invalid/model", target, "0" * 64)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_cleans_temporary_file(bootstrap, monkeypatch, tmp_path):
    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            raise OSError("connection interrupted")

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *args, **kwargs: BrokenResponse())
    with pytest.raises(bootstrap.PreparationError, match="Falha ao baixar"):
        bootstrap.ensure_download("https://example.invalid/model", tmp_path / "model.pt", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_archive_extracts_expected_structure_and_reuses_exact_existing_files(bootstrap, tmp_path):
    archive = make_zip(tmp_path, [("images/train/a.jpg", b"image"), ("labels/train/a.txt", b"0 0.5 0.5 1 1"),
                                 ("data.yaml", b"names: [person]"), ("LICENSE", b"license")])
    destination = tmp_path / "data"
    assert bootstrap.extract_dataset(archive, destination) == {"extracted": 4, "reused": 0}
    extra = destination / "user-notes.txt"
    extra.write_text("preserve", encoding="utf-8")
    assert bootstrap.extract_dataset(archive, destination) == {"extracted": 0, "reused": 4}
    assert extra.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("name", ["../escape.txt", "/tmp/escape.txt", "C:/escape.txt", "images/../../escape.txt",
                                  "images\\..\\escape.txt", "images/a.jpg:stream", "images/CON.txt", "other/a.jpg"])
def test_unsafe_archive_paths_are_rejected_before_any_extraction(bootstrap, tmp_path, name):
    archive = make_zip(tmp_path, [("images/first.jpg", b"must not be written"), (name, b"bad")])
    destination = tmp_path / "output"
    with pytest.raises(bootstrap.PreparationError):
        bootstrap.extract_dataset(archive, destination)
    assert not destination.exists()


def test_zip_symlink_is_rejected(bootstrap, tmp_path):
    info = zipfile.ZipInfo("images/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_zip(tmp_path, [(info, b"../../outside")])
    with pytest.raises(bootstrap.PreparationError, match="Link"):
        bootstrap.extract_dataset(archive, tmp_path / "output")


def test_existing_symlink_or_junction_is_not_followed(bootstrap, monkeypatch, tmp_path):
    archive = make_zip(tmp_path, [("images/a.jpg", b"image")])
    destination = tmp_path / "output"
    linked_directory = destination / "images"
    monkeypatch.setattr(bootstrap, "_is_link", lambda path: path == linked_directory)
    with pytest.raises(bootstrap.PreparationError, match="junction"):
        bootstrap.extract_dataset(archive, destination)
    assert not destination.exists()


def test_modified_dataset_file_stops_preflight_without_overwriting_or_partial_extraction(bootstrap, tmp_path):
    archive = make_zip(tmp_path, [("images/new.jpg", b"new"), ("labels/edited.txt", b"original")])
    destination = tmp_path / "output"
    (destination / "labels").mkdir(parents=True)
    edited = destination / "labels" / "edited.txt"
    edited.write_bytes(b"modified")
    with pytest.raises(bootstrap.PreparationError, match="preservado"):
        bootstrap.extract_dataset(archive, destination)
    assert edited.read_bytes() == b"modified"
    assert not (destination / "images").exists()


def test_file_directory_collision_inside_archive_is_rejected(bootstrap, tmp_path):
    archive = make_zip(tmp_path, [("images/nested", b"file"), ("images/nested/a.jpg", b"image")])
    with pytest.raises(bootstrap.PreparationError, match="arquivo e como diretório"):
        bootstrap.extract_dataset(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_atomic_publish_never_overwrites_a_file_created_concurrently(bootstrap, tmp_path):
    target, temporary = tmp_path / "model.pt", tmp_path / "pending.part"
    target.write_bytes(b"user file")
    temporary.write_bytes(b"downloaded file")
    with pytest.raises(bootstrap.PreparationError, match="preservado"):
        bootstrap._publish_new(temporary, target)
    assert target.read_bytes() == b"user file"


def test_prepare_creates_reproducible_provenance_and_preserves_cache(bootstrap, monkeypatch, tmp_path):
    payload = b"checkpoint fixture"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(bootstrap, "BASELINE_SHA256", digest)
    target = tmp_path / "models" / "ppe" / "baseline-public.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    report = bootstrap.prepare(tmp_path)
    provenance = json.loads(target.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["revision"] in provenance["url"]
    assert provenance["sha256"] == digest
    assert "not trained" in provenance["role"]
    assert "não treinado por este TCC" in report["warning"]
    assert bootstrap.prepare(tmp_path) == report


def test_changed_provenance_is_not_overwritten(bootstrap, tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text('{"sha256": "user"}', encoding="utf-8")
    with pytest.raises(bootstrap.PreparationError, match="divergente"):
        bootstrap._save_provenance(path, {"sha256": "expected"})
    assert json.loads(path.read_text(encoding="utf-8"))["sha256"] == "user"
