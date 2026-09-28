"""Dataset provenance, safe cache reuse, pagination and retry boundaries."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest


@pytest.fixture
def bootstrap(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_absence_data.py"
    spec = importlib.util.spec_from_file_location("prepare_absence_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: pytest.fail("Network is forbidden in these tests"))
    return module


def metadata(path, content, lfs=False):
    oid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    item = {"type": "file", "path": path, "size": len(content), "oid": oid}
    if lfs:
        item["lfs"] = {"oid": hashlib.sha256(content).hexdigest(), "size": len(content)}
    return item


def plan_item(path, content, lfs=False):
    item = metadata(path, content, lfs)
    return {"path": path, "size": len(content), "algorithm": "sha256" if lfs else "git-blob-sha1",
            "digest": item["lfs"]["oid"] if lfs else item["oid"]}


class Response(io.BytesIO):
    def __init__(self, content, url, link=""):
        super().__init__(content)
        self.url, self.headers = url, {"Link": link}

    def geturl(self):
        return self.url


def source_entries(module):
    files = [metadata(path, b"source") for path in sorted(module.SOURCE_FILES)]
    for split in module.SPLITS:
        files.extend([metadata(f"{split}/images/a.jpg", b"image", True),
                      metadata(f"{split}/labels/a.txt", b"2 0.5 0.5 0.5 0.5")])
    return files


def test_pagination_preserves_pinned_revision_and_digests(bootstrap, monkeypatch):
    files = source_entries(bootstrap)
    next_url = bootstrap.API_URL + "&cursor=abc"
    pages = {bootstrap.API_URL: Response(json.dumps(files[:5]).encode(), bootstrap.API_URL, f'<{next_url}>; rel="next"'),
             next_url: Response(json.dumps(files[5:]).encode(), next_url)}
    monkeypatch.setattr(bootstrap, "_request", lambda url: pages[url])
    plan = bootstrap.load_plan()
    assert len(plan) == 10
    assert {x["algorithm"] for x in plan} == {"sha256", "git-blob-sha1"}
    assert [x["path"] for x in plan] == sorted(x["path"] for x in plan)


@pytest.mark.parametrize("url", ["http://huggingface.co/", "https://example.com/x",
                                 "https://huggingface.co@evil.example/x",
                                 "https://huggingface.co/api/datasets/example/tree/main"])
def test_pagination_cannot_change_host_or_revision(bootstrap, url):
    with pytest.raises(bootstrap.PreparationError, match="paginação"):
        bootstrap._api_url(url)


def test_remote_pagination_link_is_rejected_before_request(bootstrap, monkeypatch):
    calls = []
    def request(url):
        calls.append(url)
        return Response(b"[]", url, '<https://other.example/data>; rel="next"')
    monkeypatch.setattr(bootstrap, "_request", request)
    with pytest.raises(bootstrap.PreparationError, match="paginação"):
        bootstrap.load_plan()
    assert calls == [bootstrap.API_URL]


@pytest.mark.parametrize("path", ["../outside.jpg", "train/images/../../escape.jpg", "/train/images/a.jpg",
                                  "train/images/a.jpg:stream", "train/images/CON.jpg", "train/images/a.jpg.",
                                  "train/images/a\\b.jpg", "train/images/code.py", "unknown/labels/a.txt"])
def test_unsafe_or_unexpected_paths_rejected(bootstrap, tmp_path, path):
    with pytest.raises(bootstrap.PreparationError):
        bootstrap._destination(tmp_path, path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("lfs", [True, False])
def test_cache_is_verified_for_both_lfs_sha256_and_git_blob_sha1(bootstrap, tmp_path, lfs):
    target = tmp_path / "train" / "images" / "a.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"cached data")
    item = plan_item("train/images/a.jpg", b"cached data", lfs)
    result = bootstrap.download_file(tmp_path, item)
    assert result["sha256"] == hashlib.sha256(b"cached data").hexdigest()
    target.write_bytes(b"edited data")
    with pytest.raises(bootstrap.PreparationError, match="preservado"):
        bootstrap.download_file(tmp_path, item)
    assert target.read_bytes() == b"edited data"


def test_download_published_only_after_content_hash_verification(bootstrap, monkeypatch, tmp_path):
    payload = b"verified image"
    monkeypatch.setattr(bootstrap, "_request", lambda url: io.BytesIO(payload))
    item = plan_item("valid/images/a.jpg", payload, True)
    assert bootstrap.download_file(tmp_path, item)["sha256"] == item["digest"]
    assert (tmp_path / item["path"]).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))


def test_corrupt_download_is_not_published(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_request", lambda url: io.BytesIO(b"bad"))
    with pytest.raises(bootstrap.PreparationError, match="Hash"):
        bootstrap.download_file(tmp_path, plan_item("train/images/a.jpg", b"yes", True))
    assert not (tmp_path / "train/images/a.jpg").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_download_size_limit_prevents_publication(bootstrap, monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_request", lambda url: io.BytesIO(b"too large"))
    with pytest.raises(bootstrap.PreparationError, match="excede"):
        bootstrap.download_file(tmp_path, plan_item("train/images/a.jpg", b"ok", True))
    assert not (tmp_path / "train/images/a.jpg").exists()


def test_transient_errors_retry_only_four_times_with_bounded_backoff(bootstrap, monkeypatch):
    sleeps, calls = [], []
    def failing(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 429, "limit", {}, None)
    monkeypatch.setattr(bootstrap, "urlopen", failing)
    monkeypatch.setattr(bootstrap.time, "sleep", sleeps.append)
    with pytest.raises(bootstrap.PreparationError, match="4 tentativas"):
        bootstrap._request(bootstrap.API_URL)
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]


def test_not_found_does_not_retry(bootstrap, monkeypatch):
    def missing(request, timeout):
        raise HTTPError(request.full_url, 404, "missing", {}, None)
    monkeypatch.setattr(bootstrap, "urlopen", missing)
    monkeypatch.setattr(bootstrap.time, "sleep", lambda _: pytest.fail("404 must not retry"))
    with pytest.raises(bootstrap.PreparationError, match="404"):
        bootstrap._request(bootstrap.API_URL)


def test_prepare_only_selected_split_with_reproducible_provenance(bootstrap, monkeypatch, tmp_path):
    payload = b"fixture"
    plan = [plan_item(path, payload) for path in sorted(bootstrap.SOURCE_FILES)]
    plan += [plan_item(f"{split}/{folder}/a.{suffix}", payload, folder == "images")
             for split in bootstrap.SPLITS for folder, suffix in [("images", "jpg"), ("labels", "txt")]]
    monkeypatch.setattr(bootstrap, "load_plan", lambda: plan)
    monkeypatch.setattr(bootstrap, "_request", lambda url: io.BytesIO(payload))
    report = bootstrap.prepare(tmp_path, splits=("valid",))
    assert report["files"] == 6
    assert (tmp_path / "valid/images/a.jpg").exists()
    assert not (tmp_path / "train").exists()
    provenance = json.loads(Path(report["provenance"]).read_text(encoding="utf-8"))
    assert provenance["names"][2] == "no-vest"
    assert provenance["license_declared_by_publisher"] == "CC BY 4.0"
    assert all("sha256" in f for f in provenance["files"])
    assert bootstrap.prepare(tmp_path, splits=("valid",)) == report


def test_existing_metadata_is_not_overwritten(bootstrap, tmp_path):
    path = tmp_path / "source-plan.json"
    path.write_text("user metadata", encoding="utf-8")
    with pytest.raises(bootstrap.PreparationError, match="preservado"):
        bootstrap._save_json(path, {"source": "pinned"})
    assert path.read_text(encoding="utf-8") == "user metadata"
    assert not list(tmp_path.glob("*.part"))


def test_incomplete_image_label_pair_rejected(bootstrap, monkeypatch):
    entries = source_entries(bootstrap)[:-1]
    monkeypatch.setattr(bootstrap, "_request", lambda url: Response(json.dumps(entries).encode(), url))
    with pytest.raises(bootstrap.PreparationError, match="correspondem"):
        bootstrap.load_plan()
