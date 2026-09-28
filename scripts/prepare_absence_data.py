"""Fetch the pinned RF100 PPE dataset, verifying every file before publication.

Uses only the standard library. No models or executable third-party code are
downloaded. The upstream class IDs, labels and split assignments are preserved.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "LibreYOLO/construction-safety-gsnvb"
REVISION = "342e545489a6b5f76d6c8225f1ef2629c5a4770a"
API_PATH = f"/api/datasets/{REPOSITORY}/tree/{REVISION}"
API_URL = f"https://huggingface.co{API_PATH}?recursive=true&limit=1000"
DOWNLOAD_BASE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
SPLITS = ("train", "valid", "test")
NAMES = ("helmet", "no-helmet", "no-vest", "person", "vest")
SOURCE_FILES = {"README.md", "README.dataset.txt", "README.roboflow.txt", "data.yaml"}
MAX_FILE_BYTES = 20 * 1024**2
MAX_DATASET_BYTES = 300 * 1024**2
MAX_FILES = 10000


class PreparationError(RuntimeError):
    """The requested artifact cannot be prepared without changing existing data."""


def _check_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PreparationError(f"Link ou junction não permitido: {candidate}")


def _destination(directory: Path, raw: str) -> Path:
    if not isinstance(raw, str):
        raise PreparationError("Caminho remoto inválido.")
    parts = raw.split("/")
    devices = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(not p or p in {".", ".."} or re.search(r'[\\:\x00<>"|?*]', p)
           or p.endswith((".", " ")) or p.split(".")[0].upper() in devices for p in parts):
        raise PreparationError(f"Caminho remoto inseguro: {raw!r}")
    if raw not in SOURCE_FILES:
        valid = (len(parts) == 3 and parts[0] in SPLITS and
                 ((parts[1] == "labels" and PurePosixPath(raw).suffix == ".txt") or
                  (parts[1] == "images" and PurePosixPath(raw).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})))
        if not valid:
            raise PreparationError(f"Arquivo inesperado no dataset: {raw!r}")
    result = directory.joinpath(*parts)
    _check_path(result)
    if not result.resolve().is_relative_to(directory.resolve()):
        raise PreparationError(f"Caminho escapa do dataset: {raw!r}")
    return result


def _request(url: str):
    """Four bounded attempts; never retry permanent authorization/not-found errors."""
    for attempt in range(4):
        try:
            return urlopen(Request(url, headers={"User-Agent": "TCC-PPE-dataset-preparation/1.0"}), timeout=60)
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code <= 599:
                raise PreparationError(f"Download HTTP {error.code}: {url}") from error
            error.close()
            if attempt == 3:
                raise PreparationError(f"Download HTTP {error.code} após 4 tentativas: {url}") from error
        except (URLError, OSError) as error:
            if attempt == 3:
                raise PreparationError(f"Falha de rede após 4 tentativas: {url}") from error
        time.sleep(2 ** (attempt + 1))
    raise AssertionError("Unreachable")


def _api_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co" or
            parsed.path != API_PATH or parsed.fragment):
        raise PreparationError("A paginação deve permanecer na API e revisão fixadas do Hugging Face.")


def load_plan() -> list[dict]:
    """Resolve file metadata from the immutable source revision, never from main."""
    url, pages, entries = API_URL, set(), []
    while url:
        _api_url(url)
        if url in pages or len(pages) >= 20:
            raise PreparationError("Paginação repetida ou excessiva na API.")
        pages.add(url)
        with _request(url) as response:
            _api_url(response.geturl())
            payload = response.read(4 * 1024**2 + 1)
            if len(payload) > 4 * 1024**2:
                raise PreparationError("Resposta de metadados excessiva.")
            try:
                page = json.loads(payload)
            except (ValueError, UnicodeDecodeError) as error:
                raise PreparationError("Metadados JSON inválidos.") from error
            if not isinstance(page, list):
                raise PreparationError("A API não retornou uma lista de arquivos.")
            entries.extend(page)
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        url = match.group(1) if match else None
    files, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PreparationError("Entrada inválida nos metadados.")
        if entry.get("type") == "directory" or entry.get("path") == ".gitattributes":
            continue
        if entry.get("type") != "file":
            raise PreparationError("Tipo de arquivo remoto inesperado.")
        path = entry.get("path")
        _destination(ROOT / "data" / "datasets" / "ppe-absence", path)
        if path.casefold() in seen:
            raise PreparationError(f"Arquivo repetido na API: {path}")
        seen.add(path.casefold())
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_FILE_BYTES:
            raise PreparationError(f"Tamanho remoto inválido: {path}")
        lfs = entry.get("lfs")
        algorithm, digest = ("sha256", lfs.get("oid")) if isinstance(lfs, dict) else ("git-blob-sha1", entry.get("oid"))
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}" if algorithm == "sha256" else r"[0-9a-f]{40}", digest):
            raise PreparationError(f"Hash remoto inválido: {path}")
        if isinstance(lfs, dict) and lfs.get("size") != size:
            raise PreparationError(f"Tamanho LFS divergente: {path}")
        files.append({"path": path, "size": size, "algorithm": algorithm, "digest": digest})
    if len(files) > MAX_FILES or sum(f["size"] for f in files) > MAX_DATASET_BYTES:
        raise PreparationError("Dataset excede os limites previstos.")
    if not SOURCE_FILES.issubset({f["path"] for f in files}):
        raise PreparationError("Metadados não incluem os arquivos de origem/licença esperados.")
    for split in SPLITS:
        images = {PurePosixPath(f["path"]).stem for f in files if f["path"].startswith(f"{split}/images/")}
        labels = {PurePosixPath(f["path"]).stem for f in files if f["path"].startswith(f"{split}/labels/")}
        if not images or images != labels:
            raise PreparationError(f"Imagens e labels não correspondem no split {split}.")
    return sorted(files, key=lambda f: f["path"])


def _hashes(path: Path, item: dict) -> dict:
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise PreparationError(f"Tamanho inesperado; arquivo preservado: {path}")
    sha256, git = hashlib.sha256(), hashlib.sha1()
    git.update(f"blob {item['size']}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            git.update(chunk)
    hashes = {"sha256": sha256.hexdigest(), "git-blob-sha1": git.hexdigest()}
    if hashes[item["algorithm"]] != item["digest"]:
        raise PreparationError(f"Hash inesperado; arquivo preservado: {path}")
    return hashes


def _publish(temporary: Path, destination: Path) -> None:
    _check_path(destination)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or destination.read_bytes() != temporary.read_bytes():
            raise PreparationError(f"Arquivo existente diferente; preservado: {destination}")


def _save_json(destination: Path, payload: dict) -> None:
    _check_path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output, indent=2, ensure_ascii=False, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        _publish(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_file(directory: Path, item: dict) -> dict:
    destination = _destination(directory, item["path"])
    if destination.exists():
        return {**item, "sha256": _hashes(destination, item)["sha256"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with _request(DOWNLOAD_BASE + quote(item["path"], safe="/")) as response:
                length = 0
                while chunk := response.read(1024 * 1024):
                    length += len(chunk)
                    if length > item["size"]:
                        raise PreparationError(f"Download excede o tamanho esperado: {item['path']}")
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        hashes = _hashes(temporary, item)
        _publish(temporary, destination)
        return {**item, "sha256": hashes["sha256"]}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(directory: Path | None = None, *, splits=SPLITS, workers: int = 4, plan_only: bool = False) -> dict:
    selected = tuple(split for split in SPLITS if split in splits)
    if not selected or set(splits) - set(SPLITS) or not 1 <= workers <= 6:
        raise PreparationError("Informe splits train/valid/test e entre 1 e 6 workers.")
    directory = Path(directory or ROOT / "data" / "datasets" / "ppe-absence").absolute()
    _check_path(directory)
    plan = load_plan()
    files = [item for item in plan if item["path"] in SOURCE_FILES or item["path"].split("/")[0] in selected]
    # Validate every existing selected file before downloading anything.
    for item in files:
        target = _destination(directory, item["path"])
        if target.exists():
            _hashes(target, item)
    directory.mkdir(parents=True, exist_ok=True)
    source = {
        "repository": REPOSITORY, "revision": REVISION,
        "download_base": DOWNLOAD_BASE,
        "original_source": "https://universe.roboflow.com/computer-vision/worker-safety",
        "benchmark_source": "https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1",
        "attribution": "Anonymous (computer-vision/worker-safety); Roboflow 100; mirror by LibreYOLO",
        "license_declared_by_publisher": "CC BY 4.0",
        "names": list(NAMES), "role": "Public research dataset, not an in-loco TCC validation set",
        "upstream_preprocessing": "Auto-orientation and resize stretch 640x640; no augmentation",
    }
    plan_report = {**source, "files": plan}
    _save_json(directory / "source-plan.json", plan_report)
    report = {"directory": str(directory), "splits": list(selected), "files": len(files),
              "bytes": sum(item["size"] for item in files), "revision": REVISION,
              "plan": str(directory / "source-plan.json"), "plan_only": plan_only}
    if plan_only:
        return report
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(lambda item: download_file(directory, item), files))
    manifest = {**source, "splits": list(selected), "files": verified}
    manifest_path = directory / f"provenance.{ '-'.join(selected) }.json"
    _save_json(manifest_path, manifest)
    report["provenance"] = str(manifest_path)
    report["provenance_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data" / "datasets" / "ppe-absence")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.directory, splits=args.splits, workers=args.workers, plan_only=args.plan_only), indent=2, ensure_ascii=False))
    except (PreparationError, OSError, ValueError) as error:
        parser.exit(1, f"Falha ao preparar dataset: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
