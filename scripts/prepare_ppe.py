"""Download pinned public PPE artifacts using only the Python standard library.

The public checkpoint is a baseline, not the final model trained by this TCC.
Run ``python scripts/prepare_ppe.py --dataset`` to also prepare local training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPOSITORY = "baskarmother/yolov8-ppe-construction"
BASELINE_REVISION = "3213ed51de90cbc76e577e6944e84f7c74343526"
BASELINE_URL = f"https://huggingface.co/{BASELINE_REPOSITORY}/resolve/{BASELINE_REVISION}/best.pt"
BASELINE_SHA256 = "8714b4b2bbde95b3a07dcdbe873995e34742b5ce628464a4da232721d4691ffe"
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATASET_SHA256 = "bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc"
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_ARCHIVE_FILES = 100000
BASELINE_WARNING = (
    "ATENÇÃO: baseline público, não treinado por este TCC e insuficiente para uso operacional. "
    "No teste externo observado, o recall de colete foi de 1/178 (0,56%); "
    "esse resultado entre datasets não substitui a avaliação de um modelo treinado para o projeto."
)


class PreparationError(RuntimeError):
    """A local artifact could not be prepared without changing user files."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    # Junctions/reparse points on Windows also redirect writes outside a tree.
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _check_local_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        if _is_link(candidate):
            raise PreparationError(f"Caminho contém link simbólico ou junction: {candidate}")


def _publish_new(temporary: Path, destination: Path) -> None:
    """Publish a complete same-filesystem file atomically, without replacement."""
    _check_local_path(destination)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or sha256_file(destination) != sha256_file(temporary):
            raise PreparationError(f"Arquivo existente diferente; preservado: {destination}")
    except OSError as error:
        raise PreparationError(f"Não foi possível publicar o arquivo atomicamente: {destination}: {error}") from error


def ensure_download(url: str, destination: Path, expected_sha256: str) -> Path:
    """Reuse only verified cache; a mismatch never overwrites an existing file."""
    destination = Path(destination).absolute()
    _check_local_path(destination)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise PreparationError(f"SHA256 inesperado no cache; arquivo preservado: {destination}. Mova-o antes de tentar novamente.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=90) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise PreparationError(f"SHA256 do download não corresponde à versão fixada: {destination.name}.")
        _publish_new(temporary, destination)
        return destination
    except PreparationError:
        raise
    except (OSError, ValueError) as error:
        raise PreparationError(f"Falha ao baixar {destination.name}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _zip_destination(info: zipfile.ZipInfo, directory: Path) -> Path:
    raw = info.filename
    parts = PurePosixPath(raw).parts
    if (not parts or raw.startswith("/") or "\\" in raw or "\x00" in raw
            or any(part in {".", ".."} or ":" in part or part.endswith((".", " ")) for part in raw.rstrip("/").split("/"))):
        raise PreparationError(f"Caminho inseguro no ZIP: {raw!r}")
    if parts[0] not in {"images", "labels", "data.yaml", "LICENSE"}:
        raise PreparationError(f"Entrada não esperada no dataset: {raw!r}")
    if parts[0] in {"data.yaml", "LICENSE"} and (len(parts) != 1 or info.is_dir()):
        raise PreparationError(f"Entrada não esperada no dataset: {raw!r}")
    devices = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.split(".")[0].upper() in devices for part in parts):
        raise PreparationError(f"Nome de dispositivo não permitido no ZIP: {raw!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise PreparationError(f"Link ou tipo especial não permitido no ZIP: {raw!r}")
    if bool(info.flag_bits & 1):
        raise PreparationError("ZIP criptografado não é aceito.")
    target = directory.joinpath(*parts)
    _check_local_path(target)
    try:
        target.resolve().relative_to(directory.resolve())
    except ValueError as error:
        raise PreparationError(f"Entrada escapa da pasta de destino: {raw!r}") from error
    return target


def extract_dataset(archive: Path, directory: Path) -> dict[str, int]:
    """Preflight all paths and cache contents before creating any dataset file."""
    archive, directory = Path(archive), Path(directory).absolute()
    _check_local_path(directory)
    missing, reused = [], 0
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > MAX_ARCHIVE_FILES or sum(info.file_size for info in entries) > MAX_ARCHIVE_BYTES:
                raise PreparationError("ZIP excede os limites de quantidade/tamanho do dataset esperado.")
            seen = set()
            file_paths = set()
            targets = []
            for info in entries:
                target = _zip_destination(info, directory)
                key = str(target).casefold()
                if key in seen:
                    raise PreparationError(f"Caminho duplicado no ZIP: {info.filename!r}")
                seen.add(key)
                targets.append(target)
                if not info.is_dir():
                    file_paths.add(key)
                for parent in target.parents:
                    if parent == directory:
                        break
                    if parent.exists() and not parent.is_dir():
                        raise PreparationError(f"Arquivo bloqueia diretório do dataset; preservado: {parent}")
                if info.is_dir():
                    if target.exists() and not target.is_dir():
                        raise PreparationError(f"Arquivo bloqueia diretório do dataset; preservado: {target}")
                    continue
                if target.exists():
                    if not target.is_file() or target.stat().st_size != info.file_size:
                        raise PreparationError(f"Arquivo do dataset já existe com conteúdo diferente; preservado: {target}")
                    with source.open(info) as handle:
                        expected = _sha256_stream(handle)
                    if sha256_file(target) != expected:
                        raise PreparationError(f"Arquivo do dataset modificado; preservado: {target}")
                    reused += 1
                else:
                    missing.append((info, target))
            if any(str(parent).casefold() in file_paths for target in targets for parent in target.parents):
                raise PreparationError("ZIP define um mesmo caminho como arquivo e como diretório.")
            for info, target in missing:
                _check_local_path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".extract-", suffix=".part", delete=False) as output:
                        temporary = Path(output.name)
                        with source.open(info) as handle:
                            shutil.copyfileobj(handle, output, length=1024 * 1024)
                    _publish_new(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
    except PreparationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise PreparationError(f"Não foi possível extrair o dataset: {error}") from error
    return {"extracted": len(missing), "reused": reused}


def _save_provenance(destination: Path, report: dict) -> None:
    _check_local_path(destination)
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PreparationError(f"Proveniência existente inválida; preservada: {destination}") from error
        if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in report.items()):
            raise PreparationError(f"Proveniência existente divergente; preservada: {destination}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, ensure_ascii=False)
            output.write("\n")
        _publish_new(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(root: Path = REPOSITORY_ROOT, *, dataset: bool = False) -> dict:
    root = Path(root).absolute()
    model = ensure_download(BASELINE_URL, root / "models" / "ppe" / "baseline-public.pt", BASELINE_SHA256)
    _save_provenance(model.with_suffix(".provenance.json"), {
        "repository": BASELINE_REPOSITORY, "revision": BASELINE_REVISION, "url": BASELINE_URL,
        "license_declared_by_author": "MIT", "sha256": BASELINE_SHA256,
        "role": "public pretrained baseline, not trained by this TCC",
        "model_card": f"https://huggingface.co/{BASELINE_REPOSITORY}",
    })
    report = {"baseline": str(model), "baseline_sha256": BASELINE_SHA256, "warning": BASELINE_WARNING}
    if dataset:
        directory = root / "data" / "datasets"
        archive = ensure_download(DATASET_URL, directory / "construction-ppe.zip", DATASET_SHA256)
        report["dataset"] = {"archive": str(archive), "sha256": DATASET_SHA256,
                             "directory": str(directory), **extract_dataset(archive, directory)}
        _save_provenance(directory / "construction-ppe.provenance.json", {
            "url": DATASET_URL, "sha256": DATASET_SHA256,
            "documentation": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
            "license_file": "LICENSE", "role": "public external dataset; not an in-loco TCC test set",
        })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara baseline público de EPI e dataset com versões e hashes fixados.")
    parser.add_argument("--dataset", action="store_true", help="Também baixa/verifica e extrai Construction-PPE em data/datasets")
    args = parser.parse_args(argv)
    print(BASELINE_WARNING)
    try:
        report = prepare(dataset=args.dataset)
    except (PreparationError, OSError) as error:
        print(f"Falha na preparação: {error}")
        return 1
    print(f"Baseline verificado: {report['baseline']}")
    if args.dataset:
        data = report["dataset"]
        print(f"Dataset verificado: {data['directory']} ({data['extracted']} arquivos extraídos; {data['reused']} reutilizados).")
        print("Audite antes de treinar: python -m safeguard audit-data --data data/construction-ppe.yaml --require-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
