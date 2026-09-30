"""Read-only audit of local YOLO detection datasets, without model or network I/O.

Split paths are relative to YAML ``path`` (itself relative to the YAML).
Entries inside a TXT manifest are relative to that manifest. Use the report's
resolved_splits when preparing absolute manifests for a downstream trainer.
Empty label files explicitly mark background. Missing files fail unless the
operator opts into treating them as background with allow_background=True.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

from PIL import Image, UnidentifiedImageError
import yaml


# Matches the image suffixes accepted by the pinned Ultralytics detector.
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}
SPLITS = ("train", "val", "test")


class DatasetAuditError(RuntimeError):
    """The audit report itself could not be written safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently losing class/split definitions."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(None, None, "Chave YAML inválida.", key_node.start_mark) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"Chave YAML duplicada: {key!r}.", key_node.start_mark)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _issue(report: dict, code: str, message: str, *, warning=False, **context) -> None:
    report["warnings" if warning else "errors"].append({"code": code, "message": message, **context})


def _local_path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Informe um caminho local não vazio.")
    value = value.strip()
    if "://" in value or value.lower().startswith(("http:", "https:", "ftp:", "s3:", "gs:", "file:")):
        raise ValueError("URLs não são aceitas. Prepare os arquivos localmente; nenhum download será executado.")
    if "\x00" in value:
        raise ValueError("Caminho contém caractere nulo.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _names(document: dict, report: dict) -> dict[int, str]:
    names = document.get("names")
    if isinstance(names, list):
        mapping = dict(enumerate(names))
    elif isinstance(names, dict):
        mapping = {}
        for key, value in names.items():
            if type(key) is int:
                identifier = key
            elif isinstance(key, str) and key.isascii() and key.isdecimal():
                identifier = int(key)
            else:
                _issue(report, "invalid_classes", "IDs de classes em 'names' devem ser inteiros consecutivos a partir de 0.")
                return {}
            if identifier in mapping:
                _issue(report, "invalid_classes", "Há IDs de classes repetidos após normalizar 'names'.")
                return {}
            mapping[identifier] = value
    else:
        _issue(report, "invalid_classes", "Defina 'names' como lista ou mapeamento de IDs para nomes de classes.")
        return {}
    if not mapping or sorted(mapping) != list(range(len(mapping))):
        _issue(report, "invalid_classes", "Classes devem ser não vazias e usar IDs consecutivos 0, 1, 2, ... .")
        return {}
    if any(not isinstance(name, str) or not name.strip() for name in mapping.values()):
        _issue(report, "invalid_classes", "Cada classe deve ter um nome de texto não vazio.")
        return {}
    mapping = {key: mapping[key].strip() for key in sorted(mapping)}
    if len({name.casefold() for name in mapping.values()}) != len(mapping):
        _issue(report, "invalid_classes", "Os nomes das classes devem ser únicos, inclusive sem distinguir maiúsculas.")
        return {}
    if "nc" in document and (type(document["nc"]) is not int or document["nc"] != len(mapping)):
        _issue(report, "class_count_mismatch", "'nc' deve ser inteiro e igual à quantidade de classes em 'names'.")
    return mapping


def _images(entries: Any, root: Path, split: str, report: dict, protected: set[Path]) -> list[Path]:
    entries = entries if isinstance(entries, list) else [entries]
    found: list[Path] = []
    for entry in entries:
        try:
            path = _local_path(entry, root)
            protected.add(path)
            if path.is_dir():
                found.extend(sorted(item.resolve() for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES))
            elif path.is_file() and path.suffix.lower() == ".txt":
                for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    try:
                        image = _local_path(line, path.parent)
                        if image.suffix.lower() not in IMAGE_SUFFIXES:
                            raise ValueError("Cada linha do manifesto deve apontar para uma imagem suportada.")
                        found.append(image)
                    except (ValueError, OSError) as error:
                        _issue(report, "invalid_manifest_entry", str(error), split=split, path=str(path), line=number)
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                found.append(path)
            else:
                _issue(report, "invalid_split_path", "Caminho ausente ou não suportado: use pasta de imagens, imagem ou manifesto TXT.", split=split, path=str(path))
        except (ValueError, OSError, UnicodeError) as error:
            _issue(report, "invalid_split_path", str(error), split=split)
    unique = sorted(set(found))
    if len(unique) != len(found):
        _issue(report, "repeated_image_path", "A mesma imagem foi listada mais de uma vez neste split; foi contada uma única vez.", warning=True, split=split, repeated_entries=len(found) - len(unique))
    return unique


def _label_path(image: Path) -> Path:
    # Same last-component replacement used by Ultralytics img2label_paths.
    parts = list(image.parts)
    positions = [index for index, part in enumerate(parts[:-1]) if part == "images"]
    if positions:
        parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(path: Path, names: dict[int, str], split: str, report: dict) -> tuple[str, Counter, str | None]:
    counts: Counter = Counter()
    if not path.exists():
        return "missing", counts, None
    try:
        digest = _sha256(path)
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        _issue(report, "unreadable_label", f"Não foi possível ler o rótulo: {error}", split=split, path=str(path))
        return "invalid", counts, None
    invalid = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        message = ""
        if len(fields) != 5:
            message = "Detecção exige exatamente 5 valores: classe centro_x centro_y largura altura; segmentação não é suportada."
        else:
            try:
                identifier, x, y, width, height = map(float, fields)
                if not all(math.isfinite(value) for value in (identifier, x, y, width, height)):
                    message = "Todos os valores do rótulo devem ser finitos (sem NaN ou infinito)."
                elif not identifier.is_integer() or int(identifier) not in names:
                    message = "A classe deve ser um ID inteiro definido em 'names'."
                elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    message = "Centros devem estar em [0, 1] e largura/altura em (0, 1]."
                elif min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
                    message = "A caixa ultrapassa os limites da imagem; revise centro e dimensões normalizados."
                else:
                    counts[int(identifier)] += 1
            except ValueError:
                message = "Rótulo contém valores não numéricos."
        if message:
            invalid = True
            _issue(report, "invalid_label", message, split=split, path=str(path), line=line_number)
    return ("invalid" if invalid else "annotated" if counts else "empty"), counts, digest


def _write_report(report: dict, output: str | Path | None, protected: set[Path]) -> dict:
    report["valid"] = not report["errors"]
    report["summary"] = {
        "images": sum(item["images"] for item in report["splits"].values()),
        "valid_images": sum(item["valid_images"] for item in report["splits"].values()),
        "instances": sum(item["instances"] for item in report["splits"].values()),
        "background_images": sum(item["background_images"] for item in report["splits"].values()),
        "missing_labels": sum(item["missing_labels"] for item in report["splits"].values()),
        "errors": len(report["errors"]), "warnings": len(report["warnings"]),
    }
    if output is None:
        return report
    destination = Path(output).expanduser().resolve()
    if destination in protected or (destination.exists() and any(path.exists() and os.path.samefile(destination, path) for path in protected)):
        raise DatasetAuditError("O relatório não pode substituir o YAML, manifesto, imagem ou rótulo do dataset.")
    report["report_path"] = str(destination)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(destination)
    except OSError as error:
        raise DatasetAuditError(f"Não foi possível salvar o relatório de auditoria: {error}") from error
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return report


def audit_dataset(data: str | Path, *, require_test: bool = False, allow_background: bool = False,
                  output: str | Path | None = None) -> dict[str, Any]:
    """Audit all configured splits and return findings; invalid input yields valid=False.

    Train/val are required. require_test additionally requires nonempty test.
    Missing labels are errors unless explicitly allowed; empty TXT labels always
    count as declared background. Exact SHA256 overlap across splits is an error.
    The audit does not infer class semantics or estimate mAP, nor detect near
    duplicates/same-camera scenes. Invalid datasets still produce a JSON report.
    """
    source = Path(data).expanduser().resolve()
    protected = {source}
    report: dict[str, Any] = {
        "schema_version": 1, "kind": "dataset_audit", "dataset_yaml": str(source),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "require_test": require_test, "allow_background": allow_background,
        "class_names": {}, "splits": {}, "resolved_splits": {},
        "errors": [], "warnings": [], "leakage": [], "duplicates_within_split": [],
        "scope": "Auditoria estrutural local; não mede mAP nem garante qualidade semântica das anotações. SHA256 detecta cópias idênticas, não cenas semelhantes ou frames próximos.",
    }
    try:
        document = yaml.load(source.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
        report["dataset_yaml_sha256"] = _sha256(source)
        if not isinstance(document, dict):
            raise ValueError("O YAML deve conter um mapeamento com train, val e names.")
        if "download" in document:
            raise ValueError("Remova 'download' do YAML e prepare o dataset localmente. Nenhum download foi executado.")
        names = _names(document, report)
        root = _local_path(document.get("path") or str(source.parent), source.parent)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        _issue(report, "invalid_yaml", f"Não foi possível usar o dataset YAML: {error}")
        return _write_report(report, output, protected)
    report["dataset_root"] = str(root)
    report["class_names"] = {str(key): name for key, name in names.items()}
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for split in SPLITS:
        entries = document.get(split)
        if not entries:
            if split in ("train", "val") or require_test:
                _issue(report, "missing_split", f"Defina o split '{split}' com imagens locais distintas.", split=split)
            continue
        images = _images(entries, root, split, report, protected)
        report["resolved_splits"][split] = [str(path) for path in images]
        stats = {"images": len(images), "valid_images": 0, "corrupt_images": 0, "missing_labels": 0,
                 "empty_labels": 0, "background_images": 0, "annotated_images": 0, "invalid_labels": 0,
                 "instances": 0, "class_counts": {str(key): {"name": name, "images": 0, "instances": 0} for key, name in names.items()},
                 "files": []}
        report["splits"][split] = stats
        if not images:
            _issue(report, "empty_split", f"Split '{split}' não contém imagens suportadas.", split=split)
        for image in images:
            label = _label_path(image)
            protected.update((image, label))
            record: dict[str, Any] = {"image": str(image), "label": str(label), "sha256": None, "dimensions": None}
            stats["files"].append(record)
            try:
                record["sha256"] = _sha256(image)
                by_hash[record["sha256"]].append({"split": split, "image": str(image)})
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(image) as opened:
                        opened.verify()
                    with Image.open(image) as opened:
                        opened.load()
                        if min(opened.size) < 10:
                            raise ValueError("A imagem deve ter pelo menos 10 pixels em cada dimensão.")
                        record["dimensions"] = list(opened.size)
                stats["valid_images"] += 1
            except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                stats["corrupt_images"] += 1
                _issue(report, "invalid_image", f"Imagem ausente, corrompida ou não decodificável: {error}", split=split, path=str(image))
            status, counts, label_hash = _labels(label, names, split, report)
            record.update(label_status=status, label_sha256=label_hash, instances=sum(counts.values()))
            if status == "missing":
                stats["missing_labels"] += 1
                if allow_background:
                    stats["background_images"] += 1
                    _issue(report, "missing_label_as_background", "Rótulo ausente tratado como background por opção explícita; confira se a imagem realmente não tem objetos.", warning=True, split=split, path=str(label))
                else:
                    _issue(report, "missing_label", "Rótulo ausente. Anote a imagem; para background declarado crie TXT vazio, ou use allow_background explicitamente.", split=split, path=str(label))
            elif status == "empty":
                stats["empty_labels"] += 1
                stats["background_images"] += 1
            elif status == "invalid":
                stats["invalid_labels"] += 1
            else:
                stats["annotated_images"] += 1
            stats["instances"] += sum(counts.values())
            for identifier, count in counts.items():
                stats["class_counts"][str(identifier)]["instances"] += count
                stats["class_counts"][str(identifier)]["images"] += 1
        for identifier, values in stats["class_counts"].items():
            if not values["instances"]:
                _issue(report, "class_absent", f"Classe {identifier} ({values['name']}) não possui instâncias válidas em '{split}'.", warning=True, split=split, class_id=int(identifier))
    for digest, occurrences in sorted(by_hash.items()):
        if len(occurrences) < 2:
            continue
        group = {"sha256": digest, "files": occurrences}
        if len({item["split"] for item in occurrences}) > 1:
            report["leakage"].append(group)
            _issue(report, "split_leakage", "Imagens com SHA256 idêntico aparecem em splits diferentes. Separe treino/validação/teste antes de medir desempenho.", sha256=digest, files=occurrences)
        else:
            report["duplicates_within_split"].append(group)
            _issue(report, "duplicate_image", "Há cópias idênticas dentro do mesmo split; revise a ponderação das amostras.", warning=True, sha256=digest, files=occurrences)
    return _write_report(report, output, protected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita dataset YOLO de detecção local antes do treinamento, sem downloads.")
    parser.add_argument("--data", required=True, help="YAML local com train, val e names")
    parser.add_argument("--require-test", action="store_true", help="Exige também split test para avaliação final")
    parser.add_argument("--allow-background", action="store_true", help="Permite rótulos ausentes como background; TXT vazio já é background explícito")
    parser.add_argument("--output", default="runs/dataset-audit.json", help="Relatório JSON, inclusive quando a auditoria falha")
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.data, require_test=args.require_test, allow_background=args.allow_background, output=args.output)
    except (DatasetAuditError, OSError, ValueError) as error:
        print(f"Falha de auditoria: {error}")
        return 2
    summary = report["summary"]
    print(f"Dataset {'válido' if report['valid'] else 'inválido'}: {summary['images']} imagens, {summary['instances']} instâncias, "
          f"{summary['errors']} erros, {summary['warnings']} avisos.")
    for issue in report["errors"][:20]:
        print(f"- [{issue['code']}] {issue['message']} {issue.get('path', '')}")
    print(f"Relatório: {report['report_path']}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
