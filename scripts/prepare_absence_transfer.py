"""Remap RF100 labels to a verified ten-class checkpoint without changing weights.

Only the five annotated PPE/person classes are evaluated; the other five output
classes retain their IDs but are not annotated by this dataset. Images, split
membership, box coordinates and source files are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ("helmet", "no-helmet", "no-vest", "person", "vest")
CLASS_MAPPING = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7}
SPLIT_MAPPING = {"train": "train", "val": "valid", "test": "test"}


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(path: Path) -> Path:
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PreparationError(f"Link ou junction não permitido: {candidate}")
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise PreparationError(f"Caminho deve permanecer dentro do projeto: {path}")
    return path


def load_names(weights: Path) -> dict[int, str]:
    # Lazy loading: tests can verify all data transformations without importing
    # Ultralytics or deserializing any checkpoint.
    from epi_monitor.ml import load_yolo
    model = load_yolo(str(weights))
    return dict(model.names)


def audit_source(source: Path) -> dict:
    from epi_monitor.dataset_audit import audit_dataset
    return audit_dataset(source, require_test=True)


def verify_mapping(source_names: dict, target_names: dict) -> dict[int, int]:
    from epi_monitor.detection import canonical_label
    source = {int(key): value for key, value in source_names.items()}
    target = {int(key): value for key, value in target_names.items()}
    if source != dict(enumerate(SOURCE_NAMES)):
        raise PreparationError("Os cinco IDs/nomes de origem devem corresponder ao RF100 fixado.")
    if set(target) != set(range(10)) or any(not isinstance(name, str) or not name.strip() for name in target.values()):
        raise PreparationError("Os pesos devem conter exatamente dez IDs consecutivos com nomes válidos.")
    for old, new in CLASS_MAPPING.items():
        expected = canonical_label(source[old])
        matches = [identifier for identifier, name in target.items() if canonical_label(name) == expected]
        if matches != [new]:
            raise PreparationError(f"Classe {source[old]} ausente, ambígua ou em ID inesperado nos pesos: {matches}.")
    return dict(CLASS_MAPPING)


def remap_labels(payload: bytes, mapping: dict[int, int]) -> bytes:
    try:
        lines = payload.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise PreparationError("Rótulo não é UTF-8.") from error
    output = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            values = [float(value) for value in parts]
        except ValueError as error:
            raise PreparationError(f"Valores inválidos na linha {number} do rótulo.") from error
        if (len(values) != 5 or not all(math.isfinite(value) for value in values) or
                values[0] != int(values[0]) or int(values[0]) not in mapping):
            raise PreparationError(f"ID ou caixa inválida na linha {number} do rótulo.")
        _, x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise PreparationError(f"Coordenadas inválidas na linha {number} do rótulo.")
        output.append(" ".join([str(mapping[int(values[0])]), *parts[1:]]))
    return (("\n".join(output) + "\n") if output else "").encode("utf-8")


def _verify_existing(path: Path, expected_sha256: str) -> bool:
    checked_path(path)
    if not path.exists():
        return False
    if not path.is_file() or sha256(path) != expected_sha256:
        raise PreparationError(f"Arquivo existente diferente; preservado: {path}")
    return True


def _publish(path: Path, *, payload: bytes | None = None, source: Path | None = None, expected: str) -> None:
    if _verify_existing(path, expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            if source is not None:
                with checked_path(source).open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, 1024 * 1024)
            else:
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != expected:
            raise PreparationError(f"Conteúdo mudou durante a preparação: {source or path}")
        checked_path(path)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify_existing(path, expected)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def prepare(*, source: Path | None = None, weights: Path | None = None,
            directory: Path | None = None, output_yaml: Path | None = None) -> dict:
    source = checked_path(source or ROOT / "data" / "ppe-absence.yaml")
    weights = checked_path(weights or ROOT / "models" / "ppe" / "absence-base.pt")
    directory = checked_path(directory or ROOT / "data" / "datasets" / "ppe-absence-transfer")
    output_yaml = checked_path(output_yaml or ROOT / "data" / "ppe-absence-transfer.yaml")
    if not directory.is_relative_to((ROOT / "data" / "datasets").resolve()):
        raise PreparationError("O derivado deve permanecer em data/datasets dentro do projeto.")
    if not output_yaml.is_relative_to((ROOT / "data").resolve()) or output_yaml.suffix.lower() not in {".yaml", ".yml"}:
        raise PreparationError("O YAML derivado deve ficar em data/ dentro do projeto.")
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise PreparationError("Informe pesos .pt locais existentes para preservar a cabeça de classificação.")
    audit = audit_source(source)
    if not audit.get("valid"):
        details = "; ".join(error.get("message", "erro") for error in audit.get("errors", [])[:5])
        raise PreparationError(f"A auditoria do dataset de origem falhou: {details}")
    source_root = checked_path(Path(audit["dataset_root"]))
    if directory.is_relative_to(source_root) or source_root.is_relative_to(directory):
        raise PreparationError("A origem e o derivado precisam usar diretórios independentes.")
    if output_yaml in {source, weights} or output_yaml.is_relative_to(source_root) or output_yaml.is_relative_to(directory):
        raise PreparationError("O YAML derivado não pode substituir arquivos ou ficar dentro dos datasets.")
    weights_hash = sha256(weights)
    target_names = load_names(weights)
    if sha256(weights) != weights_hash:
        raise PreparationError("Os pesos foram alterados durante a leitura dos nomes.")
    mapping = verify_mapping(audit["class_names"], target_names)
    target_names = {int(key): value for key, value in target_names.items()}
    writes, records, seen = [], [], set()
    for split, folder in SPLIT_MAPPING.items():
        files = audit["splits"].get(split, {}).get("files", [])
        if not files:
            raise PreparationError(f"O split {split} está vazio.")
        for item in files:
            image, label = checked_path(Path(item["image"])), checked_path(Path(item["label"]))
            if not image.is_relative_to(source_root) or not label.is_relative_to(source_root):
                raise PreparationError("Manifesto aponta para arquivos fora da origem declarada.")
            image_hash, label_hash = sha256(image), sha256(label)
            if image_hash != item["sha256"] or label_hash != item["label_sha256"]:
                raise PreparationError("Imagem ou rótulo mudou após a auditoria.")
            image_target = directory / folder / "images" / image.name
            label_target = directory / folder / "labels" / f"{image.stem}.txt"
            for target in (image_target, label_target):
                key = str(target).casefold()
                if key in seen:
                    raise PreparationError(f"Colisão de nomes no derivado: {target}")
                seen.add(key)
            mapped = remap_labels(label.read_bytes(), mapping)
            mapped_hash = hashlib.sha256(mapped).hexdigest()
            writes.extend([(image_target, None, image, image_hash), (label_target, mapped, None, mapped_hash)])
            records.append({"split": split, "source_image": image.relative_to(source_root).as_posix(),
                            "source_label": label.relative_to(source_root).as_posix(),
                            "image_sha256": image_hash, "source_label_sha256": label_hash,
                            "derived_image": image_target.relative_to(directory).as_posix(),
                            "derived_label": label_target.relative_to(directory).as_posix(),
                            "derived_label_sha256": mapped_hash, "instances": item["instances"]})
    provenance = {
        "schema_version": 1, "kind": "class_id_transfer_dataset", "source_yaml": str(source),
        "source_yaml_sha256": sha256(source), "source_root": str(source_root),
        "source_repository": "LibreYOLO/construction-safety-gsnvb",
        "source_revision": "342e545489a6b5f76d6c8225f1ef2629c5a4770a",
        "license_declared_by_publisher": "CC BY 4.0",
        "attribution": "Anonymous (computer-vision/worker-safety); Roboflow 100; mirror by LibreYOLO",
        "weights": str(weights), "weights_sha256": weights_hash,
        "source_names": list(SOURCE_NAMES), "target_names": target_names, "class_id_mapping": mapping,
        "annotated_target_class_ids": sorted(mapping.values()),
        "unannotated_target_class_ids": sorted(set(target_names) - set(mapping.values())),
        "note": "Five annotated classes mapped to ten existing output IDs. Other outputs are not evaluated as if annotated. No weight modification. No generated negative labels. Original split membership and box coordinates preserved. Public data, not in-loco validation.",
        "splits": {split: len(audit["splits"][split]["files"]) for split in SPLIT_MAPPING},
        "files": records,
    }
    provenance_payload = _json_bytes(provenance)
    provenance_path = directory / "transfer.provenance.json"
    writes.append((provenance_path, provenance_payload, None, hashlib.sha256(provenance_payload).hexdigest()))
    # JSON is a strict subset of YAML: original class strings remain exact.
    yaml_payload = _json_bytes({"path": Path(os.path.relpath(directory, output_yaml.parent)).as_posix(),
                               "train": "train/images", "val": "valid/images", "test": "test/images",
                               "names": [target_names[index] for index in range(10)]})
    writes.append((output_yaml, yaml_payload, None, hashlib.sha256(yaml_payload).hexdigest()))
    # Preflight everything, including provenance/config, before creating outputs.
    for target, _, _, digest in writes:
        _verify_existing(target, digest)
    for target, payload, origin, digest in writes:
        _publish(target, payload=payload, source=origin, expected=digest)
    return {"dataset_yaml": str(output_yaml), "directory": str(directory), "images": len(records),
            "splits": provenance["splits"], "class_id_mapping": mapping, "weights_sha256": weights_hash,
            "provenance": str(provenance_path), "provenance_sha256": hashlib.sha256(provenance_payload).hexdigest(),
            "warning": "Somente cinco classes estão anotadas; as outras cinco saídas não possuem validação neste dataset."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output-yaml", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(source=args.source, weights=args.weights, directory=args.directory,
                                 output_yaml=args.output_yaml), indent=2, ensure_ascii=False))
    except (PreparationError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Falha ao preparar transferência: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
