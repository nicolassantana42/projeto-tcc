"""Publish a locally trained PPE checkpoint without replacing existing models.

Run only after reviewing validation results. Promotion verifies provenance and
class compatibility; it does not establish accuracy or approve operational use.
The previous best.pt and its exported INT8 artifacts are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from prepare_ppe import PreparationError, _check_local_path, _publish_new, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load_names(weights: Path) -> dict[int, str]:
    from epi_monitor.ml import load_yolo
    return dict(load_yolo(str(weights)).names)


def _check_existing(path: Path, expected: bytes | str) -> None:
    _check_local_path(path)
    if path.exists():
        matches = (path.is_file() and (path.read_bytes() == expected if isinstance(expected, bytes)
                                      else sha256_file(path) == expected))
        if not matches:
            raise PreparationError(f"Arquivo existente divergente; preservado: {path}")


def promote(run_dir: Path | None = None, *, root: Path = ROOT) -> dict:
    from epi_monitor.config import DEFAULT_PPE_MODEL
    from epi_monitor.detection import canonical_label

    root = Path(root).absolute()
    directory = Path(run_dir or root / "runs/train/ppe_absence").absolute()
    checkpoint, training_path = directory / "weights/best.pt", directory / "training.json"
    for path in (checkpoint, training_path):
        _check_local_path(path)
    try:
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training["kind"] != "training":
            raise ValueError("o relatório não é de treinamento")
        expected_sha = training["provenance"]["best_weights"]["sha256"]
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("SHA256 de best_weights ausente ou inválido")
        if sha256_file(checkpoint) != expected_sha:
            raise ValueError("SHA256 de best.pt difere do treinamento registrado")
        recorded_names = {entry["class_id"]: entry["name"] for entry in training["per_class"]}
        if not recorded_names or len(recorded_names) != len(training["per_class"]):
            raise ValueError("classes do relatório ausentes ou duplicadas")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PreparationError(f"Treinamento inválido; nenhum modelo promovido: {error}") from error

    target = root / DEFAULT_PPE_MODEL
    provenance_path = target.with_suffix(".provenance.json")
    _check_existing(target, expected_sha)
    names = load_names(checkpoint)
    if names != recorded_names:
        raise PreparationError("Nomes/IDs do checkpoint diferem do relatório de treinamento.")
    canonical = [canonical_label(name) for name in names.values()]
    if any(canonical.count(label) != 1 for label in ("helmet", "vest", "no_helmet", "no_vest")):
        raise PreparationError("Pesos precisam ter classes únicas de capacete/colete e suas ausências explícitas.")

    report = {
        "description": "Modelo EPI treinado localmente; promoção explícita após revisão da validação.",
        "weights_sha256": expected_sha,
        "training_run": str(directory.relative_to(root)) if directory.is_relative_to(root) else str(directory),
        "training_report_sha256": sha256_file(training_path),
        "classes": names,
        "config": training.get("config", {}),
        "training_provenance": training["provenance"],
        "limitations": ["Promoção verifica integridade e classes, não valida precisão.",
                        "Validar no ambiente real e revisar alertas de ausência.",
                        "best.pt e seus exports INT8 anteriores permanecem artefatos históricos distintos."],
    }
    payload = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _check_existing(provenance_path, payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Both destinations are preflighted before publishing either file. A crash
    # between publications is recoverable by rerunning the identical command.
    for destination, source in ((target, checkpoint), (provenance_path, payload)):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".promote-", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                if isinstance(source, bytes):
                    stream.write(source)
                else:
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, stream)
            if destination == target and sha256_file(temporary) != expected_sha:
                raise PreparationError("O checkpoint mudou durante a cópia; promoção interrompida.")
            _publish_new(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"model": str(target), "sha256": expected_sha, "provenance": str(provenance_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/train/ppe_absence")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(promote(args.run_dir), ensure_ascii=False, indent=2))
    except (PreparationError, OSError, ValueError) as error:
        print(f"Promoção interrompida: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
