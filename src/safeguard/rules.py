"""Heurística explicável de EPI; não constitui avaliação de conformidade."""

import unicodedata
from dataclasses import dataclass

from .types import Detection


def _normalize(label: str) -> str:
    text = unicodedata.normalize("NFKD", label.lower().strip())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.replace("_", " ").replace("-", " ").split())


ALIASES = {
    "person": {"person", "people", "pessoa", "pessoas", "worker", "trabalhador"},
    "helmet": {"helmet", "hardhat", "hard hat", "safety helmet", "capacete", "capacete de seguranca"},
    "vest": {"vest", "safety vest", "reflective vest", "colete", "colete refletivo", "colete de seguranca"},
}


def _kind(label: str) -> str | None:
    normalized = _normalize(label)
    return next((kind for kind, labels in ALIASES.items() if normalized in labels), None)


@dataclass(frozen=True)
class PersonAssessment:
    """One spatial observation; index is a frame index, never an identity."""

    person: Detection
    index: int
    missing: tuple[str, ...]


def person_detections(detections: list[Detection]) -> list[Detection]:
    return sorted((item for item in detections if _kind(item.label) == "person"), key=lambda item: item.bbox)


def supports_ppe(names: dict[int, str] | None) -> bool:
    return bool(names) and {"person", "helmet", "vest"}.issubset({_kind(label) for label in names.values()})


def _assign(gear: Detection, people: list[Detection], kind: str) -> int | None:
    gx1, gy1, gx2, gy2 = gear.bbox
    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    candidates = []
    for index, person in enumerate(people):
        x1, y1, x2, y2 = person.bbox
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        relative_y = (cy - y1) / height
        min_y, max_y, target_y = (-0.15, 0.45, 0.12) if kind == "helmet" else (0.2, 0.8, 0.45)
        if not x1 <= cx <= x2 or not min_y <= relative_y <= max_y:
            continue
        # One item belongs to at most one person, including overlapping people.
        distance = ((cx - (x1 + x2) / 2) / width) ** 2 + (relative_y - target_y) ** 2
        candidates.append((distance, index))
    return min(candidates)[1] if candidates else None


def assess_people(detections: list[Detection], names: dict[int, str] | None) -> list[PersonAssessment]:
    """Structured spatial associations, unavailable for incomplete label sets.

    Absence is a visual hypothesis, not a verified safety violation. Negative
    classes alone are insufficient; the model must detect all positive classes.
    """
    if not supports_ppe(names):
        return []
    people = person_detections(detections)
    present = [set() for _ in people]
    for detection in detections:
        kind = _kind(detection.label)
        if kind in {"helmet", "vest"}:
            person_index = _assign(detection, people, kind)
            if person_index is not None:
                present[person_index].add(kind)
    return [
        PersonAssessment(person, index, tuple(kind for kind in ("helmet", "vest") if kind not in present[index - 1]))
        for index, person in enumerate(people, start=1)
    ]


def assess_ppe(detections: list[Detection], names: dict[int, str], demo_mode: bool = True) -> list[str]:
    """Associate each helmet/vest to one person before suggesting absence.

    Person indices are spatially ordered within a frame, not tracking IDs.
    Negative classes such as `NO-Hardhat` are deliberately not positive PPE.
    """
    if demo_mode:
        return []
    if not supports_ppe(names):
        return ["Análise de EPI indisponível: o modelo precisa ter classes de pessoa, capacete e colete. Use o modo demonstração para modelos COCO."]
    alerts = []
    for assessment in assess_people(detections, names):
        missing = [label for kind, label in (("helmet", "capacete"), ("vest", "colete")) if kind in assessment.missing]
        if missing:
            alerts.append(f"Pessoa {assessment.index}: possível ausência de {' e '.join(missing)} (revisão visual necessária).")
    return alerts
