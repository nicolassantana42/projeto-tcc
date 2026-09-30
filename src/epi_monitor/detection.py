"""Person-gated PPE inference and conservative, explicit per-person decisions.

The second detector sees the complete frame once, retaining the image scale and
context used in full-frame training. Missing detections are *not* evidence that
equipment is absent. Only an unambiguous explicit negative class can produce an
``unsafe`` observation, which still requires human review.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Literal, Mapping, Protocol
import unicodedata

import numpy as np

from .preprocessing import preprocess_frame
from .types import Detection, FrameResult


def normalize_label(label: str) -> str:
    text = unicodedata.normalize("NFKD", str(label).strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.replace("_", " ").replace("-", " ").split())


PERSON_ALIASES = frozenset({"person", "people", "pessoa", "pessoas", "worker", "trabalhador"})


@dataclass(frozen=True)
class EquipmentSpec:
    """Label vocabulary and relative vertical association region for one item.

    To add another PPE category, supply a new spec and include its key in
    ``required_ppe``. Region coordinates are fractions of the person's height;
    they are geometric association hints, not anatomical landmark estimates.
    """

    positive_labels: frozenset[str]
    negative_labels: frozenset[str]
    region_y: tuple[float, float]
    display_name: str


DEFAULT_EQUIPMENT = {
    "helmet": EquipmentSpec(
        frozenset({"helmet", "hardhat", "hard hat", "safety helmet", "capacete", "capacete de seguranca"}),
        frozenset({"no helmet", "no hardhat", "no hard hat", "without helmet", "without hardhat", "sem capacete", "sem capacete de seguranca"}),
        (-.12, .40), "capacete",
    ),
    "vest": EquipmentSpec(
        frozenset({"vest", "safety vest", "reflective vest", "colete", "colete refletivo", "colete de seguranca"}),
        frozenset({"no vest", "no safety vest", "no reflective vest", "without vest", "without safety vest", "sem colete", "sem colete refletivo"}),
        (.15, .80), "colete",
    ),
}


@dataclass(frozen=True)
class PersonPPEAssessment:
    """A frame-local observation; ``index`` never denotes a tracked identity."""

    person: Detection
    index: int
    status: Literal["ok", "unsafe", "uncertain"]
    present: tuple[str, ...]
    absent: tuple[str, ...]
    uncertain: tuple[str, ...]
    reasons: tuple[str, ...]


class Detector(Protocol):
    names: dict[int, str]

    def predict(self, frame: np.ndarray, confidence: float | None = None,
                iou: float | None = None) -> list[Detection]: ...


def _vocabulary(specs: Mapping[str, EquipmentSpec]) -> dict[str, tuple[str, bool]]:
    vocabulary = {}
    for kind, spec in specs.items():
        if not isinstance(kind, str) or not kind or kind == "person" or kind.startswith("no_"):
            raise ValueError("Identificador de EPI inválido; use nomes como helmet ou vest.")
        if not spec.positive_labels or not spec.display_name:
            raise ValueError("Cada EPI precisa de classes positivas e nome de apresentação.")
        lower, upper = spec.region_y
        if not all(math.isfinite(value) for value in (lower, upper)) or not -.5 <= lower < upper <= 1.5:
            raise ValueError("Região relativa do EPI inválida.")
        for labels, positive in ((spec.positive_labels, True), (spec.negative_labels, False)):
            for raw in labels:
                label = normalize_label(raw)
                if not label or label in PERSON_ALIASES:
                    raise ValueError("Classe de EPI inválida ou confundida com pessoa.")
                if label in vocabulary and vocabulary[label] != (kind, positive):
                    raise ValueError(f"Classe de EPI ambígua no vocabulário: {label}.")
                vocabulary[label] = (kind, positive)
        # Canonical output labels are accepted by the same vocabulary.
        for label, positive in ((normalize_label(kind), True), (normalize_label(f"no_{kind}"), False)):
            if label in vocabulary and vocabulary[label] != (kind, positive):
                raise ValueError(f"Classe canônica de EPI ambígua: {label}.")
            vocabulary[label] = (kind, positive)
    return vocabulary


def canonical_label(label: str, equipment_specs: Mapping[str, EquipmentSpec] | None = None) -> str | None:
    """Map a known training label to the cascade vocabulary; ignore unknowns."""
    normalized = normalize_label(label)
    if normalized in PERSON_ALIASES:
        return "person"
    match = _vocabulary(DEFAULT_EQUIPMENT if equipment_specs is None else equipment_specs).get(normalized)
    if match is None:
        return None
    return match[0] if match[1] else f"no_{match[0]}"


def validate_ppe_names(names: Mapping[int, str], required_ppe: tuple[str, ...] = ("helmet", "vest"),
                       equipment_specs: Mapping[str, EquipmentSpec] | None = None) -> None:
    """Reject COCO/incomplete PPE models before capture or inference starts."""
    specs = DEFAULT_EQUIPMENT if equipment_specs is None else equipment_specs
    if not required_ppe or len(set(required_ppe)) != len(required_ppe) or any(kind not in specs for kind in required_ppe):
        raise ValueError("Configure pelo menos um EPI obrigatório, sem duplicatas e com vocabulário conhecido.")
    vocabulary = _vocabulary(specs)
    represented = {match[0] for name in names.values()
                   if (match := vocabulary.get(normalize_label(name))) is not None and match[1]}
    missing = [specs[kind].display_name for kind in required_ppe if kind not in represented]
    if missing:
        raise ValueError("Modelo de EPI incompatível: faltam classes positivas de " + ", ".join(missing)
                         + ". Pesos COCO detectam pessoas, mas não substituem o modelo de EPI.")


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _intersection(first, second) -> float:
    return _area((max(first[0], second[0]), max(first[1], second[1]),
                  min(first[2], second[2]), min(first[3], second[3])))


def _candidate(gear: Detection, person: Detection, spec: EquipmentSpec) -> bool:
    x1, y1, x2, y2 = person.bbox
    gx1, gy1, gx2, gy2 = gear.bbox
    height = y2 - y1
    if _area(gear.bbox) <= 0 or x2 <= x1 or height <= 0:
        return False
    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    lower, upper = spec.region_y
    # A center-only test would attach an enormous erroneous box to a person.
    region = (x1, y1 + min(0., lower) * height, x2, y2)
    return (x1 <= cx <= x2 and lower <= (cy - y1) / height <= upper
            and _intersection(gear.bbox, region) / _area(gear.bbox) >= .5)


def _visibility_reason(person: Detection, others: list[Detection], spec: EquipmentSpec,
                       frame_shape: tuple[int, ...], minimum_person_height: float) -> str | None:
    x1, y1, x2, y2 = person.bbox
    height, width = frame_shape[:2]
    if y2 - y1 < minimum_person_height:
        return "pessoa pequena na imagem; resolução insuficiente para avaliar EPI"
    lower, upper = spec.region_y
    # A truncated person box makes the head/torso fractions unreliable. Do not
    # infer missing equipment from a body crossing the image boundary.
    if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return "pessoa cortada pela borda da imagem"
    region = (x1, y1 + max(0., lower) * (y2 - y1), x2, y1 + min(1., upper) * (y2 - y1))
    if _area(region) > 0 and any(_intersection(region, other.bbox) / _area(region) >= .35 for other in others):
        return "sobreposição de pessoas na região do EPI; possível oclusão"
    return None


def assess_person_ppe(people: list[Detection], equipment: list[Detection], frame_shape: tuple[int, ...],
                      *, required_ppe: tuple[str, ...] = ("helmet", "vest"),
                      equipment_specs: Mapping[str, EquipmentSpec] | None = None,
                      minimum_person_height: float = 80.) -> list[PersonPPEAssessment]:
    """Assign each item at most once and abstain on ambiguous/conflicting evidence.

    Positive or negative boxes eligible for multiple people are never assigned
    greedily. Multiple boxes competing for one person/category/polarity also
    cause abstention, so a box cannot establish compliance for two people.
    Bounding-box overlap is only an occlusion hint; it cannot measure visibility
    behind arbitrary objects. Unseen equipment always remains uncertain.
    """
    specs = DEFAULT_EQUIPMENT if equipment_specs is None else equipment_specs
    vocabulary = _vocabulary(specs)
    slots = {(index, kind, positive): [] for index in range(len(people))
             for kind in required_ppe for positive in (True, False)}
    ambiguous = set()
    for gear in equipment:
        match = vocabulary.get(normalize_label(gear.label))
        if match is None or match[0] not in required_ppe:
            continue
        kind, positive = match
        candidates = [index for index, person in enumerate(people) if _candidate(gear, person, specs[kind])]
        if len(candidates) == 1:
            slots[candidates[0], kind, positive].append(gear)
        elif len(candidates) > 1:
            ambiguous.update((index, kind) for index in candidates)

    assessments = []
    for index, person in enumerate(people):
        present, absent, uncertain, reasons = [], [], [], []
        for kind in required_ppe:
            spec = specs[kind]
            positives, negatives = slots[index, kind, True], slots[index, kind, False]
            reason = _visibility_reason(person, [other for idx, other in enumerate(people) if idx != index],
                                         spec, frame_shape, minimum_person_height)
            if reason is None and ((index, kind) in ambiguous or len(positives) > 1 or len(negatives) > 1):
                reason = "associação ambígua entre pessoa e EPI"
            if reason is None and positives and negatives:
                reason = "classes positivas e negativas conflitantes"
            if reason is not None:
                uncertain.append(kind)
                reasons.append(f"{spec.display_name}: {reason}.")
            elif positives:
                present.append(kind)
            elif negatives:
                absent.append(kind)
                reasons.append(f"{spec.display_name}: classe negativa explícita detectada; revisão visual necessária.")
            else:
                uncertain.append(kind)
                reasons.append(f"{spec.display_name}: sem evidência suficiente; não detectar o EPI não comprova ausência.")
        status = "unsafe" if absent else "uncertain" if uncertain else "ok"
        assessments.append(PersonPPEAssessment(person, index + 1, status, tuple(present), tuple(absent),
                                                tuple(uncertain), tuple(reasons)))
    return assessments


class CascadePipeline:
    """Run person detection first, then at most one full-frame PPE inference."""

    demo_mode = False

    def __init__(self, person_detector: Detector, ppe_detector: Detector,
                 required_ppe: tuple[str, ...] = ("helmet", "vest"), *,
                 equipment_specs: Mapping[str, EquipmentSpec] | None = None,
                 minimum_person_height: float = 80.):
        self.equipment_specs = dict(DEFAULT_EQUIPMENT if equipment_specs is None else equipment_specs)
        self.required_ppe = tuple(required_ppe)
        validate_ppe_names(ppe_detector.names, self.required_ppe, self.equipment_specs)
        if not any(normalize_label(label) in PERSON_ALIASES for label in person_detector.names.values()):
            raise ValueError("O primeiro modelo precisa de uma classe de pessoa.")
        if isinstance(minimum_person_height, bool) or not math.isfinite(minimum_person_height) or minimum_person_height <= 0:
            raise ValueError("A altura mínima da pessoa deve ser positiva e finita.")
        self.minimum_person_height = minimum_person_height
        self.person_detector = self.detector = person_detector
        self.ppe_detector = ppe_detector
        self.frame_index = 0
        self._vocabulary = _vocabulary(self.equipment_specs)
        labels = ["person", *self.equipment_specs, *(f"no_{kind}" for kind in self.equipment_specs)]
        self.names = dict(enumerate(labels))
        self._class_ids = {label: index for index, label in self.names.items()}

    def process(self, frame: np.ndarray, confidence: float | None = None,
                iou: float | None = None) -> FrameResult:
        started = perf_counter()
        prepared = preprocess_frame(frame)
        inference_started = perf_counter()
        first = self.person_detector.predict(prepared, confidence=confidence, iou=iou)
        person_ms = (perf_counter() - inference_started) * 1000
        people = sorted((Detection(0, "person", item.confidence, item.bbox) for item in first
                         if normalize_label(item.label) in PERSON_ALIASES), key=lambda item: item.bbox)
        equipment = []
        ppe_ms = 0.
        if people:
            inference_started = perf_counter()
            second = self.ppe_detector.predict(prepared, confidence=confidence, iou=iou)
            ppe_ms = (perf_counter() - inference_started) * 1000
            for item in second:
                match = self._vocabulary.get(normalize_label(item.label))
                if match is not None:
                    kind, positive = match
                    label = kind if positive else f"no_{kind}"
                    equipment.append(Detection(self._class_ids[label], label, item.confidence, item.bbox))
        assessments = assess_person_ppe(people, equipment, prepared.shape, required_ppe=self.required_ppe,
                                        equipment_specs=self.equipment_specs,
                                        minimum_person_height=self.minimum_person_height)
        alerts = [f"Pessoa {item.index}: não seguro — " + "; ".join(self.equipment_specs[kind].display_name for kind in item.absent)
                  + " com classe negativa explícita (revisão visual necessária)."
                  for item in assessments if item.status == "unsafe"]
        detections = people + equipment
        self.frame_index += 1
        return FrameResult(frame=frame, detections=detections,
                           counts=dict(Counter(item.label for item in detections)), alerts=alerts,
                           inference_ms=person_ms + ppe_ms, pipeline_ms=(perf_counter() - started) * 1000,
                           frame_index=self.frame_index, assessments=assessments,
                           stage_timings_ms={"person": person_ms, "ppe": ppe_ms}, ppe_executed=bool(people))
