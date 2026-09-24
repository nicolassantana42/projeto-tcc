"""Cascade contracts with synthetic boxes; no weights, cameras or network."""

import numpy as np
import pytest

from safeguard.detection import (
    CascadePipeline, DEFAULT_EQUIPMENT, EquipmentSpec, canonical_label,
    validate_ppe_names,
)
from safeguard.types import Detection


FRAME = np.zeros((300, 500, 3), dtype=np.uint8)
PERSON_BOX = (40, 20, 160, 280)
PPE_NAMES = {0: "Hardhat", 1: "Safety Vest", 2: "NO-Hardhat", 3: "NO-Safety Vest"}


def box(label, coordinates, class_id=0, confidence=.9):
    return Detection(class_id, label, confidence, coordinates)


def person(coordinates=PERSON_BOX):
    return box("person", coordinates)


def equipment(label, coordinates=PERSON_BOX):
    x1, y1, x2, y2 = coordinates
    width, height = x2 - x1, y2 - y1
    is_head = canonical_label(label) in {"helmet", "no_helmet"} or "goggle" in label
    low, high = (.02, .18) if is_head else (.30, .65)
    return box(label, (x1 + .25 * width, y1 + low * height, x1 + .75 * width, y1 + high * height))


class FakeDetector:
    device = "cpu"

    def __init__(self, names, detections):
        self.names = names
        self.detections = detections
        self.calls = []

    def predict(self, frame, confidence=None, iou=None):
        self.calls.append((frame, confidence, iou))
        return list(self.detections)


def cascade(people, gear, **kwargs):
    primary = FakeDetector({0: "person", 5: "bus"}, people)
    secondary = FakeDetector(PPE_NAMES, gear)
    return CascadePipeline(primary, secondary, **kwargs)


def test_no_person_skips_ppe_inference_completely():
    pipeline = cascade([box("bus", (10, 10, 480, 280), 5)], [equipment("NO-Hardhat")])
    result = pipeline.process(FRAME)
    assert len(pipeline.person_detector.calls) == 1
    assert pipeline.ppe_detector.calls == []
    assert not result.ppe_executed
    assert result.detections == []
    assert result.counts == {}
    assert result.assessments == []
    assert result.alerts == []
    assert result.stage_timings_ms["ppe"] == 0
    assert result.inference_ms == result.stage_timings_ms["person"]


def test_multiple_people_trigger_one_full_frame_ppe_pass_with_same_thresholds():
    second_box = (300, 20, 420, 280)
    pipeline = cascade([person(second_box), person()], [
        equipment("Hardhat"), equipment("Safety Vest"),
        equipment("Hardhat", second_box), equipment("Safety Vest", second_box),
        box("person", PERSON_BOX, 4),  # The second model's people are not duplicated.
    ])
    result = pipeline.process(FRAME, confidence=.61, iou=.33)
    assert len(pipeline.person_detector.calls) == len(pipeline.ppe_detector.calls) == 1
    first_frame, first_confidence, first_iou = pipeline.person_detector.calls[0]
    second_frame, second_confidence, second_iou = pipeline.ppe_detector.calls[0]
    assert first_frame is second_frame is FRAME
    assert (first_confidence, first_iou) == (second_confidence, second_iou) == (.61, .33)
    assert result.frame is FRAME
    assert result.ppe_executed
    assert [item.status for item in result.assessments] == ["ok", "ok"]
    assert result.assessments[0].person.bbox == PERSON_BOX
    assert [item.index for item in result.assessments] == [1, 2]
    assert result.counts == {"person": 2, "helmet": 2, "vest": 2}
    assert result.alerts == []
    assert result.inference_ms == pytest.approx(sum(result.stage_timings_ms.values()))
    assert result.pipeline_ms >= result.inference_ms
    assert result.frame_index == 1
    assert pipeline.process(FRAME).frame_index == 2


def test_canonical_class_ids_never_collide_between_models():
    pipeline = cascade([person()], [equipment("NO-Hardhat"), equipment("Safety Vest")])
    result = pipeline.process(FRAME)
    assert pipeline.names == {0: "person", 1: "helmet", 2: "vest", 3: "no_helmet", 4: "no_vest"}
    assert all(pipeline.names[item.class_id] == item.label for item in result.detections)
    assert {item.class_id for item in result.detections} == {0, 2, 3}
    assert pipeline.detector is pipeline.person_detector
    assert not pipeline.demo_mode


def test_missing_positive_is_uncertain_never_an_unsafe_alert():
    result = cascade([person()], [equipment("Hardhat")]).process(FRAME)
    assessment = result.assessments[0]
    assert assessment.status == "uncertain"
    assert assessment.present == ("helmet",)
    assert assessment.absent == ()
    assert assessment.uncertain == ("vest",)
    assert "não comprova ausência" in assessment.reasons[0]
    assert result.alerts == []


def test_explicit_associated_negative_is_required_for_unsafe():
    result = cascade([person()], [equipment("NO-Hardhat"), equipment("Safety Vest")]).process(FRAME)
    assessment = result.assessments[0]
    assert assessment.status == "unsafe"
    assert assessment.present == ("vest",)
    assert assessment.absent == ("helmet",)
    assert assessment.uncertain == ()
    assert len(result.alerts) == 1
    assert "classe negativa explícita" in result.alerts[0]


def test_one_explicit_negative_remains_unsafe_when_other_item_has_no_evidence():
    assessment = cascade([person()], [equipment("NO-Hardhat")]).process(FRAME).assessments[0]
    assert assessment.status == "unsafe"
    assert assessment.absent == ("helmet",)
    assert assessment.uncertain == ("vest",)


def test_positive_and_negative_conflict_abstains():
    result = cascade([person()], [equipment("Hardhat"), equipment("NO-Hardhat"), equipment("Safety Vest")]).process(FRAME)
    assessment = result.assessments[0]
    assert assessment.status == "uncertain"
    assert assessment.absent == ()
    assert assessment.uncertain == ("helmet",)
    assert any("conflitantes" in reason for reason in assessment.reasons)
    assert result.alerts == []


def test_a_single_helmet_cannot_cover_two_people():
    second_box = (110, 20, 230, 280)
    shared_helmet = box("Hardhat", (110, 30, 160, 65))
    result = cascade([person(), person(second_box)], [
        shared_helmet, equipment("Safety Vest"), equipment("Safety Vest", second_box),
    ]).process(FRAME)
    assert len(result.assessments) == 2
    assert all(item.status == "uncertain" for item in result.assessments)
    assert all("helmet" not in item.present for item in result.assessments)
    assert result.alerts == []


def test_ambiguous_negative_does_not_accuse_either_person():
    second_box = (110, 20, 230, 280)
    shared_negative = box("NO-Hardhat", (110, 30, 160, 65))
    result = cascade([person(), person(second_box)], [shared_negative]).process(FRAME)
    assert all(item.status == "uncertain" and not item.absent for item in result.assessments)
    assert result.alerts == []


def test_clear_helmet_is_associated_only_with_its_person():
    second_box = (300, 20, 420, 280)
    result = cascade([person(), person(second_box)], [
        equipment("Hardhat"), equipment("Safety Vest"), equipment("Safety Vest", second_box),
    ]).process(FRAME)
    assert result.assessments[0].status == "ok"
    assert result.assessments[1].status == "uncertain"
    assert result.assessments[1].present == ("vest",)


@pytest.mark.parametrize("label", ["Hardhat", "NO-Hardhat"])
def test_multiple_boxes_competing_for_one_slot_abstain(label):
    result = cascade([person()], [equipment(label), equipment(label), equipment("Safety Vest")]).process(FRAME)
    assert result.assessments[0].status == "uncertain"
    assert "helmet" in result.assessments[0].uncertain
    assert result.alerts == []


@pytest.mark.parametrize("coordinates", [(0, 20, 160, 280), (40, 0, 160, 280), (340, 20, 500, 280), (40, 20, 160, 300)])
def test_clipped_person_abstains_even_with_negative_class(coordinates):
    result = cascade([person(coordinates)], [
        equipment("NO-Hardhat", coordinates), equipment("Safety Vest", coordinates),
    ]).process(FRAME)
    assert result.ppe_executed  # A clipped person is still a detected person.
    assert result.assessments[0].status == "uncertain"
    assert all("borda" in reason for reason in result.assessments[0].reasons)
    assert result.alerts == []


def test_small_person_is_retained_but_assessment_is_uncertain():
    small_box = (40, 20, 80, 70)
    pipeline = cascade([person(small_box)], [equipment("NO-Hardhat", small_box), equipment("Safety Vest", small_box)])
    result = pipeline.process(FRAME)
    assert result.ppe_executed
    assert result.counts["person"] == 1
    assert result.assessments[0].status == "uncertain"
    assert any("resolução insuficiente" in reason for reason in result.assessments[0].reasons)
    assert result.alerts == []


def test_minimum_person_height_is_configurable():
    small_box = (40, 20, 80, 70)
    result = cascade([person(small_box)], [equipment("Hardhat", small_box), equipment("Safety Vest", small_box)],
                     minimum_person_height=40).process(FRAME)
    assert result.assessments[0].status == "ok"


def test_overlap_occlusion_hint_abstains_on_negatives():
    second_box = (80, 40, 200, 280)
    result = cascade([person(), person(second_box)], [equipment("NO-Hardhat"), equipment("NO-Safety Vest")]).process(FRAME)
    assert result.assessments[0].status == "uncertain"
    assert any("oclusão" in reason for reason in result.assessments[0].reasons)
    assert result.alerts == []


def test_unrelated_or_implausibly_large_gear_is_not_assigned():
    result = cascade([person()], [
        box("NO-Hardhat", (350, 30, 400, 65)),
        box("NO-Safety Vest", (0, 0, 2000, 400)),
        box("Hardhat", (80, 220, 120, 250)),  # At feet, not head.
    ]).process(FRAME)
    assert result.assessments[0].status == "uncertain"
    assert not result.assessments[0].present
    assert not result.assessments[0].absent
    assert result.alerts == []


@pytest.mark.parametrize("raw,expected", [
    ("Pessoa", "person"), ("worker", "person"), ("capacete de segurança", "helmet"),
    ("Hardhat", "helmet"), ("Safety Vest", "vest"), ("colete_refletivo", "vest"),
    ("NO-Hardhat", "no_helmet"), ("NO-Safety Vest", "no_vest"),
    ("sem_capacete", "no_helmet"), ("sem colete", "no_vest"),
    ("none", None), ("gloves", None), ("no_person", None),
])
def test_known_aliases_are_explicit(raw, expected):
    assert canonical_label(raw) == expected


@pytest.mark.parametrize("names", [
    {0: "person", 1: "bus", 2: "car"}, {},
    {0: "NO-Hardhat", 1: "NO-Safety Vest"}, {0: "Hardhat"},
])
def test_incompatible_ppe_model_fails_before_either_inference(names):
    first = FakeDetector({0: "person"}, [person()])
    second = FakeDetector(names, [])
    with pytest.raises(ValueError, match="Modelo de EPI incompatível"):
        CascadePipeline(first, second)
    assert first.calls == second.calls == []


def test_ppe_model_does_not_need_its_own_person_class():
    validate_ppe_names({0: "Hardhat", 1: "Safety Vest"})
    with pytest.raises(ValueError, match="primeiro modelo"):
        CascadePipeline(FakeDetector({0: "bus"}, []), FakeDetector(PPE_NAMES, []))


@pytest.mark.parametrize("required", [(), ("helmet", "helmet"), ("unknown",)])
def test_invalid_required_equipment_configuration(required):
    with pytest.raises(ValueError, match="EPI obrigatório"):
        cascade([], [], required_ppe=required)


@pytest.mark.parametrize("height", [0, -1, True, float("nan"), float("inf")])
def test_invalid_person_height_configuration(height):
    with pytest.raises(ValueError, match="altura mínima"):
        cascade([], [], minimum_person_height=height)


def test_extension_with_custom_required_equipment():
    specs = {**DEFAULT_EQUIPMENT, "goggles": EquipmentSpec(frozenset({"safety goggles"}),
             frozenset({"no goggles"}), (0, .4), "óculos")}
    first = FakeDetector({0: "person"}, [person()])
    second = FakeDetector({0: "safety goggles"}, [equipment("safety goggles")])
    pipeline = CascadePipeline(first, second, required_ppe=("goggles",), equipment_specs=specs)
    result = pipeline.process(FRAME)
    assert result.assessments[0].status == "ok"
    assert result.assessments[0].present == ("goggles",)
    assert canonical_label("safety goggles", specs) == "goggles"
    assert all(pipeline.names[item.class_id] == item.label for item in result.detections)


def test_invalid_image_fails_before_model_inference():
    pipeline = cascade([person()], [])
    with pytest.raises(ValueError, match="BGR"):
        pipeline.process(np.zeros((10, 10), dtype=np.uint8))
    assert not pipeline.person_detector.calls
    assert not pipeline.ppe_detector.calls
