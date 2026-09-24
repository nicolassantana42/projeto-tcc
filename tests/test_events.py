from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from safeguard.events import CameraContext, EventPolicy, EventService, EventStore
from safeguard.detection import PersonPPEAssessment
from safeguard.rules import assess_people, supports_ppe
from safeguard.types import Detection, FrameResult


NAMES = {0: "person", 1: "helmet", 2: "vest"}


def person(x=0):
    return Detection(0, "person", 0.9, (x, 0, x + 80, 200))


def helmet(x=0):
    return Detection(1, "helmet", 0.9, (x + 20, 0, x + 60, 35))


def result(*detections):
    frame = np.zeros((210, 320, 3), dtype=np.uint8)
    counts = {d.label: sum(item.label == d.label for item in detections) for d in detections}
    return FrameResult(frame, list(detections), counts, [], 3, 4, 1)


def service(tmp_path, **policy):
    return EventService(EventStore(tmp_path), CameraContext(location="Galpão A"), EventPolicy(**policy), False, NAMES)


def feed(service, now, *detections):
    observed = result(*detections)
    return service.process(observed, observed.frame, now)


def feed_ppe(service, now, *observations, status="unsafe"):
    """Simulate the cascade's explicit per-person decision, not missing boxes."""
    people = [item[0] for item in observations]
    evidence, assessments = [], []
    for index, (detected_person, absent) in enumerate(observations, 1):
        for kind in absent:
            evidence.append(Detection(3 if kind == "helmet" else 4, f"no_{kind}", .9, detected_person.bbox))
        present = tuple(kind for kind in ("helmet", "vest") if kind not in absent)
        assessments.append(PersonPPEAssessment(detected_person, index, status, present,
                                                tuple(absent) if status == "unsafe" else (),
                                                tuple(absent) if status == "uncertain" else (),
                                                ("Evidência explícita simulada para teste de persistência.",)))
    observed = result(*people, *evidence)
    observed.ppe_executed = True
    observed.assessments = assessments
    return service.process(observed, observed.frame, now)


@pytest.mark.parametrize("kwargs", [
    {"trigger": "any"}, {"confirmation_seconds": -1}, {"confirmation_seconds": float("nan")},
    {"cooldown_seconds": float("inf")}, {"cooldown_seconds": True}, {"max_events": 0}, {"max_events": 1.5},
])
def test_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EventPolicy(**kwargs)


def test_structured_rules_preserve_spatial_association():
    assessments = assess_people([person(), person(200), helmet()], NAMES)
    assert [item.missing for item in assessments] == [("vest",), ("helmet", "vest")]
    assert [item.index for item in assessments] == [1, 2]
    assert not supports_ppe({0: "person", 1: "NO-Hardhat", 2: "vest"})
    assert assess_people([person()], {0: "person"}) == []


def test_person_temporal_confirmation_and_camera_cooldown(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=2, cooldown_seconds=5)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1, person(5)) is None
    event = feed(detector, 2, person(10))
    assert event["kind"] == "person"
    assert "detectada" in event["reasons"][0]
    assert "2s de análise" in event["reasons"][0]
    assert feed(detector, 3, person(10), person(200)) is None
    assert feed(detector, 4, person(10), person(200)) is None
    assert feed(detector, 5, person(10), person(200)) is None
    assert feed(detector, 6, person(10), person(200)) is None
    assert feed(detector, 7, person(10), person(200)) is not None
    assert len(detector.store.list_events()) == 2


def test_reset_confirmation_discards_partial_observations(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=2)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1, person()) is None
    detector.reset_confirmation()
    assert feed(detector, 2, person()) is None
    assert feed(detector, 3, person()) is None
    assert feed(detector, 4, person()) is not None


def test_reset_confirmation_preserves_existing_event_cooldown(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=1, cooldown_seconds=5)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1, person()) is not None
    detector.reset_confirmation()
    for now in (2, 3, 4, 5):
        assert feed(detector, now, person()) is None
    assert feed(detector, 6, person()) is not None
    assert len(detector.store.list_events()) == 2


def test_spatially_distinct_people_do_not_share_confirmation(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=2)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1, person(200)) is None
    assert feed(detector, 2, person()) is None
    assert feed(detector, 3, person(200)) is None
    assert detector.store.list_events() == []


def test_ppe_conditions_on_different_people_never_accumulate(tmp_path):
    detector = service(tmp_path, confirmation_seconds=2)
    # Both people remain tracked, but explicit negative observations alternate.
    first = ((person(), ("vest",)), (person(200), ("helmet",)))
    second = ((person(), ("helmet",)), (person(200), ("vest",)))
    assert feed_ppe(detector, 0, *first) is None
    assert feed_ppe(detector, 1, *second) is None
    assert feed_ppe(detector, 2, *first) is None
    assert feed_ppe(detector, 3, *second) is None
    event = feed_ppe(detector, 4, *second)
    assert event is None
    assert feed_ppe(detector, 5, *second)["kind"] == "ppe"


def test_ambiguous_overlap_resets_tracking(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=2)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1, person(), person()) is None
    assert feed(detector, 2, person(), person()) is None
    assert detector.store.list_events() == []


def test_missing_frame_and_long_observation_gap_reset_confirmation(tmp_path):
    detector = service(tmp_path, trigger="person", confirmation_seconds=2)
    assert feed(detector, 0, person()) is None
    assert feed(detector, 1) is None
    assert feed(detector, 2, person()) is None
    assert feed(detector, 3, person()) is None
    assert feed(detector, 6, person()) is None
    assert feed(detector, 7, person()) is None
    assert feed(detector, 8, person()) is not None


@pytest.mark.parametrize("demo,names", [(True, NAMES), (False, {0: "person"}), (False, None)])
def test_demo_or_legacy_output_without_cascade_never_generates_violation(tmp_path, demo, names):
    detector = EventService(EventStore(tmp_path), CameraContext(), EventPolicy(confirmation_seconds=0), demo, names)
    assert feed(detector, 0, person()) is None
    assert detector.store.list_events() == []


def test_absent_positive_without_explicit_negative_never_creates_event(tmp_path):
    detector = service(tmp_path, confirmation_seconds=0)
    for now in range(5):
        # The legacy single-stage heuristic might suggest missing vest; this is
        # not a cascade decision and must never turn into a saved PPE violation.
        assert feed(detector, now, person(), helmet()) is None
    assert detector.store.list_events() == []


def test_uncertain_and_ok_cascade_assessments_do_not_create_events(tmp_path):
    detector = service(tmp_path, confirmation_seconds=0)
    assert feed_ppe(detector, 0, (person(), ("helmet",)), status="uncertain") is None
    assert feed_ppe(detector, 1, (person(), ()), status="ok") is None
    assert detector.store.list_events() == []


def test_explicit_negative_requires_two_seconds_and_uncertainty_resets_it(tmp_path):
    detector = service(tmp_path, confirmation_seconds=2)
    observation = (person(), ("helmet",))
    assert feed_ppe(detector, 0, observation) is None
    assert feed_ppe(detector, 1, observation, status="uncertain") is None
    assert feed_ppe(detector, 2, observation) is None
    assert feed_ppe(detector, 3, observation) is None
    event = feed_ppe(detector, 4, observation)
    assert event["kind"] == "ppe"
    assert "classe explícita de ausência de capacete" in event["reasons"][0]
    assert "2s observados" in event["reasons"][0]
    assert any(item["label"] == "no_helmet" for item in event["detections"])
    assert Path(event["snapshot_path"]).is_file()


def test_demo_mode_blocks_even_explicit_negative_assessments(tmp_path):
    detector = EventService(EventStore(tmp_path), CameraContext(), EventPolicy(confirmation_seconds=0), True, NAMES)
    assert feed_ppe(detector, 0, (person(), ("helmet",))) is None
    assert detector.store.list_events() == []


def test_explicit_person_trigger_in_demo_is_labeled_as_presence(tmp_path):
    detector = EventService(EventStore(tmp_path), CameraContext(), EventPolicy(trigger="person", confirmation_seconds=0), True, {0: "person"})
    event = feed(detector, 0, person())
    assert event["model_mode"] == "demo"
    assert event["kind"] == "person"
    assert "ausência" not in " ".join(event["reasons"])


def test_clock_must_be_finite_and_monotonic(tmp_path):
    detector = service(tmp_path)
    assert feed(detector, 5, person()) is None
    for invalid in (4, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            feed(detector, invalid, person())


def test_evidence_persists_metadata_banner_and_delivery_without_mutation(tmp_path):
    store = EventStore(tmp_path)
    observed = result(person())
    original = observed.frame.copy()
    event = store.save(observed, observed.frame, CameraContext(name="Entrada", location="Galpão A"), "person", ["Pessoa detectada"], True)
    assert Path(event["snapshot_path"]).is_absolute()
    assert Path(event["metadata_path"]).is_file()
    saved = cv2.imread(event["snapshot_path"])
    assert saved.shape[0] > observed.frame.shape[0]
    assert saved[observed.frame.shape[0]:].any()
    np.testing.assert_array_equal(original, observed.frame)
    store.set_delivery(event["id"], "telegram", "sent", "Entregue")
    restored = EventStore(tmp_path).list_events()[0]
    assert restored["location"] == "Galpão A"
    assert restored["camera_name"] == "Entrada"
    assert restored["detections"][0]["bbox"] == [0, 0, 80, 200]
    assert restored["deliveries"]["telegram"]["status"] == "sent"
    assert "rtsp" not in json.dumps(restored)


def test_manual_evidence_does_not_require_detections(tmp_path):
    store = EventStore(tmp_path)
    observed = result()
    event = store.save(observed, observed.frame, CameraContext(), "manual", ["Captura manual"], True)
    assert event["kind"] == "manual"
    assert event["detections"] == []
    assert event["deliveries"] == {}
    assert Path(event["snapshot_path"]).is_file()


def test_retention_deletes_only_known_complete_occurrences(tmp_path):
    store = EventStore(tmp_path, max_events=2)
    foreign = tmp_path / str(uuid4())
    foreign.mkdir()
    (foreign / "keep.txt").write_text("preserve")
    (tmp_path / "notes.txt").write_text("preserve")
    observed = result(person())
    events = [store.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True) for _ in range(4)]
    assert len(store.list_events()) == 2
    assert {item["id"] for item in store.list_events()} == {item["id"] for item in events[-2:]}
    assert not Path(events[0]["snapshot_path"]).exists()
    assert (foreign / "keep.txt").read_text() == "preserve"
    assert (tmp_path / "notes.txt").read_text() == "preserve"
    event_folder = Path(events[-1]["metadata_path"]).parent
    (event_folder / "extra.txt").write_text("foreign")
    assert store.delete(events[-1]["id"]) is False
    assert (event_folder / "extra.txt").exists()


@pytest.mark.parametrize("identifier", ["../outside", "../../event.json", "C:\\Windows", "a/b", "", None])
def test_event_ids_cannot_traverse_paths(tmp_path, identifier):
    store = EventStore(tmp_path)
    with pytest.raises(ValueError):
        store.set_delivery(identifier, "telegram", "sent")
    with pytest.raises(ValueError):
        store.delete(identifier)


def test_untrusted_metadata_paths_are_reconstructed(tmp_path):
    store = EventStore(tmp_path)
    observed = result(person())
    event = store.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True)
    path = Path(event["metadata_path"])
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["snapshot_path"] = "../../outside.jpg"
    tampered["metadata_path"] = "../../outside.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    restored = store.list_events()[0]
    assert restored["snapshot_path"] == event["snapshot_path"]
    assert restored["metadata_path"] == str(path)


def test_delivery_updates_across_store_instances_are_not_lost(tmp_path):
    first, second = EventStore(tmp_path), EventStore(tmp_path)
    observed = result(person())
    event = first.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first.set_delivery, event["id"], "telegram", "sent"),
                   pool.submit(second.set_delivery, event["id"], "outlook", "disabled")]
        for future in futures:
            future.result()
    assert set(first.list_events()[0]["deliveries"]) == {"telegram", "outlook"}
    assert not list(tmp_path.glob("**/*.tmp"))


def test_retention_failure_does_not_invalidate_save_or_reset_cooldown(tmp_path, monkeypatch):
    detector = service(tmp_path, trigger="person", confirmation_seconds=0, cooldown_seconds=60)
    assert detector.store.retention_warning == ""
    original_prune = detector.store.prune

    def fail_prune():
        raise PermissionError("sensitive filesystem detail")

    monkeypatch.setattr(detector.store, "prune", fail_prune)
    saved = feed(detector, 0, person())
    assert Path(saved["snapshot_path"]).is_file()
    assert detector.store.retention_warning
    assert "sensitive" not in detector.store.retention_warning
    assert feed(detector, 1, person()) is None
    assert len(detector.store.list_events()) == 1

    detector.store.set_delivery(saved["id"], "telegram", "accepted", "Aceito pelo provedor")
    assert detector.store.list_events()[0]["deliveries"]["telegram"]["status"] == "accepted"
    assert detector.store.retention_warning
    monkeypatch.setattr(detector.store, "prune", original_prune)
    detector.store.set_delivery(saved["id"], "telegram", "accepted", "Aceito pelo provedor")
    assert detector.store.retention_warning == ""


def test_pending_images_survive_retention_until_every_channel_finishes(tmp_path):
    store = EventStore(tmp_path, max_events=2)
    observed = result(person())
    pending = store.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True)
    store.set_delivery(pending["id"], "telegram", "pending")
    store.set_delivery(pending["id"], "email", "pending")
    latest = [store.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True) for _ in range(4)]
    expected = {pending["id"], latest[-1]["id"], latest[-2]["id"]}
    assert {event["id"] for event in store.list_events()} == expected

    # Pending work remains protected after a restart, without silently retrying.
    reopened = EventStore(tmp_path, max_events=2)
    assert reopened.prune() == 0
    assert Path(pending["snapshot_path"]).is_file()
    reopened.set_delivery(pending["id"], "telegram", "accepted")
    assert Path(pending["snapshot_path"]).is_file()
    reopened.set_delivery(pending["id"], "email", "failed")
    assert not Path(pending["snapshot_path"]).exists()
    assert {event["id"] for event in reopened.list_events()} == expected - {pending["id"]}


@pytest.mark.parametrize("changes", [
    {"reasons": None}, {"reasons": "single reason"}, {"reasons": [1]},
    {"camera_id": []}, {"camera_name": None}, {"location": ""},
    {"kind": []}, {"model_mode": None}, {"counts": None},
    {"counts": {"person": "one"}}, {"frame_index": -1},
    {"deliveries": []}, {"deliveries": {"telegram": None}},
    {"deliveries": {"telegram": {"status": [], "detail": ""}}},
    {"deliveries": {"telegram": {"status": "pending", "detail": None}}},
    {"deliveries": {"telegram": {"status": "pending", "detail": "", "updated_at_utc": "invalid"}}},
    {"detections": None}, {"detections": [{"class_id": 0, "label": "person", "confidence": 0.9, "bbox": [1]}]},
    {"detections": [{"class_id": 0, "label": "person", "confidence": 0.9, "bbox": [0, 0, 10 ** 310, 100]}]},
])
def test_malformed_occurrence_fields_are_ignored_and_never_pruned(tmp_path, changes):
    store = EventStore(tmp_path, max_events=1)
    observed = result(person())
    saved = store.save(observed, observed.frame, CameraContext(), "person", ["Pessoa"], True)
    path = Path(saved["metadata_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(changes)
    path.write_text(json.dumps(document), encoding="utf-8")
    assert store.list_events() == []
    assert store.prune() == 0
    assert path.is_file()
    assert Path(saved["snapshot_path"]).is_file()
