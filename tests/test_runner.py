"""CLI detection flow: real cascade/event rules with local fixtures and fake models."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from safeguard.capture import CaptureError, ImageSource
from safeguard.detection import CascadePipeline
from safeguard.events import EventStore
from safeguard.runner import run_detection
from safeguard.types import Detection
import safeguard.runner as runner


FRAME = np.zeros((300, 500, 3), dtype=np.uint8)
PERSON = Detection(0, "person", .95, (40, 20, 160, 280))
GEAR = [Detection(2, "NO-Hardhat", .90, (70, 25, 125, 70)),
        Detection(1, "Safety Vest", .90, (70, 100, 125, 190))]


class FakeDetector:
    device = "cpu"

    def __init__(self, names, outputs):
        self.names, self.outputs, self.calls = names, iter(outputs), 0

    def predict(self, frame, confidence=None, iou=None):
        self.calls += 1
        return next(self.outputs)


class FakeVideo:
    is_stream = False

    def __init__(self, timestamps, *, is_file=True, error=None):
        self.timestamps, self.is_file = iter(timestamps), is_file
        self.timestamp_seconds = None
        self.opened = self.closed = False
        self.error = error

    def __enter__(self):
        self.opened = True
        return self

    def read(self):
        if self.error is not None:
            raise self.error
        try:
            self.timestamp_seconds = next(self.timestamps)
        except StopIteration:
            return None
        return FRAME.copy()

    def __exit__(self, *_):
        self.closed = True


def install_pipeline(monkeypatch, people):
    first = FakeDetector({0: "person"}, people)
    second = FakeDetector({0: "Hardhat", 1: "Safety Vest", 2: "NO-Hardhat"}, [GEAR] * len(people))
    pipeline = CascadePipeline(first, second)
    monkeypatch.setattr(runner, "create_cascade", lambda *args: pipeline)
    return pipeline


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_image(path):
    success, encoded = cv2.imencode(".png", FRAME)
    assert success
    encoded.tofile(str(path))


def test_two_stage_flow_skips_ppe_without_people_and_records_real_outputs(monkeypatch, tmp_path):
    pipeline = install_pipeline(monkeypatch, [[], [PERSON]])
    capture = FakeVideo([0., 1.])
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    output, snapshot = tmp_path / "frames.jsonl", tmp_path / "última.png"
    report = run_detection(source="clip.mp4", output=output, snapshot=snapshot)
    assert capture.closed
    assert pipeline.person_detector.calls == 2 and pipeline.ppe_detector.calls == 1
    assert report["frames"] == 2
    assert report["ppe_executed_frames"] == report["ppe_skipped_frames"] == 1
    assert report["observations_by_status"] == {"unsafe": 1}
    rows = records(output)
    assert [row["source_time_seconds"] for row in rows] == [0., 1.]
    assert rows[0]["detections"] == [] and rows[0]["ppe_executed"] is False
    assert rows[1]["assessments"][0]["absent"] == ["helmet"]
    assert cv2.imdecode(np.fromfile(snapshot, np.uint8), cv2.IMREAD_COLOR).shape == FRAME.shape


def test_video_confirmation_uses_media_time_even_when_inference_is_fast(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [[PERSON]] * 3)
    capture = FakeVideo([0., 1., 2.])
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    report = run_detection(source="clip.mp4", output=tmp_path / "frames.jsonl", save_events=True,
                           event_directory=tmp_path / "events", confirmation_seconds=2.,
                           camera_name="Portaria", location="Entrada A")
    assert len(report["events_saved"]) == 1
    event = EventStore(tmp_path / "events").list_events()[0]
    assert event["frame_index"] == 3
    assert event["camera_name"] == "Portaria" and event["location"] == "Entrada A"
    assert "2s" in event["reasons"][0]
    assert Path(event["snapshot_path"]).is_file()


def test_file_without_media_clock_does_not_confirm_using_cpu_processing_time(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [[PERSON]] * 4)
    capture = FakeVideo([0., None, 2., 3.])
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    report = run_detection(source="clip.mp4", output=tmp_path / "frames.jsonl", save_events=True,
                           event_directory=tmp_path / "events", confirmation_seconds=2.)
    assert report["frames"] == 4 and not report["events_saved"]
    assert "timestamp/FPS" in report["event_timing_warning"]
    assert EventStore(tmp_path / "events").list_events() == []


def test_live_source_uses_event_services_monotonic_clock(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [[PERSON]])
    capture = FakeVideo([None], is_file=False)
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    observed = []

    class RecordingService:
        def __init__(self, *_):
            pass

        def process(self, result, annotated, now=None):
            observed.append(now)

    monkeypatch.setattr(runner, "EventService", RecordingService)
    report = run_detection(source=0, output=tmp_path / "frames.jsonl", save_events=True,
                           event_directory=tmp_path / "events")
    assert observed == [None]
    assert report["event_timing_warning"] == "" and capture.closed


def test_static_image_is_a_single_observation_and_show_waits_for_user(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [[PERSON]])
    source = tmp_path / "pessoa.png"
    write_image(source)
    capture = ImageSource(str(source))
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    waits, closed_windows = [], []
    monkeypatch.setattr(runner.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(runner.cv2, "waitKey", lambda delay: waits.append(delay) or ord("q"))
    monkeypatch.setattr(runner.cv2, "destroyAllWindows", lambda: closed_windows.append(True))
    report = run_detection(source=str(source), output=tmp_path / "frames.jsonl", save_events=True,
                           event_directory=tmp_path / "events", confirmation_seconds=120., show=True)
    assert report["frames"] == 1 and len(report["events_saved"]) == 1
    assert waits == [0] and closed_windows == [True]
    assert not capture._opened
    event = EventStore(tmp_path / "events").list_events()[0]
    assert "observação única" in event["reasons"][0]
    assert "120s" not in event["reasons"][0]


def test_video_preview_does_not_block_and_quit_releases_capture(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [[PERSON]] * 2)
    capture = FakeVideo([0., 1.])
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    waits = []
    monkeypatch.setattr(runner.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(runner.cv2, "waitKey", lambda delay: waits.append(delay) or ord("q"))
    monkeypatch.setattr(runner.cv2, "destroyAllWindows", lambda: None)
    report = run_detection(source="clip.mp4", output=tmp_path / "frames.jsonl", show=True)
    assert report["frames"] == 1 and waits == [1] and capture.closed


@pytest.mark.parametrize("stage", ["read", "infer", "write", "show"])
def test_failures_release_capture_and_preview(monkeypatch, tmp_path, stage):
    pipeline = install_pipeline(monkeypatch, [[PERSON]])
    capture = FakeVideo([0.], error=CaptureError("read failed") if stage == "read" else None)
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    closed_windows = []
    monkeypatch.setattr(runner.cv2, "destroyAllWindows", lambda: closed_windows.append(True))

    def fail(*_, **__):
        raise RuntimeError("intentional failure")

    if stage == "infer":
        monkeypatch.setattr(pipeline, "process", fail)
    elif stage == "write":
        monkeypatch.setattr(runner, "frame_record", fail)
    elif stage == "show":
        monkeypatch.setattr(runner.cv2, "imshow", fail)
    with pytest.raises((RuntimeError, CaptureError)):
        run_detection(source="clip.mp4", output=tmp_path / "frames.jsonl", show=True)
    assert capture.closed and closed_windows == [True]


def test_empty_source_is_an_error_and_releases_capture(monkeypatch, tmp_path):
    install_pipeline(monkeypatch, [])
    capture = FakeVideo([])
    monkeypatch.setattr(runner, "open_source", lambda _: capture)
    with pytest.raises(ValueError, match="não produziu"):
        run_detection(source="clip.mp4", output=tmp_path / "frames.jsonl")
    assert capture.closed


@pytest.mark.parametrize("target", ["source", "person_model", "ppe_model", "snapshot_source", "same_outputs", "model_directory"])
def test_output_collisions_fail_before_loading_models_or_capture(monkeypatch, tmp_path, target):
    source, person_model, ppe_model = tmp_path / "input.png", tmp_path / "person.pt", tmp_path / "ppe.pt"
    for path in (source, person_model, ppe_model):
        path.write_bytes(b"original input")
    options = {"source": str(source), "person_model": str(person_model), "ppe_model": str(ppe_model),
               "output": tmp_path / "frames.jsonl"}
    if target in {"source", "person_model", "ppe_model"}:
        options["output"] = options[target]
    elif target == "snapshot_source":
        options["snapshot"] = source.parent / "." / source.name
    elif target == "same_outputs":
        options["output"] = options["snapshot"] = tmp_path / "same.png"
    else:
        directory = tmp_path / "ppe_openvino_model"
        directory.mkdir()
        options.update(ppe_model=str(directory), output=directory / "metadata.yaml")
    monkeypatch.setattr(runner, "create_cascade", lambda *_: pytest.fail("weights loaded before validation"))
    monkeypatch.setattr(runner, "open_source", lambda *_: pytest.fail("source opened before validation"))
    with pytest.raises(ValueError, match="substituir|caminhos diferentes"):
        run_detection(**options)
    assert all(path.read_bytes() == b"original input" for path in (source, person_model, ppe_model))


def test_hardlink_alias_of_source_is_protected(monkeypatch, tmp_path):
    source, alias = tmp_path / "source.png", tmp_path / "output.jsonl"
    source.write_bytes(b"original")
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Filesystem does not support local hardlinks")
    monkeypatch.setattr(runner, "create_cascade", lambda *_: pytest.fail("weights loaded before validation"))
    with pytest.raises(ValueError, match="substituir"):
        run_detection(source=str(source), output=alias)
    assert source.read_bytes() == alias.read_bytes() == b"original"


def test_hardlink_alias_of_openvino_weights_is_protected(monkeypatch, tmp_path):
    directory = tmp_path / "ppe_openvino_model"
    directory.mkdir()
    model, alias = directory / "ppe.bin", tmp_path / "output.jsonl"
    model.write_bytes(b"model weights")
    try:
        os.link(model, alias)
    except OSError:
        pytest.skip("Filesystem does not support local hardlinks")
    monkeypatch.setattr(runner, "create_cascade", lambda *_: pytest.fail("weights loaded before validation"))
    with pytest.raises(ValueError, match="substituir"):
        run_detection(source=0, ppe_model=str(directory), output=alias)
    assert model.read_bytes() == alias.read_bytes() == b"model weights"


def test_report_and_snapshot_hardlink_aliases_are_rejected(monkeypatch, tmp_path):
    report, snapshot = tmp_path / "report.jsonl", tmp_path / "snapshot.png"
    report.write_bytes(b"original")
    try:
        os.link(report, snapshot)
    except OSError:
        pytest.skip("Filesystem does not support local hardlinks")
    monkeypatch.setattr(runner, "create_cascade", lambda *_: pytest.fail("weights loaded before validation"))
    with pytest.raises(ValueError, match="caminhos diferentes"):
        run_detection(source=0, output=report, snapshot=snapshot)
    assert report.read_bytes() == snapshot.read_bytes() == b"original"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_frame_limit_fails_before_model_loading(monkeypatch, tmp_path, value):
    monkeypatch.setattr(runner, "create_cascade", lambda *_: pytest.fail("weights loaded before validation"))
    with pytest.raises(ValueError, match="max_frames"):
        run_detection(source=0, max_frames=value, output=tmp_path / "frames.jsonl")
