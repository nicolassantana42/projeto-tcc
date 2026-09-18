import csv
import io
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from safeguard.reporting import build_report, encode_snapshot, frame_record


def sample_result(label="helmet"):
    return SimpleNamespace(
        frame_index=3, inference_ms=12.5, pipeline_ms=15.0,
        counts={label: 1}, alerts=["Revisar área"],
        detections=[SimpleNamespace(class_id=0, label=label, confidence=.91, bbox=(1, 2, 30, 40))],
    )


def test_report_preserves_detection_and_snapshot():
    record = frame_record(sample_result(), observed_fps=9.8)
    snapshot = encode_snapshot(np.zeros((16, 16, 3), dtype=np.uint8))
    report = build_report([record], snapshot_png=snapshot, metadata={"source": "Webcam local"})
    with ZipFile(io.BytesIO(report)) as archive:
        assert set(archive.namelist()) == {"report.json", "detections.csv", "snapshot.png"}
        document = json.loads(archive.read("report.json"))
        assert document["retained_frames"] == 1
        assert document["frames"][0]["detections"][0]["confidence"] == .91
        assert document["metadata"]["source"] == "Webcam local"
        assert archive.read("snapshot.png").startswith(b"\x89PNG\r\n\x1a\n")
        rows = list(csv.DictReader(io.StringIO(archive.read("detections.csv").decode("utf-8-sig"))))
        assert rows[0]["frame_index"] == "3"
        assert rows[0]["label"] == "helmet"
        assert rows[0]["x2"] == "30.0"


def test_illustration_never_exports_fake_inference_metrics():
    record = frame_record(sample_result(), illustrative=True, observed_fps=100)
    assert record["mode"] == "illustrative_preview"
    assert record["observed_fps"] is None
    assert record["inference_ms"] is None
    assert record["pipeline_ms"] is None
    assert record["detections"][0]["confidence"] is None


@pytest.mark.parametrize("label", ["=1+1", "+SUM(A1:A2)", "-2+3", "@SUM(A1)", " \t=cmd()"])
def test_csv_neutralizes_formula_labels(label):
    report = build_report([frame_record(sample_result(label))])
    with ZipFile(io.BytesIO(report)) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("detections.csv").decode("utf-8-sig"))))
    assert rows[0]["label"] == "'" + label


def test_empty_detection_frame_is_retained_in_csv():
    result = sample_result()
    result.detections = []
    result.counts = {}
    report = build_report([frame_record(result)])
    with ZipFile(io.BytesIO(report)) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("detections.csv").decode("utf-8-sig"))))
    assert len(rows) == 1
    assert rows[0]["label"] == ""


def test_empty_report_is_valid():
    with ZipFile(io.BytesIO(build_report([]))) as archive:
        assert json.loads(archive.read("report.json"))["retained_frames"] == 0


def test_empty_snapshot_has_clear_error():
    with pytest.raises(ValueError, match="frame válido"):
        encode_snapshot(np.empty((0, 0, 3), dtype=np.uint8))
