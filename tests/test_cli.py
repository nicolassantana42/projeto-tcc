"""Command routing and concise, truthful terminal output without model access."""
import json

import pytest

from safeguard.cli import main, parser


@pytest.mark.parametrize("valid,code", [(True, 0), (False, 1)])
def test_audit_exit_status_and_bounded_output(monkeypatch, capsys, valid, code):
    from safeguard import dataset_audit
    report = {"valid": valid, "summary": {"images": 5000}, "class_names": {0: "person"},
              "errors": [{"message": "bad label"}] * 100, "warnings": [],
              "resolved_splits": {"train": ["SENTINEL_PRIVATE_PATH"] * 5000},
              "splits": {"train": {"files": ["SENTINEL_PRIVATE_PATH"] * 5000}}}
    monkeypatch.setattr(dataset_audit, "audit_dataset", lambda **kwargs: report)
    assert main(["audit-data", "--data", "fixture.yaml", "--output", "audit.json"]) == code
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    assert result["report"] == "audit.json"
    assert len(result["errors_sample"]) == 20
    assert "SENTINEL_PRIVATE_PATH" not in stdout


def test_evaluation_keeps_error_details_in_report_only(monkeypatch, capsys):
    from safeguard import evaluation, factory
    sentinel = object()
    monkeypatch.setattr(factory, "create_cascade", lambda *args: sentinel)

    def evaluate(pipeline, **kwargs):
        assert pipeline is sentinel
        assert kwargs["split"] == "test"
        return {"micro": {"tp": 3}, "errors": [{"image": "DETAIL_SENTINEL"}] * 100,
                "report_path": kwargs["output"]}

    monkeypatch.setattr(evaluation, "evaluate_cascade", evaluate)
    assert main(["evaluate-cascade", "--data", "fixture.yaml"]) == 0
    stdout = capsys.readouterr().out
    assert json.loads(stdout)["error_records_in_report"] == 100
    assert "DETAIL_SENTINEL" not in stdout


def test_detect_defaults_to_local_trained_weights():
    args = parser().parse_args(["detect", "--source", "sample.png"])
    assert args.ppe_model == "models/ppe/best.pt"
    assert not args.show and not args.save_events


def test_benchmark_dispatches_both_model_paths(monkeypatch):
    from safeguard import ml
    calls = {}

    def benchmark(**kwargs):
        calls.update(kwargs)
        return {"kind": "cascade_benchmark"}

    monkeypatch.setattr(ml, "benchmark_model", benchmark)
    assert main(["benchmark", "--source", "sample.avi", "--model", "person.pt", "--ppe-model", "equipment.pt"]) == 0
    assert calls["model_path"] == "person.pt"
    assert calls["ppe_model"] == "equipment.pt"


@pytest.mark.parametrize("collision", ["output_source", "snapshot_source", "output_weights", "output_snapshot"])
def test_legacy_infer_protects_inputs_before_loading_model(monkeypatch, tmp_path, collision):
    from safeguard.inference import YOLODetector
    source, weights = tmp_path / "photo.png", tmp_path / "best.pt"
    source.write_bytes(b"original-image")
    weights.write_bytes(b"original-weights")
    monkeypatch.setattr(YOLODetector, "load", lambda self: pytest.fail("Model must not load for invalid destinations"))
    output, snapshot = tmp_path / "observations.jsonl", tmp_path / "snapshot.png"
    if collision == "output_source":
        output = source
    elif collision == "snapshot_source":
        snapshot = source
    elif collision == "output_weights":
        output = weights
    else:
        output = snapshot
    assert main(["infer", "--source", str(source), "--model", str(weights),
                 "--output", str(output), "--snapshot", str(snapshot)]) == 1
    assert source.read_bytes() == b"original-image"
    assert weights.read_bytes() == b"original-weights"
