"""Validate soak controls and honest reporting without models, network or cameras."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def soak():
    path = Path(__file__).resolve().parents[1] / "scripts" / "soak_test.py"
    spec = importlib.util.spec_from_file_location("soak_test_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(soak, tmp_path):
    source = tmp_path / "video.avi"
    source.write_bytes(b"source fixture")
    model = tmp_path / "local.pt"
    model.write_bytes(b"model fixture")
    return soak.parser().parse_args([
        "--source", str(source), "--person-model", str(model), "--ppe-model", str(model),
        "--warmup", "2", "--max-frames", "5", "--sample-every", "2",
        "--output", str(tmp_path / "result.json"),
    ])


class Clock:
    time = 0.

    def __call__(self):
        return self.time


class Source:
    def __init__(self, clock):
        self.clock = clock
        self.opened = False
        self.opens = self.closes = 0

    def open(self):
        self.opened = True
        self.remaining = 2
        self.opens += 1

    def read(self):
        assert self.opened
        self.clock.time += .01
        if self.remaining == 0:
            return None
        self.remaining -= 1
        return object()

    def close(self):
        self.opened = False
        self.closes += 1


class Pipeline:
    def __init__(self, clock, fail_at=None):
        self.clock, self.fail_at, self.calls = clock, fail_at, 0

    def process(self, frame):
        self.calls += 1
        self.clock.time += .1
        if self.calls == self.fail_at:
            raise RuntimeError("Secret from backend: rtsp://user:password@host/camera")
        return SimpleNamespace(counts={"person": 1, "helmet": 1}, ppe_executed=True,
                               assessments=[SimpleNamespace(status="uncertain")])


def execute(soak, args, *, fail_at=None, rss=None):
    clock = Clock()
    source, pipeline = Source(clock), Pipeline(clock, fail_at)
    factory_kwargs = {}

    def factory(*paths, **kwargs):
        factory_kwargs.update(kwargs)
        return pipeline

    report = soak.run(args, pipeline_factory=factory, source_factory=lambda path: source,
                      rss_reader=rss or (lambda: 1024 ** 2 * 100), clock=clock, cpu_clock=clock)
    return report, source, pipeline, factory_kwargs


@pytest.mark.parametrize("source", ["0", "rtsp://secret:password@host/cam", "https://host/video", "//host/share/file", "\\\\host\\share\\file"])
def test_remote_and_camera_inputs_are_rejected_without_echoing_secrets(soak, source):
    with pytest.raises(ValueError) as error:
        soak.local_path(source)
    assert "secret" not in str(error.value)
    assert "password" not in str(error.value)


@pytest.mark.parametrize("name,value", [("max_frames", 0), ("max_seconds", float("nan")),
                                        ("max_seconds", 0), ("sample_every", 0),
                                        ("cpu_threads", -1), ("warmup", -1),
                                        ("imgsz", 16), ("iou", float("inf")),
                                        ("confidence", 2), ("min_fps", 0),
                                        ("max_rss_growth_mib", -1)])
def test_invalid_limits_fail_before_loading_models(soak, args, name, value):
    setattr(args, name, value)
    with pytest.raises(ValueError):
        soak.validate_args(args)


def test_finite_eof_reports_observations_excludes_warmup_and_closes_source(soak, args):
    report, source, pipeline, kwargs = execute(soak, args)
    assert report["status"] == "completed"
    assert report["stop_reason"] == "end_of_file"
    assert report["frames"] == 2
    assert pipeline.calls == 4
    assert report["detection_observations"] == {"person": 2, "helmet": 2}
    assert report["ppe_executed_frames"] == 2
    assert report["person_state_observations"] == {"uncertain": 2}
    assert report["warmup_seconds"] == pytest.approx(.2)
    assert report["latency"]["mean_ms_all_frames"] == pytest.approx(110.)
    assert report["wall_seconds"] == pytest.approx(.23)
    assert not source.opened
    assert source.opens == 2
    assert kwargs["device"] == "cpu"
    assert not report["field_validated"] and not report["accuracy_evaluated"]
    assert json.loads(args.output.read_text(encoding="utf-8")) == report
    with args.output.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["frame"] == "0" and rows[-1]["frame"] == "2"


def test_loop_is_bounded_by_frames_and_reopens_at_eof(soak, args):
    args.loop = True
    report, source, pipeline, _ = execute(soak, args)
    assert report["frames"] == 5 and report["replays"] == 2
    assert report["stop_reason"] == "max_frames"
    assert source.opens == 4 and not source.opened
    assert pipeline.calls == 7


def test_duration_limit_checked_between_frames_and_excludes_warmup(soak, args):
    args.loop, args.max_seconds = True, .15
    report, source, _, _ = execute(soak, args)
    assert report["frames"] == 2
    assert report["stop_reason"] == "max_seconds"
    assert report["wall_seconds"] == pytest.approx(.22)
    assert not source.opened


def test_inference_failure_preserves_partial_report_and_redacts_backend_error(soak, args):
    report, source, _, _ = execute(soak, args, fail_at=4)
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert report["frames"] == 1
    assert not report["requested_checks_passed"]
    assert not source.opened
    output = args.output.read_text(encoding="utf-8")
    assert "password" not in output and "rtsp://" not in output


def test_memory_and_fps_criteria_are_explicit_and_can_fail_completed_run(soak, args):
    values = iter([100, 100, 108, 110])
    args.min_fps, args.max_rss_growth_mib = 20., 5.
    report, _, _, _ = execute(soak, args, rss=lambda: next(values) * 1024 ** 2)
    assert report["status"] == "completed"
    assert report["memory"]["sampled_peak_growth_mib"] == 10
    assert report["checks"] == {"execution_completed": True, "minimum_fps": False,
                                "maximum_sampled_rss_growth": False}
    assert not report["requested_checks_passed"]


def test_final_resource_sampling_failure_still_persists_and_closes(soak, args):
    values = iter([100, 100, 100])
    report, source, _, _ = execute(soak, args, rss=lambda: next(values))
    assert report["status"] == "failed" and report["frames"] == 2
    assert not source.opened and args.output.exists()


def test_output_cannot_overwrite_source_or_model_directory(soak, args, tmp_path):
    folder = tmp_path / "ppe_openvino_model"
    folder.mkdir()
    args.ppe_model, args.output = str(folder), folder / "result.json"
    with pytest.raises(ValueError, match="sobrescrever"):
        soak.validate_args(args)


def test_optional_boots_model_forwarded_only_when_supplied(soak, args):
    args.boots_model = args.ppe_model
    report, _, _, kwargs = execute(soak, args)
    assert "boots_model" in kwargs and "boots_model" in report["models"]


def test_percentiles_and_streaming_fingerprints(soak, tmp_path):
    assert soak.percentile([], .95) is None
    assert soak.percentile([20, 10, 30], .95) == 29.
    weights = tmp_path / "model_openvino_model"
    weights.mkdir()
    (weights / "model.xml").write_bytes(b"xml")
    (weights / "model.bin").write_bytes(b"binary")
    result = soak.fingerprint(weights)
    assert [file["file"] for file in result["files"]] == ["model.bin", "model.xml"]
    assert all(len(file["sha256"]) == 64 for file in result["files"])
