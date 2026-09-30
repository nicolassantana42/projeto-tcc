"""Measure CPU thread budgets on the real cascade without changing image size.

Run from the repository root with the project environment. Images are decoded
before timing, so the result measures the cascade, not capture/UI/network I/O.
This benchmark checks numerical equivalence; it does not measure accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter, process_time

import cv2
import numpy as np
import torch

from epi_monitor.factory import create_cascade


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def signature(result) -> dict:
    detections = sorted(result.detections, key=lambda d: (d.label, *d.bbox))
    return {
        "labels": [d.label for d in detections],
        "boxes": np.asarray([d.bbox for d in detections], dtype=float).reshape(-1, 4),
        "scores": np.asarray([d.confidence for d in detections], dtype=float),
        "states": [(a.index, a.status, a.present, a.absent, a.uncertain) for a in result.assessments],
        "counts": result.counts,
    }


def compare(reference: dict, actual: dict) -> dict:
    aligned = reference["labels"] == actual["labels"]
    boxes_equal = aligned and np.allclose(reference["boxes"], actual["boxes"], atol=.01, rtol=0)
    scores_equal = aligned and np.allclose(reference["scores"], actual["scores"], atol=1e-5, rtol=0)
    return {
        "same_counts": reference["counts"] == actual["counts"],
        "same_assessments": reference["states"] == actual["states"],
        "boxes_within_0_01_px": bool(boxes_equal),
        "scores_within_1e_5": bool(scores_equal),
        "max_box_delta_px": float(np.max(np.abs(reference["boxes"] - actual["boxes"]))) if aligned and actual["boxes"].size else None,
        "max_confidence_delta": float(np.max(np.abs(reference["scores"] - actual["scores"]))) if aligned and actual["scores"].size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/datasets/ppe-absence/valid/images"))
    parser.add_argument("--person-model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--ppe-model", type=Path, default=Path("models/ppe/absence.pt"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("runs/performance/cpu-threads.json"))
    args = parser.parse_args()
    if min(args.threads + [args.frames, args.warmup, args.imgsz]) < 1:
        parser.error("Thread counts, frames, warmup and image size must be positive.")
    paths = [next(args.images.glob(prefix + "*.jpg")) for prefix in ("ppe_0006", "ppe_0079", "ppe_0164", "ppe_0565", "ppe_0021")]
    frames = [cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) for path in paths]
    if any(frame is None for frame in frames):
        raise ValueError("One of the benchmark images could not be decoded.")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "5 validation images repeated; warmed CPU cascade only; no UI/capture/network; not an accuracy or field validation",
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(), "torch": torch.__version__, "opencv": cv2.__version__},
        "person_model": fingerprint(args.person_model), "ppe_model": fingerprint(args.ppe_model),
        "images": [fingerprint(path) for path in paths],
        "imgsz": args.imgsz, "confidence": .4, "iou": .45,
        "frames_per_variant": args.frames, "warmup_per_variant": args.warmup,
        "torch_threads_before_load": torch.get_num_threads(),
        "configured_epi_cpu_threads": os.environ.get("EPI_CPU_THREADS"),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(), "variants": [],
    }
    cascade = create_cascade(str(args.person_model), str(args.ppe_model), device="cpu", imgsz=args.imgsz, confidence=.4, iou=.45)
    # Both predictors are lazy; the first real prediction may change PyTorch's
    # global thread count inside Ultralytics select_device(). Observe it first.
    cascade.process(frames[0])
    if not cascade.ppe_detector._model.predictor:
        cascade.ppe_detector.predict(frames[0])
    report["torch_threads_after_predictor_initialization"] = torch.get_num_threads()
    references = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for requested in [None, *args.threads]:
        if requested is not None:
            torch.set_num_threads(requested)
        for index in range(args.warmup):
            cascade.process(frames[index % len(frames)])
        effective = torch.get_num_threads()
        if requested is not None and effective != requested:
            raise RuntimeError(f"Runtime overrode requested threads {requested}: effective={effective}")
        elapsed, checks, stages = [], [], []
        start_cpu, start_wall = process_time(), perf_counter()
        for index in range(args.frames):
            image_index = index % len(frames)
            start = perf_counter()
            result = cascade.process(frames[image_index])
            elapsed.append((perf_counter() - start) * 1000)
            current = signature(result)
            if requested is None:
                references.setdefault(image_index, current)
            checks.append(compare(references[image_index], current))
            stages.append(result.stage_timings_ms)
        wall, cpu = perf_counter() - start_wall, process_time() - start_cpu
        variant = {
            "label": "default_after_init" if requested is None else str(requested),
            "requested_threads": requested, "effective_threads": torch.get_num_threads(),
            "wall_seconds": wall, "process_cpu_seconds": cpu,
            "mean_utilized_logical_cores": cpu / wall,
            "estimated_host_cpu_percent": 100 * cpu / wall / (os.cpu_count() or 1),
            "fps": args.frames / wall, "mean_ms": statistics.mean(elapsed),
            "median_ms": statistics.median(elapsed), "p95_ms": float(np.percentile(elapsed, 95)),
            "per_frame_ms": elapsed,
            "mean_stage_ms": {key: statistics.mean(item.get(key, 0.) for item in stages) for key in set().union(*(item.keys() for item in stages))},
            "equivalent_outputs": all(all(check[key] for key in ("same_counts", "same_assessments", "boxes_within_0_01_px", "scores_within_1e_5")) for check in checks),
            "comparison": checks,
        }
        report["variants"].append(variant)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in variant.items() if key not in {"per_frame_ms", "comparison", "mean_stage_ms"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
