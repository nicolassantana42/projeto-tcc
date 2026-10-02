"""Measure the local CPU cascade over a bounded file replay, without UI or alerts.

This operational check does not measure model accuracy or validate a camera,
network, GPU, notification delivery, or the Streamlit application's memory.
Limits are checked between frames; they cannot interrupt a hung native call.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter, process_time


def local_path(value: str, *, model: bool = False) -> Path:
    # Never allow numeric webcam indices, remote URLs or UNC/network shares.
    text = str(value).strip()
    if not text or text.isdecimal() or "://" in text or text.startswith(("\\\\", "//")):
        raise ValueError("Informe um caminho local existente; câmera e rede não são permitidas neste ensaio.")
    path = Path(text).expanduser().resolve()
    if str(path).startswith(("\\\\", "//")):
        raise ValueError("A fonte resolvida deve estar em um caminho local.")
    if not (path.is_file() or model and path.is_dir()):
        raise ValueError("Arquivo local ou diretório de modelo não encontrado.")
    return path


def fingerprint(path: Path) -> dict:
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    hashes = []
    for item in files:
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes.append({"file": str(item.relative_to(path)) if path.is_dir() else item.name,
                       "bytes": item.stat().st_size, "sha256": digest.hexdigest()})
    return {"path": str(path), "files": hashes}


def percentile(values, quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, help="Imagem ou vídeo local. Sem webcam, URL ou UNC.")
    result.add_argument("--person-model", default="models/yolo11n.pt")
    result.add_argument("--ppe-model", default="models/ppe/absence.pt")
    result.add_argument("--boots-model", help="Modelo local opcional, somente se a factory oferecer esse suporte.")
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--confidence", type=float, default=.4)
    result.add_argument("--iou", type=float, default=.45)
    result.add_argument("--max-frames", type=int, default=5000)
    result.add_argument("--max-seconds", type=float, default=300.)
    result.add_argument("--loop", action="store_true", help="Reabra o arquivo no EOF até atingir um limite.")
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--sample-every", type=int, default=25, help="Intervalo de amostragem de RSS, em quadros.")
    result.add_argument("--max-rss-growth-mib", type=float, help="Critério opcional: pico RSS amostrado menos baseline após aquecimento.")
    result.add_argument("--min-fps", type=float, help="Critério opcional de vazão média local, não FPS da interface.")
    result.add_argument("--output", type=Path, default=Path("runs/soak/summary.json"))
    return result


def validate_args(args) -> None:
    for name in ("cpu_threads", "max_frames", "sample_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} deve ser positivo.")
    if args.warmup < 0 or args.imgsz < 32:
        raise ValueError("warmup deve ser não negativo; imgsz deve ser pelo menos 32.")
    for name in ("max_seconds", "min_fps"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} deve ser positivo e finito.")
    if args.max_rss_growth_mib is not None and (not math.isfinite(args.max_rss_growth_mib) or args.max_rss_growth_mib < 0):
        raise ValueError("max_rss_growth_mib deve ser não negativo e finito.")
    for name in ("confidence", "iou"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} deve estar entre 0 e 1.")
    args.source = local_path(args.source)
    for name in ("person_model", "ppe_model", "boots_model"):
        if getattr(args, name) is not None:
            setattr(args, name, local_path(getattr(args, name), model=True))
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".json":
        raise ValueError("O relatório de saída deve ter extensão .json.")
    for artifact in (args.output, args.output.with_suffix(".csv")):
        for source in (args.source, args.person_model, args.ppe_model, args.boots_model):
            if source is not None and (artifact == source or source.is_dir() and source in artifact.parents):
                raise ValueError("A saída não pode sobrescrever a fonte ou os modelos.")


def run(args, *, pipeline_factory=None, source_factory=None, rss_reader=None,
        clock=perf_counter, cpu_clock=process_time) -> dict:
    """Run serially and persist partial diagnostics on recoverable errors.

    Factories/clocks are injectable for tests; production uses the real cascade.
    No events, reports of people, notifiers or UI components are instantiated.
    """
    validate_args(args)
    os.environ["EPI_CPU_THREADS"] = str(args.cpu_threads)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    if pipeline_factory is None:
        from epi_monitor.factory import create_cascade
        pipeline_factory = create_cascade
    if source_factory is None:
        from epi_monitor.capture import open_source
        source_factory = open_source
    if rss_reader is None:
        try:
            import psutil
        except ImportError as error:
            raise ValueError("psutil ausente; instale as dependências do projeto antes deste ensaio.") from error
        process = psutil.Process()
        rss_reader = lambda: process.memory_info().rss

    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed", "scope": __doc__.strip(), "device": "cpu",
        "settings": {key: getattr(args, key) for key in ("cpu_threads", "imgsz", "confidence", "iou", "max_frames", "max_seconds", "loop", "warmup", "sample_every", "max_rss_growth_mib", "min_fps")},
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "versions": {}, "source": fingerprint(args.source),
        "models": {name: fingerprint(getattr(args, name)) for name in ("person_model", "ppe_model", "boots_model") if getattr(args, name) is not None},
        "frames": 0, "replays": 0, "ppe_executed_frames": 0,
        "sample_csv": str(args.output.with_suffix(".csv")),
        "accuracy_evaluated": False, "field_validated": False,
    }
    for package in ("torch", "ultralytics", "opencv-python", "openvino", "psutil"):
        try:
            report["versions"][package] = version(package)
        except PackageNotFoundError:
            report["versions"][package] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = None
    recent_latency = deque(maxlen=10000)
    labels, states = Counter(), Counter()
    latency_total = 0.
    measured_start = cpu_start = None
    baseline_rss = last_rss = peak_rss = None
    end_wall = end_cpu = None
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame", "elapsed_seconds", "process_cpu_seconds", "rss_bytes", "last_frame_ms", "replays"))
        writer.writeheader()

        def sample():
            nonlocal last_rss, peak_rss
            last_rss = rss_reader()
            peak_rss = max(peak_rss, last_rss) if peak_rss is not None else last_rss
            writer.writerow({"frame": report["frames"], "elapsed_seconds": clock() - measured_start,
                             "process_cpu_seconds": cpu_clock() - cpu_start, "rss_bytes": last_rss,
                             "last_frame_ms": recent_latency[-1] if recent_latency else None,
                             "replays": report["replays"]})
            stream.flush()

        try:
            source = source_factory(str(args.source))
            source.open()
            first = source.read()
            if first is None:
                raise ValueError("Fonte sem quadros decodificáveis.")
            load_start = clock()
            kwargs = dict(device="cpu", imgsz=args.imgsz, confidence=args.confidence, iou=args.iou)
            if args.boots_model is not None:
                kwargs["boots_model"] = str(args.boots_model)
            pipeline = pipeline_factory(str(args.person_model), str(args.ppe_model), **kwargs)
            report["model_load_seconds"] = clock() - load_start
            warm_start = clock()
            for _ in range(args.warmup):
                pipeline.process(first)
            report["warmup_seconds"] = clock() - warm_start
            # Reopening makes every measured replay start with source frame one.
            source.close()
            source.open()
            del first
            baseline_rss = peak_rss = rss_reader()
            measured_start, cpu_start = clock(), cpu_clock()
            sample()
            while True:
                if report["frames"] >= args.max_frames:
                    report["stop_reason"] = "max_frames"
                    break
                if clock() - measured_start >= args.max_seconds:
                    report["stop_reason"] = "max_seconds"
                    break
                frame_start = clock()
                frame = source.read()
                if frame is None:
                    if not args.loop:
                        report["stop_reason"] = "end_of_file"
                        break
                    source.close()
                    source.open()
                    frame = source.read()
                    if frame is None:
                        raise ValueError("A fonte ficou vazia durante o replay.")
                    report["replays"] += 1
                result = pipeline.process(frame)
                elapsed_ms = (clock() - frame_start) * 1000
                latency_total += elapsed_ms
                recent_latency.append(elapsed_ms)
                report["frames"] += 1
                report["ppe_executed_frames"] += int(result.ppe_executed)
                labels.update(result.counts)
                states.update(item.status for item in result.assessments)
                if report["frames"] % args.sample_every == 0:
                    sample()
                # Keep only aggregate counters, not frames or detection histories.
                del frame, result
            if report["frames"] == 0:
                raise ValueError("Nenhum quadro foi medido; aumente o limite de tempo.")
            report["status"] = "completed"
        except (Exception, KeyboardInterrupt) as error:
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["error_type"] = type(error).__name__
            # Third-party messages may contain source credentials. Never persist
            # raw exception text, even though this command accepts local inputs.
            report["error"] = "Ensaio interrompido; verifique fonte, modelos e dependências locais. Resultados parciais não aprovam estabilidade."
        finally:
            if measured_start is not None:
                end_wall, end_cpu = clock(), cpu_clock()
                try:
                    sample()
                except Exception as error:
                    report.update(status="failed", error_type=type(error).__name__,
                                  error="Não foi possível concluir a amostragem de recursos; resultado parcial.")
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    report.update(status="failed", error_type=type(error).__name__,
                                  error="Falha ao fechar a fonte local; resultado parcial.")

    wall = end_wall - measured_start if end_wall is not None else 0.
    cpu = end_cpu - cpu_start if end_cpu is not None else 0.
    growth_mib = (peak_rss - baseline_rss) / 1024 ** 2 if baseline_rss is not None else None
    fps = report["frames"] / wall if wall else 0.
    report.update({
        "wall_seconds": wall, "process_cpu_seconds": cpu, "fps": fps,
        "mean_utilized_logical_cores": cpu / wall if wall else None,
        "estimated_host_cpu_percent": 100 * cpu / wall / (os.cpu_count() or 1) if wall else None,
        "latency": {"scope": "local file decode plus cascade; excludes model loading and warmup",
                    "mean_ms_all_frames": latency_total / report["frames"] if report["frames"] else None,
                    "window_frames": len(recent_latency), "window_limit": recent_latency.maxlen,
                    "median_ms_last_window": statistics.median(recent_latency) if recent_latency else None,
                    "p95_ms_last_window": percentile(recent_latency, .95),
                    "max_ms_last_window": max(recent_latency) if recent_latency else None},
        "memory": {"metric": "process RSS sampled after warmup; excludes GPU memory; sampled peak may miss short spikes",
                   "baseline_bytes": baseline_rss, "final_bytes": last_rss, "sampled_peak_bytes": peak_rss,
                   "sampled_peak_growth_mib": growth_mib},
        "detection_observations": dict(labels), "person_state_observations": dict(states),
    })
    checks = {"execution_completed": report["status"] == "completed"}
    if args.min_fps is not None:
        checks["minimum_fps"] = fps >= args.min_fps
    if args.max_rss_growth_mib is not None:
        checks["maximum_sampled_rss_growth"] = growth_mib is not None and growth_mib <= args.max_rss_growth_mib
    report["checks"] = checks
    report["requested_checks_passed"] = all(checks.values())
    report["criteria_note"] = "Sem limites opcionais, completed apenas confirma término local sem erro; não comprova ausência de vazamento nem qualidade de detecção."
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        report = run(args)
    except ValueError as error:
        cli.error(str(error))
    print(json.dumps({key: report[key] for key in ("status", "frames", "wall_seconds", "fps", "memory", "checks")}, ensure_ascii=False))
    print(f"Relatório: {args.output}")
    return 0 if report["requested_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
