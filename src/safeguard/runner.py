"""Detection from images, videos or cameras, independent of Streamlit."""
from collections import Counter
from pathlib import Path
import json
import os
import time

import cv2

from .capture import ImageSource, open_source
from .events import CameraContext, EventPolicy, EventService, EventStore
from .factory import create_cascade
from .rendering import render_frame
from .reporting import frame_record


def _output_paths(source, person_model, ppe_model, output, snapshot):
    """Validate destinations before loading weights or opening any input."""
    destination = Path(output).expanduser().resolve()
    image_path = Path(snapshot).expanduser().resolve() if snapshot else None
    outputs = [destination, *([image_path] if image_path is not None else [])]
    protected = {Path(model).expanduser().resolve() for model in (person_model, ppe_model)}
    # Exported OpenVINO models may be supplied as either a directory or XML.
    model_directories = {path for path in protected if path.is_dir()}
    for directory in model_directories:
        protected.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    for path in tuple(protected):
        if path.suffix.lower() == ".xml":
            protected.add(path.with_suffix(".bin"))
            protected.add(path.parent / "metadata.yaml")
    if isinstance(source, str) and not source.strip().isdecimal() and "://" not in source:
        protected.add(Path(source.strip()).expanduser().resolve())

    def aliases(first, second):
        return first == second or (first.exists() and second.exists() and os.path.samefile(first, second))

    if image_path is not None and aliases(destination, image_path):
        raise ValueError("O relatório e o snapshot precisam de caminhos diferentes.")
    for path in outputs:
        if any(aliases(path, original) for original in protected) or any(
                path.is_relative_to(directory) for directory in model_directories):
            raise ValueError("O relatório e o snapshot não podem substituir a fonte ou os modelos.")
        if path.is_dir():
            raise ValueError("O relatório e o snapshot precisam apontar para arquivos, não diretórios.")
    return destination, image_path


def run_detection(*, source, person_model="models/yolo11n.pt", ppe_model="models/ppe/best.pt",
                  device="auto", imgsz=640, confidence=.4, iou=.45, max_frames=300,
                  output="runs/detection/frames.jsonl", snapshot=None, show=False,
                  save_events=False, camera_name="Câmera 01", location="Local não informado",
                  event_directory="reports/occurrences", confirmation_seconds=2., cooldown_seconds=60.):
    """Record all observations; only explicit unsafe observations become events.

    Image evidence is marked as a single observation. Video confirmation uses
    source timestamps (FPS fallback); live streams use a monotonic clock.
    No network notifications are sent by this entry point.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("max_frames deve ser positivo.")
    if snapshot and Path(snapshot).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Snapshot precisa ter extensão .jpg ou .png.")
    destination, image_path = _output_paths(source, person_model, ppe_model, output, snapshot)
    context = CameraContext(name=camera_name, location=location)
    policy = EventPolicy("ppe", confirmation_seconds, cooldown_seconds)
    pipeline = create_cascade(person_model, ppe_model, device, imgsz, confidence, iou)
    store = EventStore(event_directory) if save_events else None
    events = EventService(store, context, policy, False, pipeline.names) if store else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    statuses, count, executed, last, event_ids = Counter(), 0, 0, None, []
    missing_video_timestamps = 0
    started = time.perf_counter()
    capture = open_source(source)
    try:
        with capture, destination.open("w", encoding="utf-8") as stream:
            for _ in range(max_frames):
                frame = capture.read()
                if frame is None:
                    break
                last = pipeline.process(frame, confidence=confidence, iou=iou)
                row = frame_record(last)
                row["source_time_seconds"] = capture.timestamp_seconds
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                executed += last.ppe_executed
                statuses.update(item.status for item in last.assessments)
                annotated = render_frame(last) if show or save_events else None
                if events:
                    if isinstance(capture, ImageSource):
                        reasons = [f"Imagem estática, observação única: {'; '.join(item.reasons)}"
                                   for item in last.assessments if item.status == "unsafe"]
                        event = store.save(last, annotated, context, "ppe", reasons, False) if reasons else None
                    elif capture.is_file and capture.timestamp_seconds is None:
                        # Slow CPU inference must not turn a short/untimed file
                        # into a temporally confirmed alert. Restart continuity
                        # when the media clock is unavailable.
                        missing_video_timestamps += 1
                        events.reset_confirmation()
                        event = None
                    else:
                        event = events.process(last, annotated, now=capture.timestamp_seconds)
                    if event:
                        event_ids.append(event["id"])
                if show:
                    cv2.imshow("SafeGuard - deteccao | Q para sair", annotated)
                    if cv2.waitKey(0 if isinstance(capture, ImageSource) else 1) & 0xff in (ord("q"), 27):
                        break
    finally:
        if show:
            cv2.destroyAllWindows()
    if last is None:
        raise ValueError("A fonte não produziu imagens válidas.")
    if image_path is not None:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(image_path.suffix, render_frame(last))
        if not success:
            raise OSError("Falha ao codificar o snapshot.")
        encoded.tofile(str(image_path))
    return {
        "frames": count, "ppe_executed_frames": executed, "ppe_skipped_frames": count - executed,
        "person_device": pipeline.person_detector.device, "ppe_device": pipeline.ppe_detector.device,
        "person_model": str(person_model), "ppe_model": str(ppe_model),
        "observations_by_status": dict(statuses), "events_saved": event_ids,
        "elapsed_seconds": time.perf_counter() - started, "output": str(destination),
        "snapshot": str(image_path) if image_path is not None else None,
        "event_timing_warning": (f"{missing_video_timestamps} quadro(s) sem timestamp/FPS utilizável; confirmação temporal reiniciada."
                                 if missing_video_timestamps else ""),
        "retention_warning": store.retention_warning if store else "",
        "note": "Contagens por frame; OK significa EPIs detectados, não certificação de segurança. Sem envio externo.",
    }
