"""Portable command line entry point; no GUI or implicit package installation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from safeguard import ml


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("informe um inteiro positivo")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("informe um valor entre 0 e 1")
    return number


def source_value(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="safeguard", description="SafeGuard — detecção, treinamento e métricas reproduzíveis")
    cli.add_argument("--debug", action="store_true", help="mostrar traceback de diagnóstico")
    commands = cli.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--model", default="models/yolo11n.pt", help="pesos .pt ou artefato exportado local")
        command.add_argument("--device", default="auto", help="auto, cpu, cuda:0 ou mps")
        command.add_argument("--imgsz", type=positive, default=640)

    infer = commands.add_parser("infer", help="inferência sem janela gráfica e relatório JSONL")
    common(infer)
    infer.add_argument("--source", type=source_value, default=0)
    infer.add_argument("--confidence", type=probability, default=0.4)
    infer.add_argument("--iou", type=probability, default=0.45)
    infer.add_argument("--max-frames", type=positive, default=300)
    infer.add_argument("--output", default="runs/detections.jsonl")
    infer.add_argument("--snapshot", help="gravar o último frame anotado neste arquivo .jpg ou .png")
    infer.add_argument("--ppe", action="store_true", help="ativar regras EPI com pesos treinados; padrão: demonstração COCO")

    train = commands.add_parser("train", help="fine-tuning YOLO11/YOLOv8 com dataset EPI")
    common(train)
    train.add_argument("--data", required=True)
    train.add_argument("--epochs", type=positive, default=50)
    train.add_argument("--batch", type=positive, default=8)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--amp", action="store_true", help="ativar mixed precision CUDA; o teste AMP Ultralytics pode baixar pesos auxiliares")
    train.add_argument("--project", default="runs/train")
    train.add_argument("--name", default="ppe")

    validate = commands.add_parser("validate", help="mAP, matriz de confusão, curvas PR e métricas JSON")
    common(validate)
    validate.add_argument("--data", required=True)
    validate.add_argument("--batch", type=positive, default=1)
    validate.add_argument("--split", choices=("val", "test"), default="val")
    validate.add_argument("--confidence", type=probability, default=0.001)
    validate.add_argument("--iou", type=probability, default=0.7)
    validate.add_argument("--project", default="runs/validate")
    validate.add_argument("--name", default="ppe")
    validate.add_argument("--output", help="caminho do relatório JSON; padrão: pasta da validação")

    export = commands.add_parser("export", help="ONNX FP32/FP16 ou OpenVINO/TensorRT INT8 calibrado")
    common(export)
    export.add_argument("--format", choices=("onnx", "openvino", "engine"), default="onnx")
    export.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    export.add_argument("--data", help="dataset representativo de calibração; obrigatório para INT8")
    export.add_argument("--fraction", type=probability, default=1.0)
    export.add_argument("--batch", type=positive, default=1)

    benchmark = commands.add_parser("benchmark", help="warmup e latências p50/p95 medidas localmente")
    common(benchmark)
    benchmark.add_argument("--source", type=source_value, required=True)
    benchmark.add_argument("--confidence", type=probability, default=0.4)
    benchmark.add_argument("--iou", type=probability, default=0.45)
    benchmark.add_argument("--frames", type=positive, default=100)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--output", default="runs/benchmark.json")

    download = commands.add_parser("download", help="download explícito de pesos COCO oficiais para demonstração")
    download.add_argument("--model", choices=ml.OFFICIAL_MODELS, default="yolo11n.pt")
    download.add_argument("--directory", default="models")
    return cli


def run_inference(args: argparse.Namespace) -> dict:
    import cv2

    from safeguard.capture import VideoSource
    from safeguard.config import InferenceConfig
    from safeguard.inference import YOLODetector
    from safeguard.pipeline import Pipeline
    from safeguard.rendering import render_frame

    detector = YOLODetector(InferenceConfig(model_path=args.model, device=args.device,
                                          confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)).load()
    pipeline = Pipeline(detector, demo_mode=not args.ppe)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, last, totals = 0, None, {}
    with VideoSource(args.source) as source, destination.open("w", encoding="utf-8") as stream:
        for _ in range(args.max_frames):
            frame = source.read()
            if frame is None:
                break
            last = pipeline.process(frame)
            row = {"frame_index": last.frame_index, "counts": last.counts,
                   "detections": [asdict(item) for item in last.detections],
                   "alerts": [asdict(item) if hasattr(item, "__dataclass_fields__") else item for item in last.alerts],
                   "inference_ms": last.inference_ms, "pipeline_ms": last.pipeline_ms,
                   "demo_mode": not args.ppe}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            for label, amount in last.counts.items():
                totals[label] = totals.get(label, 0) + amount
    if last is None:
        raise ml.WorkflowError("A fonte não produziu frames válidos.")
    if args.snapshot:
        snapshot = Path(args.snapshot)
        if snapshot.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ml.WorkflowError("Snapshot precisa ter extensão .jpg ou .png.")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(snapshot), render_frame(last)):
            raise ml.WorkflowError(f"Não foi possível gravar snapshot em {snapshot}.")
    return {"frames": count, "device": detector.device, "output": str(destination),
            "detections_accumulated": totals,
            "count_note": "Contagens acumuladas por frame; não representam pessoas/objetos únicos.",
            "demo_mode": not args.ppe}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        values = vars(args).copy()
        command = values.pop("command")
        values.pop("debug")
        if command == "infer":
            result = run_inference(args)
        elif command == "download":
            result = {"model": str(ml.download_model(values.pop("model"), **values)),
                      "note": "Pesos COCO para demonstração; não detectam classes EPI especializadas."}
        else:
            values["model_path"] = values.pop("model")
            operation = {"train": ml.train_model, "validate": ml.validate_model,
                         "export": ml.export_model, "benchmark": ml.benchmark_model}[command]
            result = operation(**values)
        # Dense curves remain in the report, keeping terminal logs readable.
        printable = {key: value for key, value in result.items() if key not in {"curves", "confusion_matrix"}}
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        print("Operação interrompida; recursos de captura liberados.", file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        print(f"Erro: {error}\nUse --debug antes do subcomando para detalhes.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
