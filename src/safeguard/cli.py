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

    detect = commands.add_parser("detect", help="pessoa → segundo YOLO de EPI, em imagem, vídeo ou câmera")
    detect.add_argument("--source", type=source_value, required=True)
    detect.add_argument("--person-model", default="models/yolo11n.pt")
    detect.add_argument("--ppe-model", default="models/ppe/best.pt")
    detect.add_argument("--device", default="auto")
    detect.add_argument("--imgsz", type=positive, default=640)
    detect.add_argument("--confidence", type=probability, default=.4)
    detect.add_argument("--iou", type=probability, default=.45)
    detect.add_argument("--max-frames", type=positive, default=300)
    detect.add_argument("--output", default="runs/detection/frames.jsonl")
    detect.add_argument("--snapshot")
    detect.add_argument("--show", action="store_true", help="janela simples OpenCV; Q encerra")
    detect.add_argument("--save-events", action="store_true")
    detect.add_argument("--camera-name", default="Câmera 01")
    detect.add_argument("--location", default="Local não informado")
    detect.add_argument("--event-directory", default="reports/occurrences")

    audit = commands.add_parser("audit-data", help="verificar rótulos, imagens e duplicatas entre splits")
    audit.add_argument("--data", required=True)
    audit.add_argument("--require-test", action="store_true")
    audit.add_argument("--allow-background", action="store_true")
    audit.add_argument("--output", default="runs/dataset-audit.json")

    evaluate = commands.add_parser("evaluate-cascade", help="TP/FP/FN e latência do fluxo completo, sem confundir com mAP")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--person-model", default="models/yolo11n.pt")
    evaluate.add_argument("--ppe-model", default="models/ppe/best.pt")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--imgsz", type=positive, default=640)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--confidence", type=probability, default=.4)
    evaluate.add_argument("--iou", type=probability, default=.45)
    evaluate.add_argument("--match-iou", type=probability, default=.5)
    evaluate.add_argument("--max-images", type=positive)
    evaluate.add_argument("--output", default="runs/cascade-evaluation.json")

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
    benchmark.add_argument("--ppe-model", help="medir a cascata completa; --model passa a ser o detector de pessoas")
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

    from safeguard.capture import open_source
    from safeguard.config import InferenceConfig
    from safeguard.inference import YOLODetector
    from safeguard.pipeline import Pipeline
    from safeguard.rendering import render_frame

    if args.ppe:
        from safeguard.runner import run_detection
        return run_detection(source=args.source, ppe_model=args.model, device=args.device,
                             imgsz=args.imgsz, confidence=args.confidence, iou=args.iou,
                             max_frames=args.max_frames, output=args.output, snapshot=args.snapshot)

    from safeguard.runner import _output_paths
    if args.snapshot and Path(args.snapshot).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ml.WorkflowError("Snapshot precisa ter extensão .jpg ou .png.")
    destination, snapshot = _output_paths(args.source, args.model, args.model, args.output, args.snapshot)
    detector = YOLODetector(InferenceConfig(model_path=args.model, device=args.device,
                                          confidence=args.confidence, iou=args.iou, imgsz=args.imgsz)).load()
    pipeline = Pipeline(detector, demo_mode=not args.ppe)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, last, totals = 0, None, {}
    with open_source(args.source) as source, destination.open("w", encoding="utf-8") as stream:
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
    if snapshot is not None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(snapshot.suffix, render_frame(last))
        if not success:
            raise ml.WorkflowError(f"Não foi possível gravar snapshot em {snapshot}.")
        encoded.tofile(str(snapshot))
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
        if command == "detect":
            from safeguard.runner import run_detection
            result = run_detection(**values)
        elif command == "audit-data":
            from safeguard.dataset_audit import audit_dataset
            result = audit_dataset(**values)
            summary = {"valid": result["valid"], "summary": result["summary"],
                       "class_names": result["class_names"], "report": str(args.output),
                       "errors_sample": result["errors"][:20], "warnings_sample": result["warnings"][:20]}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if result["valid"] else 1
        elif command == "evaluate-cascade":
            from safeguard.factory import create_cascade
            from safeguard.evaluation import evaluate_cascade
            pipeline = create_cascade(values.pop("person_model"), values.pop("ppe_model"),
                                      values.pop("device"), values.pop("imgsz"), values["confidence"], values["iou"])
            result = evaluate_cascade(pipeline, **values)
        elif command == "infer":
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
        printable = {key: value for key, value in result.items() if key not in {"curves", "confusion_matrix", "errors"}}
        if "errors" in result:
            printable["error_records_in_report"] = len(result["errors"])
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
