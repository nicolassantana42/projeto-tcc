"""Integration smoke with real COCO weights; not an accuracy benchmark.

Run after installing .[onnx] and downloading models/yolo11n.pt:
    python scripts/smoke.py
Outputs are kept under runs/smoke and ignored by Git.
"""
from pathlib import Path
import json
import subprocess
import sys

import safeguard  # Configure local Ultralytics settings before its import.
import cv2
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "runs/smoke"
    output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise RuntimeError("A imagem de exemplo bus.jpg não está disponível no pacote Ultralytics.")
    image = cv2.resize(image, (480, 640))
    video = output / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (480, 640))
    if not writer.isOpened():
        raise RuntimeError("Codec MJPG indisponível para gerar o vídeo de teste.")
    try:
        for _ in range(35):
            writer.write(image)
    finally:
        writer.release()

    def run(*args):
        subprocess.run([sys.executable, "-m", "safeguard", *args], cwd=ROOT, check=True)

    run("infer", "--source", str(video), "--max-frames", "8", "--device", "cpu",
        "--output", str(output / "pt.jsonl"), "--snapshot", str(output / "pt.png"))
    run("export", "--format", "onnx", "--precision", "fp32", "--device", "cpu")
    run("infer", "--source", str(video), "--model", "models/yolo11n.onnx",
        "--max-frames", "8", "--device", "cpu", "--output", str(output / "onnx.jsonl"),
        "--snapshot", str(output / "onnx.png"))
    for suffix in ("pt", "onnx"):
        rows = [json.loads(line) for line in (output / f"{suffix}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 8, f"{suffix}: quantidade de frames incorreta"
        assert all(row["counts"].get("person", 0) > 0 for row in rows), f"{suffix}: nenhuma pessoa detectada"
        assert all(not row["alerts"] for row in rows), "COCO não pode acusar ausência de EPI"
        run("benchmark", "--source", str(video), "--model", f"models/yolo11n.{suffix}",
            "--frames", "20", "--warmup", "3", "--device", "cpu",
            "--output", str(output / f"benchmark-{suffix}.json"))
    print("SMOKE OK: inferência PT/ONNX, exportação, snapshots e benchmark executados em CPU.")


if __name__ == "__main__":
    main()
