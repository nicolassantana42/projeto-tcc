"""Exercise plots and optional INT8 export on a tiny, artificial COCO fixture.

This checks API integration ONLY: repeated images and approximate labels are
not a representative calibration/evaluation dataset. Do not report its mAP or
ship its quantized weights. Use docs/ML.md and real PPE data for that purpose.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import safeguard
import cv2
import yaml
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    output = ROOT / "runs/smoke-validation"
    images, labels = output / "fixture/images/val", output / "fixture/labels/val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise RuntimeError("Imagem bus.jpg ausente no pacote Ultralytics.")
    for index in range(6):
        if not cv2.imwrite(str(images / f"sample{index}.jpg"), image):
            raise RuntimeError("Não foi possível gravar fixture.")
        # Deliberately approximate label: this is a schema/plot smoke, not GT.
        (labels / f"sample{index}.txt").write_text("0 0.17 0.6 0.22 0.48\n", encoding="utf-8")
    model = ROOT / "models/yolo11n.pt"
    names = YOLO(str(model)).names
    dataset = output / "fixture.yaml"
    dataset.write_text(yaml.safe_dump({"path": str(images.parents[1]), "train": "images/val",
                                      "val": "images/val", "names": names}), encoding="utf-8")
    (output / "NOT_SCIENTIFIC_RESULTS.txt").write_text(__doc__, encoding="utf-8")

    def run(*command):
        subprocess.run([sys.executable, "-m", "safeguard", *command], cwd=ROOT, check=True)

    run("validate", "--model", str(model), "--data", str(dataset), "--device", "cpu",
        "--project", str(output), "--name", "plots")
    if args.int8:
        isolated_model = output / "smoke_only.pt"
        shutil.copyfile(model, isolated_model)
        run("export", "--model", str(isolated_model), "--format", "openvino", "--precision", "int8",
            "--data", str(dataset), "--device", "cpu")
        from safeguard.config import InferenceConfig
        from safeguard.inference import YOLODetector
        detector = YOLODetector(InferenceConfig(model_path=str(output / "smoke_only_int8_openvino_model"))).load()
        for _ in range(2):
            assert detector.predict(image), "Nenhuma saída do OpenVINO INT8"
        print("INT8 API SMOKE OK. Artefato artificial; não usar para avaliar/implantar detecção de EPI.")
    print("VALIDATION API SMOKE OK. Métricas artificiais não são resultados científicos.")


if __name__ == "__main__":
    main()
