"""Download the pinned YOLO11n PPE candidate, including explicit negative classes.

External pretrained weights, not the result of local TCC training. Evaluate on
validation before adopting them. This script never replaces the active model.
"""
from pathlib import Path
import json

from prepare_ppe import ensure_download, _save_provenance


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "yihong1120/Construction-Hazard-Detection"
REVISION = "212ee245136b4e330f84b409f67f3d35eff59f42"
FILENAME = "models/yolo11/pt/yolo11n.pt"
SHA256 = "f55600c106ba7952c64d2aec70dc673240b422bdbd989d395d301b0d5426b02b"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"


def main():
    target = ROOT / "models/ppe/absence-base.pt"
    ensure_download(URL, target, SHA256)
    report = {"source": f"https://huggingface.co/{REPOSITORY}", "revision": REVISION,
              "file": FILENAME, "sha256": SHA256, "publisher_license": "AGPL-3.0",
              "role": "External pretrained YOLO11n candidate; not local TCC training.",
              "training_overlap_with_evaluation": "unknown",
              "note": "Evaluate with evaluate-cascade using canonical class names before adoption."}
    _save_provenance(target.with_suffix(".provenance.json"), report)
    print(json.dumps({"candidate": str(target), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
