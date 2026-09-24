"""Audit local YOLO annotations: python scripts/audit_dataset.py --data data/ppe.yaml."""

from pathlib import Path
import sys

# Also usable from a checkout before an editable install (dependencies required).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safeguard.dataset_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
