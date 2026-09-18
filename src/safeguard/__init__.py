"""SafeGuard: captura, inferência e apresentação desacopladas."""

import os
from pathlib import Path

# Runtime inference must never install packages or write configuration to a
# user's home implicitly. Setup/export commands own dependency installation.
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".cache" / "ultralytics"))
try:
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # Ultralytics chooses a writable temporary location on read-only hosts.

__version__ = "2.0.0"
