"""Cria ambiente, instala o projeto e inicia o dashboard com um comando."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Abre o dashboard sem baixar pesos COCO")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.error("Use Python 3.10 a 3.13 (recomendado: 3.12).")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stamp = environment / ".safeguard-installed"
    digest = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in ("requirements.txt", "pyproject.toml"))).hexdigest()
    try:
        if not python.exists():
            print("Criando ambiente Python isolado…", flush=True)
            venv.EnvBuilder(with_pip=True).create(environment)
        if subprocess.run([str(python), "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
        if not stamp.exists() or stamp.read_text() != digest:
            subprocess.run([str(python), "-m", "pip", "install", "-e", ".[dev]"], cwd=ROOT, check=True)
            stamp.write_text(digest)
        if not args.no_download and not (ROOT / "models/yolo11n.pt").is_file():
            subprocess.run([str(python), "-m", "safeguard", "download"], cwd=ROOT, check=True)
        if args.install_only:
            return 0
        print(f"Dashboard: http://localhost:{args.port}", flush=True)
        return subprocess.call([
            str(python), "-m", "streamlit", "run", str(ROOT / "src/safeguard/ui/app.py"),
            "--server.port", str(args.port), "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false",
        ], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup interrompido: {exc}\nVerifique Python completo, acesso à internet e espaço em disco; execute novamente.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
