#!/usr/bin/env python3
# ============================================================
#  setup_and_test.py — Instalação e Validação do Sistema
#  TCC: Monitoramento de EPI com Visão Computacional
#
#  Execute PRIMEIRO, antes de qualquer outra coisa:
#    python setup_and_test.py
# ============================================================

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def banner():
    print("=" * 60)
    print("  Sistema EPI — Setup e Validação")
    print("  TCC | Visão Computacional e Inteligência Artificial")
    print("=" * 60)


# ── Verificações ──────────────────────────────────────────────

def check_python():
    v = sys.version_info
    tag = "✅" if v >= (3, 9) else "⚠️ "
    print(f"\n{tag} Python {v.major}.{v.minor}.{v.micro}", end="")
    if v < (3, 9):
        print(" (recomendado 3.9+)")
    else:
        print()


def setup_env():
    print("\n⚙️  Verificando .env...")
    env      = ROOT / ".env"
    example  = ROOT / ".env.example"

    if not env.exists():
        if example.exists():
            shutil.copy(example, env)
            print("  ℹ️  .env criado a partir do .env.example")
            print("  → Edite o .env para configurar Telegram e câmera")
        else:
            print("  ❌ .env.example não encontrado!")
            return
    else:
        print("  ✅ .env encontrado")

    # Verifica Telegram
    from dotenv import load_dotenv
    load_dotenv(env)
    token   = os.getenv("TELEGRAM_TOKEN",   "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if (token and chat_id and
            token   not in ("seu_token_aqui", "") and
            chat_id not in ("seu_chat_id_aqui", "")):
        print("  ✅ Telegram configurado")
    else:
        print("  ⚠️  Telegram NÃO configurado (alertas desabilitados)")
        print("     → Preencha TELEGRAM_TOKEN e TELEGRAM_CHAT_ID no .env")


def create_packages():
    """Cria __init__.py em todos os subpacotes."""
    packages = [
        "src",
        "src/camera",
        "src/ai",
        "src/rules",
        "src/alerts",
        "src/dashboard",
    ]
    for pkg in packages:
        init = ROOT / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text(f"# {pkg} package\n", encoding="utf-8")
    print("\n✅ Pacotes Python inicializados")


def install_requirements():
    print("\n📦 Instalando dependências (pode demorar na 1ª vez)...")
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("  ❌ requirements.txt não encontrado!")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        capture_output=False,
        text=True,
    )
    if result.returncode == 0:
        print("  ✅ Dependências instaladas!")
        return True
    else:
        print("  ❌ Falha na instalação. Verifique sua conexão.")
        return False


def test_imports():
    print("\n🔍 Verificando importações...")
    libs = [
        ("cv2",         "OpenCV"),
        ("ultralytics", "Ultralytics YOLOv8"),
        ("streamlit",   "Streamlit"),
        ("numpy",       "NumPy"),
        ("pandas",      "Pandas"),
        ("loguru",      "Loguru"),
        ("requests",    "Requests"),
        ("dotenv",      "python-dotenv"),
        ("PIL",         "Pillow"),
        ("sklearn",     "scikit-learn"),
    ]
    all_ok = True
    for mod, name in libs:
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            all_ok = False
    return all_ok


def test_camera():
    print("\n📷 Testando câmera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ Câmera OK | {w}x{h}")
                return True
            else:
                print("  ⚠️  Câmera abriu mas não leu frame")
        else:
            print("  ❌ Câmera não encontrada (índice 0)")
            print("     → Tente CAMERA_INDEX=1 no .env")
        cap.release()
    except Exception as e:
        print(f"  ❌ Erro: {e}")
    return False


def test_yolo():
    print("\n🤖 Testando YOLOv8 (pode baixar modelo ~6MB na 1ª vez)...")
    try:
        from ultralytics import YOLO
        import numpy as np
        model   = YOLO("yolov8n.pt")
        dummy   = (np.random.rand(480, 640, 3) * 255).astype("uint8")
        results = model.predict(dummy, verbose=False, conf=0.5)
        n_cls   = len(model.names)
        print(f"  ✅ YOLOv8n OK | {n_cls} classes COCO disponíveis")
        return True
    except Exception as e:
        print(f"  ❌ Erro YOLOv8: {e}")
        return False


def test_project_modules():
    print("\n📂 Testando módulos do projeto...")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    modules = [
        ("config",               "Config"),
        ("src.camera.capture",   "Camera Capture"),
        ("src.ai.detector",      "EPI Detector"),
        ("src.rules.ppe_rules",  "PPE Rules Engine"),
        ("src.alerts.logger",    "Violation Logger"),
        ("src.alerts.telegram",  "Telegram Alerter"),
    ]
    all_ok = True
    for mod, name in modules:
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    return all_ok


# ── Main ──────────────────────────────────────────────────────

def main():
    banner()
    check_python()
    create_packages()
    setup_env()

    ok = install_requirements()
    if not ok:
        print("\n❌ Corrija as dependências e execute novamente.")
        sys.exit(1)

    imp_ok  = test_imports()
    cam_ok  = test_camera()
    yolo_ok = test_yolo()
    mod_ok  = test_project_modules()

    print("\n" + "=" * 60)
    print("RESUMO DO SETUP")
    print("=" * 60)
    print(f"  Importações  : {'✅ OK' if imp_ok  else '❌ FALHA'}")
    print(f"  Câmera       : {'✅ OK' if cam_ok  else '⚠️  Não encontrada'}")
    print(f"  YOLOv8       : {'✅ OK' if yolo_ok else '❌ FALHA'}")
    print(f"  Módulos TCC  : {'✅ OK' if mod_ok  else '❌ FALHA'}")
    print("=" * 60)

    if imp_ok and yolo_ok and mod_ok:
        print("\n🚀 TUDO PRONTO! Como executar:\n")
        print("  1. Sistema principal (câmera + detecção):")
        print("       python main.py\n")
        print("  2. Dashboard (terminal separado):")
        print("       streamlit run src/dashboard/streamlit_app.py\n")
        print("  Teclas no main.py:")
        print("    Q / ESC → encerrar")
        print("    B       → benchmark de performance")
        print("    S       → salvar frame manualmente")
    else:
        print("\n⚠️  Resolva os erros acima antes de continuar.")


if __name__ == "__main__":
    main()
