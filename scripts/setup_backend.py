#!/usr/bin/env python3
"""Install runtime dependencies and launch the local Toori runtime."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_IMPORTS = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "prometheus_client",
    "httpx",
    "numpy",
    "PIL",
    "onnxruntime",
    "websockets",
    "weasyprint",
]

OPTIONAL_PERCEPTION_PACKAGES = [
    "torch>=2.1.0",
    "torchvision",
    "git+https://github.com/ChaoningZhang/MobileSAM",
]

def ensure_python_311() -> None:
    if sys.version_info[:2] == (3, 11):
        return
    python311 = shutil.which("python3.11")
    if not python311:
        print("python3.11 is not installed. Continuing with the active interpreter.")
        return
    print(f"Re-launching setup with {python311} so ONNX Runtime and perception extras can be installed on Apple Silicon.")
    os.execv(python311, [python311, __file__])

def _missing_base_imports() -> list[str]:
    missing = []
    for import_name in BASE_IMPORTS:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(import_name)
    return missing


def install_requirements(*, include_optional: bool = False) -> None:
    requirements_path = Path.cwd() / "requirements.txt"
    print(f"Installing base requirements from {requirements_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    if include_optional:
        print("Installing optional DINOv2/SAM perception extras")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *OPTIONAL_PERCEPTION_PACKAGES])

def main():
    print("=== Backend setup script ===")
    ensure_python_311()
    include_optional = os.environ.get("TOORI_INSTALL_PERCEPTION_EXTRAS") == "1"
    if _missing_base_imports():
        install_requirements(include_optional=include_optional)
    elif include_optional:
        install_requirements(include_optional=True)
    else:
        print("Base requirements already installed.")
        print("Skipping optional DINOv2/SAM installs. Set TOORI_INSTALL_PERCEPTION_EXTRAS=1 to install perception extras.")
    subprocess.check_call([sys.executable, str(Path.cwd() / "scripts" / "download_desktop_models.py")], cwd=Path.cwd())
    data_dir = Path.cwd() / ".toori"
    data_dir.mkdir(exist_ok=True)
    cmd = [
        "uvicorn",
        "cloud.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7777",
    ]
    print(f"Launching runtime with TOORI_DATA_DIR={data_dir}")
    subprocess.check_call(cmd, cwd=Path.cwd(), env={**os.environ, "TOORI_DATA_DIR": str(data_dir)})

if __name__ == "__main__":
    main()
