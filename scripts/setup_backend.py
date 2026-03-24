#!/usr/bin/env python3
"""Install runtime dependencies and launch the local Toori runtime."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("prometheus_client", "prometheus_client"),
    ("httpx", "httpx"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("onnxruntime", "onnxruntime"),
    ("websockets", "websockets"),
]

def ensure_python_311() -> None:
    if sys.version_info[:2] == (3, 11):
        return
    python311 = shutil.which("python3.11")
    if not python311:
        print("python3.11 is not installed. Continuing with the active interpreter.")
        return
    print(f"Re-launching setup with {python311} so ONNX Runtime can be installed on Apple Silicon.")
    os.execv(python311, [python311, __file__])

def check_python_packages():
    missing = []
    for pkg, import_name in REQUIREMENTS:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing Python packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required Python packages are present.")

def main():
    print("=== Backend setup script ===")
    ensure_python_311()
    check_python_packages()
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
