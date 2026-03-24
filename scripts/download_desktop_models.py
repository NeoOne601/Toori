#!/usr/bin/env python3
"""Download local desktop inference assets for Toori."""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from pathlib import Path

import certifi

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models" / "vision"

MODEL_URL = "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx?download=true"
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response:
        destination.write_bytes(response.read())
    print(f"Downloaded {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Toori desktop ONNX assets")
    parser.add_argument("--force", action="store_true", help="Re-download even if the files already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = MODELS_DIR / "mobilenetv2-12.onnx"
    labels_path = MODELS_DIR / "imagenet-simple-labels.json"

    assets = [
        (MODEL_URL, model_path),
        (LABELS_URL, labels_path),
    ]
    for url, destination in assets:
        if destination.exists() and not args.force:
            print(f"Keeping existing {destination}")
            continue
        download(url, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
