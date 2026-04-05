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

VITS14_ONNX_URL = "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx"
VITS14_ONNX_PATH = MODELS_DIR / "dinov2_vits14.onnx"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as response:
        destination.write_bytes(response.read())
    print(f"Downloaded {destination}")


def download_vits14_onnx(*, force: bool = False) -> bool:
    """Download DINOv2-ViT-S/14 ONNX (~85MB). Returns True on success."""
    if VITS14_ONNX_PATH.exists() and not force:
        size_mb = VITS14_ONNX_PATH.stat().st_size / (1024 * 1024)
        if size_mb > 70:
            print(f"  ViT-S/14 ONNX already present ({size_mb:.0f}MB): {VITS14_ONNX_PATH}")
            return True
    VITS14_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading DINOv2-ViT-S/14 ONNX from HuggingFace (~85MB)...")
    for attempt in range(3):
        try:
            download(VITS14_ONNX_URL, VITS14_ONNX_PATH)
            size_mb = VITS14_ONNX_PATH.stat().st_size / (1024 * 1024)
            if size_mb > 70:
                print(f"  ViT-S/14 ONNX downloaded ({size_mb:.0f}MB)")
                return True
            print(f"  WARNING: ViT-S/14 ONNX too small ({size_mb:.1f}MB), retrying...")
            VITS14_ONNX_PATH.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  WARNING: ViT-S/14 download attempt {attempt + 1}/3 failed: {exc}")
    print("  WARNING: DINOv2-ViT-S/14 ONNX download failed after 3 attempts (non-fatal)")
    return False


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

    # DINOv2-ViT-S/14 ONNX — honest V-JEPA2 fallback encoder
    download_vits14_onnx(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
