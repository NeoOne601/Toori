#!/usr/bin/env python3
"""Health-check and invoke MLX-VLM from a stable local wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toori MLX-VLM wrapper")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", default="128")
    parser.add_argument("--temperature", default="0.0")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        __import__("mlx_vlm")
    except ImportError:
        emit(
            {
                "success": False,
                "message": "mlx-vlm is not installed in this Python environment. Install it with `python3.11 -m pip install mlx-vlm`.",
            },
            json_mode=args.json,
            stream=sys.stderr,
        )
        return 1

    if args.healthcheck:
        emit({"success": True, "message": "mlx-vlm available"}, json_mode=args.json)
        return 0

    if not args.model_path:
        emit({"success": False, "message": "model path is required"}, json_mode=args.json, stream=sys.stderr)
        return 1
    if not args.image_path:
        emit({"success": False, "message": "image path is required"}, json_mode=args.json, stream=sys.stderr)
        return 1

    command = [
        sys.executable,
        "-m",
        "mlx_vlm",
        "generate",
        "--model",
        args.model_path,
        "--image",
        args.image_path,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temp",
        str(args.temperature),
    ]
    started_at = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    latency_ms = (time.perf_counter() - started_at) * 1000
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "mlx_vlm.generate failed"
        emit(
            {
                "success": False,
                "message": message,
                "latency_ms": round(latency_ms, 3),
            },
            json_mode=args.json,
            stream=sys.stderr,
        )
        return completed.returncode
    emit(
        {
            "success": True,
            "message": "ok",
            "text": extract_answer(completed.stdout),
            "latency_ms": round(latency_ms, 3),
            "peak_memory_gb": extract_peak_memory_gb(completed.stdout),
        },
        json_mode=args.json,
    )
    return 0


def extract_answer(output: str) -> str:
    cleaned = output.strip()
    if "<|im_start|>assistant" in cleaned:
        cleaned = cleaned.rsplit("<|im_start|>assistant", 1)[-1]
    cleaned = cleaned.lstrip()
    if cleaned.startswith("=========="):
        cleaned = cleaned.splitlines()[-1]
    if "==========" in cleaned:
        cleaned = cleaned.split("==========", 1)[0]
    lines = [line.rstrip() for line in cleaned.splitlines()]
    meaningful = [
        line for line in lines
        if line.strip()
        and not line.startswith("Files:")
        and not line.startswith("Prompt:")
        and not line.startswith("Generation:")
        and not line.startswith("Peak memory:")
    ]
    return "\n".join(meaningful).strip() or output.strip()


def extract_peak_memory_gb(output: str) -> float | None:
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Peak memory:"):
            continue
        value = stripped.split(":", 1)[1].strip().split()[0]
        try:
            return float(value)
        except ValueError:
            return None
    return None


def emit(payload: dict, *, json_mode: bool, stream=sys.stdout) -> None:
    if json_mode:
        print(json.dumps(payload), file=stream)
        return
    if payload.get("success") and payload.get("text"):
        print(payload["text"], file=stream)
        return
    print(payload.get("message", ""), file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
