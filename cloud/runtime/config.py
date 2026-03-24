from __future__ import annotations

import os
from pathlib import Path

from .models import ProviderConfig, RuntimeSettings


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.getenv("TOORI_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".toori").resolve()


def default_settings() -> RuntimeSettings:
    root = repo_root()
    default_onnx_path = os.getenv("TOORI_ONNX_MODEL") or str(root / "models" / "vision" / "mobilenetv2-12.onnx")
    default_onnx_labels = str(root / "models" / "vision" / "imagenet-simple-labels.json")
    default_mlx_command = os.getenv("TOORI_MLX_COMMAND") or f"python3.11 {root / 'scripts' / 'mlx_reasoner.py'}"
    return RuntimeSettings(
        providers={
            "onnx": ProviderConfig(
                name="onnx",
                enabled=True,
                model_path=default_onnx_path,
                metadata={
                    "input_size": 224,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "labels_path": default_onnx_labels,
                },
            ),
            "basic": ProviderConfig(name="basic", enabled=True),
            "coreml": ProviderConfig(
                name="coreml",
                enabled=False,
                model_path=os.getenv("TOORI_COREML_MODEL"),
            ),
            "tflite": ProviderConfig(
                name="tflite",
                enabled=False,
                model_path=os.getenv("TOORI_TFLITE_MODEL"),
            ),
            "ollama": ProviderConfig(
                name="ollama",
                enabled=False,
                base_url=os.getenv("TOORI_OLLAMA_HOST", "http://127.0.0.1:11434"),
                model=os.getenv("TOORI_OLLAMA_MODEL", "gemma3:4b"),
                timeout_s=float(os.getenv("TOORI_OLLAMA_TIMEOUT", "150")),
                metadata={"keep_alive": os.getenv("TOORI_OLLAMA_KEEP_ALIVE", "15m")},
            ),
            "mlx": ProviderConfig(
                name="mlx",
                enabled=False,
                model_path=os.getenv("TOORI_MLX_MODEL_PATH", "mlx-community/Qwen2-VL-2B-Instruct-4bit"),
                timeout_s=float(os.getenv("TOORI_MLX_TIMEOUT", "150")),
                metadata={"command": default_mlx_command},
            ),
            "cloud": ProviderConfig(
                name="cloud",
                enabled=True,
                base_url=os.getenv("TOORI_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("TOORI_OPENAI_MODEL", "gpt-4.1-mini"),
                api_key=os.getenv("TOORI_OPENAI_API_KEY"),
                timeout_s=float(os.getenv("TOORI_OPENAI_TIMEOUT", "20")),
            ),
            "local": ProviderConfig(name="local", enabled=True),
        }
    )
