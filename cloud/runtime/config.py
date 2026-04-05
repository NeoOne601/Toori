from __future__ import annotations

import os
from pathlib import Path

from .models import ProviderConfig, RuntimeSettings, SmritiStorageConfig


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.getenv("TOORI_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.cwd() / ".toori").resolve()


def resolve_smriti_storage(
    settings: RuntimeSettings,
    base_data_dir: str,
) -> SmritiStorageConfig:
    """
    Returns fully resolved SmritiStorageConfig.
    Applies environment variable overrides last.
    Priority: settings > env vars > defaults
    """

    resolved = settings.smriti_storage.resolve_paths(base_data_dir)

    if env_dir := os.getenv("TOORI_SMRITI_DATA_DIR"):
        resolved = resolved.model_copy(update={"data_dir": env_dir})
    if env_frames := os.getenv("TOORI_SMRITI_FRAMES_DIR"):
        resolved = resolved.model_copy(update={"frames_dir": env_frames})
    if env_thumbs := os.getenv("TOORI_SMRITI_THUMBS_DIR"):
        resolved = resolved.model_copy(update={"thumbs_dir": env_thumbs})

    return resolved


def default_settings() -> RuntimeSettings:
    root = repo_root()
    default_onnx_path = os.getenv("TOORI_ONNX_MODEL") or str(root / "models" / "vision" / "mobilenetv2-12.onnx")
    default_onnx_labels = str(root / "models" / "vision" / "imagenet-simple-labels.json")
    default_mlx_command = os.getenv("TOORI_MLX_COMMAND") or f"python3.11 {root / 'scripts' / 'mlx_reasoner.py'}"
    return RuntimeSettings(
        public_url=os.getenv("TOORI_PUBLIC_URL", "https://github.com/NeoOne601/Toori"),
        primary_perception_provider="dinov2",
        fallback_order=["dinov2", "onnx", "basic", "cloud"],
        providers={
            "dinov2": ProviderConfig(
                name="dinov2",
                enabled=True,
                model="facebookresearch/dinov2:dinov2_vits14",
                metadata={
                    "device": os.getenv("TOORI_DINOV2_DEVICE", "mps"),
                    "sam_device": os.getenv("TOORI_SAM_DEVICE", os.getenv("TOORI_DINOV2_DEVICE", "mps")),
                    "pool_dim": 128,
                    "patch_grid": [14, 14],
                },
            ),
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
                enabled=False, # Gemma 4 on-device narrator (user-enabled)
                model_path=os.getenv("TOORI_MLX_MODEL_PATH", "/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit"),
                timeout_s=float(os.getenv("TOORI_MLX_TIMEOUT", "45")),
                metadata={"command": default_mlx_command, "keep_alive": "30m"},
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
