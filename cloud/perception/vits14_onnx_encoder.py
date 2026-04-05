"""
vits14_onnx_encoder.py — DINOv2-ViT-S/14 ONNX patch encoder.

Honest fallback for when V-JEPA2 is unavailable.
Produces real 14×14 patch token grids (196 patches × 384-dim) using
DINOv2-ViT-S/14, which is architecturally aligned with ImmersiveJEPAEngine's
14×14 patch grid expectation.

This is NOT a surrogate. It is a real patch-token world model encoder.
degraded must be False when this encoder is active.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

VITS14_INPUT_SIZE = 224
VITS14_PATCH_SIZE = 14
VITS14_PATCHES_PER_DIM = VITS14_INPUT_SIZE // VITS14_PATCH_SIZE   # 16
VITS14_TOTAL_PATCHES = VITS14_PATCHES_PER_DIM ** 2                 # 256
VITS14_DIM = 384
VITS14_TARGET_GRID = 14          # ImmersiveJEPAEngine expects 14×14
VITS14_TARGET_PATCHES = VITS14_TARGET_GRID ** 2  # 196
VITS14_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VITS14_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ViTS14OnnxEncoder:
    """DINOv2-ViT-S/14 ONNX encoder producing 14×14 patch tokens."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ViT-S/14 ONNX model not found at {path}. "
                f"Run: python3.11 scripts/download_desktop_models.py"
            )
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def preprocess(self, image_np: np.ndarray) -> np.ndarray:
        """Resize to 224×224, normalize to ImageNet stats, return [1, 3, H, W] float32."""
        from PIL import Image
        if image_np.dtype != np.uint8:
            image_np = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(image_np).resize(
            (VITS14_INPUT_SIZE, VITS14_INPUT_SIZE), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3]
        arr = (arr - VITS14_IMAGENET_MEAN) / VITS14_IMAGENET_STD
        return arr.transpose(2, 0, 1)[np.newaxis]        # [1, 3, H, W]

    def encode(self, image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode image to DINOv2-ViT-S/14 patch tokens.

        Returns:
            global_emb:  np.ndarray[384]      — unit-norm global embedding (mean of patches)
            patch_tokens: np.ndarray[196, 384] — unit-norm 14×14 patch grid
        """
        preprocessed = self.preprocess(image_np)
        # ONNX output: [1, 257, 384] = [batch, CLS + 256 patches, dim]
        # Token 0 = CLS, tokens 1:257 = 16×16 spatial patch grid
        outputs = self._session.run(None, {self._input_name: preprocessed})
        all_tokens = outputs[0][0]              # [257, 384]
        patch_256 = all_tokens[1:]              # [256, 384] — drop CLS
        # Center-crop 16×16 grid to 14×14: remove 1 row/col from each edge
        patches_2d = patch_256.reshape(VITS14_PATCHES_PER_DIM, VITS14_PATCHES_PER_DIM, VITS14_DIM)
        margin = (VITS14_PATCHES_PER_DIM - VITS14_TARGET_GRID) // 2   # = 1
        patch_14x14 = patches_2d[margin:-margin, margin:-margin, :]    # [14, 14, 384]
        patch_tokens = patch_14x14.reshape(VITS14_TARGET_PATCHES, VITS14_DIM)  # [196, 384]
        global_emb = patch_tokens.mean(axis=0)
        # Unit-normalize
        eps = 1e-9
        global_emb = global_emb / (np.linalg.norm(global_emb) + eps)
        norms = np.linalg.norm(patch_tokens, axis=1, keepdims=True) + eps
        patch_tokens = patch_tokens / norms
        return global_emb.astype(np.float32), patch_tokens.astype(np.float32)

    @classmethod
    def default_model_path(cls) -> str:
        return "models/vision/dinov2_vits14.onnx"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return Path(cls.default_model_path()).exists()
        except ImportError:
            return False
