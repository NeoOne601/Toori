"""Lazy-loaded DINOv2 + SAM perception helpers.

The package keeps torch imports isolated to this directory and falls back to
deterministic numpy implementations when model weights or torch are unavailable.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .dinov2_encoder import DinoV2Embedding, Dinov2Encoder
from .sam_segmenter import SamSegmentation, SamSegmenter


DINOv2Encoder = Dinov2Encoder
SAMSegmenter = SamSegmenter


@dataclass(slots=True)
class MaskResult:
    bbox_pixels: dict[str, float]
    patch_indices: list[int]
    area_fraction: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_pixels": self.bbox_pixels,
            "patch_indices": self.patch_indices,
            "area_fraction": self.area_fraction,
            "confidence": self.confidence,
        }


class PerceptionPipeline:
    """Combined perception pipeline used by the runtime and immersive engine."""

    def __init__(self, device: str = "mps") -> None:
        self.encoder = DINOv2Encoder(device=device)
        self.segmenter = SAMSegmenter(device=device)

    def encode(self, frame: np.ndarray | Image.Image) -> tuple[np.ndarray, list[MaskResult]]:
        image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame).astype(np.uint8))
        embedding = self.encoder.encode(image)
        masks = self.segmenter.segment(image, timeout_s=0.08)
        return embedding.patch_tokens.astype(np.float32), self._mask_results(masks, image.size)

    def retrieval_embedding(self, frame: np.ndarray | Image.Image) -> np.ndarray:
        image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame).astype(np.uint8))
        return self.encoder.encode(image).pooled_embedding.astype(np.float32)

    def fallback_random_patches(self, n: int = 32) -> list[int]:
        rng = np.random.default_rng(20260327)
        total = 14 * 14
        count = min(max(n, 1), total)
        return sorted(int(index) for index in rng.choice(total, size=count, replace=False))

    def _mask_results(self, segmentation: SamSegmentation, image_size: tuple[int, int]) -> list[MaskResult]:
        width, height = image_size
        results: list[MaskResult] = []
        for index, mask in enumerate(np.asarray(segmentation.masks, dtype=bool)):
            area_fraction = float(mask.mean())
            if area_fraction <= 0.005:
                continue
            points = np.argwhere(mask)
            if points.size == 0:
                continue
            top, left = points.min(axis=0)
            bottom, right = points.max(axis=0)
            scale_x = width / mask.shape[1]
            scale_y = height / mask.shape[0]
            bbox_pixels = {
                "x": float(left * scale_x),
                "y": float(top * scale_y),
                "width": float(max((right - left + 1) * scale_x, 1.0)),
                "height": float(max((bottom - top + 1) * scale_y, 1.0)),
            }
            patch_indices = self._mask_to_patches(mask)
            results.append(
                MaskResult(
                    bbox_pixels=bbox_pixels,
                    patch_indices=patch_indices,
                    area_fraction=area_fraction,
                    confidence=float(segmentation.scores[index]) if index < len(segmentation.scores) else 0.0,
                )
            )
        return results

    def _mask_to_patches(self, mask: np.ndarray) -> list[int]:
        patch_rows, patch_cols = 14, 14
        row_step = mask.shape[0] / patch_rows
        col_step = mask.shape[1] / patch_cols
        patch_indices: list[int] = []
        for row in range(patch_rows):
            for col in range(patch_cols):
                top = int(round(row * row_step))
                bottom = int(round((row + 1) * row_step))
                left = int(round(col * col_step))
                right = int(round((col + 1) * col_step))
                if bottom <= top or right <= left:
                    continue
                if bool(mask[top:bottom, left:right].any()):
                    patch_indices.append((row * patch_cols) + col)
        return patch_indices


__all__ = [
    "DinoV2Embedding",
    "Dinov2Encoder",
    "DINOv2Encoder",
    "SamSegmentation",
    "SamSegmenter",
    "SAMSegmenter",
    "MaskResult",
    "PerceptionPipeline",
]
