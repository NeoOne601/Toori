from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

_RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC


@dataclass(slots=True)
class DinoV2Embedding:
    pooled_embedding: np.ndarray
    patch_tokens: np.ndarray
    patch_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding(self) -> np.ndarray:
        return self.pooled_embedding

    @property
    def mask(self) -> np.ndarray:
        return self.patch_mask


def _coerce_image(image: Image.Image | np.ndarray | object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


@lru_cache(maxsize=None)
def _projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    return (matrix / np.sqrt(max(in_dim, 1))).astype(np.float32)


class Dinov2Encoder:
    """Lazy DINOv2 wrapper with a deterministic numpy fallback."""

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vits14",
        device: str = "cpu",
        model_path: str | None = None,
        allow_download: bool = False,
        timeout_s: float = 10.0,
        grid_size: tuple[int, int] = (14, 14),
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_path = model_path
        self.allow_download = allow_download
        self.timeout_s = timeout_s
        self.grid_size = grid_size
        self._backend: str = "fallback"
        self._model: Any = None

    def encode(self, image: Image.Image | np.ndarray | object) -> DinoV2Embedding:
        image = _coerce_image(image)
        try:
            self._ensure_model()
            if self._backend == "torch":
                return self._encode_with_torch(image)
        except Exception:
            self._backend = "fallback"
            self._model = None
        return self._encode_with_fallback(image)

    def _ensure_model(self) -> None:
        if self._model is not None or self._backend == "fallback":
            return
        try:
            import importlib

            torch = importlib.import_module("torch")
        except Exception:
            self._backend = "fallback"
            return

        model = None
        if self.model_path and Path(self.model_path).exists():
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model = state if hasattr(state, "forward_features") else None
        elif self.allow_download:
            try:
                model = torch.hub.load("facebookresearch/dinov2", self.model_name, pretrained=True)
            except Exception:
                model = None

        if model is None:
            self._backend = "fallback"
            return

        model.eval()
        model.to(self.device)
        self._model = model
        self._backend = "torch"

    def _encode_with_torch(self, image: Image.Image) -> DinoV2Embedding:
        torch = __import__("torch")
        rgb = np.asarray(image.convert("RGB").resize((224, 224), _RESAMPLE), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None, ...]).to(self.device)
        with torch.no_grad():
            output = self._model.forward_features(tensor)
        tokens = self._extract_patch_tokens(output)
        return self._finalize_encoding(image, tokens, backend="torch")

    def _extract_patch_tokens(self, output: object) -> np.ndarray:
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "x"):
                value = output.get(key)
                if value is not None:
                    return np.asarray(value, dtype=np.float32)
        return np.asarray(output, dtype=np.float32)

    def _encode_with_fallback(self, image: Image.Image) -> DinoV2Embedding:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), _RESAMPLE), dtype=np.float32) / 255.0
        tokens, patch_mask = self._fallback_patch_tokens(rgb)
        return self._finalize_encoding(image, tokens, patch_mask=patch_mask, backend="fallback")

    def _finalize_encoding(
        self,
        image: Image.Image,
        tokens: np.ndarray,
        *,
        patch_mask: np.ndarray | None = None,
        backend: str,
    ) -> DinoV2Embedding:
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.ndim == 3:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        tokens = self._ensure_token_shape(tokens)
        if patch_mask is None:
            patch_mask = self._derive_mask(tokens)
        pooled = tokens.mean(axis=0)
        pooled_embedding = _l2_normalize(self._project_vector(pooled, 128, seed=20240327))
        metadata = {
            "backend": backend,
            "model_name": self.model_name,
            "device": self.device,
            "grid_size": self.grid_size,
            "image_size": tuple(image.size),
        }
        return DinoV2Embedding(
            pooled_embedding=pooled_embedding,
            patch_tokens=tokens.astype(np.float32),
            patch_mask=patch_mask.astype(bool),
            metadata=metadata,
        )

    def _ensure_token_shape(self, tokens: np.ndarray) -> np.ndarray:
        grid_h, grid_w = self.grid_size
        expected_patches = grid_h * grid_w
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.ndim != 2:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        if tokens.shape[1] != 384:
            tokens = self._project_matrix(tokens, 384, seed=20240328)
        if tokens.shape[0] < expected_patches:
            pad = np.repeat(tokens[-1:, :], expected_patches - tokens.shape[0], axis=0) if tokens.size else np.zeros((expected_patches, 384), dtype=np.float32)
            tokens = np.concatenate([tokens, pad], axis=0)
        elif tokens.shape[0] > expected_patches:
            tokens = tokens[:expected_patches, :]
        return tokens.astype(np.float32)

    def _fallback_patch_tokens(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid_h, grid_w = self.grid_size
        patch_h = rgb.shape[0] // grid_h
        patch_w = rgb.shape[1] // grid_w
        patches = rgb.reshape(grid_h, patch_h, grid_w, patch_w, 3).transpose(0, 2, 1, 3, 4)
        gray = rgb.mean(axis=2).reshape(grid_h, patch_h, grid_w, patch_w).transpose(0, 2, 1, 3)
        edge_x = np.abs(np.diff(gray, axis=2, prepend=gray[:, :, :1, :]))
        edge_y = np.abs(np.diff(gray, axis=3, prepend=gray[:, :, :, :1]))
        edge = np.sqrt(edge_x**2 + edge_y**2)
        mean_rgb = patches.mean(axis=(2, 3))
        std_rgb = patches.std(axis=(2, 3))
        min_rgb = patches.min(axis=(2, 3))
        max_rgb = patches.max(axis=(2, 3))
        gray_mean = gray.mean(axis=(2, 3))
        gray_std = gray.std(axis=(2, 3))
        edge_mean = edge.mean(axis=(2, 3))
        saturation = (max_rgb - min_rgb).mean(axis=-1)
        center_x = np.linspace(-1.0, 1.0, grid_w, dtype=np.float32)[None, :]
        center_y = np.linspace(-1.0, 1.0, grid_h, dtype=np.float32)[:, None]
        radius = np.sqrt(center_x**2 + center_y**2)
        luma_delta = np.abs(gray_mean - float(gray_mean.mean()))
        patch_energy = np.sqrt((mean_rgb**2).sum(axis=-1))
        base_features = np.stack(
            [
                mean_rgb[..., 0],
                mean_rgb[..., 1],
                mean_rgb[..., 2],
                std_rgb[..., 0],
                std_rgb[..., 1],
                std_rgb[..., 2],
                min_rgb.mean(axis=-1),
                max_rgb.mean(axis=-1),
                gray_mean,
                gray_std,
                edge_mean,
                saturation,
                center_x + np.zeros_like(gray_mean),
                center_y + np.zeros_like(gray_mean),
                radius,
                luma_delta,
                patch_energy,
            ],
            axis=-1,
        ).astype(np.float32)
        features = np.concatenate([base_features, base_features**2], axis=-1)
        tokens = self._project_matrix(features.reshape(-1, features.shape[-1]), 384, seed=20240329)
        patch_scores = edge_mean + (0.5 * gray_std) + (0.25 * luma_delta)
        patch_mask = patch_scores >= float(np.median(patch_scores))
        if not bool(np.any(patch_mask)):
            patch_mask.flat[int(np.argmax(patch_scores))] = True
        return tokens.astype(np.float32), patch_mask.astype(bool)

    def _derive_mask(self, tokens: np.ndarray) -> np.ndarray:
        grid_h, grid_w = self.grid_size
        norms = np.linalg.norm(tokens.reshape(grid_h * grid_w, -1), axis=1).reshape(grid_h, grid_w)
        mask = norms >= float(np.median(norms))
        if not bool(np.any(mask)):
            mask.flat[int(np.argmax(norms))] = True
        return mask

    def _project_vector(self, vector: np.ndarray, out_dim: int, *, seed: int) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        projection = _projection_matrix(vector.shape[0], out_dim, seed)
        return vector @ projection

    def _project_matrix(self, matrix: np.ndarray, out_dim: int, *, seed: int) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        projection = _projection_matrix(matrix.shape[-1], out_dim, seed)
        return matrix @ projection
