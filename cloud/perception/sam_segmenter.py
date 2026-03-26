from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageFilter

_RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC


@dataclass(slots=True)
class SamSegmentation:
    masks: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return self.masks[0] if self.masks.size else np.zeros((0, 0), dtype=bool)


def _coerce_image(image: Image.Image | np.ndarray | object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    scale = max(maximum - minimum, 1e-6)
    return ((values - minimum) / scale).astype(np.float32)


class SamSegmenter:
    """Lazy SAM wrapper with deterministic fallbacks and timeout support."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        device: str = "cpu",
        allow_download: bool = False,
        timeout_s: float = 3.0,
        max_masks: int = 3,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.allow_download = allow_download
        self.timeout_s = timeout_s
        self.max_masks = max_masks
        self._backend: str = "fallback"
        self._model: Any = None

    def segment(self, image: Image.Image | np.ndarray | object, *, timeout_s: float | None = None) -> SamSegmentation:
        image = _coerce_image(image)
        effective_timeout = self.timeout_s if timeout_s is None else timeout_s
        if effective_timeout and effective_timeout > 0:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._segment_impl, image)
            try:
                return future.result(timeout=effective_timeout)
            except TimeoutError:
                future.cancel()
                return self._fallback_segment(image, reason="timeout")
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return self._segment_impl(image)

    def _segment_impl(self, image: Image.Image) -> SamSegmentation:
        try:
            self._ensure_model()
            if self._backend == "torch":
                return self._segment_with_torch(image)
        except Exception:
            self._backend = "fallback"
            self._model = None
        return self._fallback_segment(image, reason="deterministic fallback")

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
            model = torch.load(self.model_path, map_location=self.device)
        elif self.allow_download:
            try:
                model = torch.hub.load("facebookresearch/segment-anything", "sam_vit_b")
            except Exception:
                model = None

        if model is None:
            self._backend = "fallback"
            return

        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(self.device)
        self._model = model
        self._backend = "torch"

    def _segment_with_torch(self, image: Image.Image) -> SamSegmentation:
        torch = __import__("torch")
        rgb = np.asarray(image.convert("RGB").resize((224, 224), _RESAMPLE), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None, ...]).to(self.device)
        with torch.no_grad():
            masks = self._run_torch_forward(tensor)
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        scores = self._score_masks(rgb, masks)
        return SamSegmentation(
            masks=masks,
            scores=scores,
            metadata={
                "backend": "torch",
                "device": self.device,
                "image_size": tuple(image.size),
                "num_masks": int(masks.shape[0]),
            },
        )

    def _run_torch_forward(self, tensor: Any) -> np.ndarray:
        if hasattr(self._model, "predict_masks"):
            return self._model.predict_masks(tensor)
        if hasattr(self._model, "forward"):
            return self._model(tensor)
        raise RuntimeError("Loaded SAM model does not expose a usable forward method")

    def _fallback_segment(self, image: Image.Image, *, reason: str) -> SamSegmentation:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), _RESAMPLE), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        edges = np.asarray(image.convert("L").resize((224, 224), _RESAMPLE).filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        center_x = np.linspace(-1.0, 1.0, gray.shape[1], dtype=np.float32)[None, :]
        center_y = np.linspace(-1.0, 1.0, gray.shape[0], dtype=np.float32)[:, None]
        center_prior = 1.0 - np.sqrt(center_x**2 + center_y**2) / np.sqrt(2.0)
        saliency = _normalize((0.45 * _normalize(gray)) + (0.35 * _normalize(edges)) + (0.2 * _normalize(center_prior)))
        thresholds = [0.78, 0.64, 0.52]
        masks = []
        scores = []
        for threshold in thresholds[: self.max_masks]:
            mask = saliency >= threshold
            if not bool(np.any(mask)):
                mask = saliency >= float(np.quantile(saliency, threshold))
            if not bool(np.any(mask)):
                mask = np.zeros_like(saliency, dtype=bool)
                mask[np.unravel_index(int(np.argmax(saliency)), saliency.shape)] = True
            masks.append(mask)
            scores.append(self._mask_score(saliency, mask))
        return SamSegmentation(
            masks=np.asarray(masks, dtype=bool),
            scores=np.asarray(scores, dtype=np.float32),
            metadata={
                "backend": "fallback",
                "reason": reason,
                "image_size": tuple(image.size),
                "num_masks": len(masks),
            },
        )

    def _mask_score(self, saliency: np.ndarray, mask: np.ndarray) -> float:
        coverage = float(mask.mean())
        if coverage <= 0.0:
            return 0.0
        return float((saliency[mask].mean() * 0.7) + (coverage * 0.3))

    def _score_masks(self, rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        gray = rgb.mean(axis=2)
        scores = []
        for mask in masks:
            if not bool(np.any(mask)):
                scores.append(0.0)
                continue
            scores.append(float(gray[mask].mean()))
        return np.asarray(scores, dtype=np.float32)
