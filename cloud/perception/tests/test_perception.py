from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from cloud.perception import Dinov2Encoder, SamSegmenter


def _sample_image() -> Image.Image:
    grid_x = np.arange(224, dtype=np.uint8)[None, :]
    grid_y = np.arange(224, dtype=np.uint8)[:, None]
    red = np.broadcast_to((grid_x * 3) % 255, (224, 224))
    green = np.broadcast_to((grid_y * 2) % 255, (224, 224))
    blue = (grid_x + grid_y) % 255
    array = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_dinov2_fallback_shapes_and_determinism():
    encoder = Dinov2Encoder()
    image = _sample_image()

    first = encoder.encode(image)
    second = encoder.encode(image)

    assert first.pooled_embedding.shape == (128,)
    assert first.patch_tokens.shape == (196, 384)
    assert first.patch_mask.shape == (14, 14)
    assert first.pooled_embedding.dtype == np.float32
    assert first.patch_tokens.dtype == np.float32
    assert first.patch_mask.dtype == bool
    assert np.all(np.isfinite(first.pooled_embedding))
    assert np.all(np.isfinite(first.patch_tokens))
    assert np.allclose(first.pooled_embedding, second.pooled_embedding)
    assert np.allclose(first.patch_tokens, second.patch_tokens)
    assert np.array_equal(first.patch_mask, second.patch_mask)


def test_sam_fallback_shapes():
    segmenter = SamSegmenter()
    result = segmenter.segment(_sample_image(), timeout_s=0.5)

    assert result.masks.ndim == 3
    assert result.masks.shape[1:] == (224, 224)
    assert result.scores.shape == (result.masks.shape[0],)
    assert result.masks.dtype == bool
    assert np.all(result.scores >= 0.0)


def test_sam_timeout_falls_back(monkeypatch):
    segmenter = SamSegmenter(timeout_s=0.01)

    def slow_impl(image):
        time.sleep(0.05)
        return segmenter._fallback_segment(image, reason="slow")

    monkeypatch.setattr(segmenter, "_segment_impl", slow_impl)

    result = segmenter.segment(_sample_image(), timeout_s=0.01)
    assert result.metadata["backend"] == "fallback"
    assert result.metadata["reason"] == "timeout"
    assert result.masks.shape[1:] == (224, 224)


def test_engine_source_does_not_import_torch():
    engine_path = Path("/Users/macuser/toori/cloud/jepa_service/engine.py")
    source = engine_path.read_text()
    assert "import torch" not in source
