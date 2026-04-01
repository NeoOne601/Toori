"""
Temporal Parallax Depth Separator (TPDS).

Derives per-patch depth strata using only V-JEPA temporal energy signals.
No depth sensor. No stereo camera. No generative model. Pure JEPA signal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from cloud.runtime.error_types import SmritiPipelineError
from cloud.runtime.observability import get_logger, with_fallback

log = get_logger("tpds")

FOREGROUND_THRESHOLD: float = 1.5
BACKGROUND_THRESHOLD: float = 0.4
EMA_ALPHA: float = 0.30
SMOOTHING_KERNEL: np.ndarray = np.ones((3, 3), dtype=np.float32) / 9.0
EPSILON: float = 1e-6


@dataclass
class DepthStrataMap:
    depth_proxy: np.ndarray
    foreground_mask: np.ndarray
    midground_mask: np.ndarray
    background_mask: np.ndarray
    confidence: float
    strata_entropy: float

    def to_dict(self) -> dict:
        return {
            "depth_proxy": self.depth_proxy.tolist(),
            "foreground_mask": self.foreground_mask.tolist(),
            "midground_mask": self.midground_mask.tolist(),
            "background_mask": self.background_mask.tolist(),
            "confidence": round(float(self.confidence), 4),
            "strata_entropy": round(float(self.strata_entropy), 4),
        }

    @staticmethod
    def cold_start(grid: tuple[int, int] = (14, 14)) -> "DepthStrataMap":
        h, w = grid
        return DepthStrataMap(
            depth_proxy=np.full((h, w), 0.7, dtype=np.float32),
            foreground_mask=np.zeros((h, w), dtype=bool),
            midground_mask=np.ones((h, w), dtype=bool),
            background_mask=np.zeros((h, w), dtype=bool),
            confidence=0.0,
            strata_entropy=0.0,
        )


class TemporalParallaxDepthSeparator:
    def __init__(
        self,
        foreground_threshold: float = FOREGROUND_THRESHOLD,
        background_threshold: float = BACKGROUND_THRESHOLD,
        ema_alpha: float = EMA_ALPHA,
        temporal_buffer_size: int = 8,
        grid: tuple[int, int] = (14, 14),
    ) -> None:
        self._fg_threshold = foreground_threshold
        self._bg_threshold = background_threshold
        self._ema_alpha = ema_alpha
        self._grid = grid
        h, w = grid
        self._ema_proxy: np.ndarray = np.full((h, w), 0.7, dtype=np.float32)
        self._energy_buffer: deque[np.ndarray] = deque(maxlen=temporal_buffer_size)
        self._tick_count: int = 0

    @with_fallback(fallback_value=DepthStrataMap.cold_start(), log_component="tpds")
    def update(
        self,
        energy_map_current: np.ndarray,
        energy_map_previous: np.ndarray,
        patch_tokens: np.ndarray,
        *,
        prediction_error: float | None = None,
    ) -> DepthStrataMap:
        current = np.asarray(energy_map_current, dtype=np.float32)
        previous = np.asarray(energy_map_previous, dtype=np.float32)
        _ = patch_tokens

        if current.shape != self._grid or previous.shape != self._grid:
            raise SmritiPipelineError("tpds", f"Expected energy maps shaped {self._grid}")

        self._energy_buffer.append(current.copy())
        self._tick_count += 1

        if self._tick_count < 3:
            return DepthStrataMap.cold_start(self._grid)

        temporal_delta = np.abs(current - previous).astype(np.float32)
        spatial_norm = np.minimum(current, previous) + EPSILON
        raw_proxy = temporal_delta / spatial_norm
        raw_proxy = _box_filter(raw_proxy, SMOOTHING_KERNEL)

        # Sprint 6: boost foreground sensitivity when world model prediction error is high
        if prediction_error is not None and prediction_error > 0.3:
            boost = 1.0 + min(float(prediction_error), 1.0) * 0.5
            raw_proxy = raw_proxy * boost

        self._ema_proxy = ((1.0 - self._ema_alpha) * self._ema_proxy) + (self._ema_alpha * raw_proxy)
        proxy = self._ema_proxy.copy()

        foreground_mask = proxy > self._fg_threshold
        background_mask = (proxy < self._bg_threshold) & ~foreground_mask
        midground_mask = ~foreground_mask & ~background_mask

        total = foreground_mask.size
        p_fg = foreground_mask.sum() / total
        p_bg = background_mask.sum() / total
        p_mid = midground_mask.sum() / total
        entropy = _shannon_entropy(np.array([p_fg, p_bg, p_mid], dtype=np.float32))
        confidence = float(min(entropy / 1.585, 1.0))

        log.debug(
            "tpds_update",
            foreground_pct=round(float(p_fg), 3),
            background_pct=round(float(p_bg), 3),
            confidence=round(confidence, 3),
        )

        return DepthStrataMap(
            depth_proxy=proxy,
            foreground_mask=foreground_mask,
            midground_mask=midground_mask,
            background_mask=background_mask,
            confidence=confidence,
            strata_entropy=float(entropy),
        )

    def reset(self) -> None:
        h, w = self._grid
        self._ema_proxy = np.full((h, w), 0.7, dtype=np.float32)
        self._energy_buffer.clear()
        self._tick_count = 0
        log.debug("tpds_reset")

    def get_depth_proxy_map(self) -> np.ndarray:
        return self._ema_proxy.copy()


def _box_filter(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(h):
        for j in range(w):
            out[i, j] = (padded[i : i + kh, j : j + kw] * kernel).sum()
    return out.astype(np.float32)


def _shannon_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
