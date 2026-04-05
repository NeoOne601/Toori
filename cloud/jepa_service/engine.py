"""JEPA engines for compatibility and immersive proof-surface use.

`JEPAEngine` preserves the pre-existing vector-oriented API used by legacy
tests and integration points. `ImmersiveJEPAEngine` is the DINOv2/SAM-backed
single-session engine used by the living-lens runtime path.

CONTRIBUTION SURFACE: `ImmersiveJEPAEngine._predict_vector()` intentionally
uses a simple 4-layer MLP. Community PRs are welcome for transformer,
recurrent, or attention-based predictors as long as `tick(frame) -> JEPATick`
and `forecast_last_state(k)` remain stable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter, time
from typing import Any, NamedTuple, Optional
from uuid import uuid4

import numpy as np

from cloud.perception import MaskResult, PerceptionPipeline
from cloud.perception.vjepa2_encoder import _is_test_environment, get_vjepa2_encoder
from cloud.jepa_service.confidence_gate import EpistemicConfidenceGate
from cloud.runtime.models import BoundingBox, EntityTrack, JEPATick
from cloud.jepa_service.anchor_graph import SemanticAnchorGraph
from cloud.jepa_service.depth_separator import TemporalParallaxDepthSeparator, DepthStrataMap
from cloud.jepa_service.world_model_alignment import CrossModalWorldModelAligner
from cloud.runtime.setu2 import Setu2Bridge


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear_forward(x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
    return x @ layer["weight"] + layer["bias"]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
    return float(np.dot(left, right) / denom)


def _xavier_uniform(in_dim: int, out_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    limit = np.sqrt(6.0 / max(in_dim + out_dim, 1))
    return {
        "weight": rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32),
        "bias": np.zeros(out_dim, dtype=np.float32),
    }


class TickResult(NamedTuple):
    energy_map: np.ndarray
    mean_energy: float
    threshold: float
    should_talk: bool
    talker_event: Optional[str]
    prediction_residual: np.ndarray
    sigreg_loss: float


class JEPAEngine:
    """Compatibility JEPA engine that preserves the legacy vector-first API."""

    def __init__(
        self,
        embedding_dim: int = 128,
        predictor_hidden: int = 512,
        predictor_layers: int = 4,
        patch_grid: tuple[int, int] = (7, 7),
        ema_tau: float = 0.996,
        energy_ema_alpha: float = 0.05,
        initial_threshold: float = 0.15,
        min_threshold: float = 0.08,
        sigreg_buffer_size: int = 64,
        sigreg_lambda_var: float = 1.0,
        sigreg_lambda_cov: float = 0.04,
        sigreg_gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._patch_grid = patch_grid
        self._ema_tau = ema_tau
        self._energy_ema_alpha = energy_ema_alpha
        self._min_threshold = min_threshold
        self._lambda_var = sigreg_lambda_var
        self._lambda_cov = sigreg_lambda_cov
        self._gamma = sigreg_gamma
        self._rng = np.random.default_rng(seed)
        self._context_proj = _xavier_uniform(embedding_dim, embedding_dim, self._rng)
        self._target_proj = {key: value.copy() for key, value in self._context_proj.items()}
        self._predictor_layers: list[dict[str, np.ndarray]] = []
        input_dim = embedding_dim
        for _ in range(predictor_layers - 1):
            self._predictor_layers.append(_xavier_uniform(input_dim, predictor_hidden, self._rng))
            input_dim = predictor_hidden
        self._predictor_layers.append(_xavier_uniform(input_dim, embedding_dim, self._rng))
        self._energy_ema = initial_threshold
        self._threshold = initial_threshold
        self._prev_target_embedding: Optional[np.ndarray] = None
        self._embedding_buffer: deque[np.ndarray] = deque(maxlen=sigreg_buffer_size)
        self._last_energy_map = np.zeros(patch_grid, dtype=np.float32)
        self._tick_count = 0

    def tick(
        self,
        frame_embedding: np.ndarray,
        patch_embeddings: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> TickResult:
        frame_embedding = np.asarray(frame_embedding, dtype=np.float32).ravel()
        assert frame_embedding.shape == (self._embedding_dim,), (
            f"Expected ({self._embedding_dim},), got {frame_embedding.shape}"
        )

        grid_h, grid_w = self._patch_grid
        num_patches = grid_h * grid_w
        if patch_embeddings is None:
            patch_embeddings = self._tile_with_noise(frame_embedding, num_patches)
        else:
            patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
            if patch_embeddings.shape[0] != num_patches:
                patch_embeddings = self._tile_with_noise(frame_embedding, num_patches)

        self._update_ema()
        context_patches = np.array([_linear_forward(patch, self._context_proj) for patch in patch_embeddings])
        target_patches = np.array([_linear_forward(patch, self._target_proj) for patch in patch_embeddings])
        predicted_patches = self._predict_batch(context_patches)
        residuals = target_patches - predicted_patches
        per_patch_energy = np.sum(residuals ** 2, axis=1)
        energy_map = per_patch_energy.reshape(grid_h, grid_w)
        mean_energy = float(np.mean(energy_map))
        self._update_threshold(mean_energy)
        should_talk = mean_energy > self._threshold
        context_global = _linear_forward(frame_embedding, self._context_proj)
        target_global = _linear_forward(frame_embedding, self._target_proj)
        predicted_global = self._predict(context_global)
        prediction_residual = target_global - predicted_global
        self._embedding_buffer.append(context_global)
        sigreg_loss = self._compute_sigreg(context_global, target_global)
        self._last_energy_map = energy_map.copy()
        self._prev_target_embedding = target_global.copy()
        self._tick_count += 1
        return TickResult(
            energy_map=energy_map,
            mean_energy=mean_energy,
            threshold=self._threshold,
            should_talk=should_talk,
            talker_event=None,
            prediction_residual=prediction_residual,
            sigreg_loss=sigreg_loss,
        )

    def get_energy_map(self) -> np.ndarray:
        return self._last_energy_map.copy()

    def get_threshold(self) -> float:
        return self._threshold

    def get_predictor_weights(self) -> dict[str, np.ndarray]:
        weights: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self._predictor_layers):
            weights[f"predictor.{index}.weight"] = layer["weight"].copy()
            weights[f"predictor.{index}.bias"] = layer["bias"].copy()
        weights["context_proj.weight"] = self._context_proj["weight"].copy()
        weights["context_proj.bias"] = self._context_proj["bias"].copy()
        weights["target_proj.weight"] = self._target_proj["weight"].copy()
        weights["target_proj.bias"] = self._target_proj["bias"].copy()
        return weights

    def load_predictor_weights(self, weights: dict[str, np.ndarray]) -> None:
        for index, layer in enumerate(self._predictor_layers):
            for suffix in ("weight", "bias"):
                key = f"predictor.{index}.{suffix}"
                if key in weights:
                    layer[suffix] = np.asarray(weights[key], dtype=np.float32).copy()
        for prefix, projection in (("context_proj", self._context_proj), ("target_proj", self._target_proj)):
            for suffix in ("weight", "bias"):
                key = f"{prefix}.{suffix}"
                if key in weights:
                    projection[suffix] = np.asarray(weights[key], dtype=np.float32).copy()

    def reset(self) -> None:
        self._prev_target_embedding = None
        self._embedding_buffer.clear()
        self._last_energy_map = np.zeros(self._patch_grid, dtype=np.float32)
        self._tick_count = 0

    @staticmethod
    def boxes_to_mask(boxes: list[BoundingBox], grid_h: int, grid_w: int) -> np.ndarray:
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        for box in boxes:
            col_start = int(box.x * grid_w)
            col_end = min(int((box.x + box.width) * grid_w) + 1, grid_w)
            row_start = int(box.y * grid_h)
            row_end = min(int((box.y + box.height) * grid_h) + 1, grid_h)
            mask[row_start:row_end, col_start:col_end] = True
        return mask

    def random_mask(self, ratio: float = 0.3) -> np.ndarray:
        grid_h, grid_w = self._patch_grid
        total = grid_h * grid_w
        count = max(1, int(ratio * total))
        mask = np.zeros(total, dtype=bool)
        indices = self._rng.choice(total, size=count, replace=False)
        mask[indices] = True
        return mask.reshape(grid_h, grid_w)

    def _tile_with_noise(self, embedding: np.ndarray, num_patches: int) -> np.ndarray:
        tiled = np.tile(embedding, (num_patches, 1))
        noise = self._rng.standard_normal(tiled.shape).astype(np.float32) * 0.01
        return tiled + noise

    def _update_ema(self) -> None:
        tau = self._ema_tau
        for key in ("weight", "bias"):
            self._target_proj[key] = (tau * self._target_proj[key]) + ((1.0 - tau) * self._context_proj[key])

    def _predict(self, x: np.ndarray) -> np.ndarray:
        for index, layer in enumerate(self._predictor_layers):
            x = _linear_forward(x, layer)
            if index < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        for index, layer in enumerate(self._predictor_layers):
            x = x @ layer["weight"] + layer["bias"]
            if index < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x

    def _update_threshold(self, mean_energy: float) -> None:
        alpha = self._energy_ema_alpha
        self._energy_ema = (1.0 - alpha) * self._energy_ema + (alpha * mean_energy)
        self._threshold = max(self._energy_ema * 1.5, self._min_threshold)

    def _compute_sigreg(self, context_embedding: np.ndarray, target_embedding: np.ndarray) -> float:
        l_invariance = float(np.mean((context_embedding - target_embedding) ** 2))
        buffer = list(self._embedding_buffer)
        if len(buffer) < 8:
            return l_invariance
        z = np.array(buffer, dtype=np.float32)
        stds = np.std(z, axis=0)
        l_variance = float(np.mean(np.maximum(0.0, self._gamma - stds)))
        z_centered = z - z.mean(axis=0, keepdims=True)
        n = z_centered.shape[0]
        cov = (z_centered.T @ z_centered) / max(n - 1, 1)
        d = cov.shape[0]
        mask = ~np.eye(d, dtype=bool)
        l_covariance = float(np.sum(cov[mask] ** 2)) / max(d * d, 1)
        return l_invariance + (self._lambda_var * l_variance) + (self._lambda_cov * l_covariance)


@dataclass
class _TrackState:
    id: str
    label: str
    first_seen_at: datetime
    last_seen_at: datetime
    first_observation_id: str
    last_observation_id: str
    observations: list[str]
    prototype_embedding: np.ndarray
    status: str = "visible"
    visibility_streak: int = 1
    occlusion_count: int = 0
    reidentification_count: int = 0
    persistence_confidence: float = 0.75
    continuity_score: float = 0.75
    last_similarity: float = 1.0
    status_history: list[str] = field(default_factory=lambda: ["visible"])
    bbox_pixels: dict[str, float] = field(default_factory=dict)
    patch_indices: list[int] = field(default_factory=list)
    misses: int = 0
    ghost_embedding: Optional[np.ndarray] = None
    cosine_history: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    just_created: bool = True


class ImmersiveJEPAEngine:
    """Single-session DINOv2/SAM-backed JEPA 2.1 proof engine."""

    def __init__(self, device: str = "mps", seed: int = 42) -> None:
        self.perception = PerceptionPipeline(device=device)
        self._rng = np.random.default_rng(seed)
        self._theta_ctx = np.eye(384, dtype=np.float32)
        self._theta_ctx += self._rng.standard_normal((384, 384)).astype(np.float32) * 1e-4
        self._theta_tgt = self._theta_ctx.copy()
        self._predictor_layers = [
            _xavier_uniform(384, 512, self._rng),
            _xavier_uniform(512, 512, self._rng),
            _xavier_uniform(512, 512, self._rng),
            _xavier_uniform(512, 384, self._rng),
        ]
        self._mu_E = 0.0
        self._tick_count = 0
        self._session_cov = np.zeros((384, 384), dtype=np.float32)
        self._energy_history: deque[float] = deque(maxlen=256)
        self._retrieval_history: deque[np.ndarray] = deque(maxlen=120)
        self._future_predictions: dict[int, deque[tuple[int, np.ndarray]]] = {
            1: deque(maxlen=256),
            2: deque(maxlen=256),
            5: deque(maxlen=256),
        }
        self._track_states: dict[str, _TrackState] = {}
        self._last_forecast_errors = {1: 0.0, 2: 0.0, 5: 0.0}
        self._last_state: Optional[np.ndarray] = None
        self._last_planning_ms = 0.0
        self._last_guard_active = False
        self._last_tau = 0.996
        self._guard_ticks_remaining = 0
        self._last_pred_loss = 0.0
        self._last_total_loss = 0.0
        self._tpds = TemporalParallaxDepthSeparator(grid=(14, 14))
        self._sag = SemanticAnchorGraph(max_matches_per_frame=8)
        self._cwma = CrossModalWorldModelAligner(lambda_cwma=0.15)
        self._ecgd = EpistemicConfidenceGate()
        self._setu2 = Setu2Bridge()
        self._last_energy_map: np.ndarray = np.zeros((14, 14), dtype=np.float32)
        self._last_depth_strata: DepthStrataMap = DepthStrataMap.cold_start()
        # Sprint 6: V-JEPA 2 world model integration
        self._vjepa2 = None
        self._vjepa2_loaded = False
        self._last_world_model_failure: Optional[dict[str, str]] = None
        self._last_tick_encoder_type = "surrogate"
        # Sprint 7: ViT-S/14 ONNX honest fallback (real patch encoder, NOT surrogate)
        self._fallback_encoder = None
        self._fallback_encoder_type = "none"
        self._is_true_surrogate = True
        if not _is_test_environment():
            try:
                self._vjepa2 = get_vjepa2_encoder()
            except Exception:
                self._vjepa2 = None
                self._vjepa2_loaded = False
                self._last_world_model_failure = {
                    "reason": "encoder initialization failed",
                    "stage": "init",
                }
            # Attempt ViT-S/14 ONNX as the honest fallback (real patch encoder, not surrogate)
            if self._vjepa2 is None:
                try:
                    from cloud.perception.vits14_onnx_encoder import ViTS14OnnxEncoder
                    if ViTS14OnnxEncoder.is_available():
                        self._fallback_encoder = ViTS14OnnxEncoder(ViTS14OnnxEncoder.default_model_path())
                        self._fallback_encoder_type = "dinov2-vits14-onnx"
                        self._is_true_surrogate = False
                        self._last_tick_encoder_type = "dinov2-vits14-onnx"
                    else:
                        self._fallback_encoder = None
                        self._fallback_encoder_type = "none"
                        self._is_true_surrogate = True
                except Exception as _e:
                    import logging as _lg
                    _lg.getLogger(__name__).warning("ViT-S/14 fallback init failed: %s", _e)
                    self._fallback_encoder = None
                    self._fallback_encoder_type = "none"
                    self._is_true_surrogate = True
        self._prev_predictor_embs: dict[str, np.ndarray] = {}  # per-session
        self._surprise_windows: dict[str, deque] = {}  # per-session running window

    def tick(
        self,
        frame: np.ndarray,
        *,
        session_id: str = "default",
        observation_id: Optional[str] = None,
    ) -> JEPATick:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        patch_tokens, mask_results = self.perception.encode(frame)
        if not mask_results:
            fallback_indices = self.perception.fallback_random_patches(n=32)
        else:
            fallback_indices = []

        self._ema_update()
        s_ctx = self._context_encode(patch_tokens)
        s_pred = self._predict(s_ctx, mask_results, fallback_indices=fallback_indices)
        s_tgt = self._target_encode(patch_tokens)
        energy_map = self._compute_energy(s_ctx, s_pred)
        depth_strata = self._tpds.update(
            energy_map_current=energy_map,
            energy_map_previous=self._last_energy_map,
            patch_tokens=s_ctx,
        )

        # Sprint 6: V-JEPA 2 world model prediction
        wm_encoder_emb = None
        wm_predictor_emb = None
        wm_prediction_error = None
        wm_epistemic = None
        wm_aleatoric = None
        wm_surprise = None
        wm_version = "surrogate"
        degraded = False
        degrade_reason = None
        degrade_stage = None

        if self._vjepa2 is not None:
            try:
                self._vjepa2.ensure_loaded()
                self._vjepa2_loaded = self._vjepa2.is_loaded
                enc_emb, pred_emb = self._vjepa2.encode(frame)
                wm_encoder_emb = enc_emb
                wm_predictor_emb = pred_emb
                wm_version = "vjepa2"

                # Cross-tick prediction error: compare current encoder with previous predictor
                if session_id in self._prev_predictor_embs:
                    prev_pred = self._prev_predictor_embs[session_id]
                    wm_prediction_error = float(np.sum((enc_emb - prev_pred) ** 2))
                else:
                    wm_prediction_error = 0.0  # first tick

                # Store current predictor for next tick
                self._prev_predictor_embs[session_id] = pred_emb.copy()

                # Surprise normalization: z-score within session window
                if session_id not in self._surprise_windows:
                    self._surprise_windows[session_id] = deque(maxlen=128)
                self._surprise_windows[session_id].append(wm_prediction_error)
                window = list(self._surprise_windows[session_id])
                if len(window) >= 3:
                    mu = float(np.mean(window))
                    std = float(np.std(window)) + 1e-8
                    wm_surprise = float(np.clip((wm_prediction_error - mu) / std, -3.0, 3.0))
                    wm_surprise = float((wm_surprise + 3.0) / 6.0)  # normalize to [0, 1]
                else:
                    wm_surprise = 0.5  # neutral during warmup

                # Epistemic: based on prediction error magnitude
                wm_epistemic = float(min(wm_prediction_error, 1.0))
                # Aleatoric: based on energy variance
                if len(self._energy_history) >= 3:
                    wm_aleatoric = float(min(np.var(list(self._energy_history)[-8:]), 1.0))
                else:
                    wm_aleatoric = 0.5
                self._last_world_model_failure = None
                self._last_tick_encoder_type = "vjepa2"

            except Exception as exc:
                self._vjepa2_loaded = False
                degraded = True
                degrade_stage = "encode"
                degrade_reason = str(exc)
                self._last_world_model_failure = {
                    "reason": degrade_reason,
                    "stage": degrade_stage,
                }
                self._last_tick_encoder_type = "surrogate"
        else:
            # V-JEPA2 unavailable — try ViT-S/14 ONNX honest fallback
            if self._fallback_encoder is not None:
                try:
                    enc_emb, patch_tokens = self._fallback_encoder.encode(frame)
                    wm_encoder_emb = enc_emb
                    # predictor_emb: use spatial mean as an honest approximation
                    wm_predictor_emb = patch_tokens.mean(axis=0).astype(np.float32)
                    wm_version = self._fallback_encoder_type
                    # NOT degraded — ViT-S/14 is a real patch encoder
                    degraded = False
                    degrade_reason = "vjepa2_unavailable_dinov2_vits14_active"
                    degrade_stage = None
                    self._last_tick_encoder_type = self._fallback_encoder_type

                    # Cross-tick prediction error
                    if session_id in self._prev_predictor_embs:
                        prev_pred = self._prev_predictor_embs[session_id]
                        wm_prediction_error = float(np.sum((enc_emb - prev_pred) ** 2))
                    else:
                        wm_prediction_error = 0.0
                    self._prev_predictor_embs[session_id] = wm_predictor_emb.copy()

                    # Surprise normalization
                    if session_id not in self._surprise_windows:
                        self._surprise_windows[session_id] = deque(maxlen=128)
                    self._surprise_windows[session_id].append(wm_prediction_error)
                    window = list(self._surprise_windows[session_id])
                    if len(window) >= 3:
                        mu = float(np.mean(window))
                        std = float(np.std(window)) + 1e-8
                        wm_surprise = float(np.clip((wm_prediction_error - mu) / std, -3.0, 3.0))
                        wm_surprise = float((wm_surprise + 3.0) / 6.0)
                    else:
                        wm_surprise = 0.5
                    wm_epistemic = float(min(wm_prediction_error, 1.0))
                    if len(self._energy_history) >= 3:
                        wm_aleatoric = float(min(np.var(list(self._energy_history)[-8:]), 1.0))
                    else:
                        wm_aleatoric = 0.5
                except Exception as _fb_exc:
                    import logging as _lg
                    _lg.getLogger(__name__).warning("ViT-S/14 fallback encode failed: %s", _fb_exc)
                    degraded = True
                    degrade_reason = f"vjepa2_and_vits14_unavailable: {_fb_exc}"
                    degrade_stage = "fallback_encode"
                    self._last_tick_encoder_type = "none"
            elif self._last_world_model_failure is not None:
                degraded = True
                degrade_reason = "vjepa2_and_vits14_unavailable"
                degrade_stage = self._last_world_model_failure.get("stage")
                self._last_tick_encoder_type = "none"
            else:
                # Test mode or no encoder configured
                self._last_tick_encoder_type = "surrogate"

        # Update TPDS with prediction error if available
        if wm_prediction_error is not None:
            depth_strata = self._tpds.update(
                energy_map_current=energy_map,
                energy_map_previous=self._last_energy_map,
                patch_tokens=s_ctx,
                prediction_error=wm_prediction_error,
            )
        mask_patch_lists = []
        if mask_results:
            first_mask = mask_results[0]
            if isinstance(first_mask, dict):
                mask_patch_lists = [mask.get("patch_indices", []) for mask in mask_results]
            else:
                mask_patch_lists = [mask.to_dict().get("patch_indices", []) for mask in mask_results]
        anchor_matches = self._sag.match(
            patch_tokens=s_ctx,
            depth_strata=depth_strata,
            mask_regions=mask_patch_lists,
        )
        cwma_result = self._cwma.apply_alignment(
            energy_map=energy_map,
            anchor_matches=anchor_matches,
            depth_strata=depth_strata,
        )
        if cwma_result[0] is not None:
            energy_map = cwma_result[0]
        alignment_loss = cwma_result[1]
        self._last_energy_map = energy_map.copy()
        self._last_depth_strata = depth_strata
        prediction_residual = s_tgt - s_pred
        sigreg_loss = self._sigreg(s_ctx)
        self._last_pred_loss = float(np.mean((s_pred - s_tgt) ** 2))
        self._last_total_loss = self._last_pred_loss + (0.04 * float(sigreg_loss))
        _ = prediction_residual
        current_state = s_ctx.mean(axis=0).astype(np.float32)
        forecast_errors = self._forecast(current_state)
        entity_tracks = self._update_occlusion(
            session_id=session_id,
            observation_id=observation_id,
            patch_tokens=s_ctx,
            mask_results=mask_results,
        )
        planning_time_ms = self._planning_speed_ms(current_state)
        fingerprint = self._update_fingerprint(energy_map, current_state)
        setu_descriptions: list[dict] = []
        for match in anchor_matches:
            gate_result = self._ecgd.evaluate(
                anchor_match=match,
                depth_strata=depth_strata,
                energy_history=list(self._energy_history),
                prediction_error=wm_prediction_error,
                surprise_score=wm_surprise,
            )
            description = self._setu2.describe_region(gate_result)
            setu_descriptions.append(
                {
                    "gate": gate_result.to_dict(),
                    "description": description.to_dict(),
                }
            )
        caption_score, retrieval_score = self._baselines(current_state)
        mean_energy = float(energy_map.mean())
        energy_std = float(energy_map.std())
        talker_event = self._talker(mean_energy, energy_std, entity_tracks)
        warmup = self._tick_count < 10
        if warmup:
            energy_map = np.zeros((14, 14), dtype=np.float32)
            talker_event = None
        tick = JEPATick(
            energy_map=energy_map.astype(np.float32),
            entity_tracks=entity_tracks,
            talker_event=talker_event,
            sigreg_loss=float(sigreg_loss),
            forecast_errors={int(key): float(value) for key, value in forecast_errors.items()},
            session_fingerprint=fingerprint.astype(np.float32),
            planning_time_ms=float(planning_time_ms),
            caption_score=float(caption_score),
            retrieval_score=float(retrieval_score),
            timestamp_ms=int(time() * 1000),
            warmup=warmup,
            mask_results=[mask.to_dict() for mask in mask_results],
            mean_energy=mean_energy,
            energy_std=energy_std,
            guard_active=self._last_guard_active,
            ema_tau=self._last_tau,
            depth_strata=depth_strata.to_dict(),
            anchor_matches=[match.to_dict() for match in anchor_matches],
            setu_descriptions=setu_descriptions,
            alignment_loss=float(alignment_loss),
            l2_embedding=wm_encoder_emb.tolist() if wm_encoder_emb is not None else None,
            predicted_next_embedding=wm_predictor_emb.tolist() if wm_predictor_emb is not None else None,
            prediction_error=wm_prediction_error,
            epistemic_uncertainty=wm_epistemic,
            aleatoric_uncertainty=wm_aleatoric,
            surprise_score=wm_surprise,
            world_model_version=wm_version,
            configured_encoder="vjepa2",
            last_tick_encoder_type=self._last_tick_encoder_type,
            degraded=degraded,
            degrade_reason=degrade_reason,
            degrade_stage=degrade_stage,
        )
        self._last_state = current_state.copy()
        self._tick_count += 1
        self._adapt_context(current_state)
        return tick

    def forecast_last_state(self, k: int) -> tuple[np.ndarray, Optional[float]]:
        if self._last_state is None:
            return np.zeros(384, dtype=np.float32), None
        prediction = np.asarray(self._last_state, dtype=np.float32).copy()
        for _ in range(max(k, 1)):
            prediction = self._predict_vector(prediction)
        return prediction.astype(np.float32), self._last_forecast_errors.get(int(k))

    def _ema_update(self) -> None:
        diff = float(np.linalg.norm(self._theta_tgt - self._theta_ctx))
        if diff < 0.01:
            self._guard_ticks_remaining = 10
        tau = 0.90 if self._guard_ticks_remaining > 0 else 0.996
        self._last_guard_active = self._guard_ticks_remaining > 0
        self._last_tau = tau
        self._theta_tgt = (tau * self._theta_tgt) + ((1.0 - tau) * self._theta_ctx)
        if self._guard_ticks_remaining > 0:
            self._guard_ticks_remaining -= 1

    def _context_encode(self, patch_tokens: np.ndarray) -> np.ndarray:
        return np.asarray(patch_tokens, dtype=np.float32) @ self._theta_ctx

    def _target_encode(self, patch_tokens: np.ndarray) -> np.ndarray:
        return np.asarray(patch_tokens, dtype=np.float32) @ self._theta_tgt

    def _predict(
        self,
        patch_tokens: np.ndarray,
        mask_results: list[MaskResult],
        *,
        fallback_indices: list[int],
    ) -> np.ndarray:
        predicted_all = self._predict_batch(np.asarray(patch_tokens, dtype=np.float32))
        predicted = np.asarray(patch_tokens, dtype=np.float32).copy()
        mask_indices = sorted({index for mask in mask_results for index in mask.patch_indices})
        if not mask_indices:
            mask_indices = fallback_indices
        if not mask_indices:
            return predicted_all.astype(np.float32)
        predicted[mask_indices] = predicted_all[mask_indices]
        return predicted.astype(np.float32)

    def _predict_vector(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=np.float32)
        for index, layer in enumerate(self._predictor_layers):
            x = _linear_forward(x, layer)
            if index < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x.astype(np.float32)

    def _predict_batch(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)
        for index, layer in enumerate(self._predictor_layers):
            x = x @ layer["weight"] + layer["bias"]
            if index < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x.astype(np.float32)

    def _compute_energy(self, s_ctx: np.ndarray, s_pred: np.ndarray) -> np.ndarray:
        energy = np.sum((np.asarray(s_ctx, dtype=np.float32) - np.asarray(s_pred, dtype=np.float32)) ** 2, axis=1)
        return energy.reshape(14, 14).astype(np.float32)

    def _sigreg(self, s_ctx: np.ndarray) -> float:
        centered = np.asarray(s_ctx, dtype=np.float32) - np.asarray(s_ctx, dtype=np.float32).mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rank = min(64, vt.shape[0])
        sketch = np.zeros((vt.shape[1], vt.shape[1]), dtype=np.float32)
        for index in range(rank):
            vector = vt[index]
            sketch += np.outer(vector, vector).astype(np.float32)
        identity = np.eye(sketch.shape[0], dtype=np.float32)
        return float(np.linalg.norm(sketch - identity, ord="fro") ** 2 / max(sketch.shape[0], 1))

    def _update_occlusion(
        self,
        *,
        session_id: str,
        observation_id: Optional[str],
        patch_tokens: np.ndarray,
        mask_results: list[MaskResult],
    ) -> list[EntityTrack]:
        observation_key = observation_id or f"tick_{self._tick_count}"
        detections = []
        for mask in mask_results:
            indices = mask.patch_indices or self.perception.fallback_random_patches(n=8)
            observed = patch_tokens[indices].mean(axis=0).astype(np.float32)
            detections.append((mask, observed))

        matched_tracks: set[str] = set()
        matched_detection_indices: set[int] = set()

        for detection_index, (mask, observed) in enumerate(detections):
            best_track_id: Optional[str] = None
            best_similarity = 0.0
            for track_id, state in self._track_states.items():
                similarity = _cosine_similarity(observed, state.prototype_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_track_id = track_id
            if best_track_id is None or best_similarity < 0.72:
                track_id = f"trk_{uuid4().hex[:8]}"
                now = datetime.now(timezone.utc)
                state = _TrackState(
                    id=track_id,
                    label=f"entity-{len(self._track_states) + 1}",
                    first_seen_at=now,
                    last_seen_at=now,
                    first_observation_id=observation_key,
                    last_observation_id=observation_key,
                    observations=[observation_key],
                    prototype_embedding=observed.copy(),
                    bbox_pixels=mask.bbox_pixels,
                    patch_indices=list(mask.patch_indices),
                )
                state.cosine_history.append(1.0)
                self._track_states[track_id] = state
                matched_tracks.add(track_id)
                matched_detection_indices.add(detection_index)
                continue

            state = self._track_states[best_track_id]
            ghost_similarity = _cosine_similarity(observed, state.ghost_embedding) if state.ghost_embedding is not None else best_similarity
            was_occluded = state.status == "occluded"
            state.prototype_embedding = ((0.8 * state.prototype_embedding) + (0.2 * observed)).astype(np.float32)
            state.last_similarity = best_similarity
            state.continuity_score = best_similarity
            state.persistence_confidence = min(1.0, max(0.0, (best_similarity + 1.0) / 2.0))
            state.last_seen_at = datetime.now(timezone.utc)
            state.last_observation_id = observation_key
            if not state.observations or state.observations[-1] != observation_key:
                state.observations.append(observation_key)
            state.misses = 0
            state.visibility_streak += 1
            state.bbox_pixels = mask.bbox_pixels
            state.patch_indices = list(mask.patch_indices)
            state.just_created = False
            if was_occluded and ghost_similarity > 0.72:
                state.status = "re-identified"
                state.reidentification_count += 1
            else:
                state.status = "visible"
            state.status_history.append(state.status)
            state.cosine_history.append(best_similarity)
            matched_tracks.add(best_track_id)
            matched_detection_indices.add(detection_index)

        observed_vectors = [observed for _, observed in detections]
        for track_id, state in self._track_states.items():
            if track_id in matched_tracks:
                continue
            best_similarity = max((_cosine_similarity(observed, state.prototype_embedding) for observed in observed_vectors), default=0.0)
            state.last_similarity = best_similarity
            state.cosine_history.append(best_similarity)
            state.misses += 1
            if state.misses >= 3 and best_similarity < 0.65:
                if state.status != "occluded":
                    state.occlusion_count += 1
                state.status = "occluded"
                state.ghost_embedding = self._predict_vector(state.prototype_embedding)
            elif state.misses >= 5:
                state.status = "disappeared"
            else:
                state.status = "violated prediction" if best_similarity < 0.5 else state.status
            state.persistence_confidence = max(0.0, 1.0 - (0.18 * state.misses))
            state.continuity_score = best_similarity
            state.status_history.append(state.status)

        self._prune_track_states()

        entity_tracks: list[EntityTrack] = []
        for state in self._track_states.values():
            duration_ms = int((state.last_seen_at - state.first_seen_at).total_seconds() * 1000)
            entity_tracks.append(
                EntityTrack(
                    id=state.id,
                    session_id=session_id,
                    label=state.label,
                    status=state.status,  # type: ignore[arg-type]
                    first_seen_at=state.first_seen_at,
                    last_seen_at=state.last_seen_at,
                    first_observation_id=state.first_observation_id,
                    last_observation_id=state.last_observation_id,
                    observations=state.observations[-24:],
                    visibility_streak=state.visibility_streak,
                    occlusion_count=state.occlusion_count,
                    reidentification_count=state.reidentification_count,
                    persistence_confidence=float(state.persistence_confidence),
                    continuity_score=float(state.continuity_score),
                    last_similarity=float(state.last_similarity),
                    prototype_embedding=state.prototype_embedding.astype(np.float32).tolist(),
                    status_history=state.status_history[-24:],
                    metadata={
                        "bbox_pixels": state.bbox_pixels,
                        "ghost_bbox_pixels": state.bbox_pixels if state.status == "occluded" else None,
                        "patch_indices": state.patch_indices,
                        "ghost_patch_indices": state.patch_indices if state.status == "occluded" else [],
                        "duration_ms": duration_ms,
                        "cosine_history": [float(item) for item in state.cosine_history],
                        "just_created": state.just_created,
                    },
                )
            )
            state.just_created = False
        entity_tracks.sort(key=lambda track: track.last_seen_at, reverse=True)
        return entity_tracks

    def _prune_track_states(self) -> None:
        now = datetime.now(timezone.utc)
        removable_ids: list[str] = []
        for track_id, state in self._track_states.items():
            age_s = (now - state.last_seen_at).total_seconds()
            if state.status in {"disappeared", "violated prediction"} and (state.misses >= 4 or age_s > 8.0):
                removable_ids.append(track_id)
                continue
            if state.status == "occluded" and state.misses >= 10 and age_s > 20.0:
                removable_ids.append(track_id)

        for track_id in removable_ids:
            self._track_states.pop(track_id, None)

    def _forecast(self, current_state: np.ndarray) -> dict[int, float]:
        forecast_errors = {1: self._last_forecast_errors.get(1, 0.0), 2: self._last_forecast_errors.get(2, 0.0), 5: self._last_forecast_errors.get(5, 0.0)}
        for horizon in (1, 2, 5):
            queue = self._future_predictions[horizon]
            due_predictions = [pred for due_tick, pred in list(queue) if due_tick == self._tick_count]
            self._future_predictions[horizon] = deque(
                [(due_tick, pred) for due_tick, pred in queue if due_tick > self._tick_count],
                maxlen=256,
            )
            if due_predictions:
                forecast_errors[horizon] = float(np.mean([np.linalg.norm(current_state - prediction) ** 2 for prediction in due_predictions]))
            prediction = np.asarray(current_state, dtype=np.float32).copy()
            for _ in range(horizon):
                prediction = self._predict_vector(prediction)
            self._future_predictions[horizon].append((self._tick_count + horizon, prediction.astype(np.float32)))
        self._last_forecast_errors = forecast_errors
        return forecast_errors

    def _update_fingerprint(self, energy_map: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        mean_energy = float(np.asarray(energy_map, dtype=np.float32).mean())
        self._energy_history.append(mean_energy)
        centered = current_state - float(np.mean(current_state))
        outer = np.outer(centered, centered).astype(np.float32)
        self._session_cov = (0.98 * self._session_cov) + (0.02 * outer)
        eigenvalues, eigenvectors = np.linalg.eigh(self._session_cov + (1e-6 * np.eye(self._session_cov.shape[0], dtype=np.float32)))
        _ = eigenvalues
        top_vectors = eigenvectors[:, -32:]
        fingerprint = np.concatenate(
            [
                np.array(
                    [
                        float(np.mean(self._energy_history)),
                        float(np.var(self._energy_history)),
                    ],
                    dtype=np.float32,
                ),
                top_vectors.T.reshape(-1).astype(np.float32),
            ]
        )
        return fingerprint.astype(np.float32)

    def _baselines(self, current_state: np.ndarray) -> tuple[float, float]:
        previous = self._retrieval_history[-1] if self._retrieval_history else np.asarray(current_state, dtype=np.float32)
        caption_score = max(0.0, _cosine_similarity(np.asarray(current_state, dtype=np.float32), previous))
        retrieval_score = max(
            (_cosine_similarity(np.asarray(current_state, dtype=np.float32), item) for item in self._retrieval_history),
            default=caption_score,
        )
        self._retrieval_history.append(np.asarray(current_state, dtype=np.float32).copy())
        return float(caption_score), float(retrieval_score)

    def _talker(self, mean_energy: float, energy_std: float, entity_tracks: list[EntityTrack]) -> Optional[str]:
        self._mu_E = (0.05 * mean_energy) + (0.95 * self._mu_E)
        tau_talk = self._mu_E + (2.0 * energy_std)
        if mean_energy <= tau_talk:
            return None
        if any(track.status == "re-identified" for track in entity_tracks):
            return "OCCLUSION_END"
        if any(track.status == "occluded" for track in entity_tracks):
            return "OCCLUSION_START"
        if any(track.status == "disappeared" for track in entity_tracks):
            return "ENTITY_DISAPPEARED"
        if any(bool(track.metadata.get("just_created")) for track in entity_tracks):
            return "ENTITY_APPEARED"
        return "PREDICTION_VIOLATION"

    def _planning_speed_ms(self, current_state: np.ndarray) -> float:
        started_at = perf_counter()
        prediction = np.asarray(current_state, dtype=np.float32).copy()
        for _ in range(5):
            prediction = self._predict_vector(prediction)
        _ = prediction
        self._last_planning_ms = (perf_counter() - started_at) * 1000.0
        return self._last_planning_ms

    def _adapt_context(self, current_state: np.ndarray) -> None:
        normalized = np.asarray(current_state, dtype=np.float32)
        norm = float(np.linalg.norm(normalized)) or 1.0
        normalized = normalized / norm
        update = np.outer(normalized, normalized).astype(np.float32) * 1e-4
        self._theta_ctx = np.clip(self._theta_ctx + update, -2.0, 2.0).astype(np.float32)
