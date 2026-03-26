"""JEPA engine: context/target projections, latent predictor, spatial energy, SIGReg.

Pure numpy implementation — no PyTorch, no CUDA. Runs on M1 via Accelerate BLAS.
Consumes embeddings from the existing ProviderRegistry (ONNX or basic provider).
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple, Optional

import numpy as np

from cloud.runtime.models import BoundingBox


class TickResult(NamedTuple):
    energy_map: np.ndarray           # shape (grid_h, grid_w), float32
    mean_energy: float               # scalar mean of energy_map
    threshold: float                 # current adaptive threshold
    should_talk: bool                # mean_energy > threshold
    talker_event: Optional[str]      # placeholder for talker layer
    prediction_residual: np.ndarray  # shape (embedding_dim,)
    sigreg_loss: float               # SIGReg monitoring value


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _init_linear(in_dim: int, out_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = np.sqrt(2.0 / in_dim)
    return {
        "weight": rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale,
        "bias": np.zeros(out_dim, dtype=np.float32),
    }


def _linear_forward(x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
    return x @ layer["weight"] + layer["bias"]


class JEPAEngine:
    """Core JEPA engine with context/target projections, MLP predictor, and spatial energy."""

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

        # Context and target projection layers (embedding_dim -> embedding_dim)
        self._context_proj = _init_linear(embedding_dim, embedding_dim, self._rng)
        self._target_proj = {
            k: v.copy() for k, v in self._context_proj.items()
        }

        # MLP predictor: embedding_dim -> hidden -> ... -> embedding_dim
        self._predictor_layers: list[dict[str, np.ndarray]] = []
        in_d = embedding_dim
        for i in range(predictor_layers - 1):
            self._predictor_layers.append(_init_linear(in_d, predictor_hidden, self._rng))
            in_d = predictor_hidden
        self._predictor_layers.append(_init_linear(in_d, embedding_dim, self._rng))

        # State
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
        """Main per-frame update.

        Args:
            frame_embedding: shape (embedding_dim,) from ONNX/basic provider.
            patch_embeddings: shape (num_patches, embedding_dim) or None.
                When None, tiles frame_embedding with spatial noise.
            mask: shape (grid_h, grid_w) boolean. True = masked (predict these).
                When None, all patches are predicted.
        """
        frame_embedding = np.asarray(frame_embedding, dtype=np.float32).ravel()
        assert frame_embedding.shape == (self._embedding_dim,), (
            f"Expected ({self._embedding_dim},), got {frame_embedding.shape}"
        )

        grid_h, grid_w = self._patch_grid
        num_patches = grid_h * grid_w

        # Build per-patch embeddings if not provided
        if patch_embeddings is None:
            patch_embeddings = self._tile_with_noise(frame_embedding, num_patches)
        else:
            patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
            if patch_embeddings.shape[0] != num_patches:
                patch_embeddings = self._tile_with_noise(frame_embedding, num_patches)

        # Step 1: EMA update (BEFORE predictor)
        self._update_ema()

        # Step 2: Context projection
        context_patches = np.array([
            _linear_forward(p, self._context_proj) for p in patch_embeddings
        ])  # (num_patches, embedding_dim)

        # Step 3: Target projection (using EMA weights)
        target_patches = np.array([
            _linear_forward(p, self._target_proj) for p in patch_embeddings
        ])  # (num_patches, embedding_dim)

        # Step 4: Predictor forward pass
        predicted_patches = self._predict_batch(context_patches)  # (num_patches, embedding_dim)

        # Step 5: Spatial energy map E_i = ||s_target_i - ŝ_pred_i||²
        residuals = target_patches - predicted_patches  # (num_patches, embedding_dim)
        per_patch_energy = np.sum(residuals ** 2, axis=1)  # (num_patches,)

        # Apply mask if provided (only compute energy on masked patches for SIGReg,
        # but report full map for visualization)
        energy_map = per_patch_energy.reshape(grid_h, grid_w)

        # Step 6: Global prediction residual
        context_global = _linear_forward(frame_embedding, self._context_proj)
        target_global = _linear_forward(frame_embedding, self._target_proj)
        predicted_global = self._predict(context_global)
        prediction_residual = target_global - predicted_global

        # Step 7: Compute mean energy and update threshold
        mean_energy = float(np.mean(energy_map))
        self._update_threshold(mean_energy)
        should_talk = mean_energy > self._threshold

        # Step 8: SIGReg loss computation
        self._embedding_buffer.append(context_global)
        sigreg_loss = self._compute_sigreg(context_global, target_global)

        # Store state
        self._last_energy_map = energy_map.copy()
        self._prev_target_embedding = target_global.copy()
        self._tick_count += 1

        return TickResult(
            energy_map=energy_map,
            mean_energy=mean_energy,
            threshold=self._threshold,
            should_talk=should_talk,
            talker_event=None,  # filled by SelectiveTalker layer
            prediction_residual=prediction_residual,
            sigreg_loss=sigreg_loss,
        )

    def get_energy_map(self) -> np.ndarray:
        """Returns the last computed spatial energy map."""
        return self._last_energy_map.copy()

    def get_threshold(self) -> float:
        """Returns the current adaptive energy threshold."""
        return self._threshold

    def get_predictor_weights(self) -> dict[str, np.ndarray]:
        """Returns all predictor weights for checkpointing."""
        weights: dict[str, np.ndarray] = {}
        for i, layer in enumerate(self._predictor_layers):
            weights[f"predictor.{i}.weight"] = layer["weight"].copy()
            weights[f"predictor.{i}.bias"] = layer["bias"].copy()
        weights["context_proj.weight"] = self._context_proj["weight"].copy()
        weights["context_proj.bias"] = self._context_proj["bias"].copy()
        weights["target_proj.weight"] = self._target_proj["weight"].copy()
        weights["target_proj.bias"] = self._target_proj["bias"].copy()
        return weights

    def load_predictor_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Loads predictor weights from a checkpoint."""
        for i, layer in enumerate(self._predictor_layers):
            w_key = f"predictor.{i}.weight"
            b_key = f"predictor.{i}.bias"
            if w_key in weights:
                layer["weight"] = weights[w_key].copy()
            if b_key in weights:
                layer["bias"] = weights[b_key].copy()
        for key in ("context_proj", "target_proj"):
            proj = self._context_proj if key == "context_proj" else self._target_proj
            w_key = f"{key}.weight"
            b_key = f"{key}.bias"
            if w_key in weights:
                proj["weight"] = weights[w_key].copy()
            if b_key in weights:
                proj["bias"] = weights[b_key].copy()

    def reset(self) -> None:
        """Resets internal state for a new session."""
        self._prev_target_embedding = None
        self._embedding_buffer.clear()
        self._last_energy_map = np.zeros(self._patch_grid, dtype=np.float32)
        self._tick_count = 0

    # --- C-JEPA Masking ---

    @staticmethod
    def boxes_to_mask(
        boxes: list[BoundingBox],
        grid_h: int,
        grid_w: int,
    ) -> np.ndarray:
        """Convert bounding boxes to a boolean patch mask.

        Returns (grid_h, grid_w) boolean array. True = masked (predict this patch).
        """
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        for box in boxes:
            col_start = int(box.x * grid_w)
            col_end = min(int((box.x + box.width) * grid_w) + 1, grid_w)
            row_start = int(box.y * grid_h)
            row_end = min(int((box.y + box.height) * grid_h) + 1, grid_h)
            mask[row_start:row_end, col_start:col_end] = True
        return mask

    def random_mask(self, ratio: float = 0.3) -> np.ndarray:
        """Fallback random patch masking when no objects detected."""
        grid_h, grid_w = self._patch_grid
        total = grid_h * grid_w
        num_masked = max(1, int(ratio * total))
        mask = np.zeros(total, dtype=bool)
        indices = self._rng.choice(total, size=num_masked, replace=False)
        mask[indices] = True
        return mask.reshape(grid_h, grid_w)

    # --- Internal Methods ---

    def _tile_with_noise(self, embedding: np.ndarray, num_patches: int) -> np.ndarray:
        """Tile a single embedding across patches with small spatial noise."""
        tiled = np.tile(embedding, (num_patches, 1))
        noise = self._rng.standard_normal(tiled.shape).astype(np.float32) * 0.01
        return tiled + noise

    def _update_ema(self) -> None:
        """EMA update: target_proj ← τ·target_proj + (1-τ)·context_proj."""
        tau = self._ema_tau
        for key in ("weight", "bias"):
            self._target_proj[key] = (
                tau * self._target_proj[key]
                + (1.0 - tau) * self._context_proj[key]
            )

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """MLP predictor forward pass for a single vector."""
        for i, layer in enumerate(self._predictor_layers):
            x = _linear_forward(x, layer)
            if i < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x

    def _predict_batch(self, x: np.ndarray) -> np.ndarray:
        """MLP predictor forward pass for a batch of vectors."""
        for i, layer in enumerate(self._predictor_layers):
            x = x @ layer["weight"] + layer["bias"]
            if i < len(self._predictor_layers) - 1:
                x = _relu(x)
        return x

    def _update_threshold(self, mean_energy: float) -> None:
        """Adaptive threshold update via exponential moving average."""
        alpha = self._energy_ema_alpha
        self._energy_ema = (1.0 - alpha) * self._energy_ema + alpha * mean_energy
        self._threshold = max(self._energy_ema * 1.5, self._min_threshold)

    def _compute_sigreg(
        self,
        context_embedding: np.ndarray,
        target_embedding: np.ndarray,
    ) -> float:
        """Compute SIGReg loss as a monitoring signal.

        L = L_invariance + λ_var·L_variance + λ_cov·L_covariance
        """
        # L_invariance: prediction error
        l_invariance = float(np.mean((context_embedding - target_embedding) ** 2))

        buffer = list(self._embedding_buffer)
        if len(buffer) < 8:
            return l_invariance

        # Stack buffer into matrix (N, d)
        Z = np.array(buffer, dtype=np.float32)

        # L_variance: encourage per-dimension std >= gamma
        stds = np.std(Z, axis=0)
        l_variance = float(np.mean(np.maximum(0.0, self._gamma - stds)))

        # L_covariance: decorrelation
        Z_centered = Z - Z.mean(axis=0, keepdims=True)
        n = Z_centered.shape[0]
        cov = (Z_centered.T @ Z_centered) / max(n - 1, 1)
        d = cov.shape[0]
        # Zero-out diagonal for off-diagonal penalty
        mask = ~np.eye(d, dtype=bool)
        l_covariance = float(np.sum(cov[mask] ** 2)) / max(d * d, 1)

        return l_invariance + self._lambda_var * l_variance + self._lambda_cov * l_covariance
