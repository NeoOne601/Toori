"""
clap_projector.py — CLAP-to-DINOv2 projection head.

Maps LAION-CLAP 512-dim audio-text embeddings into DINOv2-ViT-S/14 384-dim
visual space. Once trained, this enables cross-modal retrieval:
  audio query (humming/music) → projected into visual space → cosine search on visual FAISS index
  → finds the VIDEO FRAMES that match the audio.

Architecture: Linear(512→384) → LeakyReLU → LayerNorm(384) → Linear(384→384) → L2-normalize

Weights stored as numpy .npz file (no torch, no onnxruntime required — pure numpy inference).
Seeded random initialization uses seed=20260327 for consistency with AudioEncoder.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


class CLAPProjector:
    """Projects CLAP 512-dim embeddings into DINOv2 384-dim visual space."""

    _CLAP_DIM = 512
    _VISUAL_DIM = 384
    _SEED = 20260327

    def __init__(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"CLAPProjector weights not found at {path}. "
                f"Run: python3.11 scripts/train_clap_projector.py --random-init"
            )
        data = np.load(str(path))
        self._w0 = data['w0'].astype(np.float32)       # [384, 512]
        self._b0 = data['b0'].astype(np.float32)       # [384]
        self._w1 = data['w1'].astype(np.float32)       # [384, 384]
        self._b1 = data['b1'].astype(np.float32)       # [384]
        self._ln_gamma = data['ln_gamma'].astype(np.float32)  # [384]
        self._ln_beta = data['ln_beta'].astype(np.float32)    # [384]

    def project(self, clap_emb: np.ndarray) -> np.ndarray:
        """Project CLAP 512-dim embedding into DINOv2 384-dim visual space.

        Args:
            clap_emb: np.ndarray[512] — CLAP audio or text embedding, any norm
        Returns:
            np.ndarray[384] — unit-norm embedding in visual space
        """
        x = clap_emb.astype(np.float32).flatten()[:self._CLAP_DIM]
        # Layer 0: Linear + LeakyReLU
        x = np.dot(self._w0, x) + self._b0
        x = np.where(x >= 0, x, 0.1 * x)  # LeakyReLU alpha=0.1
        # LayerNorm
        mean = x.mean()
        std = np.sqrt(x.var() + 1e-6)
        x = (x - mean) / std * self._ln_gamma + self._ln_beta
        # Layer 1: Linear
        x = np.dot(self._w1, x) + self._b1
        # L2 normalize into visual embedding space
        norm = np.linalg.norm(x) + 1e-9
        return (x / norm).astype(np.float32)

    @classmethod
    def default_weights_path(cls) -> str:
        return "models/audio/clap_projector.npz"

    @classmethod
    def is_available(cls) -> bool:
        try:
            return Path(cls.default_weights_path()).exists()
        except Exception:
            return False

    @classmethod
    def create_random_init(cls, save_path: Optional[str] = None) -> 'CLAPProjector':
        """Create and save random-weight projector using seed=20260327.

        Produces internally CONSISTENT projections but NOT cross-modal semantics.
        Useful for testing the pipeline before real CLAP training data is available.
        Document clearly in UI: 'random init — no cross-modal semantics until trained'.
        """
        rng = np.random.RandomState(cls._SEED)
        # Xavier initialization for stable gradients
        w0 = rng.randn(cls._VISUAL_DIM, cls._CLAP_DIM).astype(np.float32) * np.sqrt(2.0 / cls._CLAP_DIM)
        b0 = np.zeros(cls._VISUAL_DIM, dtype=np.float32)
        w1 = rng.randn(cls._VISUAL_DIM, cls._VISUAL_DIM).astype(np.float32) * np.sqrt(2.0 / cls._VISUAL_DIM)
        b1 = np.zeros(cls._VISUAL_DIM, dtype=np.float32)
        ln_gamma = np.ones(cls._VISUAL_DIM, dtype=np.float32)
        ln_beta = np.zeros(cls._VISUAL_DIM, dtype=np.float32)

        path = save_path or cls.default_weights_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w0=w0, b0=b0, w1=w1, b1=b1, ln_gamma=ln_gamma, ln_beta=ln_beta)
        return cls(path)
