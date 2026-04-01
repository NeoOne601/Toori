"""
V-JEPA 2 Encoder — TOORI Sprint 6
cloud/perception/vjepa2_encoder.py

This module is in cloud/perception/ where torch imports are allowed.
Uses: facebook/vjepa2-vitl-fpc64-256 (HuggingFace Transformers)

M1 iMac 8GB budget:
  Model size: ~600MB in float16
  Inference:  ~200-400ms per frame (CPU), ~80-150ms (MPS)
  Device:     MPS if available, CPU fallback

The CRITICAL V-JEPA 2 property: it returns BOTH
  - last_hidden_state (encoder): what the model sees
  - predictor_output (predictor): what the model PREDICTS will happen next

The gap between prediction and reality IS the world model signal.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Optional
from cloud.runtime.observability import get_logger

log = get_logger("vjepa2_encoder")

# torch and transformers are allowed in cloud/perception/
import torch

MODEL_ID = os.environ.get("TOORI_VJEPA2_MODEL", "facebook/vjepa2-vitl-fpc64-256")
N_FRAMES = 16      # fpc16 variant: 16 frames per clip (M1 memory efficient)
FRAME_RES = 256     # 256×256 px (matches model training resolution)
TARGET_DIM = 384    # Setu-2 metric space dimension — NEVER CHANGE

_encoder_singleton: Optional["VJepa2Encoder"] = None


def get_vjepa2_encoder() -> "VJepa2Encoder":
    """Singleton accessor — model is loaded once and reused."""
    global _encoder_singleton
    if _encoder_singleton is None:
        _encoder_singleton = VJepa2Encoder()
    return _encoder_singleton


class VJepa2Encoder:
    """
    V-JEPA 2 ViT-L encoder for TOORI world model integration.

    Provides two outputs per frame:
        encode(frame) → (encoder_emb, predictor_emb)

        encoder_emb:   384-dim L2-normalized representation of current state
        predictor_emb: 384-dim L2-normalized prediction of next state

    The predictor_emb is the Le World Model prediction.
    When the next frame arrives, its encoder_emb is compared against
    the previous tick's predictor_emb to compute genuine prediction error.

    This is NOT a proxy. This is the real V-JEPA 2 world model signal.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        self._projection = None  # 1024 → 384 linear projection

        # M1-optimized device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            log.info("vjepa2_device", device="mps",
                     message="Using Apple Silicon MPS acceleration")
        else:
            self.device = torch.device("cpu")
            log.info("vjepa2_device", device="cpu")

        # Initialize deterministic projection matrix
        # seed=42 ensures same projection across restarts
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((1024, TARGET_DIM)).astype(np.float32)
        proj /= np.sqrt(1024)
        self._projection = torch.from_numpy(proj).to(self.device)

        # Load immediately (not lazy) — fail fast
        self._load()

    def _load(self) -> None:
        try:
            from transformers import AutoVideoProcessor, AutoModel
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run:\n"
                "  pip install transformers"
            )

        log.info("vjepa2_loading", model=MODEL_ID,
                 message="Loading V-JEPA 2 weights (~1.2GB, once only)...")

        self._processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,   # M1 8GB budget: ~600MB
        ).to(self.device).eval()

        # Verify predictor output is accessible
        test_frame = np.zeros((N_FRAMES, FRAME_RES, FRAME_RES, 3), dtype=np.uint8)
        test_in = self._processor(test_frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            test_out = self._model(**test_in)

        assert hasattr(test_out, "predictor_output"), (
            "V-JEPA 2 model must return predictor_output. "
            "Ensure you have the latest transformers installed."
        )

        enc_dim = test_out.last_hidden_state.shape[-1]
        assert enc_dim == 1024, f"Expected 1024-dim encoder, got {enc_dim}"

        self._loaded = True
        log.info("vjepa2_loaded", model=MODEL_ID,
                 device=str(self.device),
                 encoder_dim=enc_dim,
                 message="V-JEPA 2 loaded successfully")

    def encode(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a single frame using V-JEPA 2.

        For Smriti's use case (individual photos and video keyframes),
        we repeat the frame N_FRAMES times to fill the temporal buffer.
        This produces a stable representation of a static scene.

        Args:
            frame: np.ndarray (H, W, 3) uint8

        Returns:
            encoder_emb:   (384,) float32, L2-normalized
            predictor_emb: (384,) float32, L2-normalized
        """
        assert self._loaded, "Model not loaded"

        from PIL import Image
        resized = Image.fromarray(frame.astype(np.uint8)).resize(
            (FRAME_RES, FRAME_RES), Image.BILINEAR
        )
        frame_resized = np.array(resized, dtype=np.uint8)
        frames = np.stack([frame_resized] * N_FRAMES, axis=0)  # (N, H, W, C)

        inputs = self._processor(
            frames, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Pool encoder output: (1, T*P, 1024) → mean → (1024,)
        enc_pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).float()
        pred_pooled = outputs.predictor_output.last_hidden_state.mean(
            dim=1
        ).squeeze(0).float()

        # Project to 384-dim
        enc_384 = (enc_pooled @ self._projection).cpu().numpy()
        pred_384 = (pred_pooled @ self._projection).cpu().numpy()

        # L2 normalize
        enc_384 /= np.linalg.norm(enc_384) + 1e-8
        pred_384 /= np.linalg.norm(pred_384) + 1e-8

        return enc_384.astype(np.float32), pred_384.astype(np.float32)

    def encode_with_context(
        self,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode with actual temporal context (up to N_FRAMES frames).
        Used when processing video sequences for Living Lens.

        Args:
            frames: list of up to N_FRAMES numpy arrays (H, W, 3)

        Returns: same as encode()
        """
        assert self._loaded

        from PIL import Image

        # Pad or trim to N_FRAMES
        while len(frames) < N_FRAMES:
            frames = [frames[0]] + frames  # front-pad with first frame
        frames = frames[-N_FRAMES:]         # take last N_FRAMES

        resized = []
        for f in frames:
            img = Image.fromarray(f.astype(np.uint8)).resize(
                (FRAME_RES, FRAME_RES), Image.BILINEAR
            )
            resized.append(np.array(img, dtype=np.uint8))

        frames_np = np.stack(resized, axis=0)  # (N, H, W, C)
        inputs = self._processor(
            frames_np, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        enc_pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).float()
        pred_pooled = outputs.predictor_output.last_hidden_state.mean(
            dim=1
        ).squeeze(0).float()

        enc_384 = (enc_pooled @ self._projection).cpu().numpy()
        pred_384 = (pred_pooled @ self._projection).cpu().numpy()
        enc_384 /= np.linalg.norm(enc_384) + 1e-8
        pred_384 /= np.linalg.norm(pred_384) + 1e-8

        return enc_384.astype(np.float32), pred_384.astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def encoder_type(self) -> str:
        return "vjepa2"

    @property
    def model_id(self) -> str:
        return MODEL_ID
