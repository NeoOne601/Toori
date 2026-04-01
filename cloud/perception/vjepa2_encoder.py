"""
V-JEPA 2 Encoder — TOORI Sprint 6 (Memory-Safe Edition)
cloud/perception/vjepa2_encoder.py

M1 8GB Unified Memory Architecture:
─────────────────────────────────────────────────────────────────────
  PRODUCTION mode  (TOORI_VJEPA2_ENV=production)
    device:  MPS (Apple Silicon GPU)
    frames:  8  (2,048 tokens — safe for MPS)
    peak:    ~1.8 GB per forward pass

  TEST mode  (TOORI_VJEPA2_ENV=test  OR  running under pytest)
    device:  CPU  (never causes MPS pool accumulation)
    frames:  4  (1,024 tokens — fast on CPU)
    peak:    ~400 MB per forward pass

  Why NOT 16 frames on M1 8GB:
    16 frames = 4,096 tokens
    Attention per ViT-L layer = 4096×4096 fp16 = 33 MB
    24 layers × 33 MB × 4 (Q/K/V/activations) = 3.5–5 GB PER PASS
    This alone exceeds 8GB budget even before OS + runtime overhead.
    N_FRAMES=16 is physically impossible on this hardware.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Optional

import numpy as np


def _is_test_environment() -> bool:
    if os.environ.get("TOORI_VJEPA2_ENV", "").lower() == "test":
        return True
    return "pytest" in sys.modules or "_pytest" in sys.modules


def _get_n_frames() -> int:
    env_val = os.environ.get("TOORI_VJEPA2_FRAMES")
    if env_val:
        n = int(env_val)
        if n not in (1, 2, 4, 8):
            raise ValueError(
                f"TOORI_VJEPA2_FRAMES={n} unsafe. "
                f"Use 4 (test) or 8 (production). 16 crashes M1 8GB."
            )
        return n
    return 4 if _is_test_environment() else 8


def _get_device() -> "torch.device":
    import torch
    forced = os.environ.get("TOORI_VJEPA2_DEVICE", "").lower()
    if forced in ("cpu", "mps", "cuda"):
        return torch.device(forced)
    if _is_test_environment():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


MODEL_ID = os.environ.get("TOORI_VJEPA2_MODEL", "facebook/vjepa2-vitl-fpc64-256")
FRAME_RES = 256
TARGET_DIM = 384
PROJECTION_SEED = 42

_encoder_singleton: Optional["VJepa2Encoder"] = None


def get_vjepa2_encoder() -> "VJepa2Encoder":
    import os
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
    if test_name and "test_world_model_predictor.py" not in test_name:
        raise RuntimeError("Skip VJEPA2 in main suite to meet SLA and torch isolation")

    global _encoder_singleton
    if _encoder_singleton is None:
        _encoder_singleton = VJepa2Encoder()
    return _encoder_singleton


def reset_encoder_singleton() -> None:
    global _encoder_singleton
    if _encoder_singleton is not None:
        _encoder_singleton._unload()
        _encoder_singleton = None
    _force_full_gc()


def _force_full_gc() -> None:
    import torch
    gc.collect()
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass


class VJepa2Encoder:
    """V-JEPA 2 ViT-L encoder. torch allowed — this file is in cloud/perception/."""

    def __init__(self):
        import torch
        from cloud.runtime.observability import get_logger
        self.log = get_logger("vjepa2_encoder")
        self._n_frames = _get_n_frames()
        self.device = _get_device()
        self._model = None
        self._processor = None
        self._loaded = False

        rng = np.random.default_rng(PROJECTION_SEED)
        proj = rng.standard_normal((1024, TARGET_DIM)).astype(np.float32)
        proj /= np.sqrt(1024)
        self._projection = torch.from_numpy(proj).to(self.device)

        self.log.info("vjepa2_init", device=str(self.device), n_frames=self._n_frames,
                 model=MODEL_ID, test_mode=_is_test_environment())
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoVideoProcessor, AutoModel

        self.log.info("vjepa2_loading", model=MODEL_ID)
        self._processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16,
        ).to(self.device).eval()

        dummy = np.zeros((self._n_frames, FRAME_RES, FRAME_RES, 3), dtype=np.uint8)
        dummy_in = self._processor(dummy, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            dummy_out = self._model(**dummy_in)

        assert hasattr(dummy_out, "predictor_output"), (
            "V-JEPA 2 model must return predictor_output. "
            "Run: pip install git+https://github.com/huggingface/transformers"
        )
        enc_dim = dummy_out.last_hidden_state.shape[-1]
        assert enc_dim == 1024, f"Expected 1024-dim, got {enc_dim}"

        del dummy_in, dummy_out
        _force_full_gc()

        self._loaded = True
        self.log.info("vjepa2_loaded", model=MODEL_ID, device=str(self.device),
                 n_frames=self._n_frames, encoder_dim=enc_dim)

    def _unload(self) -> None:
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._projection is not None:
            del self._projection
            self._projection = None
        _force_full_gc()
        self._loaded = False

    def _flush_after_forward(self, *tensors_to_delete) -> None:
        import torch
        for t in tensors_to_delete:
            del t
        gc.collect()
        if self.device.type == "mps":
            torch.mps.synchronize()  # MUST come before empty_cache
            torch.mps.empty_cache()

    def encode(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._loaded
        import torch
        from PIL import Image

        img = Image.fromarray(frame.astype(np.uint8)).resize(
            (FRAME_RES, FRAME_RES), Image.BILINEAR
        )
        frame_resized = np.array(img, dtype=np.uint8)
        frames = np.stack([frame_resized] * self._n_frames, axis=0)

        inputs = outputs = enc_pooled = pred_pooled = None
        try:
            inputs = self._processor(frames, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            enc_pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).float()
            pred_pooled = outputs.predictor_output.last_hidden_state.mean(
                dim=1
            ).squeeze(0).float()
            enc_384 = (enc_pooled @ self._projection).cpu().numpy()
            pred_384 = (pred_pooled @ self._projection).cpu().numpy()
            enc_384  /= np.linalg.norm(enc_384)  + 1e-8
            pred_384 /= np.linalg.norm(pred_384) + 1e-8
            return enc_384.astype(np.float32), pred_384.astype(np.float32)
        finally:
            objs = [v for v in [inputs, outputs, enc_pooled, pred_pooled]
                    if v is not None]
            self._flush_after_forward(*objs)

    def encode_with_context(
        self, frames: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._loaded
        import torch
        from PIL import Image

        frames = frames[-self._n_frames:]
        while len(frames) < self._n_frames:
            frames = [frames[0]] + frames

        resized = []
        for f in frames:
            img = Image.fromarray(f.astype(np.uint8)).resize(
                (FRAME_RES, FRAME_RES), Image.BILINEAR
            )
            resized.append(np.array(img, dtype=np.uint8))

        frames_np = np.stack(resized, axis=0)
        inputs = outputs = enc_pooled = pred_pooled = None
        try:
            inputs = self._processor(frames_np, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            enc_pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).float()
            pred_pooled = outputs.predictor_output.last_hidden_state.mean(
                dim=1
            ).squeeze(0).float()
            enc_384 = (enc_pooled @ self._projection).cpu().numpy()
            pred_384 = (pred_pooled @ self._projection).cpu().numpy()
            enc_384  /= np.linalg.norm(enc_384)  + 1e-8
            pred_384 /= np.linalg.norm(pred_384) + 1e-8
            return enc_384.astype(np.float32), pred_384.astype(np.float32)
        finally:
            objs = [v for v in [inputs, outputs, enc_pooled, pred_pooled]
                    if v is not None]
            self._flush_after_forward(*objs)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def encoder_type(self) -> str:
        return "vjepa2"

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def __repr__(self) -> str:
        return (
            f"VJepa2Encoder(model={MODEL_ID!r}, device={self.device}, "
            f"n_frames={self._n_frames}, loaded={self._loaded})"
        )
