from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from safetensors import safe_open
from transformers import AutoTokenizer

DEFAULT_GEMMA_MODEL_PATH = "/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit"
SEMANTIC_DIM = 2048



def _stable_projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    matrix /= np.sqrt(max(in_dim, 1))
    return matrix.astype(np.float32)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


def _unpack_affine_row(
    packed_row: "torch.Tensor",
    scales_row: "torch.Tensor",
    biases_row: "torch.Tensor",
) -> np.ndarray:
    import torch
    nibble_shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int64)
    packed = packed_row.to(torch.int64)
    values = ((packed.unsqueeze(-1) >> nibble_shifts) & 0xF).reshape(-1).float()
    scales = scales_row.float().unsqueeze(-1)
    biases = biases_row.float().unsqueeze(-1)
    restored = (values.reshape(-1, 64) * scales + biases).reshape(-1)
    return restored.cpu().numpy().astype(np.float32)


class GemmaSemanticExtractor:
    """
    Stable Gemma semantic text embedding extractor.

    This avoids `mlx_vlm` generation and instead reads the local quantized Gemma
    token embeddings directly from `model.safetensors`, dequantizes them on demand,
    and produces a deterministic 2048-d pooled semantic embedding.
    """

    def __init__(self, model_path: str | Path = DEFAULT_GEMMA_MODEL_PATH) -> None:
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.exists():
            raise FileNotFoundError(f"Gemma model path not found: {self._model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path), local_files_only=True)
        tensor_path = self._model_path / "model.safetensors"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Gemma safetensors not found: {tensor_path}")
        self._tensors = safe_open(str(tensor_path), framework="pt", device="cpu")
        self._embed_weight = self._tensors.get_tensor("language_model.model.embed_tokens.weight")
        self._embed_scales = self._tensors.get_tensor("language_model.model.embed_tokens.scales")
        self._embed_biases = self._tensors.get_tensor("language_model.model.embed_tokens.biases")
        self._per_weight = self._tensors.get_tensor("language_model.model.embed_tokens_per_layer.weight")
        self._per_scales = self._tensors.get_tensor("language_model.model.embed_tokens_per_layer.scales")
        self._per_biases = self._tensors.get_tensor("language_model.model.embed_tokens_per_layer.biases")
        self._base_dim = int(self._embed_weight.shape[1] * 8)
        self._per_dim = int(self._per_weight.shape[1] * 8)
        self._base_projection = _stable_projection_matrix(self._base_dim, SEMANTIC_DIM, seed=20260409)

    @property
    def model_path(self) -> str:
        return str(self._model_path)

    def encode_text(self, text: str) -> np.ndarray:
        token_ids = self._tokenize(text)
        if not token_ids:
            return np.zeros(SEMANTIC_DIM, dtype=np.float32)
        token_vectors = np.stack([self._semantic_token_vector(token_id) for token_id in token_ids], axis=0)
        pooled = token_vectors.mean(axis=0)
        return _l2_normalize(pooled)

    def encode_texts(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self.encode_text(text) for text in texts]

    def cosine_similarity(self, left: str, right: str) -> float:
        left_vec = self.encode_text(left)
        right_vec = self.encode_text(right)
        return float(np.dot(left_vec, right_vec))

    def _tokenize(self, text: str) -> list[int]:
        payload = self._tokenizer(
            str(text or "").strip(),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [int(token_id) for token_id in payload.get("input_ids", [])]

    @lru_cache(maxsize=8192)
    def _semantic_token_vector(self, token_id: int) -> np.ndarray:
        base = _unpack_affine_row(
            self._embed_weight[token_id],
            self._embed_scales[token_id],
            self._embed_biases[token_id],
        )
        per_layer = _unpack_affine_row(
            self._per_weight[token_id],
            self._per_scales[token_id],
            self._per_biases[token_id],
        )
        semantic = self._semantic_slice(base, per_layer)
        return _l2_normalize(semantic)

    def _semantic_slice(self, base: np.ndarray, per_layer: np.ndarray) -> np.ndarray:
        if self._per_dim >= self._base_dim + (4 * SEMANTIC_DIM):
            semantic_blocks = per_layer[self._base_dim : self._base_dim + (4 * SEMANTIC_DIM)].reshape(4, SEMANTIC_DIM)
            semantic_mean = semantic_blocks.mean(axis=0)
        elif self._per_dim >= SEMANTIC_DIM:
            semantic_mean = per_layer[:SEMANTIC_DIM]
        else:
            semantic_mean = np.zeros(SEMANTIC_DIM, dtype=np.float32)
        base_projected = np.asarray(base, dtype=np.float32) @ self._base_projection
        fused = (0.75 * semantic_mean) + (0.25 * base_projected)
        return fused.astype(np.float32)


@lru_cache(maxsize=1)
def get_default_gemma_semantic_extractor() -> GemmaSemanticExtractor | None:
    try:
        return GemmaSemanticExtractor()
    except Exception:
        return None
