"""
tvlc_connector.py — Toori Vision Language Connector.

Perceiver Resampler architecture connecting DINOv2-ViT-S/14 patch tokens
to Gemma 4 e4b token embedding space.

Architecture:
  Input:  patch_tokens [196, 384] — DINOv2 unit-norm patch embeddings
  Output: visual_tokens [32, 2048] — in Gemma 4 embedding space

  32 learned query vectors attend over 196 patch tokens via multi-head
  cross-attention. This compresses the visual representation to 32 tokens
  that Gemma 4 can attend to efficiently during narration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

N_QUERIES = 32
PATCH_DIM = 384
GEMMA_DIM = 2048
N_PATCHES = 196


N_HEADS = 8
HEAD_DIM = GEMMA_DIM // N_HEADS
SEED = 20260327


class TVLCConnector:
    """Perceiver Resampler: DINOv2 patches -> Gemma 4 visual tokens."""

    def __init__(self, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"TVLC weights not found: {path}. "
                f"Run: python3.11 scripts/train_tvlc.py --random-init"
            )
        data = np.load(str(path))
        self._queries = data["queries"].astype(np.float32)
        self._wq = data["wq"].astype(np.float32)
        self._wk = data["wk"].astype(np.float32)
        self._wv = data["wv"].astype(np.float32)
        self._wo = data["wo"].astype(np.float32)
        self._ln1_g = data["ln1_gamma"].astype(np.float32)
        self._ln1_b = data["ln1_beta"].astype(np.float32)
        self._ff1 = data["ff1"].astype(np.float32)
        self._ff2 = data["ff2"].astype(np.float32)
        self._ln2_g = data["ln2_gamma"].astype(np.float32)
        self._ln2_b = data["ln2_beta"].astype(np.float32)
        self._is_random_init = bool(data.get("random_init", np.array(False)))
        # Prototype-based Search (Fixed list - legacy fallback)
        prototype_labels = data.get("prototype_labels")
        prototype_vectors = data.get("prototype_vectors")
        self._prototype_labels = (
            np.asarray(prototype_labels, dtype=str).reshape(-1).tolist()
            if prototype_labels is not None
            else []
        )
        self._prototype_vectors = (
            np.asarray(prototype_vectors, dtype=np.float32)
            if prototype_vectors is not None
            else np.zeros((0, GEMMA_DIM), dtype=np.float32)
        )

        # Latent Manifold Search (Full Semantic Vocabulary - Zero-Shot Engine)
        self._latent_labels = []
        self._latent_vectors = np.zeros((0, GEMMA_DIM), dtype=np.float32)
        
        latent_path = Path("models/vision/latent_vocab.npz")
        if latent_path.exists():
            try:
                latent_data = np.load(str(latent_path))
                self._latent_labels = latent_data["labels"].tolist()
                self._latent_vectors = latent_data["vectors"].astype(np.float32)
            except Exception:
                pass

    @property
    def is_trained(self) -> bool:
        return not self._is_random_init

    @property
    def has_semantic_prototypes(self) -> bool:
        return bool(self._prototype_labels) and self._prototype_vectors.shape == (
            len(self._prototype_labels),
            GEMMA_DIM,
        )

    def _forward_with_attention(
        self, patch_tokens: np.ndarray, valid_patches: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shared forward pass returning both visual tokens and attention.

        Returns:
            visual_tokens: np.ndarray[32, 2048] — in Gemma 4 space
            attn_weights:  np.ndarray[8, 32, 196] — cross-attention weights
        """
        patches = np.asarray(patch_tokens, dtype=np.float32)
        if patches.shape != (N_PATCHES, PATCH_DIM):
            raise ValueError(
                f"TVLCConnector expects patch tokens with shape {(N_PATCHES, PATCH_DIM)}, "
                f"got {patches.shape}"
            )

        queries = self._queries.copy()

        q = queries @ self._wq
        k = patches @ self._wk
        v = patches @ self._wv

        q = q.reshape(N_QUERIES, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = k.reshape(N_PATCHES, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(N_PATCHES, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

        scale = HEAD_DIM ** -0.5
        attn = np.matmul(q, k.transpose(0, 2, 1)) * scale

        if valid_patches is not None:
            mask = np.ones(N_PATCHES, dtype=bool)
            # Ensure at least one valid patch to avoid NaN softmaxes
            valid = valid_patches if valid_patches else [0]
            mask[valid] = False
            attn[:, :, mask] = -1e9

        attn -= attn.max(axis=-1, keepdims=True)
        attn = np.exp(attn)
        attn /= attn.sum(axis=-1, keepdims=True) + 1e-9

        # Preserve attention weights before consuming them
        attn_weights = attn.copy()

        out = np.matmul(attn, v)
        out = out.transpose(1, 0, 2).reshape(N_QUERIES, GEMMA_DIM)
        out = out @ self._wo

        queries = queries + out

        mean = queries.mean(axis=-1, keepdims=True)
        std = queries.std(axis=-1, keepdims=True) + 1e-6
        queries = (queries - mean) / std
        queries = queries * self._ln1_g + self._ln1_b

        ff = queries @ self._ff1
        ff = ff * (1.0 / (1.0 + np.exp(-1.702 * ff)))
        ff = ff @ self._ff2
        queries = queries + ff

        mean = queries.mean(axis=-1, keepdims=True)
        std = queries.std(axis=-1, keepdims=True) + 1e-6
        queries = (queries - mean) / std
        queries = queries * self._ln2_g + self._ln2_b

        return queries.astype(np.float32), attn_weights

    def project(self, patch_tokens: np.ndarray) -> np.ndarray:
        """
        Project DINOv2 patch tokens to Gemma 4 visual tokens.

        Args:
            patch_tokens: np.ndarray[196, 384] — unit-norm DINOv2 patches
        Returns:
            np.ndarray[32, 2048] — visual tokens in Gemma 4 embedding space
        """
        visual_tokens, _attn = self._forward_with_attention(patch_tokens)
        return visual_tokens

    def score_anchor(
        self,
        patch_tokens: np.ndarray,
        patch_indices: list[int],
    ) -> tuple[str, float]:
        """Score patches against the full Latent Manifold (Gemma Vocabulary).

        This is an instantaneous zero-shot identification pass. It projects
        the visual patches into the Gemma 4 semantic space and finds the 
        statistically closest token (word) from the dictionary.

        Returns:
            (label, score) — best matching Gemma word and cosine similarity.
        """
        if not patch_indices:
            return "", 0.0
        patches = np.asarray(patch_tokens, dtype=np.float32)
        if patches.shape != (N_PATCHES, PATCH_DIM):
            return "", 0.0
        valid = [i for i in patch_indices if 0 <= i < N_PATCHES]
        if not valid:
            return "", 0.0

        try:
            # Shift patches through the Perceiver Resampler to get Visual Keywords
            visual_tokens, _attn_weights = self._forward_with_attention(patches, valid_patches=valid)
        except Exception:
            return "", 0.0

        # 1. PRIMARY: Full Latent Manifold Consensus (Zero-Shot)
        if len(self._latent_labels) > 0:
            latent_norms = np.linalg.norm(self._latent_vectors, axis=1) + 1e-9
            q_norms = np.linalg.norm(visual_tokens, axis=1) + 1e-9
            # Matrix is [vocab_size, 32]
            sim_matrix = (self._latent_vectors @ visual_tokens.T) / (latent_norms[:, None] * q_norms[None, :])
            
            slot_winners = np.argmax(sim_matrix, axis=0) # [32]
            slot_scores = np.max(sim_matrix, axis=0) # [32]
            
            counts = np.bincount(slot_winners, minlength=len(self._latent_labels))
            best_idx = int(np.argmax(counts))
            vote_count = counts[best_idx]
            winning_slots_mask = (slot_winners == best_idx)
            consensus_score = float(np.mean(slot_scores[winning_slots_mask]))
            
            # Elite High-Precision Gate: Require plurality and statistical alignment
            # A slot vote count >= 3 is statistically non-random for N=32.
            # In normalized Latent Projection space, > 0.35 is already significant.
            if vote_count >= 2 and consensus_score > 0.35:
                # Add human-friendly casing
                raw_label = str(self._latent_labels[best_idx])
                label = raw_label.replace("_", " ").strip()
                # Auto-capitalize if it looks like a proper name/entity
                if any(c.isupper() for c in raw_label):
                    return label, consensus_score
                return label.capitalize(), consensus_score

        # 2. FALLBACK: Prototype-based Search (Legacy Majority Consensus)
        if not self.has_semantic_prototypes:
            return "", 0.0

        proto_norms = np.linalg.norm(self._prototype_vectors, axis=1) + 1e-9
        q_norms = np.linalg.norm(visual_tokens, axis=1) + 1e-9
        sim_matrix = (self._prototype_vectors @ visual_tokens.T) / (proto_norms[:, None] * q_norms[None, :])
        
        slot_winners = np.argmax(sim_matrix, axis=0)
        slot_scores = np.max(sim_matrix, axis=0)
        counts = np.bincount(slot_winners, minlength=len(self._prototype_labels))
        best_proto_idx = int(np.argmax(counts))
        vote_count = counts[best_proto_idx]
        winning_slots_mask = (slot_winners == best_proto_idx)
        consensus_score = float(np.mean(slot_scores[winning_slots_mask]))
        
        max_slot_score = float(np.max(slot_scores))
        if max_slot_score > 0.85:
            return str(self._prototype_labels[best_proto_idx]), max_slot_score
        if vote_count < 2 or consensus_score < 0.45:
            return "", 0.0
            
        return str(self._prototype_labels[best_proto_idx]), consensus_score

    def to_gemma_context(self, patch_tokens: np.ndarray) -> str:
        """
        Project patches and encode as a compact context string for Gemma 4 prompt.
        """
        visual_tokens = self.project(patch_tokens)
        mean_token = visual_tokens.mean(axis=0)

        rng = np.random.RandomState(SEED)
        slot_proj = rng.randn(8, GEMMA_DIM).astype(np.float32)
        slot_proj /= np.linalg.norm(slot_proj, axis=1, keepdims=True) + 1e-9
        slots = slot_proj @ mean_token
        slots = (slots - slots.min()) / (slots.max() - slots.min() + 1e-9)

        connector_type = "tvlc-trained" if self.is_trained else "tvlc-random-init"
        slot_text = ",".join(f"{float(value):.3f}" for value in slots)
        visual_entropy = float(np.std(visual_tokens))
        prototype_text = ""
        if self.has_semantic_prototypes:
            prototypes = self._prototype_vectors
            proto_norms = np.linalg.norm(prototypes, axis=1) + 1e-9
            mean_norm = float(np.linalg.norm(mean_token)) + 1e-9
            similarities = (prototypes @ mean_token) / (proto_norms * mean_norm)
            if similarities.size:
                top_indices = np.argsort(similarities)[::-1][:3]
                matches = [
                    f"{self._prototype_labels[index]}:{float(similarities[index]):.2f}"
                    for index in top_indices
                    if similarities[index] > 0.0
                ]
                if matches:
                    prototype_text = f"prototype_matches={';'.join(matches)}; "
        return (
            f"[TVLC visual context ({connector_type}): "
            f"{prototype_text}slot_activations={slot_text}; visual_entropy={visual_entropy:.3f}]"
        )

    @classmethod
    def default_weights_path(cls) -> str:
        return "models/vision/tvlc_connector.npz"

    @classmethod
    def is_available(cls) -> bool:
        try:
            return Path(cls.default_weights_path()).exists()
        except Exception:
            return False

    @classmethod
    def create_random_init(cls, save_path: Optional[str] = None) -> "TVLCConnector":
        """Random-weight init with seed=20260327. Pipeline testing only."""
        rng = np.random.RandomState(SEED)

        def _scale(fan: int) -> float:
            return float(np.sqrt(2.0 / fan))

        weights = {
            "queries": (rng.randn(N_QUERIES, GEMMA_DIM) * _scale(GEMMA_DIM)).astype(np.float32),
            "wq": (rng.randn(GEMMA_DIM, GEMMA_DIM) * _scale(GEMMA_DIM)).astype(np.float32),
            "wk": (rng.randn(PATCH_DIM, GEMMA_DIM) * _scale(PATCH_DIM)).astype(np.float32),
            "wv": (rng.randn(PATCH_DIM, GEMMA_DIM) * _scale(PATCH_DIM)).astype(np.float32),
            "wo": (rng.randn(GEMMA_DIM, GEMMA_DIM) * _scale(GEMMA_DIM)).astype(np.float32),
            "ln1_gamma": np.ones(GEMMA_DIM, dtype=np.float32),
            "ln1_beta": np.zeros(GEMMA_DIM, dtype=np.float32),
            "ff1": (rng.randn(GEMMA_DIM, GEMMA_DIM * 4) * _scale(GEMMA_DIM)).astype(np.float32),
            "ff2": (rng.randn(GEMMA_DIM * 4, GEMMA_DIM) * _scale(GEMMA_DIM * 4)).astype(np.float32),
            "ln2_gamma": np.ones(GEMMA_DIM, dtype=np.float32),
            "ln2_beta": np.zeros(GEMMA_DIM, dtype=np.float32),
            "random_init": np.array(True),
        }
        path = save_path or cls.default_weights_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **weights)
        return cls(path)
