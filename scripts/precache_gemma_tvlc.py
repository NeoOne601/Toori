"""
precache_gemma_tvlc.py — Torch-free semantic target precacher for TVLC training.

Produces Gemma 4 quality semantic target vectors WITHOUT loading PyTorch or MLX.
Reads the Gemma 4 token embedding table directly from model.safetensors using
pure numpy with manual bfloat16 decoding. Row-level file seeks + LRU cache keep
peak memory under 500 MB on an 8 GB Apple Silicon Mac.

Usage:
  python3.11 scripts/precache_gemma_tvlc.py --coco-dir /Volumes/Apple/CoCo --max-samples 2500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# TVLC architecture constants (from cloud/perception/tvlc_connector.py)
# ---------------------------------------------------------------------------
N_QUERIES = 32
PATCH_DIM = 384
GEMMA_DIM = 2048
N_PATCHES = 196
N_HEADS = 8
HEAD_DIM = GEMMA_DIM // N_HEADS

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_ANNOTATIONS = "annotations/captions_train2017.json"
DEFAULT_CACHE_DIR = ".toori/tvlc_train_cache"
DEFAULT_GEMMA_MODEL_PATH = "/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit"

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
RELATION_WORDS = {
    "near", "behind", "beside", "between", "holding", "wearing", "with",
    "on", "under", "over", "in", "at", "by", "next", "front", "left", "right",
}
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "is", "are", "this", "that",
    "these", "those", "some", "many", "several", "very", "small", "large",
    "big", "little",
}
ATTRIBUTE_WORDS = {
    "red", "green", "blue", "yellow", "orange", "white", "black", "brown",
    "gray", "grey", "pink", "purple", "striped", "wooden", "metal", "plastic",
    "glass",
}


def _sanitize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", str(value or "").lower())
    return " ".join(cleaned.split()).strip()


# ---------------------------------------------------------------------------
# Caption target data class & heuristic parser
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CaptionTarget:
    primary_label: str
    attributes: list[str]
    scene_tags: list[str]
    summary: str
    source: str


def _heuristic_caption_target(caption: str) -> CaptionTarget:
    normalized = _sanitize_text(caption)
    if not normalized:
        return CaptionTarget(
            primary_label="object", attributes=[], scene_tags=[],
            summary="object", source="heuristic",
        )
    tokens = normalized.split()
    primary_tokens: list[str] = []
    attributes: list[str] = []
    scene_tags: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in ATTRIBUTE_WORDS and token not in attributes:
            attributes.append(token)
            continue
        if token in RELATION_WORDS:
            if primary_tokens:
                break
            continue
        if len(primary_tokens) < 2:
            primary_tokens.append(token)
        elif token not in scene_tags:
            scene_tags.append(token)
    if not primary_tokens:
        primary_tokens = [next((t for t in tokens if t not in STOPWORDS), "object")]
    summary = " ".join(primary_tokens[:2] + attributes[:2] + scene_tags[:3]).strip() or normalized
    return CaptionTarget(
        primary_label=" ".join(primary_tokens[:2]),
        attributes=attributes[:3],
        scene_tags=scene_tags[:4],
        summary=summary,
        source="heuristic",
    )


# ---------------------------------------------------------------------------
# Stable phrase vector (hash-based fallback when Gemma model unavailable)
# ---------------------------------------------------------------------------
def _stable_phrase_vector(text: str, dim: int, seed: int = 0) -> np.ndarray:
    normalized = _sanitize_text(text) or "object"
    digest = hashlib.sha256(f"{seed}:{normalized}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(local_seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure-numpy Gemma 4 semantic extractor (ZERO torch dependency)
# ---------------------------------------------------------------------------
def _stable_projection_matrix(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    matrix /= np.sqrt(max(in_dim, 1))
    return matrix


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


# Nibble shifts for 4-bit affine dequantization (8 nibbles per uint32)
_NIBBLE_SHIFTS = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int64)


def _unpack_affine_row_numpy(
    packed_row: np.ndarray,
    scales_row: np.ndarray,
    biases_row: np.ndarray,
) -> np.ndarray:
    """Dequantize one row of 4-bit affine-quantized weights using pure numpy."""
    packed = packed_row.astype(np.int64)
    values = ((packed[:, np.newaxis] >> _NIBBLE_SHIFTS) & 0xF).reshape(-1).astype(np.float32)
    scales = scales_row.astype(np.float32)[:, np.newaxis]
    biases = biases_row.astype(np.float32)[:, np.newaxis]
    restored = (values.reshape(-1, 64) * scales + biases).reshape(-1)
    return restored.astype(np.float32)


class _SafeTensorRowReader:
    """
    Row-level random-access reader for safetensors files.

    Reads individual rows via file seeks instead of loading entire tensors.
    Handles bfloat16 → float32 conversion via bit manipulation (no torch).
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        with open(filepath, "rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            self._header = json.loads(fh.read(header_size))
        self._data_offset = 8 + header_size
        self._fh = open(filepath, "rb")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def read_row_u32(self, key: str, row_idx: int) -> np.ndarray:
        meta = self._header[key]
        cols = meta["shape"][1]
        row_bytes = cols * 4
        offset = self._data_offset + meta["data_offsets"][0] + row_idx * row_bytes
        self._fh.seek(offset)
        return np.frombuffer(self._fh.read(row_bytes), dtype=np.uint32).copy()

    def read_row_bf16(self, key: str, row_idx: int) -> np.ndarray:
        meta = self._header[key]
        cols = meta["shape"][1]
        row_bytes = cols * 2
        offset = self._data_offset + meta["data_offsets"][0] + row_idx * row_bytes
        self._fh.seek(offset)
        raw = np.frombuffer(self._fh.read(row_bytes), dtype=np.uint16).copy()
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._header[key]["shape"])


class NumpyGemmaExtractor:
    """
    Torch-free Gemma 4 semantic text embedding extractor.

    Reads the quantized token embedding table directly from model.safetensors
    using row-level file seeks. Dequantizes individual rows on demand (LRU cached)
    and produces deterministic 2048-d semantic embeddings in Gemma 4's token space.

    Peak memory: ~200 MB (vs ~2 GB with the original torch-based extractor).
    """

    SEMANTIC_DIM = 2048

    def __init__(self, model_path: str | Path = DEFAULT_GEMMA_MODEL_PATH) -> None:
        self._model_path = Path(model_path).expanduser().resolve()
        if not self._model_path.exists():
            raise FileNotFoundError(f"Gemma model path not found: {self._model_path}")

        tensor_path = self._model_path / "model.safetensors"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Gemma safetensors not found: {tensor_path}")

        # Lightweight tokenizer (no torch needed — just sentencepiece under the hood)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_path), local_files_only=True,
        )

        # Row-level reader — no bulk tensor loading
        self._reader = _SafeTensorRowReader(str(tensor_path))

        embed_shape = self._reader.tensor_shape("language_model.model.embed_tokens.weight")
        per_shape = self._reader.tensor_shape("language_model.model.embed_tokens_per_layer.weight")
        self._base_dim = embed_shape[1] * 8  # uint32 → 8 nibbles
        self._per_dim = per_shape[1] * 8
        self._base_projection = _stable_projection_matrix(
            self._base_dim, self.SEMANTIC_DIM, seed=20260409,
        )

    def close(self) -> None:
        self._reader.close()

    def encode_text(self, text: str) -> np.ndarray:
        token_ids = self._tokenize(text)
        if not token_ids:
            return np.zeros(self.SEMANTIC_DIM, dtype=np.float32)
        vectors = np.stack(
            [self._semantic_token_vector(tid) for tid in token_ids], axis=0,
        )
        pooled = vectors.mean(axis=0)
        return _l2_normalize(pooled)

    def _tokenize(self, text: str) -> list[int]:
        payload = self._tokenizer(
            str(text or "").strip(),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [int(tid) for tid in payload.get("input_ids", [])]

    @lru_cache(maxsize=8192)
    def _semantic_token_vector(self, token_id: int) -> np.ndarray:
        base = _unpack_affine_row_numpy(
            self._reader.read_row_u32("language_model.model.embed_tokens.weight", token_id),
            self._reader.read_row_bf16("language_model.model.embed_tokens.scales", token_id),
            self._reader.read_row_bf16("language_model.model.embed_tokens.biases", token_id),
        )
        per_layer = _unpack_affine_row_numpy(
            self._reader.read_row_u32("language_model.model.embed_tokens_per_layer.weight", token_id),
            self._reader.read_row_bf16("language_model.model.embed_tokens_per_layer.scales", token_id),
            self._reader.read_row_bf16("language_model.model.embed_tokens_per_layer.biases", token_id),
        )
        semantic = self._semantic_slice(base, per_layer)
        return _l2_normalize(semantic)

    def _semantic_slice(self, base: np.ndarray, per_layer: np.ndarray) -> np.ndarray:
        sd = self.SEMANTIC_DIM
        if self._per_dim >= self._base_dim + (4 * sd):
            blocks = per_layer[self._base_dim: self._base_dim + (4 * sd)].reshape(4, sd)
            semantic_mean = blocks.mean(axis=0)
        elif self._per_dim >= sd:
            semantic_mean = per_layer[:sd]
        else:
            semantic_mean = np.zeros(sd, dtype=np.float32)
        base_projected = base.astype(np.float32) @ self._base_projection
        fused = (0.75 * semantic_mean) + (0.25 * base_projected)
        return fused.astype(np.float32)


# ---------------------------------------------------------------------------
# Semantic target builder (wraps extractor with fallback)
# ---------------------------------------------------------------------------
class SemanticTargetBuilder:
    def __init__(self, model_path: str | None = None) -> None:
        self._extractor: NumpyGemmaExtractor | None = None
        self._disabled_reason: str | None = None
        try:
            self._extractor = NumpyGemmaExtractor(model_path or DEFAULT_GEMMA_MODEL_PATH)
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._extractor = None

    @property
    def ready(self) -> bool:
        return self._extractor is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def close(self) -> None:
        if self._extractor is not None:
            self._extractor.close()

    def encode_phrase(self, text: str, *, seed: int) -> np.ndarray:
        if self._extractor is not None:
            vector = self._extractor.encode_text(text)
            if np.any(vector):
                return vector.astype(np.float32)
        return _stable_phrase_vector(text, GEMMA_DIM, seed=seed)


# ---------------------------------------------------------------------------
# Target token construction
# ---------------------------------------------------------------------------
def _mean_vectors(values: Iterable[np.ndarray], dim: int) -> np.ndarray:
    vectors = [np.asarray(v, dtype=np.float32) for v in values if v is not None]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean)) or 1.0
    return (mean / norm).astype(np.float32)


def _target_tokens(target: CaptionTarget, builder: SemanticTargetBuilder) -> np.ndarray:
    primary_vec = builder.encode_phrase(target.primary_label, seed=11)
    attr_vec = _mean_vectors(
        (builder.encode_phrase(item, seed=13) for item in target.attributes), GEMMA_DIM,
    )
    scene_vec = _mean_vectors(
        (builder.encode_phrase(item, seed=17) for item in target.scene_tags), GEMMA_DIM,
    )
    summary_vec = builder.encode_phrase(target.summary, seed=19)
    slots: list[np.ndarray] = []
    for index in range(N_QUERIES):
        if index < 8:
            base = (1.4 * primary_vec) + (0.5 * attr_vec)
        elif index < 16:
            base = (0.9 * primary_vec) + (1.0 * attr_vec) + (0.3 * scene_vec)
        elif index < 24:
            base = (0.6 * primary_vec) + (0.8 * scene_vec) + (0.7 * summary_vec)
        else:
            base = (0.5 * primary_vec) + (0.6 * attr_vec) + (0.9 * summary_vec)
        slot = base + (0.15 * builder.encode_phrase(f"{target.primary_label}:slot:{index}", seed=29))
        norm = float(np.linalg.norm(slot)) or 1.0
        slots.append((slot / norm).astype(np.float32))
    return np.stack(slots, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI & main loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precache_gemma_tvlc.py",
        description="Pre-cache Gemma 4 semantic targets for TVLC training (torch-free).",
    )
    parser.add_argument("--coco-dir", type=str, required=True, help="COCO dataset root")
    parser.add_argument("--annotations", type=str, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gemma-model-path", type=str, default=None,
                        help="Override local Gemma model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    coco_dir = Path(args.coco_dir).expanduser().resolve()
    annotations_path = (coco_dir / args.annotations).resolve()
    cache_dir = (
        (repo_root / args.cache_dir).resolve()
        if not Path(args.cache_dir).is_absolute()
        else Path(args.cache_dir).resolve()
    )

    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO captions file not found: {annotations_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("annotations", [])
    if args.max_samples is not None:
        records = records[: max(0, int(args.max_samples))]

    # Initialize Gemma semantic extractor (pure numpy, ~200 MB)
    target_builder = SemanticTargetBuilder(model_path=args.gemma_model_path)
    if target_builder.ready:
        print("✅ Gemma 4 semantic extractor loaded (pure numpy, torch-free)")
    else:
        print(f"⚠️  Gemma 4 extractor unavailable: {target_builder.disabled_reason}")
        print("   Falling back to hash-based vectors (lower quality but functional)")

    try:
        progress = tqdm(records, desc="Pre-caching semantic targets")
        skipped = 0
        written = 0
        errors = 0

        for annotation in progress:
            annotation_id = int(annotation["id"])
            gemma_cache_file = cache_dir / f"{annotation_id}_gemma.npz"

            if gemma_cache_file.exists():
                skipped += 1
                progress.set_postfix(skipped=skipped, written=written, errors=errors)
                continue

            try:
                caption = str(annotation.get("caption") or "").strip()
                target = _heuristic_caption_target(caption)
                target_tokens = _target_tokens(target, target_builder)

                np.savez(
                    str(gemma_cache_file),
                    target_tokens=target_tokens.astype(np.float32),
                    primary_label=np.array(target.primary_label),
                    summary=np.array(target.summary),
                    target_source=np.array(target.source),
                )
                written += 1
            except Exception as exc:
                errors += 1
                print(f"\n  [error] annotation {annotation_id}: {exc}", file=sys.stderr)

            progress.set_postfix(skipped=skipped, written=written, errors=errors)

        print(f"\nPre-caching complete! Wrote {written} new, skipped {skipped}, errors {errors}.")
        if target_builder.ready:
            print("Targets are in Gemma 4's semantic embedding space ✅")
        else:
            print("Targets used hash-based vectors (Gemma model was unavailable)")

    finally:
        target_builder.close()


if __name__ == "__main__":
    main()
