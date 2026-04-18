import os
import json
import struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Reusing the optimized dequantization logic from precache_gemma_tvlc.py
# ---------------------------------------------------------------------------

_NIBBLE_SHIFTS = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int64)

def _unpack_affine_row_numpy(packed_row, scales_row, biases_row):
    packed = packed_row.astype(np.int64)
    values = ((packed[:, np.newaxis] >> _NIBBLE_SHIFTS) & 0xF).reshape(-1).astype(np.float32)
    scales = scales_row.astype(np.float32)[:, np.newaxis]
    biases = biases_row.astype(np.float32)[:, np.newaxis]
    restored = (values.reshape(-1, 64) * scales + biases).reshape(-1)
    return restored.astype(np.float32)

class SafeTensorRowReader:
    def __init__(self, filepath):
        self._filepath = filepath
        with open(filepath, "rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            self._header = json.loads(fh.read(header_size))
        self._data_offset = 8 + header_size
        self._fh = open(filepath, "rb")

    def read_row_u32(self, key, row_idx):
        meta = self._header[key]
        cols = meta["shape"][1]
        row_bytes = cols * 4
        offset = self._data_offset + meta["data_offsets"][0] + row_idx * row_bytes
        self._fh.seek(offset)
        return np.frombuffer(self._fh.read(row_bytes), dtype=np.uint32).copy()

    def read_row_bf16(self, key, row_idx):
        meta = self._header[key]
        cols = meta["shape"][1]
        row_bytes = cols * 2
        offset = self._data_offset + meta["data_offsets"][0] + row_idx * row_bytes
        self._fh.seek(offset)
        raw = np.frombuffer(self._fh.read(row_bytes), dtype=np.uint16).copy()
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def tensor_shape(self, key):
        return tuple(self._header[key]["shape"])

def _stable_projection_matrix(input_dim, output_dim, seed=20260409):
    rng = np.random.RandomState(seed)
    proj = rng.randn(input_dim, output_dim).astype(np.float32)
    return proj / (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-9)

def _l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9) if norm > 1e-9 else v

# ---------------------------------------------------------------------------
# Main extraction logic - Physical Noun Focus
# ---------------------------------------------------------------------------

def build_latent_vocab(model_path, save_path):
    print(f"[*] Analyzing Gemma 4 vocabulary from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tensor_path = Path(model_path) / "model.safetensors"
    reader = SafeTensorRowReader(str(tensor_path))

    embed_shape = reader.tensor_shape("language_model.model.embed_tokens.weight")
    per_shape = reader.tensor_shape("language_model.model.embed_tokens_per_layer.weight")
    base_dim = embed_shape[1] * 8
    per_dim = per_shape[1] * 8
    SEMANTIC_DIM = 2048
    base_projection = _stable_projection_matrix(base_dim, SEMANTIC_DIM)

    # Load high-quality physical nouns (the detox list)
    seeds_path = Path("scripts/object_seeds.txt")
    target_nouns = set()
    if seeds_path.exists():
        with open(seeds_path, "r") as f:
            content = f.read().replace('\n', ',')
            words = [w.strip() for w in content.split(',') if w.strip()]
            target_nouns.update(words)
            target_nouns.update([w.capitalize() for w in words])
    
    # We always ensure basic tracking categories exist
    target_nouns.update(["Face", "Human", "Person", "Animal", "Object", "Vehicle"])

    print(f"[*] Extracting specific targeted physical nouns (Seed size: {len(target_nouns)})...")
    
    vocab = tokenizer.get_vocab()
    # For matching, we want full BPE tokens that match our nouns.
    # Gemma's tokenizer prepends ' ' to the start of words
    
    # Pre-calculate target tokens to find matching token_ids
    target_tokens = []
    
    for noun in target_nouns:
        # Get the token ID for the noun with a space prefix (indicates start of a word)
        token_id = vocab.get(" " + noun)
        if token_id is not None:
             target_tokens.append((noun, token_id))
        else:
             # Try without space if it's a subword
             token_id = vocab.get(noun)
             if token_id is not None:
                 target_tokens.append((noun, token_id))

    # De-duplicate token IDs while keeping the best label
    unique_tokens = {}
    for label, tid in target_tokens:
        if tid not in unique_tokens or len(label) > len(unique_tokens[tid]):
            unique_tokens[tid] = label

    selected_labels = []
    selected_vectors = []

    count = 0
    for token_id, label in unique_tokens.items():
        base = _unpack_affine_row_numpy(
            reader.read_row_u32("language_model.model.embed_tokens.weight", token_id),
            reader.read_row_bf16("language_model.model.embed_tokens.scales", token_id),
            reader.read_row_bf16("language_model.model.embed_tokens.biases", token_id),
        )
        per_layer = _unpack_affine_row_numpy(
            reader.read_row_u32("language_model.model.embed_tokens_per_layer.weight", token_id),
            reader.read_row_bf16("language_model.model.embed_tokens_per_layer.scales", token_id),
            reader.read_row_bf16("language_model.model.embed_tokens_per_layer.biases", token_id),
        )
        
        if per_dim >= base_dim + (4 * SEMANTIC_DIM):
            blocks = per_layer[base_dim: base_dim + (4 * SEMANTIC_DIM)].reshape(4, SEMANTIC_DIM)
            semantic_mean = blocks.mean(axis=0)
        elif per_dim >= SEMANTIC_DIM:
            semantic_mean = per_layer[:SEMANTIC_DIM]
        else:
            semantic_mean = np.zeros(SEMANTIC_DIM, dtype=np.float32)
            
        base_projected = base.astype(np.float32) @ base_projection
        fused = (0.75 * semantic_mean) + (0.25 * base_projected)
        vector = _l2_normalize(fused)
        
        selected_labels.append(label)
        selected_vectors.append(vector)
        count += 1

    np.savez_compressed(
        save_path,
        labels=np.array(selected_labels, dtype=object),
        vectors=np.array(selected_vectors, dtype=np.float32)
    )
    print(f"[!] Saved latent manifold with {count} high-fidelity noun tokens to {save_path}")

if __name__ == "__main__":
    MODEL_PATH = "/Volumes/Apple/AI Model/gemma-4-e4b-it-4bit"
    SAVE_PATH = "models/vision/latent_vocab.npz"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    build_latent_vocab(MODEL_PATH, SAVE_PATH)
