from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cloud.perception.dinov2_encoder import Dinov2Encoder
from cloud.perception.tvlc_connector import (
    GEMMA_DIM,
    HEAD_DIM,
    N_HEADS,
    N_PATCHES,
    N_QUERIES,
    PATCH_DIM,
    SEED,
    TVLCConnector,
)
from cloud.runtime.config import default_settings
from cloud.runtime.models import ProviderConfig
from cloud.runtime.providers import MlxReasoningProvider

_MULTIWORD_OBJECTS = (
    "neck brace",
    "wall clock",
    "lab coat",
    "light switch",
    "desk lamp",
    "office chair",
    "table lamp",
    "sunglasses",
    "orange telescope",
    "telescope",
    "poster",
    "switch",
    "chair",
    "lamp",
    "tie",
)

_COLOR_WORDS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "violet",
    "purple",
    "pink",
    "white",
    "black",
    "gray",
    "grey",
    "brown",
    "gold",
    "silver",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "to",
    "with",
}

_RELATION_WORDS = {"near", "behind", "left", "right", "above", "below", "next"}

_TEACHER_SYSTEM = (
    "You convert captions into strict object evidence for a geometry-bounded vision system. "
    "Return ONLY valid JSON with keys objects, supporting_tokens, canonical_caption. "
    "objects: 1-5 atomic visible object labels. "
    "supporting_tokens: short non-relational cues such as color or material when useful. "
    "canonical_caption: a short relation-free caption grounded in those objects. "
    "Do not emit phrases containing near, behind, left of, right of. "
    "Do not emit descriptor jargon like histogram, texture, dominant color, edge map."
)


@dataclass(slots=True)
class CaptionSemantics:
    objects: list[str]
    supporting_tokens: list[str]
    canonical_caption: str
    source: str


@dataclass(slots=True)
class TVLCSample:
    image_path: Path
    raw_caption: str
    semantics: CaptionSemantics
    patch_tokens: np.ndarray
    target_vector: np.ndarray


def _normalize_phrase(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 +_-]+", " ", str(value or "").strip().lower())
    return " ".join(cleaned.replace("_", " ").split())


def _tokenize(value: str) -> list[str]:
    return [token for token in _normalize_phrase(value).split() if token]


def _fallback_semantics(caption: str) -> CaptionSemantics:
    text = _normalize_phrase(caption)
    objects: list[str] = []
    supporting: list[str] = []
    for phrase in _MULTIWORD_OBJECTS:
        if phrase in text and phrase not in objects:
            objects.append(phrase)
    for token in _tokenize(text):
        if token in _STOPWORDS or token in _RELATION_WORDS:
            continue
        if token in _COLOR_WORDS:
            if token not in supporting:
                supporting.append(token)
            continue
        if len(token) <= 2:
            continue
        if token not in objects and token not in supporting:
            objects.append(token)
        if len(objects) >= 4:
            break
    if not objects:
        objects = ["object"]
    return CaptionSemantics(
        objects=objects[:5],
        supporting_tokens=supporting[:5],
        canonical_caption=", ".join(objects[:3]),
        source="fallback_caption_normalizer",
    )


def _coerce_teacher_payload(payload: Any, raw_caption: str) -> CaptionSemantics:
    if not isinstance(payload, dict):
        return _fallback_semantics(raw_caption)
    objects = [_normalize_phrase(item) for item in payload.get("objects", []) if _normalize_phrase(item)]
    objects = [
        item
        for item in objects
        if item not in _RELATION_WORDS and " near " not in item and " behind " not in item
    ]
    supporting = [_normalize_phrase(item) for item in payload.get("supporting_tokens", []) if _normalize_phrase(item)]
    supporting = [item for item in supporting if "histogram" not in item and "descriptor" not in item]
    canonical_caption = _normalize_phrase(payload.get("canonical_caption", ""))
    if not objects:
        return _fallback_semantics(raw_caption)
    if not canonical_caption:
        canonical_caption = ", ".join(objects[:3])
    return CaptionSemantics(
        objects=objects[:5],
        supporting_tokens=supporting[:5],
        canonical_caption=canonical_caption,
        source="gemma4_teacher",
    )


def _semantic_hash_vector(semantics: CaptionSemantics, raw_caption: str, dim: int = GEMMA_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)

    def _accumulate(key: str, weight: float) -> None:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        for offset in range(4):
            idx = int.from_bytes(digest[offset * 2 : offset * 2 + 2], "little") % dim
            sign = 1.0 if (digest[8 + offset] % 2) == 0 else -1.0
            vector[idx] += weight * sign

    for obj in semantics.objects:
        tokens = _tokenize(obj)
        _accumulate(f"obj:{obj}", 1.0)
        for token in tokens:
            _accumulate(f"objtok:{token}", 0.75)
        for left, right in zip(tokens, tokens[1:]):
            _accumulate(f"objbigram:{left}_{right}", 0.55)

    for token in semantics.supporting_tokens:
        _accumulate(f"support:{token}", 0.35)

    for token in _tokenize(raw_caption):
        if token in _STOPWORDS or token in _RELATION_WORDS:
            continue
        _accumulate(f"caption:{token}", 0.12)

    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


class GemmaCaptionTeacher:
    def __init__(self, *, enabled: bool, config: ProviderConfig | None, cache_path: Path) -> None:
        self._enabled = enabled and config is not None
        self._config = config
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._provider = MlxReasoningProvider() if self._enabled else None
        self._healthy = False
        self._load_cache()
        if self._provider is not None and self._config is not None:
            try:
                self._healthy = bool(self._provider.health(self._config).healthy)
            except Exception:
                self._healthy = False

    @classmethod
    def from_defaults(
        cls,
        *,
        enabled: bool,
        cache_path: Path,
        mlx_command: str | None = None,
    ) -> "GemmaCaptionTeacher":
        settings = default_settings()
        config = settings.providers.get("mlx")
        if config is not None:
            config.enabled = bool(enabled and bool(config.model_path))
            if mlx_command:
                config.metadata["command"] = mlx_command
        return cls(enabled=enabled, config=config, cache_path=cache_path)

    @property
    def available(self) -> bool:
        return bool(self._healthy and self._provider is not None and self._config is not None)

    def close(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()
        self._save_cache()

    def canonicalize(self, caption: str) -> CaptionSemantics:
        normalized_caption = " ".join(str(caption or "").split())
        cached = self._cache.get(normalized_caption)
        if isinstance(cached, dict):
            return _coerce_teacher_payload(cached, normalized_caption)
        if not self.available:
            semantics = _fallback_semantics(normalized_caption)
            self._cache[normalized_caption] = {
                "objects": semantics.objects,
                "supporting_tokens": semantics.supporting_tokens,
                "canonical_caption": semantics.canonical_caption,
                "source": semantics.source,
            }
            return semantics
        prompt = f'Caption: "{normalized_caption}"\nReturn JSON only.'
        try:
            result = self._provider._send_receive(  # type: ignore[attr-defined]
                self._config,
                {"prompt": prompt, "system": _TEACHER_SYSTEM, "max_tokens": 96},
                timeout_s=float(self._config.timeout_s),
            )
            raw_text = str(result.get("text") or "").strip()
            payload = json.loads(re.sub(r"```(?:json)?|```", "", raw_text).strip())
            semantics = _coerce_teacher_payload(payload, normalized_caption)
        except Exception:
            semantics = _fallback_semantics(normalized_caption)
        self._cache[normalized_caption] = {
            "objects": semantics.objects,
            "supporting_tokens": semantics.supporting_tokens,
            "canonical_caption": semantics.canonical_caption,
            "source": semantics.source,
        }
        return semantics

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            payload = json.loads(self._cache_path.read_text())
            if isinstance(payload, dict):
                self._cache = {str(key): value for key, value in payload.items() if isinstance(value, dict)}
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))


class TorchTVLCConnector(nn.Module):
    def __init__(self, init_path: str | None = None, *, train_all: bool = False) -> None:
        super().__init__()
        if init_path and Path(init_path).exists():
            data = np.load(init_path)
        else:
            random_path = Path("/tmp/tvlc_random_init_training.npz")
            TVLCConnector.create_random_init(str(random_path))
            data = np.load(str(random_path))

        self.queries = nn.Parameter(torch.from_numpy(data["queries"].astype(np.float32)))
        self.wq = nn.Parameter(torch.from_numpy(data["wq"].astype(np.float32)), requires_grad=train_all)
        self.wk = nn.Parameter(torch.from_numpy(data["wk"].astype(np.float32)))
        self.wv = nn.Parameter(torch.from_numpy(data["wv"].astype(np.float32)))
        self.wo = nn.Parameter(torch.from_numpy(data["wo"].astype(np.float32)), requires_grad=train_all)
        self.ln1_gamma = nn.Parameter(torch.from_numpy(data["ln1_gamma"].astype(np.float32)))
        self.ln1_beta = nn.Parameter(torch.from_numpy(data["ln1_beta"].astype(np.float32)))
        self.ff1 = nn.Parameter(torch.from_numpy(data["ff1"].astype(np.float32)), requires_grad=train_all)
        self.ff2 = nn.Parameter(torch.from_numpy(data["ff2"].astype(np.float32)), requires_grad=train_all)
        self.ln2_gamma = nn.Parameter(torch.from_numpy(data["ln2_gamma"].astype(np.float32)))
        self.ln2_beta = nn.Parameter(torch.from_numpy(data["ln2_beta"].astype(np.float32)))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = patch_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        q = torch.matmul(queries, self.wq)
        k = torch.matmul(patch_tokens, self.wk)
        v = torch.matmul(patch_tokens, self.wv)

        q = q.view(batch_size, N_QUERIES, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = k.view(batch_size, N_PATCHES, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = v.view(batch_size, N_PATCHES, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch_size, N_QUERIES, GEMMA_DIM)
        queries = queries + torch.matmul(out, self.wo)

        queries = F.layer_norm(queries, (GEMMA_DIM,), self.ln1_gamma, self.ln1_beta, eps=1e-6)
        ff = F.gelu(torch.matmul(queries, self.ff1), approximate="tanh")
        queries = queries + torch.matmul(ff, self.ff2)
        queries = F.layer_norm(queries, (GEMMA_DIM,), self.ln2_gamma, self.ln2_beta, eps=1e-6)
        return queries

    def export(self, output_path: str, *, metadata: dict[str, Any]) -> None:
        payload = {
            "queries": self.queries.detach().cpu().numpy().astype(np.float32),
            "wq": self.wq.detach().cpu().numpy().astype(np.float32),
            "wk": self.wk.detach().cpu().numpy().astype(np.float32),
            "wv": self.wv.detach().cpu().numpy().astype(np.float32),
            "wo": self.wo.detach().cpu().numpy().astype(np.float32),
            "ln1_gamma": self.ln1_gamma.detach().cpu().numpy().astype(np.float32),
            "ln1_beta": self.ln1_beta.detach().cpu().numpy().astype(np.float32),
            "ff1": self.ff1.detach().cpu().numpy().astype(np.float32),
            "ff2": self.ff2.detach().cpu().numpy().astype(np.float32),
            "ln2_gamma": self.ln2_gamma.detach().cpu().numpy().astype(np.float32),
            "ln2_beta": self.ln2_beta.detach().cpu().numpy().astype(np.float32),
            "random_init": np.array(False),
        }
        for key, value in metadata.items():
            payload[key] = np.array(value)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **payload)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_coco_records(
    coco_dir: Path,
    *,
    captions_json: str | None,
    images_dir: str | None,
    max_samples: int,
    captions_per_image: int,
    seed: int,
) -> list[tuple[Path, str]]:
    captions_path = Path(captions_json) if captions_json else coco_dir / "annotations" / "captions_train2017.json"
    image_root = Path(images_dir) if images_dir else coco_dir / "train2017"
    payload = json.loads(captions_path.read_text())
    images = {int(entry["id"]): entry["file_name"] for entry in payload.get("images", []) if "id" in entry and "file_name" in entry}
    grouped: dict[int, list[str]] = {}
    for annotation in payload.get("annotations", []):
        image_id = int(annotation.get("image_id"))
        caption = str(annotation.get("caption") or "").strip()
        if image_id not in images or not caption:
            continue
        grouped.setdefault(image_id, []).append(caption)

    rng = random.Random(seed)
    image_ids = list(grouped.keys())
    rng.shuffle(image_ids)
    records: list[tuple[Path, str]] = []
    for image_id in image_ids:
        image_path = image_root / images[image_id]
        if not image_path.exists():
            continue
        captions = grouped[image_id][:]
        rng.shuffle(captions)
        for caption in captions[: max(captions_per_image, 1)]:
            records.append((image_path, caption))
            if len(records) >= max_samples:
                return records
    return records


def _prepare_samples(
    *,
    records: list[tuple[Path, str]],
    encoder: Dinov2Encoder,
    teacher: GemmaCaptionTeacher,
) -> list[TVLCSample]:
    samples: list[TVLCSample] = []
    patch_cache: dict[str, np.ndarray] = {}
    for image_path, caption in records:
        patch_tokens = patch_cache.get(str(image_path))
        if patch_tokens is None:
            image = Image.open(image_path).convert("RGB")
            patch_tokens = encoder.encode(image).patch_tokens.astype(np.float32)
            patch_cache[str(image_path)] = patch_tokens
        semantics = teacher.canonicalize(caption)
        target_vector = _semantic_hash_vector(semantics, caption)
        samples.append(
            TVLCSample(
                image_path=image_path,
                raw_caption=caption,
                semantics=semantics,
                patch_tokens=patch_tokens,
                target_vector=target_vector,
            )
        )
    return samples


def _train(
    *,
    samples: list[TVLCSample],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_path: str,
    init_weights: str | None,
    teacher: GemmaCaptionTeacher,
    train_all: bool,
) -> None:
    if not samples:
        raise RuntimeError("No TVLC training samples prepared")
    torch_device = torch.device(device)
    model = TorchTVLCConnector(init_weights, train_all=train_all).to(torch_device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    patches = torch.from_numpy(np.stack([sample.patch_tokens for sample in samples])).to(torch_device)
    targets = torch.from_numpy(np.stack([sample.target_vector for sample in samples])).to(torch_device)
    temperature = 0.07

    for epoch in range(max(epochs, 1)):
        permutation = torch.randperm(patches.shape[0], device=torch_device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, patches.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_patches = patches.index_select(0, batch_indices)
            batch_targets = targets.index_select(0, batch_indices)
            projected = F.normalize(model(batch_patches).mean(dim=1), dim=-1)
            target_norm = F.normalize(batch_targets, dim=-1)
            logits = torch.matmul(projected, target_norm.t()) / temperature
            labels = torch.arange(logits.shape[0], device=torch_device)
            contrastive = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
            )
            cosine = 1.0 - torch.sum(projected * target_norm, dim=-1).mean()
            loss = 0.65 * contrastive + 0.35 * cosine
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        print(f"[train_tvlc] epoch {epoch + 1}/{max(epochs, 1)} loss={epoch_loss / max(batches, 1):.4f}", flush=True)

    model.export(
        output_path,
        metadata={
            "trainer_backend": "torch_partial_finetune",
            "teacher_mode": "gemma4_teacher" if teacher.available else "fallback_caption_normalizer",
            "sample_count": len(samples),
            "seed": SEED,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TVLCConnector with Gemma-assisted caption supervision")
    parser.add_argument("--random-init", action="store_true", help="Create random-weight TVLC connector and exit")
    parser.add_argument("--coco-dir", type=str, help="Path to COCO dataset root")
    parser.add_argument("--captions-json", type=str, help="Override captions json path")
    parser.add_argument("--images-dir", type=str, help="Override image directory path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--captions-per-image", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|mps")
    parser.add_argument("--encoder-device", type=str, default="cpu", help="DINO encoder device")
    parser.add_argument("--weights-out", type=str, default=TVLCConnector.default_weights_path())
    parser.add_argument("--init-weights", type=str, default=TVLCConnector.default_weights_path())
    parser.add_argument("--teacher-cache", type=str, default=".toori/training/tvlc_teacher_cache.json")
    parser.add_argument("--disable-gemma-teacher", action="store_true")
    parser.add_argument("--mlx-command", type=str, help="Override MLX daemon command for Gemma teacher")
    parser.add_argument("--train-all", action="store_true", help="Train all connector matrices instead of partial finetune")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.random_init:
        TVLCConnector.create_random_init(args.weights_out)
        print(f"Random-init TVLC saved to: {args.weights_out}")
        print("NOTE: random init only validates pipeline shape. Use --coco-dir to train semantic slot grounding.")
        return

    if not args.coco_dir:
        parser.error("Provide --random-init or --coco-dir")

    teacher = GemmaCaptionTeacher.from_defaults(
        enabled=not args.disable_gemma_teacher,
        cache_path=Path(args.teacher_cache),
        mlx_command=args.mlx_command,
    )
    try:
        records = _load_coco_records(
            Path(args.coco_dir),
            captions_json=args.captions_json,
            images_dir=args.images_dir,
            max_samples=max(int(args.max_samples), 1),
            captions_per_image=max(int(args.captions_per_image), 1),
            seed=args.seed,
        )
        print(
            f"[train_tvlc] loaded {len(records)} caption/image pairs "
            f"(teacher={'gemma4' if teacher.available else 'fallback'})",
            flush=True,
        )
        samples = _prepare_samples(
            records=records,
            encoder=Dinov2Encoder(device=args.encoder_device),
            teacher=teacher,
        )
        print(f"[train_tvlc] prepared {len(samples)} samples", flush=True)
        _train(
            samples=samples,
            device=_resolve_device(args.device),
            epochs=max(int(args.epochs), 1),
            batch_size=max(int(args.batch_size), 1),
            lr=float(args.lr),
            output_path=args.weights_out,
            init_weights=args.init_weights if Path(args.init_weights).exists() else None,
            teacher=teacher,
            train_all=bool(args.train_all),
        )
        print(f"[train_tvlc] trained weights saved to: {args.weights_out}", flush=True)
    finally:
        teacher.close()


if __name__ == "__main__":
    main(sys.argv[1:])
