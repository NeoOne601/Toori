"""
train_tvlc_mlx.py — Train TVLCConnector with DINOv2 patches and Gemma-guided captions.

Working principle:
  1. Extract 196x384 DINOv2 patch tokens from each training image.
  2. Canonicalize paired captions into atomic object labels, attributes, and scene tags.
     Gemma 4 is used as an optional teacher via scripts/mlx_reasoner.py; if that path
     is unavailable, the trainer falls back to deterministic caption heuristics.
  3. Build a 32x2048 semantic target sequence from the canonicalized text evidence.
  4. Train a Perceiver-style resampler to map image patches into that semantic token space.
  5. Export weights in the exact .npz layout expected by TVLCConnector, plus optional
     semantic prototypes that improve TVLC context strings at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud.perception.dinov2_encoder import Dinov2Encoder
from cloud.perception.gemma_semantic_extractor import DEFAULT_GEMMA_MODEL_PATH, GemmaSemanticExtractor
from cloud.perception.tvlc_connector import GEMMA_DIM, HEAD_DIM, N_HEADS, N_PATCHES, N_QUERIES, PATCH_DIM

DEFAULT_ANNOTATIONS = "annotations/captions_train2017.json"
DEFAULT_OUTPUT = "models/vision/tvlc_connector.npz"
DEFAULT_CACHE_DIR = ".toori/tvlc_train_cache"
DEFAULT_GEMMA_SCRIPT = "scripts/mlx_reasoner.py"
TRAINER_VERSION = "tvlc_gemma_teacher_v1"
RELATION_WORDS = {
    "near",
    "behind",
    "beside",
    "between",
    "holding",
    "wearing",
    "with",
    "on",
    "under",
    "over",
    "in",
    "at",
    "by",
    "next",
    "front",
    "left",
    "right",
}
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "is",
    "are",
    "this",
    "that",
    "these",
    "those",
    "some",
    "many",
    "several",
    "very",
    "small",
    "large",
    "big",
    "little",
}
ATTRIBUTE_WORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "white",
    "black",
    "brown",
    "gray",
    "grey",
    "pink",
    "purple",
    "striped",
    "wooden",
    "metal",
    "plastic",
    "glass",
}


@dataclass(slots=True)
class CaptionTarget:
    primary_label: str
    attributes: list[str]
    scene_tags: list[str]
    summary: str
    source: str


@dataclass(slots=True)
class TrainingRecord:
    annotation_id: int
    image_id: int
    image_path: Path
    caption: str


def _sanitize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", str(value or "").lower())
    return " ".join(cleaned.split()).strip()


def _stable_phrase_vector(text: str, dim: int, seed: int = 0) -> np.ndarray:
    normalized = _sanitize_text(text) or "object"
    digest = hashlib.sha256(f"{seed}:{normalized}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(local_seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).astype(np.float32)


class SemanticTargetBuilder:
    def __init__(self, mode: str, model_path: str | None = None) -> None:
        self._mode = mode
        self._extractor: GemmaSemanticExtractor | None = None
        self._disabled_reason: str | None = None
        if mode != "off":
            try:
                self._extractor = GemmaSemanticExtractor(model_path or DEFAULT_GEMMA_MODEL_PATH)
            except Exception as exc:
                self._disabled_reason = str(exc)
                if mode == "on":
                    raise
                self._extractor = None

    @property
    def ready(self) -> bool:
        return self._extractor is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def encode_phrase(self, text: str, *, seed: int) -> np.ndarray:
        if self._extractor is not None:
            vector = self._extractor.encode_text(text)
            if np.any(vector):
                return vector.astype(np.float32)
        return _stable_phrase_vector(text, GEMMA_DIM, seed=seed)

    def cosine_similarity(self, left: str, right: str) -> float | None:
        if self._extractor is None:
            return None
        return float(self._extractor.cosine_similarity(left, right))


def _mean_vectors(values: Iterable[np.ndarray], dim: int) -> np.ndarray:
    vectors = [np.asarray(value, dtype=np.float32) for value in values if value is not None]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    stacked = np.stack(vectors, axis=0)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean)) or 1.0
    return (mean / norm).astype(np.float32)


def _heuristic_caption_target(caption: str) -> CaptionTarget:
    normalized = _sanitize_text(caption)
    if not normalized:
        return CaptionTarget(
            primary_label="object",
            attributes=[],
            scene_tags=[],
            summary="object",
            source="heuristic",
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
        primary_tokens = [next((token for token in tokens if token not in STOPWORDS), "object")]
    summary = " ".join(primary_tokens[:2] + attributes[:2] + scene_tags[:3]).strip() or normalized
    return CaptionTarget(
        primary_label=" ".join(primary_tokens[:2]),
        attributes=attributes[:3],
        scene_tags=scene_tags[:4],
        summary=summary,
        source="heuristic",
    )


class GemmaTeacher:
    def __init__(self, repo_root: Path, mode: str, script_path: str) -> None:
        self._repo_root = repo_root
        self._mode = mode
        self._script_path = str((repo_root / script_path).resolve())
        self._proc: subprocess.Popen[str] | None = None
        self._disabled_reason: str | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def canonicalize(self, caption: str) -> CaptionTarget:
        if self._mode == "off":
            return _heuristic_caption_target(caption)
        if not self._ensure_started():
            return _heuristic_caption_target(caption)
        prompt = (
            "Convert the caption into grounded semantic supervision for a vision-language connector. "
            "Return ONLY compact JSON with schema "
            '{"primary_label":str,"attributes":[str],"scene_tags":[str],"summary":str}. '
            "Rules: primary_label must be atomic, no relation phrases, no 'near/behind/left/right of', "
            "prefer common object names and attributes supported by the caption."
            f'\nCaption: "{caption}"'
        )
        payload = {
            "prompt": prompt,
            "image_base64": None,
            "system": "You are a precise caption canonicalizer. Output only JSON.",
            "max_tokens": 96,
        }
        try:
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            raw = self._proc.stdout.readline().strip()
            if not raw:
                raise RuntimeError("empty teacher response")
            result = json.loads(raw)
            text = str(result.get("text") or "").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0) if match else text)
            target = CaptionTarget(
                primary_label=_sanitize_text(parsed.get("primary_label") or ""),
                attributes=[_sanitize_text(item) for item in parsed.get("attributes") or [] if _sanitize_text(item)],
                scene_tags=[_sanitize_text(item) for item in parsed.get("scene_tags") or [] if _sanitize_text(item)],
                summary=_sanitize_text(parsed.get("summary") or caption),
                source="gemma4",
            )
            if not target.primary_label:
                raise RuntimeError("teacher did not return a primary label")
            return target
        except Exception as exc:
            if self._mode == "daemon":
                self._disabled_reason = str(exc)
            return _heuristic_caption_target(caption)

    def _ensure_started(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if self._disabled_reason and self._mode == "auto":
            return False
        command = [sys.executable, self._script_path]
        try:
            self._proc = subprocess.Popen(
                command,
                cwd=str(self._repo_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._disabled_reason = str(exc)
            self._proc = None
            return False
        probe = {"type": "healthcheck"}
        try:
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(probe) + "\n")
            self._proc.stdin.flush()
            started = time.time()
            while time.time() - started < 8.0:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"teacher exited with code {self._proc.returncode}")
                line = self._proc.stdout.readline().strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("type") == "healthcheck" and payload.get("success"):
                    return True
            raise RuntimeError("teacher healthcheck timed out")
        except Exception as exc:
            self._disabled_reason = str(exc)
            self.close()
            return False


def _target_tokens(target: CaptionTarget, builder: SemanticTargetBuilder) -> np.ndarray:
    primary_vec = builder.encode_phrase(target.primary_label, seed=11)
    attr_vec = _mean_vectors((builder.encode_phrase(item, seed=13) for item in target.attributes), GEMMA_DIM)
    scene_vec = _mean_vectors((builder.encode_phrase(item, seed=17) for item in target.scene_tags), GEMMA_DIM)
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


class TorchTVLCConnector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        scale_q = float(np.sqrt(2.0 / GEMMA_DIM))
        scale_patch = float(np.sqrt(2.0 / PATCH_DIM))
        self.queries = nn.Parameter(torch.randn(N_QUERIES, GEMMA_DIM) * scale_q)
        self.wq = nn.Parameter(torch.randn(GEMMA_DIM, GEMMA_DIM) * scale_q)
        self.wk = nn.Parameter(torch.randn(PATCH_DIM, GEMMA_DIM) * scale_patch)
        self.wv = nn.Parameter(torch.randn(PATCH_DIM, GEMMA_DIM) * scale_patch)
        self.wo = nn.Parameter(torch.randn(GEMMA_DIM, GEMMA_DIM) * scale_q)
        self.ln1 = nn.LayerNorm(GEMMA_DIM)
        self.ln2 = nn.LayerNorm(GEMMA_DIM)
        self.ff1 = nn.Linear(GEMMA_DIM, GEMMA_DIM * 4)
        self.ff2 = nn.Linear(GEMMA_DIM * 4, GEMMA_DIM)

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
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch_size, N_QUERIES, GEMMA_DIM)
        out = torch.matmul(out, self.wo)
        queries = self.ln1(queries + out)
        ff = self.ff2(F.gelu(self.ff1(queries)))
        return self.ln2(queries + ff)


class CocoTVLCDataset(Dataset[Any]):
    def __init__(
        self,
        repo_root: Path,
        coco_dir: Path,
        annotations_path: Path,
        cache_dir: Path,
        teacher: GemmaTeacher,
        target_builder: SemanticTargetBuilder,
        max_samples: int | None = None,
        dino_model_path: str | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._teacher = teacher
        self._target_builder = target_builder
        self._dino = Dinov2Encoder(device="cpu", model_path=dino_model_path, allow_download=False)
        with annotations_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        images = {
            int(item["id"]): coco_dir / "train2017" / str(item["file_name"])
            for item in payload.get("images", [])
        }
        records = [
            TrainingRecord(
                annotation_id=int(annotation["id"]),
                image_id=int(annotation["image_id"]),
                image_path=images[int(annotation["image_id"])],
                caption=str(annotation.get("caption") or "").strip(),
            )
            for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in images
        ]
        if max_samples is not None:
            records = records[: max(0, int(max_samples))]
        self._records = [record for record in records if record.image_path.exists()]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]
        cache_file = self._cache_dir / f"{record.annotation_id}.npz"
        if cache_file.exists():
            data = np.load(str(cache_file))
            return {
                "patch_tokens": torch.from_numpy(data["patch_tokens"].astype(np.float32)),
                "target_tokens": torch.from_numpy(data["target_tokens"].astype(np.float32)),
                "primary_label": str(data["primary_label"]),
                "summary": str(data["summary"]),
                "target_source": str(data["target_source"]),
            }
        with Image.open(record.image_path) as image:
            embedding = self._dino.encode(image.convert("RGB"))
        gemma_cache = self._cache_dir / f"{record.annotation_id}_gemma.npz"
        if gemma_cache.exists():
            g_data = np.load(str(gemma_cache))
            target_tokens = g_data["target_tokens"]
            primary_label = str(g_data["primary_label"])
            summary = str(g_data["summary"])
            target_source = str(g_data["target_source"])
        else:
            target = self._teacher.canonicalize(record.caption)
            target_tokens = _target_tokens(target, self._target_builder)
            primary_label = target.primary_label
            summary = target.summary
            target_source = target.source

        np.savez(
            str(cache_file),
            patch_tokens=embedding.patch_tokens.astype(np.float32),
            target_tokens=target_tokens.astype(np.float32),
            primary_label=np.array(primary_label),
            summary=np.array(summary),
            target_source=np.array(target_source),
        )
        return {
            "patch_tokens": torch.from_numpy(embedding.patch_tokens.astype(np.float32)),
            "target_tokens": torch.from_numpy(target_tokens.astype(np.float32)),
            "primary_label": primary_label,
            "summary": summary,
            "target_source": target_source,
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "patch_tokens": torch.stack([item["patch_tokens"] for item in batch], dim=0),
        "target_tokens": torch.stack([item["target_tokens"] for item in batch], dim=0),
        "primary_labels": [item["primary_label"] for item in batch],
        "summaries": [item["summary"] for item in batch],
        "sources": [item["target_source"] for item in batch],
    }


def _build_prototypes(
    model: TorchTVLCConnector,
    dataloader: DataLoader[Any],
    device: torch.device,
    prototype_count: int,
    target_builder: SemanticTargetBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    accum: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            patches = batch["patch_tokens"].to(device)
            pooled = F.normalize(model(patches).mean(dim=1), dim=-1).cpu().numpy()
            for label, vector in zip(batch["primary_labels"], pooled):
                label_text = _sanitize_text(label) or "object"
                accum[label_text].append(vector.astype(np.float32))
    counts = Counter({label: len(vectors) for label, vectors in accum.items()})
    top_labels = [label for label, _count in counts.most_common(max(1, prototype_count))]
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    for label in top_labels:
        labels.append(label)
        if target_builder.ready:
            vectors.append(target_builder.encode_phrase(label, seed=41))
        else:
            mean = np.stack(accum[label], axis=0).mean(axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean)) or 1.0
            vectors.append(mean / norm)
    if not labels:
        return np.asarray([], dtype="<U1"), np.zeros((0, GEMMA_DIM), dtype=np.float32)
    width = max(len(label) for label in labels)
    return np.asarray(labels, dtype=f"<U{max(8, width)}"), np.stack(vectors, axis=0).astype(np.float32)


def _export_weights(
    model: TorchTVLCConnector,
    output_path: Path,
    prototype_labels: np.ndarray,
    prototype_vectors: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queries": model.queries.detach().cpu().numpy().astype(np.float32),
        "wq": model.wq.detach().cpu().numpy().astype(np.float32),
        "wk": model.wk.detach().cpu().numpy().astype(np.float32),
        "wv": model.wv.detach().cpu().numpy().astype(np.float32),
        "wo": model.wo.detach().cpu().numpy().astype(np.float32),
        "ln1_gamma": model.ln1.weight.detach().cpu().numpy().astype(np.float32),
        "ln1_beta": model.ln1.bias.detach().cpu().numpy().astype(np.float32),
        "ff1": model.ff1.weight.detach().cpu().numpy().T.astype(np.float32),
        "ff2": model.ff2.weight.detach().cpu().numpy().T.astype(np.float32),
        "ln2_gamma": model.ln2.weight.detach().cpu().numpy().astype(np.float32),
        "ln2_beta": model.ln2.bias.detach().cpu().numpy().astype(np.float32),
        "random_init": np.array(False),
        "prototype_labels": prototype_labels,
        "prototype_vectors": prototype_vectors.astype(np.float32),
        "trainer_version": np.array(TRAINER_VERSION),
        "training_metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    np.savez(str(output_path), **payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_tvlc.py",
        description="Train TVLCConnector with Gemma-guided captions",
    )
    parser.add_argument("--coco-dir", type=str, required=True, help="COCO dataset root containing train2017 and annotations/")
    parser.add_argument("--annotations", type=str, default=DEFAULT_ANNOTATIONS, help="Relative path under --coco-dir to captions JSON")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to write tvlc_connector.npz")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Feature/cache directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or mps")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--prototype-count", type=int, default=64)
    parser.add_argument("--dinov2-model-path", type=str, default=None)
    parser.add_argument("--gemma-mode", choices=("auto", "daemon", "off"), default="auto")
    parser.add_argument("--gemma-embeddings", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--gemma-model-path", type=str, default=None, help="Override local Gemma model path for semantic embeddings")
    parser.add_argument("--gemma-script", type=str, default=DEFAULT_GEMMA_SCRIPT, help="Path to mlx_reasoner.py relative to repo root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    coco_dir = Path(args.coco_dir).expanduser().resolve()
    annotations_path = (coco_dir / args.annotations).resolve()
    output_path = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir).resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO captions file not found: {annotations_path}")
    teacher = GemmaTeacher(repo_root=repo_root, mode=args.gemma_mode, script_path=args.gemma_script)
    target_builder = SemanticTargetBuilder(mode=args.gemma_embeddings, model_path=args.gemma_model_path)
    try:
        dataset = CocoTVLCDataset(
            repo_root=repo_root,
            coco_dir=coco_dir,
            annotations_path=annotations_path,
            cache_dir=cache_dir,
            teacher=teacher,
            target_builder=target_builder,
            max_samples=args.max_samples,
            dino_model_path=args.dinov2_model_path,
        )
        if not len(dataset):
            raise RuntimeError("No COCO caption/image pairs were discovered")
        dataloader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_batch,
        )
        device = torch.device(args.device)
        model = TorchTVLCConnector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
        for epoch in range(max(1, int(args.epochs))):
            model.train()
            progress = tqdm(dataloader, desc=f"TVLC epoch {epoch + 1}/{args.epochs}")
            running_loss = 0.0
            for step, batch in enumerate(progress, start=1):
                patches = batch["patch_tokens"].to(device)
                targets = batch["target_tokens"].to(device)
                predictions = model(patches)
                pred_pooled = F.normalize(predictions.mean(dim=1), dim=-1)
                target_pooled = F.normalize(targets.mean(dim=1), dim=-1)
                logits = torch.matmul(pred_pooled, target_pooled.T) / float(args.temperature)
                labels = torch.arange(logits.shape[0], device=device)
                contrastive = 0.5 * (
                    F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                )
                mse = F.mse_loss(predictions, targets)
                cosine = 1.0 - F.cosine_similarity(pred_pooled, target_pooled, dim=-1).mean()
                diversity = predictions.std(dim=1).mean()
                loss = mse + cosine + (0.25 * contrastive) - (0.01 * diversity)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += float(loss.item())
                progress.set_postfix(loss=f"{running_loss / step:.4f}")
        prototype_loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_batch,
        )
        prototype_labels, prototype_vectors = _build_prototypes(
            model=model,
            dataloader=prototype_loader,
            device=device,
            prototype_count=max(1, int(args.prototype_count)),
            target_builder=target_builder,
        )
        metadata = {
            "trainer_version": TRAINER_VERSION,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "dataset_size": len(dataset),
            "gemma_mode": args.gemma_mode,
            "gemma_embeddings": args.gemma_embeddings,
            "teacher_disabled_reason": teacher.disabled_reason,
            "embedding_disabled_reason": target_builder.disabled_reason,
            "prototype_count": int(len(prototype_labels)),
            "output_path": str(output_path),
        }
        _export_weights(
            model=model,
            output_path=output_path,
            prototype_labels=prototype_labels,
            prototype_vectors=prototype_vectors,
            metadata=metadata,
        )
        print(f"TVLC weights saved to: {output_path}")
        if teacher.disabled_reason:
            print(f"Gemma teacher fallback reason: {teacher.disabled_reason}")
        print(
            "TVLC training complete. Runtime will treat this connector as trained "
            "and expose prototype-backed TVLC context during open-vocab labeling."
        )
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
