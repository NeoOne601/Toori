from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from PIL import Image, ImageFilter

from cloud.perception import PerceptionPipeline

from .models import (
    Answer,
    BoundingBox,
    Observation,
    ProviderConfig,
    ProviderHealth,
    ReasoningTraceEntry,
    RuntimeSettings,
)


def _image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _normalize_vector(vector: np.ndarray, size: int = 128) -> list[float]:
    flattened = vector.astype(np.float32).reshape(-1)
    if flattened.size == 0:
        return [0.0] * size
    if flattened.size < size:
        flattened = np.pad(flattened, (0, size - flattened.size))
    elif flattened.size > size:
        bins = np.array_split(flattened, size)
        flattened = np.array([float(chunk.mean()) for chunk in bins], dtype=np.float32)
    norm = np.linalg.norm(flattened) or 1.0
    return (flattened / norm).astype(np.float32).tolist()


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _vector_softmax_confidence(values: np.ndarray) -> float:
    flattened = values.reshape(-1).astype(np.float32)
    if flattened.size == 0:
        return 0.0
    shifted = flattened - np.max(flattened)
    exps = np.exp(shifted)
    total = float(np.sum(exps)) or 1.0
    return float(np.max(exps) / total)


def _coerce_float_list(value: object, default: list[float]) -> list[float]:
    if isinstance(value, list) and value:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return default
    return default


def _load_labels(path_value: object) -> list[str]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [str(item) for item in data]
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


GENERIC_LABEL_TOKENS = {
    "scene",
    "image",
    "photo",
    "picture",
    "frame",
    "view",
    "background",
    "foreground",
    "object",
    "objects",
    "thing",
    "things",
    "visual",
    "dominant",
    "balanced",
    "textured",
    "smooth",
    "textured",
    "bright",
    "dark",
    "light",
    "color",
    "colors",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "entity",
    "tracked",
    "track",
    "tracks",
    "proposal",
    "proposals",
    "candidate",
    "candidates",
    "descriptor",
    "descriptors",
    "label",
    "labels",
    "rgb",
    "histogram",
    "histograms",
    "basic",
    "onnx",
    "perception",
    "vector",
}

ABSURD_LABELS = {
    "hot dog near velvet",
    "hand plane near windsor tie",
    "lemon near bra",
    "rgb histogram+edge histogram",
    "rgb histogram edge histogram",
    "dominant color",
    "brightness label",
    "edge label",
}

SUSPICIOUS_CLASSIFIER_LABELS = {
    "oxygen mask",
    "bra",
    "windsor tie",
    "hot dog",
    "hand plane",
    "velvet",
    "lemon",
}

RELATION_PHRASES = (
    " near ",
    " behind ",
    " left of ",
    " right of ",
    " above ",
    " below ",
    " in front of ",
    " next to ",
    " beside ",
    " with ",
    " on top of ",
)

PLACEHOLDER_LABEL_RE = re.compile(r"^(?:entity|proposal|candidate|tracked(?: region| object)?|object)\s*[-_ ]*\d*$")


def _normalize_label(label: object) -> str:
    return (
        str(label or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .strip()
    )


def _meaningful_label(label: object) -> str:
    normalized = _normalize_label(label)
    if not normalized:
        return ""
    if normalized in ABSURD_LABELS or PLACEHOLDER_LABEL_RE.match(normalized):
        return ""
    if any(phrase in normalized for phrase in RELATION_PHRASES):
        return ""
    if "histogram" in normalized or "descriptor" in normalized:
        return ""
    tokens = [token for token in normalized.split() if len(token) >= 3 and token not in GENERIC_LABEL_TOKENS]
    if not tokens:
        return ""
    candidate = " ".join(tokens[:4])
    if candidate in ABSURD_LABELS or PLACEHOLDER_LABEL_RE.match(candidate):
        return ""
    return candidate


def _proposal_fallback_label(region: BoundingBox | None, index: int) -> str:
    region_label = _meaningful_label(getattr(region, "label", None) or "")
    if region_label and region_label not in SUSPICIOUS_CLASSIFIER_LABELS:
        return region_label
    return "object"


def _label_rank(label: str, score: float, saliency: float) -> float:
    tokens = _meaningful_label(label).split()
    if not tokens:
        return 0.0
    length_bonus = 0.02 * min(len(tokens), 3)
    return _clamp_unit((0.55 * score) + (0.35 * saliency) + length_bonus)


def _region_saliency(crop: Image.Image) -> float:
    resized = crop.convert("RGB").resize((48, 48))
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    gray = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
    edges = np.asarray(resized.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32) / 255.0
    contrast = float(gray.std())
    saturation = float(hsv[:, :, 1].mean())
    edge_density = float(edges.mean())
    brightness = float(gray.mean())
    saliency = (edge_density * 0.45) + (contrast * 0.3) + (saturation * 0.2) + (0.05 * (1.0 - abs(brightness - 0.5) * 2.0))
    return _clamp_unit(saliency)


def _grid_regions(image: Image.Image) -> list[tuple[BoundingBox, float]]:
    """Return salient candidate regions from a 5x5 grid for finer small-object coverage."""
    width, height = image.size
    regions: list[tuple[BoundingBox, float]] = []
    cols, rows = 5, 5
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    for row in range(rows):
        for col in range(cols):
            left = col * cell_width
            top = row * cell_height
            box = BoundingBox(x=left, y=top, width=cell_width, height=cell_height)
            px_box = (
                int(round(left * width)),
                int(round(top * height)),
                int(round((left + cell_width) * width)),
                int(round((top + cell_height) * height)),
            )
            crop = image.crop(px_box)
            saliency = _region_saliency(crop)
            center_bonus = 1.0 - (abs((left + cell_width / 2.0) - 0.5) + abs((top + cell_height / 2.0) - 0.5)) / 2.0
            regions.append((box, _clamp_unit((saliency * 0.88) + (center_bonus * 0.12))))
    regions.sort(key=lambda item: item[1], reverse=True)
    return regions


def _ollama_model_matches(configured: str, available: str) -> bool:
    if configured == available:
        return True
    return configured.split(":", 1)[0] == available.split(":", 1)[0]


def _short_probe_message(output: str) -> str:
    lowered = output.lower()
    if "nsrangeexception" in lowered or "index 0 beyond bounds" in lowered:
        return "MLX crashed during Metal device initialization on this machine"
    if "mlx-vlm is not installed" in lowered:
        return "mlx-vlm is not installed in python3.11"
    return output.strip().splitlines()[-1] if output.strip() else "provider probe failed"


def _short_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    lowered = message.lower()
    if "timed out" in lowered:
        return "timed out"
    if "operation not permitted" in lowered:
        return "operation not permitted"
    return message.splitlines()[0]


class BasicVisionProvider:
    name = "basic"

    def health(self, config: ProviderConfig) -> ProviderHealth:
        return ProviderHealth(
            name=self.name,
            role="perception",
            enabled=config.enabled,
            healthy=config.enabled,
            message="Classical image descriptor fallback is available",
        )

    def perceive(self, image: Image.Image) -> tuple[list[float], float, dict]:
        rgb = image.convert("RGB").resize((64, 64))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        edges = np.asarray(rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        channels = [arr[:, :, idx] for idx in range(3)]
        histograms = [
            np.histogram(channel, bins=32, range=(0.0, 1.0), density=True)[0]
            for channel in channels
        ]
        gray_hist = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)[0]
        edge_hist = np.histogram(edges, bins=16, range=(0.0, 1.0), density=True)[0]
        vector = np.concatenate(histograms + [gray_hist, edge_hist]).astype(np.float32)
        mean_rgb = arr.mean(axis=(0, 1))
        color_names = ["red", "green", "blue"]
        dominant_index = int(np.argmax(mean_rgb))
        brightness = float(gray.mean())
        edge_density = float(edges.mean())
        return _normalize_vector(vector), 0.58, {
            "descriptor": "rgb_histogram+edge_histogram",
            "dominant_color": color_names[dominant_index],
            "brightness": brightness,
            "brightness_label": _brightness_label(brightness),
            "edge_density": edge_density,
            "edge_label": "textured" if edge_density > 0.18 else "smooth",
        }


def _brightness_label(brightness: float) -> str:
    if brightness < 0.25:
        return "dark"
    if brightness > 0.75:
        return "bright"
    return "balanced"


class OnnxPerceptionProvider:
    name = "onnx"

    def __init__(self) -> None:
        self._session = None
        self._loaded_path: Optional[str] = None

    def health(self, config: ProviderConfig) -> ProviderHealth:
        if not config.enabled:
            return ProviderHealth(name=self.name, role="perception", enabled=False, message="disabled")
        if not importlib.util.find_spec("onnxruntime"):
            version = f"{sys.version_info.major}.{sys.version_info.minor}"
            return ProviderHealth(
                name=self.name,
                role="perception",
                enabled=True,
                healthy=False,
                message=f"onnxruntime not installed for Python {version}; use python3.11 or install into the active interpreter",
            )
        if not config.model_path or not Path(config.model_path).exists():
            return ProviderHealth(name=self.name, role="perception", enabled=True, healthy=False, message="model_path is missing or not found")
        return ProviderHealth(name=self.name, role="perception", enabled=True, healthy=True, message="ready")

    def perceive(self, image: Image.Image, config: ProviderConfig) -> tuple[list[float], float, dict]:
        import onnxruntime as ort

        if self._session is None or self._loaded_path != config.model_path:
            self._session = ort.InferenceSession(str(config.model_path), providers=["CPUExecutionProvider"])
            self._loaded_path = config.model_path
        session = self._session
        input_name = session.get_inputs()[0].name
        input_size = int(config.metadata.get("input_size", 224))
        resized = image.convert("RGB").resize((input_size, input_size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array(_coerce_float_list(config.metadata.get("mean"), [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(_coerce_float_list(config.metadata.get("std"), [0.229, 0.224, 0.225]), dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        outputs = session.run(None, {input_name: arr})
        output_index = int(config.metadata.get("output_index", 0))
        selected = np.asarray(outputs[output_index], dtype=np.float32).reshape(-1)
        embedding = _normalize_vector(selected)
        metadata = {"descriptor": "onnx", "input_size": input_size}
        labels = _load_labels(config.metadata.get("labels_path"))
        if labels:
            top_index = int(np.argmax(selected))
            if 0 <= top_index < len(labels):
                metadata["top_label"] = labels[top_index]
                metadata["top_index"] = top_index
        return embedding, max(0.65, _vector_softmax_confidence(selected)), metadata


class DinoV2PerceptionProvider:
    name = "dinov2"

    def __init__(self) -> None:
        self._pipeline: PerceptionPipeline | None = None
        self._device: str | None = None

    def health(self, config: ProviderConfig) -> ProviderHealth:
        if not config.enabled:
            return ProviderHealth(name=self.name, role="perception", enabled=False, message="disabled")
        device = str(config.metadata.get("device", "mps"))
        return ProviderHealth(
            name=self.name,
            role="perception",
            enabled=True,
            healthy=True,
            message=f"lazy-loaded DINOv2+SAM pipeline ready on {device}",
        )

    def perceive(self, image: Image.Image, config: ProviderConfig) -> tuple[list[float], float, dict]:
        pipeline = self._ensure_pipeline(config)
        embedding = pipeline.retrieval_embedding(image)
        return embedding.astype(np.float32).tolist(), 0.76, {
            "descriptor": "dinov2",
            "backend": "open-vocabulary",
            "device": str(config.metadata.get("device", "mps")),
            "patch_grid": [14, 14],
        }

    def object_proposals(self, image: Image.Image, config: ProviderConfig, *, max_proposals: int = 12) -> list[BoundingBox]:
        pipeline = self._ensure_pipeline(config)
        _, masks = pipeline.encode(image)
        width, height = image.size
        proposals: list[BoundingBox] = []
        for index, mask in enumerate(masks):
            bbox = mask.bbox_pixels
            w_norm = float(bbox["width"]) / max(width, 1)
            h_norm = float(bbox["height"]) / max(height, 1)
            
            # Physically reject any masks whose bounding box spans more than 60% of the screen area
            # (this prevents scattered background pixels from creating massive 100% boundary squares)
            if w_norm * h_norm > 0.60 or mask.area_fraction > 0.85:
                continue
                
            # Penalize the score of remaining large masks to prioritize target-locking individual entities
            adjusted_score = float(mask.confidence) * (1.0 - (mask.area_fraction * 0.4))
            
            proposals.append(
                BoundingBox(
                    x=float(bbox["x"]) / max(width, 1),
                    y=float(bbox["y"]) / max(height, 1),
                    width=w_norm,
                    height=h_norm,
                    label=f"entity-{index + 1}",
                    score=round(adjusted_score, 4),
                )
            )
            
        proposals.sort(key=lambda p: float(p.score or 0.0), reverse=True)
        return proposals[:max_proposals]

    def _ensure_pipeline(self, config: ProviderConfig) -> PerceptionPipeline:
        device = str(config.metadata.get("device", "mps"))
        if self._pipeline is None or self._device != device:
            self._pipeline = PerceptionPipeline(device=device)
            self._device = device
        return self._pipeline


class OllamaReasoningProvider:
    name = "ollama"

    def health(self, config: ProviderConfig) -> ProviderHealth:
        if not config.enabled:
            return ProviderHealth(name=self.name, role="reasoning", enabled=False, message="disabled")
        if not config.base_url or not config.model:
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message="base_url or model missing")
        start = time.perf_counter()
        try:
            response = httpx.get(f"{config.base_url.rstrip('/')}/api/tags", timeout=config.timeout_s)
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code >= 400:
                return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message=f"HTTP {response.status_code}", latency_ms=latency_ms)
            payload = response.json()
            names = {
                item.get("name") or item.get("model")
                for item in payload.get("models", [])
                if isinstance(item, dict)
            }
            if names and not any(_ollama_model_matches(config.model, name) for name in names):
                return ProviderHealth(
                    name=self.name,
                    role="reasoning",
                    enabled=True,
                    healthy=False,
                    message=f"model {config.model} not pulled; run `ollama pull {config.model}`",
                    latency_ms=latency_ms,
                )
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=True, message="ready", latency_ms=latency_ms)
        except Exception as exc:  # pragma: no cover - network dependent
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message=str(exc))

    def reason(
        self,
        *,
        config: ProviderConfig,
        prompt: str,
        image_bytes: Optional[bytes],
        context: list[Observation],
    ) -> Answer:
        messages = [{"role": "user", "content": self._format_prompt(prompt, context, has_image=image_bytes is not None)}]
        if image_bytes:
            messages[0]["images"] = [_image_to_base64(image_bytes)]
        payload = {"model": config.model, "messages": messages, "stream": False}
        keep_alive = config.metadata.get("keep_alive")
        if keep_alive:
            payload["keep_alive"] = keep_alive
        response = httpx.post(
            f"{config.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        return Answer(text=content, provider=self.name, confidence=0.7)

    def _format_prompt(self, prompt: str, context: list[Observation], *, has_image: bool) -> str:
        if has_image:
            lines = [
                "You are analyzing the current live camera frame.",
                "Base the answer on the current frame first.",
                "Use recent observation memory only as secondary context, and ignore it if it conflicts with the current frame.",
                "",
                prompt,
            ]
        else:
            lines = [prompt]
        if not context:
            return "\n".join(lines)
        lines.extend(["", "Recent observations (secondary context):"])
        for observation in context[:5]:
            lines.append(f"- {observation.created_at.isoformat()}: {observation.summary or 'No prior summary'}")
        return "\n".join(lines)


class CloudReasoningProvider:
    name = "cloud"

    def health(self, config: ProviderConfig) -> ProviderHealth:
        if not config.enabled:
            return ProviderHealth(name=self.name, role="reasoning", enabled=False, message="disabled")
        if not config.base_url or not config.model:
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message="base_url or model missing")
        if not config.api_key:
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message="api_key missing")
        return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=True, message="configured")

    def reason(
        self,
        *,
        config: ProviderConfig,
        prompt: str,
        image_bytes: Optional[bytes],
        context: list[Observation],
    ) -> Answer:
        content: list[dict] = [{"type": "text", "text": self._format_prompt(prompt, context, has_image=image_bytes is not None)}]
        if image_bytes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_image_to_base64(image_bytes)}"},
                }
            )
        response = httpx.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.2,
            },
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()
        return Answer(text=text, provider=self.name, confidence=0.75)

    def _format_prompt(self, prompt: str, context: list[Observation], *, has_image: bool) -> str:
        if has_image:
            lines = [
                "You are analyzing the current live camera frame.",
                "Base the answer on the current frame first.",
                "Use relevant observation memory only as secondary context.",
                "",
                prompt,
            ]
        else:
            lines = [prompt]
        if not context:
            return "\n".join(lines)
        lines.extend(["", "Relevant observation memory (secondary context):"])
        for observation in context[:5]:
            lines.append(f"- {observation.created_at.isoformat()} | {observation.summary or 'No summary'}")
        return "\n".join(lines)


class MlxReasoningProvider:
    name = "mlx"

    _HEALTH_CACHE_TTL_S = 30.0

    def __init__(self) -> None:
        import atexit
        import threading

        self._daemon: Optional[subprocess.Popen] = None
        self._daemon_lock = threading.Lock()
        self._health_cache: Optional[tuple[ProviderHealth, float]] = None
        atexit.register(self.shutdown)

    # ── lifecycle ──────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Terminate the daemon process if running. Safe to call multiple times."""
        with self._daemon_lock:
            proc = self._daemon
            self._daemon = None
        if proc is not None:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _daemon_alive(self) -> bool:
        return self._daemon is not None and self._daemon.poll() is None

    def _ensure_daemon(self, config: ProviderConfig) -> subprocess.Popen:
        """Start the daemon if not already running. Returns the Popen handle."""
        with self._daemon_lock:
            if self._daemon_alive():
                return self._daemon  # type: ignore[return-value]
            # Kill stale process if it exited
            if self._daemon is not None:
                try:
                    self._daemon.kill()
                except Exception:
                    pass
                self._daemon = None
            argv = self._resolve_wrapper_prefix(config)
            self._daemon = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            return self._daemon

    def _send_receive(self, config: ProviderConfig, payload: dict, timeout_s: float = 150.0) -> dict:
        """Send a JSON line to the daemon and read back a JSON line response."""
        import select

        daemon = self._ensure_daemon(config)
        with self._daemon_lock:
            try:
                line = json.dumps(payload) + "\n"
                daemon.stdin.write(line)  # type: ignore[union-attr]
                daemon.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError) as exc:
                self._daemon = None
                raise RuntimeError(f"daemon stdin broken: {exc}") from exc

            # Read response with loop to skip non-JSON native stdout pollution
            response_line = ""
            try:
                stdout_fd = daemon.stdout.fileno()  # type: ignore[union-attr]
                while True:
                    ready, _, _ = select.select([stdout_fd], [], [], timeout_s)
                    if not ready:
                        raise RuntimeError(f"daemon response timed out after {timeout_s}s")
                    
                    line = daemon.stdout.readline()  # type: ignore[union-attr]
                    if not line:
                        break  # EOF 
                        
                    line_str = line.strip()
                    if not line_str:
                        continue
                        
                    if line_str.startswith("{"):
                        response_line = line
                        break
                    else:
                        from .observability import get_logger
                        get_logger("runtime").warning(f"mlx background warning: {line_str}")

            except (OSError, ValueError) as exc:
                self._daemon = None
                raise RuntimeError(f"daemon stdout error: {exc}") from exc

        if not response_line or not response_line.strip():
            if not self._daemon_alive():
                self._daemon = None
                raise RuntimeError("daemon exited unexpectedly")
            raise RuntimeError("daemon returned empty response")
        try:
            result = json.loads(response_line.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"daemon returned invalid JSON: {response_line.strip()[:200]}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("daemon returned non-object JSON")
        return result

    # ── health ─────────────────────────────────────────────────────

    def health(self, config: ProviderConfig) -> ProviderHealth:
        if not config.enabled:
            return ProviderHealth(name=self.name, role="reasoning", enabled=False, message="disabled")
        if not self._command_string(config):
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message="metadata.command missing")
        if not (config.model_path or config.model):
            return ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message="model_path missing")

        # Return cached result if fresh
        if self._health_cache is not None:
            cached_result, cached_at = self._health_cache
            if (time.time() - cached_at) < self._HEALTH_CACHE_TTL_S:
                return cached_result

        # Tier 2: If daemon is already running, ping it
        if self._daemon_alive():
            try:
                result = self._send_receive(config, {"type": "healthcheck"}, timeout_s=5.0)
                if result.get("success"):
                    health = ProviderHealth(
                        name=self.name, role="reasoning", enabled=True, healthy=True,
                        message=str(result.get("message") or "daemon alive"),
                    )
                    self._health_cache = (health, time.time())
                    return health
            except Exception:
                pass  # Fall through to lightweight probe

        # Tier 1: Lightweight subprocess probe (no model load)
        try:
            result = self._run_healthcheck_subprocess(config)
        except Exception as exc:
            health = ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=False, message=_short_error_message(exc))
            self._health_cache = (health, time.time())
            return health
        if not result.get("success"):
            health = ProviderHealth(
                name=self.name, role="reasoning", enabled=True, healthy=False,
                message=str(result.get("message") or "mlx healthcheck failed"),
            )
            self._health_cache = (health, time.time())
            return health
        msg = str(result.get("message") or "configured")
        if not self._daemon_alive():
            msg += " (model loads on first use)"
        health = ProviderHealth(name=self.name, role="reasoning", enabled=True, healthy=True, message=msg)
        self._health_cache = (health, time.time())
        return health

    # ── reasoning ──────────────────────────────────────────────────

    def reason(
        self,
        *,
        config: ProviderConfig,
        prompt: str,
        image_bytes: Optional[bytes],
        context: list[Observation],
        image_path: Optional[str] = None,
    ) -> Answer:
        has_image = bool(image_bytes or image_path)
        rendered_prompt = self._format_prompt(prompt, context, has_image=has_image)
        image_b64: Optional[str] = None
        if image_bytes is not None:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        elif image_path is not None:
            try:
                image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
            except Exception:
                pass
        payload: dict = {
            "prompt": rendered_prompt,
            "image_base64": image_b64,
            "max_tokens": int(config.metadata.get("max_tokens", 512)),
        }
        result = self._send_receive(config, payload, timeout_s=config.timeout_s)
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        text = str(result.get("text") or "").strip()
        if not text:
            raise RuntimeError("mlx daemon returned empty output")
        return Answer(text=text, provider=self.name, confidence=0.6)

    # ── prompt formatting ──────────────────────────────────────────

    def _format_prompt(self, prompt: str, context: list[Observation], *, has_image: bool = False) -> str:
        if has_image:
            lines = [
                "Analyze the current live camera frame.",
                "Trust the current frame first.",
                "Use recent memory only as secondary context if it helps disambiguate the scene.",
                "",
                prompt,
            ]
        else:
            lines = [prompt]
        if not context:
            return "\n".join(lines)
        lines.extend(["", "Recent memory summaries (secondary context):"])
        for observation in context[:5]:
            lines.append(f"- {observation.summary or 'No summary'}")
        return "\n".join(lines)

    # ── internal helpers ───────────────────────────────────────────

    def _command_string(self, config: ProviderConfig) -> str:
        return str(config.metadata.get("command", "")).strip()

    def _resolve_wrapper_prefix(self, config: ProviderConfig) -> list[str]:
        command = self._command_string(config)
        parts = shlex.split(command)
        if not parts:
            raise RuntimeError("metadata.command missing")
        if "mlx_reasoner.py" not in command:
            raise RuntimeError("mlx command must reference mlx_reasoner.py")
        executable = parts[0]
        resolved = shutil.which(executable) if not Path(executable).exists() else executable
        if resolved is None:
            raise RuntimeError(f"{executable} not found")
        script_path = next((part for part in parts if part.endswith("mlx_reasoner.py")), None)
        if script_path is None:
            raise RuntimeError("mlx_reasoner.py not found in command")
        return [resolved, script_path]

    def _run_healthcheck_subprocess(self, config: ProviderConfig) -> dict:
        """Run a lightweight --healthcheck subprocess (no model loading)."""
        argv = self._resolve_wrapper_prefix(config)
        argv.append("--healthcheck")
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        # Parse the last line of stdout as JSON (in case mlx prints warnings above)
        payload: Optional[dict] = None
        if stdout:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        payload = parsed
                        break
                except json.JSONDecodeError:
                    continue
        if completed.returncode != 0:
            if payload is not None:
                return payload
            raise RuntimeError(_short_probe_message(stderr or stdout))
        if payload is None:
            raise RuntimeError("mlx healthcheck returned malformed json")
        return payload


@dataclass
class CircuitState:
    failures: int = 0
    opened_until: float = 0.0
    last_error: str = ""


@dataclass
class ReasoningOutcome:
    answer: Optional[Answer]
    trace: list[ReasoningTraceEntry]


class ProviderRegistry:
    def __init__(self) -> None:
        self.basic = BasicVisionProvider()
        self.dinov2 = DinoV2PerceptionProvider()
        self.onnx = OnnxPerceptionProvider()
        self.ollama = OllamaReasoningProvider()
        self.cloud = CloudReasoningProvider()
        self.mlx = MlxReasoningProvider()
        self._circuits: dict[str, CircuitState] = {}

    def get(self, provider_name: str):
        return getattr(self, provider_name, None)

    def health_snapshot(self, settings: RuntimeSettings) -> list[ProviderHealth]:
        providers = settings.providers
        entries: list[ProviderHealth] = []
        entries.append(self.dinov2.health(providers.get("dinov2", ProviderConfig(name="dinov2", enabled=False))))
        entries.append(self.onnx.health(providers.get("onnx", ProviderConfig(name="onnx", enabled=False))))
        entries.append(self.basic.health(providers.get("basic", ProviderConfig(name="basic", enabled=False))))
        entries.append(
            ProviderHealth(
                name="coreml",
                role="perception",
                enabled=providers.get("coreml", ProviderConfig(name="coreml", enabled=False)).enabled,
                healthy=False,
                message="Handled on-device in iOS client",
            )
        )
        entries.append(
            ProviderHealth(
                name="tflite",
                role="perception",
                enabled=providers.get("tflite", ProviderConfig(name="tflite", enabled=False)).enabled,
                healthy=False,
                message="Handled on-device in Android client",
            )
        )
        entries.append(self._provider_health("ollama", settings))
        entries.append(self._provider_health("mlx", settings))
        entries.append(self._provider_health("cloud", settings))
        entries.append(
            ProviderHealth(
                name="local",
                role="search",
                enabled=providers.get("local", ProviderConfig(name="local", enabled=True)).enabled,
                healthy=True,
                message="SQLite vector search is available",
            )
        )
        return entries

    def perceive(self, settings: RuntimeSettings, image: Image.Image) -> tuple[list[float], str, float, dict]:
        providers = settings.providers
        order = [settings.primary_perception_provider, *settings.fallback_order, "basic"]
        seen: set[str] = set()
        for provider_name in order:
            if provider_name in seen:
                continue
            seen.add(provider_name)
            if provider_name == "dinov2":
                config = providers.get("dinov2", ProviderConfig(name="dinov2", enabled=False))
                health = self.dinov2.health(config)
                if health.healthy:
                    embedding, confidence, metadata = self.dinov2.perceive(image, config)
                    return embedding, "dinov2", confidence, metadata
            if provider_name == "onnx":
                config = providers.get("onnx", ProviderConfig(name="onnx", enabled=False))
                health = self.onnx.health(config)
                if health.healthy:
                    embedding, confidence, metadata = self.onnx.perceive(image, config)
                    return embedding, "onnx", confidence, metadata
            elif provider_name == "basic":
                config = providers.get("basic", ProviderConfig(name="basic", enabled=True))
                health = self.basic.health(config)
                if health.healthy:
                    embedding, confidence, metadata = self.basic.perceive(image)
                    return embedding, "basic", confidence, metadata
        raise RuntimeError("No healthy perception provider available")

    def object_proposals(
        self,
        settings: RuntimeSettings,
        image: Image.Image,
        *,
        provider_name: str,
        max_proposals: int = 12,
    ) -> list[BoundingBox]:
        providers = settings.providers
        candidates: list[tuple[BoundingBox, float]] = []

        # 1. Gather proposal regions
        if provider_name == "dinov2":
            config = providers.get("dinov2", ProviderConfig(name="dinov2", enabled=False))
            proposals = self.dinov2.object_proposals(image, config, max_proposals=max_proposals)
            if proposals:
                candidates = [(p, p.score or 1.0) for p in proposals]

        if not candidates:
            candidates = self._proposal_candidates(image, max_proposals=max_proposals)

        if not candidates:
            return []

        # 2. Label the regions using the active classifier (ONNX or basic)
        classifier_name = "onnx" if providers.get("onnx", ProviderConfig(name="onnx", enabled=False)).enabled else "basic"
        if classifier_name == "onnx" and not self.onnx.health(providers["onnx"]).healthy:
            classifier_name = "basic"
        config = providers.get(classifier_name, ProviderConfig(name=classifier_name, enabled=classifier_name == "basic"))
        
        scored: list[BoundingBox] = []
        width, height = image.size
        
        for index, (region, saliency) in enumerate(candidates):
            left = int(round(region.x * width))
            top = int(round(region.y * height))
            right = int(round((region.x + region.width) * width))
            bottom = int(round((region.y + region.height) * height))
            if right <= left or bottom <= top:
                continue
                
            crop = image.crop((left, top, right, bottom))
            try:
                if classifier_name == "onnx":
                    _, confidence, metadata = self.onnx.perceive(crop, config)
                else:
                    _, confidence, metadata = self.basic.perceive(crop)
            except Exception:
                continue
                
            label = ""
            if classifier_name == "onnx":
                label = _meaningful_label(metadata.get("top_label") or "")
                if label in SUSPICIOUS_CLASSIFIER_LABELS:
                    label = ""
            if not label:
                label = _proposal_fallback_label(region, index)
                
            score = _label_rank(label, confidence, saliency)
            proposal = BoundingBox(
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                label=label,
                score=round(float(saliency if provider_name == "dinov2" else score), 4),
            )
            # Keep all spatially distinct proposals, do NOT deduplicate by label!
            scored.append(proposal)
                
        # Filter weak proposals immediately
        scored = [p for p in scored if float(p.score or 0.0) >= 0.15]
        scored = sorted(scored, key=lambda item: float(item.score or 0.0), reverse=True)
        return scored[:max_proposals]


    def _proposal_candidates(self, image: Image.Image, *, max_proposals: int = 12) -> list[tuple[BoundingBox, float]]:
        regions = _grid_regions(image)
        if not regions:
            return []
        # Filter out regions with low saliency so we don't always emit 5 boxes
        saliency_threshold = 0.15
        salient_regions = [(box, sal) for box, sal in regions if sal >= saliency_threshold]
        if not salient_regions:
            # If nothing passes threshold, keep only the best region
            salient_regions = [regions[0]]
        center_region = min(
            salient_regions,
            key=lambda item: abs((item[0].x + item[0].width / 2.0) - 0.5) + abs((item[0].y + item[0].height / 2.0) - 0.5),
        )
        quadrants = {
            "top_left": lambda region: region[0].x < 0.5 and region[0].y < 0.5,
            "top_right": lambda region: region[0].x >= 0.5 and region[0].y < 0.5,
            "bottom_left": lambda region: region[0].x < 0.5 and region[0].y >= 0.5,
            "bottom_right": lambda region: region[0].x >= 0.5 and region[0].y >= 0.5,
        }
        selected: list[tuple[BoundingBox, float]] = [center_region]
        seen = {
            (
                round(center_region[0].x, 4),
                round(center_region[0].y, 4),
                round(center_region[0].width, 4),
                round(center_region[0].height, 4),
            )
        }
        for predicate in quadrants.values():
            quadrant_region = next((region for region in salient_regions if predicate(region)), None)
            if quadrant_region is None:
                continue
            key = (
                round(quadrant_region[0].x, 4),
                round(quadrant_region[0].y, 4),
                round(quadrant_region[0].width, 4),
                round(quadrant_region[0].height, 4),
            )
            if key in seen:
                continue
            selected.append(quadrant_region)
            seen.add(key)
            if len(selected) >= max_proposals:
                break
        if len(selected) < max_proposals:
            for region in salient_regions:
                key = (round(region[0].x, 4), round(region[0].y, 4), round(region[0].width, 4), round(region[0].height, 4))
                if key in seen:
                    continue
                selected.append(region)
                seen.add(key)
                if len(selected) >= max_proposals:
                    break
        return selected

    def reason(
        self,
        settings: RuntimeSettings,
        *,
        prompt: str,
        image_bytes: Optional[bytes],
        image_path: Optional[str],
        context: list[Observation],
    ) -> ReasoningOutcome:
        trace: list[ReasoningTraceEntry] = []
        has_image = bool(image_bytes or image_path)
        for provider_name in self._reasoning_order(settings, has_image=has_image):
            health = self._provider_health(provider_name, settings)
            if provider_name == "mlx" and not has_image:
                trace.append(
                    ReasoningTraceEntry(
                        provider=provider_name,
                        healthy=health.healthy,
                        health_message=health.message,
                        attempted=False,
                        success=False,
                        error="image input not provided",
                    )
                )
                continue
            if not health.healthy:
                trace.append(
                    ReasoningTraceEntry(
                        provider=provider_name,
                        healthy=False,
                        health_message=health.message,
                        attempted=False,
                        success=False,
                        error=health.message,
                    )
                )
                continue
            started_at = time.perf_counter()
            try:
                if provider_name == "ollama":
                    config = settings.providers.get("ollama", ProviderConfig(name="ollama", enabled=False))
                    answer = self.ollama.reason(config=config, prompt=prompt, image_bytes=image_bytes, context=context)
                elif provider_name == "mlx":
                    config = settings.providers.get("mlx", ProviderConfig(name="mlx", enabled=False))
                    answer = self.mlx.reason(
                        config=config,
                        prompt=prompt,
                        image_bytes=image_bytes,
                        context=context,
                        image_path=image_path,
                    )
                elif provider_name == "cloud":
                    config = settings.providers.get("cloud", ProviderConfig(name="cloud", enabled=False))
                    if not self.cloud.health(config).healthy:
                        continue
                    answer = self.cloud.reason(config=config, prompt=prompt, image_bytes=image_bytes, context=context)
                else:
                    continue
                self._record_success(provider_name)
                trace.append(
                    ReasoningTraceEntry(
                        provider=provider_name,
                        healthy=True,
                        health_message=health.message,
                        attempted=True,
                        success=True,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                    )
                )
                return ReasoningOutcome(answer=answer, trace=trace)
            except Exception as exc:
                error = _short_error_message(exc)
                self._record_failure(provider_name, error)
                trace.append(
                    ReasoningTraceEntry(
                        provider=provider_name,
                        healthy=True,
                        health_message=health.message,
                        attempted=True,
                        success=False,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        error=error,
                    )
                )
                continue
        return ReasoningOutcome(answer=None, trace=trace)

    def _reasoning_order(self, settings: RuntimeSettings, *, has_image: bool) -> list[str]:
        backend = settings.reasoning_backend
        if backend == "disabled":
            return []
        if backend == "ollama":
            order = ["ollama", "cloud"]
        elif backend == "mlx":
            order = ["mlx", "ollama", "cloud"] if has_image else ["ollama", "cloud"]
        elif backend == "cloud":
            order = ["cloud"]
        else:
            order = ["ollama", "mlx", "cloud"] if has_image else ["ollama", "cloud"]
        if settings.local_reasoning_disabled:
            order = [name for name in order if name == "cloud"]
        return order

    def reset_circuits(self) -> None:
        self._circuits.clear()

    def _allow(self, provider_name: str) -> bool:
        state = self._circuits.setdefault(provider_name, CircuitState())
        return time.time() >= state.opened_until

    def _record_success(self, provider_name: str) -> None:
        state = self._circuits.setdefault(provider_name, CircuitState())
        state.failures = 0
        state.opened_until = 0.0
        state.last_error = ""

    def _record_failure(self, provider_name: str, error: str) -> None:
        state = self._circuits.setdefault(provider_name, CircuitState())
        state.failures += 1
        state.last_error = error
        if state.failures >= 2:
            state.opened_until = time.time() + 30

    def _provider_health(self, provider_name: str, settings: RuntimeSettings) -> ProviderHealth:
        providers = settings.providers
        if provider_name == "ollama":
            health = self.ollama.health(providers.get("ollama", ProviderConfig(name="ollama", enabled=False)))
        elif provider_name == "mlx":
            health = self.mlx.health(providers.get("mlx", ProviderConfig(name="mlx", enabled=False)))
        elif provider_name == "cloud":
            health = self.cloud.health(providers.get("cloud", ProviderConfig(name="cloud", enabled=False)))
        else:
            raise KeyError(provider_name)
        if health.healthy and not self._allow(provider_name):
            state = self._circuits.setdefault(provider_name, CircuitState())
            remaining_s = max(int(state.opened_until - time.time()), 0)
            reason = state.last_error or "recent failures"
            return health.model_copy(
                update={
                    "healthy": False,
                    "message": f"circuit open for {remaining_s}s after {reason}",
                }
            )
        return health
