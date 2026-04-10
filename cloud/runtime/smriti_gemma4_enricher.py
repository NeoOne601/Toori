"""
smriti_gemma4_enricher.py — Plugs Gemma 4 narration into ingestion and live tick.

Integration call sites:
  A. smriti_ingestion.py  _process_job()    — after setu_descriptions assigned
  B. service.py           living_lens_tick() — before returning response dict
"""
from __future__ import annotations
import asyncio, logging
import re
from typing import Any

from cloud.perception.gemma_semantic_extractor import get_default_gemma_semantic_extractor

from .mlx_adapter import make_mlx_bridge_call

logger = logging.getLogger(__name__)

def _attr(obj, key, default=None):
    return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)


_FALLBACK_ANCHOR_LABELS = {
    "person_torso": "person standing",
    "chair_seated": "chair",
    "cylindrical_object": "cylindrical object",
    "screen_display": "display screen",
    "desk_surface": "desk surface",
    "hand_region": "hand or arm",
    "spherical_object": "round object",
    "background_plane": "background surface",
    "telescope": "telescope",
    "neck_brace": "neck brace",
    "tie": "tie",
    "chair": "chair",
    "wall_clock": "wall clock",
    "poster": "poster",
    "switch": "switch",
    "lamp": "lamp",
    "sunglasses": "sunglasses",
    "lab_coat": "lab coat",
    "barrette": "barrette",
    "unknown": "object",
}

_ABSURD_LABELS = {
    "hot dog near velvet",
    "hand plane near windsor tie",
    "lemon near bra",
    "rgb histogram+edge histogram",
    "rgb histogram edge histogram",
    "dominant color",
    "brightness label",
    "edge label",
}

_PLACEHOLDER_LABEL_RE = re.compile(r"^(?:entity|proposal|candidate|tracked(?: region| object)?|object)\s*[-_ ]*\d*$")


def _fallback_anchor_label(anchor_name: str, depth_stratum: str = "unknown") -> str:
    cleaned = str(anchor_name or "").replace("_", " ").strip().lower() or "object"
    label = _FALLBACK_ANCHOR_LABELS.get(str(anchor_name or "").strip(), cleaned)
    if label == "object" and depth_stratum and depth_stratum != "unknown":
        return f"object {depth_stratum}".strip()
    return label


def _context_object_hint(tvlc_context: str | None) -> str | None:
    context = str(tvlc_context or "").strip().lower()
    if not context:
        return None
    if "prototype_matches=" in context:
        fragment = context.split("prototype_matches=", 1)[1]
        fragment = fragment.split("slot_activations=", 1)[0]
        for entry in fragment.split(";"):
            label, _, _score = entry.partition(":")
            normalized = " ".join(label.replace("_", " ").split()).strip()
            if normalized:
                return normalized
    curated = (
        ("orange telescope", "telescope"),
        ("telescope", "telescope"),
        ("sunglass", "sunglasses"),
        ("neck brace", "neck brace"),
        ("lab coat", "lab coat"),
        ("barrette", "barrette"),
        ("chair", "chair"),
        ("wall clock", "wall clock"),
        ("poster", "poster"),
        ("switch", "switch"),
        ("lamp", "lamp"),
        ("tie", "tie"),
    )
    for needle, label in curated:
        if needle in context:
            return label
    return None


def _is_bad_open_vocab_label(value: str) -> bool:
    normalized = " ".join(str(value or "").replace("_", " ").split()).strip().lower()
    if not normalized:
        return True
    if normalized in _ABSURD_LABELS:
        return True
    if _PLACEHOLDER_LABEL_RE.match(normalized):
        return True
    if " near " in normalized or " behind " in normalized or " left of " in normalized or " right of " in normalized:
        return True
    if "histogram" in normalized or "descriptor" in normalized:
        return True
    return len(normalized.split()) > 6


def _anchor_payload(anchor: Any) -> dict[str, Any]:
    template_name = str(_attr(anchor, "template_name", "") or _attr(anchor, "name", "") or "object")
    return {
        "template_name": template_name,
        "confidence": float(_attr(anchor, "confidence", 0.0) or 0.0),
        "patch_indices": list(_attr(anchor, "patch_indices", []) or []),
        "depth_stratum": str(_attr(anchor, "depth_stratum", "unknown") or "unknown"),
        "open_vocab_label": _attr(anchor, "open_vocab_label", None),
    }


class SmetiGemma4Enricher:
    def __init__(self, mlx_provider=None, mlx_config=None):
        self._p = mlx_provider
        self._config = mlx_config
        self._ok = mlx_provider is not None and (mlx_config is None or bool(getattr(mlx_config, "enabled", True)))
        self._semantic_checked = False
        self._semantic_extractor = None
        if not self._ok:
            logger.debug("Gemma4Enricher: no MLX provider — no-op mode")

    def _get_semantic_extractor(self):
        if self._semantic_checked:
            return self._semantic_extractor
        self._semantic_checked = True
        try:
            self._semantic_extractor = get_default_gemma_semantic_extractor()
        except Exception:
            self._semantic_extractor = None
        return self._semantic_extractor

    def _resolve_semantic_label(
        self,
        candidate: str,
        *,
        fallback: str,
        tvlc_context: str | None,
    ) -> tuple[str, dict[str, Any]]:
        normalized = str(candidate or "").strip()
        context_hint = _context_object_hint(tvlc_context)
        evidence: dict[str, Any] = {
            "policy": "strict_evidence_gate",
            "context_hint": context_hint,
            "extractor_ready": False,
            "semantic_gate_passed": True,
        }
        if not normalized or _is_bad_open_vocab_label(normalized):
            evidence["semantic_gate_passed"] = False
            evidence["reason"] = "bad_or_empty_candidate"
            return fallback, evidence
        extractor = self._get_semantic_extractor()
        if extractor is None or not context_hint:
            evidence["reason"] = "no_semantic_reference"
            return normalized, evidence
        evidence["extractor_ready"] = True
        similarity = float(extractor.cosine_similarity(normalized, context_hint))
        evidence["semantic_similarity"] = round(similarity, 4)
        evidence["semantic_reference"] = context_hint
        if normalized != context_hint and similarity < 0.18:
            evidence["semantic_gate_passed"] = False
            evidence["reason"] = "semantic_mismatch_to_tvlc_hint"
            return context_hint or fallback, evidence
        evidence["reason"] = "semantic_match"
        return normalized, evidence

    async def get_open_vocab_label_with_evidence(
        self,
        anchor_name: str,
        depth_stratum: str,
        confidence: float,
        patch_count: int,
        tvlc_context: str | None = None,
        connector_type: str | None = None,
        image_base64: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        fallback = _fallback_anchor_label(anchor_name, depth_stratum)
        if connector_type != "tvlc_trained":
            return fallback, {
                "policy": "strict_evidence_gate",
                "reason": "connector_not_trained",
                "context_hint": _context_object_hint(tvlc_context),
                "extractor_ready": self._get_semantic_extractor() is not None,
                "semantic_gate_passed": False,
            }
        context_hint = _context_object_hint(tvlc_context)
        if context_hint:
            fallback = context_hint
        if not self._ok:
            return fallback, {
                "policy": "strict_evidence_gate",
                "reason": "mlx_unavailable",
                "context_hint": context_hint,
                "extractor_ready": self._get_semantic_extractor() is not None,
                "semantic_gate_passed": False,
            }

        prompt = (
            f"JEPA geometric evidence:\n"
            f"- SAG anchor template: {anchor_name}\n"
            f"- Depth stratum: {depth_stratum}\n"
            f"- Geometric confidence: {confidence:.2f}\n"
            f"- Patch coverage: {patch_count}/196 patches\n\n"
            f"- TVLC visual context: {tvlc_context or 'none'}\n"
            f"- Connector type: {connector_type or 'unknown'}\n\n"
            "Return a short atomic object label only. "
            "Do not describe relations between objects. "
            "Do not use phrases containing 'near', 'behind', 'left of', or 'right of'. "
            "Prefer common object names over ImageNet jargon. "
            "If TVLC context strongly identifies the object, follow it. "
            "Output ONLY the object label."
        )

        try:
            result = await asyncio.wait_for(
                self._mlx(prompt=prompt, image_base64=image_base64, system=None, max_tokens=16),
                timeout=3.0,
            )
            text = str(result.get("text") or "").strip().strip("\"'")
            label, evidence = self._resolve_semantic_label(
                text,
                fallback=fallback,
                tvlc_context=tvlc_context,
            )
            evidence["mlx_model"] = result.get("model")
            return label, evidence
        except (asyncio.TimeoutError, Exception):
            return fallback, {
                "policy": "strict_evidence_gate",
                "reason": "mlx_timeout_or_error",
                "context_hint": context_hint,
                "extractor_ready": self._get_semantic_extractor() is not None,
                "semantic_gate_passed": False,
            }

    async def get_open_vocab_label(
        self,
        anchor_name: str,
        depth_stratum: str,
        confidence: float,
        patch_count: int,
        tvlc_context: str | None = None,
        connector_type: str | None = None,
        image_base64: str | None = None,
    ) -> str:
        label, _evidence = await self.get_open_vocab_label_with_evidence(
            anchor_name=anchor_name,
            depth_stratum=depth_stratum,
            confidence=confidence,
            patch_count=patch_count,
            tvlc_context=tvlc_context,
            connector_type=connector_type,
            image_base64=image_base64,
        )
        return label

    async def enrich_ingested_media(self, media):
        existing = _attr(media,"setu_descriptions",[])
        if not self._ok: return existing
        anchors = [
            payload
            for anchor in (_attr(media,"anchor_matches",[]) or [])
            for payload in [_anchor_payload(anchor)]
            if payload["template_name"] and payload["confidence"] >= 0.55
        ]
        if not anchors: return existing
        try:
            from .gemma4_bridge import Gemma4Bridge
            bridge = Gemma4Bridge(self._mlx)
            n = await asyncio.wait_for(bridge.narrate_anchor(
                anchor_matches=anchors,
                depth_strata=_attr(media,"depth_strata",None) if isinstance(_attr(media,"depth_strata",None),dict) else None,
                setu_descriptions=existing,
                observation_id=str(_attr(media,"id","unknown")),
                image_base64=None), timeout=8.0)
            rec = {"gate":{"passes":True,"consistency_score":n.confidence,"failure_reasons":[],
                           "anchor_name":n.anchor_name,"depth_stratum":n.depth_stratum,
                           "estimated_hallucination_risk":n.hallucination_risk,
                           "uncertainty_map":[],"narrator":"gemma4"},
                   "description":{"text":n.text,"confidence":n.confidence,
                                   "anchor_basis":n.anchor_name,"depth_stratum":n.depth_stratum,
                                   "is_uncertain":n.confidence<0.5,
                                   "hallucination_risk":n.hallucination_risk,
                                   "uncertainty_map":None,"narrator":"gemma4",
                                   "latency_ms":n.latency_ms}}
            return [rec]+list(existing)
        except asyncio.TimeoutError: logger.warning("Gemma4: narration timeout")
        except Exception as e: logger.warning("Gemma4: narration error: %s",e)
        return existing

    async def live_tick_alert(self, tick_result, scene_state, entity_tracks):
        if not self._ok: return {"text":"stable","is_alert":False,"latency_ms":0.0}
        sd = scene_state.dict() if hasattr(scene_state,"dict") else (scene_state if isinstance(scene_state,dict) else {})
        tl = [t.dict() if hasattr(t,"dict") else (t if isinstance(t,dict) else {}) for t in (entity_tracks or [])]
        jt = tick_result.get("jepa_tick") or tick_result or {}
        try:
            from .gemma4_bridge import Gemma4Bridge
            a = await asyncio.wait_for(Gemma4Bridge(self._mlx).proactive_alert(sd,tl,jt),timeout=5.0)
            return {"text":a.text,"is_alert":a.is_alert,"latency_ms":a.latency_ms}
        except asyncio.TimeoutError: return {"text":"stable","is_alert":False,"latency_ms":0.0}
        except Exception as e: logger.debug("Gemma4: alert error: %s",e); return {"text":"stable","is_alert":False,"latency_ms":0.0}

    async def reformulate_recall_query(self, q):
        if not self._ok: return {"setu_query":q,"depth_stratum":None,"person_filter":None,"time_hint":None,"confidence_min":0.3}
        try:
            from .gemma4_bridge import Gemma4Bridge
            r = await asyncio.wait_for(Gemma4Bridge(self._mlx).reformulate_query(q),timeout=5.0)
            return {"setu_query":r.setu_query,"depth_stratum":r.depth_stratum,
                    "person_filter":r.person_filter,"time_hint":r.time_hint,
                    "confidence_min":r.confidence_min,"gemma4_latency_ms":r.latency_ms}
        except Exception: return {"setu_query":q,"depth_stratum":None,"person_filter":None,"time_hint":None,"confidence_min":0.3}

    async def _mlx(self, prompt, image_base64=None, system=None, max_tokens=256):
        if not self._p:
            return {"text": "", "latency_ms": 0.0, "model": "none"}
        bridge_call = make_mlx_bridge_call(self._p, self._config)
        return await bridge_call(
            prompt=prompt,
            image_base64=image_base64,
            system=system,
            max_tokens=max_tokens,
        )
