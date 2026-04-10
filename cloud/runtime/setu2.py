"""
Setu-2: JEPA-to-Language EBM Bridge.

Upgraded bridge that operates on DEPTH-SEPARATED, ANCHOR-CONFIRMED,
WORLD-MODEL-ALIGNED, GATE-PASSED embeddings only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from cloud.jepa_service.confidence_gate import GateResult
from cloud.runtime.observability import get_logger

log = get_logger("setu2")

QUERY_DIM: int = 384
JEPA_DIM: int = 128


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _xavier(in_dim: int, out_dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return {
        "W": rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32),
        "b": np.zeros(out_dim, dtype=np.float32),
    }


@dataclass
class SetuDescription:
    text: str
    confidence: float
    is_uncertain: bool
    anchor_basis: str
    depth_stratum: str
    hallucination_risk: float
    uncertainty_map: Optional[list]
    tvlc_context: Optional[str] = None
    connector_type: Optional[str] = None

    def to_dict(self) -> dict:
        payload = {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "is_uncertain": self.is_uncertain,
            "anchor_basis": self.anchor_basis,
            "depth_stratum": self.depth_stratum,
            "hallucination_risk": round(float(self.hallucination_risk), 4),
            "uncertainty_map": self.uncertainty_map,
        }
        if self.tvlc_context:
            payload["tvlc_context"] = self.tvlc_context
        if self.connector_type:
            payload["connector_type"] = self.connector_type
        return payload


ANCHOR_DESCRIPTIONS: dict[str, str] = {
    "person_torso": "Person in {depth}",
    "chair_seated": "Seated surface in {depth}",
    "cylindrical_object": "Cylindrical object in {depth}",
    "screen_display": "Display screen in {depth}",
    "desk_surface": "Desk surface in {depth}",
    "hand_region": "Hand or arm in {depth}",
    "spherical_object": "Rounded object in {depth}",
    "background_plane": "Background surface",
    "unknown": "Unidentified region in {depth}",
}


class Setu2Bridge:
    def __init__(
        self,
        w_diagonal: Optional[np.ndarray] = None,
        seed: int = 20260327,
    ) -> None:
        self._layers = [
            _xavier(QUERY_DIM, 512, seed),
            _xavier(512, 256, seed + 1),
            _xavier(256, 128, seed + 2),
            _xavier(128, JEPA_DIM, seed + 3),
        ]
        self._W = w_diagonal if w_diagonal is not None else np.ones(JEPA_DIM, dtype=np.float32)
        self._tvlc_checked = False
        self._tvlc_connector: Any = None
        self._tvlc_cache_patch_id: Optional[int] = None
        self._tvlc_cache_value: tuple[Optional[str], Optional[str]] = (None, None)

    def project_query(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        if query.shape[0] != QUERY_DIM:
            if query.shape[0] > QUERY_DIM:
                query = query[:QUERY_DIM]
            else:
                query = np.pad(query, (0, QUERY_DIM - query.shape[0]))

        projected = self._forward(query)
        corpus = np.asarray(corpus_embeddings, dtype=np.float32)
        diff = corpus - projected[None, :]
        weighted = diff * self._W[None, :]
        energies = np.sum(weighted**2, axis=1)
        return energies.astype(np.float32)

    def describe_region(
        self,
        gate_result: GateResult,
        patch_tokens: Optional[np.ndarray] = None,
        anchor_match: Optional[dict[str, Any]] = None,
    ) -> SetuDescription:
        tvlc_context, connector_type = self._resolve_tvlc_context(patch_tokens)
        if not gate_result.passes:
            return SetuDescription(
                text=f"Unclear region ({', '.join(gate_result.failure_reasons[:1])})",
                confidence=gate_result.consistency_score,
                is_uncertain=True,
                anchor_basis=gate_result.anchor_name,
                depth_stratum=gate_result.depth_stratum,
                hallucination_risk=gate_result.estimated_hallucination_risk,
                uncertainty_map=gate_result.uncertainty_map.tolist(),
                tvlc_context=tvlc_context,
                connector_type=connector_type,
            )

        template = ANCHOR_DESCRIPTIONS.get(gate_result.anchor_name, "Object in {depth}")
        depth_label = gate_result.depth_stratum
        text = template.format(depth=depth_label)
        return SetuDescription(
            text=text,
            confidence=gate_result.consistency_score,
            is_uncertain=False,
            anchor_basis=gate_result.anchor_name,
            depth_stratum=depth_label,
            hallucination_risk=gate_result.estimated_hallucination_risk,
            uncertainty_map=None,
            tvlc_context=tvlc_context,
            connector_type=connector_type,
        )

    def describe(
        self,
        patch_tokens: Optional[np.ndarray],
        anchor_match: Optional[dict[str, Any]],
        gate_result: Optional[GateResult | dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if isinstance(gate_result, GateResult):
            return self.describe_region(
                gate_result,
                patch_tokens=patch_tokens,
                anchor_match=anchor_match,
            ).to_dict()
        return self._describe_from_anchor(anchor_match, gate_result, patch_tokens).to_dict()

    def update_metric_w(
        self,
        positive_pairs: list[tuple[np.ndarray, np.ndarray]],
        negative_pairs: list[tuple[np.ndarray, np.ndarray]],
        learning_rate: float = 0.01,
    ) -> None:
        for query_embedding, positive_embedding in positive_pairs:
            projected = self._forward(np.asarray(query_embedding, dtype=np.float32))
            diff = np.asarray(positive_embedding, dtype=np.float32) - projected
            gradient = -2.0 * (diff**2) * learning_rate
            self._W = np.clip(self._W + gradient, 0.01, 10.0)

        for query_embedding, negative_embedding in negative_pairs:
            projected = self._forward(np.asarray(query_embedding, dtype=np.float32))
            diff = np.asarray(negative_embedding, dtype=np.float32) - projected
            gradient = 2.0 * np.exp(-(diff**2)) * learning_rate
            self._W = np.clip(self._W + gradient, 0.01, 10.0)

        log.debug("setu2_w_updated", w_mean=float(self._W.mean()))

        # Persist updated weights — W-matrix now survives restart
        try:
            from cloud.api.main import app
            db = getattr(app.state, "runtime", None)
            if db and hasattr(db, "smriti_db") and hasattr(db.smriti_db, "_persist_wmatrix_to_db"):
                for _idx, _val in enumerate(self._W):
                    db.smriti_db._persist_wmatrix_to_db(component=f"dim_{_idx}", new_weight=float(_val), feedback_count=1)
        except Exception as _persist_e:
            pass  # logged inside _persist_wmatrix_to_db — safe to swallow here

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for index, layer in enumerate(self._layers):
            x = x @ layer["W"] + layer["b"]
            if index < len(self._layers) - 1:
                x = _relu(x)
        return x.astype(np.float32)

    def _describe_from_anchor(
        self,
        anchor_match: Optional[dict[str, Any]],
        gate_result: Optional[dict[str, Any]],
        patch_tokens: Optional[np.ndarray],
    ) -> SetuDescription:
        payload = anchor_match if isinstance(anchor_match, dict) else {}
        gate_payload = gate_result if isinstance(gate_result, dict) else {}
        anchor_name = str(payload.get("template_name") or payload.get("name") or "unknown")
        depth_stratum = str(payload.get("depth_stratum") or gate_payload.get("depth_stratum") or "unknown")
        confidence = float(payload.get("confidence", 0.0) or gate_payload.get("consistency_score", 0.0) or 0.0)
        failure_reasons = gate_payload.get("failure_reasons") or []
        passes = bool(gate_payload.get("passes", confidence >= 0.55))
        uncertainty_map = gate_payload.get("uncertainty_map")
        if hasattr(uncertainty_map, "tolist"):
            uncertainty_map = uncertainty_map.tolist()
        tvlc_context, connector_type = self._resolve_tvlc_context(patch_tokens)

        if not passes:
            failure = str(failure_reasons[0]) if failure_reasons else "insufficient evidence"
            return SetuDescription(
                text=f"Unclear region ({failure})",
                confidence=confidence,
                is_uncertain=True,
                anchor_basis=anchor_name,
                depth_stratum=depth_stratum,
                hallucination_risk=float(gate_payload.get("estimated_hallucination_risk", max(0.0, min(1.0, 1.0 - confidence)))),
                uncertainty_map=uncertainty_map,
                tvlc_context=tvlc_context,
                connector_type=connector_type,
            )

        template = ANCHOR_DESCRIPTIONS.get(anchor_name, "Object in {depth}")
        return SetuDescription(
            text=template.format(depth=depth_stratum),
            confidence=confidence,
            is_uncertain=False,
            anchor_basis=anchor_name,
            depth_stratum=depth_stratum,
            hallucination_risk=float(gate_payload.get("estimated_hallucination_risk", max(0.0, min(1.0, 1.0 - confidence)))),
            uncertainty_map=None,
            tvlc_context=tvlc_context,
            connector_type=connector_type,
        )

    def _resolve_tvlc_context(self, patch_tokens: Optional[np.ndarray]) -> tuple[Optional[str], Optional[str]]:
        if patch_tokens is None:
            return None, None
        patches = np.asarray(patch_tokens, dtype=np.float32)
        if patches.shape != (196, 384):
            return None, None
        patch_id = id(patches)
        if self._tvlc_cache_patch_id == patch_id:
            return self._tvlc_cache_value
        connector = self._get_tvlc_connector()
        if connector is None:
            return None, None
        try:
            context = connector.to_gemma_context(patches)
            connector_type = "tvlc_trained" if connector.is_trained else "tvlc_random_init"
            self._tvlc_cache_patch_id = patch_id
            self._tvlc_cache_value = (context, connector_type)
            return self._tvlc_cache_value
        except Exception as exc:
            log.debug("setu2_tvlc_context_failed", error=str(exc))
            return None, None

    def _get_tvlc_connector(self) -> Any:
        if self._tvlc_checked:
            return self._tvlc_connector

        self._tvlc_checked = True
        try:
            from cloud.perception.tvlc_connector import TVLCConnector
        except Exception as exc:
            log.debug("setu2_tvlc_import_failed", error=str(exc))
            self._tvlc_connector = None
            return None

        if not TVLCConnector.is_available():
            self._tvlc_connector = None
            return None

        try:
            self._tvlc_connector = TVLCConnector(TVLCConnector.default_weights_path())
        except Exception as exc:
            log.debug("setu2_tvlc_load_failed", error=str(exc))
            self._tvlc_connector = None
        return self._tvlc_connector
