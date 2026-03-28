"""
Setu-2: JEPA-to-Language EBM Bridge.

Upgraded bridge that operates on DEPTH-SEPARATED, ANCHOR-CONFIRMED,
WORLD-MODEL-ALIGNED, GATE-PASSED embeddings only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "is_uncertain": self.is_uncertain,
            "anchor_basis": self.anchor_basis,
            "depth_stratum": self.depth_stratum,
            "hallucination_risk": round(float(self.hallucination_risk), 4),
            "uncertainty_map": self.uncertainty_map,
        }


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
    ) -> SetuDescription:
        if not gate_result.passes:
            return SetuDescription(
                text=f"Unclear region ({', '.join(gate_result.failure_reasons[:1])})",
                confidence=gate_result.consistency_score,
                is_uncertain=True,
                anchor_basis=gate_result.anchor_name,
                depth_stratum=gate_result.depth_stratum,
                hallucination_risk=gate_result.estimated_hallucination_risk,
                uncertainty_map=gate_result.uncertainty_map.tolist(),
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
        )

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

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for index, layer in enumerate(self._layers):
            x = x @ layer["W"] + layer["b"]
            if index < len(self._layers) - 1:
                x = _relu(x)
        return x.astype(np.float32)
