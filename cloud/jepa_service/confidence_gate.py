"""
Epistemic Confidence Gate (ECGD).

Prevents description of semantically unclear regions.
"I don't know" is correct. Confident wrong answers are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cloud.jepa_service.anchor_graph import AnchorMatch
from cloud.jepa_service.depth_separator import DepthStrataMap
from cloud.runtime.observability import get_logger

log = get_logger("ecgd")

TAU_DEPTH: float = 0.50
TAU_ANCHOR: float = 0.55
TAU_VARIANCE: float = 0.25


@dataclass
class GateResult:
    passes: bool
    consistency_score: float
    failure_reasons: list[str]
    safe_embedding: Optional[np.ndarray]
    uncertainty_map: np.ndarray
    anchor_name: str
    depth_stratum: str
    estimated_hallucination_risk: float
    # Sprint 6: World Model Foundation
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passes": self.passes,
            "consistency_score": round(float(self.consistency_score), 4),
            "failure_reasons": self.failure_reasons,
            "anchor_name": self.anchor_name,
            "depth_stratum": self.depth_stratum,
            "estimated_hallucination_risk": round(float(self.estimated_hallucination_risk), 4),
            "uncertainty_map": self.uncertainty_map.tolist(),
            "epistemic_uncertainty": round(float(self.epistemic_uncertainty), 4),
            "aleatoric_uncertainty": round(float(self.aleatoric_uncertainty), 4),
        }


class EpistemicConfidenceGate:
    def __init__(
        self,
        tau_depth: float = TAU_DEPTH,
        tau_anchor: float = TAU_ANCHOR,
        tau_variance: float = TAU_VARIANCE,
    ) -> None:
        self._tau_depth = tau_depth
        self._tau_anchor = tau_anchor
        self._tau_variance = tau_variance

    def evaluate(
        self,
        anchor_match: AnchorMatch,
        depth_strata: DepthStrataMap,
        energy_history: list[float],
        *,
        prediction_error: float | None = None,
        surprise_score: float | None = None,
    ) -> GateResult:
        failures: list[str] = []

        depth_purity = self._compute_depth_purity(anchor_match, depth_strata)
        if depth_purity < self._tau_depth:
            failures.append(f"depth_purity={depth_purity:.3f} < τ={self._tau_depth}")

        if anchor_match.confidence < self._tau_anchor:
            failures.append(f"anchor_confidence={anchor_match.confidence:.3f} < τ={self._tau_anchor}")

        variance = float(np.var(energy_history[-8:])) if len(energy_history) >= 3 else 1.0
        if variance > self._tau_variance:
            failures.append(f"energy_variance={variance:.3f} > τ={self._tau_variance}")

        passes = len(failures) == 0
        consistency = float(
            0.35 * min(depth_purity / max(self._tau_depth, 1e-6), 1.0)
            + 0.40 * min(anchor_match.confidence / max(self._tau_anchor, 1e-6), 1.0)
            + 0.25 * max(0.0, 1.0 - (variance / max(self._tau_variance, 1e-6)))
        )
        consistency = round(min(consistency, 1.0), 4)
        uncertainty_map = self._build_uncertainty_map(anchor_match, depth_strata)
        hallucination_risk = round(1.0 - consistency, 4)

        # Sprint 6: Information-theoretic uncertainty from world model
        epistemic = 0.0
        aleatoric = 0.0
        if prediction_error is not None:
            # Epistemic: model's predictive uncertainty (how wrong was prediction)
            epistemic = round(min(float(prediction_error), 1.0), 4)
            # High prediction error lowers gate confidence
            if prediction_error > 0.5:
                failures.append(f"prediction_error={prediction_error:.3f} > 0.5")
                passes = len(failures) == 0
        if surprise_score is not None:
            # Aleatoric: inherent scene variability (normalized surprise)
            aleatoric = round(min(float(surprise_score), 1.0), 4)

        result = GateResult(
            passes=passes,
            consistency_score=consistency,
            failure_reasons=failures,
            safe_embedding=anchor_match.embedding_centroid.copy() if passes else None,
            uncertainty_map=uncertainty_map,
            anchor_name=anchor_match.template_name,
            depth_stratum=anchor_match.depth_stratum,
            estimated_hallucination_risk=hallucination_risk,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )
        log.debug(
            "ecgd_gate",
            anchor=anchor_match.template_name,
            passes=passes,
            consistency=consistency,
            failures=failures,
        )
        return result

    def _compute_depth_purity(
        self,
        match: AnchorMatch,
        strata: DepthStrataMap,
    ) -> float:
        if not match.patch_indices:
            return 0.0
        dominant = match.depth_stratum
        count_in_dominant = 0
        for idx in match.patch_indices:
            row, col = divmod(idx, 14)
            if dominant == "foreground" and strata.foreground_mask[row, col]:
                count_in_dominant += 1
            elif dominant == "background" and strata.background_mask[row, col]:
                count_in_dominant += 1
            elif dominant == "midground" and strata.midground_mask[row, col]:
                count_in_dominant += 1
        return count_in_dominant / len(match.patch_indices)

    def _build_uncertainty_map(
        self,
        match: AnchorMatch,
        strata: DepthStrataMap,
    ) -> np.ndarray:
        uncertainty_map = np.zeros((14, 14), dtype=np.float32)
        for idx in match.patch_indices:
            row, col = divmod(idx, 14)
            dominant = match.depth_stratum
            is_wrong_stratum = (
                (dominant == "foreground" and not strata.foreground_mask[row, col])
                or (dominant == "background" and not strata.background_mask[row, col])
                or (dominant == "midground" and not strata.midground_mask[row, col])
            )
            uncertainty_map[row, col] = 1.0 if is_wrong_stratum else 0.1
        return uncertainty_map
