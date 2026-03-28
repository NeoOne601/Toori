"""
Cross-Modal World Model Alignment (CWMA).

Injects spatial co-occurrence priors into JEPA energy computation.
Zero LLM calls at runtime. Priors stored as a static 50KB numpy lookup table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from cloud.jepa_service.anchor_graph import AnchorMatch
from cloud.jepa_service.depth_separator import DepthStrataMap
from cloud.runtime.observability import get_logger, with_fallback

log = get_logger("cwma")

TEMPLATE_NAMES: list[str] = [
    "person_torso",
    "chair_seated",
    "cylindrical_object",
    "screen_display",
    "desk_surface",
    "hand_region",
    "spherical_object",
    "background_plane",
    "unknown",
]
N_CLASSES: int = len(TEMPLATE_NAMES)

RELATIONS: list[str] = ["FOREGROUND_OF", "BACKGROUND_OF", "BESIDE", "NONE"]
N_RELATIONS: int = len(RELATIONS)


def _build_bootstrap_scpt() -> np.ndarray:
    scpt = np.zeros((N_CLASSES, N_CLASSES, N_RELATIONS), dtype=np.float32)
    scpt[:, :, 3] = 0.4

    def _set(name_a: str, name_b: str, relation: str, prob: float) -> None:
        i = TEMPLATE_NAMES.index(name_a)
        j = TEMPLATE_NAMES.index(name_b)
        r = RELATIONS.index(relation)
        scpt[i, j, r] = prob
        if prob > 0.5:
            scpt[i, j, 3] = max(0.0, scpt[i, j, 3] - (prob * 0.5))

    _set("person_torso", "chair_seated", "FOREGROUND_OF", 0.92)
    _set("person_torso", "desk_surface", "FOREGROUND_OF", 0.95)
    _set("person_torso", "screen_display", "FOREGROUND_OF", 0.88)
    _set("person_torso", "cylindrical_object", "FOREGROUND_OF", 0.90)

    _set("cylindrical_object", "person_torso", "BACKGROUND_OF", 0.89)
    _set("cylindrical_object", "desk_surface", "BESIDE", 0.72)

    _set("screen_display", "person_torso", "BACKGROUND_OF", 0.93)
    _set("screen_display", "desk_surface", "BESIDE", 0.68)

    _set("desk_surface", "person_torso", "BACKGROUND_OF", 0.96)
    _set("desk_surface", "cylindrical_object", "BESIDE", 0.71)

    _set("hand_region", "person_torso", "BESIDE", 0.91)
    _set("hand_region", "desk_surface", "FOREGROUND_OF", 0.82)

    row_sums = scpt.sum(axis=2, keepdims=True) + 1e-8
    return (scpt / row_sums).astype(np.float32)


BOOTSTRAP_SCPT: np.ndarray = _build_bootstrap_scpt()


class CrossModalWorldModelAligner:
    def __init__(
        self,
        scpt_path: Optional[str] = None,
        lambda_cwma: float = 0.15,
    ) -> None:
        self._lambda = lambda_cwma
        if scpt_path and Path(scpt_path).exists():
            self._scpt = np.load(scpt_path)["scpt"].astype(np.float32)
            log.info("scpt_loaded", path=scpt_path, shape=self._scpt.shape)
        else:
            self._scpt = BOOTSTRAP_SCPT
            log.debug("scpt_bootstrap_used")

    @with_fallback(fallback_value=(None, 0.0), log_component="cwma")
    def apply_alignment(
        self,
        energy_map: np.ndarray,
        anchor_matches: list[AnchorMatch],
        depth_strata: DepthStrataMap,
    ) -> tuple[np.ndarray | None, float]:
        energy = np.asarray(energy_map, dtype=np.float32).copy()
        if not anchor_matches or len(anchor_matches) < 2:
            return energy, 0.0

        total_penalty = 0.0
        n_patches_penalized = 0

        for i, match_a in enumerate(anchor_matches):
            idx_a = self._class_index(match_a.template_name)
            for j, match_b in enumerate(anchor_matches):
                if i == j:
                    continue
                idx_b = self._class_index(match_b.template_name)
                actual_relation = self._infer_actual_relation(match_a, match_b, depth_strata)
                actual_rel_idx = RELATIONS.index(actual_relation)
                prior_prob = float(self._scpt[idx_a, idx_b, actual_rel_idx])

                if prior_prob < 0.30:
                    penalty = self._lambda * (1.0 - prior_prob)
                    for patch_idx in match_a.patch_indices:
                        row, col = divmod(patch_idx, 14)
                        if 0 <= row < 14 and 0 <= col < 14:
                            energy[row, col] += penalty
                            total_penalty += penalty
                            n_patches_penalized += 1

        alignment_loss = total_penalty / max(n_patches_penalized, 1)
        log.debug(
            "cwma_aligned",
            n_matches=len(anchor_matches),
            alignment_loss=round(alignment_loss, 4),
            n_patches_penalized=n_patches_penalized,
        )
        return energy.astype(np.float32), float(alignment_loss)

    def _class_index(self, template_name: str) -> int:
        try:
            return TEMPLATE_NAMES.index(template_name)
        except ValueError:
            return TEMPLATE_NAMES.index("unknown")

    def _infer_actual_relation(
        self,
        match_a: AnchorMatch,
        match_b: AnchorMatch,
        depth_strata: DepthStrataMap,
    ) -> str:
        _ = depth_strata
        if match_a.depth_stratum == "foreground" and match_b.depth_stratum == "background":
            return "FOREGROUND_OF"
        if match_a.depth_stratum == "background" and match_b.depth_stratum == "foreground":
            return "BACKGROUND_OF"
        bbox_a = match_a.bbox_normalized
        bbox_b = match_b.bbox_normalized
        center_ax = bbox_a["x"] + (bbox_a["width"] / 2)
        center_bx = bbox_b["x"] + (bbox_b["width"] / 2)
        if abs(center_ax - center_bx) < 0.3:
            return "BESIDE"
        return "NONE"
