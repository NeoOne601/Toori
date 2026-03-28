"""
Semantic Anchor Graph (SAG).

Matches patch token spatial configurations against typed anchor templates
using Weisfeiler-Lehman-style graph neighborhood aggregation.

No neural network. No generative model. Pure structural pattern matching.
All arrays float32. Zero torch imports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cloud.jepa_service.depth_separator import DepthStrataMap
from cloud.runtime.observability import get_logger, with_fallback

log = get_logger("sag")

GRID_H: int = 14
GRID_W: int = 14
PATCH_DIM: int = 384
PATCH_SIMILARITY_THRESHOLD: float = 0.68


@dataclass
class SemanticAnchorTemplate:
    name: str
    nodes: list[dict]
    edges: list[tuple[int, int, str]]
    min_nodes_required: int
    depth_preference: str = "any"
    min_confidence: float = 0.65

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": self.nodes,
            "edges": [(a, b, relation) for a, b, relation in self.edges],
            "min_nodes_required": self.min_nodes_required,
            "depth_preference": self.depth_preference,
            "min_confidence": self.min_confidence,
        }

    @staticmethod
    def from_dict(data: dict) -> "SemanticAnchorTemplate":
        return SemanticAnchorTemplate(
            name=data["name"],
            nodes=data["nodes"],
            edges=[tuple(edge) for edge in data["edges"]],
            min_nodes_required=data["min_nodes_required"],
            depth_preference=data.get("depth_preference", "any"),
            min_confidence=data.get("min_confidence", 0.65),
        )


@dataclass
class AnchorMatch:
    template_name: str
    confidence: float
    patch_indices: list[int]
    depth_stratum: str
    centroid_patch: int
    embedding_centroid: np.ndarray
    is_novel: bool
    bbox_normalized: dict

    def to_dict(self) -> dict:
        return {
            "template_name": self.template_name,
            "confidence": round(float(self.confidence), 4),
            "patch_indices": self.patch_indices,
            "depth_stratum": self.depth_stratum,
            "centroid_patch": self.centroid_patch,
            "embedding_centroid": self.embedding_centroid.tolist(),
            "is_novel": self.is_novel,
            "bbox_normalized": self.bbox_normalized,
        }


def _patch_to_rowcol(patch_idx: int) -> tuple[int, int]:
    return divmod(patch_idx, GRID_W)


def _rowcol_to_patch(row: int, col: int) -> int:
    return row * GRID_W + col


def _spatial_relation(patch_a: int, patch_b: int) -> Optional[str]:
    row_a, col_a = _patch_to_rowcol(patch_a)
    row_b, col_b = _patch_to_rowcol(patch_b)
    dr = row_b - row_a
    dc = col_b - col_a
    if abs(dr) > 1 or abs(dc) > 1:
        return None
    if dr == -1 and dc == 0:
        return "ABOVE"
    if dr == 1 and dc == 0:
        return "BELOW"
    if dr == 0 and dc == -1:
        return "LEFT_OF"
    if dr == 0 and dc == 1:
        return "RIGHT_OF"
    return "ADJACENT"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float32)
    b = b.ravel().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _get_dominant_stratum(
    patch_indices: list[int],
    depth_strata: DepthStrataMap,
) -> str:
    counts = {"foreground": 0, "midground": 0, "background": 0}
    for idx in patch_indices:
        row, col = _patch_to_rowcol(idx)
        if depth_strata.foreground_mask[row, col]:
            counts["foreground"] += 1
        elif depth_strata.background_mask[row, col]:
            counts["background"] += 1
        else:
            counts["midground"] += 1
    return max(counts, key=counts.get)


def _bbox_from_patches(patch_indices: list[int]) -> dict:
    rows = [_patch_to_rowcol(index)[0] for index in patch_indices]
    cols = [_patch_to_rowcol(index)[1] for index in patch_indices]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    return {
        "x": round(c_min / GRID_W, 4),
        "y": round(r_min / GRID_H, 4),
        "width": round((c_max - c_min + 1) / GRID_W, 4),
        "height": round((r_max - r_min + 1) / GRID_H, 4),
    }


def _build_bootstrap_templates() -> list[SemanticAnchorTemplate]:
    return [
        SemanticAnchorTemplate(
            name="person_torso",
            nodes=[
                {"role": "upper_torso", "depth_preference": "foreground"},
                {"role": "lower_torso", "depth_preference": "foreground"},
                {"role": "shoulder_l", "depth_preference": "foreground"},
                {"role": "shoulder_r", "depth_preference": "foreground"},
            ],
            edges=[(0, 1, "ABOVE"), (0, 2, "ADJACENT"), (0, 3, "ADJACENT")],
            min_nodes_required=2,
            depth_preference="foreground",
            min_confidence=0.72,
        ),
        SemanticAnchorTemplate(
            name="chair_seated",
            nodes=[
                {"role": "seat_surface", "depth_preference": "midground"},
                {"role": "chair_back", "depth_preference": "midground"},
                {"role": "armrest", "depth_preference": "midground"},
            ],
            edges=[(0, 1, "ABOVE"), (0, 2, "ADJACENT")],
            min_nodes_required=2,
            depth_preference="midground",
            min_confidence=0.65,
        ),
        SemanticAnchorTemplate(
            name="cylindrical_object",
            nodes=[
                {"role": "top_end", "depth_preference": "any"},
                {"role": "mid_body_1", "depth_preference": "any"},
                {"role": "mid_body_2", "depth_preference": "any"},
                {"role": "bottom_end", "depth_preference": "any"},
            ],
            edges=[(0, 1, "BELOW"), (1, 2, "BELOW"), (2, 3, "BELOW")],
            min_nodes_required=3,
            depth_preference="background",
            min_confidence=0.62,
        ),
        SemanticAnchorTemplate(
            name="screen_display",
            nodes=[
                {"role": "top_edge", "depth_preference": "background"},
                {"role": "left_edge", "depth_preference": "background"},
                {"role": "right_edge", "depth_preference": "background"},
                {"role": "interior", "depth_preference": "background"},
            ],
            edges=[
                (0, 1, "BELOW"),
                (0, 2, "BELOW"),
                (1, 3, "RIGHT_OF"),
                (2, 3, "LEFT_OF"),
            ],
            min_nodes_required=3,
            depth_preference="background",
            min_confidence=0.70,
        ),
        SemanticAnchorTemplate(
            name="desk_surface",
            nodes=[
                {"role": "surface_l", "depth_preference": "background"},
                {"role": "surface_c", "depth_preference": "background"},
                {"role": "surface_r", "depth_preference": "background"},
            ],
            edges=[(0, 1, "RIGHT_OF"), (1, 2, "RIGHT_OF")],
            min_nodes_required=2,
            depth_preference="background",
            min_confidence=0.60,
        ),
        SemanticAnchorTemplate(
            name="hand_region",
            nodes=[
                {"role": "palm", "depth_preference": "foreground"},
                {"role": "finger", "depth_preference": "foreground"},
            ],
            edges=[(0, 1, "ADJACENT")],
            min_nodes_required=1,
            depth_preference="foreground",
            min_confidence=0.58,
        ),
        SemanticAnchorTemplate(
            name="spherical_object",
            nodes=[
                {"role": "top", "depth_preference": "any"},
                {"role": "center", "depth_preference": "any"},
                {"role": "bottom", "depth_preference": "any"},
                {"role": "left", "depth_preference": "any"},
                {"role": "right", "depth_preference": "any"},
            ],
            edges=[
                (0, 1, "BELOW"),
                (1, 2, "BELOW"),
                (3, 1, "RIGHT_OF"),
                (1, 4, "RIGHT_OF"),
            ],
            min_nodes_required=3,
            depth_preference="any",
            min_confidence=0.60,
        ),
        SemanticAnchorTemplate(
            name="background_plane",
            nodes=[
                {"role": "bg_patch_a", "depth_preference": "background"},
                {"role": "bg_patch_b", "depth_preference": "background"},
                {"role": "bg_patch_c", "depth_preference": "background"},
                {"role": "bg_patch_d", "depth_preference": "background"},
            ],
            edges=[
                (0, 1, "ADJACENT"),
                (1, 2, "ADJACENT"),
                (2, 3, "ADJACENT"),
            ],
            min_nodes_required=3,
            depth_preference="background",
            min_confidence=0.55,
        ),
    ]


BOOTSTRAP_TEMPLATES: list[SemanticAnchorTemplate] = _build_bootstrap_templates()


class SemanticAnchorGraph:
    def __init__(
        self,
        templates: list[SemanticAnchorTemplate] | None = None,
        similarity_threshold: float = PATCH_SIMILARITY_THRESHOLD,
        max_matches_per_frame: int = 8,
    ) -> None:
        self._templates = templates or list(BOOTSTRAP_TEMPLATES)
        self._sim_threshold = similarity_threshold
        self._max_matches = max_matches_per_frame
        self._learned_templates: list[SemanticAnchorTemplate] = []

    @with_fallback(fallback_value=[], log_component="sag")
    def match(
        self,
        patch_tokens: np.ndarray,
        depth_strata: DepthStrataMap,
        mask_regions: list[list[int]],
    ) -> list[AnchorMatch]:
        patch_tokens = np.asarray(patch_tokens, dtype=np.float32)
        if not mask_regions:
            mask_regions = self._fallback_grid_regions(depth_strata)

        all_templates = self._templates + self._learned_templates
        matches: list[AnchorMatch] = []
        used_patches: set[int] = set()

        for region_patches in mask_regions:
            if not region_patches:
                continue
            overlap = len(set(region_patches) & used_patches)
            if overlap > len(region_patches) * 0.5:
                continue

            best_match = self._best_template_match(
                patch_tokens,
                region_patches,
                depth_strata,
                all_templates,
            )
            matches.append(best_match)
            used_patches.update(region_patches)

            if len(matches) >= self._max_matches:
                break

        log.debug("sag_match", n_regions=len(mask_regions), n_matches=len(matches))
        return matches

    def _best_template_match(
        self,
        patch_tokens: np.ndarray,
        region_patches: list[int],
        depth_strata: DepthStrataMap,
        templates: list[SemanticAnchorTemplate],
    ) -> AnchorMatch:
        dom_stratum = _get_dominant_stratum(region_patches, depth_strata)
        embedding_centroid = patch_tokens[region_patches].mean(axis=0)
        centroid_patch = region_patches[len(region_patches) // 2]
        bbox = _bbox_from_patches(region_patches)

        best_name = "unknown"
        best_conf = 0.0

        for template in templates:
            if template.depth_preference != "any" and template.depth_preference != dom_stratum:
                depth_penalty = 0.25
            else:
                depth_penalty = 0.0

            conf = self._topology_score(region_patches, template) - depth_penalty
            if conf > best_conf:
                best_conf = conf
                best_name = template.name

        is_novel = best_conf < 0.40
        if is_novel:
            best_name = "unknown"

        return AnchorMatch(
            template_name=best_name,
            confidence=round(float(max(best_conf, 0.0)), 4),
            patch_indices=list(region_patches),
            depth_stratum=dom_stratum,
            centroid_patch=centroid_patch,
            embedding_centroid=embedding_centroid.astype(np.float32),
            is_novel=is_novel,
            bbox_normalized=bbox,
        )

    def _topology_score(
        self,
        region_patches: list[int],
        template: SemanticAnchorTemplate,
    ) -> float:
        if len(region_patches) < template.min_nodes_required:
            return 0.0
        if not template.edges:
            return min(len(region_patches) / max(len(template.nodes), 1), 1.0)

        region_set = set(region_patches)
        satisfied_edges = 0
        total_edges = len(template.edges)

        for _, _, relation in template.edges:
            for patch in region_patches:
                row, col = _patch_to_rowcol(patch)
                if relation == "ABOVE" and row > 0:
                    if _rowcol_to_patch(row - 1, col) in region_set:
                        satisfied_edges += 1
                        break
                elif relation == "BELOW" and row < GRID_H - 1:
                    if _rowcol_to_patch(row + 1, col) in region_set:
                        satisfied_edges += 1
                        break
                elif relation == "LEFT_OF" and col > 0:
                    if _rowcol_to_patch(row, col - 1) in region_set:
                        satisfied_edges += 1
                        break
                elif relation == "RIGHT_OF" and col < GRID_W - 1:
                    if _rowcol_to_patch(row, col + 1) in region_set:
                        satisfied_edges += 1
                        break
                elif relation == "ADJACENT":
                    found_adjacent = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        r2, c2 = row + dr, col + dc
                        if 0 <= r2 < GRID_H and 0 <= c2 < GRID_W:
                            if _rowcol_to_patch(r2, c2) in region_set:
                                satisfied_edges += 1
                                found_adjacent = True
                                break
                    if found_adjacent:
                        break

        edge_ratio = satisfied_edges / total_edges
        return float(edge_ratio * template.min_confidence + edge_ratio * (1 - template.min_confidence) * 0.5)

    def _fallback_grid_regions(
        self,
        depth_strata: DepthStrataMap,
    ) -> list[list[int]]:
        _ = depth_strata
        regions: list[list[int]] = []
        for block_row in range(3):
            for block_col in range(3):
                patches = []
                for row in range(block_row * 4, min((block_row + 1) * 5, 14)):
                    for col in range(block_col * 4, min((block_col + 1) * 5, 14)):
                        patches.append(_rowcol_to_patch(row, col))
                if patches:
                    regions.append(patches)
        return regions

    def learn_template_from_confirmation(
        self,
        region_patches: list[int],
        confirmed_label: str,
        patch_tokens: np.ndarray,
    ) -> SemanticAnchorTemplate:
        _ = patch_tokens
        nodes = [{"role": f"patch_{index}", "depth_preference": "any"} for index in range(min(len(region_patches), 6))]
        edges: list[tuple[int, int, str]] = []
        sampled = region_patches[:6]
        for i, patch_a in enumerate(sampled):
            for j, patch_b in enumerate(sampled):
                if i >= j:
                    continue
                relation = _spatial_relation(patch_a, patch_b)
                if relation:
                    edges.append((i, j, relation))

        template = SemanticAnchorTemplate(
            name=confirmed_label,
            nodes=nodes,
            edges=edges,
            min_nodes_required=max(1, len(nodes) // 2),
            depth_preference="any",
            min_confidence=0.60,
        )
        self._learned_templates.append(template)
        log.info("template_learned", name=confirmed_label, n_nodes=len(nodes))
        return template

    def save_learned_templates(self, path: str) -> None:
        data = [template.to_dict() for template in self._learned_templates]
        Path(path).write_text(json.dumps(data, indent=2))

    def load_learned_templates(self, path: str) -> None:
        template_path = Path(path)
        if not template_path.exists():
            return
        data = json.loads(template_path.read_text())
        self._learned_templates = [SemanticAnchorTemplate.from_dict(item) for item in data]
        log.info("templates_loaded", n=len(self._learned_templates))

    def reset(self) -> None:
        self._learned_templates.clear()
