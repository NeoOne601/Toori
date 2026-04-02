"""Epistemic Atlas: in-memory entity relationship graph built from JEPA energy + EntityTracks.

Nodes represent entities, edges represent co-occurrence and interaction energy.
Persisted as JSON in SceneState.metadata — no new SQLite tables needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from cloud.runtime.models import (
    AtlasEdge,
    AtlasNode,
    BoundingBox,
    EntityTrack,
    SceneState,
    TrackStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _box_center(box: BoundingBox) -> tuple[float, float]:
    return (box.x + box.width / 2.0, box.y + box.height / 2.0)


def _spatial_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


class EpistemicAtlas:
    """In-memory entity relationship graph updated each tick."""

    def __init__(self, stale_seconds: float = 30.0) -> None:
        self._nodes: dict[str, AtlasNode] = {}
        self._edges: dict[tuple[str, str], AtlasEdge] = {}
        self._stale_seconds = stale_seconds

    def update(
        self,
        entity_tracks: list[EntityTrack],
        scene_state: SceneState,
        energy_map: Optional[np.ndarray] = None,
    ) -> None:
        """Update atlas nodes and edges from current tick state."""
        now = scene_state.created_at

        # Build label → centroid lookup from proposal boxes
        box_centroids: dict[str, tuple[float, float]] = {}
        for box in scene_state.proposal_boxes:
            if box.label:
                box_centroids[box.label.lower().strip()] = _box_center(box)

        # Update / create nodes for every entity track
        current_track_ids: list[str] = []
        for track in entity_tracks:
            current_track_ids.append(track.id)
            centroid = box_centroids.get(
                track.label.lower().strip(), (0.5, 0.5)
            )

            # Get energy at centroid from energy map
            last_energy = 0.0
            if energy_map is not None:
                grid_h, grid_w = energy_map.shape
                row = min(int(centroid[1] * grid_h), grid_h - 1)
                col = min(int(centroid[0] * grid_w), grid_w - 1)
                last_energy = float(energy_map[row, col])

            if track.id in self._nodes:
                existing = self._nodes[track.id]
                self._nodes[track.id] = existing.model_copy(update={
                    "label": track.label,
                    "centroid": centroid,
                    "track_length": len(track.observations),
                    "confidence": track.persistence_confidence,
                    "last_energy": last_energy,
                    "status": track.status,
                    "last_seen_at": now,
                })
            else:
                self._nodes[track.id] = AtlasNode(
                    entity_id=track.id,
                    label=track.label,
                    centroid=centroid,
                    track_length=len(track.observations),
                    confidence=track.persistence_confidence,
                    last_energy=last_energy,
                    status=track.status,
                    first_seen_at=track.first_seen_at,
                    last_seen_at=now,
                )

        # Update edges: co-occurring entities in this scene
        visible_ids = [
            tid for tid in current_track_ids
            if tid in self._nodes and self._nodes[tid].status in ("visible", "re-identified")
        ]

        for i, a_id in enumerate(visible_ids):
            for b_id in visible_ids[i + 1:]:
                edge_key = (min(a_id, b_id), max(a_id, b_id))
                a_centroid = self._nodes[a_id].centroid
                b_centroid = self._nodes[b_id].centroid
                distance = _spatial_distance(a_centroid, b_centroid)
                proximity = max(0.0, 1.0 - distance)

                # Interaction energy: mean of both nodes' energy
                interaction = (
                    self._nodes[a_id].last_energy + self._nodes[b_id].last_energy
                ) / 2.0

                if edge_key in self._edges:
                    existing_edge = self._edges[edge_key]
                    self._edges[edge_key] = existing_edge.model_copy(update={
                        "interaction_energy": interaction,
                        "spatial_proximity": proximity,
                        "co_occurrence_count": existing_edge.co_occurrence_count + 1,
                        "last_seen_together": now,
                        "status": "active",
                    })
                else:
                    self._edges[edge_key] = AtlasEdge(
                        source_id=edge_key[0],
                        target_id=edge_key[1],
                        interaction_energy=interaction,
                        spatial_proximity=proximity,
                        co_occurrence_count=1,
                        last_seen_together=now,
                        status="active",
                    )

        # Age out stale edges
        for key, edge in list(self._edges.items()):
            age = (now - edge.last_seen_together).total_seconds()
            if age > self._stale_seconds * 2:
                self._edges[key] = edge.model_copy(update={"status": "broken"})
            elif age > self._stale_seconds:
                self._edges[key] = edge.model_copy(update={"status": "stale"})

        removable_nodes = [
            track_id
            for track_id, node in self._nodes.items()
            if (
                node.status in {"disappeared", "violated prediction"}
                or (now - node.last_seen_at).total_seconds() > self._stale_seconds
            )
        ]
        for track_id in removable_nodes:
            self._nodes.pop(track_id, None)

        for key, edge in list(self._edges.items()):
            if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
                self._edges.pop(key, None)

    def get_nodes(self) -> list[AtlasNode]:
        return list(self._nodes.values())

    def get_edges(self) -> list[AtlasEdge]:
        return list(self._edges.values())

    def to_dict(self) -> dict:
        """Serializable dict for API responses and metadata persistence."""
        return {
            "nodes": [node.model_dump(mode="json") for node in self._nodes.values()],
            "edges": [edge.model_dump(mode="json") for edge in self._edges.values()],
        }

    def reset(self) -> None:
        """Clear all atlas state."""
        self._nodes.clear()
        self._edges.clear()
