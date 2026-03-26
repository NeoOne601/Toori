"""Selective Talker: fires structured events only when energy exceeds adaptive threshold.

Replaces the previous approach of calling Ollama/MLX on every frame.
Language reasoning providers are demoted to query-only (explicit user prompts).
"""

from __future__ import annotations

from cloud.runtime.models import (
    EntityTrack,
    PersistenceSignal,
    TalkerEvent,
)


class SelectiveTalker:
    """Evaluates JEPA energy + entity state to produce structured talker events."""

    def evaluate(
        self,
        mean_energy: float,
        threshold: float,
        should_talk: bool,
        entity_tracks: list[EntityTrack],
        previous_tracks: list[EntityTrack],
        persistence_signal: PersistenceSignal,
    ) -> TalkerEvent | None:
        """Decide whether to emit a talker event and which type.

        Returns None when energy is below threshold (SCENE_STABLE is still
        emitted as a heartbeat every time should_talk is False, but only
        if there has been at least one tick).
        """
        if not should_talk:
            return TalkerEvent(
                event_type="SCENE_STABLE",
                confidence=1.0 - min(mean_energy / max(threshold, 1e-6), 1.0),
                energy_summary=mean_energy,
                description="Scene is stable; energy below adaptive threshold.",
            )

        # Priority 1: Occlusion recovery
        if persistence_signal.recovered_track_ids:
            recovered_labels = self._track_labels(
                persistence_signal.recovered_track_ids, entity_tracks
            )
            return TalkerEvent(
                event_type="OCCLUSION_END",
                confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
                entity_ids=persistence_signal.recovered_track_ids,
                energy_summary=mean_energy,
                description=f"Re-identified after occlusion: {', '.join(recovered_labels) or 'entity'}.",
            )

        # Priority 2: Prediction violation
        if persistence_signal.violated_track_ids:
            violated_labels = self._track_labels(
                persistence_signal.violated_track_ids, entity_tracks
            )
            return TalkerEvent(
                event_type="PREDICTION_VIOLATION",
                confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
                entity_ids=persistence_signal.violated_track_ids,
                energy_summary=mean_energy,
                description=f"Prediction violated: expected {', '.join(violated_labels) or 'entity'} but not found.",
            )

        # Priority 3: New entity appeared
        prev_visible_ids = {t.id for t in previous_tracks if t.status in ("visible", "re-identified")}
        curr_visible_ids = {t.id for t in entity_tracks if t.status in ("visible", "re-identified")}
        new_appeared = curr_visible_ids - prev_visible_ids
        if new_appeared:
            appeared_labels = self._track_labels(list(new_appeared), entity_tracks)
            return TalkerEvent(
                event_type="ENTITY_APPEARED",
                confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
                entity_ids=list(new_appeared),
                energy_summary=mean_energy,
                description=f"New entity appeared: {', '.join(appeared_labels) or 'entity'}.",
            )

        # Priority 4: Occlusion start
        prev_occluded_ids = {t.id for t in previous_tracks if t.status == "occluded"}
        curr_occluded_ids = {t.id for t in entity_tracks if t.status == "occluded"}
        new_occluded = curr_occluded_ids - prev_occluded_ids
        if new_occluded:
            occluded_labels = self._track_labels(list(new_occluded), entity_tracks)
            return TalkerEvent(
                event_type="OCCLUSION_START",
                confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
                entity_ids=list(new_occluded),
                energy_summary=mean_energy,
                description=f"Entity occluded: {', '.join(occluded_labels) or 'entity'}.",
            )

        # Priority 5: Entity disappeared
        if persistence_signal.disappeared_track_ids:
            disappeared_labels = self._track_labels(
                persistence_signal.disappeared_track_ids, entity_tracks
            )
            return TalkerEvent(
                event_type="ENTITY_DISAPPEARED",
                confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
                entity_ids=persistence_signal.disappeared_track_ids,
                energy_summary=mean_energy,
                description=f"Entity disappeared: {', '.join(disappeared_labels) or 'entity'}.",
            )

        # High energy but no structural change — prediction violation (general)
        return TalkerEvent(
            event_type="PREDICTION_VIOLATION",
            confidence=min(mean_energy / max(threshold, 1e-6), 1.0),
            energy_summary=mean_energy,
            description="Unexpected change detected; prediction residual is high.",
        )

    @staticmethod
    def _track_labels(track_ids: list[str], tracks: list[EntityTrack]) -> list[str]:
        lookup = {t.id: t.label for t in tracks}
        return [lookup.get(tid, tid) for tid in track_ids]
