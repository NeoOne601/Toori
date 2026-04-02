"""Tests for the Epistemic Atlas graph model."""

from datetime import datetime, timezone

import numpy as np

from cloud.runtime.atlas import EpistemicAtlas
from cloud.runtime.models import (
    BoundingBox,
    EntityTrack,
    PredictionWindow,
    SceneState,
    WorldModelMetrics,
)
from cloud.runtime.service import RuntimeContainer


def _make_track(
    track_id: str,
    label: str,
    status: str = "visible",
) -> EntityTrack:
    now = datetime.now(timezone.utc)
    return EntityTrack(
        id=track_id,
        session_id="test",
        label=label,
        status=status,
        first_seen_at=now,
        last_seen_at=now,
        first_observation_id="obs_1",
        last_observation_id="obs_2",
        observations=["obs_1", "obs_2"],
        visibility_streak=2,
        persistence_confidence=0.8,
        continuity_score=0.7,
        last_similarity=0.9,
        prototype_embedding=[0.1] * 128,
        status_history=[status],
    )


def _make_scene_state(
    boxes: list[BoundingBox] | None = None,
    track_ids: list[str] | None = None,
) -> SceneState:
    now = datetime.now(timezone.utc)
    return SceneState(
        id="ws_test",
        session_id="test",
        created_at=now,
        observation_id="obs_1",
        proposal_boxes=boxes or [],
        entity_track_ids=track_ids or [],
        prediction_window=PredictionWindow(),
        metrics=WorldModelMetrics(),
    )


def test_nodes_created_from_tracks():
    atlas = EpistemicAtlas()
    tracks = [
        _make_track("trk_1", "cup"),
        _make_track("trk_2", "person"),
    ]
    scene = _make_scene_state(track_ids=["trk_1", "trk_2"])
    atlas.update(tracks, scene)

    nodes = atlas.get_nodes()
    assert len(nodes) == 2
    labels = {n.label for n in nodes}
    assert "cup" in labels
    assert "person" in labels


def test_edges_from_cooccurrence():
    atlas = EpistemicAtlas()
    tracks = [
        _make_track("trk_1", "cup"),
        _make_track("trk_2", "person"),
    ]
    scene = _make_scene_state(
        boxes=[
            BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2, label="cup"),
            BoundingBox(x=0.6, y=0.6, width=0.2, height=0.2, label="person"),
        ],
        track_ids=["trk_1", "trk_2"],
    )
    atlas.update(tracks, scene)
    edges = atlas.get_edges()
    assert len(edges) == 1
    assert edges[0].co_occurrence_count == 1
    assert edges[0].status == "active"


def test_edge_count_increments():
    atlas = EpistemicAtlas()
    tracks = [
        _make_track("trk_1", "cup"),
        _make_track("trk_2", "person"),
    ]
    scene = _make_scene_state(track_ids=["trk_1", "trk_2"])
    atlas.update(tracks, scene)
    atlas.update(tracks, scene)
    edges = atlas.get_edges()
    assert edges[0].co_occurrence_count == 2


def test_energy_map_applied_to_nodes():
    atlas = EpistemicAtlas()
    tracks = [_make_track("trk_1", "cup")]
    scene = _make_scene_state(
        boxes=[BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2, label="cup")],
        track_ids=["trk_1"],
    )
    energy_map = np.ones((7, 7), dtype=np.float32) * 0.42
    atlas.update(tracks, scene, energy_map=energy_map)

    nodes = atlas.get_nodes()
    assert len(nodes) == 1
    assert nodes[0].last_energy > 0.0


def test_to_dict_serialization():
    atlas = EpistemicAtlas()
    tracks = [
        _make_track("trk_1", "cup"),
        _make_track("trk_2", "person"),
    ]
    scene = _make_scene_state(track_ids=["trk_1", "trk_2"])
    atlas.update(tracks, scene)

    data = atlas.to_dict()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1


def test_reset_clears_atlas():
    atlas = EpistemicAtlas()
    tracks = [_make_track("trk_1", "cup")]
    scene = _make_scene_state(track_ids=["trk_1"])
    atlas.update(tracks, scene)
    assert len(atlas.get_nodes()) == 1
    atlas.reset()
    assert len(atlas.get_nodes()) == 0
    assert len(atlas.get_edges()) == 0


def test_disappeared_nodes_are_pruned():
    atlas = EpistemicAtlas(stale_seconds=30.0)
    visible_track = _make_track("trk_1", "cup", status="visible")
    scene = _make_scene_state(track_ids=["trk_1"])
    atlas.update([visible_track], scene)
    assert len(atlas.get_nodes()) == 1

    disappeared_track = visible_track.model_copy(update={"status": "disappeared"})
    later_scene = scene.model_copy(update={"id": "ws_test_2", "created_at": datetime.now(timezone.utc)})
    atlas.update([disappeared_track], later_scene)

    assert len(atlas.get_nodes()) == 0
    assert len(atlas.get_edges()) == 0


def test_world_state_uses_session_scoped_atlas(tmp_path):
    runtime = RuntimeContainer(data_dir=tmp_path)

    session_a_track = _make_track("trk_a", "cup").model_copy(update={"session_id": "session-a"})
    session_b_track = _make_track("trk_b", "chair").model_copy(update={"session_id": "session-b"})
    session_a_scene = _make_scene_state(track_ids=["trk_a"]).model_copy(update={"session_id": "session-a", "id": "ws_a"})
    session_b_scene = _make_scene_state(track_ids=["trk_b"]).model_copy(update={"session_id": "session-b", "id": "ws_b"})

    runtime._atlas_for_session("session-a").update([session_a_track], session_a_scene)
    runtime._atlas_for_session("session-b").update([session_b_track], session_b_scene)

    session_a_world = runtime.get_world_state("session-a")
    session_b_world = runtime.get_world_state("session-b")

    assert [node["label"] for node in session_a_world.atlas["nodes"]] == ["cup"]
    assert [node["label"] for node in session_b_world.atlas["nodes"]] == ["chair"]
