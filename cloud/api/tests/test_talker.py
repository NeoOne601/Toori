"""Tests for the Selective Talker event generation."""

from datetime import datetime, timezone

from cloud.runtime.models import EntityTrack, PersistenceSignal
from cloud.runtime.talker import SelectiveTalker


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


def test_scene_stable_when_below_threshold():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.05,
        threshold=0.15,
        should_talk=False,
        entity_tracks=[_make_track("trk_1", "cup")],
        previous_tracks=[_make_track("trk_1", "cup")],
        persistence_signal=PersistenceSignal(
            visible_track_ids=["trk_1"],
            persistence_confidence=0.9,
        ),
    )
    assert result is not None
    assert result.event_type == "SCENE_STABLE"
    assert result.energy_summary == 0.05


def test_entity_appeared():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.3,
        threshold=0.15,
        should_talk=True,
        entity_tracks=[
            _make_track("trk_1", "cup"),
            _make_track("trk_2", "person"),
        ],
        previous_tracks=[_make_track("trk_1", "cup")],
        persistence_signal=PersistenceSignal(
            visible_track_ids=["trk_1", "trk_2"],
            persistence_confidence=0.8,
        ),
    )
    assert result is not None
    assert result.event_type == "ENTITY_APPEARED"
    assert "trk_2" in result.entity_ids
    assert "person" in result.description


def test_occlusion_end():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.25,
        threshold=0.15,
        should_talk=True,
        entity_tracks=[
            _make_track("trk_1", "cup", status="re-identified"),
        ],
        previous_tracks=[
            _make_track("trk_1", "cup", status="occluded"),
        ],
        persistence_signal=PersistenceSignal(
            visible_track_ids=[],
            recovered_track_ids=["trk_1"],
            persistence_confidence=0.7,
        ),
    )
    assert result is not None
    assert result.event_type == "OCCLUSION_END"
    assert "trk_1" in result.entity_ids


def test_prediction_violation():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.4,
        threshold=0.15,
        should_talk=True,
        entity_tracks=[
            _make_track("trk_1", "cup", status="violated prediction"),
        ],
        previous_tracks=[
            _make_track("trk_1", "cup", status="visible"),
        ],
        persistence_signal=PersistenceSignal(
            violated_track_ids=["trk_1"],
            persistence_confidence=0.3,
        ),
    )
    assert result is not None
    assert result.event_type == "PREDICTION_VIOLATION"
    assert "trk_1" in result.entity_ids


def test_occlusion_start():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.2,
        threshold=0.15,
        should_talk=True,
        entity_tracks=[
            _make_track("trk_1", "cup", status="occluded"),
        ],
        previous_tracks=[
            _make_track("trk_1", "cup", status="visible"),
        ],
        persistence_signal=PersistenceSignal(
            occluded_track_ids=["trk_1"],
            persistence_confidence=0.5,
        ),
    )
    assert result is not None
    assert result.event_type == "OCCLUSION_START"
    assert "trk_1" in result.entity_ids


def test_entity_disappeared():
    talker = SelectiveTalker()
    result = talker.evaluate(
        mean_energy=0.25,
        threshold=0.15,
        should_talk=True,
        entity_tracks=[
            _make_track("trk_1", "cup", status="disappeared"),
        ],
        previous_tracks=[
            _make_track("trk_1", "cup", status="visible"),
            _make_track("trk_2", "book", status="visible"),
        ],
        persistence_signal=PersistenceSignal(
            disappeared_track_ids=["trk_2"],
            persistence_confidence=0.4,
        ),
    )
    assert result is not None
    assert result.event_type == "ENTITY_DISAPPEARED"
    assert "trk_2" in result.entity_ids
