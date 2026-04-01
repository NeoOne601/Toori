"""Sprint 6: Surprise Indexing Tests.

Tests the surprise-score based query surface exposed by SmetiDB
and the /v1/smriti/surprising route.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from cloud.runtime.smriti_storage import SmetiDB


def _embedding(seed: int, dim: int = 128) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[seed % dim] = 1.0
    return v


def _create_media(db, *, file_hash: str, embedding: np.ndarray, status: str = "complete"):
    return db.create_smriti_media(
        observation_id=None,
        file_path=f"/test/{file_hash}.png",
        file_hash=file_hash,
        media_type="image",
        ingestion_status=status,
        embedding=embedding,
    )


def test_get_surprising_media_returns_empty_by_default(tmp_path):
    db = SmetiDB(tmp_path)
    results = db.get_surprising_media(threshold=0.5, limit=10)
    assert results == []


def test_get_surprising_media_filters_by_threshold(tmp_path):
    db = SmetiDB(tmp_path)
    m1 = _create_media(db, file_hash="high-surprise", embedding=_embedding(0))
    m2 = _create_media(db, file_hash="low-surprise", embedding=_embedding(1))

    db.update_media_world_model_fields(m1.id, surprise_score=0.85)
    db.update_media_world_model_fields(m2.id, surprise_score=0.2)

    results = db.get_surprising_media(threshold=0.5, limit=10)
    assert len(results) == 1
    assert results[0].id == m1.id


def test_get_surprising_media_ordered_by_descending_surprise(tmp_path):
    db = SmetiDB(tmp_path)
    m1 = _create_media(db, file_hash="med", embedding=_embedding(0))
    m2 = _create_media(db, file_hash="high", embedding=_embedding(1))
    m3 = _create_media(db, file_hash="low", embedding=_embedding(2))

    db.update_media_world_model_fields(m1.id, surprise_score=0.6)
    db.update_media_world_model_fields(m2.id, surprise_score=0.95)
    db.update_media_world_model_fields(m3.id, surprise_score=0.51)

    results = db.get_surprising_media(threshold=0.5, limit=10)
    assert len(results) == 3
    assert results[0].id == m2.id  # highest
    assert results[1].id == m1.id
    assert results[2].id == m3.id


def test_get_surprising_media_respects_limit(tmp_path):
    db = SmetiDB(tmp_path)
    for i in range(5):
        m = _create_media(db, file_hash=f"s-{i}", embedding=_embedding(i))
        db.update_media_world_model_fields(m.id, surprise_score=0.7 + i * 0.01)

    results = db.get_surprising_media(threshold=0.5, limit=2)
    assert len(results) == 2


def test_get_surprising_media_excludes_pending(tmp_path):
    db = SmetiDB(tmp_path)
    m = _create_media(db, file_hash="pending-surprise", embedding=_embedding(0), status="pending")
    db.update_media_world_model_fields(m.id, surprise_score=0.9)

    results = db.get_surprising_media(threshold=0.5, limit=10)
    assert len(results) == 0


def test_update_media_world_model_fields_sets_all(tmp_path):
    db = SmetiDB(tmp_path)
    m = _create_media(db, file_hash="wm-update", embedding=_embedding(0))

    db.update_media_world_model_fields(
        m.id,
        l2_embedding=[0.1] * 128,
        prediction_error=0.42,
        surprise_score=0.75,
        world_model_version="vjepa2",
    )

    updated = db.get_smriti_media(m.id)
    assert updated.prediction_error is not None
    assert abs(updated.prediction_error - 0.42) < 0.01
    assert updated.surprise_score is not None
    assert abs(updated.surprise_score - 0.75) < 0.01
    assert updated.world_model_version == "vjepa2"


def test_get_media_by_world_model_version(tmp_path):
    db = SmetiDB(tmp_path)
    m1 = _create_media(db, file_hash="vjepa2-media", embedding=_embedding(0))
    m2 = _create_media(db, file_hash="surrogate-media", embedding=_embedding(1))

    db.update_media_world_model_fields(m1.id, world_model_version="vjepa2")
    db.update_media_world_model_fields(m2.id, world_model_version="surrogate")

    vjepa2_results = db.get_media_by_world_model_version("vjepa2")
    assert len(vjepa2_results) == 1
    assert vjepa2_results[0].id == m1.id


def test_update_media_world_model_fields_returns_false_when_empty(tmp_path):
    db = SmetiDB(tmp_path)
    m = _create_media(db, file_hash="empty-update", embedding=_embedding(0))
    result = db.update_media_world_model_fields(m.id)
    assert result is False
