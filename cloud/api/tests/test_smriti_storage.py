from __future__ import annotations

import concurrent.futures
import io
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

from cloud.runtime.smriti_storage import SmetiDB
from cloud.runtime.storage import ObservationStore


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> tuple[Image.Image, bytes]:
    image = Image.new("RGB", size, color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


def _embedding(seed: int, dim: int = 128) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[seed % dim] = 1.0
    if seed + 1 < dim:
        vector[(seed + 1) % dim] = 0.25
    return vector


def _create_observation(store: ObservationStore, *, session_id: str, color: tuple[int, int, int], embedding: np.ndarray):
    image, raw_bytes = _png_bytes(color)
    return store.create_observation(
        image=image,
        raw_bytes=raw_bytes,
        embedding=embedding.astype(np.float32).tolist(),
        session_id=session_id,
        confidence=0.9,
        novelty=0.1,
        source_query=None,
        tags=["test"],
        providers=["basic"],
        metadata={"source": "test"},
    )


def _create_media(
    db: SmetiDB,
    *,
    observation_id: str | None,
    file_path: str,
    file_hash: str,
    embedding: np.ndarray,
    status: str = "complete",
    media_type: str = "image",
    created_at: datetime | None = None,
    text: str = "test description",
    anchor: str = "anchor",
    location_id: str | None = None,
    depth_label: str = "foreground",
):
    return db.create_smriti_media(
        observation_id=observation_id,
        file_path=file_path,
        file_hash=file_hash,
        media_type=media_type,
        original_created_at=created_at,
        ingested_at=created_at or datetime.now(timezone.utc),
        depth_strata={depth_label: 1.0},
        anchor_matches=[{"name": anchor, "confidence": 0.92, "patch_indices": [0, 1], "energy": 0.1}],
        setu_descriptions=[{"text": text, "confidence": 0.88, "anchor_basis": anchor, "region_id": "r1"}],
        hallucination_risk=0.08,
        ingestion_status=status,
        location_id=location_id,
        embedding=embedding,
    )


def test_schema_migrations_are_idempotent(tmp_path):
    db = SmetiDB(tmp_path)

    db._apply_migrations()
    db._apply_migrations()

    with db._connect() as connection:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='smriti_media'"
        ).fetchone()

    assert row is not None
    assert db._schema_manager.current_version() == 4


def test_schema_version_increments_correctly(tmp_path):
    db = SmetiDB(tmp_path)

    assert db._schema_manager.current_version() == 4


def test_faiss_search_returns_correct_nearest_neighbor(tmp_path):
    db = SmetiDB(tmp_path)
    first = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "first.png"),
        file_hash="hash-first",
        embedding=_embedding(0),
    )
    second = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "second.png"),
        file_hash="hash-second",
        embedding=_embedding(1),
    )

    results = db.faiss_search(_embedding(0), top_k=2)

    assert results[0][0] == first.id
    assert results[0][1] >= results[1][1]
    assert second.id in {media_id for media_id, _ in results}


def test_faiss_add_increments_index_size(tmp_path):
    db = SmetiDB(tmp_path)
    first = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "first.png"),
        file_hash="hash-first",
        embedding=_embedding(2),
        status="pending",
    )
    second = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "second.png"),
        file_hash="hash-second",
        embedding=_embedding(3),
        status="pending",
    )

    assert db.faiss_add(first.id, _embedding(2)) == 0
    assert db.faiss_add(second.id, _embedding(3)) == 1
    assert db.faiss_search(_embedding(2), top_k=2, filter_status="pending")


def test_faiss_rebuild_from_smriti_media_table(tmp_path):
    db = SmetiDB(tmp_path)
    first = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "first.png"),
        file_hash="hash-first",
        embedding=_embedding(4),
    )
    _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "second.png"),
        file_hash="hash-second",
        embedding=_embedding(5),
    )

    db._faiss_index_path.unlink(missing_ok=True)
    rebuilt = SmetiDB(tmp_path)
    results = rebuilt.faiss_search(_embedding(4), top_k=1)

    assert results[0][0] == first.id


def test_media_hash_prevents_duplicate_ingestion(tmp_path):
    db = SmetiDB(tmp_path)
    first = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "first.png"),
        file_hash="duplicate-hash",
        embedding=_embedding(6),
    )
    duplicate = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "first.png"),
        file_hash="duplicate-hash",
        embedding=_embedding(6),
    )

    with db._connect() as connection:
        count = connection.execute("SELECT COUNT(*) AS count FROM smriti_media").fetchone()["count"]

    assert first.id == duplicate.id
    assert count == 1


def test_person_tag_propagates_to_high_similarity_media(tmp_path):
    db = SmetiDB(tmp_path)
    person = db.create_person("Ada", _embedding(0))
    keep = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "keep.png"),
        file_hash="keep-hash",
        embedding=_embedding(0),
    )
    _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "skip.png"),
        file_hash="skip-hash",
        embedding=_embedding(9),
    )

    linked = db.propagate_person_tag(person.id, threshold=0.82)
    with db._connect() as connection:
        rows = connection.execute(
            "SELECT media_id FROM smriti_person_media WHERE person_id = ?",
            (person.id,),
        ).fetchall()

    assert linked == 1
    assert {row["media_id"] for row in rows} == {keep.id}


def test_person_tag_does_not_propagate_below_threshold(tmp_path):
    db = SmetiDB(tmp_path)
    person = db.create_person("Grace", _embedding(0))
    _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "low.png"),
        file_hash="low-hash",
        embedding=_embedding(9),
    )

    linked = db.propagate_person_tag(person.id, threshold=0.99)

    assert linked == 0


def test_recall_search_respects_person_filter(tmp_path):
    db = SmetiDB(tmp_path)
    person = db.create_person("Mira", _embedding(0))
    keep = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "keep.png"),
        file_hash="keep-hash",
        embedding=_embedding(0),
    )
    skip = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "skip.png"),
        file_hash="skip-hash",
        embedding=_embedding(0),
    )
    db.link_person_to_media(person.id, keep.id, 0.98)

    results = db.recall_search(_embedding(0), top_k=5, person_filter=person.name)

    assert results and all(result.media_id == keep.id for result in results)
    assert skip.id not in {result.media_id for result in results}


def test_recall_search_respects_time_range(tmp_path):
    db = SmetiDB(tmp_path)
    old_time = datetime.now(timezone.utc) - timedelta(days=5)
    recent_time = datetime.now(timezone.utc)
    old = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "old.png"),
        file_hash="old-hash",
        embedding=_embedding(0),
        created_at=old_time,
    )
    recent = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "recent.png"),
        file_hash="recent-hash",
        embedding=_embedding(0),
        created_at=recent_time,
    )

    results = db.recall_search(
        _embedding(0),
        top_k=5,
        time_start=datetime.now(timezone.utc) - timedelta(days=1),
    )

    assert results and {result.media_id for result in results} == {recent.id}
    assert old.id not in {result.media_id for result in results}


def test_hybrid_search_combines_vector_and_fts_scores(tmp_path):
    db = SmetiDB(tmp_path)
    primary = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "primary.png"),
        file_hash="primary-hash",
        embedding=_embedding(0),
        text="red coffee mug on desk",
        anchor="mug",
    )
    _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "secondary.png"),
        file_hash="secondary-hash",
        embedding=_embedding(1),
        text="blue chair near window",
        anchor="chair",
    )

    results = db.hybrid_search(_embedding(0), "coffee mug", top_k=2)

    assert results[0].media_id == primary.id
    assert results[0].hybrid_score >= results[1].hybrid_score


def test_mandala_data_is_cached_and_served_fast(tmp_path, monkeypatch):
    db = SmetiDB(tmp_path)
    media = _create_media(
        db,
        observation_id=None,
        file_path=str(tmp_path / "cluster.png"),
        file_hash="cluster-hash",
        embedding=_embedding(0),
    )
    db.update_clusters([media.id], np.array([_embedding(0)]))

    calls = {"count": 0}
    original = db._build_mandala_data

    def wrapped():
        calls["count"] += 1
        return original()

    monkeypatch.setattr(db, "_build_mandala_data", wrapped)

    first = db.get_mandala_data()
    second = db.get_mandala_data()

    assert first == second
    assert calls["count"] == 1


def test_concurrent_reads_do_not_block_each_other(tmp_path):
    db = SmetiDB(tmp_path)
    media = [
        _create_media(
            db,
            observation_id=None,
            file_path=str(tmp_path / f"{index}.png"),
            file_hash=f"hash-{index}",
            embedding=_embedding(index),
        )
        for index in range(4)
    ]

    def read_task(index: int) -> str:
        item = db.get_smriti_media(media[index % len(media)].id)
        assert item is not None
        hits = db.recall_search(_embedding(index % len(media)), top_k=2)
        assert hits
        return item.id

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_task, range(32)))

    assert len(results) == 32


def test_smriti_schema_does_not_break_existing_observation_tests(tmp_path):
    store = ObservationStore(tmp_path)
    observation = _create_observation(store, session_id="baseline", color=(10, 20, 30), embedding=_embedding(0))

    smriti = SmetiDB(tmp_path)

    assert smriti.get_observation(observation.id) is not None
    assert smriti.list_observations(session_id="baseline", limit=5)[0].id == observation.id
