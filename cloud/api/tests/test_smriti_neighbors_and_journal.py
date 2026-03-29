"""Additional Sprint 5 coverage for neighbors and journal payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from cloud.runtime.app import create_app


def _create_media(runtime, media_id: str, file_hash: str, embedding: np.ndarray, path: Path) -> None:
    runtime.smriti_db.create_smriti_media(
        id=media_id,
        file_path=str(path),
        file_hash=file_hash,
        media_type="image",
        ingestion_status="complete",
        embedding=embedding.astype(np.float32).tolist(),
        setu_descriptions=[{"text": "Object in background", "confidence": 0.8, "anchor_basis": "unknown"}],
    )


def test_neighbors_endpoint_registered(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    routes = {route.path for route in app.routes}
    assert "/v1/smriti/media/{media_id}/neighbors" in routes


def test_neighbors_returns_empty_for_unknown_media(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.get("/v1/smriti/media/does-not-exist/neighbors?top_k=3")
    assert response.status_code == 200
    assert response.json() == {"neighbors": []}


def test_neighbors_returns_related_media(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    _create_media(runtime, "media_a", "hash_a", np.ones(128, dtype=np.float32), tmp_path / "a.png")
    _create_media(runtime, "media_b", "hash_b", np.ones(128, dtype=np.float32) * 0.99, tmp_path / "b.png")
    _create_media(runtime, "media_c", "hash_c", np.concatenate([np.ones(64), -np.ones(64)]).astype(np.float32), tmp_path / "c.png")

    response = client.get("/v1/smriti/media/media_a/neighbors?top_k=2")
    assert response.status_code == 200
    body = response.json()
    assert body["neighbors"]
    neighbor_ids = [item["media_id"] for item in body["neighbors"]]
    assert "media_b" in neighbor_ids


def test_person_journal_includes_atlas_when_empty(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.get("/v1/smriti/person/Unknown/journal")
    assert response.status_code == 200
    body = response.json()
    assert "atlas" in body
    assert body["entries"] == []


def test_person_journal_returns_entries_for_linked_person(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    embedding = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    _create_media(runtime, "media_person", "hash_person", embedding, tmp_path / "person.png")
    person = runtime.smriti_db.create_person("Asha", embedding)
    runtime.smriti_db.link_person_to_media(person.id, "media_person", 1.0)

    response = client.get("/v1/smriti/person/Asha/journal")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    assert body["entries"][0]["media_id"] == "media_person"
    assert "atlas" in body
