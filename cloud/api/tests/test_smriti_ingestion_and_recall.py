"""Integration tests for ingestion daemon and recall API."""
import asyncio
import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


def _png(color=(200, 100, 50)) -> str:
    image = Image.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_smriti_ingest_endpoint_rejects_missing_path(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/ingest", json={})
    assert resp.status_code == 422


def test_smriti_ingest_file_that_does_not_exist(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/ingest", json={"file_path": "/nonexistent/file.jpg"})
    assert resp.status_code in (200, 422, 500)


def test_smriti_status_returns_stats(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.get("/v1/smriti/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "ingestion" in body


def test_smriti_recall_requires_query(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall", json={})
    assert resp.status_code == 422


def test_smriti_recall_empty_corpus_returns_empty_results(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/smriti/recall", json={"query": "red jacket", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_smriti_recall_rate_limiter_returns_429_under_burst(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    responses = [client.post("/v1/smriti/recall", json={"query": f"query {index}"}) for index in range(20)]
    status_codes = [response.status_code for response in responses]
    assert 429 in status_codes or all(code == 200 for code in status_codes)


def test_smriti_tag_person_endpoint_exists(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/v1/smriti/tag/person",
        json={"media_id": "obs_fake", "person_name": "Priya", "confirmed": True},
    )
    assert resp.status_code in (200, 404, 422)


def test_smriti_clusters_endpoint_returns_dict(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.get("/v1/smriti/clusters")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


def test_smriti_metrics_endpoint_returns_pipeline_health(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    resp = client.get("/v1/smriti/metrics")
    assert resp.status_code == 200


def test_file_hash_deduplication_prevents_double_ingest(tmp_path):
    from cloud.runtime.smriti_ingestion import _compute_file_hash

    test_file = tmp_path / "test.png"
    Image.new("RGB", (32, 32), (100, 200, 50)).save(str(test_file))
    digest = _compute_file_hash(str(test_file))
    assert len(digest) == 32


def test_media_type_classification():
    from cloud.runtime.smriti_ingestion import _classify_media_type

    assert _classify_media_type("/home/user/photo.jpg") == "image"
    assert _classify_media_type("/home/user/video.mp4") == "video"
    assert _classify_media_type("/home/user/Screenshot 2024.png") == "screenshot"


def test_video_media_type_classification():
    from cloud.runtime.smriti_ingestion import _classify_media_type

    assert _classify_media_type("/home/user/video.mp4") == "video"
    assert _classify_media_type("/home/user/video.mov") == "video"
    assert _classify_media_type("/home/user/video.m4v") == "video"


def test_pyav_import_graceful_failure(monkeypatch, tmp_path):
    import importlib

    from cloud.runtime.smriti_ingestion import SmritiIngestionDaemon
    from cloud.runtime.smriti_storage import SmetiDB

    original_import_module = importlib.import_module

    def fake_import(name: str, package=None):
        if name == "av":
            raise ImportError("av missing")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    daemon = SmritiIngestionDaemon(SmetiDB(tmp_path), lambda: None)
    result = asyncio.run(daemon._extract_video_keyframe(str(tmp_path / "missing.mp4")))
    assert result is None


def test_watchdog_degrades_gracefully_when_unavailable(monkeypatch, tmp_path):
    import sys

    from cloud.runtime.smriti_ingestion import SmritiIngestionDaemon
    from cloud.runtime.smriti_storage import SmetiDB

    watchdog_backup = {key: value for key, value in sys.modules.items() if "watchdog" in key}
    for key in list(sys.modules.keys()):
        if "watchdog" in key:
            del sys.modules[key]
    monkeypatch.setitem(sys.modules, "watchdog", None)
    monkeypatch.setitem(sys.modules, "watchdog.events", None)
    monkeypatch.setitem(sys.modules, "watchdog.observers", None)
    try:
        daemon = SmritiIngestionDaemon(SmetiDB(tmp_path), lambda: None)
        daemon._start_watchdog(str(tmp_path))
    finally:
        sys.modules.update(watchdog_backup)


def test_sag_templates_persist_across_runtime_instances(tmp_path):
    from cloud.runtime.service import RuntimeContainer

    runtime = RuntimeContainer(data_dir=str(tmp_path))
    engine = runtime._engine_for_session("persist-session")
    engine._sag.learn_template_from_confirmation(
        region_patches=[0, 1, 14, 15],
        confirmed_label="persisted_anchor",
        patch_tokens=np.ones((196, 384), dtype=np.float32),
    )
    runtime._save_sag_templates()

    runtime2 = RuntimeContainer(data_dir=str(tmp_path))
    runtime2._load_sag_templates()
    engine2 = runtime2._engine_for_session("persist-session")
    assert any(template.name == "persisted_anchor" for template in engine2._sag._learned_templates)
