"""Tests for Smriti data migration service."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


def _make_fake_smriti_data(data_dir: Path) -> None:
    smriti = data_dir / "smriti"
    smriti.mkdir(parents=True, exist_ok=True)
    (smriti / "frames").mkdir(exist_ok=True)
    (smriti / "thumbs").mkdir(exist_ok=True)
    (smriti / "smriti_faiss.index").write_bytes(b"FAISS-LITE")
    (smriti / "sag_templates.json").write_text("[]")

    connection = sqlite3.connect(smriti / "smriti.sqlite3")
    connection.execute("CREATE TABLE IF NOT EXISTS demo (id TEXT PRIMARY KEY, value TEXT)")
    connection.execute("INSERT OR REPLACE INTO demo (id, value) VALUES (?, ?)", ("row_1", "ok"))
    connection.commit()
    connection.close()

    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    frame_buf = io.BytesIO()
    image.save(frame_buf, format="PNG")
    (smriti / "frames" / "obs_001.png").write_bytes(frame_buf.getvalue())

    thumb_buf = io.BytesIO()
    image.resize((8, 8)).save(thumb_buf, format="PNG")
    (smriti / "thumbs" / "obs_001.png").write_bytes(thumb_buf.getvalue())


def test_migration_dry_run_reports_what_would_be_moved(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    _make_fake_smriti_data(tmp_path)
    client = TestClient(app)
    target = tmp_path / "new_location"
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(target), "dry_run": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["success"] is True
    assert not target.exists()


def test_migration_copies_files_to_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    _make_fake_smriti_data(source)
    app = create_app(data_dir=str(source))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["files_moved"] > 0
    assert (destination / "smriti.sqlite3").exists()
    assert (destination / "frames" / "obs_001.png").exists()


def test_migration_preserves_source_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    _make_fake_smriti_data(source)
    source_smriti = source / "smriti"

    app = create_app(data_dir=str(source))
    client = TestClient(app)
    client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )

    assert source_smriti.exists()
    assert (source_smriti / "smriti.sqlite3").exists()


def test_migration_updates_config_on_success(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    _make_fake_smriti_data(source)

    app = create_app(data_dir=str(source))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

    storage_response = client.get("/v1/smriti/storage")
    assert storage_response.status_code == 200
    assert str(destination) in (storage_response.json().get("data_dir") or "")


def test_migration_to_nonwritable_directory_fails(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": "/root/no_permission_here_12345", "dry_run": False},
    )
    body = response.json()
    if response.status_code == 200:
        assert body["success"] is False
        assert body["errors"]


def test_migration_reports_bytes_moved(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    _make_fake_smriti_data(source)
    app = create_app(data_dir=str(source))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )
    body = response.json()
    assert body["bytes_moved"] > 0
    assert "B" in body["bytes_moved_human"]


def test_migration_does_not_break_existing_functionality(tmp_path):
    import base64

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    app = create_app(data_dir=str(source))
    client = TestClient(app)
    client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )

    image = Image.new("RGB", (32, 32), color=(100, 150, 200))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.post(
        "/v1/analyze",
        json={"image_base64": b64, "session_id": "post_migration", "decode_mode": "off"},
    )
    assert response.status_code == 200


def test_migration_endpoint_registered(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    routes = {route.path for route in app.routes}
    assert "/v1/smriti/storage/migrate" in routes


def test_migration_result_has_rollback_available_flag(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    _make_fake_smriti_data(source)
    app = create_app(data_dir=str(source))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(destination), "dry_run": False},
    )
    assert "rollback_available" in response.json()


def test_config_not_updated_on_dry_run(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    before = client.get("/v1/smriti/storage").json()
    client.post(
        "/v1/smriti/storage/migrate",
        json={"target_data_dir": str(tmp_path / "fake_target"), "dry_run": True},
    )
    after = client.get("/v1/smriti/storage").json()
    assert before.get("data_dir") == after.get("data_dir")
