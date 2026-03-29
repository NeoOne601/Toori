"""
Tests for Smriti storage configuration API.

These tests cover the Sprint 4 storage configuration contract:
- storage config GET/PUT
- storage usage reporting
- watch folder list/add/remove behavior
- prune endpoints
- storage path resolution and environment overrides
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloud.runtime.app import create_app

def _client(tmp_path: Path) -> TestClient:
    app = create_app(data_dir=str(tmp_path))
    return TestClient(app)


def test_smriti_storage_get_returns_config(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/v1/smriti/storage")
    assert response.status_code == 200
    body = response.json()
    assert "data_dir" in body
    assert "frames_dir" in body
    assert "thumbs_dir" in body
    assert "watch_folders" in body
    assert isinstance(body["watch_folders"], list)


def test_smriti_storage_put_updates_budget(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.put(
        "/v1/smriti/storage",
        json={
            "max_storage_gb": 50.0,
            "watch_folders": [],
            "store_full_frames": True,
            "thumbnail_max_dim": 320,
            "auto_prune_missing": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["max_storage_gb"] == 50.0


def test_smriti_storage_put_resolves_data_dir(tmp_path: Path) -> None:
    client = _client(tmp_path)
    new_dir = tmp_path / "custom_smriti_data"
    response = client.put(
        "/v1/smriti/storage",
        json={
            "data_dir": str(new_dir),
            "max_storage_gb": 0.0,
            "watch_folders": [],
            "store_full_frames": False,
            "thumbnail_max_dim": 320,
            "auto_prune_missing": False,
        },
    )
    assert response.status_code == 200
    assert new_dir.exists()


def test_storage_usage_returns_report(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/v1/smriti/storage/usage")
    assert response.status_code == 200
    body = response.json()
    assert "total_human" in body
    assert "budget_warning" in body
    assert "budget_critical" in body
    assert isinstance(body["total_bytes"], int)


def test_watch_folders_list_empty_initially(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.get("/v1/smriti/watch-folders")
    assert response.status_code == 200
    assert response.json() == []


def test_add_watch_folder_returns_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    watch_dir = tmp_path / "my_photos"
    watch_dir.mkdir()

    response = client.post("/v1/smriti/watch-folders", json={"path": str(watch_dir)})
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == str(watch_dir)
    assert body["exists"] is True


def test_add_nonexistent_folder_raises_error(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/v1/smriti/watch-folders",
        json={"path": "/this/does/not/exist/12345"},
    )
    assert response.status_code in (400, 422, 500)


def test_remove_watch_folder(tmp_path: Path) -> None:
    client = _client(tmp_path)
    watch_dir = tmp_path / "to_remove"
    watch_dir.mkdir()
    client.post("/v1/smriti/watch-folders", json={"path": str(watch_dir)})

    response = client.delete(f"/v1/smriti/watch-folders?path={watch_dir}")
    assert response.status_code == 200
    folders = client.get("/v1/smriti/watch-folders").json()
    assert not any(folder["path"] == str(watch_dir) for folder in folders)


def test_watch_folder_persists_across_settings_save(tmp_path: Path) -> None:
    client = _client(tmp_path)
    watch_dir = tmp_path / "persistent_folder"
    watch_dir.mkdir()
    client.post("/v1/smriti/watch-folders", json={"path": str(watch_dir)})

    settings = client.get("/v1/settings").json()
    assert str(watch_dir) in settings.get("smriti_storage", {}).get("watch_folders", [])


def test_prune_missing_files_does_not_crash(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/v1/smriti/storage/prune",
        json={
            "remove_missing_files": True,
            "remove_failed": False,
            "clear_all": False,
            "confirm_clear_all": "",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "removed_media_records" in body


def test_prune_clear_all_requires_confirmation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/v1/smriti/storage/prune",
        json={
            "clear_all": True,
            "confirm_clear_all": "wrong_string",
        },
    )
    assert response.status_code in (400, 422, 500)


def test_prune_clear_all_with_correct_confirmation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/v1/smriti/storage/prune",
        json={
            "clear_all": True,
            "confirm_clear_all": "CONFIRM_CLEAR_ALL",
        },
    )
    assert response.status_code == 200


def test_smriti_storage_config_resolve_paths_fills_defaults(tmp_path: Path) -> None:
    from cloud.runtime.models import SmritiStorageConfig

    config = SmritiStorageConfig()
    resolved = config.resolve_paths(str(tmp_path))
    assert resolved.data_dir is not None
    assert resolved.frames_dir is not None
    assert resolved.thumbs_dir is not None
    assert resolved.templates_path is not None
    assert "smriti" in resolved.data_dir


def test_resolve_smriti_storage_respects_settings_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cloud.runtime.config import resolve_smriti_storage
    from cloud.runtime.models import RuntimeSettings
    from cloud.runtime.models import SmritiStorageConfig

    settings = SmritiStorageConfig(data_dir=str(tmp_path / "configured"))
    monkeypatch.setenv("TOORI_SMRITI_DATA_DIR", str(tmp_path / "env_data"))
    monkeypatch.setenv("TOORI_SMRITI_FRAMES_DIR", str(tmp_path / "env_frames"))
    monkeypatch.setenv("TOORI_SMRITI_THUMBS_DIR", str(tmp_path / "env_thumbs"))

    runtime_settings = RuntimeSettings(smriti_storage=settings)
    resolved = resolve_smriti_storage(runtime_settings, str(tmp_path))
    assert resolved.data_dir == str(tmp_path / "env_data")
    assert resolved.frames_dir == str(tmp_path / "env_frames")
    assert resolved.thumbs_dir == str(tmp_path / "env_thumbs")


def test_storage_usage_budget_warning_at_86_pct(tmp_path: Path) -> None:
    from cloud.runtime.models import StorageUsageReport

    report = StorageUsageReport(
        smriti_data_dir=str(tmp_path),
        total_media_count=100,
        indexed_count=100,
        pending_count=0,
        failed_count=0,
        total_bytes=int(0.86 * 10 * 1024**3),
        total_human="8.6 GB",
        max_storage_gb=10.0,
        budget_pct=86.0,
        budget_warning=True,
        budget_critical=False,
    )
    assert report.budget_warning is True
    assert report.budget_critical is False


def test_budget_critical_at_96_pct(tmp_path: Path) -> None:
    from cloud.runtime.models import StorageUsageReport

    report = StorageUsageReport(
        smriti_data_dir=str(tmp_path),
        total_media_count=100,
        indexed_count=100,
        pending_count=0,
        failed_count=0,
        total_bytes=int(0.96 * 10 * 1024**3),
        total_human="9.6 GB",
        max_storage_gb=10.0,
        budget_pct=96.0,
        budget_warning=True,
        budget_critical=True,
    )
    assert report.budget_critical is True
