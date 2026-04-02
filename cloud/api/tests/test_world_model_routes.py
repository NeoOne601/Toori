from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient

from cloud.runtime.app import create_app


class _DummyVJepa2Encoder:
    def __init__(self) -> None:
        self._loaded = True
        self.device = "mps"
        self._n_frames = 8

    def ensure_loaded(self) -> None:
        return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def encoder_type(self) -> str:
        return "vjepa2"

    @property
    def model_id(self) -> str:
        return "facebook/vjepa2-vitl-fpc64-256"

    @property
    def n_frames(self) -> int:
        return self._n_frames


def test_world_model_status_and_config_routes(tmp_path, monkeypatch):
    model_dir = tmp_path / "vjepa2-model"
    model_dir.mkdir()

    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    monkeypatch.setattr("cloud.perception.vjepa2_encoder._is_test_environment", lambda: False)
    monkeypatch.setattr("cloud.perception.vjepa2_encoder.get_vjepa2_encoder", lambda: _DummyVJepa2Encoder())

    status = client.get("/v1/world-model/status")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["encoder_type"] == "vjepa2"
    assert status_body["configured_encoder"] == "vjepa2"
    assert status_body["model_loaded"] is True
    assert status_body["device"] == "mps"
    assert status_body["n_frames"] == 8
    assert status_body["last_tick_encoder_type"] in {"not_loaded", "vjepa2", "surrogate"}
    assert status_body["telescope_test"] == "PASSED"

    update = client.put(
        "/v1/world-model/config",
        json={"model_path": str(model_dir), "n_frames": 4},
    )
    assert update.status_code == 200
    update_body = update.json()
    assert update_body["model_path"] == str(model_dir)
    assert update_body["n_frames"] == 4
    assert update_body["effective_model"] == str(model_dir.resolve())

    settings_mirror = Path(tmp_path) / "settings.json"
    assert settings_mirror.exists()
    assert '"vjepa2_model_path":' in settings_mirror.read_text()

    monkeypatch.delenv("TOORI_VJEPA2_MODEL", raising=False)
    monkeypatch.delenv("TOORI_VJEPA2_FRAMES", raising=False)
    monkeypatch.setenv("TOORI_VJEPA2_ENV", "production")
    monkeypatch.setenv("TOORI_DATA_DIR", str(tmp_path))

    import cloud.perception.vjepa2_encoder as encoder_module

    reloaded = reload(encoder_module)
    assert reloaded._is_test_environment() is False
    assert reloaded._resolve_model_id() == str(model_dir.resolve())
    assert reloaded._resolve_n_frames() == 4


def test_tool_state_planning_and_recovery_routes(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    observe = client.post(
        "/v1/tool-state/observe",
        json={
            "session_id": "hybrid",
            "state_domain": "browser",
            "current_url": "https://example.com/checkout",
            "view_id": "payment-modal",
            "visible_entities": [
                {
                    "id": "dialog-1",
                    "label": "payment dialog",
                    "kind": "dialog",
                    "state_domain": "browser",
                    "status": "visible",
                    "confidence": 0.9,
                    "properties": {},
                }
            ],
            "affordances": [
                {
                    "id": "retry-1",
                    "label": "retry payment",
                    "kind": "browser.retry",
                    "state_domain": "browser",
                    "availability": "error",
                    "confidence": 0.8,
                    "properties": {},
                }
            ],
            "error_banners": ["Payment failed"],
        },
    )
    assert observe.status_code == 200
    observe_body = observe.json()
    assert observe_body["scene_state"]["state_domain"] == "browser"
    assert observe_body["scene_state"]["grounded_entities"][0]["label"] == "payment dialog"

    rollout = client.post(
        "/v1/planning/rollout",
        json={"session_id": "hybrid", "horizon": 2},
    )
    assert rollout.status_code == 200
    rollout_body = rollout.json()
    assert rollout_body["comparison"]["ranked_branches"]
    assert rollout_body["comparison"]["chosen_branch_id"] is not None

    benchmark = client.post(
        "/v1/benchmarks/recovery/run",
        json={"session_id": "hybrid"},
    )
    assert benchmark.status_code == 200
    benchmark_body = benchmark.json()
    assert benchmark_body["scenarios"]
    benchmark_id = benchmark_body["id"]

    fetched = client.get(f"/v1/benchmarks/recovery/{benchmark_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == benchmark_id
