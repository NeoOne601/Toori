from importlib import reload
from pathlib import Path

from fastapi.testclient import TestClient

from cloud.runtime.app import create_app
from cloud.runtime.jepa_worker import JEPAWorkerPreflightResult


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
    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir()

    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    monkeypatch.setattr("cloud.perception.vjepa2_encoder._is_test_environment", lambda: False)
    monkeypatch.setattr("cloud.perception.vjepa2_encoder.has_local_vjepa2_weights", lambda *args, **kwargs: True)
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
    assert status_body["active_backend"] in {"not_loaded", "vjepa2"}
    assert status_body["native_ready"] is False
    assert status_body["preflight_status"] == "not_run"
    assert status_body["telescope_test"] == "PASSED"

    update = client.put(
        "/v1/world-model/config",
        json={"model_path": str(model_dir), "cache_dir": str(cache_dir), "n_frames": 4},
    )
    assert update.status_code == 200
    update_body = update.json()
    assert update_body["model_path"] == str(model_dir)
    assert update_body["cache_dir"] == str(cache_dir.resolve())
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
    assert reloaded._resolve_cache_dir() == cache_dir.resolve()
    assert reloaded._resolve_n_frames() == 4


def test_runtime_quarantines_native_path_after_preflight_failure(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    runtime = app.state.runtime

    monkeypatch.setattr(
        "cloud.runtime.service.run_jepa_worker_preflight",
        lambda timeout_s=45.0: JEPAWorkerPreflightResult(
            ok=False,
            stage="failed",
            error="worker process exited without returning a result",
            crash_fingerprint="deadbeefcafe",
        ),
    )

    pool = runtime._ensure_jepa_pool()
    assert pool._disable_vjepa2 is True
    assert runtime._jepa_safe_fallback_reason == "worker process exited without returning a result"

    status = runtime.get_world_model_status()
    assert status.native_ready is False
    assert status.preflight_status == "failed"
    assert status.crash_fingerprint == "deadbeefcafe"
    assert status.degraded is True


def test_cpu_worker_defaults_native_jepa_to_four_frames(monkeypatch):
    monkeypatch.delenv("TOORI_VJEPA2_FRAMES", raising=False)
    monkeypatch.delenv("TOORI_VJEPA2_ENV", raising=False)
    monkeypatch.setenv("TOORI_VJEPA2_DEVICE", "cpu")

    import cloud.perception.vjepa2_encoder as encoder_module

    reloaded = reload(encoder_module)
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


def test_challenge_evaluate_uses_stored_scene_state_surprise_semantics(tmp_path):
    """
    Regression: /v1/challenges/evaluate must score `surprise_increases_on_change` using
    SceneState.metrics.surprise_score (world-model semantics), not heatmap energy.
    """
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    def analyze(session_id: str, seed: int):
        return client.post(
            "/v1/analyze",
            json={
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ccH1n8AAAAASUVORK5CYII=",
                "session_id": session_id,
                "decode_mode": "off",
                "query": f"seed {seed}",
            },
        ).json()["observation"]

    session_id = "challenge-semantics"
    obs_a = analyze(session_id, 1)
    obs_b = analyze(session_id, 2)
    obs_c = analyze(session_id, 3)

    from datetime import datetime, timezone
    from cloud.runtime.models import SceneState, WorldModelMetrics

    stable = WorldModelMetrics(
        prediction_consistency=0.9,
        surprise_score=0.1,
        temporal_continuity_score=0.9,
        persistence_confidence=0.8,
        occlusion_recovery_score=0.0,
    )
    changed = WorldModelMetrics(
        prediction_consistency=0.3,
        surprise_score=0.7,
        temporal_continuity_score=0.4,
        persistence_confidence=0.7,
        occlusion_recovery_score=0.0,
    )

    ws_a = SceneState(
        id=f"ws_{obs_a['id']}",
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        observation_id=obs_a["id"],
        observed_elements=["desk"],
        changed_elements=[],
        metrics=stable,
    )
    ws_b = SceneState(
        id=f"ws_{obs_b['id']}",
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        observation_id=obs_b["id"],
        observed_elements=["desk", "lamp"],
        changed_elements=["new:lamp"],
        metrics=changed,
    )
    ws_c = SceneState(
        id=f"ws_{obs_c['id']}",
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        observation_id=obs_c["id"],
        observed_elements=["desk", "lamp"],
        changed_elements=[],
        metrics=stable,
    )

    for obs, ws in [(obs_a, ws_a), (obs_b, ws_b), (obs_c, ws_c)]:
        runtime.store.save_scene_state(ws)
        runtime.store.update_observation(obs["id"], world_state_id=ws.id)

    result = client.post(
        "/v1/challenges/evaluate",
        json={"session_id": session_id, "challenge_set": "live", "proof_mode": "both", "limit": 10},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["success_criteria"]["surprise_increases_on_change"] is True
