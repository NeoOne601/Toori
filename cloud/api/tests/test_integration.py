import asyncio
import base64
import io
from collections import deque
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app
from cloud.runtime.models import Answer, BoundingBox, JEPATick, ProviderHealth, SceneState, WorldModelMetrics


def _encoded_png(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_analyze_query_and_observation_history(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    first = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((255, 0, 0)), "session_id": "demo", "decode_mode": "off"},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert len(first_body["observation"]["embedding"]) == 128
    assert first_body["hits"] == []

    second = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((250, 10, 10)), "session_id": "demo", "decode_mode": "off"},
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["hits"], "second observation should retrieve the first one"

    runtime = app.state.runtime
    runtime.store.update_observation(
        first_body["observation"]["id"],
        summary="red coffee mug on desk",
        providers=["basic", "manual"],
        metadata={"summary_source": "test"},
    )

    query = client.post("/v1/query", json={"query": "coffee mug", "session_id": "demo", "top_k": 5})
    assert query.status_code == 200
    query_body = query.json()
    assert query_body["hits"], "text query should find the annotated observation"

    observations = client.get("/v1/observations", params={"session_id": "demo", "limit": 5})
    assert observations.status_code == 200
    assert len(observations.json()["observations"]) == 2

    observations_summary = client.get(
        "/v1/observations",
        params={"session_id": "demo", "limit": 5, "summary_only": True},
    )
    assert observations_summary.status_code == 200
    assert "embedding" not in observations_summary.json()["observations"][0]

    thumbnail_path = second_body["observation"]["thumbnail_path"]
    file_response = client.get("/v1/file", params={"path": thumbnail_path})
    assert file_response.status_code == 200
    assert file_response.content

    external = tmp_path.parent / "outside-runtime.txt"
    external.write_text("outside")
    forbidden = client.get("/v1/file", params={"path": str(external)})
    assert forbidden.status_code == 403


def test_runtime_snapshot_skips_provider_health_probe(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    analyzed = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((20, 80, 200)), "session_id": "snapshot", "decode_mode": "off"},
    )
    assert analyzed.status_code == 200

    runtime = app.state.runtime
    monkeypatch.setattr(
        runtime.providers.mlx,
        "health",
        lambda config: (_ for _ in ()).throw(AssertionError("snapshot should not call mlx health")),
    )

    snapshot = client.get("/v1/runtime/snapshot", params={"session_id": "snapshot", "observation_limit": 8})
    assert snapshot.status_code == 200
    body = snapshot.json()
    assert body["session_id"] == "snapshot"
    assert body["observation_count"] >= 1
    assert body["observations"]


def test_object_proposals_use_existing_onnx_path(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    runtime = app.state.runtime
    settings = runtime.get_settings()
    settings.providers["onnx"].enabled = True
    runtime.update_settings(settings)

    image = Image.new("RGB", (96, 96), color=(255, 255, 255))
    for x in range(0, 48):
        for y in range(0, 96):
            image.putpixel((x, y), (220, 40, 40))
    for x in range(48, 96):
        for y in range(0, 96):
            image.putpixel((x, y), (35, 35, 210))

    monkeypatch.setattr(
        runtime.providers.onnx,
        "health",
        lambda config: ProviderHealth(name="onnx", role="perception", enabled=True, healthy=True, message="ready"),
    )

    def fake_onnx_perceive(crop, config):
        r, g, b = crop.convert("RGB").resize((1, 1)).getpixel((0, 0))
        label = "chair" if r >= b else "person"
        return [0.1] * 128, 0.92, {"descriptor": "onnx", "top_label": label}

    monkeypatch.setattr(runtime.providers.onnx, "perceive", fake_onnx_perceive)

    proposals = runtime.providers.object_proposals(settings, image, provider_name="onnx")
    assert proposals, "ONNX crop proposals should produce labeled regions"
    assert all(box.label in {"chair", "person"} for box in proposals)
    assert any(box.label == "chair" for box in proposals)
    assert any(box.label == "person" for box in proposals)


def test_living_lens_refines_generic_answers_with_object_proposals(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "mlx")

    monkeypatch.setattr(
        runtime.providers.mlx,
        "health",
        lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    monkeypatch.setattr(
        runtime.providers.mlx,
        "reason",
        lambda **kwargs: Answer(text="red dominant balanced textured scene", provider="mlx", confidence=0.61),
    )
    monkeypatch.setattr(
        runtime.providers,
        "object_proposals",
        lambda settings, image, provider_name, max_proposals=5: [
            BoundingBox(x=0.18, y=0.2, width=0.42, height=0.56, label="person", score=0.93),
            BoundingBox(x=0.56, y=0.22, width=0.3, height=0.42, label="chair", score=0.88),
        ],
    )

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "proposal-summary",
            "query": "Describe it",
            "decode_mode": "force",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["text"] == "red dominant balanced textured scene"
    assert body["observation"]["summary"] != "red dominant balanced textured scene"
    assert "person" in body["observation"]["summary"]
    assert body["scene_state"]["primary_object_label"] == "person"
    assert len(body["scene_state"]["proposal_boxes"]) == 2
    assert body["observation"]["metadata"]["primary_object_label"] == "person"
    assert len(body["observation"]["metadata"]["object_proposals"]) == 2


def _enable_local_reasoning(runtime, backend: str) -> None:
    settings = runtime.get_settings()
    settings.local_reasoning_disabled = False
    settings.reasoning_backend = backend
    settings.providers["ollama"].enabled = True
    settings.providers["mlx"].enabled = True
    settings.providers["cloud"].enabled = False
    runtime.update_settings(settings)


def test_forced_ollama_analyze_returns_reasoning_trace(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "ollama")

    monkeypatch.setattr(
      runtime.providers.ollama,
      "health",
      lambda config: ProviderHealth(name="ollama", role="reasoning", enabled=True, healthy=True, message="ready"),
    )
    monkeypatch.setattr(
      runtime.providers.ollama,
      "reason",
      lambda **kwargs: Answer(text="desk lamp", provider="ollama", confidence=0.82),
    )

    response = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((90, 120, 200)), "session_id": "trace", "query": "Describe it", "decode_mode": "force"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["provider"] == "ollama"
    assert body["reasoning_trace"][0]["provider"] == "ollama"
    assert body["reasoning_trace"][0]["success"] is True
    stored = runtime.store.get_observation(body["observation"]["id"])
    assert stored is not None
    assert stored.metadata["reasoning_trace"][0]["provider"] == "ollama"


def test_auto_reasoning_falls_back_from_ollama_to_mlx(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "auto")

    monkeypatch.setattr(
      runtime.providers.ollama,
      "health",
      lambda config: ProviderHealth(name="ollama", role="reasoning", enabled=True, healthy=True, message="ready"),
    )
    monkeypatch.setattr(
      runtime.providers.ollama,
      "reason",
      lambda **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "health",
      lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "reason",
      lambda **kwargs: Answer(text="fallback mlx answer", provider="mlx", confidence=0.61),
    )

    response = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((255, 120, 10)), "session_id": "trace", "query": "Describe it", "decode_mode": "force"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["provider"] == "mlx"
    assert [entry["provider"] for entry in body["reasoning_trace"][:2]] == ["ollama", "mlx"]
    assert body["reasoning_trace"][0]["success"] is False
    assert body["reasoning_trace"][0]["error"] == "timed out"
    assert body["reasoning_trace"][1]["success"] is True


def _dummy_tick(*, ts_ms: int = 1_700_000_000_000, mean_energy: float = 0.2, backend: str = "dinov2-vits14-onnx") -> JEPATick:
    return JEPATick(
        energy_map=__import__("numpy").zeros((14, 14), dtype=__import__("numpy").float32),
        entity_tracks=[],
        talker_event=None,
        sigreg_loss=0.42,
        forecast_errors={},
        session_fingerprint=__import__("numpy").zeros((128,), dtype=__import__("numpy").float32),
        planning_time_ms=0.0,
        caption_score=0.0,
        retrieval_score=0.0,
        timestamp_ms=int(ts_ms),
        warmup=False,
        mask_results=[],
        mean_energy=float(mean_energy),
        energy_std=0.0,
        world_model_version=backend,
        configured_encoder="vjepa2",
        last_tick_encoder_type=backend,
        degraded=False,
        degrade_reason=None,
        degrade_stage=None,
        gemma4_alert=None,
    )


def test_proof_report_generate_does_not_use_non_callable_mlx_provider(tmp_path, monkeypatch):
    """
    Regression: proof export must not pass the MlxReasoningProvider object directly into Gemma4Bridge,
    because Gemma4Bridge expects an async callable (`await self._call(...)`).
    """
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    session_id = "proof-noncallable"
    runtime._jepa_ticks_by_session[session_id] = deque([_dummy_tick(mean_energy=0.25)], maxlen=256)

    settings = runtime.get_settings()
    settings.providers["mlx"].enabled = True
    runtime.update_settings(settings)

    class _FakeMlxProvider:
        name = "mlx"

        def health(self, config):
            return ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="ok")

        async def aquery(self, *, prompt, image_base64=None, system=None, max_tokens=256):
            return {"text": "proof narration ok", "model": "fake", "latency_ms": 1.0}

        def reason(self, **kwargs):
            return Answer(text="proof narration ok", provider="mlx", confidence=0.7)

        def query(self, **kwargs):
            return {"text": "proof narration ok", "model": "fake", "latency_ms": 1.0}

    fake = _FakeMlxProvider()
    original_get = runtime.providers.get
    monkeypatch.setattr(runtime.providers, "get", lambda name: fake if name == "mlx" else original_get(name))

    response = client.post("/v1/proof-report/generate", json={"session_id": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["generated"] is True

    latest = client.get("/v1/proof-report/latest")
    assert latest.status_code == 200
    content = latest.content
    assert b"object is not callable" not in content


def test_living_lens_tick_does_not_overwrite_surprise_with_mean_energy(tmp_path, monkeypatch):
    """
    Regression: scene_state.metrics.surprise_score is owned by world_model.py and must not be
    overwritten with jepa_tick.mean_energy (energy activation).
    """
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    analyzed = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((10, 20, 30)), "session_id": "surprise", "decode_mode": "off"},
    )
    assert analyzed.status_code == 200
    obs_id = analyzed.json()["observation"]["id"]
    observation = runtime.store.get_observation(obs_id)
    assert observation is not None

    expected_surprise = 0.12
    expected_consistency = 0.88
    scene_state = SceneState(
        id=f"ws_{obs_id}",
        session_id="surprise",
        created_at=datetime.now(timezone.utc),
        observation_id=obs_id,
        metrics=WorldModelMetrics(
            prediction_consistency=expected_consistency,
            surprise_score=expected_surprise,
            temporal_continuity_score=0.5,
            persistence_confidence=0.5,
            occlusion_recovery_score=0.0,
        ),
    )

    def fake_analyze_with_world_model(request, _live):
        from cloud.runtime.models import AnalyzeResponse

        return AnalyzeResponse(observation=observation, hits=[], answer=None, provider_health=[], reasoning_trace=[]), scene_state, []

    async def fake_run_jepa_tick(*args, **kwargs):
        tick = _dummy_tick(mean_energy=0.91, backend="dinov2-vits14-onnx").to_payload().model_dump(mode="json")
        return type(
            "_Result",
            (),
            {
                "correlation_id": "corr",
                "session_id": "surprise",
                "observation_id": obs_id,
                "jepa_tick_dict": tick,
                "pipeline_trace": {},
                "error": None,
                "state_vector": [0.0] * 384,
                "worker_id": 0,
            },
        )()

    monkeypatch.setattr(runtime, "_analyze_with_world_model", fake_analyze_with_world_model)
    monkeypatch.setattr(runtime, "_run_jepa_tick", fake_run_jepa_tick)
    monkeypatch.setattr(runtime, "_load_frame_array", lambda _path: __import__("numpy").zeros((256, 256, 3), dtype=__import__("numpy").uint8))
    monkeypatch.setattr(runtime, "_update_observation_from_tick", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "_prime_forecast_engine", lambda *a, **k: None)

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((20, 40, 80)),
            "session_id": "surprise",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert abs(body["scene_state"]["metrics"]["surprise_score"] - expected_surprise) < 1e-6
    assert abs(body["scene_state"]["metrics"]["surprise_score"] - 0.91) > 1e-3


def test_forced_mlx_http_analyze_returns_answer(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "mlx")

    monkeypatch.setattr(
      runtime.providers.mlx,
      "health",
      lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "reason",
      lambda **kwargs: Answer(text="mlx direct answer", provider="mlx", confidence=0.63),
    )

    response = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((15, 240, 15)), "session_id": "mlx", "query": "What is this?", "decode_mode": "force"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]["provider"] == "mlx"
    assert body["observation"]["providers"] in (["basic", "mlx"], ["onnx", "mlx"], ["dinov2", "mlx"])
    assert body["reasoning_trace"][0]["provider"] == "mlx"
    assert body["reasoning_trace"][0]["success"] is True


def test_mlx_malformed_output_produces_failed_trace(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "mlx")

    monkeypatch.setattr(
      runtime.providers.ollama,
      "health",
      lambda config: ProviderHealth(name="ollama", role="reasoning", enabled=True, healthy=False, message="unsupported in this test"),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "health",
      lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "reason",
      lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mlx returned malformed json")),
    )

    response = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((30, 30, 30)), "session_id": "mlx", "query": "What is this?", "decode_mode": "force"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["reasoning_trace"][0]["provider"] == "mlx"
    assert body["reasoning_trace"][0]["success"] is False
    assert body["reasoning_trace"][0]["error"] == "mlx returned malformed json"


def test_text_query_skips_mlx_without_image_input(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "mlx")

    monkeypatch.setattr(
      runtime.providers.ollama,
      "health",
      lambda config: ProviderHealth(name="ollama", role="reasoning", enabled=True, healthy=True, message="ready"),
    )
    monkeypatch.setattr(
      runtime.providers.ollama,
      "reason",
      lambda **kwargs: Answer(text="memory-only answer", provider="ollama", confidence=0.71),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "health",
      lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    monkeypatch.setattr(
      runtime.providers.mlx,
      "reason",
      lambda **kwargs: (_ for _ in ()).throw(AssertionError("mlx should not be called for text-only queries")),
    )

    first = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((200, 30, 30)), "session_id": "text-query", "decode_mode": "off"},
    )
    assert first.status_code == 200

    response = client.post(
        "/v1/query",
        json={"query": "red scene", "session_id": "text-query", "top_k": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hits"], "text-only query should still return memory hits"
    assert body["answer"]["provider"] == "ollama"
    assert body["reasoning_trace"][0]["provider"] == "ollama"
    assert body["reasoning_trace"][0]["success"] is True


def test_provider_health_reports_circuit_open_consistently(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime
    _enable_local_reasoning(runtime, "mlx")

    monkeypatch.setattr(
      runtime.providers.mlx,
      "health",
      lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="configured"),
    )
    runtime.providers._record_failure("mlx", "timed out")
    runtime.providers._record_failure("mlx", "timed out")

    response = client.get("/v1/providers/health")
    assert response.status_code == 200
    health = next(item for item in response.json()["providers"] if item["name"] == "mlx")
    assert health["healthy"] is False
    assert "circuit open" in health["message"]


def test_living_lens_tick_exposes_scene_state_and_tracks(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "proof",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scene_state"]["observation_id"] == body["observation"]["id"]
    assert body["observation"]["world_state_id"] == body["scene_state"]["id"]
    assert body["scene_state"]["metrics"]["prediction_consistency"] >= 0.0
    assert body["entity_tracks"], "living lens should expose at least one entity track"

    world_state = client.get("/v1/world-state", params={"session_id": "proof"})
    assert world_state.status_code == 200
    world_body = world_state.json()
    assert world_body["current"]["id"] == body["scene_state"]["id"]
    assert world_body["entity_tracks"], "world-state endpoint should return persisted tracks"


def test_provider_health_includes_feature_statuses(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    response = client.get("/v1/providers/health")
    assert response.status_code == 200
    body = response.json()
    assert "providers" in body
    assert "features" in body
    feature_names = {item["name"] for item in body["features"]}
    assert {
        "live-lens-jepa",
        "energy-heatmap",
        "open-vocab-labels",
        "tvlc-connector",
        "proof-report-gemma",
    }.issubset(feature_names)


def test_living_lens_tick_prefers_open_vocab_summary(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    original_run_jepa_tick = runtime._run_jepa_tick

    async def _fake_run_jepa_tick(*args, **kwargs):
        result = await original_run_jepa_tick(*args, **kwargs)
        result.jepa_tick_dict["anchor_matches"] = [
            {
                "template_name": "person_torso",
                "confidence": 0.84,
                "patch_indices": [84, 85, 98, 99],
                "depth_stratum": "foreground",
                "open_vocab_label": "graphic print shirt",
            }
        ]
        result.jepa_tick_dict["setu_descriptions"] = [
            {
                "gate": {"passes": True},
                "description": {
                    "text": "person in foreground",
                    "confidence": 0.84,
                    "anchor_basis": "person_torso",
                    "depth_stratum": "foreground",
                    "is_uncertain": False,
                    "hallucination_risk": 0.16,
                    "uncertainty_map": None,
                },
            }
        ]
        return result

    async def _skip_background_enrichment(**_kwargs):
        return None

    monkeypatch.setattr(runtime, "_run_jepa_tick", _fake_run_jepa_tick)
    monkeypatch.setattr(runtime, "_finish_living_lens_enrichment", _skip_background_enrichment)

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "open-vocab-summary",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["summary"].startswith("graphic print shirt")

    stored = runtime.store.get_observation(body["observation"]["id"])
    assert stored is not None
    assert stored.summary.startswith("graphic print shirt")
    assert stored.metadata.get("summary_source") == "anchor_matches"


def test_living_lens_tick_returns_degraded_tick_when_worker_pool_fails(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    class BrokenPool:
        async def submit(self, _work_item):
            raise RuntimeError("worker unavailable")

        def recycle_session_worker(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(runtime, "_ensure_jepa_pool", lambda: BrokenPool())
    monkeypatch.setattr(
        runtime,
        "_inline_jepa_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inline fallback must not run")),
    )

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((80, 120, 180)),
            "session_id": "inline-fallback",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["jepa_tick"] is not None
    assert body["jepa_tick"]["degraded"] is True
    assert body["jepa_tick"]["degrade_stage"] == "submit"
    assert body["jepa_tick"]["last_tick_encoder_type"] == "unavailable"
    assert "worker unavailable" in body["jepa_tick"]["degrade_reason"]
    assert body["scene_state"]["observation_id"] == body["observation"]["id"]
    assert runtime._jepa_ticks_by_session["inline-fallback"]
    assert runtime._jepa_circuit_reason == "worker unavailable"

    monkeypatch.setattr(
        runtime,
        "_ensure_jepa_pool",
        lambda: (_ for _ in ()).throw(AssertionError("circuit-open path must skip worker pool")),
    )

    second = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((90, 90, 180)),
            "session_id": "inline-fallback",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["jepa_tick"]["degraded"] is True
    assert second_body["jepa_tick"]["degrade_stage"] == "disabled"
    assert second_body["jepa_tick"]["degrade_reason"] == "worker unavailable"


def test_living_lens_tick_returns_degraded_tick_when_worker_times_out(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    class SlowPool:
        def submit(self, _work_item):
            return None

        def recycle_session_worker(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(runtime, "_ensure_jepa_pool", lambda: SlowPool())
    monkeypatch.setattr(
        runtime,
        "_inline_jepa_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inline fallback must not run")),
    )
    monkeypatch.setattr(
        "cloud.runtime.service.asyncio.wait_for",
        lambda _awaitable, _timeout: (_ for _ in ()).throw(asyncio.TimeoutError("JEPA worker timed out")),
    )

    response = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "timeout-fallback",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["jepa_tick"] is not None
    assert body["jepa_tick"]["degraded"] is True
    assert body["jepa_tick"]["degrade_stage"] == "timeout"
    assert body["jepa_tick"]["last_tick_encoder_type"] == "unavailable"
    assert "timed out" in body["jepa_tick"]["degrade_reason"]


def test_proof_report_generation_with_enabled_mlx_does_not_require_ready_attr(tmp_path, monkeypatch):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    runtime = app.state.runtime

    settings = runtime.get_settings()
    settings.providers["mlx"].enabled = True
    runtime.update_settings(settings)

    monkeypatch.setattr(
        runtime.providers.mlx,
        "health",
        lambda config: ProviderHealth(name="mlx", role="reasoning", enabled=True, healthy=True, message="ready"),
    )

    from cloud.runtime.gemma4_bridge import Gemma4Bridge

    async def _fake_narrate(_self, summary_stats):
        return f"Ticks: {summary_stats.get('total_ticks', 0)}"

    monkeypatch.setattr(Gemma4Bridge, "narrate_proof_report", _fake_narrate)

    tick = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "proof-mlx",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert tick.status_code == 200

    generate = client.post(
        "/v1/proof-report/generate",
        json={"session_id": "proof-mlx", "chart_b64": None},
    )
    assert generate.status_code == 200
    body = generate.json()
    assert body["generated"] is True


def test_observation_share_endpoint_returns_grounded_copy_and_metrics(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    captured = client.post(
        "/v1/analyze",
        json={"image_base64": _encoded_png((220, 40, 40)), "session_id": "share-demo", "decode_mode": "off"},
    )
    assert captured.status_code == 200
    body = captured.json()

    runtime = app.state.runtime
    runtime.store.update_observation(
        body["observation"]["id"],
        summary="red mug on desk",
        providers=["basic"],
        metadata={"summary_source": "test"},
    )

    share = client.post(
        "/v1/share/observation",
        json={"session_id": "share-demo", "observation_id": body["observation"]["id"]},
    )
    assert share.status_code == 200
    share_body = share.json()
    assert share_body["observation_id"] == body["observation"]["id"]
    assert share_body["summary"] == "red mug on desk"
    assert "red mug on desk" in share_body["share_text"]
    assert share_body["share_url"] == "https://github.com/NeoOne601/Toori"

    recorded = client.post(
        "/v1/share/observation/event",
        json={
            "session_id": "share-demo",
            "observation_id": body["observation"]["id"],
            "event_type": "share_copied",
        },
    )
    assert recorded.status_code == 200
    assert recorded.json()["recorded"] is True

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'toori_share_events_total{event_type="share_clicked"} 1.0' in metrics.text
    assert 'toori_share_events_total{event_type="share_copied"} 1.0' in metrics.text


def test_challenge_evaluate_returns_baseline_comparison(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    for color in ((200, 30, 30), (180, 50, 40), (30, 30, 200), (200, 30, 30)):
        response = client.post(
            "/v1/living-lens/tick",
            json={
                "image_base64": _encoded_png(color),
                "session_id": "challenge",
                "decode_mode": "off",
                "proof_mode": "both",
            },
        )
        assert response.status_code == 200

    challenge = client.post(
        "/v1/challenges/evaluate",
        json={"session_id": "challenge", "challenge_set": "live", "proof_mode": "both", "limit": 4},
    )
    assert challenge.status_code == 200
    body = challenge.json()
    assert body["baseline_comparison"]["winner"] in {"jepa_hybrid", "frame_captioning", "embedding_retrieval"}
    assert body["baseline_comparison"]["summary"]
    assert body["success_criteria"]
