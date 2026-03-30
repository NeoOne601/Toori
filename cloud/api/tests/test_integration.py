import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app
from cloud.runtime.models import Answer, BoundingBox, ProviderHealth


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

    thumbnail_path = second_body["observation"]["thumbnail_path"]
    file_response = client.get("/v1/file", params={"path": thumbnail_path})
    assert file_response.status_code == 200
    assert file_response.content

    external = tmp_path.parent / "outside-runtime.txt"
    external.write_text("outside")
    forbidden = client.get("/v1/file", params={"path": str(external)})
    assert forbidden.status_code == 403


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
