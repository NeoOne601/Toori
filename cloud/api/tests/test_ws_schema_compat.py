import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


def _encoded_png(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (48, 48), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_events_schema_remains_additive(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    with client.websocket_connect("/v1/events") as websocket:
        response = client.post(
            "/v1/living-lens/tick",
            json={
                "image_base64": _encoded_png((90, 140, 210)),
                "session_id": "ws-proof",
                "decode_mode": "off",
                "proof_mode": "both",
            },
        )
        assert response.status_code == 200

        event_types: set[str] = set()
        jepa_payload = None
        for _ in range(16):
            message = websocket.receive_json()
            event_types.add(message["type"])
            if message["type"] == "jepa_tick":
                jepa_payload = message["payload"]["payload"]
                break

        assert "observation.created" in event_types
        assert "search.ready" in event_types
        assert "world_state.updated" in event_types
        assert "jepa.energy_map" in event_types
        assert "jepa_tick" in event_types
        assert jepa_payload is not None
        assert "energy_map" in jepa_payload
        assert "forecast_errors" in jepa_payload
        assert "configured_encoder" in jepa_payload
        assert "last_tick_encoder_type" in jepa_payload
        assert "degraded" in jepa_payload


def test_proof_report_endpoints_return_pdf(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    tick = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((120, 80, 40)),
            "session_id": "proof-report",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert tick.status_code == 200

    generate = client.post(
        "/v1/proof-report/generate",
        json={"session_id": "proof-report", "chart_b64": None},
    )
    assert generate.status_code == 200
    generate_body = generate.json()
    assert Path(generate_body["path"]).read_bytes().startswith(b"%PDF")

    latest = client.get("/v1/proof-report/latest")
    assert latest.status_code == 200
    assert latest.headers["content-type"] == "application/pdf"
    assert latest.content.startswith(b"%PDF")


def test_jepa_forecast_endpoint_returns_prediction(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    tick = client.post(
        "/v1/living-lens/tick",
        json={
            "image_base64": _encoded_png((40, 150, 90)),
            "session_id": "forecast",
            "decode_mode": "off",
            "proof_mode": "both",
        },
    )
    assert tick.status_code == 200

    forecast = client.post("/v1/jepa/forecast?k=5", json={"session_id": "forecast", "k": 1})
    assert forecast.status_code == 200
    body = forecast.json()
    assert body["session_id"] == "forecast"
    assert body["k"] == 5
    assert body["ready"] is True
    assert len(body["prediction"]) == 384
