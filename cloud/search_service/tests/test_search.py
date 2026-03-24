import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app
from cloud.search_service.main import create_search_app


def _png(color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (16, 16), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_search_service_returns_real_hits(tmp_path):
    runtime_app = create_app(data_dir=str(tmp_path))
    runtime_client = TestClient(runtime_app)
    first = runtime_client.post("/v1/analyze", json={"image_base64": _png((0, 0, 255)), "session_id": "s", "decode_mode": "off"}).json()
    runtime_client.post("/v1/analyze", json={"image_base64": _png((0, 10, 245)), "session_id": "s", "decode_mode": "off"})

    app = create_search_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post("/search", json={"observation_id": first["observation"]["id"], "k": 5})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits
    assert hits[0]["observation_id"] != first["observation"]["id"]
