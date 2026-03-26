import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from ..app import create_perception_app


def test_embed_returns_real_embedding(tmp_path):
    app = create_perception_app(data_dir=str(tmp_path))
    client = TestClient(app)

    image = Image.new("RGB", (24, 24), color=(10, 120, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {"image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8")}

    response = client.post("/embed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] in {"dinov2", "onnx", "basic"}
    assert len(data["embedding"]) == 128
    assert any(abs(value) > 0 for value in data["embedding"])
