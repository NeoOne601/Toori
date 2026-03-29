"""Tests for Setu-2 feedback loop."""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from cloud.runtime.app import create_app


def _png_b64(color: tuple[int, int, int] = (100, 150, 200)) -> str:
    image = Image.new("RGB", (32, 32), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_feedback_endpoint_registered(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    routes = {route.path for route in app.routes}
    assert "/v1/smriti/recall/feedback" in routes


def test_feedback_confirmed_returns_updated_true_when_media_found(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    analyze = client.post(
        "/v1/analyze",
        json={"image_base64": _png_b64(), "session_id": "fb_test", "decode_mode": "off"},
    )
    assert analyze.status_code == 200
    observation_id = analyze.json()["observation"]["id"]

    response = client.post(
        "/v1/smriti/recall/feedback",
        json={
            "query": "person in office",
            "media_id": observation_id,
            "confirmed": True,
            "session_id": "fb_test",
        },
    )
    assert response.status_code == 200
    assert response.json()["updated"] is True


def test_feedback_rejected_returns_result(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/recall/feedback",
        json={
            "query": "test query",
            "media_id": "nonexistent_obs",
            "confirmed": False,
            "session_id": "test",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "updated" in body
    assert "message" in body


def test_feedback_missing_media_id_returns_not_updated(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/recall/feedback",
        json={"query": "anything", "media_id": "obs_this_does_not_exist_xyz", "confirmed": True},
    )
    assert response.status_code == 200
    assert response.json()["updated"] is False


def test_feedback_requires_query_field(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/recall/feedback",
        json={"media_id": "obs_xyz", "confirmed": True},
    )
    assert response.status_code == 422


def test_feedback_result_has_w_mean_field(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/v1/smriti/recall/feedback",
        json={"query": "q", "media_id": "obs_fake", "confirmed": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert "w_mean" in body
    assert isinstance(body["w_mean"], (int, float))


def test_multiple_feedbacks_do_not_crash(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)
    for index in range(10):
        response = client.post(
            "/v1/smriti/recall/feedback",
            json={
                "query": f"query {index}",
                "media_id": f"obs_{index}",
                "confirmed": index % 2 == 0,
            },
        )
        assert response.status_code == 200
