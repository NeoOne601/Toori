import pytest
from fastapi.testclient import TestClient
from cloud.api.main import app
import base64

client = TestClient(app, raise_server_exceptions=False)

def test_audio_query_route_exists():
    response = client.post("/v1/audio/query", json={})
    assert response.status_code == 422, "Route should exist and enforce Pydantic validation"

def test_audio_query_pydantic_validation():
    payload = {
        "audio_base64": "dummy",
        "sample_rate": -1,
        "top_k": 10
    }
    response = client.post("/v1/audio/query", json=payload)
    assert response.status_code == 422
    assert "sample_rate" in response.text
    assert "greater than or equal to 8000" in response.text or "Input should be" in response.text

def test_audio_query_handles_invalid_base64():
    client_exceptions = TestClient(app, raise_server_exceptions=True)
    payload = {
        "audio_base64": "not-base-64!!!",
        "sample_rate": 16000,
        "top_k": 5
    }
    with pytest.raises(ValueError) as exc_info:
        client_exceptions.post("/v1/audio/query", json=payload)
    assert "Invalid audio_base64" in str(exc_info.value)

def test_audio_query_happy_path(monkeypatch):
    class MockDB:
        def audio_faiss_search(self, emb, top_k):
            return [
                {"media_id": "media-1", "audio_score": 0.95},
                {"media_id": "media-2", "audio_score": 0.85}
            ]
        def get_smriti_media(self, media_id):
            class MockMedia:
                def __init__(self, id, thumbnail_path, audio_energy, audio_duration_seconds):
                    self.id = id
                    self.thumbnail_path = thumbnail_path
                    self.audio_energy = audio_energy
                    self.audio_duration_seconds = audio_duration_seconds
                    self.setu_descriptions = []
            if media_id == "media-1":
                return MockMedia("media-1", "/thumb1.jpg", 0.5, 2.0)
            return MockMedia("media-2", "/thumb2.jpg", 0.4, 1.5)
        _audio_media_ids = ["media-1", "media-2"]

    monkeypatch.setattr(app.state.runtime, "smriti_db", MockDB())

    class MockAE:
        def __init__(self, *args, **kwargs):
            pass
        def encode_bytes(self, _bytes, sample_rate):
            return [0.1] * 384
            
    monkeypatch.setattr("cloud.perception.audio_encoder.AudioEncoder", MockAE)

    dummy_wav = b"RIFF$" + b"\x00"*36 # Dummy bytes
    payload = {
        "audio_base64": base64.b64encode(dummy_wav).decode("utf-8"),
        "sample_rate": 16000,
        "top_k": 5
    }
    
    response = client.post("/v1/audio/query", json=payload)
    assert response.status_code == 200, f"Happy path failed: {response.text}"
    
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["media_id"] == "media-1"
    assert data["results"][0]["audio_score"] == 0.95
    assert data["index_size"] == 2
    assert data["encoder"] == "audio_jepa_phase1"
