import json
from fastapi.testclient import TestClient
from cloud.api.main import app, call_jepa, call_search, get_current_user


def test_search_endpoint(monkeypatch):
    # Prepare deterministic data
    dummy_embedding = [0.0] * 128
    refined_embedding = [1.0] * 128
    search_results = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]

    # Monkey‑patch the helper functions
    def mock_call_jepa(embedding):
        assert embedding == dummy_embedding
        return refined_embedding

    def mock_call_search(refined):
        assert refined == refined_embedding
        return search_results

    monkeypatch.setattr('cloud.api.main.call_jepa', mock_call_jepa)
    monkeypatch.setattr('cloud.api.main.call_search', mock_call_search)

    monkeypatch.setattr('cloud.api.auth.get_current_user', lambda: {"sub": "test_user"})
    client = TestClient(app)
    # Override auth dependency for testing
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test_user"}
    response = client.post("/search", json=dummy_embedding)
    assert response.status_code == 200
    data = response.json()
    assert data["refined"] == refined_embedding
    assert data["results"] == search_results
