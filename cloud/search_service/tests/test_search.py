"""Tests for the FAISS search service skeleton.

The test ensures that when the service is started with an empty index it
returns an empty list of results for any query.
"""

import pytest
from fastapi.testclient import TestClient

from cloud.search_service.main import app

client = TestClient(app)

@pytest.fixture
def empty_index(monkeypatch):
    # Ensure index is None (should already be default)
    from cloud.search_service import main
    monkeypatch.setattr(main, "index", None)
    return None

def test_search_empty_index(empty_index):
    response = client.post("/search", json={"query": "test", "k": 5})
    assert response.status_code == 200
    assert response.json() == []
