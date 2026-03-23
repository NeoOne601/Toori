import pytest
from fastapi.testclient import TestClient
from cloud.api.main import app

client = TestClient(app)

def test_protected_route_without_token():
    response = client.get("/protected")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Authorization header"
