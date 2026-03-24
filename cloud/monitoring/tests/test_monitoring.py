import pytest
from fastapi.testclient import TestClient
from cloud.jepa_service.app import app as jepa_app
from cloud.search_service.main import app as search_app

@pytest.mark.parametrize("service,client", [
    ("jepa", TestClient(jepa_app)),
    ("search", TestClient(search_app))
])
def test_metrics_endpoint(service, client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "process_cpu_seconds_total" in response.text
