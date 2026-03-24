import pytest
from fastapi.testclient import TestClient

from cloud.jepa_service.app import create_perception_app
from cloud.search_service.main import create_search_app


@pytest.mark.parametrize(
    "client",
    [
        TestClient(create_perception_app(data_dir=".toori/test-metrics-perception")),
        TestClient(create_search_app(data_dir=".toori/test-metrics-search")),
    ],
)
def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "process_cpu_seconds_total" in response.text
