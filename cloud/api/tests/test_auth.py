from fastapi.testclient import TestClient

from cloud.runtime.app import create_app


def test_settings_roundtrip_with_api_key_auth(tmp_path):
    app = create_app(data_dir=str(tmp_path))
    client = TestClient(app)

    settings = client.get("/v1/settings").json()
    settings["auth_mode"] = "api-key"
    settings["providers"]["cloud"]["api_key"] = "secret-token"

    response = client.put("/v1/settings", json=settings)
    assert response.status_code == 200

    unauthorized = client.get("/v1/settings")
    assert unauthorized.status_code == 401

    authorized = client.get("/v1/settings", headers={"X-API-Key": "secret-token"})
    assert authorized.status_code == 200
    assert authorized.json()["auth_mode"] == "api-key"
