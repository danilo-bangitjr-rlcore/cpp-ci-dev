import pytest
import yaml
from fastapi.testclient import TestClient

from corerl.config import MainConfig
from corerl.web.app import app


@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)

def test_healthcheck(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OK"

def test_json_config_file(test_client: TestClient):
    with open("config/mountain_car_continuous.yaml", "r") as file:
        config = yaml.safe_load(file)

    response = test_client.post("/configuration/file", json=config, headers={"Accept": "application/json"})
    content = response.json()

    res_config = MainConfig(**content)
    assert response.status_code == 200
    assert isinstance(res_config, MainConfig)
    assert response.headers["Content-Type"] == "application/json"

def test_yaml_config_file(test_client: TestClient):
    with open("config/mountain_car_continuous.yaml", "r") as file:
        config = yaml.safe_load(file)

    response = test_client.post("/configuration/file", json=config, headers={"Accept": "application/yaml"})
    content = yaml.safe_load(response.text)

    res_config = MainConfig(**content)

    assert response.status_code == 200
    assert isinstance(res_config, MainConfig)
    assert response.headers["Content-Type"] == "application/yaml"
