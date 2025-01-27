import pytest
import json
import time
from fastapi.testclient import TestClient
import yaml
import tempfile

from corerl.web.app import app
from corerl.config import MainConfig
from corerl.configs.loader import direct_load_config, config_to_dict


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
    # print(f"{response.status_code=},\n{response.headers=},\n{response.text=}")
    content = response.json()
    # print(f"{content=}")

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
