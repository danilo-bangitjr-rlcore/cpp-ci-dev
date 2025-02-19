import datetime
import json

import pytest
import yaml
from fastapi.testclient import TestClient

from corerl.config import MainConfig
from corerl.web.app import app


@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)

def test_healthcheck(test_client: TestClient):
    response = test_client.get("/api/corerl/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OK"

@pytest.mark.parametrize("req_type", ["application/json", "application/yaml"])
@pytest.mark.parametrize("res_type", ["application/json", "application/yaml"])
def test_config_file(req_type: str, res_type: str, test_client: TestClient):

    with open("config/mountain_car_continuous.yaml", "r") as file:
        config = yaml.safe_load(file)

    if req_type == "application/json":
        config_data = json.dumps(config)
    elif req_type == "application/yaml":
        config_data = yaml.safe_dump(config)
    else:
        raise NotImplementedError

    response = test_client.post(
        "/api/corerl/configuration/file",
        content=config_data,
        headers={
            "Content-Type": req_type,
            "Accept": res_type,
        }
    )

    if res_type == "application/json":
        content = response.json()
        assert response.headers["Content-Type"] == "application/json"
    elif res_type == "application/yaml":
        content = yaml.safe_load(response.text)
        assert response.headers["Content-Type"] == "application/yaml"
    else:
        raise NotImplementedError

    res_config = MainConfig(**content)
    assert response.status_code == 200
    assert isinstance(res_config.env.obs_period, datetime.timedelta)
    assert isinstance(res_config, MainConfig)
