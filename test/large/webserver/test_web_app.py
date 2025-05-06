import datetime
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from corerl.config import MainConfig
from corerl.web import get_coreio_sqlite_path
from corerl.web.app import app


@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)

def test_healthcheck(test_client: TestClient):
    response = test_client.get("/api/corerl/health")
    assert response.status_code == 200
    payload = response.json()

    coreio_db = Path(get_coreio_sqlite_path())
    expected_status = "OK" if coreio_db.is_file() else "ERROR"

    assert payload["status"] == expected_status

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
    assert isinstance(res_config.interaction.obs_period, datetime.timedelta)
    assert isinstance(res_config, MainConfig)
