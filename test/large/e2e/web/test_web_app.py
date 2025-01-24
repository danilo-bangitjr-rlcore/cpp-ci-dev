import pytest
from fastapi.testclient import TestClient
import yaml

from corerl.web.app import app


@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)

def test_healthcheck(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OK"

def test_item_json(test_client: TestClient):
    sample_item = {"name": "Water", "description": "Bottle", "price": 1.59, "tax": 0.2}
    response = test_client.post("/items/", json=sample_item, headers={"Accept": "application/json"})

    print(f"\nStatus Code: {response.status_code},\nHeaders: {dict(response.headers)},\nText: {response.text}")  # Can also be response.content
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"

def test_item_yaml(test_client: TestClient):
    sample_item = {"name": "Water", "description": "Bottle", "price": 1.59, "tax": 0.2}
    response = test_client.post("/items/", json=sample_item, headers={"Accept": "application/yaml"})

    print(f"\nStatus Code: {response.status_code},\nHeaders: {dict(response.headers)},\nText: {response.text}")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/yaml"

def test_gen_config_file(test_client: TestClient):
    with open("config/opc_mountain_car_continuous.yaml", "r") as file:
        config = yaml.safe_load(file)

    response = test_client.post("/configuration/file", json=config)
    print(f"\nStatus Code: {response.status_code},\nHeaders: {dict(response.headers)},\nText: {response.text}")
    # assert response.status_code == 200

