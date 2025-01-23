import pytest
from fastapi.testclient import TestClient

from corerl.web.app import app


@pytest.fixture(scope="module")
def test_client():
    yield TestClient(app)

def test_healthcheck(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OK"
