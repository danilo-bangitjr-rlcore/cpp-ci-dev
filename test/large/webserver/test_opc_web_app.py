from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pytest import FixtureRequest

from corerl.web.app import OpcNodeResponse
from test.large.webserver.test_web_app import test_client  # noqa: F401
from test.medium.utils.fixture_opc import FakeOpcSyncServer, sync_server  # noqa: F401


@pytest.fixture
def skip_coverage_fixture(request: FixtureRequest):
    if request.config.getoption("--cov"):
        raise pytest.skip("Test skipped because coverage is emitted")


skip_test_if_coverage = pytest.mark.usefixtures("skip_coverage_fixture")

@skip_test_if_coverage
def test_read_opc(sync_server: FakeOpcSyncServer, test_client: TestClient): # noqa: F811

    encoded_url = quote(sync_server.url)
    response = test_client.get(f"/api/opc/nodes?opc_url={encoded_url}")
    data = response.json()
    OpcNodeResponse.model_validate(data)
    assert response.status_code == 200

@skip_test_if_coverage
def test_search_opc(sync_server: FakeOpcSyncServer, test_client: TestClient): # noqa: F811
    encoded_url = quote(sync_server.url)
    query = "vPLC1"
    response = test_client.get(f"/api/opc/nodes?opc_url={encoded_url}&query={query}")
    data = response.json()
    node_response = OpcNodeResponse.model_validate(data)
    assert response.status_code == 200
    assert len(node_response.nodes) == 2
