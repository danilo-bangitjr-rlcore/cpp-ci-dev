from pathlib import Path

import pytest
import requests

from tests.utils.api_client import CoredinatorAPIClient
from tests.utils.factories import create_dummy_config
from tests.utils.service_fixtures import CoredinatorService
from tests.utils.timeout_multiplier import apply_timeout_multiplier

# Platform-adjusted timeout value for test decorators
TIMEOUT = int(apply_timeout_multiplier(30))


@pytest.mark.timeout(TIMEOUT)
def test_coreio_service_basic_lifecycle(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test basic CoreIO service start/stop/status lifecycle.

    Verifies that CoreIO services can be started, queried for status,
    and stopped cleanly through the HTTP API.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)

    # Start CoreIO service
    service_id = api_client.start_coreio_service(str(config_file))
    assert service_id == f"{config_file.stem}-coreio"

    # Verify service status
    status_data = api_client.get_coreio_status(service_id)
    assert status_data["service_id"] == service_id
    assert status_data["status"]["state"] == "running"

    # Verify service appears in listing
    response = requests.get(f"{api_client.base_url}/api/io/", timeout=10.0)
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 1
    assert list_data["coreio_services"][0]["service_id"] == service_id

    # Stop CoreIO service
    api_client.stop_coreio_service(service_id)

    # Verify service is removed from listing
    response = requests.get(f"{api_client.base_url}/api/io/", timeout=10.0)
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 0


@pytest.mark.timeout(TIMEOUT)
def test_coreio_service_custom_id(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test CoreIO service start with custom service ID.

    Verifies that custom coreio_id parameter works correctly
    and overrides the default naming convention.
    """
    api_client = CoredinatorAPIClient(coredinator_service.base_url)
    custom_id = "custom-coreio-test"

    service_id = api_client.start_coreio_service(str(config_file), coreio_id=custom_id)
    assert service_id == custom_id

    # Verify status endpoint uses custom ID
    status_data = api_client.get_coreio_status(custom_id)
    assert status_data["service_id"] == custom_id


@pytest.mark.timeout(TIMEOUT)
def test_coreio_service_error_scenarios(
    coredinator_service: CoredinatorService,
    tmp_path: Path,
):
    """
    Test error handling for invalid requests and missing resources.

    Verifies proper HTTP status codes and error messages for
    various failure scenarios.
    """
    base_url = coredinator_service.base_url

    # Test invalid config file path
    invalid_config = tmp_path / "nonexistent.yaml"
    response = requests.post(
        f"{base_url}/api/io/start",
        json={"config_path": str(invalid_config)},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

    # Test status for nonexistent service
    response = requests.get(f"{base_url}/api/io/nonexistent-service/status", timeout=10.0)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

    # Test stop for nonexistent service
    response = requests.post(f"{base_url}/api/io/nonexistent-service/stop")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.timeout(TIMEOUT)
def test_coreio_service_list_empty_and_multiple(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test CoreIO service listing with empty and populated states.

    Verifies that the listing endpoint correctly reports service states.
    """
    base_url = coredinator_service.base_url

    # Initially empty service list
    response = requests.get(f"{base_url}/api/io/", timeout=10.0)
    assert response.status_code == 200
    list_data = response.json()
    assert list_data["coreio_services"] == []

    # Start multiple services with different configurations
    config2 = config_file.parent / "config2.yaml"
    create_dummy_config(config2)

    api_client = CoredinatorAPIClient(base_url)

    # Start first service
    service_id1 = api_client.start_coreio_service(str(config_file))

    # Start second service with custom ID
    service_id2 = api_client.start_coreio_service(str(config2), "custom-id")

    # Verify listing shows correct services and metadata
    response = requests.get(f"{base_url}/api/io/", timeout=10.0)
    assert response.status_code == 200
    list_data = response.json()

    services = list_data["coreio_services"]
    assert len(services) == 2

    service_ids = {s["service_id"] for s in services}
    assert service_id1 in service_ids
    assert service_id2 in service_ids
