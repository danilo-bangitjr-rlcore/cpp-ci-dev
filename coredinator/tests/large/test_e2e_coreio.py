from pathlib import Path

import pytest
import requests

from tests.utils.api_client import CoredinatorAPIClient
from tests.utils.factories import create_dummy_config
from tests.utils.service_fixtures import CoredinatorService


@pytest.mark.timeout(25)
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
    assert len(status_data["owners"]) == 1
    assert not status_data["is_shared"]

    # Verify service appears in listing
    response = requests.get(f"{api_client.base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 1
    assert list_data["coreio_services"][0]["service_id"] == service_id

    # Stop CoreIO service
    api_client.stop_coreio_service(service_id)

    # Verify service is removed from listing
    response = requests.get(f"{api_client.base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 0


@pytest.mark.timeout(25)
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

    # Start CoreIO service with custom ID
    service_id = api_client.start_coreio_service(str(config_file), coreio_id=custom_id)
    assert service_id == custom_id

    # Verify status endpoint uses custom ID
    status_data = api_client.get_coreio_status(custom_id)
    assert status_data["service_id"] == custom_id


@pytest.mark.timeout(30)
def test_coreio_service_sharing(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test CoreIO service sharing with multiple instances.

    Verifies that multiple services can share the same CoreIO instance
    and that service ownership tracking works correctly.
    """
    base_url = coredinator_service.base_url
    shared_id = "shared-coreio-test"

    # Create additional config files for testing
    config2 = config_file.parent / "config2.yaml"
    create_dummy_config(config2)

    api_client = CoredinatorAPIClient(base_url)

    # Start first service with shared ID
    api_client.start_coreio_service(str(config_file), shared_id)

    # Start second service with same shared ID
    api_client.start_coreio_service(str(config2), shared_id)

    # Verify both services share the same CoreIO instance
    status_data = api_client.get_coreio_status(shared_id)
    assert status_data["service_id"] == shared_id
    assert status_data["is_shared"]

    # Verify only one CoreIO service appears in listing (shared instance)
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 1
    assert list_data["coreio_services"][0]["service_id"] == shared_id
    assert list_data["coreio_services"][0]["is_shared"]


@pytest.mark.timeout(25)
def test_coreio_service_stop_shared_protection(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test that shared CoreIO services cannot be stopped while in use.

    Verifies the safety mechanism that prevents stopping CoreIO services
    that are still being used by other components.
    """
    base_url = coredinator_service.base_url
    shared_id = "protected-coreio-test"

    # Start two services with shared CoreIO
    config2 = config_file.parent / "config2.yaml"
    create_dummy_config(config2)

    api_client = CoredinatorAPIClient(base_url)

    api_client.start_coreio_service(str(config_file), shared_id)
    api_client.start_coreio_service(str(config2), shared_id)

    # Attempt to stop shared service should fail with 409 conflict
    response = requests.post(f"{base_url}/api/io/{shared_id}/stop")
    assert response.status_code == 409
    assert "still in use" in response.json()["detail"]

    # Verify service is still running
    status_data = api_client.get_coreio_status(shared_id)
    assert status_data["status"]["state"] == "running"


@pytest.mark.timeout(25)
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
    response = requests.get(f"{base_url}/api/io/nonexistent-service/status")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

    # Test stop for nonexistent service
    response = requests.post(f"{base_url}/api/io/nonexistent-service/stop")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.timeout(25)
def test_coreio_service_list_empty_and_multiple(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test CoreIO service listing with empty and populated states.

    Verifies that the listing endpoint correctly reports service
    states and metadata for various scenarios.
    """
    base_url = coredinator_service.base_url

    # Initially empty service list
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert list_data["coreio_services"] == []

    # Start multiple services with different configurations
    config2 = config_file.parent / "config2.yaml"
    create_dummy_config(config2)

    api_client = CoredinatorAPIClient(base_url)

    # Independent service
    api_client.start_coreio_service(str(config_file))

    # Shared services
    shared_id = "shared-test"
    api_client.start_coreio_service(str(config2), shared_id)

    config3 = config_file.parent / "config3.yaml"
    create_dummy_config(config3)
    api_client.start_coreio_service(str(config3), shared_id)

    # Verify listing shows correct services and metadata
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()

    services = list_data["coreio_services"]
    assert len(services) == 2  # One independent, one shared

    # Find independent and shared services
    independent_service = next((s for s in services if not s["is_shared"]), None)
    shared_service = next((s for s in services if s["is_shared"]), None)

    assert independent_service is not None
    assert independent_service["service_id"] == f"{config_file.stem}-coreio"
    assert not independent_service["is_shared"]

    assert shared_service is not None
    assert shared_service["service_id"] == shared_id
    assert shared_service["is_shared"]
