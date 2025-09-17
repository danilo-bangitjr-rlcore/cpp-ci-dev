from pathlib import Path

import pytest
import requests

from coredinator.test_utils import CoredinatorService


@pytest.mark.timeout(15)
def test_coreio_service_basic_lifecycle(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test basic CoreIO service start/stop/status lifecycle.

    Verifies that CoreIO services can be started, queried for status,
    and stopped cleanly through the HTTP API.
    """
    base_url = coredinator_service.base_url

    # Start CoreIO service
    response = requests.post(
        f"{base_url}/api/io/start",
        json={"config_path": str(config_file)},
    )
    assert response.status_code == 200
    data = response.json()
    service_id = data["service_id"]
    assert service_id == f"{config_file.stem}-coreio"
    assert data["status"]["state"] == "running"

    # Verify service status
    response = requests.get(f"{base_url}/api/io/{service_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["service_id"] == service_id
    assert status_data["status"]["state"] == "running"
    assert len(status_data["owners"]) == 1
    assert not status_data["is_shared"]

    # Verify service appears in listing
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 1
    assert list_data["coreio_services"][0]["service_id"] == service_id

    # Stop CoreIO service
    response = requests.post(f"{base_url}/api/io/{service_id}/stop")
    assert response.status_code == 200

    # Verify service is removed from listing
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 0


@pytest.mark.timeout(15)
def test_coreio_service_custom_id(
    coredinator_service: CoredinatorService,
    config_file: Path,
):
    """
    Test CoreIO service start with custom service ID.

    Verifies that custom coreio_id parameter works correctly
    and overrides the default naming convention.
    """
    base_url = coredinator_service.base_url
    custom_id = "custom-coreio-test"

    # Start CoreIO service with custom ID
    response = requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config_file),
            "coreio_id": custom_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["service_id"] == custom_id

    # Verify status endpoint uses custom ID
    response = requests.get(f"{base_url}/api/io/{custom_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["service_id"] == custom_id


@pytest.mark.timeout(20)
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
    config2.write_text("dummy: true\n")

    # Start first service with shared ID
    response1 = requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config_file),
            "coreio_id": shared_id,
        },
    )
    assert response1.status_code == 200

    # Start second service with same shared ID
    response2 = requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config2),
            "coreio_id": shared_id,
        },
    )
    assert response2.status_code == 200

    # Verify both services share the same CoreIO instance
    response = requests.get(f"{base_url}/api/io/{shared_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["service_id"] == shared_id
    assert status_data["is_shared"]

    # Verify only one CoreIO service appears in listing (shared instance)
    response = requests.get(f"{base_url}/api/io/")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["coreio_services"]) == 1
    assert list_data["coreio_services"][0]["service_id"] == shared_id
    assert list_data["coreio_services"][0]["is_shared"]


@pytest.mark.timeout(15)
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
    config2.write_text("dummy: true\n")

    requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config_file),
            "coreio_id": shared_id,
        },
    )
    requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config2),
            "coreio_id": shared_id,
        },
    )

    # Attempt to stop shared service should fail with 409 conflict
    response = requests.post(f"{base_url}/api/io/{shared_id}/stop")
    assert response.status_code == 409
    assert "still in use" in response.json()["detail"]

    # Verify service is still running
    response = requests.get(f"{base_url}/api/io/{shared_id}/status")
    assert response.status_code == 200
    assert response.json()["status"]["state"] == "running"


@pytest.mark.timeout(15)
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


@pytest.mark.timeout(15)
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
    config2.write_text("dummy: true\n")

    # Independent service
    requests.post(
        f"{base_url}/api/io/start",
        json={"config_path": str(config_file)},
    )

    # Shared services
    shared_id = "shared-test"
    requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config2),
            "coreio_id": shared_id,
        },
    )
    config3 = config_file.parent / "config3.yaml"
    config3.write_text("dummy: true\n")
    requests.post(
        f"{base_url}/api/io/start",
        json={
            "config_path": str(config3),
            "coreio_id": shared_id,
        },
    )

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
