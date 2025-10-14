"""End-to-end API tests for system metrics endpoints."""

import pytest
from coretelemetry.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient for system metrics endpoints."""
    return TestClient(create_app(config_path="clean/"))


class TestSystemMetricsEndpoints:
    """E2E tests for system metrics API endpoints."""

    @pytest.mark.timeout(10)
    def test_get_platform_returns_200(self, test_client: TestClient):
        """Test platform endpoint returns valid platform."""
        response = test_client.get("/api/v1/coretelemetry/system/platform")

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"] in ["Linux", "Windows", "Darwin"]

    @pytest.mark.timeout(10)
    def test_get_cpu_returns_valid_percent(self, test_client: TestClient):
        """Test CPU endpoint returns valid percentage."""
        response = test_client.get("/api/v1/coretelemetry/system/cpu")

        assert response.status_code == 200
        data = response.json()
        assert "percent" in data
        assert 0 <= data["percent"] <= 100

    @pytest.mark.timeout(10)
    def test_get_cpu_per_core_returns_list(self, test_client: TestClient):
        """Test per-core CPU endpoint returns list."""
        response = test_client.get("/api/v1/coretelemetry/system/cpu_per_core")

        assert response.status_code == 200
        data = response.json()
        assert "percent" in data
        assert isinstance(data["percent"], list)
        assert len(data["percent"]) > 0  # At least 1 core
        assert all(0 <= p <= 100 for p in data["percent"])

    @pytest.mark.timeout(10)
    def test_get_ram_returns_valid_stats(self, test_client: TestClient):
        """Test RAM endpoint returns valid statistics."""
        response = test_client.get("/api/v1/coretelemetry/system/ram")

        assert response.status_code == 200
        data = response.json()
        assert "percent" in data
        assert "used_gb" in data
        assert "total_gb" in data
        assert 0 <= data["percent"] <= 100
        assert data["used_gb"] <= data["total_gb"]
        assert data["total_gb"] > 0

    @pytest.mark.timeout(10)
    def test_get_disk_returns_valid_stats(self, test_client: TestClient):
        """Test disk endpoint returns valid statistics."""
        response = test_client.get("/api/v1/coretelemetry/system/disk")

        assert response.status_code == 200
        data = response.json()
        assert "percent" in data
        assert "used_gb" in data
        assert "total_gb" in data
        assert 0 <= data["percent"] <= 100
        assert data["used_gb"] <= data["total_gb"]
        assert data["total_gb"] > 0
