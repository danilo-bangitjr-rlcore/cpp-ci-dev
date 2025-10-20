"""End-to-end API tests with FastAPI TestClient and real database."""

from pathlib import Path

import pytest
from coretelemetry.app import create_app, get_agent_metrics_manager
from coretelemetry.utils.sql import DBConfig
from fastapi.testclient import TestClient

pytest_plugins = [
    "test.infrastructure.networking",
    "test.infrastructure.utils.tsdb",
]


@pytest.fixture
def test_client(db_config_from_engine: DBConfig, sample_config_dir: Path) -> TestClient:
    """FastAPI TestClient with real database configuration."""
    # Configure manager before creating client
    agent_metrics_manager = get_agent_metrics_manager()
    agent_metrics_manager.set_db_config(db_config_from_engine)
    agent_metrics_manager.set_config_path(sample_config_dir)
    agent_metrics_manager.clear_cache()  # Clear any stale state

    return TestClient(create_app(config_path=sample_config_dir))


class TestBasicEndpoints:
    """Tests for basic API endpoints."""

    @pytest.mark.timeout(10)
    def test_root_redirects_to_docs(self, test_client: TestClient):
        """Test root endpoint redirects to /docs."""
        response = test_client.get("/", follow_redirects=False)

        assert response.status_code in (307, 308)  # Redirect status codes
        assert "/docs" in response.headers["location"]

    @pytest.mark.timeout(10)
    def test_health_check(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestGetTelemetryEndpoint:
    """Tests for GET /api/data/{agent_id} endpoint."""

    @pytest.mark.timeout(15)
    def test_get_telemetry_latest_value(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test getting latest telemetry value without time params."""
        response = test_client.get("/api/data/test_agent?metric=temperature")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "timestamp" in data[0]
        assert "value" in data[0]
        assert data[0]["value"] == 27.5

    @pytest.mark.timeout(15)
    def test_get_telemetry_with_time_range(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test getting telemetry data with time range."""
        response = test_client.get(
            "/api/data/test_agent"
            "?metric=temperature"
            "&start_time=2024-01-01 10:00:00%2B00"
            "&end_time=2024-01-01 11:30:00%2B00",
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @pytest.mark.timeout(15)
    def test_get_telemetry_time_reserved_word(self, test_client: TestClient):
        """Test 'time' as metric returns 400 error."""
        response = test_client.get("/api/data/test_agent?metric=time")

        assert response.status_code == 400

    @pytest.mark.timeout(15)
    def test_get_telemetry_missing_agent(self, test_client: TestClient):
        """Test 500 error when agent config doesn't exist."""
        response = test_client.get("/api/data/nonexistent_agent?metric=temperature")

        assert response.status_code == 500

    @pytest.mark.timeout(15)
    def test_get_telemetry_table_not_found(self, test_client: TestClient, sample_config_dir: Path):
        """Test 404 error when table doesn't exist."""
        import yaml

        # Create config with nonexistent table
        bad_config = {"metrics": {"table_name": "nonexistent_table_xyz"}}
        config_file = sample_config_dir / "bad_table_agent.yaml"
        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        response = test_client.get("/api/data/bad_table_agent?metric=temperature")

        assert response.status_code == 404

    @pytest.mark.timeout(15)
    def test_get_telemetry_column_not_found(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test 404 error when column doesn't exist."""
        response = test_client.get("/api/data/test_agent?metric=nonexistent_column")

        assert response.status_code == 404

    @pytest.mark.timeout(15)
    def test_get_telemetry_no_data(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test 404 error when no data found for time range."""
        response = test_client.get(
            "/api/data/test_agent?metric=temperature&start_time=2025-01-01&end_time=2025-01-02",
        )

        assert response.status_code == 404


class TestGetAvailableMetricsEndpoint:
    """Tests for GET /api/data/{agent_id}/metrics endpoint."""

    @pytest.mark.timeout(15)
    def test_get_available_metrics(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test getting available metrics for an agent."""
        response = test_client.get("/api/data/test_agent/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test_agent"
        assert "time" not in data["data"]
        assert "temperature" in data["data"]
        assert "pressure" in data["data"]
        assert len(data["data"]) == 2

    @pytest.mark.timeout(15)
    def test_get_available_metrics_excludes_time(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test that 'time' column is excluded from metrics."""
        response = test_client.get("/api/data/test_agent/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "time" not in data["data"]


class TestConfigEndpoints:
    """Tests for configuration management endpoints."""

    @pytest.mark.timeout(10)
    def test_get_db_config(self, test_client: TestClient):
        """Test GET /api/config/db endpoint."""
        response = test_client.get("/api/config/db")

        assert response.status_code == 200
        data = response.json()
        assert "username" in data
        assert "ip" in data
        assert "port" in data
        assert "db_name" in data

    @pytest.mark.timeout(10)
    def test_set_db_config(self, test_client: TestClient):
        """Test POST /api/config/db endpoint."""
        new_config = {
            "drivername": "postgresql+psycopg2",
            "username": "new_user",
            "password": "new_pass",
            "ip": "192.168.1.1",
            "port": 5433,
            "db_name": "new_db",
            "schema": "public",
        }

        response = test_client.post("/api/config/db", json=new_config)

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["config"]["username"] == "new_user"
        assert data["config"]["port"] == 5433

    @pytest.mark.timeout(10)
    def test_get_config_path(self, test_client: TestClient):
        """Test GET /api/config/path endpoint."""
        response = test_client.get("/api/config/path")

        assert response.status_code == 200
        data = response.json()
        assert "config_path" in data
        assert isinstance(data["config_path"], str)

    @pytest.mark.timeout(10)
    def test_set_config_path(self, test_client: TestClient):
        """Test POST /api/config/path endpoint."""
        response = test_client.post("/api/config/path?path=/tmp/new_configs")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["config_path"] == "/tmp/new_configs"

    @pytest.mark.timeout(10)
    def test_clear_cache(self, test_client: TestClient, sample_metrics_table: tuple[str, str]):
        """Test POST /api/config/clear_cache endpoint."""
        # First, populate cache by making a request
        test_client.get("/api/data/test_agent?metric=temperature")

        # Clear the cache
        response = test_client.post("/api/config/clear_cache")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Cache cleared successfully"


class TestCORSHeaders:
    """Tests for CORS middleware configuration."""

    @pytest.mark.timeout(10)
    def test_cors_headers(self, test_client: TestClient):
        """Test CORS headers are properly configured."""
        headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        }

        response = test_client.options("/api/config/db", headers=headers)

        # Check that CORS middleware is configured
        assert response.status_code in (200, 204)
        # Note: TestClient may not fully simulate CORS preflight, but we verify no errors
