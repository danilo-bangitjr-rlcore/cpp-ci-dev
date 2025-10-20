from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient


class TestBasicEndpoints:
    """Test basic FastAPI application endpoints."""

    def test_root_redirects_to_docs(self, test_client: TestClient) -> None:
        """Test root endpoint redirects to /docs."""
        response = test_client.get("/", follow_redirects=False)
        assert response.status_code in (307, 308)
        assert response.headers.get("location") == "/docs"

    def test_version_header_present(self, test_client: TestClient) -> None:
        """Test X-CoreRL-Version header is present in all responses."""
        response = test_client.get("/health")
        assert "X-CoreRL-Version" in response.headers
        assert response.headers["X-CoreRL-Version"] == "0.0.1"


class TestHealthEndpoint:
    """Test health check endpoint with multiple services."""

    def test_health_check_all_healthy(
        self,
        test_client: TestClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health endpoint when all services are healthy."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_httpx_client.get.return_value = mock_response

        response = test_client.get("/health")
        data = response.json()

        # Verify both services were checked
        assert mock_httpx_client.get.call_count == 2
        calls = [call[0][0] for call in mock_httpx_client.get.call_args_list]
        assert "http://localhost:7000/api/healthcheck" in calls
        assert "http://localhost:7001/health" in calls

        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert data["services"]["coredinator"] == "healthy"
        assert data["services"]["coretelemetry"] == "healthy"

    def test_health_check_coredinator_unhealthy(
        self,
        test_client: TestClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health endpoint when coredinator is unhealthy."""

        def get_side_effect(url: str, **kwargs: dict) -> AsyncMock:
            mock_response = AsyncMock()
            if "7000" in url:
                mock_response.status_code = 503
            else:
                mock_response.status_code = 200
            return mock_response

        mock_httpx_client.get.side_effect = get_side_effect

        response = test_client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["services"]["coredinator"] == "unhealthy"
        assert data["services"]["coretelemetry"] == "healthy"

    def test_health_check_coretelemetry_unhealthy(
        self,
        test_client: TestClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health endpoint when coretelemetry is unhealthy."""

        def get_side_effect(url: str, **kwargs: dict) -> AsyncMock:
            mock_response = AsyncMock()
            if "7001" in url:
                mock_response.status_code = 503
            else:
                mock_response.status_code = 200
            return mock_response

        mock_httpx_client.get.side_effect = get_side_effect

        response = test_client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["services"]["coredinator"] == "healthy"
        assert data["services"]["coretelemetry"] == "unhealthy"

    def test_health_check_all_unhealthy(
        self,
        test_client: TestClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health endpoint when all services are unhealthy."""
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_httpx_client.get.return_value = mock_response

        response = test_client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["services"]["coredinator"] == "unhealthy"
        assert data["services"]["coretelemetry"] == "unhealthy"

    def test_health_check_network_error(
        self,
        test_client: TestClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Test health endpoint handles network errors gracefully."""
        mock_httpx_client.get.side_effect = httpx.NetworkError("Connection failed")

        response = test_client.get("/health")
        data = response.json()

        assert response.status_code == 503
        assert data["status"] == "degraded"
        assert data["services"]["coredinator"] == "unhealthy"
        assert data["services"]["coretelemetry"] == "unhealthy"

