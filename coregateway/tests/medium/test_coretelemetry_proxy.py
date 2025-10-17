from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from coregateway.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx.AsyncClient for network isolation."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"result": "ok"}
    mock_response.text = '{"result": "ok"}'
    mock_client.request.return_value = mock_response
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    mock_client.aclose.return_value = None
    return mock_client


@pytest.fixture
def stub_async_http_transport() -> Iterator[MagicMock]:
    """Stub httpx.AsyncHTTPTransport to avoid real network setup."""
    with patch("httpx.AsyncHTTPTransport", MagicMock()) as stub:
        yield stub


@pytest.fixture
def test_client(
    mock_httpx_client: AsyncMock,
    stub_async_http_transport: MagicMock,
) -> Iterator[TestClient]:
    """Test client with mocked HTTP requests."""
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        app = create_app(port=8001)
        app.state.httpx_client = mock_httpx_client
        with TestClient(app) as client:
            yield client


class TestCoretelemetryProxy:
    """Test proxy functionality to coretelemetry service."""

    def test_proxy_get_request_success(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test GET request is properly proxied to coretelemetry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"config_path": "../config"}
        mock_response.text = '{"config_path": "../config"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coretelemetry/api/config/path")

        # Verify request was proxied correctly
        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "http://localhost:7001/api/config/path"

        assert response.status_code == 200
        assert response.json() == {"config_path": "../config"}

    def test_proxy_get_with_query_params(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test GET request with query parameters is properly proxied."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = [{"timestamp": "2024-01-01", "value": 1.0}]
        mock_response.text = '[{"timestamp": "2024-01-01", "value": 1.0}]'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get(
            "/api/v1/coretelemetry/api/data/test_agent?metric=reward",
        )

        # Verify request was proxied correctly
        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "http://localhost:7001/api/data/test_agent"
        assert call_args[1]["params"] == {"metric": "reward"}

        assert response.status_code == 200

    def test_proxy_post_request_with_body(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test POST request with query parameters is properly proxied."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "Config path updated"}
        mock_response.text = '{"message": "Config path updated"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.post(
            "/api/v1/coretelemetry/api/config/path?path=../config",
        )

        # Verify request was proxied correctly
        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:7001/api/config/path"
        # Query parameters should be passed through
        assert call_args[1]["params"] == {"path": "../config"}

        assert response.status_code == 200

    def test_proxy_headers_cleaned(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test hop-by-hop headers are stripped from responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "x-custom-header": "telemetry",
            "content-length": "20",
        }
        mock_response.json.return_value = {"status": "ok"}
        mock_response.text = '{"status": "ok"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coretelemetry/health")

        # Hop-by-hop headers should be stripped
        assert "connection" not in response.headers
        assert "transfer-encoding" not in response.headers
        # Custom headers should be preserved
        assert response.headers.get("x-custom-header") == "telemetry"

    @pytest.mark.parametrize(
        "exception,expected_status",
        [
            (httpx.TimeoutException("Request timeout"), 504),
            (httpx.NetworkError("Network error"), 502),
            (httpx.HTTPError("HTTP error"), 502),
        ],
    )
    def test_proxy_error_handling(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
        exception: Exception,
        expected_status: int,
    ) -> None:
        """Test proxy returns correct error codes for different upstream exceptions."""
        mock_httpx_client.request.side_effect = exception

        response = test_client.get("/api/v1/coretelemetry/api/config/path")

        assert response.status_code == expected_status
        error_detail = response.json()["detail"]
        assert "Gateway" in error_detail or "gateway" in error_detail

    def test_proxy_upstream_http_error_status(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test proxy forwards upstream HTTP error status codes."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"detail": "Data not found"}
        mock_response.text = '{"detail": "Data not found"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coretelemetry/api/data/nonexistent")

        assert response.status_code == 404
        assert response.json() == {"detail": "Data not found"}
