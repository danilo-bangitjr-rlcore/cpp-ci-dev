from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient


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
        mock_response.content = b'{"config_path": "../config"}'
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
        mock_response.content = b'[{"timestamp": "2024-01-01", "value": 1.0}]'
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
        mock_response.content = b'{"message": "Config path updated"}'
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
        mock_response.content = b'{"status": "ok"}'
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
        mock_response.content = b'{"detail": "Data not found"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coretelemetry/api/data/nonexistent")

        assert response.status_code == 404
        assert response.json() == {"detail": "Data not found"}
