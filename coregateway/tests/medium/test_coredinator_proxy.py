from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient


class TestCoredinatorProxy:
    """Test proxy functionality to coredinator service."""

    def test_proxy_get_request_success(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test GET request is properly proxied to coredinator."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"agents": []}
        mock_response.content = b'{"agents": []}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coredinator/agents")

        # Verify request was proxied correctly
        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "http://localhost:7000/agents"

        assert response.status_code == 200
        assert response.json() == {"agents": []}

    def test_proxy_post_request_with_body(
        self,
        test_client: TestClient,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test POST request with JSON body is properly proxied."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"agent_id": "test-agent"}
        mock_response.content = b'{"agent_id": "test-agent"}'
        mock_httpx_client.request.return_value = mock_response

        payload = {"config_path": "/tmp/test.yaml", "coreio_id": None}
        response = test_client.post("/api/v1/coredinator/agents/start", json=payload)

        # Verify request was proxied correctly
        mock_httpx_client.request.assert_called_once()
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:7000/agents/start"
        # Body is now forwarded as raw bytes via content parameter
        # Decode and parse to compare JSON content (formatting may differ)
        import json
        actual_body = json.loads(call_args[1]["content"].decode())
        assert actual_body == payload

        assert response.status_code == 201
        assert response.json() == {"agent_id": "test-agent"}

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
            "custom-header": "value",
            "content-length": "15",
        }
        mock_response.json.return_value = {"result": "ok"}
        mock_response.content = b'{"result": "ok"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coredinator/status")

        # Hop-by-hop headers should be stripped
        assert "connection" not in response.headers
        assert "transfer-encoding" not in response.headers
        # Custom headers should be preserved
        assert response.headers.get("custom-header") == "value"

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

        response = test_client.get("/api/v1/coredinator/agents")

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
        mock_response.json.return_value = {"detail": "Agent not found"}
        mock_response.content = b'{"detail": "Agent not found"}'
        mock_httpx_client.request.return_value = mock_response

        response = test_client.get("/api/v1/coredinator/agents/nonexistent")

        assert response.status_code == 404
        assert response.json() == {"detail": "Agent not found"}
