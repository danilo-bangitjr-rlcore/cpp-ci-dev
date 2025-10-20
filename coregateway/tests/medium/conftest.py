"""Shared test fixtures for medium tests.

This conftest.py file contains fixtures that are automatically discovered
by pytest and can be used across all test files in this directory without
explicit imports.
"""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from coregateway.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def mock_logger() -> MagicMock:
    """Mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx.AsyncClient for network isolation."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"result": "ok"}
    mock_response.content = b'{"result": "ok"}'
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
    mock_logger: MagicMock,
    stub_async_http_transport: MagicMock,
) -> Iterator[TestClient]:
    """Test client with mocked HTTP requests and logger.

    This fixture sets up a FastAPI TestClient with:
    - Mocked httpx.AsyncClient to isolate network calls
    - Mocked logger for testing logging behavior
    - Stubbed HTTP transport to avoid real network setup

    The mock httpx client is pre-configured with default successful responses,
    but individual tests can override the behavior as needed.
    """
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        app = create_app(port=8001)
        # Replace the real logger with mock
        app.state.logger = mock_logger
        # Ensure the mock httpx client is used by the app
        app.state.httpx_client = mock_httpx_client
        with TestClient(app) as client:
            yield client
