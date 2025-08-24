from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest
from pydantic import BaseModel

from corecli.utils import http


class _TestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/ok":
            payload = {"foo": "bar", "n": 1}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/invalid-json":
            data = b"{not: valid json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/slow":
            # delay longer than typical client timeout
            time.sleep(2)
            payload = {"foo": "slow"}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/not-json":
            data = b"hello world"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/server-error":
            self.send_response(500)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Silence the default logging to keep test output clean
        return


@pytest.fixture(scope="module")
def http_server_url() -> Generator[str]:
    server = HTTPServer(('localhost', 0), _TestHandler)

    host = server.server_address[0]
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://{host}:{port}"

    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)


class SimpleModel(BaseModel):
    foo: str
    n: int | None = None


# =====================
# == Core Tests =======
# =====================

def test_maybe_validate_success():
    """Test maybe_validate with valid data"""
    data = {"foo": "bar", "n": 42}
    result = http.maybe_validate(SimpleModel, data)

    assert result.is_some()
    model = result.expect("Expected valid model")
    assert model.foo == "bar"
    assert model.n == 42


def test_maybe_validate_missing_required_field():
    """Test maybe_validate with missing required field"""
    data = {"n": 42}  # missing 'foo'
    result = http.maybe_validate(SimpleModel, data)

    assert result.is_none()


def test_maybe_open_url_invalid_url():
    """Test maybe_open_url with malformed URL"""
    result = http.maybe_open_url("not-a-url", timeout=1.0)

    assert result.is_none()


def test_maybe_get_json_success(http_server_url: str) -> None:
    """Test successful HTTP + JSON + validation flow"""
    url = f"{http_server_url}/ok"
    res = http.maybe_get_json(url, SimpleModel, timeout=1.0)

    model = res.expect("expected value")
    assert model.foo == "bar"
    assert model.n == 1


def test_maybe_get_json_invalid_json(http_server_url: str) -> None:
    """Test that malformed JSON is handled gracefully (our bug fix!)"""
    url = f"{http_server_url}/invalid-json"
    res = http.maybe_get_json(url, SimpleModel, timeout=1.0)

    # This is the critical test - invalid JSON should result in Maybe(None)
    assert res.is_none()


def test_maybe_get_json_not_json(http_server_url: str) -> None:
    """Test that non-JSON responses are handled"""
    url = f"{http_server_url}/not-json"
    res = http.maybe_get_json(url, SimpleModel, timeout=1.0)

    assert res.is_none()


def test_maybe_get_json_validation_error(http_server_url: str) -> None:
    """Test that pydantic validation errors are handled"""
    class StrictModel(BaseModel):
        missing_field: int

    url = f"{http_server_url}/ok"
    res = http.maybe_get_json(url, StrictModel, timeout=1.0)

    assert res.is_none()


def test_maybe_get_json_timeout(http_server_url: str) -> None:
    """Test that timeouts are handled"""
    url = f"{http_server_url}/slow"
    res = http.maybe_get_json(url, SimpleModel, timeout=0.1)

    assert res.is_none()


def test_maybe_get_json_http_error(http_server_url: str) -> None:
    """Test that HTTP errors are handled"""
    url = f"{http_server_url}/server-error"
    res = http.maybe_get_json(url, SimpleModel, timeout=1.0)

    assert res.is_none()


@patch('corecli.utils.http.urlopen')
def test_maybe_get_json_network_error(mock_urlopen: Mock) -> None:
    """Test that network errors are handled"""
    mock_urlopen.side_effect = URLError("Network unreachable")

    result = http.maybe_get_json("http://example.com/api", SimpleModel, timeout=1.0)
    assert result.is_none()
