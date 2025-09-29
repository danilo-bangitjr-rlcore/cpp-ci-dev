import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner

from corecli.main import cli


class _TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Silence the default logging to keep test output clean
        return


@pytest.fixture(scope="session")
def http_server_url():
    """Start a simple HTTP server in a background thread and yield the base URL.

    The server binds to an ephemeral port on localhost. Yields something like
    'http://127.0.0.1:12345'.
    """

    # bind to an ephemeral port
    server = None
    for host in ("127.0.0.1", "localhost"):
        try:
            server = HTTPServer((host, 0), _TestHandler)
            break
        except OSError:
            server = None
    assert server is not None, "failed to bind test HTTP server"

    host = server.server_address[0]
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://{host}:{port}"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_app():
    return cli
