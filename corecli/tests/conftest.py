import json
import tempfile
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner as ClickCliRunner
from test.infrastructure.networking import get_free_port

from corecli.main import cli
from corecli.utils.coredinator import is_coredinator_running, wait_for_coredinator_stop
from tests.utils.cli import CliRunner
from tests.utils.waiting import wait_for_event


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
def runner() -> ClickCliRunner:
    return ClickCliRunner()


@pytest.fixture
def cli_app():
    return cli


@pytest.fixture
def running_coredinator(
    free_port: int,
    corecli_runner: CliRunner,
    coredinator_log_file: Path,
    coredinator_base_path: Path,
) -> Generator[int]:
    start_result = corecli_runner.start_coredinator(free_port, coredinator_log_file, coredinator_base_path)
    assert start_result.returncode == 0, f"Start command failed: {start_result.stderr}"

    wait_for_event(
        lambda: is_coredinator_running(free_port),
        timeout=30.0,
        description=f"coredinator to start on port {free_port}",
    )

    yield free_port

    # Cleanup
    corecli_runner.stop_coredinator(free_port)
    wait_for_coredinator_stop(free_port, timeout=10.0)


@pytest.fixture
def free_port() -> int:
    return get_free_port("localhost")


@pytest.fixture
def monorepo_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def corecli_runner(monorepo_root: Path) -> CliRunner:
    return CliRunner(monorepo_root)


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def coredinator_log_file(temp_log_dir: Path) -> Path:
    return temp_log_dir / "coredinator.log"


@pytest.fixture
def coredinator_base_path(tmp_path: Path) -> Path:
    return tmp_path / "coredinator_data"
