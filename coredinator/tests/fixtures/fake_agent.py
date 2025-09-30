#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def _parse_args(argv: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {"config-name": None}
    it = iter(argv)
    for a in it:
        if a == "--config-name":
            try:
                out["config-name"] = next(it)
            except StopIteration:
                pass
    return out


def _install_sigterm_exit():
    def handler(signum: int, frame: object | None) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)


class _HealthcheckHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/api/healthcheck":
            self.send_error(404)
            return

        status = os.environ.get("FAKE_AGENT_HEALTHCHECK", "healthy")
        if status == "unhealthy":
            body = json.dumps({"detail": "Service is unhealthy"})
            self.send_response(500)
        else:
            body = json.dumps({"status": "ok"})
            self.send_response(200)

        encoded = body.encode("utf-8")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default logging to keep tests quiet.
        return


def _start_server() -> HTTPServer:
    """Start lightweight HTTP server providing the fake agent healthcheck."""

    port = int(os.environ.get("FAKE_AGENT_PORT", "0"))
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    server = HTTPServer(("127.0.0.1", port), _HealthcheckHandler)

    # Persist the chosen port for tests to read if requested
    port_file = os.environ.get("FAKE_AGENT_PORT_FILE")
    if port_file:
        with open(port_file, "w", encoding="utf-8") as f:
            f.write(str(server.server_address[1]))

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main(argv: list[str]) -> int:
    _ = _parse_args(argv)

    mode = os.environ.get("FAKE_AGENT_BEHAVIOR", "long")
    if mode == "exit-0":
        return 0
    if mode == "exit-1":
        return 1

    _install_sigterm_exit()

    # Start lightweight HTTP server in background thread (only for long-running mode)
    server = _start_server()

    # Stay alive until killed; sleep in small increments to react to signals.
    try:
        while True:
            time.sleep(0.1)
    except SystemExit as e:
        if server is not None:
            server.shutdown()
            server.server_close()
        return int(e.code) if e.code is not None else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
