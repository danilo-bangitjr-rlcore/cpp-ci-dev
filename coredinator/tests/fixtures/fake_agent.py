#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI


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


def _create_app():
    """Create FastAPI app with healthcheck endpoint."""
    app = FastAPI()

    @app.get("/api/healthcheck")
    async def _():
        # Check if we should return unhealthy (check each time, not just at startup)
        import os
        if os.environ.get("FAKE_AGENT_HEALTHCHECK", "healthy") == "unhealthy":
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Service is unhealthy")
        return {"status": "ok"}

    return app


def _run_server():
    """Run FastAPI server in background thread."""
    import socket

    app = _create_app()
    if app is None:
        return

    # Find an available port
    port = int(os.environ.get("FAKE_AGENT_PORT", "0"))
    if port == 0:
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

    # Write the port to a file so tests can read it
    port_file = os.environ.get("FAKE_AGENT_PORT_FILE")
    if port_file:
        with open(port_file, "w") as f:
            f.write(str(port))

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="critical",  # Suppress logs during tests
    )
    server = uvicorn.Server(config)
    server.run()


def main(argv: list[str]) -> int:
    _ = _parse_args(argv)

    mode = os.environ.get("FAKE_AGENT_BEHAVIOR", "long")
    if mode == "exit-0":
        return 0
    if mode == "exit-1":
        return 1

    _install_sigterm_exit()

    # Start FastAPI server in background thread (only for long-running mode)
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    # Stay alive until killed; sleep in small increments to react to signals.
    try:
        while True:
            time.sleep(0.1)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
