import argparse
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from lib_events.server.manager import EventBusManager

from coredinator.agent.agent_manager import AgentManager
from coredinator.logging_config import get_logger, setup_structured_logging
from coredinator.service.service_manager import ServiceManager
from coredinator.web.agent_manager import router as agent_manager
from coredinator.web.coreio_manager import router as coreio_manager

# Structured logger for application events
logger = get_logger(__name__)

version = "0.0.1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "CoreRL server starting up",
        base_path=str(app.state.base_path),
        version=version,
    )

    app.state.event_bus_manager.start()
    logger.info("Event bus started")

    yield

    logger.info("CoreRL server shutting down")
    app.state.event_bus_manager.stop()
    logger.info("Event bus stopped")

class CliArgs(NamedTuple):
    base_path: Path
    port: int
    log_file: Path | None
    log_level: str
    console_output: bool
    reload: bool
    event_bus_host: str
    event_bus_port: int

def parse_args():
    parser = argparse.ArgumentParser(description="Coredinator Service")
    parser.add_argument("--base-path", type=Path, required=True, help="Path to microservice executables")
    parser.add_argument("--port", type=int, default=7000, help="Port to run the service on (default: 7000)")
    parser.add_argument("--log-file", type=Path, help="Path to log file for structured logging with rotation")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO)")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging output")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--event-bus-host", type=str, default="*",
                        help="Event bus host address (default: *)")
    parser.add_argument(
        "--event-bus-port", type=int, default=5580,
        help="Event bus DEALER/ROUTER port (default: 5580)",
    )

    args = parser.parse_args()

    return CliArgs(
        base_path=args.base_path,
        port=args.port,
        log_file=args.log_file,
        log_level=args.log_level,
        console_output=not args.no_console,
        reload=args.reload,
        event_bus_host=args.event_bus_host,
        event_bus_port=args.event_bus_port,
    )

# pyright: reportUnusedFunction=false
def _prepare_base_path(base_path: Path | str) -> Path:
    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    resolved = base_path.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Base path must be a directory: {resolved}")

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def create_app(
    base_path: Path,
    event_bus_host: str = "*",
    event_bus_port: int = 5580,
) -> FastAPI:
    """Factory function to create FastAPI app with given base_path."""
    prepared_base_path = _prepare_base_path(base_path)

    app = FastAPI(lifespan=lifespan)
    service_manager = ServiceManager(base_path=prepared_base_path)
    event_bus_manager = EventBusManager(
        host=event_bus_host,
        port=event_bus_port,
    )
    app.state.service_manager = service_manager
    app.state.event_bus_manager = event_bus_manager
    app.state.base_path = prepared_base_path
    app.state.agent_manager = AgentManager(base_path=prepared_base_path, service_manager=service_manager)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_core_rl_version(request: Request, call_next: Callable):
        logger.debug(
            "Processing request",
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None,
        )

        response = await call_next(request)
        response.headers["X-CoreRL-Version"] = version

        logger.debug(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            version=version,
        )

        return response

    @app.get("/")
    async def redirect():
        return RedirectResponse(url="/docs")

    @app.get("/api/healthcheck")
    async def health(request: Request):
        logger.debug("Health check requested")
        event_bus_healthy = request.app.state.event_bus_manager.is_healthy()
        return {
            "status": "healthy" if event_bus_healthy else "degraded",
            "process_id": os.getpid(),
            "service": "coredinator",
            "version": version,
            "event_bus": {
                "status": "running" if event_bus_healthy else "stopped",
                "config": request.app.state.event_bus_manager.get_config(),
            },
        }

    app.include_router(agent_manager, prefix="/api/agents", tags=["Agent"])
    app.include_router(coreio_manager, prefix="/api/io", tags=["CoreIO"])

    return app

def get_app() -> FastAPI:
    # Parsing args again inside get_app to get consistent config across reloads
    cli_args = parse_args()

    return create_app(
        cli_args.base_path,
        cli_args.event_bus_host,
        cli_args.event_bus_port,
    )


if __name__ == "__main__":
    cli_args = parse_args()

    if cli_args.reload:
        # Dev mode: use uvicorn reload (bypasses service framework)
        setup_structured_logging(
            log_file_path=cli_args.log_file,
            log_level=cli_args.log_level,
            console_output=cli_args.console_output,
        )
        uvicorn.run(
            "coredinator.app:get_app",
            host="0.0.0.0",
            port=cli_args.port,
            reload=True,
            factory=True,
        )
    else:
        # Production mode: use service framework
        from coredinator.core_service import CoredinatorService

        service = CoredinatorService(
            base_path=cli_args.base_path,
            port=cli_args.port,
            log_file=cli_args.log_file,
            log_level=cli_args.log_level,
            console_output=cli_args.console_output,
            event_bus_host=cli_args.event_bus_host,
            event_bus_port=cli_args.event_bus_port,
        )
        service.run_forever(max_retries=5, retry_window_hours=1, enable_retry=True)
