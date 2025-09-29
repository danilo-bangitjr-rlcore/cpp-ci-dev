import argparse
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

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
        port=main_port,
        version=version,
    )
    yield
    logger.info("CoreRL server shutting down")

def parse_args():
    parser = argparse.ArgumentParser(description="Coredinator Service")
    parser.add_argument("--base-path", type=Path, required=True, help="Path to microservice executables")
    parser.add_argument("--port", type=int, default=7000, help="Port to run the service on (default: 7000)")
    parser.add_argument("--log-file", type=Path, help="Path to log file for structured logging with rotation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level (default: INFO)")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging output")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    return args.base_path, args.port, args.log_file, args.log_level, not args.no_console, args.reload


# pyright: reportUnusedFunction=false
def create_app(base_path: Path) -> FastAPI:
    """Factory function to create FastAPI app with given base_path."""
    app = FastAPI(lifespan=lifespan)
    service_manager = ServiceManager(base_path=base_path)
    app.state.service_manager = service_manager
    app.state.base_path = base_path
    app.state.agent_manager = AgentManager(base_path=base_path, service_manager=service_manager)

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
    async def health():
        logger.debug("Health check requested")
        return {
            "status": "healthy",
            "process_id": os.getpid(),
            "service": "coredinator",
            "version": version,
        }

    app.include_router(agent_manager, prefix="/api/agents", tags=["Agent"])
    app.include_router(coreio_manager, prefix="/api/io", tags=["CoreIO"])

    return app


if __name__ == "__main__":
    base_path, main_port, log_file, log_level, console_output, reload = parse_args()

    # Initialize structured logging
    setup_structured_logging(
        log_file_path=log_file,
        log_level=log_level,
        console_output=console_output,
    )

    app = create_app(base_path)
    uvicorn.run(app, host="0.0.0.0", port=main_port, reload=reload)
