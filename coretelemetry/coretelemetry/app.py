import argparse
from pathlib import Path

import uvicorn
from coretelemetry.agent_metrics_api.agent_metrics_routes import AgentMetricsManager, agent_metrics_router
from coretelemetry.agent_metrics_api.exceptions import AgentMetricsException
from coretelemetry.agent_metrics_api.services import get_agent_metrics_manager
from coretelemetry.system_metrics_api.system_metrics_routes import system_metrics_router
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

__version__ = "0.1.0"

class CoretelemetryConfig(BaseModel):
    port: int = 7001
    config_path: str = "clean/"

coretelemetry_config = CoretelemetryConfig()

def parse_args():
    parser = argparse.ArgumentParser(description="CoreTelemetry API")
    parser.add_argument(
        "--config-path",
        type=str,
        default="clean/",
        help="Path to the configuration directory (default: clean/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7001,
        help="Port to run the server on (default: 7001)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser.parse_args()

# pyright: reportUnusedFunction=false
def create_app(config_path: str | Path) -> FastAPI:
    app = FastAPI(title="CoreTelemetry API")

    # Global exception handler for all domain exceptions
    @app.exception_handler(AgentMetricsException)
    async def telemetry_exception_handler(request: Request, exc: AgentMetricsException):
        """Convert domain exceptions to HTTP responses with appropriate status codes."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message},
        )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent_metrics_manager = get_agent_metrics_manager()
    agent_metrics_manager.set_config_path(Path(config_path))

    app.include_router(agent_metrics_router)
    app.include_router(system_metrics_router)


    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health_check(agent_metrics_manager: AgentMetricsManager = Depends(get_agent_metrics_manager)): # noqa: B008
        db_connected = agent_metrics_manager.test_db_connection()
        return {"status": "healthy", "db_connected": db_connected}

    return app

def get_app() -> FastAPI:
    # Parsing args inside get_app to get consistent config across reloads
    args = parse_args()
    coretelemetry_config.port = args.port
    coretelemetry_config.config_path = args.config_path

    return create_app(coretelemetry_config.config_path)

if __name__ == "__main__":
    args = parse_args()

    if args.reload:
        # Use string import for reload support (dev only)
        uvicorn.run(
            "coretelemetry.app:get_app",
            host="0.0.0.0",
            port=args.port,
            reload=True,
            factory=True,
        )
    else:
        # Use direct app instance for executable compatibility
        app = get_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
