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

__version__ = "0.1.0"


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

if __name__ == "__main__":

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

    args = parser.parse_args()

    app = create_app(args.config_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=args.reload)
