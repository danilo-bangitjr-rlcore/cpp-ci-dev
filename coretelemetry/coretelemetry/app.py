import argparse
from pathlib import Path

import uvicorn
from coretelemetry.agent_metrics_api.agent_metrics_routes import agent_metrics_router
from coretelemetry.agent_metrics_api.exceptions import TelemetryException
from coretelemetry.agent_metrics_api.services import get_agent_metrics_manager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

__version__ = "0.1.0"


# pyright: reportUnusedFunction=false
def create_app(config_path: str | Path) -> FastAPI:
    app = FastAPI(title="CoreTelemetry API")

    # Global exception handler for all domain exceptions
    @app.exception_handler(TelemetryException)
    async def telemetry_exception_handler(request: Request, exc: TelemetryException):
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
        default=8001,
        help="Port to run the server on (default: 8001)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    app = create_app(args.config_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=args.reload)
