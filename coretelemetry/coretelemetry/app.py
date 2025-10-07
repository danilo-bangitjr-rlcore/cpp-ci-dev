from pathlib import Path

from coretelemetry.services import (
    DBConfig,
    TelemetryManager,
    get_telemetry_manager,
)
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(title="CoreTelemetry API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# TODO: Add endpoint for refresh, call it clear cache...

@app.get("/api/v1/telemetry/{agent_id}")
async def get_telemetry(
    agent_id: str,
    metric: str,
    start_time: str | None = None,
    end_time: str | None = None,
    manager: TelemetryManager = Depends(get_telemetry_manager), # noqa: B008
):
    """
    Get telemetry data for a specific agent and metric within a date range.

    Args:
        agent_id: The ID of the agent
        metric: The name of the metric to retrieve
        start_time: Start date/time for the telemetry data (query param 'start_time', optional)
        end_time: End date/time for the telemetry data (query param 'end_time', optional)

    Returns:
        Telemetry data for the specified parameters

    Note:
        If neither start_time nor end_time is specified, returns the latest value only.
        If only one of start_time or end_time is specified, the query may return too many rows
        and result in a 413 error. It is recommended to specify both start_time and end_time
        for time-range queries.
    """
    return manager.get_telemetry_data(agent_id, metric, start_time, end_time)

@app.get("/api/v1/telemetry/{agent_id}/metrics")
async def get_available_metrics(
    agent_id: str,
    manager: TelemetryManager = Depends(get_telemetry_manager), # noqa: B008
):
    """
    Get all available metrics for a specific agent.

    Args:
        agent_id: The ID of the agent

    Returns:
        List of available metric names for the agent
    """
    return manager.get_available_metrics(agent_id)

@app.get("/api/v1/config/db", response_model=DBConfig)
async def get_db_config(manager: TelemetryManager = Depends(get_telemetry_manager)): # noqa: B008
    """
    Get the current database configuration.

    Returns:
        Current database configuration settings
    """
    return manager.get_db_config()

@app.post("/api/v1/config/db")
async def set_db_config(
    config: DBConfig,
    manager: TelemetryManager = Depends(get_telemetry_manager), # noqa: B008
):
    """
    Update the database configuration.

    Args:
        config: New database configuration settings

    Returns:
        Updated database configuration
    """
    updated_config = manager.set_db_config(config)
    return {"message": "Database configuration updated successfully", "config": updated_config}

@app.get("/api/v1/config/path")
async def get_config_path(manager: TelemetryManager = Depends(get_telemetry_manager)): # noqa: B008
    """
    Get the current configuration path.

    Returns:
        Current configuration path as a string
    """
    return {"config_path": str(manager.get_config_path())}

@app.post("/api/v1/config/path")
async def set_config_path(
    path: str,
    manager: TelemetryManager = Depends(get_telemetry_manager), # noqa: B008
):
    """
    Update the configuration path.

    Args:
        path: New configuration path as a string

    Returns:
        Updated configuration path
    """
    updated_path = manager.set_config_path(Path(path))
    return {"message": "Configuration path updated successfully", "config_path": str(updated_path)}

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="CoreTelemetry API")
    parser.add_argument(
        "--config-path",
        type=str,
        default="clean/",
        help="Path to the configuration directory (default: clean/)",
    )
    args = parser.parse_args()

    # Set the config path from command line args
    manager = get_telemetry_manager()
    manager.set_config_path(Path(args.config_path))

    uvicorn.run(app, host="0.0.0.0", port=8001)
