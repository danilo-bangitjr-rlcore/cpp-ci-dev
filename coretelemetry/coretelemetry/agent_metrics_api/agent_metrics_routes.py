from pathlib import Path

from coretelemetry.agent_metrics_api.services import (
    AgentMetricsManager,
    DBConfig,
    get_agent_metrics_manager,
)
from fastapi import APIRouter, Depends

agent_metrics_router = APIRouter(
    tags=["Agent Metrics"],
)

@agent_metrics_router.post("/api/config/clear_cache")
async def clear_cache(manager: AgentMetricsManager = Depends(get_agent_metrics_manager)): # noqa: B008
    """
    Clear all cached data including YAML configuration cache.

    Returns:
        Success message confirming cache was cleared
    """
    manager.clear_cache()
    return {"message": "Cache cleared successfully"}

@agent_metrics_router.get("/api/data/{agent_id}")
async def get_telemetry(
    agent_id: str,
    metric: str,
    start_time: str | None = None,
    end_time: str | None = None,
    manager: AgentMetricsManager = Depends(get_agent_metrics_manager), # noqa: B008
):
    """
    Get telemetry data for a specific agent and metric within a date range.

    Args:
        agent_id: The ID of the agent
        metric: The name of the metric to retrieve
        start_time: Start date/time for the telemetry data (query param 'start_time', optional).
                   If timezone is not specified, UTC is assumed.
        end_time: End date/time for the telemetry data (query param 'end_time', optional).
                 If timezone is not specified, UTC is assumed.

    Returns:
        Telemetry data for the specified parameters

    Note:
        If neither start_time nor end_time is specified, returns the latest value only.
        If only one of start_time or end_time is specified, the query may return too many rows
        and result in a 413 error. It is recommended to specify both start_time and end_time
        for time-range queries.
    """
    # Add UTC timezone if not present
    if start_time and not start_time.endswith(("+00", "Z", "UTC")) and "+" not in start_time[-6:]:
        start_time = f"{start_time}+00"
    if end_time and not end_time.endswith(("+00", "Z", "UTC")) and "+" not in end_time[-6:]:
        end_time = f"{end_time}+00"

    return manager.get_telemetry_data(agent_id, metric, start_time, end_time)

@agent_metrics_router.get("/api/data/{agent_id}/metrics")
async def get_available_metrics(
    agent_id: str,
    manager: AgentMetricsManager = Depends(get_agent_metrics_manager), # noqa: B008
):
    """
    Get all available metrics for a specific agent.

    Args:
        agent_id: The ID of the agent

    Returns:
        List of available metric names for the agent
    """
    return manager.get_available_metrics(agent_id)

@agent_metrics_router.get("/api/config/db", response_model=DBConfig)
async def get_db_config(manager: AgentMetricsManager = Depends(get_agent_metrics_manager)): # noqa: B008
    """
    Get the current database configuration.

    Returns:
        Current database configuration settings
    """
    return manager.get_db_config()

@agent_metrics_router.post("/api/config/db")
async def set_db_config(
    config: DBConfig,
    manager: AgentMetricsManager = Depends(get_agent_metrics_manager), # noqa: B008
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

@agent_metrics_router.get("/api/config/path")
async def get_config_path(manager: AgentMetricsManager = Depends(get_agent_metrics_manager)): # noqa: B008
    """
    Get the current configuration path.

    Returns:
        Current configuration path as a string
    """
    return {"config_path": str(manager.get_config_path())}

@agent_metrics_router.post("/api/config/path")
async def set_config_path(
    path: str,
    manager: AgentMetricsManager = Depends(get_agent_metrics_manager), # noqa: B008
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
