import asyncio
import platform

import psutil
from fastapi import APIRouter
from pydantic import BaseModel


class PlatformResponse(BaseModel):
    data: str


class CPUResponse(BaseModel):
    percent: float


class CPUPerCoreResponse(BaseModel):
    percent: list[float]


class MemoryResponse(BaseModel):
    percent: float
    used_gb: float
    total_gb: float


class DiskResponse(BaseModel):
    percent: float
    used_gb: float
    total_gb: float


class SystemMetricsResponse(BaseModel):
    platform: str
    cpu: CPUResponse
    cpu_per_core: CPUPerCoreResponse
    ram: MemoryResponse
    disk: DiskResponse


system_metrics_router = APIRouter(
    tags=["System Metrics"],
)

@system_metrics_router.get("/api/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get all system metrics in a single request.

    Returns:
        Dictionary containing platform, CPU, RAM, and disk usage information
    """
    # Fetch all metrics in parallel for better performance
    platform_task = asyncio.create_task(asyncio.to_thread(platform.system))
    cpu_per_core_task = asyncio.create_task(asyncio.to_thread(psutil.cpu_percent, 1, True))
    ram_task = asyncio.create_task(asyncio.to_thread(psutil.virtual_memory))
    disk_task = asyncio.create_task(asyncio.to_thread(psutil.disk_usage, '/'))

    platform_data, cpu_per_core, ram, disk = await asyncio.gather(
        platform_task, cpu_per_core_task, ram_task, disk_task,
    )

    # Calculate average CPU from per-core values
    cpu_avg = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0.0

    return {
        "platform": platform_data,
        "cpu": {"percent": cpu_avg},
        "cpu_per_core": {"percent": cpu_per_core},
        "ram": {"percent": ram.percent, "used_gb": ram.used / (1024**3), "total_gb": ram.total / (1024**3)},
        "disk": {"percent": disk.percent, "used_gb": disk.used / (1024**3), "total_gb": disk.total / (1024**3)},
    }

@system_metrics_router.get("/api/system/platform", response_model=PlatformResponse)
async def get_system_platform():
    """
    Get the operating system platform.

    Returns:
        Dictionary containing the platform name (e.g., 'Linux', 'Windows', 'Darwin')
    """
    return {"data": platform.system()}

@system_metrics_router.get("/api/system/cpu", response_model=CPUResponse)
async def get_cpu():
    """
    Get overall CPU usage percentage.

    Returns:
        Dictionary containing CPU usage percentage averaged across all cores
    """
    return {"percent": psutil.cpu_percent(interval=1)}

@system_metrics_router.get("/api/system/cpu_per_core", response_model=CPUPerCoreResponse)
async def get_cpu_per_core():
    """
    Get CPU usage percentage for each individual core.

    Returns:
        Dictionary containing a list of CPU usage percentages, one per core
    """
    return {"percent": psutil.cpu_percent(interval=1, percpu=True)}

@system_metrics_router.get("/api/system/ram", response_model=MemoryResponse)
async def get_ram():
    """
    Get RAM memory usage statistics.

    Returns:
        Dictionary containing RAM usage percentage, used memory in GB, and total memory in GB
    """
    ram = psutil.virtual_memory()
    return {"percent": ram.percent, "used_gb": ram.used / (1024**3), "total_gb": ram.total / (1024**3)}

@system_metrics_router.get("/api/system/disk", response_model=DiskResponse)
async def get_disk():
    """
    Get disk usage statistics for the root filesystem.

    Returns:
        Dictionary containing disk usage percentage, used space in GB, and total space in GB
    """
    disk = psutil.disk_usage('/')
    return {"percent": disk.percent, "used_gb": disk.used / (1024**3), "total_gb": disk.total / (1024**3)}
