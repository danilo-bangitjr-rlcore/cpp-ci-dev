from fastapi import APIRouter
import platform
import psutil

system_metrics_router = APIRouter(
    tags=["System Metrics"],
)

@system_metrics_router.get("/api/v1/coretelemetry/system/platform")
async def get_system_patform():
    return {"data": platform.system()}

@system_metrics_router.get("/api/v1/coretelemetry/system/cpu")
async def get_cpu():
    return {"percent": psutil.cpu_percent(interval=1)}

@system_metrics_router.get("/api/v1/coretelemetry/system/cpu_per_core")
async def get_cpu_per_core():
    return {"percent": psutil.cpu_percent(interval=1, percpu=True)}

@system_metrics_router.get("/api/v1/coretelemetry/system/ram")
async def get_ram():
    ram = psutil.virtual_memory()
    return {"percent": ram.percent, "used_gb": ram.used / (1024**3), "total_gb": ram.total / (1024**3)}

@system_metrics_router.get("/api/v1/coretelemetry/system/disk")
async def get_disk():
    disk = psutil.disk_usage('/')
    return {"percent": disk.percent, "used_gb": disk.used / (1024**3), "total_gb": disk.total / (1024**3)}
