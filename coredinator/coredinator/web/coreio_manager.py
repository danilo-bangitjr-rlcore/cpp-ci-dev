import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from lib_utils.list import filter_instance
from pydantic import BaseModel

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID
from coredinator.service.service_manager import ServiceManager
from coredinator.services.coreio import CoreIOService
from coredinator.utils.http import convert_to_http_exception

router = APIRouter()
log = get_logger(__name__)


# Dependency injection for service_manager
def get_service_manager(request: Request) -> ServiceManager:
    return request.app.state.service_manager


class StartCoreIORequestPayload(BaseModel):
    config_path: Path
    coreio_id: ServiceID | None = None


@router.post("/start")
def coreio_start(req_payload: StartCoreIORequestPayload, request: Request):
    service_manager = get_service_manager(request)
    cfg = req_payload.config_path
    coreio_id = req_payload.coreio_id

    return _start_coreio(service_manager, cfg, coreio_id, request)


@convert_to_http_exception(FileNotFoundError, status_code=400)
def _start_coreio(service_manager: ServiceManager, cfg: Path, coreio_id: ServiceID | None, request: Request):
    """Start a new CoreIO service instance."""
    if not cfg.exists():
        log.warning(
            "CoreIO start aborted: config missing",
            config_path=str(cfg),
        )
        raise HTTPException(status_code=400, detail=f"Config file not found at {cfg}")

    if coreio_id is None:
        coreio_id = ServiceID(f"{cfg.stem}-coreio")


    request_start = time.perf_counter()
    log.info(
        "CoreIO start request received",
        service_id=coreio_id,
        config_path=str(cfg),
    )

    base_path = request.app.state.base_path
    service_exists = service_manager.has_service(coreio_id)
    if service_exists:
        log.info(
            "Reusing existing CoreIO service",
            service_id=coreio_id,
            config_path=str(cfg),
        )

    service = service_manager.get_or_register_service(
        coreio_id,
        lambda: CoreIOService(
            id=coreio_id,
            config_path=cfg,
            base_path=base_path,
        ),
    )

    start_elapsed_start = time.perf_counter()
    service.start()
    service_start_elapsed = round(time.perf_counter() - start_elapsed_start, 3)
    total_elapsed = round(time.perf_counter() - request_start, 3)
    log.info(
        "CoreIO service.start completed",
        service_id=coreio_id,
        elapsed_seconds=service_start_elapsed,
        total_elapsed_seconds=total_elapsed,
        reused=service_exists,
    )

    status_start = time.perf_counter()
    status = service.status()
    status_elapsed = round(time.perf_counter() - status_start, 3)
    total_elapsed = round(time.perf_counter() - request_start, 3)
    log.info(
        "CoreIO service status evaluated",
        service_id=coreio_id,
        state=status.state.value,
        intended=status.intended_state.value,
        status_elapsed_seconds=status_elapsed,
        total_elapsed_seconds=total_elapsed,
    )
    return {
        "service_id": coreio_id,
        "message": f"CoreIO service '{coreio_id}' started successfully",
        "config_path": str(cfg),
        "status": status,
    }


@router.post("/{coreio_id}/stop")
def coreio_stop(coreio_id: ServiceID, request: Request):
    service_manager = get_service_manager(request)
    log.info("CoreIO stop request received", service_id=coreio_id)

    stop_timer_start = time.perf_counter()

    service = service_manager.get_service(coreio_id)
    if service is None:
        log.warning("CoreIO stop failed: service not found", service_id=coreio_id)
        raise HTTPException(status_code=404, detail=f"CoreIO service with ID '{coreio_id}' not found")

    service.stop()
    service_manager.remove_service(coreio_id)

    stop_elapsed = round(time.perf_counter() - stop_timer_start, 3)
    log.info(
        "CoreIO service stopped",
        service_id=coreio_id,
        elapsed_seconds=stop_elapsed,
    )

    return {"message": f"CoreIO service '{coreio_id}' stopped successfully"}


@router.get("/{coreio_id}/status")
def coreio_status(coreio_id: ServiceID, request: Request):
    service_manager = get_service_manager(request)
    log.debug("CoreIO status request received", service_id=coreio_id)

    service = service_manager.get_service(coreio_id)
    if service is None:
        log.warning("CoreIO status failed: service not found", service_id=coreio_id)
        raise HTTPException(status_code=404, detail=f"CoreIO service with ID '{coreio_id}' not found")

    return {
        "service_id": coreio_id,
        "status": service.status(),
    }


@router.get("/")
def coreio_list(request: Request):
    service_manager = get_service_manager(request)
    log.debug("CoreIO list request received")

    all_services = service_manager.list_services()
    all_service_objects = [service_manager.get_service(service_id) for service_id in all_services]

    coreio_service_objects = filter_instance(CoreIOService, all_service_objects)

    coreio_services: list[dict[str, Any]] = []
    for service in coreio_service_objects:
        service_id = service.id
        coreio_services.append(
            {
                "service_id": service_id,
                "status": service.status(),
            },
        )

    log.info(
        "CoreIO services listed",
        total_services=len(coreio_services),
    )

    return {"coreio_services": coreio_services}
