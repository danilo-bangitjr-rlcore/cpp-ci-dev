from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from coredinator.service.protocols import ServiceBundleID, ServiceID
from coredinator.service.service_manager import ServiceManager
from coredinator.services.coreio import CoreIOService
from coredinator.utils.http import convert_to_http_exception

router = APIRouter()


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
        raise HTTPException(status_code=400, detail=f"Config file not found at {cfg}")

    if coreio_id is None:
        coreio_id = ServiceID(f"{cfg.stem}-coreio")

    # Create a unique owner ID for this API call (using config path as identifier)
    api_owner_id = ServiceBundleID(f"coreio-api-{cfg.name}")

    base_path = request.app.state.base_path
    service = service_manager.get_or_register_service(
        coreio_id,
        lambda: CoreIOService(
            id=coreio_id,
            config_path=cfg,
            base_path=base_path,
        ),
    )

    # Register this API call as an owner of the service
    service_manager.add_service_owner(coreio_id, api_owner_id)

    service.start()
    return {
        "service_id": coreio_id,
        "message": f"CoreIO service '{coreio_id}' started successfully",
        "config_path": str(cfg),
        "status": service.status(),
    }


@router.post("/{coreio_id}/stop")
def coreio_stop(coreio_id: ServiceID, request: Request):
    service_manager = get_service_manager(request)

    service = service_manager.get_service(coreio_id)
    if service is None:
        raise HTTPException(status_code=404, detail=f"CoreIO service with ID '{coreio_id}' not found")

    # Check if service can be safely stopped (not shared by other API calls)
    if service_manager.is_service_shared(coreio_id):
        owners = service_manager.get_service_owners(coreio_id)
        raise HTTPException(
            status_code=409,
            detail=f"Cannot stop CoreIO service '{coreio_id}' - still in use by: {list(owners)}",
        )

    # Remove all ownership and stop the service
    owners = list(service_manager.get_service_owners(coreio_id))
    for owner_id in owners:
        service_manager.remove_service_owner(coreio_id, owner_id)

    service.stop()
    service_manager.remove_service(coreio_id)

    return {"message": f"CoreIO service '{coreio_id}' stopped successfully"}


@router.get("/{coreio_id}/status")
def coreio_status(coreio_id: ServiceID, request: Request):
    service_manager = get_service_manager(request)

    service = service_manager.get_service(coreio_id)
    if service is None:
        raise HTTPException(status_code=404, detail=f"CoreIO service with ID '{coreio_id}' not found")

    owners = service_manager.get_service_owners(coreio_id)
    is_shared = service_manager.is_service_shared(coreio_id)

    return {
        "service_id": coreio_id,
        "status": service.status(),
        "owners": list(owners),
        "is_shared": is_shared,
    }


@router.get("/")
def coreio_list(request: Request):
    service_manager = get_service_manager(request)

    all_services = service_manager.list_services()
    all_service_objects = [service_manager.get_service(service_id) for service_id in all_services]

    # Simple filter without lib_utils to test basic functionality
    coreio_service_objects = [
        service for service in all_service_objects
        if service is not None and hasattr(service, '__class__') and service.__class__.__name__ == 'CoreIOService'
    ]

    coreio_services: list[dict[str, Any]] = []
    for service in coreio_service_objects:
        service_id = service.id
        owners = service_manager.get_service_owners(service_id)
        is_shared = service_manager.is_service_shared(service_id)
        coreio_services.append(
            {
                "service_id": service_id,
                "status": service.status(),
                "owners": list(owners),
                "is_shared": is_shared,
            },
        )

    return {"coreio_services": coreio_services}
