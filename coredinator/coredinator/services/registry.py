"""Service registry for dynamic service instantiation."""

from pathlib import Path

from lib_utils.errors import fail_gracefully

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID
from coredinator.services.coreio import CoreIOService
from coredinator.services.corerl import CoreRLService
from coredinator.services.demos.tep import TEPService
from coredinator.services.demos.uaserver import UAServer

logger = get_logger(__name__)


@fail_gracefully(logger)
def create_service_instance(
    service_id: ServiceID,
    service_type: str,
    config_path: Path,
    base_path: Path,
    version: str | None = None,
):
    if service_type == "CoreIOService":
        return CoreIOService(id=service_id, config_path=config_path, base_path=base_path)
    if service_type == "CoreRLService":
        return CoreRLService(id=service_id, config_path=config_path, base_path=base_path)
    if service_type == "TEPService":
        return TEPService(id=service_id, config_path=config_path, base_path=base_path)
    if service_type == "UAServer":
        return UAServer(id=service_id, config_path=config_path, base_path=base_path)

    logger.error(
        "Unknown service type",
        service_type=service_type,
        service_id=service_id,
    )
    return None
