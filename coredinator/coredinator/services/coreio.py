from pathlib import Path

from coredinator.service.service import Service, ServiceConfig, ServiceID
from coredinator.utils.executable import find_service_executable


class CoreIOService(Service):
    def __init__(self, id: ServiceID, config_path: Path, base_path: Path, config: ServiceConfig | None = None):
        super().__init__(
            id=id,
            base_path=base_path,
            config_path=config_path,
            config=config,
        )

    def _find_executable(self) -> Path:
        return find_service_executable(self._base_path, "coreio")
