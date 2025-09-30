from pathlib import Path

from coredinator.service.service import Service, ServiceConfig, ServiceID
from coredinator.utils.executable import find_service_executable


class CoreIOService(Service):
    def __init__(self, id: ServiceID, config_path: Path, base_path: Path, config: ServiceConfig | None = None):
        executable_path = self._find_executable(base_path)
        super().__init__(
            id=id,
            executable_path=executable_path,
            config_path=config_path,
            config=config,
        )

    @staticmethod
    def _find_executable(base_path: Path) -> Path:
        return find_service_executable(base_path, "coreio")
