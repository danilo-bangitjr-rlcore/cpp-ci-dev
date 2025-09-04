import glob
from pathlib import Path

from coredinator.service.service import Service, ServiceConfig, ServiceID


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
        exe_pattern = str(base_path / "**/*coreio-*")
        matches = glob.glob(exe_pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(f"No coreio executable found in {base_path} matching '**/*coreio-*'")
        return Path(matches[0])
