import glob
from pathlib import Path

from coredinator.service.service import Service, ServiceConfig, ServiceID


class UAServer(Service):
    def __init__(
        self,
        id: ServiceID,
        config_path: Path,
        base_path: Path,
        config: ServiceConfig | None = None,
        version: str | None = None,
    ):
        super().__init__(
            id=id,
            base_path=base_path,
            config_path=config_path,
            config=config,
            version=version,
        )

    def _find_executable(self) -> Path:
        exe_pattern = str(self._base_path / "**/*opc_server*")
        matches = glob.glob(exe_pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(f"No opc_server executable found in {self._base_path} matching '**/*opc_server*'")
        return Path(matches[0])

    def _build_args(self, exe: Path, cfg: Path) -> list[str]:
        return [str(exe)]
