from pathlib import Path

from coredinator.service.service import Service, ServiceID


class CoreRLService(Service):
    EXECUTABLE_PATH = Path("dist/corerl")
    def __init__(self, id: ServiceID, config_path: Path):
        super().__init__(
            id=id,
            executable_path=self.EXECUTABLE_PATH,
            config_path=config_path,
        )
