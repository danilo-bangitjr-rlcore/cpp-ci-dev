from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from coredinator.service.protocols import ServiceID, ServiceLike, ServiceState
from coredinator.services import CoreIOService, CoreRLService

AgentID = NewType("AgentID", str)


@dataclass(frozen=True)
class AgentStatus:
    id: AgentID
    state: ServiceState
    config_path: Path | None


class Agent(ServiceLike):
    def __init__(self, id: AgentID, config_path: Path):
        self._id = id
        self._config_path = config_path
        self._corerl_service = CoreRLService(
            id=ServiceID(f"{id}-corerl"),
            config_path=config_path,
        )
        self._coreio_service = CoreIOService(
            id=ServiceID(f"{id}-coreio"),
            config_path=config_path,
        )

    @property
    def id(self):
        return ServiceID(self._id)

    def start(self):
        self._coreio_service.start()
        self._corerl_service.start()

    def stop(self, grace_seconds: float = 5.0):
        self._corerl_service.stop(grace_seconds)
        self._coreio_service.stop(grace_seconds)

    def restart(self):
        self.stop()
        self.start()

    def status(self):
        corerl_status = self._corerl_service.status()
        coreio_status = self._coreio_service.status()

        statuses = [corerl_status, coreio_status]

        if any(s.state == "failed" for s in statuses):
            state = "failed"
        elif all(s.state == "running" for s in statuses):
            state = "running"
        elif all(s.state == "stopped" for s in statuses):
            state = "stopped"
        else:
            # This covers mixed states like starting/stopping
            state = "starting"

        return AgentStatus(
            id=self._id,
            state=state,
            config_path=self._config_path,
        )
