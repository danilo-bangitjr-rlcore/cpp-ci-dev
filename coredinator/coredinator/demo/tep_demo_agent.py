from pathlib import Path

from coredinator.agent.agent import Agent, AgentID, AgentStatus
from coredinator.service.service import ServiceID
from coredinator.services.demos.tep import TEPService
from coredinator.services.demos.uaserver import UAServer


class TEPDemoAgent(Agent):
    def __init__(self, id: AgentID, config_path: Path, base_path: Path):
        super().__init__(id=id, config_path=config_path, base_path=base_path)
        self._tep_service = TEPService(
            id=ServiceID(f"{id}-tep"),
            config_path=config_path,
            base_path=base_path,
        )

        self._uaserver_service = UAServer(
            id=ServiceID(f"{id}-uaserver"),
            config_path=config_path,
            base_path=base_path,
        )

    def start(self):
        self._uaserver_service.start()
        self._tep_service.start()
        super().start()

    def stop(self, grace_seconds: float = 5.0):
        super().stop(grace_seconds)
        self._tep_service.stop(grace_seconds)
        self._uaserver_service.stop(grace_seconds)

    def status(self):
        corerl_status = self._corerl_service.status()
        coreio_status = self._coreio_service.status()
        tep_status = self._tep_service.status()
        uaserver_status = self._uaserver_service.status()

        statuses = [corerl_status, coreio_status, tep_status, uaserver_status]
        state = self._get_joint_status([s.state for s in statuses])

        service_statuses = {
            "corerl": corerl_status,
            "coreio": coreio_status,
            "tep": tep_status,
            "uaserver": uaserver_status,
        }

        return AgentStatus(
            id=self._id,
            state=state,
            config_path=self._config_path,
            service_statuses=service_statuses,
        )

    def get_process_ids(self) -> list[int | None]:
        return (
            super().get_process_ids()
            + self._tep_service.get_process_ids()
            + self._uaserver_service.get_process_ids()
        )
