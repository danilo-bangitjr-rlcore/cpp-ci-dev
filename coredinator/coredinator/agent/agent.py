from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NewType

from coredinator.service.protocols import ServiceID, ServiceState
from coredinator.service.service import ServiceStatus
from coredinator.service.service_manager import ServiceManager
from coredinator.services import CoreIOService, CoreRLService

AgentID = NewType("AgentID", str)


@dataclass(frozen=True)
class AgentStatus:
    id: AgentID
    state: ServiceState
    config_path: Path | None
    service_statuses: dict[str, ServiceStatus] = field(default_factory=dict)


class Agent:
    def __init__(self, id: AgentID, config_path: Path, base_path: Path, service_manager: ServiceManager):
        self._id = id
        self._config_path = config_path
        self._service_manager = service_manager

        # Create service IDs
        self._corerl_service_id = ServiceID(f"{id}-corerl")
        self._coreio_service_id = ServiceID(f"{id}-coreio")

        # Get or create services using the service manager
        self._corerl_service = self._service_manager.get_or_register_service(
            self._corerl_service_id,
            lambda: CoreRLService(
                id=self._corerl_service_id,
                config_path=config_path,
                base_path=base_path,
            ),
        )
        self._coreio_service = self._service_manager.get_or_register_service(
            self._coreio_service_id,
            lambda: CoreIOService(
                id=self._coreio_service_id,
                config_path=config_path,
                base_path=base_path,
            ),
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


    def status(self):
        corerl_status = self._corerl_service.status()
        coreio_status = self._coreio_service.status()
        statuses = [corerl_status, coreio_status]
        state = self._get_joint_status([s.state for s in statuses])

        service_statuses = {
            "corerl": corerl_status,
            "coreio": coreio_status,
        }

        return AgentStatus(
            id=self._id,
            state=state,
            config_path=self._config_path,
            service_statuses=service_statuses,
        )

    def _get_joint_status(self, service_statuses: list[ServiceState]) -> ServiceState:
        if any(s == ServiceState.FAILED for s in service_statuses):
            return ServiceState.FAILED

        if all(s == ServiceState.RUNNING for s in service_statuses):
            return ServiceState.RUNNING

        if all(s == ServiceState.STOPPED for s in service_statuses):
            return ServiceState.STOPPED

        return ServiceState.STARTING

    def get_process_ids(self) -> list[int | None]:
        """Get process IDs as a 2-element list: [corerl_pid, coreio_pid]."""
        corerl_pids = self._corerl_service.get_process_ids()
        coreio_pids = self._coreio_service.get_process_ids()

        # Each service returns exactly one element (int or None)
        return [corerl_pids[0], coreio_pids[0]]



    def reattach_processes(self, corerl_pid: int | None, coreio_pid: int | None) -> tuple[bool, bool]:
        """Reattach to existing CoreRL and CoreIO processes.

        Returns tuple of (corerl_success, coreio_success).
        """
        corerl_success = False
        coreio_success = False

        if corerl_pid is not None:
            corerl_success = self._corerl_service.reattach_process(corerl_pid)

        if coreio_pid is not None:
            coreio_success = self._coreio_service.reattach_process(coreio_pid)

        return corerl_success, coreio_success
