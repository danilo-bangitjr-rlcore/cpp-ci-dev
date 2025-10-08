from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NewType, Protocol

ServiceID = NewType("ServiceID", str)


class ServiceIntendedState(StrEnum):
    """Desired operational state of a service."""
    RUNNING = "running"
    STOPPED = "stopped"


class ServiceState(StrEnum):
    """Current observed state of a service."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    intended_state: ServiceIntendedState
    config_path: Path | None


class StatusLike(Protocol):
    """Minimal status contract shared by services and service bundles."""

    @property
    def id(self) -> ServiceID:
        ...

    @property
    def state(self) -> ServiceState:
        ...

    @property
    def config_path(self) -> Path | None:
        ...


class ServiceLike(Protocol):
    """Minimal lifecycle and inspection API for a managed service."""

    @property
    def id(self) -> ServiceID:
        ...

    def start(self) -> object:
        ...

    def stop(self, grace_seconds: float = 5.0) -> object:
        ...

    def restart(self) -> None:
        ...

    def status(self) -> ServiceStatus:
        ...

    def get_pid(self) -> int | None:
        """Get process ID of the service, or None if not running."""
        ...

    def get_process_ids(self) -> list[int | None]:
        """Get process IDs of all running processes for this service."""
        ...

    def reattach_process(self, pid: int) -> bool:
        """Reattach to an existing process if it exists and is running.

        Returns True if successfully reattached, False otherwise.
        """
        ...

    def get_version(self) -> str | None:
        """Get the version of the service executable, or None if not started."""
        ...
