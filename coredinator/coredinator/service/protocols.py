from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NewType, Protocol

ServiceID = NewType("ServiceID", str)


class ServiceState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


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

    def start(self) -> None:
        ...

    def stop(self, grace_seconds: float = 5.0) -> None:
        ...

    def restart(self) -> None:
        ...

    def status(self) -> object:
        ...

    def get_process_ids(self) -> list[int | None]:
        """Get process IDs of all running processes for this service."""
        ...

    def reattach_process(self, pid: int) -> bool:
        """Reattach to an existing process if it exists and is running.

        Returns True if successfully reattached, False otherwise.
        """
        ...


@dataclass(frozen=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    config_path: Path | None
