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


@dataclass(frozen=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    config_path: Path | None
