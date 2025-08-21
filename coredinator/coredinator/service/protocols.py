from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from coredinator.agent.agent_process import AgentID as ServiceID

ServiceState = Literal["starting", "running", "stopping", "stopped", "failed"]


class StatusLike(Protocol):
    """Minimal status contract shared by services and service bundles."""

    id: ServiceID
    state: ServiceState
    config_path: Path | None


class ServiceLike(Protocol):
    """Minimal lifecycle and inspection API for a managed service."""

    id: ServiceID

    def start(self) -> None:
        ...

    def stop(self, grace_seconds: float = 5.0) -> None:
        ...

    def restart(self) -> None:
        ...

    def status(self) -> StatusLike:
        ...
