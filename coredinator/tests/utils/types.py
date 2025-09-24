"""Type definitions for coredinator test utilities."""

from typing import Any, Literal, Protocol

# Type aliases for better code readability
AgentState = Literal["running", "stopped", "failed", "starting"]


# Type protocols for test client interfaces
class TestClient(Protocol):
    """Protocol for FastAPI TestClient interface."""
    def get(self, url: str) -> Any: ...


class AgentManager(Protocol):
    """Protocol for AgentManager interface."""
    def get_agent_status(self, agent_id: Any) -> Any: ...


class Service(Protocol):
    """Protocol for Service interface."""
    def status(self) -> Any: ...
