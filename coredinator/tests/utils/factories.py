from pathlib import Path

from coredinator.agent.agent_manager import AgentManager
from coredinator.service.service_manager import ServiceManager


def create_agent_manager(base_path: Path | str) -> AgentManager:
    """
    Create an AgentManager with a ServiceManager using the standard pattern.

    This factory eliminates the duplication of creating both AgentManager and
    ServiceManager with the same base_path parameter.
    """
    base_path = Path(base_path)
    service_manager = ServiceManager(base_path=base_path)
    return AgentManager(base_path=base_path, service_manager=service_manager)


def create_dummy_config(config_path: Path | str) -> Path:
    """
    Create a dummy configuration file for testing.
    """
    config_path = Path(config_path)
    config_path.write_text("dummy: true\n")
    return config_path


def create_agent_type_config(config_path: Path | str, agent_type: str) -> Path:
    """
    Create a configuration file for a specific agent type.
    """
    config_path = Path(config_path)
    config_path.write_text(f"agent_type: {agent_type}\nprocess_params: {{}}\n")
    return config_path


def create_multiple_dummy_configs(base_path: Path, count: int, prefix: str = "config") -> list[Path]:
    """
    Create multiple dummy configuration files.
    """
    configs = []
    for i in range(count):
        config_path = base_path / f"{prefix}{i + 1}.yaml"
        create_dummy_config(config_path)
        configs.append(config_path)
    return configs
