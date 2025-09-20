"""Configuration creation utilities for coredinator testing."""

from pathlib import Path


def create_test_configs(base_path: Path, config_names: list[str]) -> dict[str, Path]:
    """
    Create test configuration files.

    Args:
        base_path: Directory to create config files in
        config_names: List of config names to create

    Returns:
        Dictionary mapping config names to their file paths
    """
    configs = {}
    for name in config_names:
        config_path = base_path / f"{name}_config.yaml"
        if name in ["backwash", "coag"]:
            config_path.write_text(f"agent_type: {name}\nprocess_params: {{}}\n")
        else:
            config_path.write_text("dummy: true\n")
        configs[name] = config_path
    return configs
