from click.testing import CliRunner

from corecli.config import config


def test_config_help(runner: CliRunner) -> None:
    """Test config command group help."""
    result = runner.invoke(config, ["--help"])
    assert result.exit_code == 0
    assert "Configuration management commands" in result.output


def test_validate_command_exists(runner: CliRunner) -> None:
    """Test validate command exists."""
    result = runner.invoke(config, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Validate a configuration file" in result.output
