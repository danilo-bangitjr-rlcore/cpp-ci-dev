from click.testing import CliRunner

from corecli.main import cli


def test_cli_help(runner: CliRunner) -> None:
    """Test that CLI shows help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CoreCLI - Command line tools" in result.output


def test_version_command(runner: CliRunner) -> None:
    """Test version command."""
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_dev_group_exists(runner: CliRunner) -> None:
    """Test that dev command group exists."""
    result = runner.invoke(cli, ["dev", "--help"])
    assert result.exit_code == 0
    assert "Development workflow commands" in result.output


def test_coredinator_group_exists(runner: CliRunner) -> None:
    """Test that coredinator command group exists."""
    result = runner.invoke(cli, ["coredinator", "--help"])
    assert result.exit_code == 0
    assert "Coredinator management commands" in result.output


def test_agent_group_exists(runner: CliRunner) -> None:
    """Test that agent command group exists."""
    result = runner.invoke(cli, ["agent", "--help"])
    assert result.exit_code == 0
    assert "Agent management commands" in result.output


def test_config_group_exists(runner: CliRunner) -> None:
    """Test that config command group exists."""
    result = runner.invoke(cli, ["config", "--help"])
    assert result.exit_code == 0
    assert "Configuration management commands" in result.output


def test_global_options(runner: CliRunner) -> None:
    """Test global options are available."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output
    assert "--quiet" in result.output
    assert "--config" in result.output
