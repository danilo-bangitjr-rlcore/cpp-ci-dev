
from click.testing import CliRunner

from corecli.coredinator import coredinator


def test_coredinator_help(runner: CliRunner) -> None:
    """Test coredinator command group help."""
    result = runner.invoke(coredinator, ["--help"])
    assert result.exit_code == 0
    assert "Coredinator management commands" in result.output


def test_status_command_exists(runner: CliRunner) -> None:
    """Test status command exists."""
    result = runner.invoke(coredinator, ["status", "--help"])
    assert result.exit_code == 0
    assert "health and status" in result.output


def test_start_command_exists(runner: CliRunner) -> None:
    """Test start command exists."""
    result = runner.invoke(coredinator, ["start", "--help"])
    assert result.exit_code == 0
    assert "Start the coredinator service" in result.output


def test_stop_command_exists(runner: CliRunner) -> None:
    """Test stop command exists."""
    result = runner.invoke(coredinator, ["stop", "--help"])
    assert result.exit_code == 0
    assert "stop the coredinator" in result.output
