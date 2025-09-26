from click.testing import CliRunner

from corecli.dev import dev


def test_dev_help(runner: CliRunner) -> None:
    """Test dev command group help."""
    result = runner.invoke(dev, ["--help"])
    assert result.exit_code == 0
    assert "Development workflow commands" in result.output


def test_start_sim_command_exists(runner: CliRunner) -> None:
    """Test start-sim command exists."""
    result = runner.invoke(dev, ["start-sim", "--help"])
    assert result.exit_code == 0
    assert "Start development simulation" in result.output


def test_stop_sim_command_exists(runner: CliRunner) -> None:
    """Test stop-sim command exists."""
    result = runner.invoke(dev, ["stop-sim", "--help"])
    assert result.exit_code == 0
    assert "shut down the simulation" in result.output


def test_logs_command_exists(runner: CliRunner) -> None:
    """Test logs command exists."""
    result = runner.invoke(dev, ["logs", "--help"])
    assert result.exit_code == 0
    assert "View logs from simulation" in result.output
