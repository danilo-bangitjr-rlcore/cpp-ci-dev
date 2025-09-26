from click.testing import CliRunner

from corecli.agent import agent


def test_agent_help(runner: CliRunner) -> None:
    """Test agent command group help."""
    result = runner.invoke(agent, ["--help"])
    assert result.exit_code == 0
    assert "Agent management commands" in result.output


def test_start_command_exists(runner: CliRunner) -> None:
    """Test start command exists."""
    result = runner.invoke(agent, ["start", "--help"])
    assert result.exit_code == 0
    assert "Start a new RL agent" in result.output


def test_stop_command_exists(runner: CliRunner) -> None:
    """Test stop command exists."""
    result = runner.invoke(agent, ["stop", "--help"])
    assert result.exit_code == 0
    assert "Stop a running agent" in result.output


def test_status_command_exists(runner: CliRunner) -> None:
    """Test status command exists."""
    result = runner.invoke(agent, ["status", "--help"])
    assert result.exit_code == 0
    assert "detailed status information" in result.output


def test_list_command_exists(runner: CliRunner) -> None:
    """Test list command exists."""
    result = runner.invoke(agent, ["list", "--help"])
    assert result.exit_code == 0
    assert "List all agents known to coredinator" in result.output


def test_logs_command_exists(runner: CliRunner) -> None:
    """Test logs command exists."""
    result = runner.invoke(agent, ["logs", "--help"])
    assert result.exit_code == 0
    assert "View logs from a specific agent" in result.output
