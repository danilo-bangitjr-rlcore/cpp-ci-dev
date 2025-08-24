from unittest.mock import Mock, patch

from click.testing import CliRunner
from lib_utils.maybe import Maybe

from corecli.coredinator import coredinator
from corecli.utils.coredinator import HealthModel


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


class TestCoredinatorStartCommand:
    """Test coredinator start command with fakes."""

    @patch("corecli.coredinator.start.start_coredinator")
    @patch("corecli.coredinator.start.wait_for_coredinator_start")
    def test_start_command_success(
        self,
        mock_wait: Mock,
        mock_start: Mock,
        runner: CliRunner,
    ) -> None:
        """Test successful start command."""
        mock_start.return_value = 12345
        mock_wait.return_value = True

        result = runner.invoke(coredinator, ["start", "--port", "8080"])

        assert result.exit_code == 0
        assert "started with PID 12345" in result.output
        assert "ready on port 8080" in result.output
        mock_start.assert_called_once()
        mock_wait.assert_called_once()

    @patch("corecli.coredinator.start.start_coredinator")
    def test_start_command_already_running(
        self,
        mock_start: Mock,
        runner: CliRunner,
    ) -> None:
        """Test start command when already running."""
        mock_start.side_effect = RuntimeError("already running")

        result = runner.invoke(coredinator, ["start"])

        assert result.exit_code == 1
        assert "already running" in result.output


class TestCoredinatorStopCommand:
    """Test coredinator stop command with fakes."""

    @patch("corecli.coredinator.stop.stop_coredinator")
    @patch("corecli.coredinator.stop.wait_for_coredinator_stop")
    def test_stop_command_success(
        self,
        mock_wait: Mock,
        mock_stop: Mock,
        runner: CliRunner,
    ) -> None:
        """Test successful stop command."""
        mock_stop.return_value = True
        mock_wait.return_value = True

        result = runner.invoke(coredinator, ["stop"])

        assert result.exit_code == 0
        assert "stopped successfully" in result.output
        mock_stop.assert_called_once()
        mock_wait.assert_called_once()

    @patch("corecli.coredinator.stop.stop_coredinator")
    def test_stop_command_failure(
        self,
        mock_stop: Mock,
        runner: CliRunner,
    ) -> None:
        """Test stop command failure."""
        mock_stop.return_value = False

        result = runner.invoke(coredinator, ["stop"])

        assert result.exit_code == 1
        assert "Failed to stop" in result.output


class TestCoredinatorStatusCommand:
    """Test coredinator status command with fakes."""

    @patch("corecli.coredinator.status._check_healthcheck")
    def test_status_command_running(
        self,
        mock_healthcheck: Mock,
        runner: CliRunner,
    ) -> None:
        """Test status command when coredinator is running."""
        mock_healthcheck.return_value = Maybe(
            HealthModel(
                status="healthy",
                process_id=1234,
                service="coredinator",
                version="0.0.0",
            ),
        )

        result = runner.invoke(coredinator, ["status"])

        assert result.exit_code == 0
        assert "healthy on port 8000" in result.output
        assert "Process ID: 1234" in result.output

    @patch("corecli.coredinator.status._check_healthcheck")
    def test_status_command_not_running(
        self,
        mock_healthcheck: Mock,
        runner: CliRunner,
    ) -> None:
        """Test status command when coredinator is not running."""
        mock_healthcheck.return_value = Maybe(None)

        result = runner.invoke(coredinator, ["status"])

        assert result.exit_code == 0
        assert "No coredinator responding" in result.output
