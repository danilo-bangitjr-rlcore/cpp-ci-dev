from pathlib import Path

from corecli.utils.coredinator import is_coredinator_running
from tests.utils.cli import CliRunner
from tests.utils.waiting import wait_for_event


class TestCoredinatorE2EWorkflow:
    def test_start_status_stop_workflow(self, running_coredinator: int, corecli_runner: CliRunner) -> None:
        """
        Test complete workflow: start coredinator, check status, stop it.
        """
        port = running_coredinator

        status_result = corecli_runner.status_coredinator(port)
        assert status_result.returncode == 0, f"Status command failed: {status_result.stderr}"
        assert "healthy" in status_result.stdout, "Status should show healthy"

    def test_start_already_running_error(
        self,
        running_coredinator: int,
        corecli_runner: CliRunner,
        coredinator_log_file: Path,
        coredinator_base_path: Path,
    ) -> None:
        """
        Test that starting coredinator when already running raises appropriate error.
        """
        port = running_coredinator

        start_again_result = corecli_runner.start_coredinator(
            port,
            coredinator_log_file,
            coredinator_base_path,
            check=False,
        )
        assert start_again_result.returncode != 0, "Second start should fail when already running"
        error_output = start_again_result.stdout + start_again_result.stderr
        assert "already running" in error_output, (
            f"Error should mention already running. Output: {error_output}"
        )

    def test_stop_when_not_running_succeeds(self, free_port: int, corecli_runner: CliRunner) -> None:
        """
        Test that stopping coredinator when not running succeeds gracefully.
        """
        assert not is_coredinator_running(free_port)

        stop_result = corecli_runner.stop_coredinator(free_port)
        assert stop_result.returncode == 0, "Stop should succeed even when not running"

    def test_multiple_start_stop_cycles(
        self,
        free_port: int,
        corecli_runner: CliRunner,
        temp_log_dir: Path,
        coredinator_base_path: Path,
    ) -> None:
        """
        Test that multiple start/stop cycles work reliably.
        """
        for cycle in range(3):
            log_file = temp_log_dir / f"coredinator_cycle_{cycle}.log"

            start_result = corecli_runner.start_coredinator(free_port, log_file, coredinator_base_path)
            assert start_result.returncode == 0, f"Start failed on cycle {cycle}"

            wait_for_event(
                lambda: is_coredinator_running(free_port),
                timeout=30.0,
                description=f"coredinator to start on cycle {cycle}",
            )

            stop_result = corecli_runner.stop_coredinator(free_port)
            assert stop_result.returncode == 0, f"Stop failed on cycle {cycle}"

            wait_for_event(
                lambda: not is_coredinator_running(free_port),
                timeout=10.0,
                description=f"coredinator to stop on cycle {cycle}",
            )

            assert not is_coredinator_running(free_port)
