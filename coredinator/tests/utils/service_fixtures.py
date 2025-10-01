"""Service management and fixtures for coredinator testing."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from tests.utils.timeout_multiplier import apply_timeout_multiplier


@dataclass
class CoredinatorService:
    """Information about a running coredinator service."""

    base_url: str
    process_id: int
    command: list[str]
    env: dict[str, str]
    cwd: Path
    log_file: Path | None = None


def wait_for_service_healthy(
    base_url: str,
    max_attempts: int = 30,
    sleep_interval: float = 0.1,
    process: subprocess.Popen | None = None,
    log_file: Path | None = None,
) -> None:
    """
    Wait for coredinator service to become healthy.

    Args:
        base_url: Service base URL
        max_attempts: Maximum number of health check attempts (will be adjusted for platform)
        sleep_interval: Time between health check attempts
        process: Optional process object for error reporting
        log_file: Optional path to service log file for diagnostics

    Raises:
        RuntimeError: If service fails to become healthy
    """
    # Adjust max attempts for platform reliability, but keep individual HTTP timeouts short
    adjusted_max_attempts = int(apply_timeout_multiplier(max_attempts))
    for _ in range(adjusted_max_attempts):
        time.sleep(sleep_interval)
        try:
            # Keep HTTP timeout short - the reliability comes from more attempts
            response = requests.get(f"{base_url}/api/healthcheck", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            ...

    # Service failed to become healthy
    error_msg = f"Service at {base_url} failed to become healthy after {adjusted_max_attempts} attempts"

    if process is not None:
        # Get detailed error information from the process
        try:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            error_msg = (
                f"Coredinator service failed to start on {base_url}\n"
                f"stdout: {stdout.decode() if stdout else ''}\n"
                f"stderr: {stderr.decode() if stderr else ''}"
            )
        except subprocess.TimeoutExpired:
            process.kill()
            error_msg += "\nProcess did not terminate gracefully and was killed"
        except Exception as e:
            error_msg += f"\nError getting process output: {e}"

    if log_file is not None:
        error_msg += f"\nSee log file for details: {log_file}"

    raise RuntimeError(error_msg) from None
