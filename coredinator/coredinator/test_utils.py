"""Test utilities for coredinator testing."""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class CoredinatorService:
    """Information about a running coredinator service."""

    base_url: str
    process_id: int
    command: list[str]
    env: dict[str, str]
    cwd: Path


def wait_for_service_healthy(
    base_url: str,
    max_attempts: int = 30,
    sleep_interval: float = 0.1,
    process: subprocess.Popen | None = None,
) -> None:
    for _ in range(max_attempts):
        time.sleep(sleep_interval)
        try:
            response = requests.get(f"{base_url}/api/healthcheck", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            ...

    # Service failed to become healthy
    error_msg = f"Service at {base_url} failed to become healthy after {max_attempts} attempts"

    if process is not None:
        # Get detailed error information from the process
        try:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            error_msg = (
                f"Coredinator service failed to start on {base_url}\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            process.kill()
            error_msg += "\nProcess did not terminate gracefully and was killed"
        except Exception as e:
            error_msg += f"\nError getting process output: {e}"

    raise RuntimeError(error_msg) from None
