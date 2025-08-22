from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from subprocess import DEVNULL, Popen
from urllib.error import URLError
from urllib.request import urlopen

from coredinator.service.protocols import ServiceID, ServiceState


class ServiceMode(StrEnum):
    STARTED = "started"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    config_path: Path | None



@dataclass
class ServiceConfig:
    heartbeat_interval: timedelta = timedelta(seconds=5)
    degraded_wait: timedelta = timedelta(seconds=30)
    host: str = "127.0.0.1"
    port: int = 8080
    healthcheck_timeout: timedelta = timedelta(seconds=3)
    healthcheck_enabled: bool = False


class Service:
    def __init__(self, id: ServiceID, executable_path: Path, config_path: Path, config: ServiceConfig | None = None):
        self._mode = ServiceMode.STOPPED
        self.config = config if config is not None else ServiceConfig()

        self.id = id
        self._exe_path: Path = executable_path
        self._config_path: Path = config_path

        self._process: Popen | None = None
        self._keep_alive_thread: threading.Thread | None = None


    # ------------
    # -- Public --
    # ------------
    def is_running(self):
        if not self._is_process_running():
            return False

        if self.config.healthcheck_enabled:
            return self._is_healthy()

        return True

    def start(self):
        self._mode = ServiceMode.STARTED
        if self.is_running():
            return

        exe = self._ensure_executable()
        cfg = self._ensure_config()
        cfg_name = cfg.stem

        args = [str(exe), "--config-name", cfg_name]
        self._process = Popen(args, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
        self._keep_alive()


    def stop(self, grace_seconds: float = 5.0) -> None:
        self._mode = ServiceMode.STOPPED
        if not self._process:
            return

        stop_process(self._process, grace_seconds)
        self._process = None


    def restart(self):
        self.stop()
        self.start()

    def status(self):
        if self._process is None:
            return ServiceStatus(
                id=self.id,
                state=ServiceState.STOPPED,
                config_path=self._config_path,
            )

        code = self._process.poll()
        if code is not None:
            # Process has exited - determine if it was a clean stop or failure
            if code == 0 and self._mode == ServiceMode.STOPPED:
                state = ServiceState.STOPPED
            else:
                state = ServiceState.FAILED
            return ServiceStatus(
                id=self.id,
                state=state,
                config_path=self._config_path,
            )

        state = ServiceState.RUNNING if self._is_healthy() else ServiceState.FAILED
        return ServiceStatus(
            id=self.id,
            state=state,
            config_path=self._config_path,
        )

    # -----------------
    # -- Validations --
    # -----------------
    def _ensure_executable(self):
        if not self._exe_path.exists():
            raise FileNotFoundError(f"Service executable not found at {self._exe_path}")
        if not os.access(self._exe_path, os.X_OK):
            raise PermissionError(f"Service executable is not executable: {self._exe_path}")

        return self._exe_path

    def _ensure_config(self):
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self._config_path}")

        return self._config_path

    def _is_healthy(self) -> bool:
        """Perform HTTP healthcheck on the service's /api/healthcheck endpoint."""
        if not self.config.healthcheck_enabled:
            return True

        try:
            url = f"http://{self.config.host}:{self.config.port}/api/healthcheck"
            with urlopen(url, timeout=self.config.healthcheck_timeout.total_seconds()) as response:
                return response.status == 200
        except (URLError, OSError):
            return False

    def _is_process_running(self) -> bool:
        """Check if the service process is running without performing healthcheck."""
        return self._process is not None and self._process.poll() is None

    # -------------
    # -- Private --
    # -------------
    def _keep_alive(self):
        if self._keep_alive_thread is not None and self._keep_alive_thread.is_alive():
            return

        def monitor():
            degraded_start: datetime | None = None
            while True:
                time.sleep(self.config.heartbeat_interval.total_seconds())

                if self._mode == ServiceMode.STOPPED:
                    self._keep_alive_thread = None
                    return

                if self._mode == ServiceMode.STARTED and self.is_running():
                    degraded_start = None
                    continue

                if degraded_start is None:
                    degraded_start = datetime.now()

                if degraded_start is not None:
                    elapsed = datetime.now() - degraded_start
                    if elapsed >= self.config.degraded_wait:
                        self.restart()
                        degraded_start = None

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        self._keep_alive_thread = t


def stop_process(proc: Popen, grace_seconds: float) -> None:
    """
    Attempt to gracefully stop a Popen process, escalating to kill if needed.
    """
    if proc.poll() is not None:
        return

    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        proc.kill()
    finally:
        pass
