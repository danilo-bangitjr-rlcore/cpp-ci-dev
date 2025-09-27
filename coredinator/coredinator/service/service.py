from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import psutil
from lib_utils.errors import fail_gracefully

from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceState, ServiceStatus

IS_WINDOWS = sys.platform.startswith("win")


class ServiceMode(StrEnum):
    STARTED = "started"
    STOPPED = "stopped"



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

        self._process: psutil.Process | None = None
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

    @fail_gracefully()
    def start(self):
        self._mode = ServiceMode.STARTED
        if self.is_running():
            return

        exe = self._ensure_executable()
        cfg = self._ensure_config()

        args = self._build_args(exe, cfg)
        popen_kwargs: dict[str, Any] = {
            "stdin": DEVNULL,
            "stdout": DEVNULL,
            "stderr": DEVNULL,
            "start_new_session": True,  # Detach from parent process
        }

        if IS_WINDOWS:
            detached_process = getattr(subprocess, "DETACHED_PROCESS", 0)
            create_new_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            creationflags = detached_process | create_new_process_group | create_no_window
            if creationflags:
                popen_kwargs["creationflags"] = creationflags

        popen = Popen(args, **popen_kwargs)
        self._process = psutil.Process(popen.pid)
        self._keep_alive()


    @fail_gracefully()
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
        # Map internal mode to intended state
        intended_state = (
            ServiceIntendedState.RUNNING
            if self._mode == ServiceMode.STARTED
            else ServiceIntendedState.STOPPED
        )

        if self._process is None:
            return ServiceStatus(
                id=self.id,
                state=ServiceState.STOPPED,
                intended_state=intended_state,
                config_path=self._config_path,
            )

        if not self._process.is_running() or self._process.status() == psutil.STATUS_ZOMBIE:
            return ServiceStatus(
                id=self.id,
                state=ServiceState.FAILED,
                intended_state=intended_state,
                config_path=self._config_path,
            )

        state = ServiceState.RUNNING if self._is_healthy() else ServiceState.FAILED
        return ServiceStatus(
            id=self.id,
            state=state,
            intended_state=intended_state,
            config_path=self._config_path,
        )

    def get_process_ids(self) -> list[int | None]:
        """Get process ID of the main process for this service."""
        if self._process is not None:
            return [self._process.pid]

        return [None]

    def reattach_process(self, pid: int) -> bool:
        """Reattach to an existing process if it exists and is running.

        Returns True if successfully reattached, False otherwise.
        """
        try:
            proc = psutil.Process(pid)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                self._process = proc
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return False

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
        if not self.config.healthcheck_enabled:
            return True

        try:
            url = f"http://{self.config.host}:{self.config.port}/api/healthcheck"
            with urlopen(url, timeout=self.config.healthcheck_timeout.total_seconds()) as response:
                return response.status == 200
        except (URLError, OSError):
            return False

    def _is_process_running(self) -> bool:
        if self._process is None:
            return False

        try:
            if not self._process.is_running():
                return False

            status = self._process.status()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        # Check for terminated states based on platform
        if IS_WINDOWS and status == psutil.STATUS_DEAD:
            return False
        if status == psutil.STATUS_ZOMBIE:
            return False

        return True

    def _build_args(self, exe: Path, cfg: Path) -> list[str]:
        return [str(exe), "--config-name", str(cfg)]


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


def _terminate_process_gracefully(process: psutil.Process, grace_seconds: float) -> None:
    try:
        process.terminate()
        process.wait(timeout=grace_seconds)
    except TimeoutExpired:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


@fail_gracefully()
def stop_process(proc: psutil.Process, grace_seconds: float = 5.0) -> None:
    for child in proc.children(recursive=True):
        _terminate_process_gracefully(child, grace_seconds)

    _terminate_process_gracefully(proc, grace_seconds)
