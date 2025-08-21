from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import Literal

import backoff

from coredinator.service.protocols import ServiceID, ServiceState

ServiceMode = Literal["started", "stopped"]


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    config_path: Path | None



@dataclass
class ServiceConfig:
    heartbeat_interval: timedelta = timedelta(seconds=5)
    degraded_wait: timedelta = timedelta(seconds=30)


class Service:
    def __init__(self, id: ServiceID, executable_path: Path, config_path: Path, config: ServiceConfig | None = None):
        self._mode: ServiceMode = "stopped"
        self.config = config if config is not None else ServiceConfig()

        self.id = id
        self._exe_path: Path = executable_path
        self._config_path: Path = config_path

        self._process: Popen | None = None
        self._keep_alive_thread: threading.Thread | None = None


    def _keep_alive(self):
        if self._keep_alive_thread is not None and self._keep_alive_thread.is_alive():
            return

        def monitor():
            degraded_start: datetime | None = None
            while True:
                time.sleep(self.config.heartbeat_interval.total_seconds())

                if self._mode == "stopped":
                    self._keep_alive_thread = None
                    return

                if self._mode == "running" and self.is_running():
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

    def is_running(self):
        return self._process is not None and self._process.poll() is None


    @backoff.on_exception(
        backoff.expo,
        (FileNotFoundError, PermissionError, OSError),
        max_time=10,
        jitter=backoff.full_jitter,
    )
    def start(self):
        self._mode = "started"
        if self.is_running():
            return

        exe = self._ensure_executable()
        cfg = self._ensure_config()
        cfg_name = cfg.stem

        args = [str(exe), "--config-name", cfg_name]
        self._process = Popen(args, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
        self._keep_alive()


    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=10,
        jitter=backoff.full_jitter,
    )
    def stop(self, grace_seconds: float = 5.0) -> None:
        self._mode = "stopped"
        if not self._process:
            return

        proc = self._process
        if proc.poll() is not None:
            self._process = None
            return

        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            # If sending SIGTERM fails (rare), fall back to kill
            try:
                proc.kill()
            except Exception:
                pass
            self._process = None
            return

        # Wait up to grace period
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._process = None
                return
            time.sleep(0.05)

        # Escalate
        try:
            proc.kill()
        finally:
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
        if code is None:
            return ServiceStatus(
                id=self.id,
                state=ServiceState.RUNNING,
                config_path=self._config_path,
            )

        return ServiceStatus(
            id=self.id,
            state=ServiceState.FAILED,
            config_path=self._config_path,
        )
