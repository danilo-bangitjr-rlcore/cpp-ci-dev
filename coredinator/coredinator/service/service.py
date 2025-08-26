from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, Popen

import backoff

from coredinator.service.protocols import ServiceID, ServiceState


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    id: ServiceID
    state: ServiceState
    config_path: Path | None


class Service:
    def __init__(self, id: ServiceID, executable_path: Path, config_path: Path):
        self.id = id
        self._process: Popen | None = None
        self._exe_path: Path = executable_path
        self._config_path: Path = config_path

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
        if self.is_running():
            return

        exe = self._ensure_executable()
        cfg = self._ensure_config()
        cfg_name = cfg.stem

        args = [str(exe), "--config-name", cfg_name]
        self._process = Popen(args, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=10,
        jitter=backoff.full_jitter,
    )
    def stop(self, grace_seconds: float = 5.0) -> None:
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
