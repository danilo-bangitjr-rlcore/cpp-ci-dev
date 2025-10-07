from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from lib_process.process import Process
from lib_utils.errors import fail_gracefully

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceState, ServiceStatus
from coredinator.service.service_monitor import ServiceMonitor
from coredinator.utils.healthcheck import check_http_health

log = get_logger(__name__)


@dataclass
class ServiceConfig:
    heartbeat_interval: timedelta = timedelta(seconds=5)
    degraded_wait: timedelta = timedelta(seconds=30)
    host: str = "127.0.0.1"
    port: int = 8080
    healthcheck_timeout: timedelta = timedelta(seconds=3)
    healthcheck_enabled: bool = False


class Service(ABC):
    def __init__(self, id: ServiceID, base_path: Path, config_path: Path, config: ServiceConfig | None = None):
        self._intended_state = ServiceIntendedState.STOPPED
        self.config = config if config is not None else ServiceConfig()

        self.id = id
        self._base_path: Path = base_path
        self._config_path: Path = config_path

        self._process: Process | None = None
        self._monitor: ServiceMonitor | None = None
        self._failed: bool = False
        self._version: str | None = None

    @abstractmethod
    def _find_executable(self) -> Path:
        pass


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
        log.info("Service start requested", service_id=self.id)
        self._intended_state = ServiceIntendedState.RUNNING
        self._failed = False
        if self.is_running():
            log.info("Service already running, skipping start", service_id=self.id)
            return

        try:
            exe_path = self._find_executable()
            self._extract_version(exe_path)
            exe = self._ensure_executable(exe_path)
            cfg = self._ensure_config()

            args = self._build_args(exe, cfg)
            log.debug("Service command args", service_id=self.id, args_preview=args[:2])

            self._process = Process.start_in_background(args)
            log.info("Service started process", service_id=self.id, pid=self._process.psutil.pid, version=self._version)

        except Exception:
            log.exception("Service failed to start", service_id=self.id)
            self._failed = True
            raise

        # Start background monitoring
        if self._monitor is None:
            self._monitor = ServiceMonitor(self, self.config)
        self._monitor.start_monitoring()

    @fail_gracefully()
    def stop(self, grace_seconds: float = 5.0) -> None:
        log.info("Stopping service", service_id=self.id)
        self._intended_state = ServiceIntendedState.STOPPED
        self._failed = False

        # Stop monitoring
        if self._monitor is not None:
            self._monitor.stop_monitoring()

        if not self._process:
            return

        process_id = self._process.psutil.pid
        log.info("Service stopping process", service_id=self.id, pid=process_id)
        self._process.terminate_tree(timeout=grace_seconds)
        self._process = None
        log.info("Service process cleanup completed", service_id=self.id, pid=process_id)

    def restart(self):
        current_pid = self._process.psutil.pid if self._process else None
        log.info(
            "Service restart requested",
            service_id=self.id,
            intended_state=self._intended_state,
            current_pid=current_pid,
        )
        # Only restart if we're supposed to be running
        if self._intended_state != ServiceIntendedState.RUNNING:
            log.info("Service restart skipped - not in RUNNING state", service_id=self.id)
            return
        log.info("Service performing restart sequence", service_id=self.id, stopping_pid=current_pid)
        self.stop()
        self.start()
        new_pid = self._process.psutil.pid if self._process else None
        log.info("Service restart sequence completed", service_id=self.id, old_pid=current_pid, new_pid=new_pid)

    def status(self):
        if self._process is None:
            state = ServiceState.FAILED if self._failed else ServiceState.STOPPED
            return ServiceStatus(
                id=self.id,
                state=state,
                intended_state=self._intended_state,
                config_path=self._config_path,
            )

        if not self._process.is_running() or self._process.is_zombie():
            return ServiceStatus(
                id=self.id,
                state=ServiceState.FAILED,
                intended_state=self._intended_state,
                config_path=self._config_path,
            )

        state = ServiceState.RUNNING if self._is_healthy() else ServiceState.FAILED
        return ServiceStatus(
            id=self.id,
            state=state,
            intended_state=self._intended_state,
            config_path=self._config_path,
        )

    def get_pid(self) -> int | None:
        """Get process ID of the service, or None if not running."""
        if self._process is not None:
            return self._process.psutil.pid
        return None

    def get_process_ids(self) -> list[int | None]:
        """Get process ID of the main process for this service."""
        return [self.get_pid()]

    def reattach_process(self, pid: int) -> bool:
        """
        Reattach to an existing process if it exists and is running.

        Returns True if successfully reattached, False otherwise.
        """
        proc = Process.from_pid(pid)
        if proc.is_running() and not proc.is_zombie():
            self._process = proc
            return True

        return False

    def get_version(self) -> str | None:
        """Get the version of the service executable, or None if not started."""
        return self._version

    # -----------------
    # -- Validations --
    # -----------------
    def _ensure_executable(self, exe_path: Path):
        if not exe_path.exists():
            raise FileNotFoundError(f"Service executable not found at {exe_path}")
        if not os.access(exe_path, os.X_OK):
            raise PermissionError(f"Service executable is not executable: {exe_path}")

        return exe_path

    def _ensure_config(self):
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self._config_path}")

        return self._config_path

    def _extract_version(self, exe_path: Path) -> None:
        """Extract version string from executable filename."""
        from coredinator.utils.semver import parse_version_from_filename

        version_obj = parse_version_from_filename(exe_path.name)
        if version_obj:
            self._version = f"{version_obj.major}.{version_obj.minor}.{version_obj.patch}"
        else:
            self._version = None

    def _is_healthy(self) -> bool:
        if not self.config.healthcheck_enabled:
            return True

        return check_http_health(
            self.config.host,
            self.config.port,
            self.config.healthcheck_timeout,
        )

    def _is_process_running(self) -> bool:
        if self._process is None:
            return False

        return self._process.is_running() and not self._process.is_zombie()

    def _build_args(self, exe: Path, cfg: Path) -> list[str]:
        return [str(exe), "--config-name", str(cfg)]
