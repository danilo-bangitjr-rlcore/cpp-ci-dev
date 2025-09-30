from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from lib_process.process import Process
from lib_utils.errors import fail_gracefully

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceState, ServiceStatus

log = get_logger(__name__)


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

        self._process: Process | None = None
        self._keep_alive_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

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
        self._mode = ServiceMode.STARTED
        self._stop_event.clear()  # Clear stop signal for new service instance
        if self.is_running():
            log.info("Service already running, skipping start", service_id=self.id)
            return

        log.info("Service preparing to launch process", service_id=self.id)
        exe = self._ensure_executable()
        cfg = self._ensure_config()

        args = self._build_args(exe, cfg)
        log.debug("Service command args", service_id=self.id, args_preview=args[:2])  # Log first two args for security

        self._process = Process.start_in_background(args)
        log.info("Service started process", service_id=self.id, pid=self._process.psutil.pid)
        self._keep_alive()

    @fail_gracefully()
    def stop(self, grace_seconds: float = 5.0) -> None:
        log.info("Stopping service", service_id=self.id)
        self._mode = ServiceMode.STOPPED
        self._stop_event.set()  # Signal the monitor thread to stop

        # Wait for background thread to finish if it's running
        thread = self._keep_alive_thread
        if thread is not None and thread.is_alive():
            # Check if we're trying to join the current thread (which would cause RuntimeError)

            current_thread = threading.current_thread()
            if thread is current_thread:
                log.debug("Service cannot join current thread, skipping wait", service_id=self.id)
                # Clear thread reference and let it exit naturally
                self._keep_alive_thread = None
            else:
                log.debug("Service waiting for background thread to finish", service_id=self.id)
                # Give the thread a moment to see the stop signal and exit
                thread.join(timeout=1.0)
                if thread.is_alive():
                    log.warning("Service background thread did not finish in time", service_id=self.id)
                else:
                    log.debug("Service background thread finished", service_id=self.id)

        if not self._process:
            return

        process_id = self._process.psutil.pid
        log.info("Service stopping process", service_id=self.id, pid=process_id)
        self._process.terminate_tree(timeout=grace_seconds)
        self._process = None
        log.info("Service process cleanup completed", service_id=self.id, pid=process_id)

    def restart(self):
        current_pid = self._process.psutil.pid if self._process else None
        log.info("Service restart requested", service_id=self.id, mode=self._mode, current_pid=current_pid)
        # Only restart if we're supposed to be running
        if self._mode != ServiceMode.STARTED:
            log.info("Service restart skipped - not in STARTED mode", service_id=self.id)
            return
        log.info("Service performing restart sequence", service_id=self.id, stopping_pid=current_pid)
        self.stop()
        self.start()
        new_pid = self._process.psutil.pid if self._process else None
        log.info("Service restart sequence completed", service_id=self.id, old_pid=current_pid, new_pid=new_pid)

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

        if not self._process.is_running() or self._process.is_zombie():
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
            return [self._process.psutil.pid]

        return [None]

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

        return self._process.is_running() and not self._process.is_zombie()

    def _build_args(self, exe: Path, cfg: Path) -> list[str]:
        return [str(exe), "--config-name", str(cfg)]

    # -------------
    # -- Private --
    # -------------
    def _get_current_pid(self) -> int | None:
        return self._process.psutil.pid if self._process else None

    def _monitor_exit(self) -> None:
        monitor_pid = self._get_current_pid()
        log.debug("Service monitor thread exiting", service_id=self.id, was_monitoring_pid=monitor_pid)
        self._keep_alive_thread = None

    def _monitor_service_stopped(self) -> None:
        monitor_pid = self._get_current_pid()
        log.info(
            "Service monitor thread stopping - service mode is STOPPED",
            service_id=self.id,
            was_monitoring_pid=monitor_pid,
        )
        self._keep_alive_thread = None

    def _monitor_service_recovered(self) -> None:
        current_pid = self._get_current_pid()
        log.info("Service recovered - process is running again", service_id=self.id, pid=current_pid)

    def _monitor_service_degraded(self) -> datetime:
        failed_pid = self._get_current_pid()
        log.warning(
            "Service entered degraded state - process not running",
            service_id=self.id,
            failed_pid=failed_pid,
        )
        return datetime.now()

    def _monitor_check_restart_needed(self, degraded_start: datetime) -> bool:
        elapsed = datetime.now() - degraded_start
        if elapsed >= self.config.degraded_wait:
            elapsed_seconds = elapsed.total_seconds()
            log.warning(
                "Service triggering restart after degraded state",
                service_id=self.id,
                elapsed_seconds=round(elapsed_seconds, 1),
            )
            self.restart()
            return True
        return False

    def _keep_alive(self):
        if self._keep_alive_thread is not None and self._keep_alive_thread.is_alive():
            log.debug("Service background monitor already running", service_id=self.id)
            return

        log.info("Service starting background monitor thread", service_id=self.id)

        def monitor():
            current_pid = self._get_current_pid()
            log.info("Service background monitor thread started", service_id=self.id, monitoring_pid=current_pid)
            degraded_start: datetime | None = None
            while True:
                # Wait for the heartbeat interval or until we receive a stop signal
                if self._stop_event.wait(timeout=self.config.heartbeat_interval.total_seconds()):
                    # Stop event was set - time to exit
                    self._monitor_exit()
                    return

                if self._mode == ServiceMode.STOPPED:
                    self._monitor_service_stopped()
                    return

                if self._mode == ServiceMode.STARTED and self.is_running():
                    if degraded_start is not None:
                        self._monitor_service_recovered()
                    degraded_start = None
                    continue

                if degraded_start is None:
                    degraded_start = self._monitor_service_degraded()

                if degraded_start is not None:
                    if self._monitor_check_restart_needed(degraded_start):
                        degraded_start = None

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        self._keep_alive_thread = t
