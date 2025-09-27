from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import psutil
from lib_utils.errors import fail_gracefully

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceID, ServiceIntendedState, ServiceState, ServiceStatus
from coredinator.utils.process import safe_get_process_status, safe_is_process_running, terminate_process_tree

log = get_logger(__name__)

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
        log.info("Service started process", service_id=self.id, pid=popen.pid)
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

        process_id = self._process.pid
        log.info("Service stopping process", service_id=self.id, pid=process_id)
        stop_process(self._process, grace_seconds)
        self._process = None
        log.info("Service process cleanup completed", service_id=self.id, pid=process_id)

    def restart(self):
        current_pid = self._process.pid if self._process else "None"
        log.info("Service restart requested", service_id=self.id, mode=self._mode, current_pid=current_pid)
        # Only restart if we're supposed to be running
        if self._mode != ServiceMode.STARTED:
            log.info("Service restart skipped - not in STARTED mode", service_id=self.id)
            return
        log.info("Service performing restart sequence", service_id=self.id, stopping_pid=current_pid)
        self.stop()
        self.start()
        new_pid = self._process.pid if self._process else "None"
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

        if not safe_is_process_running(self._process):
            return ServiceStatus(
                id=self.id,
                state=ServiceState.FAILED,
                intended_state=intended_state,
                config_path=self._config_path,
            )

        status = safe_get_process_status(self._process)
        if status == psutil.STATUS_ZOMBIE:
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
        """
        Reattach to an existing process if it exists and is running.

        Returns True if successfully reattached, False otherwise.
        """
        try:
            proc = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        if not safe_is_process_running(proc):
            return False

        status = safe_get_process_status(proc)
        if status is None or status == psutil.STATUS_ZOMBIE:
            return False

        self._process = proc
        return True

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

        if not safe_is_process_running(self._process):
            return False

        status = safe_get_process_status(self._process)
        if status is None:
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
            log.debug("Service background monitor already running", service_id=self.id)
            return

        log.info("Service starting background monitor thread", service_id=self.id)

        def monitor():
            current_pid = self._process.pid if self._process else "None"
            log.info("Service background monitor thread started", service_id=self.id, monitoring_pid=current_pid)
            degraded_start: datetime | None = None
            while True:
                # Wait for the heartbeat interval or until we receive a stop signal
                if self._stop_event.wait(timeout=self.config.heartbeat_interval.total_seconds()):
                    # Stop event was set - time to exit
                    monitor_pid = self._process.pid if self._process else "None"
                    log.debug("Service monitor thread exiting", service_id=self.id, was_monitoring_pid=monitor_pid)
                    self._keep_alive_thread = None
                    return

                if self._mode == ServiceMode.STOPPED:
                    monitor_pid = self._process.pid if self._process else "None"
                    log.info(
                        "Service monitor thread stopping - service mode is STOPPED",
                        service_id=self.id,
                        was_monitoring_pid=monitor_pid,
                    )
                    self._keep_alive_thread = None
                    return

                if self._mode == ServiceMode.STARTED and self.is_running():
                    if degraded_start is not None:
                        current_pid = self._process.pid if self._process else "None"
                        log.info("Service recovered - process is running again", service_id=self.id, pid=current_pid)
                    degraded_start = None
                    continue

                if degraded_start is None:
                    failed_pid = self._process.pid if self._process else "None"
                    log.warning(
                        "Service entered degraded state - process not running",
                        service_id=self.id,
                        failed_pid=failed_pid,
                    )
                    degraded_start = datetime.now()

                if degraded_start is not None:
                    elapsed = datetime.now() - degraded_start
                    if elapsed >= self.config.degraded_wait:
                        elapsed_seconds = elapsed.total_seconds()
                        log.warning(
                            "Service triggering restart after degraded state",
                            service_id=self.id,
                            elapsed_seconds=round(elapsed_seconds, 1),
                        )
                        self.restart()
                        degraded_start = None

        t = threading.Thread(target=monitor, daemon=True)
        t.start()
        self._keep_alive_thread = t


@fail_gracefully()
def stop_process(proc: psutil.Process, grace_seconds: float = 5.0) -> None:
    try:
        pid = proc.pid
        log.info("Stopping process tree", pid=pid, grace_seconds=grace_seconds)
        terminate_process_tree(proc, timeout=grace_seconds)
        log.info("Process tree termination completed", pid=pid)
    except psutil.NoSuchProcess:
        log.info("Process no longer exists - already terminated")
    except Exception as e:
        log.warning("Exception during process termination", error=str(e))
