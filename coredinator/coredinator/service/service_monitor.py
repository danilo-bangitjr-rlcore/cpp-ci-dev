from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING

from coredinator.logging_config import get_logger
from coredinator.service.protocols import ServiceIntendedState

if TYPE_CHECKING:
    from coredinator.service.service import Service, ServiceConfig

log = get_logger(__name__)


class ServiceMonitor:
    def __init__(self, service: Service, config: ServiceConfig):
        self._service = service
        self._config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()


    def start_monitoring(self):
        if self._thread is not None and self._thread.is_alive():
            log.debug("Service background monitor already running", service_id=self._service.id)
            return

        log.info("Service starting background monitor thread", service_id=self._service.id)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()


    def stop_monitoring(self):
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

            if self._thread.is_alive():
                log.warning("Service monitor thread did not finish in time", service_id=self._service.id)


    def _monitor_loop(self):
        current_pid = self._get_current_pid()
        log.info("Service background monitor thread started", service_id=self._service.id, monitoring_pid=current_pid)

        degraded_start: datetime | None = None

        while True:
            # Wait for the heartbeat interval or until we receive a stop signal
            if self._stop_event.wait(timeout=self._config.heartbeat_interval.total_seconds()):
                self._log_monitor_exit()
                return

            if self._service._intended_state == ServiceIntendedState.STOPPED:
                self._log_monitor_service_stopped()
                return

            if self._service._intended_state == ServiceIntendedState.RUNNING and self._service.is_running():
                if degraded_start is not None:
                    self._log_service_recovered()
                degraded_start = None
                continue

            if degraded_start is None:
                degraded_start = self._log_service_degraded()

            if degraded_start is not None:
                if self._check_restart_needed(degraded_start):
                    degraded_start = None


    def _get_current_pid(self):
        return self._service._process.psutil.pid if self._service._process else None


    def _log_monitor_exit(self):
        monitor_pid = self._get_current_pid()
        log.debug("Service monitor thread exiting", service_id=self._service.id, was_monitoring_pid=monitor_pid)
        self._thread = None


    def _log_monitor_service_stopped(self):
        monitor_pid = self._get_current_pid()
        log.info(
            "Service monitor thread stopping - service state is STOPPED",
            service_id=self._service.id,
            was_monitoring_pid=monitor_pid,
        )
        self._thread = None


    def _log_service_recovered(self):
        current_pid = self._get_current_pid()
        log.info("Service recovered - process is running again", service_id=self._service.id, pid=current_pid)


    def _log_service_degraded(self):
        failed_pid = self._get_current_pid()
        log.warning(
            "Service entered degraded state - process not running",
            service_id=self._service.id,
            failed_pid=failed_pid,
        )
        return datetime.now()


    def _check_restart_needed(self, degraded_start: datetime):
        elapsed = datetime.now() - degraded_start
        if elapsed >= self._config.degraded_wait:
            elapsed_seconds = elapsed.total_seconds()
            log.warning(
                "Service triggering restart after degraded state",
                service_id=self._service.id,
                elapsed_seconds=round(elapsed_seconds, 1),
            )
            self._service.restart()
            return True
        return False
