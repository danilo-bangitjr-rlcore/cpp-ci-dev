from __future__ import annotations

import time

import psutil


class Process:
    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(self, proc: psutil.Process):
        self._proc = proc

    @staticmethod
    def from_pid(pid: int) -> Process:
        return Process(psutil.Process(pid))

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def psutil(self) -> psutil.Process:
        return self._proc

    # ============================================================================
    # Status Checks
    # ============================================================================

    def is_running(self) -> bool:
        try:
            return self._proc.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def is_zombie(self) -> bool:
        try:
            return self._proc.status() == psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    # ============================================================================
    # Process Tree
    # ============================================================================

    def children(self) -> list[Process]:
        try:
            return [Process(child) for child in self._proc.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    # ============================================================================
    # Lifecycle Management
    # ============================================================================

    def terminate(self):
        try:
            self._proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def kill(self):
        try:
            self._proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def terminate_tree(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        for child in self.children():
            child.terminate()

        self.terminate()

        return self.wait_for_termination(timeout, poll_interval)

    def kill_tree(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        for child in self.children():
            child.kill()

        self.kill()

        return self.wait_for_termination(timeout, poll_interval)

    def wait_for_termination(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_running() or self.is_zombie():
                return True

            time.sleep(poll_interval)

        if self.is_running() and not self.is_zombie():
            self.kill()
            time.sleep(0.1)

        return not self.is_running() or self.is_zombie()
