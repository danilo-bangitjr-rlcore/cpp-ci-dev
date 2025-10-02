from __future__ import annotations

import time

import psutil
from lib_utils.errors import fail_gracefully


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

    @fail_gracefully()
    def terminate(self):
        self._proc.terminate()

    @fail_gracefully()
    def kill(self):
        self._proc.kill()

    def terminate_tree(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        children = self.children()

        for child in children:
            child.terminate()

        self.terminate()

        return self.wait_for_termination(timeout, poll_interval, children)

    def kill_tree(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        children = self.children()

        for child in children:
            child.kill()

        self.kill()

        return self.wait_for_termination(timeout, poll_interval, children)

    def wait_for_termination(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.1,
        children: list[Process] | None = None,
    ) -> bool:
        children = children or []
        deadline = time.time() + timeout

        while time.time() < deadline:
            parent_done = not self.is_running() or self.is_zombie()
            children_done = all(not c.is_running() or c.is_zombie() for c in children)

            if parent_done and children_done:
                return True

            time.sleep(poll_interval)

        survivors = [c for c in children if c.is_running() and not c.is_zombie()]
        for survivor in survivors:
            survivor.kill()

        if self.is_running() and not self.is_zombie():
            self.kill()
            time.sleep(0.1)

        parent_done = not self.is_running() or self.is_zombie()
        children_done = all(not c.is_running() or c.is_zombie() for c in children)
        return parent_done and children_done
