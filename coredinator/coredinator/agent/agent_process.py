from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import Literal, NewType

import backoff

AgentID = NewType("AgentID", str)

# TODO: make this configurable
# TODO: add automatic discovery
AGENT_EXECUTABLE = Path("dist/corerl")


AgentState = Literal["starting", "running", "stopping", "stopped", "failed"]


@dataclass(frozen=True, slots=True)
class AgentStatus:
    id: AgentID
    state: AgentState
    config_path: Path | None


class AgentProcess:
    def __init__(self, id: AgentID, config_path: Path):
        self.id = id

        self._process: Popen | None = None
        self._exe_path: Path = AGENT_EXECUTABLE
        self._config_path: Path = config_path


    def _ensure_executable(self):
        if not self._exe_path.exists():
            raise FileNotFoundError(f"Agent executable not found at {self._exe_path}")
        if not os.access(self._exe_path, os.X_OK):
            raise PermissionError(f"Agent executable is not executable: {self._exe_path}")

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
            return AgentStatus(
                id=self.id,
                state="stopped",
                config_path=self._config_path,
            )

        code = self._process.poll()
        if code is None:
            return AgentStatus(
                id=self.id,
                state="running",
                config_path=self._config_path,
            )

        return AgentStatus(
            id=self.id,
            state="failed",
            config_path=self._config_path,
        )
