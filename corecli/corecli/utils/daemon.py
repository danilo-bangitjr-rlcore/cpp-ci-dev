import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import psutil
from lib_utils.maybe import Maybe


def read_log_file_safely(log_file: Path | None) -> str:
    if not log_file:
        return "No log file specified"

    if not log_file.exists():
        return f"Log file does not exist: {log_file}"

    try:
        log_contents = log_file.read_text()
        if log_contents.strip():
            return f"Log file contents:\n{log_contents}"
        return "Log file is empty"
    except Exception as e:
        return f"Could not read log file {log_file}: {e}"


def start_daemon_process(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> int:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)

    stdin = subprocess.DEVNULL

    def setup_log_file(log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return open(log_path, "a")

    log_handle = (
        Maybe(log_file)
        .map(setup_log_file)
        .unwrap()
    )
    stdout = stderr = log_handle if log_handle else subprocess.DEVNULL

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=process_env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

        time.sleep(0.1)

        if process.poll() is not None:
            # Process failed immediately - get error details with log context
            error_details = f"Command {command} failed with exit code {process.returncode}"
            log_info = read_log_file_safely(log_file)
            error_details += f"\n{log_info}"

            raise subprocess.CalledProcessError(process.returncode, command, output=error_details)

        return process.pid

    finally:
        if log_handle is not None:
            log_handle.close()


def stop_process_gracefully(pid: int, timeout: float = 10.0) -> bool:
    def get_process() -> psutil.Process | None:
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None
        except psutil.AccessDenied:
            return None

    def terminate_process(proc: psutil.Process) -> bool:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return True
        except psutil.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
                return True
            except psutil.TimeoutExpired:
                return False

    return (
        Maybe(get_process())
        .map(terminate_process)
        .or_else(True)
    )


def wait_for_event(predicate: Callable[[], bool], interval: float, timeout: float) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if predicate():
            return True
        time.sleep(interval)
    return False
