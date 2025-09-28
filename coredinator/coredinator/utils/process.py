import time

import psutil


def terminate_process_tree(proc: psutil.Process, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
    """
    Terminate a process and all its children, then wait for termination.
    """
    try:
        for child in proc.children(recursive=True):
            safe_terminate_process(child)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Process is already gone, nothing to do
        pass

    safe_terminate_process(proc)
    return wait_for_termination(proc, timeout, poll_interval)


def wait_for_termination(proc: psutil.Process, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
    """
    Wait for a process to terminate, handling zombie state and force kill if needed.
    Returns True if the process is no longer running within the timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not safe_is_process_running(proc):
            return True

        status = safe_get_process_status(proc)
        if status is None:
            return True
        if status == psutil.STATUS_ZOMBIE:
            return True

        time.sleep(poll_interval)

    if safe_is_process_running(proc):
        status = safe_get_process_status(proc)
        if status is not None and status != psutil.STATUS_ZOMBIE:
            safe_kill_process(proc)
            time.sleep(0.1)

    return not safe_is_process_running(proc) or safe_get_process_status(proc) == psutil.STATUS_ZOMBIE


def safe_get_process_name(proc: psutil.Process) -> str | None:
    """
    Get process name safely, returning None if process is inaccessible.
    """
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def safe_is_process_running(proc: psutil.Process) -> bool:
    """
    Check if process is running safely, returning False if process is inaccessible.
    """
    try:
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def safe_get_process_status(proc: psutil.Process) -> str | None:
    """
    Get process status safely, returning None if process is inaccessible.
    """
    try:
        return proc.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def safe_terminate_process(proc: psutil.Process) -> bool:
    """
    Terminate process safely (SIGTERM), returning True if successful or process was already gone.
    """
    try:
        proc.terminate()
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True


def safe_kill_process(proc: psutil.Process) -> bool:
    """
    Kill process safely (SIGKILL), returning True if successful or process was already gone.
    """
    try:
        proc.kill()
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True


def safe_iter_processes() -> list[psutil.Process]:
    """
    Iterate over all processes safely, filtering out inaccessible ones.
    """
    try:
        return [proc for proc in psutil.process_iter() if safe_get_process_name(proc) is not None]
    except Exception:
        return []


def find_processes_by_name_patterns(name_patterns: list[str]) -> list[psutil.Process]:
    """
    Find processes matching any of the given name patterns.
    """
    matching_processes = []
    for proc in safe_iter_processes():
        name = safe_get_process_name(proc)
        if name and any(pattern in name.lower() for pattern in name_patterns):
            matching_processes.append(proc)

    return matching_processes
