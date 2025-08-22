import time

import psutil


def terminate_process_tree(proc: psutil.Process, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
    """
    Terminate a process and all its children, then wait for termination.
    """
    try:
        # Terminate all children first
        for child in proc.children(recursive=True):
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Terminate the main process
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Process might already be dead
        pass

    # Wait for termination
    return wait_for_termination(proc, timeout, poll_interval)


def wait_for_termination(proc: psutil.Process, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
    """
    Wait for a process to terminate, handling zombie state and force kill if needed.
    Returns True if the process is no longer running within the timeout, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Check if process is no longer running
            if not proc.is_running():
                return True

            # If process is zombie, it's effectively terminated
            status = proc.status()
            if status == psutil.STATUS_ZOMBIE:
                return True

        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            # Process might be terminating, give it a moment
            time.sleep(poll_interval)
            continue

        time.sleep(poll_interval)

    # Timeout reached - try force kill as last resort
    try:
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            proc.kill()
            # Give it a moment to die
            time.sleep(0.1)
            return not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True
    except Exception:
        pass

    # Final check
    try:
        return not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True
