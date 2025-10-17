import subprocess
import sys
import time

import psutil
import pytest

from lib_process.process import Process

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sleep_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100)"])
    yield proc
    try:
        proc.kill()
        proc.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


@pytest.fixture
def process_tree():
    script = """
import subprocess
import sys
import time

children = []
for _ in range(3):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100)"])
    children.append(child)

time.sleep(100)
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    time.sleep(0.5)
    yield proc
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        proc.kill()
        proc.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired, psutil.NoSuchProcess):
        pass


# ============================================================================
# Status Checks
# ============================================================================


def test_is_running_alive_process(sleep_process: subprocess.Popen):
    """
    is_running returns True for alive process
    """
    process = Process.from_pid(sleep_process.pid)
    assert process.is_running() is True


def test_is_running_dead_process(sleep_process: subprocess.Popen):
    """
    is_running returns False after process terminates
    """
    process = Process.from_pid(sleep_process.pid)
    sleep_process.kill()
    sleep_process.wait()
    time.sleep(0.1)
    assert process.is_running() is False


def test_is_running_nonexistent_process(sleep_process: subprocess.Popen):
    """
    is_running returns False after process dies
    """
    process = Process.from_pid(sleep_process.pid)
    sleep_process.kill()
    sleep_process.wait()
    assert process.is_running() is False


def test_is_zombie_normal_process(sleep_process: subprocess.Popen):
    """
    is_zombie returns False for normal running process
    """
    process = Process.from_pid(sleep_process.pid)
    assert process.is_zombie() is False


def test_is_zombie_dead_process(sleep_process: subprocess.Popen):
    """
    is_zombie returns False after process is killed
    """
    process = Process.from_pid(sleep_process.pid)
    sleep_process.kill()
    sleep_process.wait()
    time.sleep(0.1)
    assert process.is_zombie() is False


# ============================================================================
# Process Tree
# ============================================================================


def test_children_no_children(sleep_process: subprocess.Popen):
    """
    children returns empty list when process has no children
    """
    process = Process.from_pid(sleep_process.pid)
    assert process.children() == []


def test_children_with_children(process_tree: subprocess.Popen):
    """
    children returns all descendants recursively
    """
    time.sleep(0.2)
    process = Process.from_pid(process_tree.pid)
    children = process.children()
    assert all(isinstance(child, Process) for child in children)
    assert all(child.is_running() for child in children)


# ============================================================================
# Lifecycle Management - Single Process
# ============================================================================


def test_terminate(sleep_process: subprocess.Popen):
    """
    terminate sends SIGTERM to process
    """
    process = Process.from_pid(sleep_process.pid)
    process.terminate()
    sleep_process.wait(timeout=1)
    time.sleep(0.2)
    assert not process.is_running()


def test_terminate_already_dead():
    """
    terminate handles already-dead process gracefully
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
    process = Process.from_pid(proc.pid)
    proc.wait()
    time.sleep(0.1)
    process.terminate()


def test_kill(sleep_process: subprocess.Popen):
    """
    kill sends SIGKILL to process
    """
    process = Process.from_pid(sleep_process.pid)
    process.kill()
    sleep_process.wait(timeout=1)
    time.sleep(0.2)
    assert not process.is_running()


def test_kill_already_dead():
    """
    kill handles already-dead process gracefully
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
    process = Process.from_pid(proc.pid)
    proc.wait()
    time.sleep(0.1)
    process.kill()


# ============================================================================
# Lifecycle Management - Process Tree
# ============================================================================


def test_terminate_tree_with_children(process_tree: subprocess.Popen):
    """
    terminate_tree terminates parent and all children
    """
    time.sleep(0.2)
    process = Process.from_pid(process_tree.pid)
    children = process.children()
    assert len(children) > 1

    assert process.terminate_tree(timeout=3.0)


def test_terminate_tree_no_children(sleep_process: subprocess.Popen):
    """
    terminate_tree works on process without children
    """
    process = Process.from_pid(sleep_process.pid)
    assert process.terminate_tree(timeout=3.0)


def test_kill_tree_with_children(process_tree: subprocess.Popen):
    """
    kill_tree kills parent and all children
    """
    time.sleep(0.2)
    process = Process.from_pid(process_tree.pid)
    children = process.children()
    assert len(children) > 1

    assert process.kill_tree(timeout=3.0)


def test_kill_tree_no_children(sleep_process: subprocess.Popen):
    """
    kill_tree works on process without children
    """
    process = Process.from_pid(sleep_process.pid)
    assert process.kill_tree(timeout=3.0)


# ============================================================================
# Wait for Termination
# ============================================================================


def test_wait_for_termination_quick_exit():
    """
    wait_for_termination returns True when process exits quickly
    """
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"])
    process = Process.from_pid(proc.pid)

    assert process.wait_for_termination(timeout=2.0, poll_interval=0.05)


def test_wait_for_termination_after_terminate(sleep_process: subprocess.Popen):
    """
    wait_for_termination returns True after terminating process
    """
    process = Process.from_pid(sleep_process.pid)
    process.terminate()

    assert process.wait_for_termination(timeout=3.0, poll_interval=0.05)


def test_wait_for_termination_force_kill():
    """
    wait_for_termination force kills if timeout reached
    """
    script = """
import signal
import time

def handler(signum, frame):
    pass

signal.signal(signal.SIGTERM, handler)
time.sleep(100)
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    process = Process.from_pid(proc.pid)
    process.terminate()

    assert process.wait_for_termination(timeout=0.5, poll_interval=0.05)


def test_wait_for_termination_already_dead():
    """
    wait_for_termination returns True for already-dead process
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    process = Process.from_pid(proc.pid)
    proc.wait()
    time.sleep(0.1)

    assert process.wait_for_termination(timeout=1.0)


# ============================================================================
# Process Creation
# ============================================================================


def test_start_in_background_creates_detached_process():
    """
    Process.start_in_background creates a detached process that continues after parent exits
    """
    process = Process.start_in_background([sys.executable, "-c", "import time; time.sleep(0.5)"])

    assert process.is_running()
    process.terminate_tree(timeout=2.0)
