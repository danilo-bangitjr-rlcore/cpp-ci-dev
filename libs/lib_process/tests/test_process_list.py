import subprocess
import sys

from lib_process.process import Process
from lib_process.process_list import find_processes_by_name_patterns


def test_find_processes_by_name_patterns_finds_python():
    """
    find_processes_by_name_patterns finds processes matching pattern
    """

    processes = find_processes_by_name_patterns(["python"])

    assert len(processes) > 0
    assert all(isinstance(p, Process) for p in processes)


def test_find_processes_by_name_patterns_case_insensitive():
    """
    find_processes_by_name_patterns is case-insensitive
    """

    lower_results = find_processes_by_name_patterns(["python"])
    upper_results = find_processes_by_name_patterns(["PYTHON"])

    assert len(lower_results) == len(upper_results)


def test_find_processes_by_name_patterns_multiple_patterns():
    """
    find_processes_by_name_patterns can search for multiple patterns
    """

    proc1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100)"])
    proc2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100)"])

    try:
        processes = find_processes_by_name_patterns(["python", "nonexistent_process"])

        pids = [p.psutil.pid for p in processes]
        assert proc1.pid in pids
        assert proc2.pid in pids
    finally:
        proc1.kill()
        proc2.kill()
        try:
            proc1.wait(timeout=1)
            proc2.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
