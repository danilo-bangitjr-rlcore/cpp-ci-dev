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
