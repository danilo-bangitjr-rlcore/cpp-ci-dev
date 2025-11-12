import psutil

from lib_process.process import Process


def find_processes_by_name_patterns(patterns: list[str]) -> list[Process]:
    matched_processes: list[Process] = []

    for proc in psutil.process_iter(['name']):
        try:
            proc_name = proc.info['name']
            if proc_name and any(pattern.lower() in proc_name.lower() for pattern in patterns):
                matched_processes.append(Process(proc))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return matched_processes
